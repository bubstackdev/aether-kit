#!/usr/bin/env python3
"""Export Qwen3 bucketed-contract variants for the time-to-first-token probe (development tooling only).

The 0.0.2 bucketed graph builds its cache key positions with a `cumsum` over the capacity axis, which converts to
Core AI's `scan`; the running graph has none. `keypos` takes the key positions as a host input instead
(`key_positions` [1, capacity] = 0..<capacity), so the graph has no scan. Everything else is the bucketed graph.

    Forge/.venv/bin/python -I Spikes/ttft/export_variant.py --variant <bucketed|keypos|running> \
        [--checkpoint <hf snapshot> | --layers N] --output <new .aimodel>

`--layers N` exports N random-weight layers at Qwen3-0.6B's dimensions (no checkpoint; seconds of CPU).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Forge" / "language"))
from export_qwen3 import drop_identity_copies  # noqa: E402
from qwen3_decode import Qwen3Decoder, load_weights, read_config  # noqa: E402

QWEN3_06B = {"vocab_size": 151936, "hidden_size": 1024, "num_attention_heads": 16, "num_key_value_heads": 8,
             "head_dim": 128, "num_hidden_layers": 28, "intermediate_size": 3072, "rms_norm_eps": 1e-6,
             "rope_theta": 1_000_000.0, "tie_word_embeddings": True}


class KeyPositionDecoder(Qwen3Decoder):
    """Bucketed graph with host-provided key positions instead of an in-graph cumsum."""

    def forward(self, input_ids, position_ids, key_positions, key_cache, value_cache):
        q_len = input_ids.shape[1]
        x = self.embed(input_ids)
        new_positions = position_ids[0]
        cache_visible = key_positions < new_positions[:1, None]                   # [1, cap]
        chunk_visible = new_positions[None, :] <= new_positions[:, None]
        visible = torch.cat((cache_visible.expand(q_len, -1), chunk_visible), dim=1)
        cos, sin = self.rope(new_positions, x.dtype)
        keys, values = [], []
        for i, block in enumerate(self.blocks):
            x, k, v = block.bucketed(x, cos, sin, key_cache, value_cache, i, visible)
            keys.append(k)
            values.append(v)
        return self.head(x[:, -1:]), torch.stack(keys), torch.stack(values)

    def example(self, max_len: int):
        q_dim = torch.export.Dim("q", min=1, max=max_len)
        cap_dim = torch.export.Dim("capacity", min=2, max=self.max_capacity)
        ex_q, ex_start, cap = 8, 16, 56
        kc, vc = self.empty_state(cap)
        inputs = (torch.arange(ex_q, dtype=torch.int32)[None] + 100,
                  torch.arange(ex_start, ex_start + ex_q, dtype=torch.int32)[None],
                  torch.arange(cap, dtype=torch.int32)[None], kc, vc)
        return inputs, ({1: q_dim}, {1: q_dim}, {1: cap_dim}, {3: cap_dim}, {3: cap_dim})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["bucketed", "keypos", "running"], required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.checkpoint is None) == (args.layers is None):
        raise SystemExit("pass exactly one of --checkpoint or --layers")
    if args.output.exists():
        raise SystemExit("output exists")
    import coreai_torch as ct

    torch.manual_seed(0)
    cfg = read_config(args.checkpoint) if args.checkpoint else dict(QWEN3_06B, num_hidden_layers=args.layers)
    contract = "running" if args.variant == "running" else "bucketed"
    cls = KeyPositionDecoder if args.variant == "keypos" else Qwen3Decoder
    module = cls(cfg, contract)
    if args.checkpoint:
        load_weights(module, args.checkpoint)
    else:
        with torch.no_grad():
            for n, p in module.named_parameters():
                p.fill_(1.0) if "norm" in n else p.normal_(0, 0.02)
    module = module.to(torch.float16).eval()
    example, dynamic = module.example(512)
    with torch.no_grad():
        ep = torch.export.export(module, example, dynamic_shapes=dynamic)
        ep = ep.run_decompositions(ct.get_decomp_table())
    drop_identity_copies(ep)
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    if args.variant == "running":
        converter.add_exported_program(ep, input_names=["input_ids", "position_ids"], output_names=["logits"],
                                       state_names=["keyCache", "valueCache"], entrypoint_name="main")
    else:
        names = ["input_ids", "cache_positions"] + (["key_positions"] if args.variant == "keypos" else []) + ["keyCache", "valueCache"]
        converter.add_exported_program(ep, input_names=names, output_names=["logits", "keyUpdate", "valueUpdate"],
                                       state_names=[], entrypoint_name="main")
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

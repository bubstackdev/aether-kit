#!/usr/bin/env python3
"""Hybrid-state 0.0.3 spike baseline: the 0.0.2 report's `static-lean` decode layout, rebuilt on Forge's port.

One `main` with fixed shapes: input_ids [1, 1], position_ids [1, 1] holding the current position, KV caches as states
with a fixed 1,024-row capacity (the new row blended in at the position by a one-hot, the mask `arange(C) <= pos`),
and the one-token Mamba recurrence. It is not Kit-compatible; it anchors the 0.0.2 numbers on today's host.

    Forge/.venv/bin/python -I Spikes/hybrid-state-003/export_baseline.py --checkpoint <snapshot> --output <new dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Forge" / "language"))
from export_qwen3 import drop_identity_copies  # noqa: E402
from granite_hybrid import GraniteHybridDecoder, load_weights, read_config  # noqa: E402

CAPACITY = 1024


class StaticDecode(torch.nn.Module):
    def __init__(self, inner: GraniteHybridDecoder) -> None:
        super().__init__()
        self.inner = inner

    def attention(self, attn, x, key_cache, value_cache, slot, pos):
        q = attn.q_proj(x).view(1, 1, attn.heads, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(1, 1, attn.kv_heads, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(x).view(1, 1, attn.kv_heads, attn.head_dim).transpose(1, 2)
        idx = torch.cumsum(torch.ones_like(key_cache[0, 0, 0, :, 0], dtype=torch.int32), 0, dtype=torch.int32) - 1
        hot = (idx == pos).to(k.dtype)[None, None, :, None]
        keys = key_cache[slot] * (1 - hot) + k * hot
        values = value_cache[slot] * (1 - hot) + v * hot
        key_cache[slot:slot + 1] = keys[None]
        value_cache[slot:slot + 1] = values[None]
        groups = attn.heads // attn.kv_heads
        keys = keys[:, :, None].expand(1, attn.kv_heads, groups, CAPACITY, attn.head_dim).reshape(1, attn.heads, CAPACITY, attn.head_dim)
        values = values[:, :, None].expand(1, attn.kv_heads, groups, CAPACITY, attn.head_dim).reshape(1, attn.heads, CAPACITY, attn.head_dim)
        mask = torch.where(idx <= pos, 0.0, -30000.0).to(q.dtype)[None, None, None, :]
        out = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask, scale=attn.scale)
        return attn.o_proj(out.transpose(1, 2).reshape(1, 1, attn.heads * attn.head_dim))

    def forward(self, input_ids, position_ids, key_cache, value_cache, conv_state, ssm_state):
        m = self.inner
        pos = position_ids[0, 0]
        x = m.embed(input_ids) * m.embedding_multiplier
        for layer, slot in zip(m.layers, m.slot):
            h = layer.input_norm(x)
            if layer.kind == "mamba":
                h = layer.mixer.recurrent(h, conv_state, ssm_state, slot)
            else:
                h = self.attention(layer.mixer, h, key_cache, value_cache, slot, pos)
            x = x + h * m.residual_multiplier
            x = x + layer.mlp(layer.post_norm(x)) * m.residual_multiplier
        return F.linear(m.norm(x), m.embed.weight) / m.logits_scaling


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output directory already exists")
    import coreai_torch as ct

    module = GraniteHybridDecoder(read_config(args.checkpoint))
    load_weights(module, args.checkpoint)
    module = module.to(torch.float16).eval()
    kc, vc, cs, ss = module.empty_state(CAPACITY)
    example = (torch.tensor([[100]], dtype=torch.int32), torch.tensor([[5]], dtype=torch.int32), kc, vc, cs, ss)
    with torch.no_grad():
        ep = torch.export.export(StaticDecode(module), example).run_decompositions(ct.get_decomp_table())
    drop_identity_copies(ep)
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    converter.add_exported_program(ep, input_names=["input_ids", "position_ids"], output_names=["logits"],
                                   state_names=["keyCache", "valueCache", "convState", "ssmState"], entrypoint_name="main")
    program = converter.to_coreai()
    program.optimize()
    args.output.mkdir(parents=True)
    program.save_asset(args.output / "model.aimodel")
    print(f"saved {args.output / 'model.aimodel'}")


if __name__ == "__main__":
    main()

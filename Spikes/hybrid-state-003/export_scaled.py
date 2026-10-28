#!/usr/bin/env python3
"""Hybrid-state 0.0.3 spike: real-dimension Granite hybrids with fewer layers, random weights, `main` only, to find
how Core AI's GPU specialization time scales with depth (the full 32-layer export did not open within 6 minutes on a
busy host; the 4-layer tiny model opens in seconds).

    Forge/.venv/bin/python -I Spikes/hybrid-state-003/export_scaled.py --checkpoint <snapshot> --layers <pattern> --output <new dir>

`--layers` is a pattern of m (Mamba) and a (attention), e.g. mmam, or `real:A:B` for the checkpoint's layers A..<B.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Forge" / "language"))
from export_granite_hybrid import INPUTS, OUTPUTS, STATES, strip_runtime_asserts  # noqa: E402
from export_qwen3 import drop_identity_copies  # noqa: E402
from granite_hybrid import Decode, GraniteHybridDecoder, Main, examples, read_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode", action="store_true", help="also export the q = 1 `decode` function (C2 layout)")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output exists")
    import coreai_torch as ct

    cfg = read_config(args.checkpoint)
    if args.layers.startswith("real:"):
        a, b = (int(x) for x in args.layers.split(":")[1:])
        kinds = cfg["layer_types"][a:b]
    else:
        kinds = ["mamba" if c == "m" else "attention" for c in args.layers]
    if "attention" not in kinds:                      # the module needs a KV layout; keep one attention, unused below
        raise SystemExit("include at least one attention layer")
    cfg = dict(cfg, layer_types=kinds, num_hidden_layers=len(kinds))
    torch.manual_seed(0)
    module = GraniteHybridDecoder(cfg)
    with torch.no_grad():
        for n, p in module.named_parameters():
            if "norm" in n:
                p.fill_(1.0)
            elif n.endswith("A_log"):
                p.copy_(torch.log(torch.rand_like(p) * 8 + 1))
            else:
                p.normal_(0, 0.02)
    module = module.to(torch.float16).eval()
    main_ex, main_dyn, dec_ex, dec_dyn = examples(module, 512)
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    entries = [("main", Main(module), main_ex, main_dyn)] + ([("decode", Decode(module), dec_ex, dec_dyn)] if args.decode else [])
    for name, wrapper, example, dynamic in entries:
        with torch.no_grad():
            ep = torch.export.export(wrapper, example, dynamic_shapes=dynamic, prefer_deferred_runtime_asserts_over_guards=True)
            ep = ep.run_decompositions(ct.get_decomp_table())
        strip_runtime_asserts(ep)
        drop_identity_copies(ep)
        converter.add_exported_program(ep, input_names=INPUTS, output_names=OUTPUTS, state_names=STATES, entrypoint_name=name)
    program = converter.to_coreai()
    program.optimize()
    args.output.mkdir(parents=True)
    program.save_asset(args.output / "model.aimodel")
    print(f"saved {args.output} ({''.join(k[0] for k in kinds)})")


if __name__ == "__main__":
    main()

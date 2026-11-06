"""CPU reference for the H3 device-chain gate (Tests/AetherRuntimeTests/H3ChainGateTests.swift).

Runs the first BLOCKS transformer blocks of a bundle's body (`weights/body.json`, weights read from their extents)
chained on the CPU with the original block module (PEFT LoRA included), from seeded inputs, and writes the inputs and
each block's output as raw tensors with a spec.json. The Swift gate runs the same blocks through `GraphChain` and
`WeightBanks` on the GPU and compares (relative L2, the prototype's per-graph limit 0.02).

    Forge/.venv/bin/python -I Forge/h3/reference_chain_h3.py --bundle <export>/bundle --output <export>/chain-reference
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h3_graphs as hg  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--blocks", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(8)
    table_doc = json.loads((args.bundle / "weights/body.json").read_text())
    build = json.loads((args.bundle.parent / "work/build.json").read_text())
    specs = build["body-weights"]["inputs"]
    assert len(table_doc["blocks"]) >= args.blocks

    model = hg.meta_model(num_layers=1)
    block = model.transformer_blocks[0].to_empty(device="cpu").to(torch.bfloat16)
    hg.zero_turbo(model)
    block.adaln_proj = hg.CachedModulation()
    names = [n for n, _ in hg.BlockCall(block).named_parameters()]
    assert names == [s["source"] for s in specs], "the module's parameter order differs from the table's"
    call = hg.BlockCall(block).eval()

    g = torch.Generator().manual_seed(4242)
    r = args.rows
    hidden = torch.randn(1, r, hg.HIDDEN, generator=g).to(torch.bfloat16)
    table = (0.1 * torch.randn(6, 6, hg.HIDDEN, generator=g)).to(torch.bfloat16)
    tags = torch.randint(0, 6, (r,), generator=g, dtype=torch.int32)
    angles = torch.rand(r, 96, generator=g) * 6.2831853
    cos, sin = angles.cos(), angles.sin()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    tensors = {"hidden": hg.raw_tensor(out / "hidden.raw", hidden), "table": hg.raw_tensor(out / "table.raw", table),
               "tags": hg.raw_tensor(out / "tags.raw", tags), "cos": hg.raw_tensor(out / "cos.raw", cos),
               "sin": hg.raw_tensor(out / "sin.raw", sin)}
    x = hidden
    for b in range(args.blocks):
        weights = {s["source"]: hg.read_extent(args.bundle, table_doc["blocks"][b][s["name"]], getattr(torch, s["dtype"]), s["shape"])
                   for s in specs}
        with torch.inference_mode():
            x = torch.func.functional_call(call, weights, (x, table, tags, cos, sin))
        tensors[f"block{b}"] = hg.raw_tensor(out / f"block{b}.raw", x)
        print(json.dumps({"block": b, "rms": float(x.float().pow(2).mean().sqrt())}), flush=True)
    (out / "spec.json").write_text(json.dumps({"rows": r, "blocks": args.blocks, "tensors": tensors,
                                               "bundle": str(args.bundle.resolve())}, indent=1))


if __name__ == "__main__":
    main()

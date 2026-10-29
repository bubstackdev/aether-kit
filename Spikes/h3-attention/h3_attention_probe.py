#!/usr/bin/env python3
"""H3 attention go/no-go (docs/spikes/h3-attention-2026-10.md). Development tooling only.

H3's body block calls `F.scaled_dot_product_attention` (maskless, non-causal) on [1, 56, R, 128] bf16 queries, keys and
values, and coreai-torch exports it as an SDPA composite. This probe exports that attention alone, weight-free, three
ways, with R dynamic in 64...40960 like the body:
  sdpa_bf16      the body's form: SDPA on bf16;
  explicit_bf16  matmul, softmax, matmul on bf16 (what an unfused graph would be);
  sdpa_fp16      SDPA on fp16.

    Forge/h3-oracle/.venv/bin/python -I -B Spikes/h3-attention/h3_attention_probe.py export <variant> --output <new .aimodel>
    Forge/h3-oracle/.venv/bin/python -I -B Spikes/h3-attention/h3_attention_probe.py compare <results directory>

`h3-attention-probe.swift` runs an asset on the GPU. Inputs come from an integer hash both sides compute exactly, so
`compare` rebuilds them and scores the sampled outputs the Swift probe wrote against a float64 reference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HEADS, DIM, MIN_ROWS, MAX_ROWS = 56, 128, 64, 40960
SAMPLE_HEADS = [0, 13, 27, 41, 55]
SAMPLE_ROWS = 64
# Query scale: the logits' spread then resembles a trained model's rather than a near-uniform softmax.
SCALES = {"q": 8.0, "k": 1.0, "v": 1.0}
SEEDS = {"q": 1, "k": 2, "v": 3}


class SDPA(torch.nn.Module):
    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v)


class Explicit(torch.nn.Module):
    def forward(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-1, -2)) * (DIM ** -0.5)
        return torch.matmul(torch.softmax(scores, dim=-1), v)


VARIANTS = {"sdpa_bf16": (SDPA, torch.bfloat16), "explicit_bf16": (Explicit, torch.bfloat16), "sdpa_fp16": (SDPA, torch.float16)}


def export(variant: str, output: Path) -> None:
    import coreai_torch as ct
    module, dtype = VARIANTS[variant]
    if output.exists():
        sys.exit(f"{output} exists; give a new path")
    rows = torch.export.Dim("rows", min=MIN_ROWS, max=MAX_ROWS)
    example = tuple(torch.randn(1, HEADS, 1024, DIM).to(dtype) for _ in range(3))
    shapes = {name: {2: rows} for name in ("q", "k", "v")}
    with torch.inference_mode():
        program = torch.export.export(module().eval(), example, dynamic_shapes=shapes).run_decompositions(ct.get_decomp_table())
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    converter.add_exported_program(program, input_names=["q", "k", "v"], output_names=["out"], state_names=[], entrypoint_name="main")
    asset = converter.to_coreai()
    asset.optimize()
    asset.save_asset(output)
    print(json.dumps({"variant": variant, "asset": str(output)}))


def hashed(name: str, rows: int, head: int) -> np.ndarray:
    """One head of the [1, HEADS, rows, DIM] input, float32: u = ((i * 2654435761 + seed * 40503) mod 2^32) / 2^32
    over the flat index i, mapped to [-scale, scale). The Swift probe computes the same values."""
    index = np.arange(head * rows * DIM, (head + 1) * rows * DIM, dtype=np.uint64)
    bits = (index * np.uint64(2654435761) + np.uint64(SEEDS[name] * 40503)) % np.uint64(1 << 32)
    unit = bits.astype(np.float64) / float(1 << 32)
    return ((unit * 2 - 1) * SCALES[name]).astype(np.float32).reshape(rows, DIM)


def rounded(array: np.ndarray, dtype: torch.dtype) -> np.ndarray:
    return torch.from_numpy(array).to(dtype).to(torch.float64).numpy()


def sample_rows(rows: int) -> list[int]:
    return [(j * (rows - 1)) // (SAMPLE_ROWS - 1) for j in range(SAMPLE_ROWS)]


def reference(rows: int, dtype: torch.dtype) -> np.ndarray:
    """Float64 attention of the sampled query rows, [len(SAMPLE_HEADS), SAMPLE_ROWS, DIM], from dtype-rounded inputs."""
    picks = sample_rows(rows)
    out = np.empty((len(SAMPLE_HEADS), SAMPLE_ROWS, DIM))
    for slot, head in enumerate(SAMPLE_HEADS):
        q, k, v = (rounded(hashed(name, rows, head), dtype) for name in ("q", "k", "v"))
        logits = q[picks] @ k.T * (DIM ** -0.5)
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=1, keepdims=True)
        out[slot] = weights @ v
    return out


def compare(results: Path) -> None:
    """Scores every `<variant>-<rows>.f32` sample against the float64 reference, and the variants against each other."""
    report = {}
    for path in sorted(results.glob("*.f32")):
        variant, rows = path.stem.rsplit("-", 1)
        rows = int(rows)
        dtype = VARIANTS[variant][1]
        sample = np.fromfile(path, dtype=np.float32).astype(np.float64).reshape(len(SAMPLE_HEADS), SAMPLE_ROWS, DIM)
        ref = reference(rows, dtype)
        relative = float(np.linalg.norm(sample - ref) / np.linalg.norm(ref))
        report.setdefault(str(rows), {})[variant] = {"relativeL2": relative, "finite": bool(np.isfinite(sample).all())}
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export")
    e.add_argument("variant", choices=sorted(VARIANTS))
    e.add_argument("--output", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("results", type=Path)
    args = parser.parse_args()
    if args.command == "export":
        export(args.variant, args.output)
    else:
        compare(args.results)


if __name__ == "__main__":
    main()

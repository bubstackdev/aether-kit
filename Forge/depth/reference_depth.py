#!/usr/bin/env python3
"""Upstream-PyTorch Depth Anything 3 small references for the Core AI gate (check_depth.py) and the Swift driver gate.

    Forge/.venv/bin/python -I Forge/depth/reference_depth.py --output <new reference dir>

Runs the unmodified upstream `DepthAnything3Net.forward` (pinned DA3-SMALL weights) on the CPU in float32, on exactly
the tensor the exported graph is fed: raw RGB on [0,1], [1,3,504,504], already rounded through float16 (the graph's
input dtype), normalized as upstream's InputProcessor does. Preprocessing is not under test here: each case's input
tensor is written out, and gates feed it as is. For each case:

    <case>/case.json        source, preprocessing recipe, shapes, SHA-256 of each file, value ranges
    <case>/input.f32        the graph input [1,3,504,504], little-endian float32 (every value fp16-representable)
    <case>/depth.f32        upstream depth [1,504,504], little-endian float32 (larger is farther)
    <case>/confidence.f32   upstream depth_conf [1,504,504], little-endian float32

and an index `references.json`. Cases: upstream's two example frames (SOH/000.png, SOH/010.png at the pinned commit),
stretched with Pillow bicubic, and one seeded synthetic image. CPU only; no GPU lock needed.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import depth_graphs as dg  # noqa: E402

CASES = [
    {"name": "soh-000", "image": "assets/examples/SOH/000.png"},
    {"name": "soh-010", "image": "assets/examples/SOH/010.png"},
    {"name": "synthetic-7", "synthetic": 7},
]
SHAPES = {"input": [1, 3, dg.SIZE, dg.SIZE], "depth": [1, dg.SIZE, dg.SIZE], "confidence": [1, dg.SIZE, dg.SIZE]}


def stats(x: np.ndarray) -> dict:
    return {"min": float(x.min()), "max": float(x.max()), "mean": float(x.mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--snapshot", type=Path, default=dg.DEFAULT_SNAPSHOT)
    ap.add_argument("--upstream", type=Path, default=dg.DEFAULT_UPSTREAM)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    verified = dg.verify_sources(args.snapshot, args.upstream)
    args.output.mkdir(parents=True)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    net = dg.build_net(args.upstream, args.snapshot)
    index = {
        "schema": 1, "generator": "Forge/depth/reference_depth.py",
        "scripts": {p.name: dg.sha256_file(p) for p in sorted(HERE.glob("*.py"))},
        "source": {"repository": dg.REPO, "revision": dg.REVISION,
                   "upstreamCode": {"repository": dg.CODE_REPO, "commit": dg.CODE_COMMIT}, "verified": verified},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "numpy", "pillow")},
        "host": {"machine": platform.machine(), "python": platform.python_version()},
        "precision": "float32 on the CPU (upstream DepthAnything3Net.forward, unmodified)",
        "shapes": SHAPES, "layout": "row-major, little-endian float32",
        "graphInput": "image = input.f32 converted to float16 (exact: every value is fp16-representable)",
        "polarity": "largerIsFarther",
        "cases": [],
    }
    for case in CASES:
        started = time.monotonic()
        out = args.output / case["name"]
        out.mkdir()
        if "image" in case:
            x = dg.image_tensor(args.upstream / case["image"])
            source = {"kind": "image", "upstreamFile": case["image"],
                      "sha256": dg.PINS["referenceImages"]["files"][case["image"]],
                      "preprocess": "Pillow convert('RGB'), resize to 504x504 BICUBIC (stretch), /255, NCHW, "
                                    "rounded through float16"}
        else:
            x = dg.synthetic_tensor(case["synthetic"])
            source = {"kind": "synthetic", "seed": case["synthetic"], "recipe": "depth_graphs.synthetic_tensor"}
        depth, conf = dg.upstream_depth(net, torch.from_numpy(x))
        depth, conf = depth.numpy(), conf.numpy()
        if not (np.isfinite(depth).all() and np.isfinite(conf).all()):
            raise SystemExit(f"{case['name']}: non-finite upstream output")
        # Polarity sanity on the natural frames: the bottom of the frame (ground, near) is nearer than the top.
        if "image" in case and not depth[0, -dg.SIZE // 8:].mean() < depth[0, : dg.SIZE // 8].mean():
            raise SystemExit(f"{case['name']}: bottom rows are not nearer than top rows; check the polarity")
        files = {"input.f32": x, "depth.f32": depth, "confidence.f32": conf}
        for name, array in files.items():
            dg.write_f32(out / name, array)
        record = {"name": case["name"], "source": source, "shapes": SHAPES,
                  "files": {name: dg.sha256_file(out / name) for name in files},
                  "depth": stats(depth), "confidence": stats(conf),
                  "seconds": round(time.monotonic() - started, 2)}
        (out / "case.json").write_text(json.dumps(record, indent=2) + "\n")
        index["cases"].append({"name": case["name"], "caseJSONSHA256": dg.sha256_file(out / "case.json")})
        print(f"[reference] {case['name']}: depth {record['depth']}, {record['seconds']} s", file=sys.stderr,
              flush=True)
    index["peakRSSGB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
    (args.output / "references.json").write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    main()

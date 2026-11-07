#!/usr/bin/env python3
"""Upstream-PyTorch AdcSR x4 references for the Core AI gate (check_adcsr.py) and the Swift driver gate.

    Forge/.venv/bin/python -I Forge/adcsr/reference_adcsr.py --output <new reference dir>

Runs upstream's network, assembled as its test.py does (`adcsr_graphs.build_upstream`), on the CPU in float32. Each case
is an 8-bit LR image. The tensor upstream sees is ToTensor * 2 - 1 of it, the graph's own input convention. The case
records:

    <case>/case.json           source, recipe, shapes, the tile grid, SHA-256 of each file, PSNRs
    <case>/lr.png              the LR image, 8-bit RGB (what an app hands the driver)
    <case>/input.f32           [1,3,H,W] float32 on [-1,1]
    <case>/tiles/<i>.lr.f32    tile i's input [1,3,128,128] (the tiling rule's crop, edge-clamped)
    <case>/tiles/<i>.sr.f32    upstream's raw output on that tile [1,3,512,512]
    <case>/sr-raw.f32          the tiles stitched (the driver's blend) [1,3,4H,4W], raw
    <case>/sr.f32              then colour-matched to the LR, x / 2 + 0.5, clamped: the final image on [0,1]
    <case>/sr.png              sr.f32 quantized as upstream (ToPILImage: x * 255 truncated)
    <case>/full-raw.f32        (sides multiples of 16) upstream test.py on the whole image, raw
    <case>/full.f32, full.png  upstream test.py's final image (colour match, clamp) and its PNG

All .f32 files are little-endian float32, row-major. `tiledVsFull` (whole-image tiled result against upstream's own
whole-image inference) is recorded per case. It measures the tiling, not the export. AdcSR's self-attention and
GroupNorm see the whole input, so a tile's output depends on its window. Cases: four CC0 Wikimedia Commons photos
(source-pins.json), centre-cropped and downscaled with Pillow bicubic to sizes that do and do not divide by the tile
and by 16, including two smaller than the tile, plus one seeded synthetic image. CPU only; no GPU lock needed.
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
import adcsr_graphs as ag  # noqa: E402

FIDELITY_BAR = 40.0       # docs/M4.md's super-resolution bar, applied to tiled-vs-whole-image as the brief asks


def stats(x: np.ndarray) -> dict:
    return {"min": float(x.min()), "max": float(x.max()), "mean": float(x.mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--snapshot", type=Path, default=ag.DEFAULT_SNAPSHOT)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    verified = ag.verify_sources(args.snapshot)
    args.output.mkdir(parents=True)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    model = ag.build_upstream(args.snapshot)

    def run_tile(tile: np.ndarray) -> np.ndarray:
        return model(torch.from_numpy(tile)).numpy()

    index = {
        "schema": 1, "generator": "Forge/adcsr/reference_adcsr.py",
        "scripts": {p.name: ag.sha256_file(p) for p in sorted(HERE.glob("*.py"))},
        "source": {"repository": ag.REPO, "revision": ag.REVISION, "base": ag.BASE["model"], "verified": verified},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "diffusers", "numpy", "pillow")},
        "host": {"machine": platform.machine(), "python": platform.python_version()},
        "precision": "float32 on the CPU (upstream test.py's network, unmodified)",
        "tile": [1, 3, ag.TILE, ag.TILE], "srTile": [1, 3, ag.SR_TILE, ag.SR_TILE], "scale": ag.SCALE,
        "tiling": ag.TILING, "layout": "row-major, little-endian float32",
        "graphInput": "lr = tiles/<i>.lr.f32 as is (float32)",
        "psnr": "10 log10(1 / MSE) on [0,1] images (final), or on raw [-1,1] outputs with peak-to-peak 2",
        "cases": [],
    }
    for case in ag.CASES:
        started = time.monotonic()
        out = args.output / case["name"]
        (out / "tiles").mkdir(parents=True)
        lr_u8 = ag.case_lr(case)
        h, w = lr_u8.shape[:2]
        lr = ag.lr_tensor(lr_u8)
        ag.write_png(out / "lr.png", lr_u8)
        ag.write_f32(out / "input.f32", lr)
        raw, tiles = ag.upscale_tiled(run_tile, lr)
        for i, (x0, y0, sr) in enumerate(tiles):
            ag.write_f32(out / f"tiles/{i}.lr.f32", ag.crop_tile(lr, x0, y0))
            ag.write_f32(out / f"tiles/{i}.sr.f32", sr)
        final = ag.postprocess(raw, lr)
        if not (np.isfinite(raw).all() and np.isfinite(final).all()):
            raise SystemExit(f"{case['name']}: non-finite upstream output")
        ag.write_f32(out / "sr-raw.f32", raw)
        ag.write_f32(out / "sr.f32", final)
        ag.write_png(out / "sr.png", ag.to_uint8(final))
        if "image" in case:
            pin = ag.PINS["referenceImages"]["files"][case["image"]]
            source = {"kind": "photo", "file": case["image"], "sha256": pin["sha256"], "license": pin["license"],
                      "page": pin["page"], "recipe": "adcsr_graphs.photo_lr: Pillow convert('RGB'), centre crop to "
                      "the aspect, resize BICUBIC to the size"}
        else:
            source = {"kind": "synthetic", "seed": case["synthetic"], "recipe": "adcsr_graphs.synthetic_lr"}
        record = {"name": case["name"], "source": source, "size": [w, h], "srSize": [w * ag.SCALE, h * ag.SCALE],
                  "shapes": {"input": [1, 3, h, w], "sr": [1, 3, h * ag.SCALE, w * ag.SCALE]},
                  "tiles": [{"index": i, "x0": x0, "y0": y0} for i, (x0, y0, _) in enumerate(tiles)],
                  "edgeClamped": w < ag.TILE or h < ag.TILE, "sr": stats(final), "srRaw": stats(raw)}
        if w % ag.SIDE_MULTIPLE == 0 and h % ag.SIDE_MULTIPLE == 0:
            full_raw, full = ag.upstream_full(model, lr)
            ag.write_f32(out / "full-raw.f32", full_raw)
            ag.write_f32(out / "full.f32", full)
            ag.write_png(out / "full.png", ag.to_uint8(full))
            value = ag.psnr(final, full)
            record["tiledVsFull"] = {
                "psnr": round(value, 3) if np.isfinite(value) else "inf", "rawPSNR": round(ag.psnr_raw(raw, full_raw), 3),
                "uint8PSNR": round(ag.psnr(ag.to_uint8(final).astype(np.float32),
                                           ag.to_uint8(full).astype(np.float32), 255.0), 3),
                "bar": FIDELITY_BAR, "met": bool(value >= FIDELITY_BAR)}
        else:
            record["tiledVsFull"] = None      # upstream's whole-image inference needs sides that are multiples of 16
        record["files"] = {str(p.relative_to(out)): ag.sha256_file(p) for p in sorted(out.rglob("*")) if p.is_file()}
        record["seconds"] = round(time.monotonic() - started, 2)
        (out / "case.json").write_text(json.dumps(record, indent=2) + "\n")
        index["cases"].append({"name": case["name"], "caseJSONSHA256": ag.sha256_file(out / "case.json")})
        print(f"[reference] {case['name']}: {len(tiles)} tiles, tiledVsFull {record['tiledVsFull'] and record['tiledVsFull']['psnr']}, "
              f"{record['seconds']} s", file=sys.stderr, flush=True)
    index["peakRSSGB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
    (args.output / "references.json").write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""One denoiser segment's run time on the GPU in several text-mask forms (development tooling only).

    Forge/.venv/bin/python -I Forge/krea2/bench_krea2_segment.py --workdir <new dir> --report <new .json> \
        [--index 1] [--variants exact bool]

Builds segment `--index` from the pinned checkpoint (vanilla weights: timing doesn't depend on the adapter), exports
each variant, runs it on the GPU at 1024² (4096 image tokens) with 27 real text rows, and records the median of
`--runs` calls after a warm-up, then deletes the asset. Variants: `exact` (no mask, dynamic text axis at 512 rows: the
cost of the length alone) and `bool` (the masked export: static 512 rows, boolean key mask). One-function assets, so
`bench_krea2_file.py` on a bundle's file shows what a multi-function file costs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import krea2_graphs as kg  # noqa: E402
from export_krea2 import DEFAULT_SNAPSHOT, convert  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--rows", type=int, default=27)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--variants", nargs="*", default=["bool", "exact"], choices=["bool", "exact"])
    args = parser.parse_args()
    if args.workdir.exists() or args.report.exists():
        raise SystemExit("workdir and report must be new")
    args.workdir.mkdir(parents=True)
    import coreai.runtime as rt
    torch.set_grad_enabled(False)
    checkpoint = kg.Checkpoint(DEFAULT_SNAPSHOT)
    rng = np.random.default_rng(0)
    n, length = 4096, kg.MASK_TEXT_ROWS
    hidden = kg.HIDDEN
    image = (rng.standard_normal((n, hidden)) * 0.5).astype(np.float32)
    text = np.zeros((length, hidden), dtype=np.float32)
    text[:args.rows] = (rng.standard_normal((args.rows, hidden)) * 0.5).astype(np.float32)
    cond = (rng.standard_normal((kg.CONDITION_ROWS, hidden)) * 0.1).astype(np.float32)
    rope = kg.rotary_table(64, 64).numpy()
    mask = kg.text_mask(args.rows, length).numpy()
    results, reference = {}, None
    for variant in args.variants:
        masked = variant != "exact"
        seg = kg.DenoiserSegment.load(checkpoint, args.index, masked=masked)
        example, dynamic = seg.example()
        path = args.workdir / f"{variant}.aimodel"
        info = convert(seg, example, dynamic, seg.input_names, seg.output_names, "segment", path)
        del seg
        if args.index == 0:
            embeds = np.zeros((length, kg.CONDITIONING), dtype=np.float16)
            embeds[:args.rows] = (rng.standard_normal((args.rows, kg.CONDITIONING)) * 0.1).astype(np.float16)
            feeds = {"latents": rng.standard_normal((n, kg.LATENT_CHANNELS)).astype(np.float16), "text_embeds": embeds,
                     "sigma": np.array([0.75], dtype=np.float32), "rope_image": rope}
        else:
            feeds = {f"image_{args.index}": image, f"text_{args.index}": text, "cond": cond, "rope_image": rope}
        if masked:
            feeds["text_mask"] = mask

        async def go():
            options = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
            model = await rt.AIModel.load(path, options)
            fn = model.load_function("segment")
            inputs = {k: rt.NDArray(np.ascontiguousarray(v)) for k, v in feeds.items()}
            times, out = [], None
            for i in range(args.runs + 1):
                t = time.monotonic()
                out = {k: v.numpy().copy() for k, v in (await fn(inputs, {})).items()}
                times.append(time.monotonic() - t)
            return times, out
        times, out = asyncio.run(go())
        img = out[f"image_{args.index + 1}"]
        if variant == "bool":
            reference = img
        results[variant] = {"firstSeconds": round(times[0], 2), "medianSeconds": round(statistics.median(times[1:]), 3),
                            "convertSeconds": info["convertSeconds"], "bytes": info["bytes"],
                            "finite": bool(np.isfinite(img).all())}
        if reference is not None and variant != "bool":
            a, b = reference.astype(np.float64).ravel(), img.astype(np.float64).ravel()
            results[variant]["cosineToBool"] = round(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))), 6)
        print(f"[segment] {variant}: {results[variant]}", file=sys.stderr, flush=True)
        shutil.rmtree(path)
    args.report.write_text(json.dumps({"index": args.index, "rows": args.rows, "imageTokens": n, "textLength": length,
                                       "variants": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Denoiser step time on the GPU, with only the four segments loaded (development tooling only).

    Forge/.venv/bin/python -I Forge/krea2/bench_krea2_steps.py --bundle <bundle> --rows 27 --report <new .json> \
        [--size 1024] [--steps 6] [--function segment]

Loads the bundle's four denoiser segments (one function; no text encoder, no VAE), then runs `--steps` chained
steps (the four segments, carries fed forward, the host Euler update) at the canvas's image tokens with `--rows`
conditioning rows. Masked bundles take the rows padded to the function's fixed length with `text_mask`; exact-length
bundles run at `--rows` itself, so `--rows 512` on an exact-length bundle measures a 512-row text axis without a mask.
Inputs are small random values (timing is data-independent). Records each step's seconds and `vm.swapusage` before
and after: the first step includes Core AI's preparation of the shape.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import krea2_graphs as kg  # noqa: E402
from check_krea2 import Graph  # noqa: E402


def swap() -> str:
    return subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--function", default="segment")
    args = parser.parse_args()
    if args.report.exists():
        raise SystemExit("report already exists; choose a new path")
    manifest = json.loads((args.bundle / ".aether/aether.json").read_text())
    files = {role: args.bundle / g["file"].split(":", 1)[1] for role, g in manifest["graphs"].items()}
    masked = "textMask" in manifest["graphs"]["denoiser0"]["inputs"]
    length = {"segment": kg.MASK_TEXT_ROWS, "segmentReferences": kg.REFERENCE_FUNCTION_ROWS}[args.function] if masked \
        else args.rows
    if args.rows > length:
        raise SystemExit(f"{args.rows} rows don't fit {length}")
    record = {"bundle": str(args.bundle), "masked": masked, "function": args.function, "rows": args.rows,
              "textLength": length, "size": args.size, "swapBefore": swap()}
    t0 = time.monotonic()
    segments = [Graph(files[f"denoiser{i}"], args.function) for i in range(4)]
    record["loadSeconds"] = round(time.monotonic() - t0, 1)
    side = args.size // kg.PIXELS_PER_TOKEN
    n = side * side
    rng = np.random.default_rng(0)
    text = np.zeros((length, kg.CONDITIONING), dtype=np.float16)
    text[:args.rows] = (rng.standard_normal((args.rows, kg.CONDITIONING)) * 0.1).astype(np.float16)
    extra = {"text_mask": kg.text_mask(args.rows, length).numpy()} if masked else {}
    rope = kg.rotary_table(side, side).numpy()
    latents = rng.standard_normal((n, kg.LATENT_CHANNELS)).astype(np.float32)
    sigmas, timesteps = kg.schedule(kg.STEPS)
    seconds = []
    for s in range(args.steps):
        t = time.monotonic()
        k = s % kg.STEPS
        sigma = np.array([timesteps[k]], dtype=np.float32)
        o0 = segments[0](latents=latents.astype(np.float16), text_embeds=text, sigma=sigma, rope_image=rope, **extra)
        o1 = segments[1](image_1=o0["image_1"], text_1=o0["text_1"], cond=o0["cond"], rope_image=rope, **extra)
        o2 = segments[2](image_2=o1["image_2"], text_2=o1["text_2"], cond=o0["cond"], rope_image=rope, **extra)
        o3 = segments[3](image_3=o2["image_3"], text_3=o2["text_3"], cond=o0["cond"], rope_image=rope, **extra)
        v = o3["velocity"].astype(np.float32)
        latents = latents + (np.float32(sigmas[k + 1]) - np.float32(sigmas[k])) * v
        seconds.append(round(time.monotonic() - t, 3))
        print(f"[bench] step {s}: {seconds[-1]} s finite {bool(np.isfinite(v).all())}", file=sys.stderr, flush=True)
    for g in segments:
        g.close()
    later = seconds[1:]
    record.update({"stepSeconds": seconds, "firstStepSeconds": seconds[0],
                   "laterStepMeanSeconds": round(sum(later) / max(1, len(later)), 3), "swapAfter": swap()})
    args.report.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()

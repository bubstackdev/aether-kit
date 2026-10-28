"""Tiled VAE decoding against one 60 s window, at a chosen precision: overlap, crossfade, time and memory.

usage: USE_OS_COREAI=1 tiled_vae_coreai.py <export-dir> <references-dir> <coreai-dir> <out.json>
           [--window vae-dec-250-fp32] [--full vae-dec-1500-fp32] [--overlaps 16,32,64] [--no-full]
Runs under the GPU lock, one process per precision so /usr/bin/time -l gives each its own peak footprint. For each
prompt it decodes the Core AI trajectory's final latents (<id>-coreai.npz) in windows of the window asset's length,
stepping by window − overlap, with a linear crossfade across each overlap (the last window right-aligned), and compares
the result with one decode by the full-length asset: SNR, the gate's STFT distance and log-mel cosine. --no-full
loads and runs only the window asset, so /usr/bin/time -l reports the tiler's own footprint.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import coreai.runtime as rt

sys.path.insert(0, str(Path(__file__).parent))
from gate_trajectory import audio_metrics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("exports")
ap.add_argument("references")
ap.add_argument("coreai")
ap.add_argument("out")
ap.add_argument("--window", default="vae-dec-250-fp32")
ap.add_argument("--full", default="vae-dec-1500-fp32")
ap.add_argument("--overlaps", default="16,32,64")
ap.add_argument("--no-full", action="store_true")
a = ap.parse_args() if __name__ == "__main__" else None
HOP = 1920


async def tile_decode(fn, latents, window, overlap, dt):
    """latents [1, 64, T] → audio [2, T·1920]: windows stepping by window − overlap, the last right-aligned, with a
    linear crossfade across each overlap."""
    frames = latents.shape[2]
    starts = list(range(0, frames - window + 1, window - overlap))
    if starts[-1] + window < frames:
        starts.append(frames - window)
    out = np.zeros((2, frames * HOP))
    weight = np.zeros(frames * HOP)
    for k, start in enumerate(starts):
        piece = (await fn({"latents": rt.NDArray(latents[:, :, start:start + window].astype(dt))}))[
            "audio"].numpy()[0].astype(np.float64)
        ramp = np.ones(window * HOP)
        lead = (starts[k - 1] + window - start) * HOP if k else 0
        if lead > 0:
            ramp[:lead] = np.linspace(0, 1, lead + 2)[1:-1]
        trail = (start + window - starts[k + 1]) * HOP if k + 1 < len(starts) else 0
        if trail > 0:
            ramp[-trail:] = np.minimum(ramp[-trail:], np.linspace(1, 0, trail + 2)[1:-1])
        out[:, start * HOP:(start + window) * HOP] += piece * ramp
        weight[start * HOP:(start + window) * HOP] += ramp
    return (out / np.maximum(weight, 1e-12)).astype(np.float32), len(starts)


async def main():
    E, REF, C = Path(a.exports), Path(a.references), Path(a.coreai)
    opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    window_fn = (await rt.AIModel.load(E / f"{a.window}.aimodel", opts)).load_function("main")
    full_fn = None if a.no_full else (await rt.AIModel.load(E / f"{a.full}.aimodel", opts)).load_function("main")
    dt = np.float16 if a.window.endswith("fp16") else np.float32
    window = int(a.window.split("-")[2])
    report = {"window": a.window, "full": None if a.no_full else a.full, "prompts": {}}
    for path in sorted(REF.glob("*.npz")):
        latents = np.load(C / f"{path.stem}-coreai.npz")["coreai_x0"].transpose(0, 2, 1).astype(np.float32)
        rep = {"tiled": {}}
        whole = None
        if full_fn is not None:
            t0 = time.perf_counter()
            whole = (await full_fn({"latents": rt.NDArray(latents.astype(dt))}))["audio"].numpy()[0].astype(np.float32)
            rep["fullSeconds"] = time.perf_counter() - t0
        for overlap in [int(v) for v in a.overlaps.split(",")]:
            t0 = time.perf_counter()
            tiled, windows = await tile_decode(window_fn, latents, window, overlap, dt)
            rep["tiled"][str(overlap)] = {"windows": windows, "seconds": time.perf_counter() - t0,
                                          **(audio_metrics(tiled, whole) if whole is not None else {})}
        report["prompts"][path.stem] = rep
        print(path.stem, json.dumps(rep), flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    asyncio.run(main())

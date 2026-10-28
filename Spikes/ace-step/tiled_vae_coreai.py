"""Tiled VAE decoding against one 60 s window, at a chosen precision: overlap, crossfade, time and memory.

usage: USE_OS_COREAI=1 tiled_vae_coreai.py <export-dir> <references-dir> <coreai-dir> <out.json>
           [--window vae-dec-250-fp32] [--full vae-dec-1500-fp32] [--overlaps 16,32,64]
Runs under the GPU lock, one process per precision so /usr/bin/time -l gives each its own peak footprint. For each
prompt it decodes the Core AI trajectory's final latents (<id>-coreai.npz) in windows of the window asset's length,
stepping by window − overlap, with a linear crossfade across each overlap (the last window right-aligned), and compares
the result with one decode by the full-length asset: SNR, the gate's STFT distance and log-mel cosine.
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
a = ap.parse_args()
E, REF, C = Path(a.exports), Path(a.references), Path(a.coreai)
HOP = 1920


async def main():
    opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    window_fn = (await rt.AIModel.load(E / f"{a.window}.aimodel", opts)).load_function("main")
    full_fn = (await rt.AIModel.load(E / f"{a.full}.aimodel", opts)).load_function("main")
    dt = np.float16 if a.window.endswith("fp16") else np.float32
    window = int(a.window.split("-")[2])
    report = {"window": a.window, "full": a.full, "prompts": {}}
    for path in sorted(REF.glob("*.npz")):
        latents = np.load(C / f"{path.stem}-coreai.npz")["coreai_x0"].transpose(0, 2, 1).astype(np.float32)
        frames = latents.shape[2]
        t0 = time.perf_counter()
        whole = (await full_fn({"latents": rt.NDArray(latents.astype(dt))}))["audio"].numpy()[0].astype(np.float32)
        rep = {"fullSeconds": time.perf_counter() - t0, "tiled": {}}
        for overlap in [int(v) for v in a.overlaps.split(",")]:
            starts = list(range(0, frames - window + 1, window - overlap))
            if starts[-1] + window < frames:
                starts.append(frames - window)
            out = np.zeros((2, frames * HOP))
            weight = np.zeros(frames * HOP)
            t0 = time.perf_counter()
            for k, start in enumerate(starts):
                piece = (await window_fn({"latents": rt.NDArray(latents[:, :, start:start + window].astype(dt))}))[
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
            tiled = (out / np.maximum(weight, 1e-12)).astype(np.float32)
            rep["tiled"][str(overlap)] = {"windows": len(starts), "seconds": time.perf_counter() - t0,
                                          **audio_metrics(tiled, whole)}
        report["prompts"][path.stem] = rep
        print(path.stem, json.dumps(rep), flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1) + "\n")


asyncio.run(main())

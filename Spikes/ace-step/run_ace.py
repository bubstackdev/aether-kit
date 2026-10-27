"""Time the exported ACE-Step 1.5 stages on the GPU and compare with eager fp32 references.

usage: USE_OS_COREAI=1 run_ace.py <export-dir> <out.json> <asset-name>…   (run under the GPU lock)
"""
import asyncio
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import coreai.runtime as rt

E, OUT = Path(sys.argv[1]), Path(sys.argv[2])
report = json.loads(OUT.read_text()) if OUT.exists() else {}


def stats(x, ref):
    x64, r64 = x.astype(np.float64).ravel(), ref.astype(np.float64).ravel()
    return {"relL2": float(np.linalg.norm(x64 - r64) / np.linalg.norm(r64)),
            "maxAbs": float(np.abs(x64 - r64).max()), "refMaxAbs": float(np.abs(r64).max()),
            "cosine": float(x64 @ r64 / (np.linalg.norm(x64) * np.linalg.norm(r64))),
            "finite": bool(np.isfinite(x64).all())}


async def run(name):
    opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    t0 = time.perf_counter()
    model = await rt.AIModel.load(E / f"{name}.aimodel", opts)
    fn = model.load_function("main")
    load = time.perf_counter() - t0
    dt = np.float16 if name.endswith("fp16") else np.float32
    if name.startswith("vae-dec"):
        frames = int(name.split("-")[2])
        feeds = {"latents": np.load(E / f"vae-ref-latents-{frames}.npy")}
        ref, key = np.load(E / f"vae-ref-audio-{frames}.npy"), "audio"
        audio_seconds = frames / 25
    else:
        shape = name.split("-")[1]
        feeds = {n: np.load(E / f"dit-ref-{n}-{shape}.npy") for n in ("xt", "t", "context", "encoder")}
        ref, key = np.load(E / f"dit-ref-v-{shape}.npy"), "v"
        audio_seconds = int(shape.split("x")[0]) / 25
    feed = {k: rt.NDArray(v.astype(dt)) for k, v in feeds.items()}
    times = []
    for _ in range(4):
        t = time.perf_counter()
        y = (await fn(feed))[key].numpy()
        times.append(time.perf_counter() - t)
    r = {"loadSeconds": load, "firstSeconds": times[0], "warmMedianSeconds": float(np.median(times[1:])),
         "audioSeconds": audio_seconds, "outputShape": list(y.shape), **stats(y.astype(np.float32), ref),
         "maxRSSBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    report[name] = r
    print(name, json.dumps(r))
    OUT.write_text(json.dumps(report, indent=1))


async def main():
    for name in sys.argv[3:]:
        try:
            await run(name)
        except Exception as e:      # record failures as evidence, keep going
            report[name] = {"error": repr(e)[:2000]}
            print(name, "FAILED", repr(e)[:500])
            OUT.write_text(json.dumps(report, indent=1))


asyncio.run(main())

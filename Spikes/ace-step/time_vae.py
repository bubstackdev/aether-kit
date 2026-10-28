"""Warm decode time of the 60 s VAE window in fp16 and fp32 (the fp16 VAE's noise floor fails the gate's log-spectral
bar, so the choice is between them). Decodes one reference's fp32 latents with each asset (a shorter window asset gets
the first window of them): one cold call, then five warm calls. Runs under the GPU lock; give one asset per process to
read its own peak footprint from /usr/bin/time -l.

usage: USE_OS_COREAI=1 time_vae.py <export-dir> <reference.npz> <out.json> [asset ...]
       (default assets: vae-dec-1500-fp16 vae-dec-1500-fp32)
"""
import asyncio
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import coreai.runtime as rt

E, REF, OUT = Path(sys.argv[1]), np.load(sys.argv[2]), Path(sys.argv[3])
ASSETS = sys.argv[4:] or ["vae-dec-1500-fp16", "vae-dec-1500-fp32"]


async def main():
    opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    report = {}
    for name in ASSETS:
        dt = np.float16 if name.endswith("fp16") else np.float32
        frames = int(name.split("-")[2])
        t0 = time.perf_counter()
        fn = (await rt.AIModel.load(E / f"{name}.aimodel", opts)).load_function("main")
        load = time.perf_counter() - t0
        latents = rt.NDArray(REF["fp32_x0"].transpose(0, 2, 1)[:, :, :frames].astype(dt))
        times = []
        for _ in range(6):
            t0 = time.perf_counter()
            (await fn({"latents": latents}))["audio"].numpy()
            times.append(time.perf_counter() - t0)
        report[name] = {"loadSeconds": load, "firstSeconds": times[0], "warmSeconds": times[1:],
                        "warmMedianSeconds": float(np.median(times[1:])),
                        "maxRSSAfter": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        print(name, json.dumps(report[name]), flush=True)
    OUT.write_text(json.dumps(report, indent=1) + "\n")


asyncio.run(main())

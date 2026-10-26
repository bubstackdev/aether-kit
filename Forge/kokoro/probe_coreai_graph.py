#!/usr/bin/env python3
"""Loads one exported Kokoro graph with a compute preference and runs it on synthetic inputs.

Development bisection probe for Core AI specialization failures (run under the GPU lock).
"""
import argparse
import asyncio
import sys
import time

import numpy as np


async def main(args):
    import coreai.runtime as rt

    if args.compute == "cpu":
        options = rt.SpecializationOptions.cpu_only()
    elif args.compute == "gpu":
        options = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    else:
        options = rt.SpecializationOptions.default()
    t0 = time.monotonic()
    model = await rt.AIModel.load(args.asset, options)
    fn = model.load_function(args.function)
    print(f"loaded {args.function} in {time.monotonic() - t0:.2f}s", flush=True)
    rng = np.random.default_rng(0)
    if args.function == "durations":
        feed = {"input_ids": np.array([[0] + [43] * (args.length - 2) + [0]], np.int32),
                "style": rng.standard_normal((1, 256)).astype(np.float32) * 0.1}
    else:
        f = args.length
        feed = {"prosody": rng.standard_normal((1, 640, f)).astype(np.float32),
                "text": rng.standard_normal((1, 512, f)).astype(np.float32),
                "style": rng.standard_normal((1, 256)).astype(np.float32) * 0.1,
                "noise": rng.standard_normal((1, f, 600, 9)).astype(np.float32)}
    for i in range(2):
        t1 = time.monotonic()
        out = await fn({k: rt.NDArray(v) for k, v in feed.items()})
        shapes = {k: v.numpy().shape for k, v in out.items()}
        print(f"run {i}: {time.monotonic() - t1:.3f}s {shapes}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("asset")
    ap.add_argument("function", choices=["durations", "decode"])
    ap.add_argument("--compute", choices=["cpu", "gpu", "default"], default="gpu")
    ap.add_argument("--length", type=int, default=31)
    asyncio.run(main(ap.parse_args()))
    sys.exit(0)

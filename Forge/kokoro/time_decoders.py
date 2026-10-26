#!/usr/bin/env python3
"""Interleaved GPU timing of Kokoro decoder assets at given shapes (development probe; GPU lock).

Each entry is ASSET:SHAPE[:VALID]; VALID marks a bucketed decoder that takes a `frames` input.
Runs one warm-up per entry, then REPEATS rounds that call every entry once, and prints medians.
"""
import argparse
import asyncio
import json
import statistics
import time

import numpy as np


async def main(args):
    import coreai.runtime as rt

    options = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    entries = []
    for spec in args.entries:
        parts = spec.split(":")
        asset, shape = parts[0], int(parts[1])
        valid = int(parts[2]) if len(parts) > 2 else None
        model = await rt.AIModel.load(asset, options)
        fn = model.load_function("decode")
        rng = np.random.default_rng(0)
        feed = {"prosody": rng.standard_normal((1, 640, shape)).astype(np.float32),
                "text": rng.standard_normal((1, 512, shape)).astype(np.float32),
                "style": (rng.standard_normal((1, 256)) * 0.1).astype(np.float32),
                "noise": rng.standard_normal((1, shape, 600, 9)).astype(np.float32)}
        if valid is not None:
            feed["frames"] = np.array([valid], np.int32)
        arrays = {k: rt.NDArray(v) for k, v in feed.items()}
        t0 = time.monotonic()
        await fn(arrays)
        entries.append({"spec": spec, "fn": fn, "arrays": arrays, "first": time.monotonic() - t0, "times": []})
    for _ in range(args.repeats):
        for e in entries:
            t0 = time.monotonic()
            await e["fn"](e["arrays"])
            e["times"].append(time.monotonic() - t0)
    print(json.dumps([{"spec": e["spec"].split("/")[-3] + ":" + ":".join(e["spec"].split(":")[1:]), "first_s": round(e["first"], 3),
                       "median_s": round(statistics.median(e["times"]), 3), "min_s": round(min(e["times"]), 3)} for e in entries], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("entries", nargs="+")
    ap.add_argument("--repeats", type=int, default=5)
    asyncio.run(main(ap.parse_args()))

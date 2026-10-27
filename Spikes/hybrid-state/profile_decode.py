"""Capture Core AI runtime profiler events for a few decode steps of one asset/function (GPU, OS Core AI runtime).

usage: USE_OS_COREAI=1 profile_decode.py <asset> <function> <out.jsonl> [steps]
Runs under the GPU lock. The states start at zero; the token/position inputs are a plausible running sequence.
"""
import asyncio
import json
import time
import sys
from pathlib import Path

import numpy as np
import coreai.runtime as rt

asset, fname, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 6
events = []


def on_event(e):
    events.append({"kind": "event", "t": e.timestamp, "id": e.event_id, "phase": e.phase, "data": e.data,
                   "meta": dict(e.metadata)})


def on_begin(e):
    events.append({"kind": "begin", "t": e.timestamp, "id": e.event_id, "phase": e.phase, "data": e.data,
                   "meta": dict(e.metadata)})
    return len(events)


def on_end(e, interval):
    events.append({"kind": "end", "t": e.timestamp, "id": e.event_id, "phase": e.phase, "data": e.data,
                   "meta": dict(e.metadata), "interval": interval})


async def main():
    opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    model = await rt.AIModel.load(asset, opts)
    prof = rt.Profiler(on_log_event=on_event, on_log_event_begin=on_begin, on_log_event_end=on_end)
    fn = model.load_function(fname, profiler=prof)
    desc = fn.desc if not callable(getattr(fn, "desc", None)) else fn.desc()
    states = {}
    for name in desc.state_names:
        d = desc.state_descriptor(name)
        shape = [s if s is not None and s >= 0 else 1024 for s in d.shape]
        dtype = np.float16 if "16" in d.dtype else np.float32
        states[name] = rt.NDArray(np.zeros(shape, dtype=dtype))
    scalar = list(desc.input_descriptor("position_ids").shape) == [1, 1]
    for i in range(steps):
        marker = len(events)
        pos = np.array([[i]], np.int32) if scalar else np.arange(i + 1, dtype=np.int32)[None]
        feed = {"input_ids": rt.NDArray(np.array([[100 + i]], np.int32)), "position_ids": rt.NDArray(pos)}
        t0 = time.perf_counter_ns()
        await fn(feed, states)
        events.append({"kind": "step", "step": i, "firstEvent": marker, "wallNs": time.perf_counter_ns() - t0})
    with out.open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    print(len(events), "events")


asyncio.run(main())

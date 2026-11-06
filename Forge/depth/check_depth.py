#!/usr/bin/env python3
"""The Depth Anything 3 small gate: the exported Core AI graph (GPU) against the upstream CPU float32 references.

    Forge/.venv/bin/python -I Forge/depth/check_depth.py --bundle <bundle dir> --reference <reference dir> \
        --report <new check-report.json>

With the bundle's own manifest (graph file, function, role names, compute placement), for every reference case
(reference_depth.py): the case's input tensor, converted to the graph's float16 input, gives
- depth relative L2 <= 0.02 against upstream (docs/M4.md's starting depth bar);
- confidence relative L2 <= 0.02 (the optional `confidence` role, held to the same bar);
- a repeat run that is bit-identical.
The graph's descriptor is also checked against the manifest's assertions: image [1,3,504,504] float16; depth and
depth_conf [1,504,504]. Writes each case's graph outputs beside the report (`<report stem>/<case>/*.f32`). Exits nonzero
when a bar fails. Runs one graph on the GPU for a few seconds.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import depth_graphs as dg  # noqa: E402

DEPTH_BAR, CONFIDENCE_BAR = 0.02, 0.02


class Graph:
    def __init__(self, path: Path, function: str, compute: str) -> None:
        import coreai.runtime as rt
        self.rt = rt
        options = (rt.SpecializationOptions.cpu_only() if compute == "cpu" else
                   rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
        self.model = asyncio.run(rt.AIModel.load(path, options))
        self.function = self.model.load_function(function)

    def descriptor(self) -> dict:
        d = self.function.desc
        describe = lambda x: {"dtype": str(x.dtype), "shape": list(x.shape)}  # noqa: E731
        return {"inputs": {n: describe(d.input_descriptor(n)) for n in d.input_names},
                "outputs": {n: describe(d.output_descriptor(n)) for n in d.output_names}}

    def __call__(self, **feeds: np.ndarray) -> dict[str, np.ndarray]:
        rt = self.rt

        async def go():
            out = await self.function({k: rt.NDArray(np.ascontiguousarray(v)) for k, v in feeds.items()}, {})
            return {k: v.numpy().copy() for k, v in out.items()}
        return asyncio.run(go())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise SystemExit("report already exists; choose a new path")
    manifest = json.loads((args.bundle / ".aether/aether.json").read_text())
    index = json.loads((args.reference / "references.json").read_text())
    spec = manifest["graphs"]["estimator"]
    image_name = spec["inputs"]["image"]
    depth_name, conf_name = spec["outputs"]["depth"], spec["outputs"]["confidence"]
    dump = args.report.with_suffix("")
    checks: list[dict] = []

    def record(name: str, value: float, bar: float, passed: bool, **extra) -> None:
        checks.append({"check": name, "value": value, "bar": bar, "passed": bool(passed), **extra})
        print(f"[check] {'PASS' if passed else 'FAIL'} {name}: {value} (bar {bar})", file=sys.stderr, flush=True)

    started = time.monotonic()
    t0 = time.monotonic()
    graph = Graph(args.bundle / spec["file"].split(":", 1)[1], spec["function"], spec["compute"])
    load_seconds = round(time.monotonic() - t0, 2)
    descriptor = graph.descriptor()
    shapes = index["shapes"]
    expected = {"inputs": {image_name: {"dtype": "float16", "shape": shapes["input"]}},
                "outputs": {depth_name: {"shape": shapes["depth"]}, conf_name: {"shape": shapes["confidence"]}}}
    ok = (descriptor["inputs"].get(image_name, {}).get("dtype", "").endswith("float16")
          and descriptor["inputs"][image_name]["shape"] == shapes["input"]
          and all(descriptor["outputs"].get(n, {}).get("shape") == s["shape"] for n, s in expected["outputs"].items()))
    record("descriptor", float(ok), 1.0, ok, descriptor=descriptor)

    timings = []
    for entry in index["cases"]:
        case = json.loads((args.reference / entry["name"] / "case.json").read_text())
        folder = args.reference / case["name"]
        x = dg.read_f32(folder / "input.f32", shapes["input"])
        want_depth = dg.read_f32(folder / "depth.f32", shapes["depth"])
        want_conf = dg.read_f32(folder / "confidence.f32", shapes["confidence"])
        feed = x.astype(np.float16)
        if not np.array_equal(feed.astype(np.float32), x):
            raise SystemExit(f"{case['name']}: input.f32 is not fp16-representable")
        t0 = time.monotonic()
        out = graph(**{image_name: feed})
        timings.append(round(time.monotonic() - t0, 4))
        got_depth = out[depth_name].astype(np.float32).reshape(shapes["depth"])
        got_conf = out[conf_name].astype(np.float32).reshape(shapes["confidence"])
        e = dg.rel_l2(got_depth, want_depth)
        record(f"{case['name']}.depth.relL2", round(e, 6), DEPTH_BAR, e <= DEPTH_BAR,
               maxAbs=float(np.abs(got_depth - want_depth).max()), dtype=str(out[depth_name].dtype))
        c = dg.rel_l2(got_conf, want_conf)
        record(f"{case['name']}.confidence.relL2", round(c, 6), CONFIDENCE_BAR, c <= CONFIDENCE_BAR,
               maxAbs=float(np.abs(got_conf - want_conf).max()))
        again = graph(**{image_name: feed})
        same = all(np.array_equal(again[k], out[k]) for k in (depth_name, conf_name))
        record(f"{case['name']}.repeat.exact", float(same), 1.0, same)
        (dump / case["name"]).mkdir(parents=True, exist_ok=True)
        dg.write_f32(dump / case["name"] / "depth.f32", got_depth)
        dg.write_f32(dump / case["name"] / "confidence.f32", got_conf)
    del graph
    passed = all(c["passed"] for c in checks)
    args.report.write_text(json.dumps({"passed": passed, "seconds": round(time.monotonic() - started, 1),
                                       "loadSeconds": load_seconds, "runSeconds": timings,
                                       "bundle": str(args.bundle), "reference": str(args.reference),
                                       "aetherJSONSHA256": dg.sha256_file(args.bundle / ".aether/aether.json"),
                                       "checks": checks}, indent=2) + "\n")
    if not passed:
        raise SystemExit("parity check failed")


if __name__ == "__main__":
    main()

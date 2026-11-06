#!/usr/bin/env python3
"""The GLiNER2.5-Decide gate: the exported Core AI graph (GPU) against gliner2's CPU float32 references.

    Forge/.venv/bin/python -I Forge/gliner/check_gliner.py --bundle <bundle dir> --reference <reference dir> \
        --report <new check-report.json>

With the bundle's own manifest (graph file, bucket functions, role names, compute placement), for every reference case
(reference_gliner.py), fed the case's padded int32 inputs at its bucket:
- every task's probabilities within 0.01 of upstream's, the same argmax and the same decision (docs/M4.md's GLiNER bar;
  multi-label decisions at the task's threshold);
- every case that fits 256 ids also through the s512 function, under the same bars (both buckets cover it);
- repeat runs bit-identical (the first 16 cases per bucket).
Also: each function's descriptor matches the manifest's assertions (input_ids, attention_mask int32 [1,S], label_idx
int32 [1,32]; logits float32 [1,32]). Exits nonzero when a bar fails.

GPU use: the whole run holds the shared GPU lease (/tmp/atelier-gpu.lock, the mkdir protocol of
Sources/AetherHost/GPU/SharedGPULease.swift: mkdir, then the pid written to pid.tmp.<pid> and renamed to pid; released
by moving the directory aside while pid still names this process). A lease held by a live process is waited for; one
whose holder is gone is reported, never reclaimed here. A few minutes of one graph on the GPU.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import signal
import statistics
import sys
import time
import uuid
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gliner_graphs as gg  # noqa: E402

PROB_BAR = 0.01
REPEATS = 16
LEASE = Path("/tmp/atelier-gpu.lock")


def _holder() -> int | None:
    try:
        text = (LEASE / "pid").read_text()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextlib.contextmanager
def gpu_lease(poll: float = 0.2):
    pid = os.getpid()
    reported = None
    while True:
        try:
            os.mkdir(LEASE, 0o755)
            break
        except FileExistsError:
            holder = _holder()
            if holder is not None and not _alive(holder):
                raise SystemExit(f"{LEASE} names pid {holder}, which no longer exists; remove it by hand if no "
                                 "process is using the GPU")
            if holder != reported:
                print(f"[check] waiting for the GPU lease (holder {holder})", file=sys.stderr, flush=True)
                reported = holder
            time.sleep(poll)
    temporary = LEASE / f"pid.tmp.{pid}"
    temporary.write_text(str(pid))
    os.rename(temporary, LEASE / "pid")
    previous = {s: signal.signal(s, lambda *_, code=128 + s: sys.exit(code)) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        yield
    finally:
        if _holder() == pid:
            aside = LEASE.with_name(f"{LEASE.name}.released.{pid}.{uuid.uuid4().hex}")
            os.rename(LEASE, aside)
            shutil.rmtree(aside, ignore_errors=True)
        for s, handler in previous.items():
            signal.signal(s, handler)


class Graph:
    def __init__(self, path: Path, functions: dict[int, str], compute: str) -> None:
        import coreai.runtime as rt
        self.rt = rt
        options = (rt.SpecializationOptions.cpu_only() if compute == "cpu" else
                   rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
        self.model = asyncio.run(rt.AIModel.load(path, options))
        self.functions = {s: self.model.load_function(f) for s, f in functions.items()}

    def descriptor(self, sequence: int) -> dict:
        d = self.functions[sequence].desc
        describe = lambda x: {"dtype": str(x.dtype), "shape": list(x.shape)}  # noqa: E731
        return {"inputs": {n: describe(d.input_descriptor(n)) for n in d.input_names},
                "outputs": {n: describe(d.output_descriptor(n)) for n in d.output_names}}

    def __call__(self, sequence: int, **feeds: np.ndarray) -> dict[str, np.ndarray]:
        rt = self.rt

        async def go():
            out = await self.functions[sequence]({k: rt.NDArray(np.ascontiguousarray(v)) for k, v in feeds.items()},
                                                 {})
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
    spec = manifest["graphs"]["classifier"]
    names = (spec["inputs"]["tokens"], spec["inputs"]["attentionMask"], spec["inputs"]["labelPositions"])
    out_name = spec["outputs"]["labelLogits"]
    functions = {e["value"]: e["function"] for e in spec["buckets"]["entries"]}
    if spec["buckets"]["axis"] != "sequence" or sorted(functions) != index["sequences"] \
            or spec["limits"]["staticSequence"] != index["sequences"]:
        raise SystemExit(f"manifest buckets {functions} differ from the reference's sequences {index['sequences']}")
    checks: list[dict] = []

    def record(name: str, value: float, bar: float, passed: bool, **extra) -> None:
        checks.append({"check": name, "value": value, "bar": bar, "passed": bool(passed), **extra})
        if not passed:
            print(f"[check] FAIL {name}: {value} (bar {bar})", file=sys.stderr, flush=True)

    started = time.monotonic()
    with gpu_lease():
        t0 = time.monotonic()
        graph = Graph(args.bundle / spec["file"].split(":", 1)[1], functions, spec["compute"])
        load_seconds = round(time.monotonic() - t0, 2)
        descriptors = {}
        for s in functions:
            d = descriptors[s] = graph.descriptor(s)
            ok = (all(d["inputs"].get(n, {}).get("dtype", "").endswith("int32") for n in names)
                  and d["inputs"][names[0]]["shape"] == [1, s] and d["inputs"][names[1]]["shape"] == [1, s]
                  and d["inputs"][names[2]]["shape"] == [1, gg.MAX_LABELS]
                  and d["outputs"].get(out_name, {}).get("dtype", "").endswith("float32")
                  and d["outputs"][out_name]["shape"] == [1, gg.MAX_LABELS])
            record(f"descriptor.s{s}", float(ok), 1.0, ok, descriptor=d)

        timings: dict[int, list[float]] = {s: [] for s in functions}
        repeated = {s: 0 for s in functions}
        rows = []
        for case in index["cases"]:
            ids, positions = np.asarray(case["inputIds"], np.int32), np.asarray(case["labelPositions"], np.int32)
            for s in sorted(functions):
                if s < case["bucket"]:
                    continue
                feeds = dict(zip(names, gg.graph_inputs(ids, positions, s)))
                t = time.monotonic()
                out = graph(s, **feeds)
                timings[s].append(time.monotonic() - t)
                logits = out[out_name].astype(np.float32).reshape(1, gg.MAX_LABELS)
                metrics = gg.task_metrics(logits, case["results"])
                where = "own" if s == case["bucket"] else "cross"
                name = f"{case['name']}@s{s}"
                worst = max(m["maxAbsProb"] for m in metrics)
                same = all(m["sameArgmax"] for m in metrics)
                decided = all(m["sameDecision"] for m in metrics)
                finite = bool(np.isfinite(logits).all())
                record(f"{name}.maxAbsProb", round(worst, 7), PROB_BAR, finite and worst <= PROB_BAR)
                record(f"{name}.sameArgmax", float(same), 1.0, same)
                record(f"{name}.sameDecision", float(decided), 1.0, decided)
                rows.append({"case": case["name"], "set": case["set"], "sequence": s, "bucket": where,
                             "seqLen": case["seqLen"], "labels": case["labels"], "maxAbsProb": worst,
                             "maxAbsLogit": max(m["maxAbsLogit"] for m in metrics), "sameArgmax": same,
                             "sameDecision": decided, "tasks": len(metrics)})
                if repeated[s] < REPEATS:
                    again = graph(s, **feeds)
                    exact = np.array_equal(again[out_name], out[out_name])
                    record(f"{name}.repeat.exact", float(exact), 1.0, exact)
                    repeated[s] += 1
        del graph

    summary = {}
    for (s, where) in sorted({(r["sequence"], r["bucket"]) for r in rows}):
        group = [r for r in rows if r["sequence"] == s and r["bucket"] == where]
        summary[f"s{s}.{where}"] = {
            "cases": len(group), "tasks": sum(r["tasks"] for r in group),
            "maxAbsProb": max(r["maxAbsProb"] for r in group), "maxAbsLogit": max(r["maxAbsLogit"] for r in group),
            "sameArgmax": sum(r["sameArgmax"] for r in group), "sameDecision": sum(r["sameDecision"] for r in group),
            "medianMs": round(1000 * statistics.median(timings[s]), 1) if timings[s] else None}
    by_set = {}
    for r in rows:
        if r["bucket"] != "own":
            continue
        g = by_set.setdefault(r["set"], {"cases": 0, "maxAbsProb": 0.0, "maxAbsLogit": 0.0, "sameDecision": 0})
        g["cases"] += 1
        g["maxAbsProb"] = max(g["maxAbsProb"], r["maxAbsProb"])
        g["maxAbsLogit"] = max(g["maxAbsLogit"], r["maxAbsLogit"])
        g["sameDecision"] += r["sameDecision"]
    passed = all(c["passed"] for c in checks)
    report = {"passed": passed, "seconds": round(time.monotonic() - started, 1), "loadSeconds": load_seconds,
              "bundle": str(args.bundle), "reference": str(args.reference),
              "aetherJSONSHA256": gg.sha256_file(args.bundle / ".aether/aether.json"),
              "precision": manifest.get("precision"), "summary": summary, "bySet": by_set,
              "failed": [c for c in checks if not c["passed"]], "rows": rows, "checks": checks}
    args.report.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"passed": passed, "summary": summary, "bySet": by_set}, indent=1), file=sys.stderr)
    if not passed:
        raise SystemExit("parity check failed")


if __name__ == "__main__":
    main()

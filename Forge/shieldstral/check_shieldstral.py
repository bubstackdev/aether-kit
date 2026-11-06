#!/usr/bin/env python3
"""The Shieldstral gate: the exported Core AI graph (GPU) against the upstream CPU float32 references.

    scripts/gpu-lease -- Forge/.venv/bin/python -I Forge/shieldstral/check_shieldstral.py --bundle <bundle dir> \
        --reference <reference dir> --report <new check-report.json>

With the bundle's own manifest (graph file, function, role names, compute placement, `postprocess.yesNo`), for every
reference case (reference_shieldstral.py): the case's input_ids and attention_mask, fed as int32, give
- probabilities within 0.01 of upstream's, both entries (docs/M4.md's yes/no bar);
- the same decision (P(yes) > threshold, the manifest's threshold, upstream's strict comparison);
- probabilities that sum to 1 (|sum - 1| <= 1e-3: the softmax is in the graph);
- a repeat run that is bit-identical.
Also: the graph's descriptor matches the manifest's assertions (input_ids, attention_mask int32 [1,512]; probs
float32 [1,2]). Reported, not gated: each case's gap to the model card's helper score. Writes each case's probs beside
the report (`<report stem>/<case>/probs.f32`). Exits nonzero when a bar fails. One graph on the GPU, about a minute.
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
import shieldstral_graphs as sg  # noqa: E402

PROB_BAR, SUM_BAR = 0.01, 1e-3


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
    spec = manifest["graphs"]["classifier"]
    ids_name, mask_name = spec["inputs"]["tokens"], spec["inputs"]["attentionMask"]
    out_name = spec["outputs"]["probs"]
    yes_no = manifest["postprocess"]["probs"]
    positive, threshold = yes_no["positiveIndex"], yes_no["threshold"]
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
    inputs, outputs = descriptor["inputs"], descriptor["outputs"]
    ok = (all(inputs.get(n, {}).get("dtype", "").endswith("int32") and inputs[n]["shape"] == shapes[k]
              for n, k in ((ids_name, "input_ids"), (mask_name, "attention_mask")))
          and outputs.get(out_name, {}).get("dtype", "").endswith("float32")
          and outputs[out_name]["shape"] == shapes["probs"]
          and spec.get("limits", {}).get("staticSequence") == shapes["input_ids"][1]
          and positive == index["labels"]["positiveIndex"])
    record("descriptor", float(ok), 1.0, ok, descriptor=descriptor)

    timings, summary = [], []
    for entry in index["cases"]:
        case = json.loads((args.reference / entry["name"] / "case.json").read_text())
        folder = args.reference / case["name"]
        ids = sg.read_i32(folder / "input_ids.i32", shapes["input_ids"])
        mask = sg.read_i32(folder / "attention_mask.i32", shapes["attention_mask"])
        want = sg.read_f32(folder / "probs.f32", shapes["probs"]).astype(np.float64)
        t0 = time.monotonic()
        out = graph(**{ids_name: ids, mask_name: mask})
        timings.append(round(time.monotonic() - t0, 4))
        got = out[out_name].astype(np.float64).reshape(shapes["probs"])
        delta = float(np.abs(got - want).max())
        p, p_ref = float(got[0, positive]), float(want[0, positive])
        record(f"{case['name']}.probs.maxAbs", round(delta, 7), PROB_BAR, bool(np.isfinite(got).all()) and delta <= PROB_BAR,
               pYes=p, pYesReference=p_ref, expectation=case["expectation"], tokens=case["realTokens"],
               dtype=str(out[out_name].dtype))
        same = bool((p > threshold) == (p_ref > threshold))
        record(f"{case['name']}.decision", float(same), 1.0, same, flagged=p > threshold,
               flaggedReference=p_ref > threshold)
        s = float(got.sum())
        record(f"{case['name']}.sum", round(s, 7), SUM_BAR, abs(s - 1) <= SUM_BAR)
        again = graph(**{ids_name: ids, mask_name: mask})
        exact = np.array_equal(again[out_name], out[out_name])
        record(f"{case['name']}.repeat.exact", float(exact), 1.0, exact)
        summary.append({"case": case["name"], "expectation": case["expectation"], "pYes": p, "pYesReference": p_ref,
                        "delta": p - p_ref, "helperScore": case["helperScore"], "minusHelper": p - case["helperScore"]})
        (dump / case["name"]).mkdir(parents=True, exist_ok=True)
        sg.write_f32(dump / case["name"] / "probs.f32", got.astype(np.float32))
    del graph
    passed = all(c["passed"] for c in checks)
    worst = max(abs(s["delta"]) for s in summary)
    print(f"[check] worst |dP(yes)| {worst:.6f} over {len(summary)} cases", file=sys.stderr, flush=True)
    args.report.write_text(json.dumps({"passed": passed, "seconds": round(time.monotonic() - started, 1),
                                       "loadSeconds": load_seconds, "runSeconds": timings,
                                       "worstAbsDeltaPYes": worst, "summary": summary,
                                       "bundle": str(args.bundle), "reference": str(args.reference),
                                       "aetherJSONSHA256": sg.sha256_file(args.bundle / ".aether/aether.json"),
                                       "checks": checks}, indent=2) + "\n")
    if not passed:
        raise SystemExit("parity check failed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The EmbeddingGemma gate: the exported Core AI graph (GPU) against the upstream CPU float32 references.

    Forge/.venv/bin/python -I Forge/embeddinggemma/check_embeddinggemma.py --bundle <bundle dir> \
        --reference <reference dir> --report <new check-report.json>

With the bundle's own manifest (graph file, function, role names, compute placement), for every reference case
(reference_embeddinggemma.py): the case's input_ids and attention_mask, fed as int32, give
- an embedding with cosine >= 0.999 against upstream's (docs/M4.md's embedding bar);
- unit length (|norm - 1| <= 1e-3: the normalization is in the graph);
- a repeat run that is bit-identical.
Also: the graph's descriptor matches the manifest's assertions (input_ids, attention_mask int32 [1,256]; embedding
float32 [1,768]), and the query x document cosine matrix from the graph is within 0.002 of upstream's with the same best
document per query. Writes each case's embedding beside the report (`<report stem>/<case>/embedding.f32`). Exits
nonzero when a bar fails. Runs one graph on the GPU for a few seconds.
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
import embeddinggemma_graphs as eg  # noqa: E402

COSINE_BAR, NORM_BAR, MATRIX_BAR = 0.999, 1e-3, 0.002


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
    spec = manifest["graphs"]["encoder"]
    ids_name, mask_name = spec["inputs"]["tokens"], spec["inputs"]["attentionMask"]
    out_name = spec["outputs"]["embedding"]
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
          and outputs[out_name]["shape"] == shapes["embedding"]
          and spec.get("limits", {}).get("staticSequence") == shapes["input_ids"][1])
    record("descriptor", float(ok), 1.0, ok, descriptor=descriptor)

    timings, got_all, want_all = [], {}, {}
    for entry in index["cases"]:
        case = json.loads((args.reference / entry["name"] / "case.json").read_text())
        folder = args.reference / case["name"]
        ids = eg.read_i32(folder / "input_ids.i32", shapes["input_ids"])
        mask = eg.read_i32(folder / "attention_mask.i32", shapes["attention_mask"])
        want = eg.read_f32(folder / "embedding.f32", shapes["embedding"])
        t0 = time.monotonic()
        out = graph(**{ids_name: ids, mask_name: mask})
        timings.append(round(time.monotonic() - t0, 4))
        got = out[out_name].astype(np.float32).reshape(shapes["embedding"])
        c = eg.cosine(got, want)
        record(f"{case['name']}.cosine", round(c, 10), COSINE_BAR, c >= COSINE_BAR, oneMinusCosine=float(1 - c),
               maxAbs=float(np.abs(got - want).max()), tokens=case["keptTokens"], dtype=str(out[out_name].dtype))
        n = float(np.linalg.norm(got.astype(np.float64)))
        record(f"{case['name']}.norm", round(n, 7), NORM_BAR, abs(n - 1) <= NORM_BAR)
        again = graph(**{ids_name: ids, mask_name: mask})
        same = np.array_equal(again[out_name], out[out_name])
        record(f"{case['name']}.repeat.exact", float(same), 1.0, same)
        got_all[case["name"]], want_all[case["name"]] = got, want
        (dump / case["name"]).mkdir(parents=True, exist_ok=True)
        eg.write_f32(dump / case["name"] / "embedding.f32", got)
    del graph

    similarity = index["similarity"]
    worst, same_best = 0.0, True
    for q in similarity["queries"]:
        row = {d: eg.cosine(got_all[q], got_all[d]) for d in similarity["documents"]}
        worst = max([worst] + [abs(row[d] - similarity["cosine"][q][d]) for d in row])
        same_best &= max(row, key=row.get) in similarity["expectedBest"][q]
    record("similarity.maxAbsDelta", round(worst, 7), MATRIX_BAR, worst <= MATRIX_BAR)
    record("similarity.bestDocument", float(same_best), 1.0, same_best)
    passed = all(c["passed"] for c in checks)
    args.report.write_text(json.dumps({"passed": passed, "seconds": round(time.monotonic() - started, 1),
                                       "loadSeconds": load_seconds, "runSeconds": timings,
                                       "bundle": str(args.bundle), "reference": str(args.reference),
                                       "aetherJSONSHA256": eg.sha256_file(args.bundle / ".aether/aether.json"),
                                       "checks": checks}, indent=2) + "\n")
    if not passed:
        raise SystemExit("parity check failed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The ColModernVBERT gate: the exported Core AI graphs (GPU) against the upstream CPU float32 references.

    scripts/gpu-lease -- Forge/.venv/bin/python -I Forge/colmodernvbert/check_colmodernvbert.py \
        --bundle <bundle dir> --reference <reference dir> --report <new check-report.json>

With the bundle's own manifest (graph files, functions, role names, compute placement, the token span):
- the graphs' descriptors against the manifest's assertions: tokens and attention_mask [1,64] int32 -> [1,64,128];
  image [1,1,3,512,512] -> [1,89,128];
- every page (reference_colmodernvbert.py), its float32 input converted to the graph input's type: cosine >= 0.999
  for every one of the 89 doc vectors against upstream (docs/M4.md's embedding bar), and a repeat run that is
  bit-identical;
- every query row with its mask: cosine >= 0.999 for every real row, padding rows exactly zero, and a repeat that is
  bit-identical;
- MaxSim from the graphs' vectors against upstream's score matrix: every query's ranking of the pages equals
  upstream's (and each expected page is first); the largest score difference and the smallest top-two margin are
  recorded.
Writes the graphs' vectors beside the report (`<report stem>/`). Exits nonzero when a bar fails. Loads two graphs on
the GPU for a few seconds; run it under scripts/gpu-lease.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import colmodernvbert_graphs as cg  # noqa: E402

COSINE_BAR = 0.999
DTYPES = {"float16": np.float16, "float32": np.float32, "int32": np.int32}


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


def input_dtype(descriptor: dict, name: str):
    text = descriptor["inputs"][name]["dtype"]
    for key, value in DTYPES.items():
        if text.endswith(key):
            return value
    raise SystemExit(f"unexpected input dtype {text} for {name}")


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
    q_spec, d_spec = manifest["graphs"]["queryEncoder"], manifest["graphs"]["docEncoder"]
    tokens_in, mask_in = q_spec["inputs"]["tokens"], q_spec["inputs"]["attentionMask"]
    q_out = q_spec["outputs"]["tokenEmbeddings"]
    image_in, d_out = d_spec["inputs"]["image"], d_spec["outputs"]["tokenEmbeddings"]
    length = q_spec["limits"]["staticSequence"]
    span = manifest["postprocess"]["retrieval"]["imageTokenSpan"]
    grid = manifest["postprocess"]["retrieval"]["patchGrid"]
    shapes = index["shapes"]
    dump = args.report.with_suffix("")
    dump.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []

    def record(name: str, value: float, bar: float, passed: bool, **extra) -> None:
        checks.append({"check": name, "value": value, "bar": bar, "passed": bool(passed), **extra})
        print(f"[check] {'PASS' if passed else 'FAIL'} {name}: {value} (bar {bar})", file=sys.stderr, flush=True)

    started = time.monotonic()
    ok = (span == index["docPrompt"]["imageTokenSpan"] and grid == index["docPrompt"]["patchGrid"]
          and length == shapes["tokens"][1] and manifest["tokenizer"]["text"]["padTo"] == length)
    record("manifest.matchesReference", float(ok), 1.0, ok)

    # MARK: doc encoder
    t0 = time.monotonic()
    graph = Graph(args.bundle / d_spec["file"].split(":", 1)[1], d_spec["function"], d_spec["compute"])
    load = {"docEncoder": round(time.monotonic() - t0, 2)}
    d = graph.descriptor()
    ok = (d["inputs"].get(image_in, {}).get("shape") == shapes["image"]
          and d["outputs"].get(d_out, {}).get("shape") == [1, *shapes["docEmbedding"]])
    record("doc.descriptor", float(ok), 1.0, ok, descriptor=d)
    feed_type = input_dtype(d, image_in)
    pages, timings = {}, {"doc": []}
    for entry in index["cases"]:
        folder = args.reference / "pages" / entry["name"]
        x = cg.read_f32(folder / "input.f32", shapes["image"])
        want = cg.read_f32(folder / "embedding.f32", shapes["docEmbedding"])
        feed = x.astype(feed_type)
        t = time.monotonic()
        out = graph(**{image_in: feed})
        timings["doc"].append(round(time.monotonic() - t, 4))
        got = out[d_out].astype(np.float32).reshape(shapes["docEmbedding"])
        cos = cg.cosine(got, want)
        worst = int(np.argmin(cos))
        record(f"doc.{entry['name']}.cosine.min", round(float(cos.min()), 7), COSINE_BAR,
               bool((cos >= COSINE_BAR).all()),
               worstRow=worst, mean=round(float(cos.mean()), 7),
               imageRowsMin=round(float(cos[span[0]:span[1]].min()), 7),
               maxAbs=float(np.abs(got - want).max()), dtype=str(out[d_out].dtype),
               normRange=[float(np.linalg.norm(got, axis=-1).min()), float(np.linalg.norm(got, axis=-1).max())])
        again = graph(**{image_in: feed})
        same = np.array_equal(again[d_out], out[d_out])
        record(f"doc.{entry['name']}.repeat.exact", float(same), 1.0, same)
        pages[entry["name"]] = got
        cg.write_f32(dump / f"doc-{entry['name']}.f32", got)
    del graph

    # MARK: query encoder
    n = shapes["tokens"][0]
    ids = cg.read_i32(args.reference / "queries/tokens.i32", shapes["tokens"])
    masks = cg.read_i32(args.reference / "queries/mask.i32", shapes["tokens"])
    want_q = cg.read_f32(args.reference / "queries/embedding.f32", shapes["queryEmbedding"])
    t0 = time.monotonic()
    graph = Graph(args.bundle / q_spec["file"].split(":", 1)[1], q_spec["function"], q_spec["compute"])
    load["queryEncoder"] = round(time.monotonic() - t0, 2)
    d = graph.descriptor()
    ok = (all(d["inputs"].get(r, {}).get("dtype", "").endswith("int32")
              and d["inputs"][r]["shape"] == [1, length] for r in (tokens_in, mask_in))
          and d["outputs"].get(q_out, {}).get("shape") == [1, length, cg.EMBED_DIM])
    record("query.descriptor", float(ok), 1.0, ok, descriptor=d)
    queries, times = np.zeros_like(want_q), []
    items = index["queries"]["items"]
    for i in range(n):
        count = items[i]["count"]
        feeds = {tokens_in: ids[i:i + 1], mask_in: masks[i:i + 1]}
        t = time.monotonic()
        out = graph(**feeds)
        times.append(time.monotonic() - t)
        got = out[q_out].astype(np.float32).reshape(length, cg.EMBED_DIM)
        cos = cg.cosine(got[:count], want_q[i, :count])
        label = f"query.{i}"
        record(f"{label}.cosine.min", round(float(cos.min()), 7), COSINE_BAR, bool((cos >= COSINE_BAR).all()),
               text=items[i]["text"][:80], count=count, worstRow=int(np.argmin(cos)),
               maxAbs=float(np.abs(got[:count] - want_q[i, :count]).max()))
        zero = bool((got[count:] == 0).all())
        record(f"{label}.padding.zero", float(zero), 1.0, zero)
        again = graph(**feeds)
        same = np.array_equal(again[q_out], out[q_out])
        record(f"{label}.repeat.exact", float(same), 1.0, same)
        queries[i] = got
    timings["query"] = round(statistics.median(times), 5)
    cg.write_f32(dump / "queries.f32", queries)
    del graph

    # MARK: MaxSim
    names = index["maxSim"]["pages"]
    ref_scores = np.asarray(index["maxSim"]["scores"])
    scores = np.array([[cg.maxsim(queries[i], pages[p]) for p in names] for i in range(n)])
    for i in range(n):
        ranking = [names[j] for j in np.argsort(-scores[i], kind="stable")]
        same = ranking == index["maxSim"]["rankings"][i]
        expected = items[i].get("expectedPage")
        top = np.sort(scores[i])[::-1]
        record(f"maxSim.query.{i}.ranking", float(same), 1.0, same and (expected is None or ranking[0] == expected),
               text=items[i]["text"][:80], top=ranking[0], expected=expected, ranking=ranking,
               maxAbsDelta=round(float(np.abs(scores[i] - ref_scores[i]).max()), 5),
               margin=round(float(top[0] - top[1]), 4))
    delta = float(np.abs(scores - ref_scores).max())
    rel = float((np.abs(scores - ref_scores) / np.abs(ref_scores)).max())
    record("maxSim.scores.maxAbsDelta", round(delta, 5), 0.0, True, maxRelDelta=round(rel, 7),
           note="reported, not gated: rankings are the bar")

    passed = all(c["passed"] for c in checks)
    args.report.write_text(json.dumps({"passed": passed, "seconds": round(time.monotonic() - started, 1),
                                       "loadSeconds": load, "runSeconds": timings,
                                       "bundle": str(args.bundle), "reference": str(args.reference),
                                       "precision": manifest.get("precision"),
                                       "aetherJSONSHA256": cg.sha256_file(args.bundle / ".aether/aether.json"),
                                       "scores": scores.tolist(), "checks": checks}, indent=2) + "\n")
    if not passed:
        raise SystemExit("parity check failed")


if __name__ == "__main__":
    main()

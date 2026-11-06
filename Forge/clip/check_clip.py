#!/usr/bin/env python3
"""The CLIP ViT-B/32 gate: the exported Core AI graphs (GPU) against the upstream CPU float32 references.

    Forge/.venv/bin/python -I Forge/clip/check_clip.py --bundle <bundle dir> --reference <reference dir> \
        --report <new check-report.json>

With the bundle's own manifest (graph files, functions, buckets, role names, compute placement):
- every image case (reference_clip.py), its input tensor converted to the graph's float16 input: image embedding
  cosine >= 0.999 against upstream (docs/M4.md's embedding bar), and a repeat run that is bit-identical;
- every reference text through each text bucket (batch 1 one row per call; batch 8 in chunks, the unused rows filled
  with the manifest's `unusedBatchRows`): text embedding cosine >= 0.999 per row, and a repeat that is bit-identical;
- zero-shot: for every image, argmax over the label rows of logitScale * cosine from the graphs' embeddings equals
  upstream's `logits_per_image` argmax (the top-two margin is recorded);
- the graphs' descriptors against the manifest's assertions: image [1,3,224,224] float16 -> [1,512]; tokens [B,77]
  int32 -> [B,512] for each bucket.
Writes the graphs' embeddings beside the report (`<report stem>/`). Exits nonzero when a bar fails. Loads two graphs on
the GPU for a few seconds.
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
import clip_graphs as cg  # noqa: E402

COSINE_BAR = 0.999


class Graph:
    def __init__(self, path: Path, functions: list[str], compute: str) -> None:
        import coreai.runtime as rt
        self.rt = rt
        options = (rt.SpecializationOptions.cpu_only() if compute == "cpu" else
                   rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
        self.model = asyncio.run(rt.AIModel.load(path, options))
        self.functions = {f: self.model.load_function(f) for f in functions}

    def descriptor(self, function: str) -> dict:
        d = self.functions[function].desc
        describe = lambda x: {"dtype": str(x.dtype), "shape": list(x.shape)}  # noqa: E731
        return {"inputs": {n: describe(d.input_descriptor(n)) for n in d.input_names},
                "outputs": {n: describe(d.output_descriptor(n)) for n in d.output_names}}

    def __call__(self, function: str, **feeds: np.ndarray) -> dict[str, np.ndarray]:
        rt = self.rt

        async def go():
            out = await self.functions[function]({k: rt.NDArray(np.ascontiguousarray(v)) for k, v in feeds.items()},
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
    image_spec, text_spec = manifest["graphs"]["imageEncoder"], manifest["graphs"]["textEncoder"]
    image_in, image_out = image_spec["inputs"]["image"], image_spec["outputs"]["imageEmbedding"]
    tokens_in, text_out = text_spec["inputs"]["tokens"], text_spec["outputs"]["textEmbedding"]
    buckets = {e["value"]: e["function"] for e in text_spec["buckets"]["entries"]}
    filler = manifest["tokenizer"]["text"]["unusedBatchRows"]
    pad = manifest["tokenizer"]["text"]["padId"]
    scale = manifest["postprocess"]["embedding"]["logitScale"]
    shapes = index["shapes"]
    dump = args.report.with_suffix("")
    dump.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []

    def record(name: str, value: float, bar: float, passed: bool, **extra) -> None:
        checks.append({"check": name, "value": value, "bar": bar, "passed": bool(passed), **extra})
        print(f"[check] {'PASS' if passed else 'FAIL'} {name}: {value} (bar {bar})", file=sys.stderr, flush=True)

    started = time.monotonic()
    if abs(scale - index["zeroShot"]["logitScale"]) > 1e-4:
        raise SystemExit(f"manifest logitScale {scale} differs from upstream's {index['zeroShot']['logitScale']}")

    # MARK: image tower
    t0 = time.monotonic()
    graph = Graph(args.bundle / image_spec["file"].split(":", 1)[1], [image_spec["function"]], image_spec["compute"])
    load = {"imageEncoder": round(time.monotonic() - t0, 2)}
    d = graph.descriptor(image_spec["function"])
    ok = (d["inputs"].get(image_in, {}).get("dtype", "").endswith("float16")
          and d["inputs"][image_in]["shape"] == shapes["image"]
          and d["outputs"].get(image_out, {}).get("shape") == shapes["imageEmbedding"])
    record("image.descriptor", float(ok), 1.0, ok, descriptor=d)
    image_emb, timings = {}, {"image": []}
    for entry in index["cases"]:
        folder = args.reference / "images" / entry["name"]
        x = cg.read_f32(folder / "input.f32", shapes["image"])
        want = cg.read_f32(folder / "embedding.f32", shapes["imageEmbedding"])
        feed = x.astype(np.float16)
        if not np.array_equal(feed.astype(np.float32), x):
            raise SystemExit(f"{entry['name']}: input.f32 is not fp16-representable")
        t = time.monotonic()
        out = graph(image_spec["function"], **{image_in: feed})
        timings["image"].append(round(time.monotonic() - t, 4))
        got = out[image_out].astype(np.float32).reshape(shapes["imageEmbedding"])
        c = float(cg.cosine(got, want)[0])
        record(f"image.{entry['name']}.cosine", round(c, 7), COSINE_BAR, c >= COSINE_BAR,
               maxAbs=float(np.abs(got - want).max()), norm=float(np.linalg.norm(got)),
               dtype=str(out[image_out].dtype))
        again = graph(image_spec["function"], **{image_in: feed})
        same = np.array_equal(again[image_out], out[image_out])
        record(f"image.{entry['name']}.repeat.exact", float(same), 1.0, same)
        image_emb[entry["name"]] = got
        cg.write_f32(dump / f"image-{entry['name']}.f32", got)
    del graph

    # MARK: text tower, every bucket
    n = shapes["tokens"][0]
    ids = cg.read_i32(args.reference / "texts/tokens.i32", shapes["tokens"])
    want_text = cg.read_f32(args.reference / "texts/embedding.f32", shapes["textEmbedding"])
    t0 = time.monotonic()
    graph = Graph(args.bundle / text_spec["file"].split(":", 1)[1], list(buckets.values()), text_spec["compute"])
    load["textEncoder"] = round(time.monotonic() - t0, 2)
    text_emb = {}
    for batch, function in sorted(buckets.items()):
        d = graph.descriptor(function)
        ok = (d["inputs"].get(tokens_in, {}).get("dtype", "").endswith("int32")
              and d["inputs"][tokens_in]["shape"] == [batch, cg.TEXT_LENGTH]
              and d["outputs"].get(text_out, {}).get("shape") == [batch, cg.EMBED_DIM])
        record(f"text.{function}.descriptor", float(ok), 1.0, ok, descriptor=d)
        rows, times, repeat_same = [], [], True
        for start in range(0, n, batch):
            chunk = ids[start:start + batch]
            feed = np.full((batch, cg.TEXT_LENGTH), pad, dtype=np.int32)
            feed[:, :len(filler)] = filler
            feed[:len(chunk)] = chunk
            t = time.monotonic()
            out = graph(function, **{tokens_in: feed})
            times.append(time.monotonic() - t)
            again = graph(function, **{tokens_in: feed})
            repeat_same &= bool(np.array_equal(again[text_out], out[text_out]))
            rows.append(out[text_out].astype(np.float32).reshape(batch, cg.EMBED_DIM)[:len(chunk)])
        got = np.concatenate(rows)
        cos = cg.cosine(got, want_text)
        worst = int(np.argmin(cos))
        record(f"text.{function}.cosine.min", round(float(cos.min()), 7), COSINE_BAR, bool((cos >= COSINE_BAR).all()),
               perRow=[round(float(c), 7) for c in cos], worstText=index["texts"]["strings"][worst],
               maxAbs=float(np.abs(got - want_text).max()))
        record(f"text.{function}.repeat.exact", float(repeat_same), 1.0, repeat_same)
        timings[f"text.{function}"] = round(statistics.median(times), 5)
        text_emb[function] = got
        cg.write_f32(dump / f"text-{function}.f32", got)
    del graph
    fns = sorted(text_emb)
    across = cg.cosine(text_emb[fns[0]], text_emb[fns[-1]])
    record(f"text.{fns[0]}-vs-{fns[-1]}.cosine.min", round(float(across.min()), 7), COSINE_BAR,
           bool((across >= COSINE_BAR).all()))

    # MARK: zero-shot
    rows = index["zeroShot"]["textRows"]
    labels = index["zeroShot"]["labels"]
    for name, emb in image_emb.items():
        want = index["zeroShot"]["perImage"][name]
        for function, texts in sorted(text_emb.items()):
            logits = scale * (emb @ texts[rows].T)[0]
            top = np.argsort(-logits)
            same = int(top[0]) == want["argmax"]
            ref_logits = np.asarray(want["logits"])
            ref_top = np.argsort(-ref_logits)
            record(f"zeroShot.{name}.{function}.argmax", float(same), 1.0, same, label=labels[int(top[0])],
                   expected=labels[want["argmax"]], margin=round(float(logits[top[0]] - logits[top[1]]), 4),
                   upstreamMargin=round(float(ref_logits[ref_top[0]] - ref_logits[ref_top[1]]), 4),
                   maxLogitDelta=round(float(np.abs(logits - ref_logits).max()), 4))

    passed = all(c["passed"] for c in checks)
    args.report.write_text(json.dumps({"passed": passed, "seconds": round(time.monotonic() - started, 1),
                                       "loadSeconds": load, "runSeconds": timings,
                                       "bundle": str(args.bundle), "reference": str(args.reference),
                                       "precision": manifest.get("precision"),
                                       "aetherJSONSHA256": cg.sha256_file(args.bundle / ".aether/aether.json"),
                                       "checks": checks}, indent=2) + "\n")
    if not passed:
        raise SystemExit("parity check failed")


if __name__ == "__main__":
    main()

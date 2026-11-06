#!/usr/bin/env python3
"""The Qwen3-VL 2B gate: the exported Core AI graphs (GPU) against the upstream CPU float32 references.

    Forge/.venv/bin/python -I Forge/qwen3vl/check_qwen3vl.py --bundle <bundle dir> --reference <reference dir> \
        --report <new check-report.json> [--vision-from <bundle dir>] [--graphs vision,decoder]

With the bundle's own manifest (graph files, role names, compute placement), for every reference case
(reference_qwen3vl.py):

- **vision** (image cases): upstream's pixel_values, zero-padded to the tower's capacity, and the case's grid give
  `image_embeds` and the three DeepStack levels with cosine >= 0.999 each against upstream's (docs/M4.md's embedding
  bar); rel L2 and the worst per-token cosine are reported; the rows past the image are exactly zero; a repeat run is
  bit-identical.
- **decoder**, fed as the Swift driver will feed it (the GPU tower's own outputs as the static image inputs, extension
  ids for the image pads, the rope shift from the case's grid; text-only: zero buffers and the sentinel), the prompt
  prefilled in chunks of `limits.prefillChunk`, then upstream's greedy tokens fed back one at a time (teacher forcing;
  until the first step whose argmax differs this is the graph's own greedy decoding, token for token): the graph's
  greedy tokens must equal upstream's for >= 32 tokens (the count until the first difference is reported, with
  upstream's top-2 probability margin there); first-token logits cosine and rel L2, and every step's cosine, rel L2 and
  argmax agreement, are reported. Also reported: the first-token logits with upstream's own CPU tower outputs as the
  image inputs (the decoder's error alone).
- the graphs' descriptors match the manifest's assertions (shapes, dtypes, the dynamic token and KV axes).

Run one case per process and merge (`--cases <case> --report <part>.json`, then `--merge <parts...> --report <json>`):
in one process the Python runtime's retained specializations of the dynamic decoder end the run with a SIGKILL a
case or two in (measured); the Swift engine specializes with `expectFrequentReshapes` and is not affected.

Writes the GPU outputs beside the report (`<report stem>/<case>/`). Exits nonzero when a bar fails. Loads the 3.4 GB
decoder; a few minutes of GPU time for the five cases (the Python runtime re-specializes the
dynamic decoder for each new length: it has no `expectFrequentReshapes`).
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qwen3vl_graphs as qg  # noqa: E402

COSINE_BAR = 0.999          # embeddings, per vector set (docs/M4.md)
GREEDY_BAR = 32             # greedy tokens equal to upstream's


class Graph:
    def __init__(self, path: Path, function: str, compute: str) -> None:
        import coreai.runtime as rt
        self.rt = rt
        options = (rt.SpecializationOptions.cpu_only() if compute == "cpu" else
                   rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
        self.path, self.name, self.options = path, function, options
        self.model = asyncio.run(rt.AIModel.load(path, options))
        self.function = self.model.load_function(function)
        self.calls = 0
        self.reload_every = 0

    def reload(self) -> None:
        """A fresh load of the same graph. The Python runtime has no `expectFrequentReshapes`: each new input shape
        is specialized anew and the specializations' GPU memory is retained, so a long run of distinct decode lengths
        (about a dozen on the 28-layer decoder, measured) ends in Metal command-buffer errors. A reload frees them;
        the states live in host arrays and carry over."""
        del self.function, self.model
        gc.collect()
        self.model = asyncio.run(self.rt.AIModel.load(self.path, self.options))
        self.function = self.model.load_function(self.name)
        self.reloads = getattr(self, "reloads", 0) + 1

    def descriptor(self) -> dict:
        d = self.function.desc
        describe = lambda x: {"dtype": str(x.dtype), "shape": list(x.shape)}  # noqa: E731
        return {"inputs": {n: describe(d.input_descriptor(n)) for n in d.input_names},
                "outputs": {n: describe(d.output_descriptor(n)) for n in d.output_names},
                "states": {n: describe(d.state_descriptor(n)) for n in d.state_names}}

    def state(self, shapes: dict[str, list[int]]) -> dict:
        return {n: self.rt.NDArray(np.zeros(s, dtype=np.float16)) for n, s in shapes.items()}

    def __call__(self, feeds: dict[str, np.ndarray], state: dict | None = None) -> dict[str, np.ndarray]:
        rt = self.rt
        if self.reload_every and self.calls and self.calls % self.reload_every == 0:
            self.reload()
        self.calls += 1

        async def go():
            out = await self.function({k: rt.NDArray(np.ascontiguousarray(v)) for k, v in feeds.items()}, state or {})
            return {k: v.numpy().copy() for k, v in out.items()}
        return asyncio.run(go())


class DecoderRun:
    """The host side of the decoder contract: positions, chunked prefill, one-token steps, in-place KV states."""

    def __init__(self, graph: Graph, spec: dict, statics: dict[str, np.ndarray], capacity: int) -> None:
        self.graph, self.spec, self.statics = graph, spec, statics
        d = graph.descriptor()["states"]
        self.state = graph.state({n: [capacity if s == -1 else s for s in d[n]["shape"]] for n in d})
        self.inputs = spec["inputs"]
        self.chunk = spec["limits"]["prefillChunk"]
        self.position = 0

    def feed(self, tokens: np.ndarray) -> np.ndarray:
        logits = None
        for a in range(0, len(tokens), self.chunk):
            part = tokens[a:a + self.chunk]
            end = self.position + len(part)
            feeds = {self.inputs["tokens"]: part[None].astype(np.int32),
                     self.inputs["positions"]: np.arange(end, dtype=np.int32)[None], **self.statics}
            logits = self.graph(feeds, self.state)[self.spec["outputs"]["logits"]["name"]]
            self.position = end
        return logits.reshape(-1).astype(np.float32)


def statics_for(spec: dict, case: dict, image: np.ndarray | None, deep: np.ndarray | None, slots: int) -> dict:
    """The decoder's static inputs for one case (image or text-only), by the manifest's role names."""
    names = {role: (b["name"] if isinstance(b, dict) else b) for role, b in spec["inputs"].items()}
    if case.get("image"):
        _, h, w = case["gridTHW"]
        rc = qg.image_rc(h, w, slots=slots)
        start, amount = case["ropeShiftStart"], case["ropeShiftAmount"]
    else:
        image = np.zeros((slots, 2048), np.float32)
        deep = np.zeros((qg.DEEPSTACK_LEVELS * slots, 2048), np.float32)
        rc = np.zeros((slots, 2), np.int32)
        start, amount = qg.TEXT_ONLY_SHIFT_START, 0
    return {names["imageEmbeds"]: image, names["deepstackEmbeds"]: deep, names["imageRC"]: rc,
            names["ropeShiftStart"]: np.array([start], np.int32), names["ropeShiftAmount"]: np.array([amount], np.int32)}


def pad_reference(image: np.ndarray, deep: np.ndarray, slots: int) -> tuple[np.ndarray, np.ndarray]:
    n = image.shape[0]
    pi = np.zeros((slots, image.shape[1]), np.float32)
    pi[:n] = image
    pd = np.zeros((qg.DEEPSTACK_LEVELS * slots, image.shape[1]), np.float32)
    for k in range(qg.DEEPSTACK_LEVELS):
        pd[k * slots:k * slots + n] = deep[k]
    return pi, pd


def merge(paths: list[Path], report: Path) -> None:
    """One report from per-case runs (`--cases`, one process each: the Python runtime's retained specializations
    end a long single-process run with a SIGKILL, measured on the 28-layer decoder; see Graph.reload)."""
    if report.exists():
        raise SystemExit("report already exists; choose a new path")
    parts = [json.loads(p.read_text()) for p in paths]
    seen, checks, summary = set(), [], []
    for part in parts:
        for c in part["checks"]:
            if c["check"] not in seen:          # the descriptor checks repeat in every part
                seen.add(c["check"])
                checks.append(c)
        summary += part["summary"]
    merged = {k: parts[0][k] for k in ("bundle", "visionFrom", "reference", "precision", "aetherJSONSHA256", "bars")}
    merged.update({"passed": all(p["passed"] for p in parts), "parts": [str(p) for p in paths],
                   "seconds": round(sum(p["seconds"] for p in parts), 1),
                   "loadSeconds": [p["loadSeconds"] for p in parts],
                   "timings": {k: sum((p["timings"][k] for p in parts), []) for k in parts[0]["timings"]},
                   "decoderReloads": [p["decoderReloads"] for p in parts],
                   "peakRSSGB": max(p["peakRSSGB"] for p in parts), "summary": summary, "checks": checks})
    report.write_text(json.dumps(merged, indent=2) + "\n")
    if not merged["passed"]:
        raise SystemExit("parity check failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--vision-from", type=Path, help="take the vision graph from this bundle instead")
    parser.add_argument("--graphs", default="vision,decoder")
    parser.add_argument("--cases", help="comma-separated subset")
    parser.add_argument("--reload-every", type=int, default=6,
                        help="reload the decoder every N calls (Python-runtime workaround, see Graph.reload; 0: never)")
    parser.add_argument("--forced", action=argparse.BooleanOptionalAction, default=True,
                        help="also the first-token logits with upstream's own tower outputs (reported, not gated)")
    parser.add_argument("--merge", nargs="+", type=Path,
                        help="combine these per-case reports into --report (no graph runs)")
    args = parser.parse_args()
    if args.merge:
        merge(args.merge, args.report)
        return
    if args.report.exists():
        raise SystemExit("report already exists; choose a new path")
    graphs = set(args.graphs.split(","))
    manifest = json.loads((args.bundle / ".aether/aether.json").read_text())
    vision_bundle = args.vision_from or args.bundle
    vspec = json.loads((vision_bundle / ".aether/aether.json").read_text())["graphs"]["vision"]
    dspec = manifest["graphs"]["decoder"]
    index = json.loads((args.reference / "references.json").read_text())
    vocab, steps = index["vocab"], index["steps"]
    capacity = index["patchCapacity"]
    slots = capacity // 4
    dump = args.report.with_suffix("")
    checks: list[dict] = []

    def record(name: str, value, bar, passed: bool, gated: bool = True, **extra) -> None:
        checks.append({"check": name, "value": value, "bar": bar, "passed": bool(passed), "gated": gated, **extra})
        tag = ("PASS" if passed else "FAIL") if gated else "INFO"
        print(f"[check] {tag} {name}: {value} (bar {bar})", file=sys.stderr, flush=True)

    started = time.monotonic()
    timings: dict[str, list] = {"encodeSeconds": [], "prefillSeconds": [], "decodeMsPerToken": []}
    t0 = time.monotonic()
    vision = Graph(vision_bundle / vspec["file"].split(":", 1)[1], vspec["function"], vspec["compute"])
    load = {"vision": round(time.monotonic() - t0, 2)}
    vd = vision.descriptor()
    vi, vo = vspec["inputs"], vspec["outputs"]
    ok = (vd["inputs"][vi["patches"]] == {"dtype": "float32", "shape": [capacity, 1536]}
          and vd["inputs"][vi["grid"]] == {"dtype": "int32", "shape": [2]}
          and vd["outputs"][vo["imageEmbeds"]] == {"dtype": "float32", "shape": [slots, 2048]}
          and vd["outputs"][vo["deepstackEmbeds"]] == {"dtype": "float32", "shape": [3 * slots, 2048]}
          and not vd["states"])
    record("vision.descriptor", float(ok), 1.0, ok, descriptor=vd)
    wanted = set(args.cases.split(",")) if args.cases else None
    encoded, rows = {}, {}
    for entry in index["cases"]:
        name = entry["name"]
        if wanted and name not in wanted:
            continue
        folder = args.reference / name
        case = json.loads((folder / "case.json").read_text())
        out_dir = dump / name
        out_dir.mkdir(parents=True, exist_ok=True)
        row = {"case": name, "grid": case.get("gridTHW"), "promptTokens": case["promptTokens"]}
        image = deep = None
        if case.get("image"):
            _, h, w = case["gridTHW"]
            n = h * w // 4
            pv = qg.read_f32(folder / "pixel_values.f32", [h * w, 1536])
            want_img = qg.read_f32(folder / "image_embeds.f32", [n, 2048])
            want_deep = qg.read_f32(folder / "deepstack_embeds.f32", [3, n, 2048])
            feeds = {vi["patches"]: qg.pad_patches(pv, capacity), vi["grid"]: np.array([h, w], np.int32)}
            t0 = time.monotonic()
            out = vision(feeds)
            timings["encodeSeconds"].append(round(time.monotonic() - t0, 4))
            image, deep = out[vo["imageEmbeds"]], out[vo["deepstackEmbeds"]]
            c = qg.cosine(image[:n], want_img)
            record(f"{name}.vision.imageEmbeds.cosine", round(c, 10), COSINE_BAR, c >= COSINE_BAR,
                   relL2=qg.rel_l2(image[:n], want_img),
                   minTokenCosine=float(qg.row_cosines(image[:n], want_img).min()), tokens=n)
            row.update({"imageEmbedsCosine": c, "imageEmbedsRelL2": qg.rel_l2(image[:n], want_img)})
            for k in range(qg.DEEPSTACK_LEVELS):
                got = deep[k * slots:k * slots + n]
                ck = qg.cosine(got, want_deep[k])
                record(f"{name}.vision.deepstack{k}.cosine", round(ck, 10), COSINE_BAR, ck >= COSINE_BAR,
                       relL2=qg.rel_l2(got, want_deep[k]), minTokenCosine=float(qg.row_cosines(got, want_deep[k]).min()))
                row[f"deepstack{k}Cosine"], row[f"deepstack{k}RelL2"] = ck, qg.rel_l2(got, want_deep[k])
            pad_zero = bool(not image[n:].any()
                            and not any(deep[k * slots + n:(k + 1) * slots].any() for k in range(qg.DEEPSTACK_LEVELS)))
            record(f"{name}.vision.paddingZero", float(pad_zero), 1.0, pad_zero)
            again = vision(feeds)
            same = all(np.array_equal(again[k], out[k]) for k in out)
            record(f"{name}.vision.repeat.exact", float(same), 1.0, same)
            qg.write_f32(out_dir / "image_embeds.f32", image)
            qg.write_f32(out_dir / "deepstack_embeds.f32", deep)
        encoded[name] = (image, deep)
        rows[name] = row

    # The vision graph is released before the decoder loads (one model resident at a time, as the driver's residency
    # plan may also choose; with both resident, the decode steps after a long prefill failed with Metal command-buffer errors in the Python runtime on this host).
    del vision
    gc.collect()
    decoder = None
    if "decoder" in graphs:
        t0 = time.monotonic()
        decoder = Graph(args.bundle / dspec["file"].split(":", 1)[1], dspec["function"], dspec["compute"])
        load["decoder"] = round(time.monotonic() - t0, 2)
        decoder.reload_every = args.reload_every
        dd = decoder.descriptor()
        names = {role: (b["name"] if isinstance(b, dict) else b) for role, b in dspec["inputs"].items()}
        expect = {names["tokens"]: ("int32", [1, -1]), names["positions"]: ("int32", [1, -1]),
                  names["imageEmbeds"]: ("float32", [slots, 2048]),
                  names["deepstackEmbeds"]: ("float32", [3 * slots, 2048]),
                  names["imageRC"]: ("int32", [slots, 2]), names["ropeShiftStart"]: ("int32", [1]),
                  names["ropeShiftAmount"]: ("int32", [1])}
        ok = (all(dd["inputs"].get(n) == {"dtype": t, "shape": s} for n, (t, s) in expect.items())
              and len(dd["inputs"]) == len(expect)
              and dd["outputs"].get(dspec["outputs"]["logits"]["name"]) == {"dtype": "float32",
                                                                           "shape": [1, 1, vocab]}
              and all(dd["states"].get(n, {}).get("shape") == [28, 1, 8, -1, 128]
                      for n in dspec["states"]["sequence"][0]["tensors"])
              and dspec["inputEncodings"]["tokens"]["extensionIds"] == {"base": vocab, "count": slots})
        record("decoder.descriptor", float(ok), 1.0, ok, descriptor=dd)

    for entry in index["cases"]:
        name = entry["name"]
        if decoder is None or (wanted and name not in wanted):
            continue
        folder = args.reference / name
        case = json.loads((folder / "case.json").read_text())
        out_dir = dump / name
        row = rows[name]
        image, deep = encoded[name]
        ids = qg.read_i32(folder / "input_ids.i32", [1, case["promptTokens"]])[0]
        tokens = qg.extension_ids(ids, vocab)
        ref_greedy = qg.read_i32(folder / "greedy.i32", [steps])
        ref_logits = qg.read_f32(folder / "logits.f32", [steps, vocab])
        cap = len(tokens) + steps + 8
        statics = statics_for(dspec, case, image, deep, slots)
        run = DecoderRun(decoder, dspec, statics, cap)
        t0 = time.monotonic()
        first = run.feed(tokens)
        timings["prefillSeconds"].append(round(time.monotonic() - t0, 3))
        c, r = qg.cosine(first, ref_logits[0]), qg.rel_l2(first, ref_logits[0])
        record(f"{name}.decoder.firstLogits.relL2", round(r, 6), None, True, gated=False, cosine=c,
               argmaxEqual=int(first.argmax()) == int(ref_greedy[0]))
        # Teacher-forced: upstream's greedy tokens fed back one at a time. Up to and including the first step whose
        # argmax differs, this is exactly greedy decoding from the graph's own argmax (the fed tokens are the same),
        # so one pass gives both the greedy agreement and the per-step logits error.
        forced = [first]
        t0 = time.monotonic()
        for i in range(steps - 1):
            forced.append(run.feed(ref_greedy[i:i + 1]))
        timings["decodeMsPerToken"].append(round((time.monotonic() - t0) * 1000 / max(steps - 1, 1), 2))
        forced = np.stack(forced)
        argmax = forced.argmax(-1).astype(np.int32)
        diff = np.nonzero(argmax != ref_greedy)[0]
        matched = int(diff[0]) if len(diff) else steps
        margin = case["probabilityMargins"][matched] if matched < steps else None
        record(f"{name}.decoder.greedyMatched", matched, GREEDY_BAR, matched >= GREEDY_BAR, steps=steps,
               referenceMarginAtDivergence=margin, firstEOS=case["firstEOS"],
               graphTokenAtDivergence=int(argmax[matched]) if matched < steps else None)
        cos = qg.row_cosines(forced, ref_logits)
        rel = np.linalg.norm(forced - ref_logits, axis=-1) / np.linalg.norm(ref_logits, axis=-1)
        agree = float((argmax == ref_greedy).mean())
        record(f"{name}.decoder.forced", round(float(cos.min()), 6), None, True, gated=False,
               minCosine=float(cos.min()), maxRelL2=float(rel.max()), meanRelL2=float(rel.mean()),
               argmaxAgreement=agree)
        qg.write_i32(out_dir / "greedy.i32", argmax[:matched + 1] if matched < steps else argmax)
        row.update({"firstLogitsCosine": c, "firstLogitsRelL2": r, "greedyMatched": matched,
                    "marginAtDivergence": margin, "forcedMinCosine": float(cos.min()),
                    "forcedMaxRelL2": float(rel.max()), "forcedArgmaxAgreement": agree})
        if args.forced:
            if case.get("image"):
                _, h, w = case["gridTHW"]
                n = h * w // 4
                ri, rd = pad_reference(qg.read_f32(folder / "image_embeds.f32", [n, 2048]),
                                       qg.read_f32(folder / "deepstack_embeds.f32", [3, n, 2048]), slots)
                run = DecoderRun(decoder, dspec, statics_for(dspec, case, ri, rd, slots), cap)
                alone = run.feed(tokens)
                record(f"{name}.decoder.referenceEmbeds.firstLogits.relL2",
                       round(qg.rel_l2(alone, ref_logits[0]), 6), None, True, gated=False,
                       cosine=qg.cosine(alone, ref_logits[0]))
                row["referenceEmbedsFirstLogitsRelL2"] = qg.rel_l2(alone, ref_logits[0])
    summary = list(rows.values())
    reloads = getattr(decoder, "reloads", 0) if decoder else 0
    del decoder
    passed = all(c["passed"] for c in checks if c["gated"])
    args.report.write_text(json.dumps({
        "passed": passed, "seconds": round(time.monotonic() - started, 1), "loadSeconds": load, "timings": timings,
        "decoderReloads": {"every": args.reload_every, "count": reloads},
        "peakRSSGB": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),
        "bundle": str(args.bundle), "visionFrom": str(vision_bundle), "reference": str(args.reference),
        "precision": manifest.get("precision"),
        "aetherJSONSHA256": qg.sha256_file(args.bundle / ".aether/aether.json"),
        "bars": {"embeddingCosine": COSINE_BAR, "greedyTokens": GREEDY_BAR},
        "summary": summary, "checks": checks}, indent=2) + "\n")
    if not passed:
        raise SystemExit("parity check failed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The YOLOX-S gate: the exported Core AI graph (GPU) against the upstream CPU float32 references.

    Forge/.venv/bin/python -I Forge/yolox/check_yolox.py --bundle <bundle dir> --reference <reference dir> \
        --report <new check-report.json>

With the bundle's own manifest (graph file, function, role names, compute placement, postprocess defaults), for every
reference case (reference_yolox.py): the case's input tensor, converted to the graph's input dtype, gives
- the detection bar (docs/M4.md): after the same host postprocess (`yolox_graphs.dense_nms` with the manifest's
  threshold, IoU and max), the same class set, every reference detection paired with a graph detection of its class
  at IoU >= 0.95, scores within 0.01;
- the raw output: preds relative L2 separately for the box columns (cx, cy, w, h) and the probability columns (obj,
  cls), and the largest absolute difference of score = obj * max(cls) over all 8400 anchors. docs/M4.md sets no raw bar;
  these are set from the first measurement (6 cases, M-series GPU), at about 3x the worst value: float32 graph 2e-6
  (boxes), 1.5e-5 (probabilities), 2e-5 (score), measured 7e-7, 4.2e-6, 5.5e-6; float16 graph 0.005, 0.025, 0.025,
  measured 0.0015, 0.0082, 0.0089;
- a repeat run that is bit-identical.
The graph's descriptor is also checked against the manifest's assertions: image [1,3,640,640] (float16 or float32, as
the manifest's precision says); preds [1,8400,85]. Not gated, recorded: the same detection matching at score threshold
0.05 (many more, lower-confidence boxes). Writes each case's graph output and detections beside the report
(`<report stem>/<case>/`). Exits nonzero when a bar fails. Runs one small graph on the GPU for a few seconds.
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
import yolox_graphs as yg  # noqa: E402

IOU_BAR, SCORE_BAR = 0.95, 0.01          # docs/M4.md's detection bar
# Raw preds, by graph precision: (boxes rel L2, probabilities rel L2, |score diff| over all anchors).
RAW_BARS = {"fp32": (2e-6, 1.5e-5, 2e-5), "fp16": (0.005, 0.025, 0.025)}
LOW_THRESHOLD = 0.05


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


def scores(preds: np.ndarray) -> np.ndarray:
    p = preds.reshape(-1, yg.CHANNELS).astype(np.float64)
    return p[:, 4] * p[:, 5:].max(1)


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
    spec = manifest["graphs"]["detector"]
    post = manifest["postprocess"]
    threshold, iou = post["defaults"]["scoreThreshold"], post["nms"]["iou"]
    max_det = post["defaults"]["maxDetections"]
    if (post["kind"], post["boxFormat"], post["boxSpace"], post["nms"]["type"]) != (
            "denseNms", "cxcywh", "letterboxPixels", "perClass"):
        raise SystemExit(f"unexpected postprocess block {post}")
    image_name, preds_name = spec["inputs"]["image"], spec["outputs"]["preds"]
    input_dtype = np.float16 if manifest["precision"]["weights"] == "fp16" else np.float32
    box_bar, prob_bar, score_bar = RAW_BARS[manifest["precision"]["weights"]]
    dump = args.report.with_suffix("")
    checks: list[dict] = []
    info: dict[str, dict] = {}

    def record(name: str, value: float, bar: float, passed: bool, **extra) -> None:
        checks.append({"check": name, "value": value, "bar": bar, "passed": bool(passed), **extra})
        print(f"[check] {'PASS' if passed else 'FAIL'} {name}: {value} (bar {bar})", file=sys.stderr, flush=True)

    started = time.monotonic()
    t0 = time.monotonic()
    graph = Graph(args.bundle / spec["file"].split(":", 1)[1], spec["function"], spec["compute"])
    load_seconds = round(time.monotonic() - t0, 2)
    descriptor = graph.descriptor()
    shapes = index["shapes"]
    got_in, got_out = descriptor["inputs"].get(image_name, {}), descriptor["outputs"].get(preds_name, {})
    ok = (got_in.get("dtype", "").endswith(np.dtype(input_dtype).name) and got_in.get("shape") == shapes["input"]
          and got_out.get("shape") == shapes["preds"])
    record("descriptor", float(ok), 1.0, ok, descriptor=descriptor)

    timings = []
    for entry in index["cases"]:
        case = json.loads((args.reference / entry["name"] / "case.json").read_text())
        folder = args.reference / case["name"]
        x = yg.read_f32(folder / "input.f32", shapes["input"])
        want = yg.read_f32(folder / "preds.f32", shapes["preds"])
        want_dets = json.loads((folder / "detections.json").read_text())
        feed = x.astype(input_dtype)
        if not np.array_equal(feed.astype(np.float32), x):
            raise SystemExit(f"{case['name']}: input.f32 is not exact in {np.dtype(input_dtype).name}")
        t0 = time.monotonic()
        out = graph(**{image_name: feed})
        timings.append(round(time.monotonic() - t0, 4))
        got = out[preds_name].astype(np.float32).reshape(shapes["preds"])
        name = case["name"]

        boxes = yg.rel_l2(got[..., :4], want[..., :4])
        record(f"{name}.preds.boxes.relL2", round(boxes, 9), box_bar, boxes <= box_bar,
               maxAbsPixels=float(np.abs(got[..., :4] - want[..., :4]).max()), dtype=str(out[preds_name].dtype))
        probs = yg.rel_l2(got[..., 4:], want[..., 4:])
        record(f"{name}.preds.probabilities.relL2", round(probs, 9), prob_bar, probs <= prob_bar,
               maxAbs=float(np.abs(got[..., 4:] - want[..., 4:]).max()))
        score_diff = float(np.abs(scores(got) - scores(want)).max())
        record(f"{name}.preds.score.maxAbs", round(score_diff, 9), score_bar, score_diff <= score_bar)

        got_dets = yg.dense_nms(got, threshold, iou, max_det)
        m = yg.match_detections(got_dets, want_dets)
        record(f"{name}.detections.sameClassSet", float(m["sameClassSet"]), 1.0,
               m["sameClassSet"] and m["unmatchedReference"] == 0 and m["unmatchedGraph"] == 0,
               reference=m["reference"], graph=m["graph"], matched=m["matched"])
        record(f"{name}.detections.minIoU", round(m["minIoU"], 6), IOU_BAR, m["minIoU"] >= IOU_BAR)
        record(f"{name}.detections.maxScoreDiff", round(m["maxScoreDiff"], 6), SCORE_BAR,
               m["maxScoreDiff"] <= SCORE_BAR)
        low = yg.match_detections(yg.dense_nms(got, LOW_THRESHOLD, iou, max_det),
                                  yg.dense_nms(want, LOW_THRESHOLD, iou, max_det))
        info[name] = {"scoreThreshold": LOW_THRESHOLD, **low}

        again = graph(**{image_name: feed})
        same = np.array_equal(again[preds_name], out[preds_name])
        record(f"{name}.repeat.exact", float(same), 1.0, same)
        (dump / name).mkdir(parents=True, exist_ok=True)
        yg.write_f32(dump / name / "preds.f32", got)
        (dump / name / "detections.json").write_text(json.dumps(got_dets, indent=1) + "\n")
    del graph
    passed = all(c["passed"] for c in checks)
    args.report.write_text(json.dumps({"passed": passed, "seconds": round(time.monotonic() - started, 1),
                                       "loadSeconds": load_seconds, "runSeconds": timings,
                                       "bundle": str(args.bundle), "reference": str(args.reference),
                                       "aetherJSONSHA256": yg.sha256_file(args.bundle / ".aether/aether.json"),
                                       "precision": manifest["precision"],
                                       "checks": checks, "lowThresholdInfo": info}, indent=2) + "\n")
    if not passed:
        raise SystemExit("parity check failed")


if __name__ == "__main__":
    main()

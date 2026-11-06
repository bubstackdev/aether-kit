#!/usr/bin/env python3
"""Eager CPU self-test of the export graph and the host postprocess, before any Core AI conversion.

    Forge/.venv/bin/python -I Forge/yolox/selftest_yolox.py [--reference <reference dir>] [--report <json>]

- `YoloxGraph` in float32 equals upstream's own eval `YOLOX.forward` bit for bit (so moving the decode after the
  network's output changes nothing).
- The float16 graph, run eagerly, is within the gate's detection bar (check_yolox.py).
- `dense_nms` (the host postprocess the Swift driver is gated against) selects exactly what upstream's own
  `yolox.utils.postprocess` (class-aware torchvision `batched_nms`) selects, at the default threshold 0.3 and at 0.05,
  where much more is suppressed: the same anchors' classes, scores and boxes, in the same order (before the max-50 cut,
  which upstream does not apply).
- `--sample-manifest`'s manifest names the `detector-dense@1` roles (MANIFEST.md §1.3) and the denseNms postprocess.
With `--reference`, the cases are reference_yolox.py's (and upstream's float32 output must reproduce them exactly);
otherwise upstream's dog.jpg and one seeded synthetic image. Exits nonzero when a check fails. CPU only; a few seconds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_yolox  # noqa: E402
import yolox_graphs as yg  # noqa: E402
from check_yolox import IOU_BAR, SCORE_BAR  # noqa: E402


def upstream_selection(boxes_module, preds: torch.Tensor, threshold: float) -> list[dict]:
    """Upstream's postprocess (class-aware), as [{classIndex, score, box xyxy}] in its output order."""
    out = boxes_module.postprocess(preds.clone(), yg.NUM_CLASSES, threshold, yg.NMS_IOU, class_agnostic=False)[0]
    if out is None:
        return []
    return [{"classIndex": int(r[6]), "score": float(r[4] * r[5]), "box": [float(v) for v in r[:4]]} for r in out]


def same_selection(mine: list[dict], theirs: list[dict]) -> bool:
    if len(mine) != len(theirs):
        return False
    for m, t in zip(mine, theirs, strict=True):
        xyxy = yg.cxcywh_to_xyxy(np.array(m["box"], dtype=np.float64))
        if (m["classIndex"] != t["classIndex"] or abs(m["score"] - t["score"]) > 1e-6
                or np.abs(xyxy - np.array(t["box"])).max() > 1e-3):
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--weights", type=Path, default=yg.DEFAULT_WEIGHTS)
    ap.add_argument("--upstream", type=Path, default=yg.DEFAULT_UPSTREAM)
    args = ap.parse_args()
    yg.verify_sources(args.weights, args.upstream)
    torch.set_grad_enabled(False)
    checks: list[dict] = []

    def record(name: str, value: float, bar: float, passed: bool) -> None:
        checks.append({"check": name, "value": value, "bar": bar, "passed": bool(passed)})
        print(f"[selftest] {'PASS' if passed else 'FAIL'} {name}: {value} (bar {bar})", file=sys.stderr, flush=True)

    cases = []
    if args.reference:
        index = json.loads((args.reference / "references.json").read_text())
        for entry in index["cases"]:
            folder = args.reference / entry["name"]
            cases.append((entry["name"], yg.read_f32(folder / "input.f32", index["shapes"]["input"]),
                          yg.read_f32(folder / "preds.f32", index["shapes"]["preds"])))
    else:
        cases.append(("dog", yg.letterbox_tensor(args.upstream / "assets/dog.jpg")[0], None))
        cases.append(("synthetic-3", yg.synthetic_tensor(3), None))

    upstream = yg.build_net(args.upstream, args.weights)
    _, boxes_module = yg.load_upstream(args.upstream)
    exact = yg.YoloxGraph(yg.build_net(args.upstream, args.weights), torch.float32).eval()
    mixed = yg.YoloxGraph(yg.build_net(args.upstream, args.weights), torch.float16).eval()
    for name, x, recorded in cases:
        image = torch.from_numpy(x)
        want = yg.upstream_preds(upstream, image)
        same = torch.equal(exact(image), want)
        record(f"{name}.float32.exact", float(same), 1.0, same)
        if recorded is not None:
            again = np.array_equal(want.numpy(), recorded)
            record(f"{name}.reference.reproduced", float(again), 1.0, again)
        for threshold in (yg.SCORE_THRESHOLD, 0.05):
            mine = yg.dense_nms(want.numpy(), threshold, yg.NMS_IOU, max_detections=yg.ANCHORS)
            agree = same_selection(mine, upstream_selection(boxes_module, want, threshold))
            record(f"{name}.nms@{threshold}.matchesUpstream", float(agree), 1.0, agree)
        half = mixed(image.half()).numpy()
        m = yg.match_detections(yg.dense_nms(half), yg.dense_nms(want.numpy()))
        same_set = m["sameClassSet"] and m["unmatchedReference"] == 0 and m["unmatchedGraph"] == 0
        record(f"{name}.float16.sameClassSet", float(same_set), 1.0, same_set)
        record(f"{name}.float16.minIoU", round(m["minIoU"], 6), IOU_BAR, m["minIoU"] >= IOU_BAR)
        record(f"{name}.float16.maxScoreDiff", round(m["maxScoreDiff"], 6), SCORE_BAR, m["maxScoreDiff"] <= SCORE_BAR)

    sample = export_yolox.sample_manifest("macos")
    graph = sample["graphs"]["detector"]
    post = sample["postprocess"]
    roles = (sample["driver"] == "detector-dense@1" and graph["inputs"] == {"image": "image"}
             and graph["outputs"] == {"preds": "preds"}
             and export_yolox.INPUTS == ["image"] and export_yolox.OUTPUTS == ["preds"]
             and post["kind"] == "denseNms" and post["boxFormat"] == "cxcywh" and post["boxSpace"] == "letterboxPixels"
             and post["nms"] == {"type": "perClass", "iou": yg.NMS_IOU}
             and post["defaults"] == {"scoreThreshold": yg.SCORE_THRESHOLD, "maxDetections": yg.MAX_DETECTIONS}
             and sample["labels"]["set"] == "coco-91"
             and sample["preprocess"]["image"]["channelOrder"] == "BGR"
             and sample["preprocess"]["image"]["resize"] == {"mode": "letterbox", "size": [yg.SIZE, yg.SIZE],
                                                             "anchor": "topLeft", "padValue": yg.PAD,
                                                             "filter": "vImage"})
    record("manifest.roles", float(roles), 1.0, roles)
    passed = all(c["passed"] for c in checks)
    if args.report:
        args.report.write_text(json.dumps({"passed": passed, "checks": checks}, indent=2) + "\n")
    if not passed:
        raise SystemExit("self-test failed")


if __name__ == "__main__":
    main()

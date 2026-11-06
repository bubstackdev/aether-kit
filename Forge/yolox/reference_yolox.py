#!/usr/bin/env python3
"""Upstream-PyTorch YOLOX-S references for the Core AI gate (check_yolox.py) and the Swift driver gate.

    Forge/.venv/bin/python -I Forge/yolox/reference_yolox.py --output <new reference dir>

Runs the unmodified upstream `YOLOX.forward` (eval, with the head's inference decode; pinned yolox_s.pth) on the CPU in
float32, on exactly the tensor the exported graph is fed: BGR on 0..255, letterboxed top-left to [1,3,640,640] with
pad 114 (integers, so exact in the graph's float16 input). Then the host postprocess (`yolox_graphs.dense_nms`: score
= obj * max(cls), threshold 0.3, per-class NMS at IoU 0.45, at most 50) in numpy. Preprocessing is not under test
here: each case's input tensor is written out, and gates feed it as is. For each case:

    <case>/case.json         source, preprocessing recipe and letterbox geometry, shapes, SHA-256 of each file
    <case>/input.f32         the graph input [1,3,640,640], little-endian float32
    <case>/preds.f32         upstream preds [1,8400,85], little-endian float32 (cx,cy,w,h letterbox px, obj, cls[80])
    <case>/detections.json   the postprocess at the driver defaults: class index, COCO id, name, score, box (cxcywh
                             letterbox pixels) and box in the source image (xyxy, normalized by its width and height)

and an index `references.json` (with the 80 class names and COCO ids). Cases: upstream's dog.jpg; its demo.png collage
whole (a 5.8:1 strip, so a 640 x 111 letterbox) and three of its four panels cropped; one seeded synthetic image. CPU
only; no GPU lock needed.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import yolox_graphs as yg  # noqa: E402

CASES = [
    {"name": "dog", "image": "assets/dog.jpg"},
    {"name": "demo", "image": "assets/demo.png"},
    {"name": "demo-traffic", "image": "assets/demo.png", "crop": [0, 0, 610, 424]},
    {"name": "demo-soccer", "image": "assets/demo.png", "crop": [611, 0, 1221, 424]},
    {"name": "demo-bikes", "image": "assets/demo.png", "crop": [1833, 0, 2443, 424]},
    {"name": "synthetic-7", "synthetic": 7},
]
SHAPES = {"input": [1, 3, yg.SIZE, yg.SIZE], "preds": [1, yg.ANCHORS, yg.CHANNELS]}
# Sanity on upstream's own demo photo: YOLOX-S finds the dog and the bicycle (README demo).
EXPECTED = {"dog": {"dog", "bicycle"}}


def detections(preds: np.ndarray, names: tuple[str, ...], letterbox: dict | None) -> list[dict]:
    out = []
    for d in yg.dense_nms(preds):
        record = {**d, "cocoId": yg.COCO91_IDS[d["classIndex"]], "label": names[d["classIndex"]]}
        if letterbox is not None:
            (w, h), r = letterbox["source"], letterbox["ratio"]
            x1, y1, x2, y2 = yg.cxcywh_to_xyxy(np.array(d["box"], dtype=np.float64))
            record["boxSourceNormalized"] = [x1 / r / w, y1 / r / h, x2 / r / w, y2 / r / h]
        out.append(record)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=yg.DEFAULT_WEIGHTS)
    ap.add_argument("--upstream", type=Path, default=yg.DEFAULT_UPSTREAM)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    verified = yg.verify_sources(args.weights, args.upstream)
    args.output.mkdir(parents=True)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    net = yg.build_net(args.upstream, args.weights)
    names = yg.coco_classes(args.upstream)
    index = {
        "schema": 1, "generator": "Forge/yolox/reference_yolox.py",
        "scripts": {p.name: yg.sha256_file(p) for p in sorted(HERE.glob("*.py"))},
        "source": {"repository": yg.REPO, "weights": yg.WEIGHTS,
                   "upstreamCode": {"repository": yg.CODE_REPO, "commit": yg.CODE_COMMIT}, "verified": verified},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "numpy", "pillow")},
        "host": {"machine": platform.machine(), "python": platform.python_version()},
        "precision": "float32 on the CPU (upstream YOLOX.forward, eval, unmodified); postprocess in float64 numpy",
        "shapes": SHAPES, "layout": "row-major, little-endian float32",
        "graphInput": "image = input.f32 (converted to float16 for a float16 graph: exact, every value is an integer "
                      "on 0..255)",
        "postprocess": {"kind": "denseNms", "score": "obj * max(cls)", "scoreThreshold": yg.SCORE_THRESHOLD,
                        "nms": {"type": "perClass", "iou": yg.NMS_IOU, "suppress": "IoU > iou, xyxy, no +1"},
                        "maxDetections": yg.MAX_DETECTIONS, "rule": "yolox_graphs.dense_nms"},
        "labels": {"set": "coco-91", "classIndex": "contiguous",
                   "classes": [{"index": i, "cocoId": yg.COCO91_IDS[i], "name": n} for i, n in enumerate(names)]},
        "cases": [],
    }
    for case in CASES:
        started = time.monotonic()
        out = args.output / case["name"]
        out.mkdir()
        letterbox = None
        if "image" in case:
            crop = tuple(case["crop"]) if "crop" in case else None
            x, letterbox = yg.letterbox_tensor(args.upstream / case["image"], crop)
            source = {"kind": "image", "upstreamFile": case["image"], "crop": case.get("crop"),
                      "sha256": yg.PINS["referenceImages"]["files"][case["image"]],
                      "preprocess": "Pillow convert('RGB'), crop (left, top, right, bottom) if given, r = min(640/h, "
                                    "640/w), resize to (int(w r), int(h r)) BILINEAR, top-left on 640x640 of 114, "
                                    "BGR, NCHW float32 0..255",
                      "letterbox": letterbox}
        else:
            x = yg.synthetic_tensor(case["synthetic"])
            source = {"kind": "synthetic", "seed": case["synthetic"], "recipe": "yolox_graphs.synthetic_tensor"}
        preds = yg.upstream_preds(net, torch.from_numpy(x)).numpy()
        if not np.isfinite(preds).all():
            raise SystemExit(f"{case['name']}: non-finite upstream output")
        dets = detections(preds, names, letterbox)
        missing = EXPECTED.get(case["name"], set()) - {d["label"] for d in dets}
        if missing:
            raise SystemExit(f"{case['name']}: upstream did not detect {sorted(missing)}; check the preprocessing")
        yg.write_f32(out / "input.f32", x)
        yg.write_f32(out / "preds.f32", preds)
        (out / "detections.json").write_text(json.dumps(dets, indent=1) + "\n")
        record = {"name": case["name"], "source": source, "shapes": SHAPES,
                  "files": {name: yg.sha256_file(out / name) for name in ("input.f32", "preds.f32",
                                                                          "detections.json")},
                  "detections": len(dets), "labels": sorted({d["label"] for d in dets}),
                  "seconds": round(time.monotonic() - started, 2)}
        (out / "case.json").write_text(json.dumps(record, indent=2) + "\n")
        index["cases"].append({"name": case["name"], "caseJSONSHA256": yg.sha256_file(out / "case.json")})
        print(f"[reference] {case['name']}: {len(dets)} detections {record['labels']}, {record['seconds']} s",
              file=sys.stderr, flush=True)
    index["peakRSSGB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
    (args.output / "references.json").write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    main()

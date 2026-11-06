#!/usr/bin/env python3
"""Upstream-PyTorch RF-DETR references for the Core AI gate (check_rfdetr.py) and the Swift driver gate.

    Forge/.venv/bin/python -I Forge/rfdetr/reference_rfdetr.py --output <new reference dir> [--variant nano|seg-nano]

Runs the unmodified upstream `LWDETR.forward` (built and loaded by upstream's own builder from the pinned checkpoint) on
the CPU in float32, on exactly the tensor the exported graph is fed: raw RGB on [0,1], [1,3,S,S], normalized as
upstream's `predict` does. Preprocessing is not under test here: each case's input tensor is written out, and gates feed
it as is. For each case:

    <case>/case.json          source, preprocessing recipe, shapes, SHA-256 of each file, the postprocessed detections
    <case>/input.f32          the graph input [1,3,S,S], little-endian float32
    <case>/dets.f32           upstream pred_boxes [1,Q,4] (cxcywh, normalized), little-endian float32
    <case>/labels.f32         upstream pred_logits [1,Q,91], little-endian float32
    <case>/masks.f32          (seg-nano) upstream pred_masks [1,Q,S/4,S/4], little-endian float32

and an index `references.json`. `case.json`'s `detections` are upstream's PostProcess at threshold 0.5 (scores, COCO
ids and names, xyxy boxes normalized to [0,1]): the expected answer a Swift driver gate compares against. Cases: five
CC0 Wikimedia Commons photos (street, bus, bicycles, a cat; source-pins.json), resized as upstream predict does, and one
seeded synthetic image. CPU only; no GPU lock needed.
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
import rfdetr_graphs as rg  # noqa: E402

CASES = [
    {"name": "burnside-irt", "image": "burnside-irt.jpg"},
    {"name": "wood-green-1981", "image": "wood-green-1981.jpg"},
    {"name": "bicycle-parade-271", "image": "bicycle-parade-271.jpg"},
    {"name": "bicycle-parade-285", "image": "bicycle-parade-285.jpg"},
    {"name": "black-cat-sofa", "image": "black-cat-sofa.jpg"},
    {"name": "synthetic-7", "synthetic": 7},
]


def detections_record(dets: dict[str, np.ndarray], size: int, names: list) -> list[dict]:
    return [{"score": float(s), "classId": int(c), "label": names[int(c)],
             "boxXYXY": [round(float(v) / size, 6) for v in b]}
            for s, c, b in zip(dets["scores"], dets["labels"], dets["boxes"])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--variant", choices=sorted(rg.VARIANTS), default="nano")
    ap.add_argument("--upstream", type=Path, default=rg.DEFAULT_UPSTREAM)
    ap.add_argument("--weights", type=Path, default=rg.DEFAULT_WEIGHTS)
    ap.add_argument("--images", type=Path, default=rg.DEFAULT_IMAGES)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    spec = rg.VARIANTS[args.variant]
    size = spec["size"]
    verified = rg.verify_sources(args.variant, args.upstream, args.weights, args.images)
    args.output.mkdir(parents=True)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    net, _, _ = rg.build_net(args.variant, args.upstream, args.weights)
    names = rg.coco_labels(args.upstream)
    shapes = rg.output_shapes(args.variant)
    index = {
        "schema": 1, "generator": "Forge/rfdetr/reference_rfdetr.py", "variant": args.variant, "id": spec["id"],
        "scripts": {p.name: rg.sha256_file(p) for p in sorted(HERE.glob("*.py"))},
        "source": {"upstreamCode": {"repository": rg.CODE_REPO, "tag": rg.CODE_TAG, "commit": rg.CODE_COMMIT},
                   "checkpoint": rg.PINS["checkpoints"][spec["checkpoint"]], "verified": verified},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "numpy", "pillow")},
        "host": {"machine": platform.machine(), "python": platform.python_version()},
        "precision": "float32 on the CPU (upstream LWDETR.forward, unmodified)",
        "shapes": shapes, "layout": "row-major, little-endian float32",
        "graphInput": "image = input.f32 (float32)",
        "postprocess": {"code": "upstream rfdetr.models.postprocess.PostProcess", "numSelect": spec["queries"],
                        "scoreThreshold": rg.SCORE_THRESHOLD, "targetSize": [size, size]},
        "cases": [],
    }
    for case in CASES:
        started = time.monotonic()
        out = args.output / case["name"]
        out.mkdir()
        if "image" in case:
            x = rg.image_tensor(args.images / case["image"], size)
            pin = rg.PINS["referenceImages"]["files"][case["image"]]
            source = {"kind": "image", "file": case["image"], **pin,
                      "preprocess": (f"Pillow convert('RGB'), to float32 / 255, torchvision resize to {size}x{size} "
                                     "bilinear antialias=False (stretch), NCHW (upstream predict)")}
        else:
            x = rg.synthetic_tensor(case["synthetic"], size)
            source = {"kind": "synthetic", "seed": case["synthetic"], "recipe": "rfdetr_graphs.synthetic_tensor"}
        result = {k: v.numpy() for k, v in rg.upstream_forward(net, torch.from_numpy(x)).items()}
        if not all(np.isfinite(v).all() for v in result.values()):
            raise SystemExit(f"{case['name']}: non-finite upstream output")
        dets = rg.postprocess(args.upstream, args.variant, result)
        if "image" in case and not len(dets["scores"]):
            raise SystemExit(f"{case['name']}: upstream detects nothing; not a useful reference case")
        files = {"input.f32": x, **{f"{k}.f32": v for k, v in result.items()}}
        for name, array in files.items():
            rg.write_f32(out / name, array)
        record = {"name": case["name"], "source": source, "shapes": shapes,
                  "files": {name: rg.sha256_file(out / name) for name in files},
                  "detections": detections_record(dets, size, names),
                  "classes": sorted({names[int(c)] for c in dets["labels"]}),
                  "seconds": round(time.monotonic() - started, 2)}
        (out / "case.json").write_text(json.dumps(record, indent=2) + "\n")
        index["cases"].append({"name": case["name"], "caseJSONSHA256": rg.sha256_file(out / "case.json"),
                               "detections": len(record["detections"])})
        print(f"[reference] {case['name']}: {len(record['detections'])} detections {record['classes']}, "
              f"{record['seconds']} s", file=sys.stderr, flush=True)
    index["peakRSSGB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
    (args.output / "references.json").write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    main()

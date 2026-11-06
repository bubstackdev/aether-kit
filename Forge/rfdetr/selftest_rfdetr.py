#!/usr/bin/env python3
"""Eager CPU self-test of the export graph, before any Core AI conversion.

    Forge/.venv/bin/python -I Forge/rfdetr/selftest_rfdetr.py [--variant nano|seg-nano] [--reference <dir>] [--report <json>]

- `DetrGraph` in float32 (upstream's `forward_export` behind the in-graph normalization) equals upstream's own
  `LWDETR.forward` behind `predict`'s normalization bit for bit, on every output.
- The exported program after the Core AI decomposition table (grid sampling as gathers), run on the CPU, passes the
  detection bars of check_rfdetr.py against upstream (so the decomposition changes nothing that matters).
- `--sample-manifest`'s manifest names the `detector-detr@1` roles (MANIFEST.md §1.3) and matches the graph's I/O.
With `--reference`, the cases are reference_rfdetr.py's (and upstream's float32 output must reproduce them exactly);
otherwise one seeded synthetic image. Exits nonzero when a check fails. CPU only; under a minute.
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
import rfdetr_graphs as rg  # noqa: E402
import export_rfdetr  # noqa: E402
from check_rfdetr import IOU_BAR, MASK_IOU_BAR, SCORE_BAR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(rg.VARIANTS), default="nano")
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--upstream", type=Path, default=rg.DEFAULT_UPSTREAM)
    ap.add_argument("--weights", type=Path, default=rg.DEFAULT_WEIGHTS)
    args = ap.parse_args()
    rg.verify_sources(args.variant, args.upstream, args.weights, images=None)
    torch.set_grad_enabled(False)
    spec = rg.VARIANTS[args.variant]
    shapes = rg.output_shapes(args.variant)
    checks: list[dict] = []

    def record(name: str, value, bar, passed: bool) -> None:
        checks.append({"check": name, "value": value, "bar": bar, "passed": bool(passed)})
        print(f"[selftest] {'PASS' if passed else 'FAIL'} {name}: {value} (bar {bar})", file=sys.stderr, flush=True)

    cases = []
    if args.reference:
        index = json.loads((args.reference / "references.json").read_text())
        if index["variant"] != args.variant:
            raise SystemExit(f"--reference holds {index['variant']} references, not {args.variant}")
        for entry in index["cases"]:
            folder = args.reference / entry["name"]
            cases.append((entry["name"], rg.read_f32(folder / "input.f32", shapes["input"]),
                          {k: rg.read_f32(folder / f"{k}.f32", shapes[k]) for k in spec["outputs"]}))
    else:
        cases.append(("synthetic-3", rg.synthetic_tensor(3, spec["size"]), None))

    upstream, _, _ = rg.build_net(args.variant, args.upstream, args.weights)
    graph = rg.DetrGraph(upstream, torch.float32).eval()
    with torch.no_grad():
        ep = torch.export.export(graph, (torch.from_numpy(cases[0][1]),), strict=False)
        decomposed = ep.run_decompositions(rg.decomposition_table(args.upstream)).module()
    for name, x, recorded in cases:
        image = torch.from_numpy(x)
        want = {k: v.numpy() for k, v in rg.upstream_forward(upstream, image).items()}
        eager = dict(zip(spec["outputs"], (t.numpy() for t in graph(image))))
        same = all(np.array_equal(eager[k], want[k]) for k in spec["outputs"])
        record(f"{name}.float32.exact", float(same), 1.0, same)
        if recorded is not None:
            again = all(np.array_equal(want[k], recorded[k]) for k in spec["outputs"])
            record(f"{name}.reference.reproduced", float(again), 1.0, again)
        with torch.no_grad():
            lowered = dict(zip(spec["outputs"], (t.numpy() for t in decomposed(image))))
        m = rg.match_detections(rg.postprocess(args.upstream, args.variant, want),
                                rg.postprocess(args.upstream, args.variant, lowered))
        record(f"{name}.decomposed.labels.relL2", round(rg.rel_l2(lowered["labels"], want["labels"]), 8), None, True)
        record(f"{name}.decomposed.classSet", float(m["referenceClasses"] == m["graphClasses"]), 1.0,
               m["referenceClasses"] == m["graphClasses"])
        record(f"{name}.decomposed.minIoU", round(m["minIoU"], 6), IOU_BAR, m["minIoU"] >= IOU_BAR)
        record(f"{name}.decomposed.maxScoreDiff", round(m["maxScoreDiff"], 6), SCORE_BAR,
               m["maxScoreDiff"] <= SCORE_BAR)
        hard = [u for u in m["unmatched"] if not u["borderline"]]
        record(f"{name}.decomposed.unmatched", len(hard), 0, not hard)
        if m["minMaskIoU"] is not None:
            record(f"{name}.decomposed.minMaskIoU", round(m["minMaskIoU"], 6), MASK_IOU_BAR,
                   m["minMaskIoU"] >= MASK_IOU_BAR)

    sample = export_rfdetr.sample_manifest(args.variant, "macos")
    detector = sample["graphs"]["detector"]
    want_outputs = {"boxes": "dets", "classLogits": "labels"}
    if "masks" in spec["outputs"]:
        want_outputs["masks"] = "masks"
    roles = (sample["driver"] == "detector-detr@1" and detector["inputs"] == {"image": "image"}
             and detector["outputs"] == want_outputs and set(want_outputs.values()) == set(spec["outputs"])
             and sample["postprocess"]["kind"] == "detrSigmoid" and sample["labels"] == {**sample["labels"],
                                                                                         "set": "coco-91"}
             and ("masks" in sample["postprocess"]) == ("masks" in spec["outputs"])
             and sample["preprocess"]["image"]["resize"] == {"mode": "stretch", "size": [spec["size"], spec["size"]],
                                                             "filter": "torchBilinear"})
    record("manifest.roles", float(roles), 1.0, roles)
    passed = all(c["passed"] for c in checks)
    if args.report:
        args.report.write_text(json.dumps({"passed": passed, "variant": args.variant, "checks": checks}, indent=2)
                               + "\n")
    if not passed:
        raise SystemExit("self-test failed")


if __name__ == "__main__":
    main()

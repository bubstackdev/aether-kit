#!/usr/bin/env python3
"""The RF-DETR gate: the exported Core AI graph (GPU) against the upstream CPU float32 references.

    Forge/.venv/bin/python -I Forge/rfdetr/check_rfdetr.py --bundle <bundle dir> --reference <reference dir> \
        --report <new check-report.json>

With the bundle's own manifest (graph file, function, role names, compute placement), for every reference case
(reference_rfdetr.py), the case's input tensor is fed as is and both sides' raw outputs go through the identical Python
postprocess (upstream's `PostProcess`, threshold 0.5). Detections are matched class by class (Hungarian on box IoU).
The bars (docs/M4.md, detection):
- the same class set;
- every matched pair: box IoU >= 0.95 and |score difference| <= 0.01;
- no unmatched detection on either side, except one whose score is within 0.01 of the threshold (it may fall on
  either side of it);
- seg-nano only: every matched pair's mask IoU (upsampled to the input, logit > 0) >= 0.95 (a starting bar set here;
  M4.md has none for masks);
- a repeat run that is bit-identical.
Raw relative L2 of `dets`, `labels` (and `masks`) over all queries is reported, not gated: it is dominated by the
low-score queries the threshold discards, and a different two-stage query selection reorders queries.
The graph's descriptor is also checked: image [1,3,S,S] float32 and the output shapes. Writes each case's graph outputs
beside the report (`<report stem>/<case>/*.f32`). Exits nonzero when a bar fails. Runs one graph on the GPU for a few
seconds.
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
import rfdetr_graphs as rg  # noqa: E402

IOU_BAR, SCORE_BAR, MASK_IOU_BAR = 0.95, 0.01, 0.95


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
    parser.add_argument("--upstream", type=Path, default=rg.DEFAULT_UPSTREAM, help="for upstream's PostProcess")
    args = parser.parse_args()
    if args.report.exists():
        raise SystemExit("report already exists; choose a new path")
    manifest = json.loads((args.bundle / ".aether/aether.json").read_text())
    index = json.loads((args.reference / "references.json").read_text())
    variant = index["variant"]
    if manifest["id"] != rg.VARIANTS[variant]["id"]:
        raise SystemExit(f"bundle {manifest['id']} does not match the {variant} references")
    rg.verify_sources(variant, args.upstream, images=None)
    spec = manifest["graphs"]["detector"]
    image_name = spec["inputs"]["image"]
    names = {role: spec["outputs"][role] for role in ("boxes", "classLogits", "masks") if role in spec["outputs"]}
    export_name = {"boxes": "dets", "classLogits": "labels", "masks": "masks"}
    shapes = index["shapes"]
    dump = args.report.with_suffix("")
    checks: list[dict] = []

    def record(name: str, value, bar, passed: bool, **extra) -> None:
        checks.append({"check": name, "value": value, "bar": bar, "passed": bool(passed), **extra})
        print(f"[check] {'PASS' if passed else 'FAIL'} {name}: {value} (bar {bar})", file=sys.stderr, flush=True)

    started = time.monotonic()
    t0 = time.monotonic()
    graph = Graph(args.bundle / spec["file"].split(":", 1)[1], spec["function"], spec["compute"])
    load_seconds = round(time.monotonic() - t0, 2)
    descriptor = graph.descriptor()
    image_desc = descriptor["inputs"].get(image_name, {})
    ok = (image_desc.get("dtype", "").endswith("float32") and image_desc.get("shape") == shapes["input"]
          and all(descriptor["outputs"].get(n, {}).get("shape") == shapes[export_name[role]]
                  for role, n in names.items()))
    record("descriptor", float(ok), 1.0, ok, descriptor=descriptor)

    timings, summary = [], []
    for entry in index["cases"]:
        case = json.loads((args.reference / entry["name"] / "case.json").read_text())
        folder = args.reference / case["name"]
        x = rg.read_f32(folder / "input.f32", shapes["input"])
        want = {export_name[role]: rg.read_f32(folder / f"{export_name[role]}.f32", shapes[export_name[role]])
                for role in names}
        t0 = time.monotonic()
        out = graph(**{image_name: x})
        timings.append(round(time.monotonic() - t0, 4))
        got = {export_name[role]: out[n].astype(np.float32).reshape(shapes[export_name[role]])
               for role, n in names.items()}
        prefix = case["name"]
        for key in got:
            e = rg.rel_l2(got[key], want[key])
            record(f"{prefix}.{key}.relL2", float(f"{e:.3e}"), None, True, gating=False,
                   maxAbs=float(np.abs(got[key] - want[key]).max()))
        want_dets = rg.postprocess(args.upstream, variant, want)
        got_dets = rg.postprocess(args.upstream, variant, got)
        m = rg.match_detections(want_dets, got_dets)
        same = m["referenceClasses"] == m["graphClasses"]
        record(f"{prefix}.classSet", float(same), 1.0, same, reference=m["referenceClasses"],
               graph=m["graphClasses"])
        record(f"{prefix}.minIoU", round(m["minIoU"], 6), IOU_BAR, m["minIoU"] >= IOU_BAR, matched=m["matched"],
               referenceCount=m["referenceCount"], graphCount=m["graphCount"])
        record(f"{prefix}.maxScoreDiff", round(m["maxScoreDiff"], 6), SCORE_BAR, m["maxScoreDiff"] <= SCORE_BAR)
        hard = [u for u in m["unmatched"] if not u["borderline"]]
        record(f"{prefix}.unmatched", len(hard), 0, not hard, unmatched=m["unmatched"])
        if m["minMaskIoU"] is not None:
            record(f"{prefix}.minMaskIoU", round(m["minMaskIoU"], 6), MASK_IOU_BAR, m["minMaskIoU"] >= MASK_IOU_BAR)
        again = graph(**{image_name: x})
        same_run = all(np.array_equal(again[n], out[n]) for n in names.values())
        record(f"{prefix}.repeat.exact", float(same_run), 1.0, same_run)
        summary.append({"case": prefix, "reference": m["referenceCount"], "graph": m["graphCount"],
                        "matched": m["matched"], "minIoU": round(m["minIoU"], 6),
                        "maxScoreDiff": round(m["maxScoreDiff"], 6),
                        **({"minMaskIoU": round(m["minMaskIoU"], 6)} if m["minMaskIoU"] is not None else {})})
        (dump / prefix).mkdir(parents=True, exist_ok=True)
        for key, array in got.items():
            rg.write_f32(dump / prefix / f"{key}.f32", array)
    del graph
    passed = all(c["passed"] for c in checks)
    args.report.write_text(json.dumps({"passed": passed, "seconds": round(time.monotonic() - started, 1),
                                       "loadSeconds": load_seconds, "runSeconds": timings,
                                       "bundle": str(args.bundle), "reference": str(args.reference),
                                       "precision": manifest.get("precision"),
                                       "aetherJSONSHA256": rg.sha256_file(args.bundle / ".aether/aether.json"),
                                       "bars": {"boxIoU": IOU_BAR, "scoreDiff": SCORE_BAR, "maskIoU": MASK_IOU_BAR,
                                                "scoreThreshold": rg.SCORE_THRESHOLD},
                                       "summary": summary, "checks": checks}, indent=2) + "\n")
    if not passed:
        raise SystemExit("parity check failed")


if __name__ == "__main__":
    main()

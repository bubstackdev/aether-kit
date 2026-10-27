#!/usr/bin/env python3
"""Compose the Qwen3-VL vision-encoder evidence record (development tooling only; standard library only).

    python3 Forge/language/record_vision_evidence.py --native <parity report .json> --reference <reference dir> \
        --export-report <vision export-report.json> --model <vision.aimodel> \
        --stability <stability report .json> [--stability ...] [--control <dynamic control .json> ...] \
        --output docs/evidence/<name>.json

Binds the native parity report, the stability reports (one per process), the dynamic-shape control, the upstream
reference (record and per-image files) and the export (report and model files) by SHA-256, and summarizes them. The
native reports are copied next to the record (`<name>-native.json`, `<name>-stability-<i>.json`,
`<name>-dynamic-control-<i>.json`).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def copy(source: Path, target: Path) -> dict:
    target.write_text(source.read_text())
    return {"file": target.name, "sha256": sha256(target)}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("native", "reference", "export-report", "model", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--stability", type=Path, action="append", required=True)
    parser.add_argument("--control", type=Path, action="append", default=[])
    args = parser.parse_args()
    native = json.loads(args.native.read_text())
    reference = json.loads((args.reference / "reference.json").read_text())
    export = json.loads(args.export_report.read_text())
    out = args.output
    here = Path(__file__).resolve().parent
    worst = lambda rows: min(min([r["imageEmbedsCosine"], *r["deepstackCosine"]]) for r in rows)  # noqa: E731

    stability_files, per_function = [], collections.defaultdict(lambda: {"encodes": 0, "differing": 0, "outputs": set()})
    for i, path in enumerate(args.stability, 1):
        r = json.loads(path.read_text())
        stability_files.append({**copy(path, out.with_name(f"{out.stem}-stability-{i}.json")), "passed": r["passed"],
                                "footprintMiB": {f["phase"]: round(f["bytes"] / 2**20) for f in r["footprint"]}})
        for g in r["grids"]:
            entry = per_function[g["function"]]
            entry["encodes"] += g["encodes"]
            entry["differing"] += g["differingFromFirst"]
            entry["outputs"].add((g["firstEmbedsHash"], g["firstDeepstackHash"]))
            entry["meanEncodeSeconds"] = round(g["meanEncodeSeconds"], 4)
    stability = {f: {"encodes": e["encodes"], "processes": len(args.stability), "differingFromFirst": e["differing"],
                     "sameFirstOutputAcrossProcesses": len(e["outputs"]) == 1, "meanEncodeSeconds": e["meanEncodeSeconds"]}
                 for f, e in per_function.items()}

    control_files, control_rows, control_outputs = [], 0, collections.Counter()
    bad_cosines = set()
    for i, path in enumerate(args.control, 1):
        r = json.loads(path.read_text())
        control_files.append({**copy(path, out.with_name(f"{out.stem}-dynamic-control-{i}.json")), "grid": r["grid"],
                              "encodes": r["encodes"], "differingFromFirst": sum(not x["sameAsFirst"] for x in r["rows"])})
        for x in r["rows"]:
            control_rows += 1
            control_outputs[(x["embedsHash"], x["deepstackHash"])] += 1
            if not x["sameAsFirst"]:
                bad_cosines.add((round(x["embedsCosine"], 6), round(x["deepstackCosine"], 6)))
    differing = sum(c["differingFromFirst"] for c in control_files)

    record = {
        "schema": 2,
        "kind": "qwen3-vl-vision-encoder-qualification",
        "passed": native["passed"] and all(s["passed"] for s in stability_files)
                  and all(v["differingFromFirst"] == 0 and v["sameFirstOutputAcrossProcesses"] for v in stability.values()),
        "device": native["device"],
        "hostLoad": native["hostLoad"],
        "supersedes": "qwen3-vl-vision-encoder-2026-10-26.json (one dynamic-shape `encode` function; see dynamicControl)",
        "model": {"repository": reference["source"]["repository"], "revision": reference["source"]["revision"],
                  "license": reference["source"]["license"], "gated": False,
                  "checkpointSHA256": reference["checkpointSHA256"]},
        "asset": {"directory": str(args.model), "functions": [g["function"] for g in export["imageGrids"]],
                  "imageGrids": export["imageGrids"], "shapes": "static: no dynamic dimension in any function",
                  "precision": "fp16 weights, float32 compute", "reportSHA256": sha256(args.export_report),
                  "modelFiles": export["modelFiles"], "versions": export["versions"],
                  "cpuOnlySelfCheck": export["selfCheck"]},
        "reference": {"oracle": reference["oracle"], "versions": reference["versions"],
                      "recordSHA256": sha256(args.reference / "reference.json"), "preprocessing": reference["preprocessing"],
                      "images": [{"name": i["name"], "size": [i["width"], i["height"]], "grid": i["grid"],
                                  "patches": i["patches"], "tokens": i["tokens"],
                                  "fileSHA256": {k: v["sha256"] for k, v in i["files"].items()},
                                  "eagerPortVsReference": i["eagerPortVsReference"]} for i in reference["images"]]},
        "hostTables": "AetherVision.Qwen3VLVisionTables (bit-exact with Forge host_inputs, CPU-tested)",
        "parity": {"gate": native["gate"], "results": native["results"], "worstCosine": worst(native["results"]),
                   "footprintMiB": {f["phase"]: round(f["bytes"] / 2**20) for f in native["footprint"]},
                   "report": copy(args.native, out.with_name(f"{out.stem}-native.json"))},
        "stability": {"gate": "every encode of every grid bit-identical to that grid's first encode; same first output "
                              "in every process", "byFunction": stability, "reports": stability_files},
        "dynamicControl": {
            "what": "the superseded dynamic-shape `encode` asset, same photo and Kit preprocessing (grid 1x48x40), "
                    "LanguageVisionEncoder at f63aa76, specialized without expectFrequentReshapes, GPU-preferred",
            "encodes": control_rows, "processes": len(control_files), "differingFromFirst": differing,
            "rate": round(differing / control_rows, 4) if control_rows else None,
            "distinctOutputs": len(control_outputs), "badOutputCosines": sorted(bad_cosines),
            "reports": control_files,
        } if control_files else None,
        "scriptSHA256": {p.name: sha256(p) for p in sorted(here.glob("*.py"))},
        "nativeTests": "Tests/AetherNativeTests/Qwen3VLVisionQualificationTests.swift (AETHER_NATIVE_QWEN3VL_VISION=1 "
                       "parity, AETHER_NATIVE_QWEN3VL_STABILITY=1 stability; release)",
    }
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"passed": record["passed"], "parityWorstCosine": record["parity"]["worstCosine"],
                      "stability": stability, "dynamicControl": {k: record["dynamicControl"][k] for k in
                      ("encodes", "differingFromFirst", "rate", "distinctOutputs")} if control_files else None}, indent=1))


if __name__ == "__main__":
    main()

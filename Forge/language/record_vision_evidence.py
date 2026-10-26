#!/usr/bin/env python3
"""Compose the Qwen3-VL vision-encoder evidence record (development tooling only; standard library only).

    python3 Forge/language/record_vision_evidence.py --native <native report .json> --reference <reference dir> \
        --export <vision export dir> --output docs/evidence/<name>.json

Binds the native report, the upstream reference (record and per-image files) and the export (report and model files)
by SHA-256, and summarizes the parity results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("native", "reference", "export", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    native = json.loads(args.native.read_text())
    reference = json.loads((args.reference / "reference.json").read_text())
    export = json.loads((args.export / "export-report.json").read_text())
    copied = args.output.with_name(args.output.stem + "-native.json")
    copied.write_text(args.native.read_text())
    here = Path(__file__).resolve().parent
    worst = lambda rows: min(min([r["imageEmbedsCosine"], *r["deepstackCosine"]]) for r in rows)  # noqa: E731
    record = {
        "schema": 1,
        "kind": "qwen3-vl-vision-encoder-qualification",
        "passed": native["passed"],
        "device": native["device"],
        "hostLoad": native["hostLoad"],
        "model": {"repository": reference["source"]["repository"], "revision": reference["source"]["revision"],
                  "license": reference["source"]["license"], "gated": False,
                  "checkpointSHA256": reference["checkpointSHA256"]},
        "asset": {"directory": str(args.export), "function": "encode", "precision": "fp16 weights, float32 compute",
                  "reportSHA256": sha256(args.export / "export-report.json"), "modelFiles": export["modelFiles"],
                  "versions": export["versions"], "cpuOnlySelfCheck": export["selfCheck"]},
        "reference": {"oracle": reference["oracle"], "versions": reference["versions"],
                      "recordSHA256": sha256(args.reference / "reference.json"), "preprocessing": reference["preprocessing"],
                      "images": [{"name": i["name"], "size": [i["width"], i["height"]], "grid": i["grid"],
                                  "patches": i["patches"], "tokens": i["tokens"],
                                  "fileSHA256": {k: v["sha256"] for k, v in i["files"].items()},
                                  "eagerPortVsReference": i["eagerPortVsReference"]} for i in reference["images"]]},
        "hostTables": "AetherVision.Qwen3VLVisionTables (bit-exact with Forge host_inputs, CPU-tested for 8 grids)",
        "gate": native["gate"],
        "native": {"reshapesOff": native["reshapesOff"], "reshapesOffWorstCosine": worst(native["reshapesOff"]),
                   "reshapesOnDiagnostic": native["reshapesOnDiagnostic"],
                   "reshapesOnWorstCosine": worst(native["reshapesOnDiagnostic"]),
                   "reshapesOnNote": "Specialized with expectFrequentReshapes the dynamic-shape encoder computes wrong "
                                     "outputs on the GPU (engine F-P3-26 reproduced); the asset must be specialized "
                                     "without it."},
        "nativeReport": {"file": copied.name, "sha256": sha256(copied)},
        "scriptSHA256": {p.name: sha256(p) for p in sorted(here.glob("*.py"))},
        "nativeTest": "Tests/AetherNativeTests/Qwen3VLVisionQualificationTests.swift (AETHER_NATIVE_QWEN3VL_VISION=1, release)",
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"passed": record["passed"], "reshapesOffWorstCosine": record["native"]["reshapesOffWorstCosine"],
                      "reshapesOnWorstCosine": record["native"]["reshapesOnWorstCosine"]}))


if __name__ == "__main__":
    main()

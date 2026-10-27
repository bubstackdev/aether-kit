#!/usr/bin/env python3
"""Compose the Qwen3-VL image + text evidence record (development tooling only; standard library only).

    python3 Forge/language/record_vl_evidence.py --bundle <bundle> --reference <dir> [--reference <dir> ...] \
        --photos <photo dir with provenance.json> [--decoded-photos <Kit-decoded PNG dir with provenance.json>] \
        --native NAME=<report.json> [...] --output docs/evidence/<name>.json

Copies each native report next to the record (`<name>-<NAME>.json`) and binds the references, the photos (with their
licences), the bundle registration and the exports' self-checks by SHA-256. A report marked `"gated": false` (Kit
preprocessing on an independent decode) is recorded and summarized but does not decide `passed`. The model's licence and
base model come from the reference's `source` when it records them.
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


def reference_summary(directory: Path) -> dict:
    r = json.loads((directory / "reference.json").read_text())
    return {"recordSHA256": sha256(directory / "reference.json"), "oracle": r["oracle"], "versions": r["versions"],
            "maximumPixels": r.get("maximumPixels"), "imageGrids": r.get("imageGrids"),
            "gridSnapping": r.get("gridSnapping"),
            "cases": [{"name": c["name"], "question": c["question"], "image": c["image"], "grid": c["grid"],
                       "imageTokens": c["imageTokens"], "promptTokens": len(c["promptTokens"]),
                       "greedyText": c["greedyText"], "eagerPortVsReference": c["eagerPortVsReference"],
                       "pixelsSHA256": c["pixels"]["sha256"], "forcedLogitsSHA256": c["forcedLogits"]["sha256"]}
                      for c in r["cases"]]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--photos", type=Path, required=True)
    parser.add_argument("--decoded-photos", type=Path)
    parser.add_argument("--native", action="append", required=True, metavar="NAME=REPORT")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    b = args.bundle
    manifest = json.loads((b / "language-model.json").read_text())
    text_export = json.loads((b / "export-report.json").read_text())
    vision_export = json.loads((b / "vision-export-report.json").read_text())
    natives = {}
    for spec in args.native:
        name, path = spec.split("=", 1)
        report = json.loads(Path(path).read_text())
        copied = args.output.with_name(f"{args.output.stem}-{name}.json")
        copied.write_text(Path(path).read_text())
        natives[name] = {"file": copied.name, "sha256": sha256(copied), "passed": report["passed"],
                         "gated": report.get("gated", True), "inputs": report.get("inputs"),
                         "gate": report.get("gate"), "cases": [{k: v for k, v in c.items() if not k.endswith("Tokens")
                                                               or k in ("promptTokens", "imageTokens", "upstreamTokens",
                                                                        "upstreamAgreementTokens")}
                                                              for c in report["cases"]]}
        for key in ("memory", "tokenizerFilesVerified", "tokenizerRendersReferencePrompts", "closedAfterUnload"):
            if key in report:
                natives[name][key] = report[key]
    here = Path(__file__).resolve().parent
    source = json.loads((args.reference[0] / "reference.json").read_text()).get("source", {})
    model = {"repository": manifest["sourceRepository"], "revision": manifest["sourceRevision"],
             "license": source.get("license", "apache-2.0"), "gated": False}
    if manifest.get("baseModel") or source.get("baseModel"):
        model["baseModel"] = manifest.get("baseModel") or source.get("baseModel")
    record = {
        "schema": 2,
        "kind": "qwen3-vl-image-text-qualification",
        "passed": all(n["passed"] for n in natives.values() if n["gated"]),
        "model": model,
        "bundle": {"directory": str(b), "manifestSHA256": sha256(b / "language-model.json"), "manifest": manifest,
                   "textExport": {"recipe": text_export["recipe"], "versions": text_export["versions"],
                                  "selfCheck": text_export["selfCheck"], "modelFiles": text_export["modelFiles"]},
                   "visionExport": {"recipe": vision_export["recipe"], "imageGrids": vision_export.get("imageGrids"),
                                    "selfCheck": vision_export["selfCheck"], "modelFiles": vision_export["modelFiles"]}},
        "photos": json.loads((args.photos / "provenance.json").read_text()),
        **({"decodedPhotos": json.loads((args.decoded_photos / "provenance.json").read_text())} if args.decoded_photos else {}),
        "references": [reference_summary(r) for r in args.reference],
        "native": natives,
        "scriptSHA256": {p.name: sha256(p) for p in sorted(here.glob("*.py"))},
        "nativeTests": ["Tests/AetherNativeTests/Qwen3VLImageTextQualificationTests.swift",
                        "Tests/AetherNativeTests/VisionLanguageSessionTests.swift"],
    }
    args.output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"passed": record["passed"], "native": {k: {"passed": v["passed"], "gated": v["gated"]}
                                                              for k, v in natives.items()}}))


if __name__ == "__main__":
    main()

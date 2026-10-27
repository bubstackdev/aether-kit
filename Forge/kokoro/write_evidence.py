#!/usr/bin/env python3
"""Summarizes a Kokoro export, its references and a native report into docs/evidence records."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--references", type=Path, required=True)
    ap.add_argument("--evidence", type=Path, required=True, help="docs/evidence directory")
    ap.add_argument("--date", required=True)
    args = ap.parse_args()
    contract = json.loads((args.bundle / "kokoro.json").read_text())
    export = json.loads((args.bundle / "export-report.json").read_text())
    index = json.loads((args.references / "references.json").read_text())
    keys = ["name", "voice", "speed", "seed", "phonemes", "input_ids", "frames", "samples", "seconds", "audio_sha256",
            "perturbed_sha256", "frames_match_export_graph", "upstream_cpu_seconds", "sensitivity_floor",
            "float64_upstream", "eager_export_graph"]
    cases = []
    for entry in index["cases"]:
        record = json.loads((args.references / entry["name"] / "case.json").read_text())
        cases.append({k: record[k] for k in keys})
    references = {**index, "location": f"~/.cache/huggingface/aether-kit/kokoro-82m/{args.references.name}",
                  "references_json_sha256": sha(args.references / "references.json"), "case_records": cases}
    (args.evidence / f"kokoro-references-{args.date}.json").write_text(json.dumps(references, indent=2, ensure_ascii=False) + "\n")
    bundle = {"location": f"~/.cache/huggingface/aether-kit/kokoro-82m/{args.bundle.name}",
              "kokoro_json_sha256": sha(args.bundle / "kokoro.json"), "export_report": export,
              "models": contract["models"], "precision": contract["precision"], "maximumFrames": contract["maximumFrames"],
              "source": contract["source"],
              "voices": [{k: v[k] for k in ("id", "language", "sha256", "source_sha256")} for v in contract["voices"]]}
    (args.evidence / f"kokoro-export-{args.date}.json").write_text(json.dumps(bundle, indent=2) + "\n")
    print(bundle["kokoro_json_sha256"], references["references_json_sha256"])


if __name__ == "__main__":
    main()

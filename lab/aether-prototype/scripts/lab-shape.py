#!/usr/bin/env python3
"""Translate retained lab export provenance into an Aether registration.

Writes only a small JSON file; never exports, copies or alters model weights.
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("krea_lab", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
rows = json.loads((args.krea_lab / "records/round4/fixed-assets.json").read_text())
assert [row["segment"] for row in rows] == [0, 1, 2, 3]
spec = dict(image_tokens=4032, text_rows=20,
    source_sha256=[row["original"]["sourceBytecodeSHA256"] for row in rows],
    specialized_sha256=[row["fixed"]["sourceBytecodeSHA256"] for row in rows],
    segments=[str((args.krea_lab / "forge/fixed-first" / Path(row["fixed"]["path"]).name).resolve()) for row in rows])
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(spec, indent=2) + "\n")
print(args.output)

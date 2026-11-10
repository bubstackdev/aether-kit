"""Record the memory gate's measured envelopes into an H3 bundle's manifest (standard library only).

    python3 -I Forge/h3/record_envelopes_h3.py --bundle <export>/bundle --renders <gate records> [--dry-run]

Reads every per-render record the H2 memory gate wrote (`scripts/h3-memory-gate`, schema
`aether-h3-memory-render/1`), groups them by admission key (`<profile>` for text to video, `<profile>/fl2va1` for one
keyframe, `<profile>/fl2va2` for first + last) and writes one envelope per key into `h3.envelopes` (MANIFEST.md
§5.10): each phase's largest peak and held bytes over the key's renders, the recipes measured, the Core AI
architecture, the export the renders measured (`Manifest.measurementDigest`: the manifest without its envelopes, so
recording them doesn't change it) and the sampling margin. The bar and the physical memory a profile needs stay in its
`memory` block.

Refuses, writing nothing: records of different exports or architectures, a record for an export other than this
manifest's (checked by the profile list), a key with any failed render (no envelope is better than an optimistic
one), a phase peak plus the margin above the profile's bar, and a manifest file that is hard-linked (an extended
export's manifest is always a fresh copy; the source export's is never written). Updates the manifest's line in
SHA256SUMS. Existing envelopes for the same key and architecture are replaced; others are kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

SCHEMA = "aether-h3-memory-render/1"
PHASES = ("perPrompt", "perStep", "perImage")
TASKS = {"t2va": "", "fl2va1": "/fl2va1", "fl2va2": "/fl2va2"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(directory: Path) -> list[dict]:
    records = []
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("schema") == SCHEMA:
            record["_file"] = path.name
            records.append(record)
    return records


def envelopes(manifest: dict, records: list[dict]) -> tuple[list[dict], list[str]]:
    """The envelopes `records` measure for `manifest`, and the keys left out (with why)."""
    if not records:
        raise SystemExit("no render records")
    exports = {r["export"] for r in records}
    architectures = {r["architecture"] for r in records}
    if len(exports) != 1 or len(architectures) != 1:
        raise SystemExit(f"the records measured {len(exports)} exports on {len(architectures)} architectures; record one at a time")
    profiles = {p["key"]: p for p in manifest["h3"]["profiles"]}
    groups: dict[str, list[dict]] = {}
    for r in records:
        render = r["render"]
        if render["profile"] not in profiles:
            raise SystemExit(f"{r['_file']}: profile {render['profile']} is not in this manifest (another export?)")
        key = render["profile"] + TASKS[render["task"]]
        if r["admissionKey"] != key:
            raise SystemExit(f"{r['_file']}: admission key {r['admissionKey']}, expected {key}")
        groups.setdefault(key, []).append(r)
    out, skipped = [], []
    for key in sorted(groups):
        rs = groups[key]
        failed = [r["_file"] for r in rs if not r.get("passed")]
        if failed:
            skipped.append(f"{key}: failed render(s) {', '.join(failed)}")
            continue
        profile = profiles[key.split("/")[0]]
        memory = profile.get("memory")
        if memory is None:
            raise SystemExit(f"profile {profile['key']} has no memory bar; extend the manifest first (extend_h3.py)")
        margin = max(r["marginBytes"] for r in rs)
        phases = {}
        for phase in PHASES:
            measured = [r["phases"][phase] for r in rs if phase in r["phases"]]
            if len(measured) != len(rs):
                raise SystemExit(f"{key}: a render didn't measure {phase}")
            phases[phase] = {"peakBytes": max(m["peakBytes"] for m in measured),
                             "heldBytes": max(m["heldBytes"] for m in measured)}
            if phases[phase]["peakBytes"] + margin > memory["barBytes"]:
                raise SystemExit(f"{key}: {phase} peaks at {phases[phase]['peakBytes']} B, over the bar with the margin")
        recipes = sorted({r["render"]["recipe"] for r in rs})
        out.append({"key": key, "recipes": recipes, "architecture": rs[0]["architecture"], "export": rs[0]["export"],
                    "marginBytes": margin, **phases, "renders": len(rs),
                    "record": ", ".join(sorted(r["_file"] for r in rs))})
    return out, skipped


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--renders", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    path = args.bundle / ".aether/aether.json"
    manifest = json.loads(path.read_text())
    found, skipped = envelopes(manifest, load_records(args.renders))
    replaced = {(e["key"], e["architecture"]) for e in found}
    kept = [e for e in manifest["h3"].get("envelopes", []) if (e["key"], e["architecture"]) not in replaced]
    manifest["h3"]["envelopes"] = sorted(kept + found, key=lambda e: (e["key"], e["architecture"]))
    print(json.dumps({"recorded": [e["key"] for e in found], "kept": [e["key"] for e in kept], "skipped": skipped}, indent=1))
    if args.dry_run:
        return
    if path.stat().st_nlink > 1:
        raise SystemExit(f"{path} is hard-linked; refusing to write it (record into an extended export's fresh manifest)")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)
    sums = args.bundle / "SHA256SUMS"
    if sums.exists():
        if sums.stat().st_nlink > 1:
            raise SystemExit(f"{sums} is hard-linked; the manifest was written but its checksum line was not")
        digest = sha256_file(path)
        lines = [f"{digest}  .aether/aether.json" if line.endswith("  .aether/aether.json") else line
                 for line in sums.read_text().splitlines()]
        temporary = sums.with_name(sums.name + ".tmp")
        temporary.write_text("\n".join(lines) + "\n")
        os.replace(temporary, sums)


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()

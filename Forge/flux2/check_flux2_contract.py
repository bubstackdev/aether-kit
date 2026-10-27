#!/usr/bin/env python3
"""Diff the execution contract of two finalized FLUX.2 bundles (development tooling only).

    Forge/.venv/bin/python -I Forge/flux2/check_flux2_contract.py <reference bundle> <candidate bundle> [--json <path>]

Reads declarations only (model bytecode signatures, not weights; nothing is loaded or run). Must match exactly:
- `metadata.json`, and `aether.json` except the fields a re-export legitimately changes (`provenance.files`,
  `provenance.exportedOn`, `provenance.forge`, `diffusion.assetBytes`, `memory.weightsBytes`), reported side by side;
- per asset: the public function names and, for each function, the graph type (input/output dtypes, ranks, static
  sizes and dynamic axes), the argument and result attributes (names, enumerated-shape specializations if any), the
  OS compatibility version, and the sets of storage and compute scalar types;
- the tokenizer and LICENSE files, byte for byte, with no extra tokenizer file in the candidate.

Reported, not gated: per-asset bytes, storage element counts by type, private composite graphs (attention,
pixel shuffle) and the operation histogram, which differ between implementations of the same contract.
Exits 1 on any gated difference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

VOLATILE = {("provenance", "files"), ("provenance", "exportedOn"), ("provenance", "forge"),
            ("diffusion", "assetBytes"), ("memory", "weightsBytes")}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(asset: Path) -> dict:
    from coreai.authoring.asset import AIModelAsset

    loaded = AIModelAsset.load(asset)
    summary = loaded.summary(include_statistics=True)
    program = loaded.program
    module = program._mlir_module            # noqa: SLF001 (read-only walk of the public graphs)
    public, private = {}, []
    with module.context:
        def walk(region) -> None:
            for op in region.blocks[0].operations:
                name = op.operation.name
                if name == "udml.namespace":
                    walk(op.regions[0])
                if name != "coreai.graph":
                    continue
                attrs = {a.name: str(a.attr) for a in op.operation.attributes}
                entry = {"type": str(op.type), "argAttrs": attrs.get("arg_attrs"), "resAttrs": attrs.get("res_attrs"),
                         "other": {k: v for k, v in attrs.items()
                                   if k not in ("sym_name", "function_type", "arg_attrs", "res_attrs", "param_decls",
                                                "composite_decl", "priv", "no_inline")}}
                symbol = attrs["sym_name"].strip('"')
                if "priv" in attrs:
                    kind = attrs.get("composite_decl", "").split('"')[1] if '"' in attrs.get("composite_decl", "") else "?"
                    private.append({"composite": kind, "type": entry["type"]})
                else:
                    public[symbol] = entry
        walk(module.body.region)
    return {
        "functions": public,
        "summaryFunctions": {f: {"inputs": summary.function_inputs(f), "outputs": summary.function_outputs(f)}
                             for f in summary.function_names},
        "compatibility": summary._compatibility_version,                 # noqa: SLF001
        "storageTypes": sorted(t for t, _ in summary.storage_types),
        "computeTypes": sorted(summary.compute_types),
        "storageCounts": dict(summary.storage_types),
        "composites": sorted(private, key=lambda p: (p["composite"], p["type"])),
        "operations": dict(summary.operation_distribution),
        "bytes": sum(p.stat().st_size for p in asset.rglob("*") if p.is_file()),
    }


def flatten(value, prefix=()):
    if isinstance(value, dict):
        for key, inner in value.items():
            yield from flatten(inner, prefix + (key,))
    else:
        yield prefix, value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--json", type=Path, help="write the full comparison here")
    args = parser.parse_args()
    ref, new = args.reference, args.candidate
    differences: list[str] = []
    lines: list[str] = []
    record: dict = {"reference": str(ref), "candidate": str(new)}

    meta = [json.loads((b / "metadata.json").read_text()) for b in (ref, new)]
    record["metadata.json"] = "identical" if meta[0] == meta[1] else {"reference": meta[0], "candidate": meta[1]}
    if meta[0] != meta[1]:
        differences.append("metadata.json differs")
    lines.append(f"metadata.json: {'identical' if meta[0] == meta[1] else 'DIFFERS'}")

    manifests = [json.loads((b / "aether.json").read_text()) for b in (ref, new)]
    flat = [dict(flatten(m)) for m in manifests]
    gated = sorted(k for k in set(flat[0]) | set(flat[1]) if not any(k[:len(v)] == v for v in VOLATILE))
    manifest_diffs = [".".join(k) for k in gated if flat[0].get(k, "<absent>") != flat[1].get(k, "<absent>")]
    differences += [f"aether.json {k}" for k in manifest_diffs]
    record["aether.json"] = {"gatedFields": len(gated), "differences": manifest_diffs, "volatile": {}}
    lines.append(f"aether.json: {len(gated)} gated fields, {len(manifest_diffs)} differ"
                 + (f" ({', '.join(manifest_diffs)})" if manifest_diffs else ""))
    for path in sorted(VOLATILE):
        values = []
        for m in manifests:
            v = m
            for key in path:
                v = v.get(key) if isinstance(v, dict) else None
            values.append(v)
        if path == ("provenance", "files"):
            values = [len(v or {}) for v in values]
            label = "files pinned"
        else:
            label = ".".join(path)
        record["aether.json"]["volatile"][".".join(path)] = {"reference": values[0], "candidate": values[1]}
        lines.append(f"  volatile {label}: {json.dumps(values[0], sort_keys=True)} -> {json.dumps(values[1], sort_keys=True)}")

    record["assets"] = {}
    for role, relative in sorted(meta[0]["assets"].items()):
        other = meta[1]["assets"].get(role)
        if other is None:
            differences.append(f"{role}: missing")
            continue
        a, b = describe(ref / relative), describe(new / other)
        row = {"reference": a, "candidate": b, "differences": []}
        for key in ("functions", "summaryFunctions", "compatibility", "storageTypes", "computeTypes"):
            if json.dumps(a[key], sort_keys=True) != json.dumps(b[key], sort_keys=True):
                row["differences"].append(key)
        differences += [f"{role}: {d}" for d in row["differences"]]
        record["assets"][role] = row
        lines.append(f"{role} ({relative}): {'IDENTICAL contract' if not row['differences'] else 'DIFFERS ' + str(row['differences'])}")
        for name, fn in b["functions"].items():
            lines.append(f"  {name}: {fn['type']}")
            lines.append(f"    args {fn['argAttrs']} -> {fn['resAttrs']}")
        lines.append(f"  compat {b['compatibility']}; storage {b['storageTypes']}; compute {b['computeTypes']}")
        counts = sorted(set(a["storageCounts"]) | set(b["storageCounts"]))
        lines.append("  storage elements " + ", ".join(
            f"{t} {a['storageCounts'].get(t, 0):,} -> {b['storageCounts'].get(t, 0):,}" for t in counts
            if t in ("Int8", "Float16", "Float32")))
        lines.append(f"  bytes {a['bytes']:,} -> {b['bytes']:,}")
        composites = [sorted({c['composite'] + ' ' + c['type'] for c in x['composites']}) for x in (a, b)]
        if composites[0] != composites[1]:
            lines.append(f"  composites (not gated) {composites[0]} -> {composites[1]}")

    record["files"] = {}
    listed = [sorted(str(p.relative_to(b)) for p in (b / "tokenizer").rglob("*") if p.is_file()) for b in (ref, new)]
    for name in listed[0] + ["LICENSE.md"]:
        same = (new / name).is_file() and sha256(ref / name) == sha256(new / name)
        record["files"][name] = same
        if not same:
            differences.append(f"{name} differs")
    extra = sorted(set(listed[1]) - set(listed[0]))
    record["extraFiles"] = extra
    differences += [f"{name} only in the candidate" for name in extra]
    lines.append(f"tokenizer + LICENSE.md: {sum(record['files'].values())}/{len(record['files'])} files identical"
                 + (f", {len(extra)} extra in the candidate: {extra}" if extra else ", no extra files"))

    record["differences"] = differences
    record["identical"] = not differences
    lines.append("RESULT: " + ("contract identical" if not differences else f"{len(differences)} differences: {differences}"))
    if args.json:
        args.json.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")
    print("\n".join(lines))
    sys.exit(0 if not differences else 1)


if __name__ == "__main__":
    main()

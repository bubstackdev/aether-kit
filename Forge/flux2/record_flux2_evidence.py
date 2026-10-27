#!/usr/bin/env python3
"""Write the evidence record of a Forge FLUX.2 export (development tooling only).

    python3 Forge/flux2/record_flux2_evidence.py --export <export dir> --finalize <finalizer stdout .json> \
        --contract <check_flux2_contract.py --json output> \
        --parity <new parity report.json> [...] --shipped-parity <shipped bundle's report.json> [...] \
        --lifecycle <new lifecycle report.json> [--shipped-lifecycle <report.json>] \
        --command "<export command>" --output docs/evidence/<name>.json

Parity reports are the `flux2NativeParity` JSON documents; several per bundle are merged by (case, check), so a
small-size run and a large-size run combine. Every check and recorded metric appears side by side, new against
shipped, with its bar and the per-check difference. Standard library only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def merged(paths: list[Path]) -> tuple[dict, list[dict]]:
    rows, reports = {}, []
    for path in paths:
        report = json.loads(path.read_text())
        reports.append({"file": str(path), "kind": report.get("kind"), "passed": report.get("passed"),
                        "checkCount": report.get("checkCount"), "fixtures": report.get("fixtures"),
                        "bundle": report.get("bundle"), "elapsedSeconds": report.get("elapsedSeconds"),
                        "phases": report.get("phases")})
        for item in report.get("checks", []):
            rows[(item["case"], item["name"])] = {"value": item["value"], "bar": item.get("bar"), "passed": item["passed"]}
        for item in report.get("records", []):
            rows[(item["case"], item["name"])] = {"value": item["value"], "bar": None, "passed": None}
    return rows, reports


def lifecycle(path: Path | None) -> dict | None:
    if path is None:
        return None
    report = json.loads(path.read_text())
    return {"file": str(path), "passed": report.get("passed"), "checkCount": report.get("checkCount"),
            "passedChecks": sum(1 for c in report.get("checks", []) if c.get("passed")),
            "nativeCalls": report.get("nativeCalls"), "modelInstances": report.get("modelInstances"),
            "functionLoads": report.get("functionLoads"), "functionReuses": report.get("functionReuses"),
            "elapsedSeconds": report.get("elapsedSeconds"), "bundle": report.get("bundle")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--finalize", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--parity", type=Path, nargs="+", required=True)
    parser.add_argument("--shipped-parity", type=Path, nargs="+", required=True)
    parser.add_argument("--lifecycle", type=Path)
    parser.add_argument("--shipped-lifecycle", type=Path)
    parser.add_argument("--command", action="append", default=[])
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads((args.export / "export-report.json").read_text())
    finalize = json.loads(args.finalize.read_text())
    contract = json.loads(args.contract.read_text())
    new, new_reports = merged(args.parity)
    old, old_reports = merged(args.shipped_parity)
    table = []
    for case, name in sorted(set(new) | set(old)):
        a, b = old.get((case, name)), new.get((case, name))
        row = {"case": case, "check": name, "bar": (b or a or {}).get("bar"),
               "shipped": None if a is None else a["value"], "new": None if b is None else b["value"],
               "newPassed": None if b is None else b["passed"]}
        if isinstance(row["shipped"], (int, float)) and isinstance(row["new"], (int, float)):
            row["delta"] = row["new"] - row["shipped"]
        table.append(row)
    gated = [r for r in table if r["bar"] is not None and r["new"] is not None]
    minima = {}
    for r in gated:
        family = r["check"].split("[")[0]
        for side in ("shipped", "new"):
            if isinstance(r[side], (int, float)):
                minima.setdefault(family, {}).setdefault(side, r[side])
                minima[family][side] = min(minima[family][side], r[side])
    record = {
        "schema": 1,
        "kind": "flux2-forge-export",
        "date": report["exportedAt"][:10],
        "commands": args.command,
        "notes": args.note,
        "export": {key: report.get(key) for key in ("recipe", "host", "versions", "scriptSHA256", "source",
                                                     "quantization", "conversion", "selfCheck", "stages", "peakRSSGB",
                                                     "bundleBytes", "command")},
        "bundle": {"root": finalize["bundle"], "bytes": finalize["bytes"], "files": finalize["files"],
                   "aether.json": finalize["aether.json"], "metadata.json": finalize["metadata.json"]},
        "contract": {"reference": contract["reference"], "identical": contract["identical"],
                     "differences": contract["differences"], "aetherJSON": contract["aether.json"],
                     "metadataJSON": contract["metadata.json"], "files": contract["files"],
                     "assets": {role: {"differences": row["differences"],
                                       "functions": row["candidate"]["functions"],
                                       "storageCounts": {"shipped": row["reference"]["storageCounts"],
                                                         "new": row["candidate"]["storageCounts"]},
                                       "bytes": {"shipped": row["reference"]["bytes"], "new": row["candidate"]["bytes"]}}
                                for role, row in contract["assets"].items()}},
        "parity": {"new": new_reports, "shipped": old_reports,
                   "passed": all(r["newPassed"] for r in gated),
                   "gatedChecks": len(gated), "minima": minima, "table": table},
        "lifecycle": {"new": lifecycle(args.lifecycle), "shipped": lifecycle(args.shipped_lifecycle)},
    }
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"contractIdentical": contract["identical"], "parityPassed": record["parity"]["passed"],
                      "gatedChecks": len(gated), "minima": minima,
                      "lifecycle": record["lifecycle"]["new"] and {k: record["lifecycle"]["new"][k]
                                                                    for k in ("passed", "passedChecks", "checkCount")}},
                     indent=1))


if __name__ == "__main__":
    main()

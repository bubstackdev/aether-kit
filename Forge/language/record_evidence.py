#!/usr/bin/env python3
"""Compose the language decode-contract evidence record (development tooling only; standard library only).

    python3 Forge/language/record_evidence.py --native <native report .json> --reference <reference dir> \
        --bucketed <bucketed export dir> [--running <running export dir>] [--chat <native chat report .json>] \
        --output docs/evidence/<name>.json
A model qualified on the bucketed contract alone omits `--running` (no before/after comparison). `--chat` binds the
native chat check (LanguageChatQualificationTests: tokenizer render, bundle open, memory) the same way.

Copies the native report next to the record (`<name>-native.json`) and binds everything by SHA-256: the native
report, the reference record and logits, each export report and model file, and this directory's scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def timing(runs: list[dict]) -> dict:
    tps = [(r["generatedTokens"] - 1) / (r["totalSeconds"] - r["timeToFirstTokenSeconds"]) for r in runs]
    return {"promptTokens": runs[0]["promptTokens"], "generatedTokens": runs[0]["generatedTokens"], "runs": len(runs),
            "timeToFirstTokenSecondsMedian": round(statistics.median(r["timeToFirstTokenSeconds"] for r in runs), 4),
            "decodeTokensPerSecondMedian": round(statistics.median(tps), 1),
            "decodeTokensPerSecond": [round(x, 1) for x in tps]}


def summary(report: dict, export_dir: Path) -> dict:
    export = json.loads((export_dir / "export-report.json").read_text())
    return {
        "contract": report["contract"],
        "export": {"directory": str(export_dir), "reportSHA256": sha256(export_dir / "export-report.json"),
                   "modelFiles": export["modelFiles"], "versions": export["versions"],
                   "exportBounds": export.get("exportBounds"),
                   "cpuOnlySelfCheck": {"passed": export["selfCheck"]["passed"],
                                        "minCosine": min(s["cosine"] for s in export["selfCheck"]["steps"]),
                                        "lostCacheRows": sum(s["lostCacheRows"] for s in export["selfCheck"]["steps"])}},
        "teacherForced": {"rows": len(report["forcedRows"]), "minimumCosine": round(report["forcedMinimumCosine"], 7),
                          "maximumAbs": round(report["forcedMaximumAbs"], 5),
                          "argmaxMismatches": report["forcedArgmaxMismatches"], "nearTies": report["forcedNearTies"]},
        "greedy": {"tokens": len(report["greedyTokens"]), "firstDivergence": report.get("greedyFirstDivergence"),
                   "divergenceReferenceMargin": (report.get("greedyDivergenceRow") or {}).get("referenceMargin"),
                   "divergenceNativeRankInReference": (report.get("greedyDivergenceRow") or {}).get("nativeRankInReference")},
        "longRun": {"decodeSteps": report["longRunSteps"], "modelReloads": report["longRunModelReloads"],
                    "distinctStepShapes": report["longRunDistinctStepShapes"],
                    "allocatedCapacity": report["longRunAllocatedCapacity"],
                    "physicalFootprintMiB": [[s["generated"], s["physicalFootprintBytes"] >> 20]
                                             for s in report["longRunFootprint"]]},
    }


def tps(x: dict) -> float:
    return (x["generatedTokens"] - 1) / (x["totalSeconds"] - x["timeToFirstTokenSeconds"])


def ratios(b: list[dict], r: list[dict]) -> dict:
    """Per-round bucketed/baseline ratios (same round, adjacent in time) and their medians."""
    decode = [round(tps(x) / tps(y), 3) for x, y in zip(b, r)]
    ttft = [round(x["timeToFirstTokenSeconds"] / y["timeToFirstTokenSeconds"], 3) for x, y in zip(b, r)]
    return {"decodeSpeedupPerRound": decode, "decodeSpeedupMedian": round(statistics.median(decode), 3),
            "timeToFirstTokenRatioPerRound": ttft, "timeToFirstTokenRatioMedian": round(statistics.median(ttft), 3)}


def paired(native: dict) -> dict:
    """Medians per session and prompt, and per-round bucketed ratios against the running baseline with its default
    recycling (what 0.0.1 consumers get, reload costs included) and with recycling disabled (per-step cost)."""
    t = native["pairedTimings"]
    out = {"rounds": len(native["pairedLoadAverages"]), "loadAverage1m": native["pairedLoadAverages"],
           "method": "all sessions resident; each round runs every session on every prompt (256 tokens, greedy), "
                     "rotating which goes first; one warm-up request each beforehand"}
    for prompt in ("short", "long"):
        b = t[f"bucketed-{prompt}"]
        out[prompt] = {"bucketed": timing(b), "running": timing(t[f"running-{prompt}"]),
                       "runningNoRecycle": timing(t[f"runningNoRecycle-{prompt}"]),
                       "vsRunning": ratios(b, t[f"running-{prompt}"]),
                       "vsRunningNoRecycle": ratios(b, t[f"runningNoRecycle-{prompt}"])}
    return out


def long_context(native: dict) -> dict | None:
    probe = native.get("longContext") or {}
    if not probe:
        return None
    out = {}
    for name, r in probe.items():
        out[name] = {"promptTokens": r["promptTokens"], "generatedTokens": r["timing"]["generatedTokens"],
                     "timeToFirstTokenSeconds": round(r["timing"]["timeToFirstTokenSeconds"], 3),
                     "decodeTokensPerSecond": round(tps(r["timing"]), 1), "allocatedCapacity": r["allocatedCapacity"],
                     "kvCacheMiB": r["allocatedBytes"] >> 20, "modelReloads": r["modelReloads"],
                     "distinctStepShapes": r["distinctStepShapes"],
                     "physicalFootprintMiB": [[s["generated"], s["physicalFootprintBytes"] >> 20] for s in r["footprint"]]}
    return out


def chat(report_path: Path, output: Path) -> dict:
    """The native chat check's memory and timings, with its report copied next to the record."""
    report = json.loads(report_path.read_text())
    copied = output.with_name(output.stem + "-chat.json")
    shutil.copyfile(report_path, copied)
    keys = ("openSeconds", "unloadSeconds", "idleFootprintMiB", "peakFootprintMiB", "afterUnloadFootprintMiB",
            "idleMetalMiB", "peakMetalMiB", "afterUnloadMetalMiB", "greedyRepeatIdentical")
    return {**{k: report.get(k) for k in keys},
            "runs": [{k: r[k] for k in ("name", "promptTokens", "replyTokens", "stopReason", "timeToFirstTokenSeconds",
                                        "totalSeconds", "text")} for r in report["runs"]],
            "report": {"file": copied.name, "sha256": sha256(copied)}}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("native", "reference", "bucketed", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--running", type=Path)
    parser.add_argument("--chat", type=Path)
    args = parser.parse_args()
    native = json.loads(args.native.read_text())
    reference = json.loads((args.reference / "reference.json").read_text())
    copied = args.output.with_name(args.output.stem + "-native.json")
    shutil.copyfile(args.native, copied)
    here = Path(__file__).resolve().parent
    record = {
        "schema": 1,
        "kind": "language-decode-contract-qualification",
        "passed": native["passed"],
        "device": native["device"],
        "hostLoad": native["hostLoad"],
        "model": {"repository": reference["source"]["repository"], "revision": reference["source"]["revision"],
                  "license": reference["source"]["license"], "precision": "float16 weights and caches",
                  "checkpointSHA256": reference["checkpointSHA256"]},
        "reference": {"oracle": reference["oracle"], "versions": reference["versions"],
                      "promptTokens": len(reference["promptTokens"]), "forcedRows": reference["forcedLogits"]["rows"],
                      "recordSHA256": sha256(args.reference / "reference.json"),
                      "logitsSHA256": reference["forcedLogits"]["sha256"],
                      "eagerPortVsReference": reference["eagerPortVsReference"]},
        "tolerance": native["tolerance"],
        "stopTokenNote": native["stopTokenNote"],
        "bucketed": summary(native["bucketed"], args.bucketed),
        "running": summary(native["running"], args.running) if args.running else None,
        "longRunTokensIdenticalPrefix": native.get("bucketedLongRunMatchesRunning") if args.running else None,
        "pairedTiming": paired(native) if args.running else None,
        "chat": chat(args.chat, args.output) if args.chat else None,
        "longContext": long_context(native),
        "nativeReport": {"file": copied.name, "sha256": sha256(copied)},
        "scriptSHA256": {p.name: sha256(p) for p in sorted(here.glob("*.py"))},
        "nativeTest": "Tests/AetherNativeTests/LanguageDecodeQualificationTests.swift (AETHER_NATIVE_LANGUAGE_QUALIFICATION=1, release)",
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("passed", "device")}, indent=1))


if __name__ == "__main__":
    main()

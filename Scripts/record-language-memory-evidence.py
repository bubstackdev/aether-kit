"""Assembles the memory-budget, prefill-width and Core AI reshape-retention evidence from the native reports.

Usage: python3 Scripts/record-language-memory-evidence.py <requalify dir> <retention dir> <chunk sweep .json> [<bounds dir>]

<requalify dir>: the re-qualification at the fixed prefill width and the default budget: upstream parity reports
(<label>.json from languageBucketedDecodeQualification, qwen3VLImageTextQualification and
visionLanguageSessionReproducesQualifiedRun), overlap.json from languageOverlappedPrefillMatchesOtherPlans, and
<label>-divergence-margin.json for a VL greedy divergence past the reference's forced rows. <bounds dir> (default the
requalify dir): bound-qwen3-0.6b.json, bound-writer.json, bound-vl-2b.json, bound-vl-4b.json from
languageMemoryBudgetBoundsRepeatedRequests. <retention dir>: retention-reshapes-1.json and retention-reshapes-0.json
from coreAIRetainsMemoryPerReshapeUntilRelease. The chunk sweep: languagePrefillChunkSweep on the writer.
"""
import hashlib, json, pathlib, sys
kit = pathlib.Path(__file__).resolve().parent.parent
requalify, retention_runs, sweep_file = (pathlib.Path(a) for a in sys.argv[1:4])
bounds = pathlib.Path(sys.argv[4]) if len(sys.argv) > 4 else requalify
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(pathlib.Path(p).read_text())
host = "Shared host: GPU jobs of cooperating agents are serialized by a lock, other host work is not; memory is Metal's currentAllocatedSize in the test process"
sources = {p: sha(kit / p) for p in ["Sources/AetherLanguageCoreAI/LanguageGenerationSession.swift",
    "Sources/AetherLanguageCoreAI/LanguageMemoryBudget.swift", "Sources/AetherLanguageCoreAI/LanguageRequestAdmission.swift",
    "Sources/AetherLanguageCoreAI/LanguageStepContract.swift", "Sources/AetherLanguageCoreAI/LanguageVisionPrompt.swift",
    "Sources/AetherLanguageCoreAI/VisionLanguageSession.swift"]}
sourceNote = ("sourceSHA256 are the committed sources; the native runs used the same sources except a doc-comment wording "
              "fix in LanguageVisionPrompt.swift (power-of-two chunks to the fixed width), made after the runs")
def tests(*names): return {f"Tests/AetherNativeTests/{n}.swift": sha(kit / f"Tests/AetherNativeTests/{n}.swift") for n in names}
bundles = {"qwen3-0.6b": "qwen3-0.6b-bucketed-fp16/model.aimodel (raw Forge export, Qwen/Qwen3-0.6B)",
           "writer": "qwen3-4b-instruct-2507-abliterated-bucketed-fp16 (huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated)",
           "vl-2b": "qwen3-vl-2b-bundle-macos (Qwen/Qwen3-VL-2B-Instruct)",
           "vl-4b": "qwen3-vl-4b-abliterated-bundle-macos (huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated)"}

# The 60-request bound at the default budget.
models, reports = [], {}
for name in bundles:
    f = bounds / f"bound-{name}.json"
    r = load(f)
    assert r.get("error") is None, name
    reports[name] = {"file": f.name, "sha256": sha(f)}
    for m in r["models"]:
        models.append({k: m[k] for k in ["name", "promptTokens", "replies", "warmMetalMiB", "maximumMetalMiB", "maximumRequestGrowthMiB",
                                          "boundMiB", "reloads", "idleReloads", "requestReloads", "mismatchedReplies",
                                          "unloadedMetalMiB", "seconds", "passed"]}
                      | {"requests": r["requests"], "metalMiBAfterEachRequest": [s["metalMiB"] for s in m["samples"]],
                         "reloadsAfterEachRequest": [s["reloads"] for s in m["samples"]]})
budget = {"schema": 2, "kind": "language-memory-budget-qualification", "passed": all(m["passed"] for m in models),
          "date": "2026-10-27", "device": r["device"], "hostLoad": host, "budgetMiB": r["budgetMiB"], "gate": r["gate"],
          "prefillWidth": "bucketed prefill at the default fixed width (64)", "bundles": bundles,
          "models": models, "nativeReports": reports, "sourceSHA256": sources, "sourceNote": sourceNote,
          "nativeTest": "Tests/AetherNativeTests/LanguageMemoryBudgetQualificationTests.swift (AETHER_NATIVE_LANGUAGE_MEMORY_BUDGET=1, release)",
          "testSHA256": tests("LanguageMemoryBudgetQualificationTests")}

# The prefill width: the sweep that chose it, its exactness on every model, and upstream parity at it.
sweep = load(sweep_file)
overlap_file = requalify / "overlap.json"
overlap = load(overlap_file)
assert overlap.get("error") is None
parity = {}
for f in sorted(requalify.glob("*.json")):
    if f.name == "overlap.json" or f.name.startswith("bound-") or f.name.endswith("-divergence-margin.json"):
        continue
    r = load(f)
    entry = {"file": f.name, "sha256": sha(f), "passed": r["passed"]}
    if "bucketed" in r:        # text: languageBucketedDecodeQualification
        b = r["bucketed"]
        entry |= {"test": "languageBucketedDecodeQualification", "forcedRows": len(b["forcedRows"]),
                  "forcedMinimumCosine": b["forcedMinimumCosine"], "forcedArgmaxMismatches": b["forcedArgmaxMismatches"],
                  "forcedNearTies": b["forcedNearTies"], "reference": r["referenceSHA256"]}
    elif "memory" in r:        # visionLanguageSessionReproducesQualifiedRun
        entry |= {"test": "visionLanguageSessionReproducesQualifiedRun", "closedAfterUnload": r["closedAfterUnload"]}
    else:                      # qwen3VLImageTextQualification
        entry |= {"test": "qwen3VLImageTextQualification", "cases": [
            {k: c.get(k) for k in ["name", "promptTokens", "forcedRows", "forcedMinimumCosine", "forcedArgmaxMismatches",
                                   "forcedMismatchesAtNearTies", "greedyFirstDivergence", "greedyDivergenceReferenceMargin",
                                   "greedyAccepted", "preprocessing", "swiftGridMatches", "swiftPixelCosine"]} for c in r["cases"]]}
        margin = requalify / f"{f.stem}-divergence-margin.json"
        if margin.exists():
            # A divergence past the forced rows has no margin in the report; the upstream margin decides it.
            m = load(margin)
            source = load(requalify.parent / m["reference"].split("/")[0] / "reference.json")["source"]
            script = "Forge/language/qwen3_vl_divergence_margin.py"
            entry["divergenceMargin"] = m | {"file": margin.name, "sha256": sha(margin),
                "marginSource": ("a CPU float32 upstream forward pass on the reference record's own pixels, grid and prompt "
                                 "ids, not the stored reference (its forced rows end at 32)"),
                "script": script, "scriptSHA256": sha(kit / script),
                "command": (f"python -I {script} <{source['repository']} checkpoint at revision {source['revision']}> "
                            f"<reference directory {m['reference'].split('/')[0]}> {m['case']} {m['divergence']} "
                            f"{m['kitToken']['id']} {f.name} {margin.name}"),
                "context": ("on 2026-10-26 (power-of-two prefill chunks) this case matched upstream for every greedy token; the "
                            "fixed 64-wide prefill changed the rounding enough to flip this near-tie")}
            others = [c for c in r["cases"] if c["name"] != m["case"]]
            entry["acceptedAtNearTie"] = m["nearTie"] and all(c["greedyAccepted"] for c in others)
    parity[f.stem] = entry
prefill = {"schema": 1, "kind": "language-prefill-width-qualification", "date": "2026-10-27", "device": overlap["device"],
           "hostLoad": host, "width": 64, "bundles": bundles,
           "chunkSweep": {"bundle": sweep["bundle"], "promptTokens": sweep["promptTokens"], "defaultBudgetMiB": sweep["defaultBudgetMiB"],
                          "rows": [{k: v for k, v in row.items() if k not in ("longReply", "shortReply")} for row in sweep["rows"]],
                          "file": sweep_file.name, "sha256": sha(sweep_file)},
           "overlap": {"gate": overlap["gate"], "passed": overlap["passed"], "models": overlap["models"],
                       "file": overlap_file.name, "sha256": sha(overlap_file)},
           "upstreamParity": parity,
           "passed": overlap["passed"] and all(p["passed"] or p.get("acceptedAtNearTie", False) for p in parity.values()),
           "sourceSHA256": sources, "sourceNote": sourceNote,
           "nativeTests": tests("LanguagePrefillChunkSweepTests", "LanguageOverlappedPrefillTests", "LanguageDecodeQualificationTests",
                                "Qwen3VLImageTextQualificationTests", "VisionLanguageSessionTests")}

retention = {"schema": 1, "kind": "core-ai-reshape-retention-reproduction", "date": "2026-10-27", "hostLoad": host,
             "model": "qwen3-0.6b-bucketed-fp16/model.aimodel, function main",
             "nativeTest": "Tests/AetherNativeTests/CoreAIReshapeRetentionTests.swift (AETHER_NATIVE_CORE_AI_RETENTION=1, release)",
             "testSHA256": sha(kit / "Tests/AetherNativeTests/CoreAIReshapeRetentionTests.swift"), "runs": {}}
for flag in ["1", "0"]:
    f = retention_runs / f"retention-reshapes-{flag}.json"
    retention["runs"]["expectFrequentReshapes=" + ("true" if flag == "1" else "false")] = load(f) | {"sha256": sha(f)}
for name, record in [("language-memory-budget-2026-10-27.json", budget), ("language-prefill-width-2026-10-27.json", prefill),
                     ("core-ai-reshape-retention-2026-10-27.json", retention)]:
    (kit / "docs/evidence" / name).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print("budget passed", budget["passed"], "prefill width passed", prefill["passed"])

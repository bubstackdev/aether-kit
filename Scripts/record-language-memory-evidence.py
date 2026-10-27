"""Assembles the memory-budget and Core AI reshape-retention evidence from the native reports.

Usage: python3 Scripts/record-language-memory-evidence.py <directory with the native reports>
(qwen3-0.6b.json, writer.json, vl-2b.json, vl-4b.json from languageMemoryBudgetBoundsRepeatedRequests, and
retention-reshapes-1.json, retention-reshapes-0.json from coreAIRetainsMemoryPerReshapeUntilRelease).
"""
import hashlib, json, pathlib, sys
kit = pathlib.Path(__file__).resolve().parent.parent
runs = pathlib.Path(sys.argv[1])
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
host = "Shared host: GPU jobs of cooperating agents are serialized by a lock, other host work is not; memory is Metal's currentAllocatedSize in the test process"
sources = {p: sha(kit / p) for p in ["Sources/AetherLanguageCoreAI/LanguageGenerationSession.swift",
    "Sources/AetherLanguageCoreAI/LanguageMemoryBudget.swift", "Sources/AetherLanguageCoreAI/VisionLanguageSession.swift"]}
models, reports = [], {}
for name in ["qwen3-0.6b", "writer", "vl-2b", "vl-4b"]:
    f = runs / f"{name}.json"
    r = json.loads(f.read_text())
    assert r.get("error") is None and r["passed"], name
    reports[name] = {"file": f.name, "sha256": sha(f)}
    for m in r["models"]:
        models.append({k: m[k] for k in ["name", "promptTokens", "replies", "warmMetalMiB", "maximumMetalMiB", "maximumRequestGrowthMiB",
                                          "boundMiB", "reloads", "mismatchedReplies", "unloadedMetalMiB", "seconds", "passed"]}
                      | {"requests": r["requests"], "metalMiBAfterEachRequest": [s["metalMiB"] for s in m["samples"]],
                         "reloadsAfterEachRequest": [s["reloads"] for s in m["samples"]]})
budget = {"schema": 1, "kind": "language-memory-budget-qualification", "passed": all(m["passed"] for m in models),
          "date": "2026-10-27", "device": r["device"], "hostLoad": host, "budgetMiB": r["budgetMiB"], "gate": r["gate"],
          "bundles": {"qwen3-0.6b": "qwen3-0.6b-bucketed-fp16/model.aimodel (raw Forge export, Qwen/Qwen3-0.6B)",
                      "writer": "qwen3-4b-instruct-2507-abliterated-bucketed-fp16 (huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated)",
                      "vl-2b": "qwen3-vl-2b-bundle-macos (Qwen/Qwen3-VL-2B-Instruct)",
                      "vl-4b": "qwen3-vl-4b-abliterated-bundle-macos (huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated)"},
          "models": models, "nativeReports": reports, "sourceSHA256": sources,
          "nativeTest": "Tests/AetherNativeTests/LanguageMemoryBudgetQualificationTests.swift (AETHER_NATIVE_LANGUAGE_MEMORY_BUDGET=1, release)",
          "testSHA256": sha(kit / "Tests/AetherNativeTests/LanguageMemoryBudgetQualificationTests.swift")}
retention = {"schema": 1, "kind": "core-ai-reshape-retention-reproduction", "date": "2026-10-27", "hostLoad": host,
             "model": "qwen3-0.6b-bucketed-fp16/model.aimodel, function main",
             "nativeTest": "Tests/AetherNativeTests/CoreAIReshapeRetentionTests.swift (AETHER_NATIVE_CORE_AI_RETENTION=1, release)",
             "testSHA256": sha(kit / "Tests/AetherNativeTests/CoreAIReshapeRetentionTests.swift"), "runs": {}}
for flag in ["1", "0"]:
    f = runs / f"retention-reshapes-{flag}.json"
    retention["runs"]["expectFrequentReshapes=" + ("true" if flag == "1" else "false")] = json.loads(f.read_text()) | {"sha256": sha(f)}
for name, record in [("language-memory-budget-2026-10-27.json", budget), ("core-ai-reshape-retention-2026-10-27.json", retention)]:
    (kit / "docs/evidence" / name).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print("budget passed", budget["passed"])

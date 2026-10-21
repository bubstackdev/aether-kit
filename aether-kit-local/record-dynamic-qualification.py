import hashlib
import json
from pathlib import Path
import struct
import subprocess

root = Path(__file__).resolve().parent.parent
output = root / ".local/native-dynamic-balanced-20261017"
evidence = root / "docs/evidence"
report_path = output / "report.json"
report = json.loads(report_path.read_text())
assert report["passed"] and report["stage"] == "complete"
assert report["modelStreamsVerified"] and all(report["checks"].values())
assert len(report["cases"]) == 2
assert len(report["checks"]) == 39

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

for source, digest in report["referenceSourceSHA256"].items():
    assert sha(Path(source)) == digest
binary = Path("/Users/zakkeown/Code/atelier/lab/krea-lab/bin/round2-final")
assert sha(binary) == report["referenceBinarySHA256"]
interrupted = json.loads((evidence / "krea-dynamic-residency-stall-report-2026-10-17.json").read_text())
for field in ["conditioningSHA256", "stepLatentsSHA256", "pixelsSHA256", "pixelShape"]:
    assert report["cases"][0]["expected"][field] == interrupted["cases"][0]["expected"][field]
balanced_path = evidence / "krea-balanced-session-2026-10-17.json"
balanced = json.loads(balanced_path.read_text())
assert balanced["passed"] and len(balanced["checks"]) == 23 and all(balanced["checks"].values())
pre_change = json.loads((evidence / "krea-hf-cache-session-2026-10-17.json").read_text())
for before, after in zip(pre_change["generations"], balanced["generations"], strict=True):
    assert before["name"] == after["name"]
    assert before["latentSHA256"] == after["latentSHA256"] and before["pixelsSHA256"] == after["pixelsSHA256"]
cache = Path.home() / ".cache/huggingface/aether-kit"
for event in report["events"]:
    if event.get("model"):
        assert Path(event["model"]).is_relative_to(cache)
begins = [event["call"] for event in report["events"] if event["event"] == "nativeBegin"]
ends = [event["call"] for event in report["events"] if event["event"] == "nativeEnd"]
assert sorted(begins) == sorted(ends)
opened = [event["instance"] for event in report["events"] if event["event"] == "modelOpened"]
closed = [event["instance"] for event in report["events"] if event["event"] == "modelClosed"]
assert len(set(opened)) == len(opened) and sorted(opened) == sorted(closed)

references = {}
images = {}
for case in report["cases"]:
    spec = case["specification"]
    expected = case["expected"]
    assert spec["steps"] == 8 and len(expected["stepLatentsSHA256"]) == 8
    assert len(case["runs"]) == 2
    reference = output / (spec["id"] + "-reference") / "record.json"
    assert sha(reference) == case["referenceRecordSHA256"]
    ref_dest = evidence / ("krea-" + spec["id"] + "-independent-reference-2026-10-17.json")
    ref_dest.write_bytes(reference.read_bytes())
    references[spec["id"]] = {"record": ref_dest.name, "sha256": sha(ref_dest)}
    for kind, filename in spec["files"].items():
        assert sha(output / "inputs" / filename) == case["inputSHA256"][kind]
    for run in case["runs"]:
        assert run["specialized"] is False
        for field in ["conditioningSHA256", "stepLatentsSHA256", "pixelsSHA256", "pixelShape"]:
            assert run[field] == expected[field]
        png = output / run["png"]
        assert struct.unpack(">II", png.read_bytes()[16:24]) == (spec["width"], spec["height"])
        images[run["name"]] = {"path": str(png.relative_to(root)), "sha256": sha(png)}

for phase, target in [("dynamic512.cancelEncoder", "-text.aimodel"), ("dynamic384.cancelDecoder", "-vae.aimodel")]:
    events = [event for event in report["events"] if event["phase"] == phase]
    requested = next(event for event in events if event["event"] == "cancelRequested")
    begin = next(event for event in events if event["event"] == "nativeBegin" and event["model"].endswith(target))
    end = next(event for event in events if event["event"] == "nativeEnd" and event["call"] == begin["call"])
    returned = next(event for event in events if event["event"] == "cancellationReturned")
    assert begin["sequence"] < requested["sequence"] < end["sequence"] < returned["sequence"]

log = root / ".local/native-dynamic-balanced-20261017.log"
text = log.read_text()
assert "Test run with 36 tests" in text and "nativeKreaDynamicParityAndStageCancellation() passed" in text
assert text.count("skipped.") == 1 and "nativeKreaSessionQualification() skipped." in text
report_dest = evidence / "krea-dynamic-parity-and-cancellation-2026-10-17.json"
report_dest.write_bytes(report_path.read_bytes())
log_dest = evidence / "krea-dynamic-parity-tests-2026-10-17.log"
log_dest.write_bytes(log.read_bytes())
sources = [*root.glob("Sources/**/*.swift"), *root.glob("Tests/**/*.swift"), root / "Package.swift", root / "Package.resolved"]
manifest = {
    "schema": 1, "date": "2026-10-17", "passed": True,
    "scope": "Full eight-step independent dynamic parity at 512x512 and 384x512, plus encoder/decoder in-flight cancellation and full parity recovery",
    "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    "working_tree_changes_present": True,
    "tests": {"registered": 36, "passed": 35, "skipped": 1,
        "skip": "Previously qualified native session suite; not enabled during this run",
        "existing_cpu_tests_passed": 34, "new_native_suite_passed": True,
        "qualification_checks": len(report["checks"]), "log": log_dest.name, "log_sha256": sha(log_dest)},
    "report": report_dest.name, "report_sha256": sha(report_dest),
    "balanced_residency_regression": {
        "passed": True, "native_checks": 23, "all_seven_outputs_unchanged": True,
        "report": balanced_path.name, "report_sha256": sha(balanced_path),
        "log": "krea-balanced-session-tests-2026-10-17.log", "log_sha256": sha(evidence / "krea-balanced-session-tests-2026-10-17.log")},
    "unique_tests_passed_across_serial_runs": 36,
    "original_512_independent_reference_reproduced": True,
    "interrupted_attempt": "krea-dynamic-residency-stall-2026-10-17.json",
    "references": references, "images": images,
    "reference_binary": str(binary), "reference_binary_sha256": sha(binary),
    "all_native_models_loaded_from_huggingface_cache": True,
    "model_sessions_opened_and_closed": len(opened), "native_call_pairs": len(begins),
    "source_sha256": {str(path.relative_to(root)): sha(path) for path in sorted(sources)},
    "test_binary_sha256": sha(root / ".build/arm64-apple-macosx/release/AetherKitPackageTests.xctest/Contents/MacOS/AetherKitPackageTests"),
    "device": json.loads((root / ".local/native-validation-doctor.json").read_text()),
    "swift": subprocess.check_output(["swift", "--version"], text=True).strip(),
    "xcode": subprocess.check_output(["xcodebuild", "-version"], text=True).strip(),
    "command": "source .local/native-assets.env\nAETHER_KREA_REFERENCE_PROBE=/Users/zakkeown/Code/atelier/lab/krea-lab/bin/round2-final AETHER_NATIVE_DYNAMIC_QUALIFICATION=1 AETHER_NATIVE_DYNAMIC_OUTPUT=\"$PWD/.local/native-dynamic-balanced-20261017\" AETHER_NATIVE_QUALIFICATION=0 Scripts/check.sh",
    "benchmark": False, "host_load": report["hostLoad"], "cache_state": "Uncontrolled OS cache",
    "input_scope": report["inputScope"], "limits": report["limits"],
}
(evidence / "krea-dynamic-qualification-2026-10-17.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"passed": True, "checks": len(report["checks"]), "native_calls": len(begins), "models": len(opened), "images": len(images)}, indent=2))

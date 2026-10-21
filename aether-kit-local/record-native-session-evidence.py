import hashlib
import json
from pathlib import Path
import struct
import subprocess

root = Path(__file__).resolve().parent.parent
run = root / ".local/native-session-20261017-native-boundary"
evidence = root / "docs/evidence"
report_path = run / "report.json"
report = json.loads(report_path.read_text())
assert report["passed"] and report["stage"] == "complete"
assert len(report["checks"]) >= 20 and all(report["checks"].values())
assert len(report["generations"]) == 7
assert report["modelStreamsVerified"]

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

events = report["events"]
assert [event["sequence"] for event in events] == list(range(len(events)))
begins = [event["call"] for event in events if event["event"] == "nativeBegin"]
ends = [event["call"] for event in events if event["event"] == "nativeEnd"]
assert sorted(begins) == sorted(ends)
opened = [event["instance"] for event in events if event["event"] == "modelOpened"]
closed = [event["instance"] for event in events if event["event"] == "modelClosed"]
assert sorted(opened) == sorted(closed) and len(opened) == len(set(opened))

images = {}
for generation in report["generations"]:
    png = run / generation["png"]
    assert struct.unpack(">II", png.read_bytes()[16:24]) == (generation["width"], generation["height"])
    images[generation["name"]] = {"path": str(png.relative_to(root)), "sha256": sha(png)}

log = root / ".local/native-session-20261017-native-boundary.log"
assert "Test run with 34 tests" in log.read_text() and "skipped." not in log.read_text()
report_dest = evidence / "krea-native-session-2026-10-17.json"
report_dest.write_bytes(report_path.read_bytes())
log_dest = evidence / "krea-native-session-tests-2026-10-17.log"
log_dest.write_bytes(log.read_bytes())
paths = [*root.glob("Sources/**/*.swift"), *root.glob("Tests/**/*.swift"), root / "Package.swift", root / "Package.resolved"]
bundle = "/Users/zakkeown/Code/aether-engine/forge/out/krea-2-turbo-realism/variants/macos-any-gpu"
fixtures = "/Users/zakkeown/Code/aether-engine/forge/out/krea-2-turbo-realism/fixtures/4c12ae019488ff1c"
manifest = {
    "schema": 1,
    "date": "2026-10-17",
    "scope": "Native model-session reuse, cancellation/recovery, dynamic fallback, and unload during active work",
    "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    "working_tree_changes_present": True,
    "passed": True,
    "benchmark": False,
    "host_load": "Shared host under significant concurrent load, confirmed by user; performance qualification deferred",
    "cache_state": "Uncontrolled OS specialization cache",
    "execution": "Approved execution outside the agent sandbox",
    "device": json.loads((root / ".local/native-validation-doctor.json").read_text()),
    "swift": subprocess.check_output(["swift", "--version"], text=True).strip(),
    "xcode": subprocess.check_output(["xcodebuild", "-version"], text=True).strip(),
    "tests": {"passed": 34, "skipped": 0, "native_qualification_checks": len(report["checks"]),
              "log": log_dest.name, "log_sha256": sha(log_dest)},
    "report": report_dest.name,
    "report_sha256": sha(report_dest),
    "model_sessions_opened_and_closed": len(opened),
    "native_call_pairs": len(begins),
    "images": images,
    "command": ["env", f"AETHER_KREA_BUNDLE={bundle}", f"AETHER_KREA_FIXTURES={fixtures}",
        "AETHER_KREA_LAB=/Users/zakkeown/Code/atelier/lab/krea-lab",
        "AETHER_KREA_SHAPE=/Users/zakkeown/Code/bubstackdev/aether/lab/aether-prototype/.local/tower-20.json",
        "AETHER_NATIVE_QUALIFICATION=1", f"AETHER_NATIVE_OUTPUT={run}", "Scripts/check.sh"],
    "source_sha256": {str(path.relative_to(root)): sha(path) for path in sorted(paths)},
    "test_binary_sha256": sha(root / ".build/arm64-apple-macosx/release/AetherKitPackageTests.xctest/Contents/MacOS/AetherKitPackageTests"),
    "limits": report["limits"],
}
destination = evidence / "krea-native-session-validation-2026-10-17.json"
destination.write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"passed": True, "checks": len(report["checks"]), "model_sessions": len(opened),
                  "native_calls": len(begins), "images": len(images)}, indent=2))

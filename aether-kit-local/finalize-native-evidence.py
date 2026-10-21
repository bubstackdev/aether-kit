import hashlib
import json
from pathlib import Path
import struct
import subprocess

root = Path(__file__).resolve().parent.parent
evidence = root / "docs/evidence"

def sha(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()

bundle = "/Users/zakkeown/Code/aether-engine/forge/out/krea-2-turbo-realism/variants/macos-any-gpu"
fixtures = "/Users/zakkeown/Code/aether-engine/forge/out/krea-2-turbo-realism/fixtures/4c12ae019488ff1c"
lab = "/Users/zakkeown/Code/atelier/lab/krea-lab"
registration = "../lab/aether-prototype/.local/tower-20.json"
binary = ".build/arm64-apple-macosx/release/aether-probe"
record = {
    "schema": 1,
    "date": "2026-10-17",
    "scope": "Native correctness of the retained Krea tower fixture, provided and encoded conditioning",
    "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    "working_tree_changes_present": True,
    "execution": "Approved execution outside the agent sandbox; normal Core AI cache access",
    "host_load": "Shared host under significant concurrent load, confirmed by user",
    "benchmark": False,
    "cache_state": "Uncontrolled OS specialization cache; no cold/warm performance claims",
    "doctor": json.loads((root / ".local/native-validation-doctor.json").read_text()),
    "toolchain": {"swift": "6.4 (swiftlang-6.4.0.34.1 clang-2100.3.34.1)", "xcode": "27.1 (27A9275)"},
    "assets": {"bundle": bundle, "fixtures": fixtures, "lab": lab, "registration": registration},
    "affine_smoke": {
        "passed": True,
        "policies": ["cpuOnly", "automatic", "preferGPU"],
        "expected_output": [-3, 1, 3, 7],
        "checks": "Two executions and output retention after unload for each policy",
        "command_template": [binary, "smoke", ".local/smoke-v1/affine.aimodel", "POLICY"],
        "observation": "All three commands exited 0 earlier in this session",
    },
    "release_tests": {
        "passed": True, "count": 33, "skipped": 0,
        "command": ["env", f"AETHER_KREA_BUNDLE={bundle}", f"AETHER_KREA_FIXTURES={fixtures}", f"AETHER_KREA_LAB={lab}", "Scripts/check.sh"],
        "log": "krea-native-tests-2026-10-17.log",
        "log_sha256": sha(evidence / "krea-native-tests-2026-10-17.log"),
    },
    "parity": {},
    "source_sha256": {},
    "binary_sha256": sha(root / binary),
    "limitations": [
        "One retained 768x1344, eight-step, seed-71344 fixture with its source-linked specialization",
        "No qualification of other shapes, prompts, dynamic fallback, or physical iOS execution",
        "Native repeated-session residency, cancellation/drain, and malformed-input gates remain pending",
        "Host-copy reference executor; device-resident optimization and performance benchmarking remain pending",
    ],
}
log = (evidence / record["release_tests"]["log"]).read_text()
assert "Test run with 33 tests" in log and "skipped." not in log
for mode in ("provided", "encoded"):
    output = f".local/native-parity-{mode}-20261017"
    report_path = root / output / "report.json"
    report = json.loads(report_path.read_text())
    assert report["passed"] and report["stage"] == "complete"
    assert report["modelStreamsVerified"] and report["shapeRegistrationVerified"] and report["tokenIDsMatch"]
    steps = [event for event in report["events"] if event["event"] == "step"]
    assert [event["step"] for event in steps] == list(range(8))
    assert all(event.get("passed", True) for event in report["events"])
    assert report["events"][-1]["event"] == "pixels"
    image = root / output / "image.png"
    assert struct.unpack(">II", image.read_bytes()[16:24]) == (768, 1344)
    destination = evidence / f"krea-native-{mode}-2026-10-17.json"
    destination.write_bytes(report_path.read_bytes())
    record["parity"][mode] = {
        "passed": True,
        "report": destination.name,
        "report_sha256": sha(destination),
        "output_png": f"{output}/image.png",
        "output_png_sha256": sha(image),
        "command": [binary, "parity-krea", bundle, lab, fixtures, registration, output, mode],
    }
for path in sorted(root.glob("Sources/**/*.swift")):
    record["source_sha256"][str(path.relative_to(root))] = sha(path)
for name in ("Package.swift", "Package.resolved"):
    record["source_sha256"][name] = sha(root / name)
destination = evidence / "krea-native-validation-2026-10-17.json"
destination.write_text(json.dumps(record, indent=2) + "\n")
print("Recorded verified provided/encoded reports, image hashes, test log, source hashes, and shared-host limits.")

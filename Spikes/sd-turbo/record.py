"""Write a compact evidence record after successful qualification, without copying model assets."""

import argparse
import hashlib
import json
import re
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--qualification", type=Path, required=True)
p.add_argument("--cpu-log", type=Path, required=True)
p.add_argument("--native-log", type=Path, required=True)
p.add_argument("--floor-log", type=Path, required=True)
p.add_argument("--ios-floor-log", type=Path)
p.add_argument("--api-log", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
root = a.qualification
native_log = a.native_log.read_text()
if "Test sdTurboNativeParity() passed" not in native_log or "✘" in native_log:
    raise SystemExit("Native qualification did not pass")
cpu_log = a.cpu_log.read_text()
if "✘" in cpu_log or "error:" in cpu_log:
    raise SystemExit("CPU suite has failures")
registered = sum(map(int, re.findall(r"Test run with (\d+) tests", cpu_log)))
skipped = len(re.findall(r"Test .* skipped", cpu_log))
if registered <= skipped:
    raise SystemExit("CPU log has no executed test runs")
floor_log = a.floor_log.read_text()
if "macOS floor consumer: minos 14.0, CoreAI weak-linked" not in floor_log:
    raise SystemExit("macOS floor gate did not pass")
api_log = a.api_log.read_text()
for module in ["AetherKit", "AetherAssets", "AetherLanguage", "AetherLanguageMacros"]:
    if f"{module}: matches the baseline" not in api_log:
        raise SystemExit("Frozen API gate did not pass")
ios_log = a.ios_floor_log.read_text() if a.ios_floor_log else ""
if (
    a.ios_floor_log
    and "iOS floor consumer: minos 17.0, CoreAI weak-linked" not in ios_log
):
    raise SystemExit("iOS build/link floor gate did not pass")
prepare = json.loads((root / "prepare-report.json").read_text())
manifest = json.loads((root / "bundle/diffusion.json").read_text())


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


repo = Path(__file__).resolve().parents[2]
code_paths = [repo / "Package.swift", Path(__file__).resolve()]
for directory in [
    "Sources/AetherDiffusion",
    "Sources/AetherDiffusionCoreAI",
    "Sources/AetherSDTurbo",
    "Tests/AetherDiffusionTests",
]:
    code_paths.extend(sorted((repo / directory).glob("*.swift")))
record = {
    "date": datetime.now(timezone.utc).date().isoformat(),
    "status": "pass",
    "scope": "v0.0.6 SD-Turbo spike; fixed 512x512 float32 native Core AI reference path",
    "host": {
        "hardware": subprocess.check_output(
            ["sysctl", "-n", "hw.model"], text=True
        ).strip(),
        "architecture": platform.machine(),
        "os": prepare["host"],
        "swift": "6.4",
    },
    "compute": {
        "policy": "preferGPU",
        "coordination": "AetherKit SharedGPULease",
        "operatorPlacement": "not instrumented",
    },
    "source": prepare["source"],
    "versions": prepare["versions"],
    "scriptSHA256": prepare["scriptSHA256"],
    "runtimeAndTestsSHA256": {
        str(path.relative_to(repo)): sha(path) for path in code_paths
    },
    "sourceFileSHA256": prepare["checkpointSHA256"],
    "bundleFileSHA256": manifest["files"],
    "geometry": [manifest["width"], manifest["height"]],
    "schedules": manifest["schedules"],
    "native": json.loads((root / "native-report.json").read_text()),
    "tokenizer": {
        "cases": 8,
        "comparison": "exact CLIPTokenizerFast IDs; all four generation prompts also exact",
    },
    "cpu": {
        "registered": registered,
        "skipped": skipped,
        "executed": registered - skipped,
        "status": "pass",
    },
    "deploymentFloor": {
        "macOS14ProductAndWeakLink": "pass",
        "iOS17ProductAndWeakLink": "pass" if a.ios_floor_log else "not run",
        "iOSModelExecution": "not qualified",
    },
    "frozenAPI": "four modules match committed baselines",
    "gateLogSHA256": {
        "cpu": sha(a.cpu_log),
        "native": sha(a.native_log),
        "macosFloor": sha(a.floor_log),
        "api": sha(a.api_log),
        **({"iosFloor": sha(a.ios_floor_log)} if a.ios_floor_log else {}),
    },
    "artifactSHA256": {
        str(f.relative_to(root)): sha(f)
        for f in sorted(root.rglob("*"))
        if f.is_file() and ("reference" in f.parts or f.name.startswith("native-"))
    },
    "limitations": [
        "No iOS qualification",
        "Float32, host-copy reference implementation",
        "All four step counts measured at fixed 512x512",
        "No quantization, residency performance, memory bound, or operator-placement claim",
        "Stability AI Community License applies to model assets; no model publication",
    ],
}
a.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

"""Write a compact evidence record after successful qualification, without copying model assets."""

import argparse
import hashlib
import json
import re
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--qualification", type=Path, required=True)
p.add_argument("--cpu-log", type=Path, required=True)
p.add_argument("--native-log", type=Path, required=True)
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
prepare = json.loads((root / "prepare-report.json").read_text())
manifest = json.loads((root / "bundle/diffusion.json").read_text())


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


record = {
    "date": "2026-11-02",
    "status": "pass",
    "scope": "v0.0.6 SD-Turbo spike; fixed 512x512 float32 native Core AI reference path",
    "host": {
        "hardware": "MacBook Pro Mac17,6, Apple M5 Max, 64 GB",
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
    "sourceFileSHA256": prepare["checkpointSHA256"],
    "bundleFileSHA256": manifest["files"],
    "geometry": [manifest["width"], manifest["height"]],
    "schedules": manifest["schedules"],
    "native": json.loads((root / "native-report.json").read_text()),
    "tokenizer": {
        "cases": 8,
        "comparison": "exact CLIPTokenizerFast IDs; two generation prompts also exact",
    },
    "cpu": {
        "registered": registered,
        "skipped": skipped,
        "executed": registered - skipped,
        "status": "pass",
    },
    "deploymentFloor": {"macOS14ProductBuild": "pass", "iOS": "not qualified"},
    "artifactSHA256": {
        str(f.relative_to(root)): sha(f)
        for f in sorted(root.rglob("*"))
        if f.is_file() and ("reference" in f.parts or f.name.startswith("native-"))
    },
    "limitations": [
        "No iOS qualification",
        "Float32, host-copy reference implementation",
        "One and four steps measured; two and three schedules exported",
        "No quantization, residency performance, memory bound, or operator-placement claim",
        "Stability AI Community License applies to model assets; no model publication",
    ],
}
a.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

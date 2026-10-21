import hashlib
import json
from pathlib import Path
import struct

root = Path(__file__).resolve().parent.parent
evidence = root / "docs/evidence"
cache = Path.home() / ".cache/huggingface/aether-kit"
run = root / ".local/native-session-hf-cache-20261017"
report_path = run / "report.json"
report = json.loads(report_path.read_text())
baseline_path = evidence / "krea-native-session-2026-10-17.json"
baseline = json.loads(baseline_path.read_text())
migration_path = evidence / "model-cache-migration-2026-10-17.json"
migration = json.loads(migration_path.read_text())
assert report["passed"] and report["stage"] == "complete" and all(report["checks"].values())
assert report["modelStreamsVerified"] and len(report["checks"]) == 23
assert migration["passed"] and len(report["generations"]) == 7
assert report["bundleFingerprint"] == baseline["bundleFingerprint"]
assert report["shapeRegistrationSHA256"] == migration["registration"]["cache_sha256"]

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

for move in migration["moves"]:
    source = Path(move["resolved_original_path"])
    destination = Path(move["cache_path"])
    assert source.is_symlink() and source.resolve() == destination
    assert destination.is_dir() and not destination.is_symlink()
for event in report["events"]:
    if event.get("model"):
        assert Path(event["model"]).is_relative_to(cache), event
for generation, original in zip(report["generations"], baseline["generations"], strict=True):
    assert generation["name"] == original["name"]
    assert generation["latentSHA256"] == original["latentSHA256"]
    assert generation["pixelsSHA256"] == original["pixelsSHA256"]
    png = run / generation["png"]
    assert struct.unpack(">II", png.read_bytes()[16:24]) == (generation["width"], generation["height"])

report_destination = evidence / "krea-hf-cache-session-2026-10-17.json"
report_destination.write_bytes(report_path.read_bytes())
log = root / ".local/native-session-hf-cache-20261017.log"
assert "Test run with 34 tests" in log.read_text() and "skipped." not in log.read_text()
log_destination = evidence / "krea-hf-cache-tests-2026-10-17.log"
log_destination.write_bytes(log.read_bytes())
smoke = {}
for policy in ["cpuOnly", "automatic", "preferGPU"]:
    result = json.loads((root / f".local/hf-cache-smoke-{policy}.json").read_text())
    assert result["passed"] and result["output"] == [-3, 1, 3, 7]
    smoke[policy] = result
previous_manifest_path = evidence / "krea-native-session-validation-2026-10-17.json"
previous_manifest = json.loads(previous_manifest_path.read_text())
for name, expected in previous_manifest["source_sha256"].items():
    assert sha(root / name) == expected, name

manifest = {
    "schema": 1, "date": "2026-10-17", "passed": True,
    "scope": "All native session qualification repeated from Hugging Face cache paths after hash-verified model relocation",
    "cache_root": str(cache),
    "environment": {
        "AETHER_KREA_BUNDLE": str(cache / "krea-2-turbo-realism/macos-any-gpu"),
        "AETHER_KREA_SHAPE": str(cache / "krea-2-turbo-realism/tower-20.json"),
        "AETHER_SMOKE_MODEL": str(cache / "smoke-v1/affine.aimodel"),
    },
    "project_environment_file": ".local/native-assets.env",
    "environment_file_sha256": sha(root / ".local/native-assets.env"),
    "migration": migration_path.name, "migration_sha256": sha(migration_path),
    "native_report": report_destination.name, "native_report_sha256": sha(report_destination),
    "tests": {"passed": 34, "skipped": 0, "native_checks_passed": len(report["checks"]),
              "log": log_destination.name, "log_sha256": sha(log_destination)},
    "all_model_loads_from_cache": True,
    "all_seven_outputs_match_before_migration": True,
    "affine_smoke": smoke,
    "source_and_toolchain_record": previous_manifest_path.name,
    "source_and_toolchain_record_sha256": sha(previous_manifest_path),
    "sources_unchanged_since_that_record": True,
    "command": "source .local/native-assets.env\nAETHER_NATIVE_QUALIFICATION=1 AETHER_NATIVE_OUTPUT=\"$PWD/.local/native-session-hf-cache-20261017\" Scripts/check.sh",
    "images": {g["name"]: {"path": str((run / g["png"]).relative_to(root)), "sha256": sha(run / g["png"])} for g in report["generations"]},
    "benchmark": False,
    "host_load": "Shared host under significant concurrent load; quiet-window benchmarking deferred",
    "limits": report["limits"],
}
(evidence / "huggingface-cache-validation-2026-10-17.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Verified cached native session checks, all seven output hashes, affine policies, compatibility symlinks, and source identities.")

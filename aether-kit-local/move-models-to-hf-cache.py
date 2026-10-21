import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

root = Path(__file__).resolve().parent.parent
cache = Path.home() / ".cache/huggingface/aether-kit"
models = cache / "krea-2-turbo-realism"
registration_source = root.parent / "lab/aether-prototype/.local/tower-20.json"
plan = [
    (Path("/Users/zakkeown/Code/aether-engine/forge/out/krea-2-turbo-realism/variants/macos-any-gpu"), models / "macos-any-gpu"),
    (Path("/Users/zakkeown/Code/atelier/lab/krea-lab/forge/fixed-first"), models / "tower-20"),
    (root / ".local/smoke-v1", cache / "smoke-v1"),
]

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def inventory(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"unexpected internal symlink: {path}")
        if path.is_file():
            result[str(path.relative_to(directory))] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    for model in directory.glob("*.aimodel"):
        assert result[f"{model.name}/main.mlirb"]["sha256"] == (model / "main.hash").read_bytes().hex()
    return result

report = {"schema": 1, "date": "2026-10-17", "passed": False, "cache_root": str(cache), "moves": []}
report_file = root / "docs/evidence/model-cache-migration-2026-10-17.json"

def write():
    report_file.write_text(json.dumps(report, indent=2) + "\n")

assert json.loads((root / ".local/native-session-20261017-native-boundary/report.json").read_text())["passed"]
for source, destination in plan:
    assert source.is_dir() and not source.is_symlink(), source
    assert not destination.exists() and not destination.is_symlink(), destination
assert not (models / "tower-20.json").exists()
original_registration = json.loads(registration_source.read_text())
cache.mkdir(parents=True, exist_ok=True)
write()
for logical_source, destination in plan:
    source = logical_source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Hashing source: {source}", flush=True)
    before = inventory(source)
    same_device = source.stat().st_dev == destination.parent.stat().st_dev
    if same_device:
        source.rename(destination)
        try:
            source.symlink_to(destination, target_is_directory=True)
        except BaseException:
            destination.rename(source)
            raise
    else:
        staging = destination.with_name(destination.name + ".import-" + uuid.uuid4().hex)
        print(f"Copying to cache staging: {staging}", flush=True)
        shutil.copytree(source, staging)
        assert inventory(staging) == before, "cache staging differs from source; original retained"
        staging.rename(destination)
        backup = source.with_name(source.name + ".migrating-" + uuid.uuid4().hex)
        source.rename(backup)
        try:
            source.symlink_to(destination, target_is_directory=True)
        except BaseException:
            backup.rename(source)
            raise
        # The user's requested move authorizes removal of the verified source copy.
        # All bytes were checked against the cache before replacing the old path.
        shutil.rmtree(backup)
    assert inventory(destination) == before
    assert source.is_symlink() and source.resolve() == destination.resolve()
    report["moves"].append({
        "original_path": str(logical_source), "resolved_original_path": str(source),
        "cache_path": str(destination), "compatibility_symlink": str(source),
        "method": "rename" if same_device else "hash-verified copy then remove source",
        "bytes": sum(item["bytes"] for item in before.values()), "files": before,
    })
    write()
    print(f"Moved and verified: {destination}", flush=True)

registration = dict(original_registration)
registration["segments"] = [str(models / "tower-20" / Path(path).name) for path in registration["segments"]]
for path in registration["segments"]:
    assert Path(path).is_dir()
registration_destination = models / "tower-20.json"
registration_destination.write_text(json.dumps(registration, indent=2) + "\n")
report["registration"] = {
    "original": str(registration_source), "original_sha256": digest(registration_source),
    "cache_path": str(registration_destination), "cache_sha256": digest(registration_destination),
    "only_segments_changed": {k:v for k,v in registration.items() if k != "segments"} ==
                             {k:v for k,v in original_registration.items() if k != "segments"},
}
assert report["registration"]["only_segments_changed"]

environment = {
    "AETHER_KREA_BUNDLE": str(models / "macos-any-gpu"),
    "AETHER_KREA_SHAPE": str(registration_destination),
    "AETHER_KREA_FIXTURES": "/Users/zakkeown/Code/aether-engine/forge/out/krea-2-turbo-realism/fixtures/4c12ae019488ff1c",
    "AETHER_KREA_LAB": "/Users/zakkeown/Code/atelier/lab/krea-lab",
    "AETHER_SMOKE_MODEL": str(cache / "smoke-v1/affine.aimodel"),
}
import shlex
environment_file = root / ".local/native-assets.env"
environment_file.write_text("# Local model locations; source before running native checks.\n" +
    "".join(f"export {key}={shlex.quote(value)}\n" for key,value in environment.items()))
report["project_environment"] = str(environment_file.relative_to(root))
report["passed"] = True
write()
print("Migration complete; original paths are compatibility symlinks and .local/native-assets.env references the cache.", flush=True)

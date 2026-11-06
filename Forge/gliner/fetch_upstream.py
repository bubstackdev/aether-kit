#!/usr/bin/env python3
"""Fetches the pinned GLiNER2.5-Decide sources: the gliner2 library wheel (the upstream code) and the checkpoint.

    Forge/.venv/bin/python -I Forge/gliner/fetch_upstream.py [--output DIR] [--no-snapshot] [--zoo]

Upstream code (fastino-ai/GLiNER2, PyPI `gliner2` 2.0.0, Apache-2.0): the release wheel from PyPI, checked against its
pinned SHA-256 (PyPI's own digest), unpacked under the output directory (default:
~/.cache/aether-kit/gliner/upstream/gliner2-2.0.0) with an UNPACKED.json stamp. The package is never installed:
`gliner_graphs.load_upstream` imports it read-only. Its runtime needs (pydantic, requests, tqdm, transformers, torch,
safetensors) are already in the Forge environment, so nothing is added to Forge/pyproject.toml.

Checkpoint (fastino/GLiNER2.5-Decide, Apache-2.0): the files `source-pins.json` lists, downloaded into the Hugging Face
cache at the pinned revision, then verified. With `--zoo`, also the zoo bundle's sidecars (`zooFixtures`: the
classifier.json contract and gate fixtures of mlboydaisuke/GLiNER2.5-Decide-CoreAI, never bundled), verified the same way.
Idempotent: an existing, matching unpack or snapshot is left alone.
"""
import argparse
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "source-pins.json").read_text())
CODE = PINS["upstreamCode"]
DEFAULT = Path.home() / f".cache/aether-kit/gliner/upstream/gliner2-{CODE['version']}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_code(output: Path) -> None:
    stamp = output / "UNPACKED.json"
    if stamp.exists() and json.loads(stamp.read_text()).get("wheelSHA256") == CODE["wheelSHA256"]:
        return
    with urllib.request.urlopen(CODE["wheelURL"], timeout=120) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != CODE["wheelSHA256"]:
        raise SystemExit(f"{CODE['wheel']} has SHA-256 {digest}, not the pinned {CODE['wheelSHA256']}")
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as wheel:
        for member in wheel.namelist():
            target = (output / member).resolve()
            if not str(target).startswith(str(output.resolve()) + "/"):
                raise SystemExit(f"wheel member {member} escapes the unpack directory")
        wheel.extractall(output)
    for relative, expected in CODE["files"].items():
        if sha256_file(output / relative) != expected:
            raise SystemExit(f"{relative} in the unpacked wheel differs from its pin")
    stamp.write_text(json.dumps({"wheel": CODE["wheel"], "wheelSHA256": CODE["wheelSHA256"]}, indent=1) + "\n")


def fetch_repo(repository: str, revision: str, files: dict[str, str]) -> Path:
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(repository, revision=revision, allow_patterns=list(files)))
    for relative, expected in files.items():
        if sha256_file(path / relative) != expected:
            raise SystemExit(f"{relative} in {path} differs from the pinned {revision[:12]} file")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT)
    ap.add_argument("--no-snapshot", action="store_true", help="fetch only the upstream code")
    ap.add_argument("--zoo", action="store_true", help="also fetch the zoo bundle's contract and gate fixtures")
    args = ap.parse_args()
    fetch_code(args.output)
    print(args.output)
    if not args.no_snapshot:
        print(fetch_repo(PINS["repository"], PINS["revision"], PINS["files"]))
    if args.zoo:
        zoo = PINS["zooFixtures"]
        print(fetch_repo(zoo["repository"], zoo["revision"], zoo["files"]))
    sys.stdout.flush()


if __name__ == "__main__":
    main()

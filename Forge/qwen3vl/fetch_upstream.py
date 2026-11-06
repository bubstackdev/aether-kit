#!/usr/bin/env python3
"""Fetches the pinned Qwen3-VL 2B sources: the checkpoint snapshot, the upstream Apache-2.0 text and the test photos.

    Forge/.venv/bin/python -I Forge/qwen3vl/fetch_upstream.py [--output DIR] [--no-snapshot]

Checkpoint (Qwen/Qwen3-VL-2B-Instruct, Apache-2.0, not gated): downloaded into the Hugging Face cache at the pinned
revision, only the files `source-pins.json` lists, each verified by SHA-256. The model code is transformers' Qwen3-VL at
the version Forge/uv.lock pins, so there is no upstream code to unpack.

Into the output directory (default ~/.cache/aether-kit/qwen3vl): `LICENSE` (QwenLM/Qwen3-VL's Apache-2.0 text at the
pinned commit, which the bundle ships) and `images/` (the three Wikimedia Commons photographs, CC0 or public domain,
the reference cases' inputs), each verified. Idempotent: matching files are left alone.
"""
import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "source-pins.json").read_text())
DEFAULT = Path.home() / ".cache/aether-kit/qwen3vl"
USER_AGENT = "aether-kit-forge/0.1 (https://github.com/bubstackdev/aether-kit)"   # Wikimedia requires one


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, target: Path, expected: str) -> None:
    if target.is_file() and sha256(target) == expected:
        return
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected:
        raise SystemExit(f"{url} has SHA-256 {digest}, not the pinned {expected}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def fetch_extras(output: Path) -> None:
    lic = PINS["upstreamLicense"]
    raw = f"https://raw.githubusercontent.com/{lic['repository']}/{lic['commit']}/"
    for relative, expected in lic["files"].items():
        fetch(raw + relative, output / relative, expected)
    for name, meta in PINS["referenceImages"]["files"].items():
        fetch(meta["url"], output / "images" / name, meta["sha256"])


def fetch_snapshot() -> Path:
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(PINS["repository"], revision=PINS["revision"], allow_patterns=list(PINS["files"])))
    for relative, expected in PINS["files"].items():
        if sha256(path / relative) != expected:
            raise SystemExit(f"{relative} in {path} differs from the pinned {PINS['revision'][:12]} file")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT)
    ap.add_argument("--no-snapshot", action="store_true", help="fetch only the license text and the photos")
    args = ap.parse_args()
    fetch_extras(args.output)
    print(args.output)
    if not args.no_snapshot:
        print(fetch_snapshot())
    sys.stdout.flush()


if __name__ == "__main__":
    main()

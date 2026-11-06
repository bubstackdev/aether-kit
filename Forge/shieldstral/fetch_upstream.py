#!/usr/bin/env python3
"""Fetches the pinned Shieldstral 1.0 3B snapshot (mistralai/Shieldstral-1.0-3B, public) into the Hugging Face cache.

    Forge/.venv/bin/python -I Forge/shieldstral/fetch_upstream.py

Downloads only the files `source-pins.json` lists, at the pinned revision (about 7.7 GB, almost all of it the bf16
`model.safetensors`; the duplicate Mistral-format `consolidated.safetensors` is skipped), and verifies each one's
SHA-256 and the checked-in Apache-2.0 text. There is no separate upstream code unpack: the decoder is transformers'
Ministral 3 code at the version Forge/uv.lock pins. Idempotent: a matching snapshot is left alone.
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "source-pins.json").read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_snapshot() -> Path:
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(PINS["repository"], revision=PINS["revision"], allow_patterns=list(PINS["files"])))
    for relative, expected in PINS["files"].items():
        if sha256(path / relative) != expected:
            raise SystemExit(f"{relative} in {path} differs from the pinned {PINS['revision'][:12]} file")
    for name, meta in PINS["terms"]["files"].items():
        if sha256(HERE / "terms" / name) != meta["sha256"]:
            raise SystemExit(f"terms/{name} differs from its pin")
    return path


def main() -> None:
    print(fetch_snapshot())
    sys.stdout.flush()


if __name__ == "__main__":
    main()

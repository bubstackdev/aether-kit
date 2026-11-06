#!/usr/bin/env python3
"""Fetches the pinned EmbeddingGemma 300M snapshot (google/embeddinggemma-300m, gated) into the Hugging Face cache.

    Forge/.venv/bin/python -I Forge/embeddinggemma/fetch_upstream.py

The repository is gated: accept the Gemma terms on its Hub page first; huggingface_hub then uses the token it already
has (`hf auth login`, or HF_TOKEN). Downloads only the files `source-pins.json` lists, at the pinned revision, and
verifies each one's SHA-256. There is no separate upstream code unpack: the encoder is transformers' Gemma 3 code at the
version Forge/uv.lock pins, and the Gemma terms the bundle carries are checked in under `terms/`. Idempotent: a matching
snapshot is left alone.
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "source-pins.json").read_text())


def fetch_snapshot() -> Path:
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(PINS["repository"], revision=PINS["revision"], allow_patterns=list(PINS["files"])))
    for relative, expected in PINS["files"].items():
        h = hashlib.sha256()
        with (path / relative).open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        if h.hexdigest() != expected:
            raise SystemExit(f"{relative} in {path} differs from the pinned {PINS['revision'][:12]} file")
    for name, meta in PINS["terms"]["files"].items():
        if hashlib.sha256((HERE / "terms" / name).read_bytes()).hexdigest() != meta["sha256"]:
            raise SystemExit(f"terms/{name} differs from its pin")
    return path


def main() -> None:
    print(fetch_snapshot())
    sys.stdout.flush()


if __name__ == "__main__":
    main()

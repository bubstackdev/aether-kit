#!/usr/bin/env python3
"""Fetches the pinned Depth Anything 3 sources: the upstream model code and example frames, and the DA3-SMALL snapshot.

    Forge/.venv/bin/python -I Forge/depth/fetch_upstream.py [--output DIR] [--no-snapshot]

Upstream code (ByteDance-Seed/Depth-Anything-3, Apache-2.0): only the files the harness imports (the network, its
DINOv2 backbone and DualDPT head, and the small helpers they import) plus LICENSE and the two example frames the
reference cases use, each fetched from the pinned commit and checked against `source-pins.json`. They are written under
the output directory (default: ~/.cache/aether-kit/depth/upstream/depth-anything-3-<commit12>) with an UNPACKED.json
stamp. The package is never installed: `depth_graphs.load_upstream` imports the model modules read-only, so none of
upstream's other dependencies (xformers, gradio, opencv, ...) is needed.

Checkpoint (depth-anything/DA3-SMALL, Apache-2.0): downloaded into the Hugging Face cache at the pinned revision, then
verified. Idempotent: an existing, matching unpack or snapshot is left alone.
"""
import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "source-pins.json").read_text())
CODE = PINS["upstreamCode"]
COMMIT = CODE["commit"]
DEFAULT = Path.home() / f".cache/aether-kit/depth/upstream/depth-anything-3-{COMMIT[:12]}"
RAW = f"https://raw.githubusercontent.com/{CODE['repository']}/{COMMIT}/"


def wanted() -> dict[str, str]:
    return {**CODE["files"], **PINS["referenceImages"]["files"]}


def fetch_code(output: Path) -> None:
    stamp = output / "UNPACKED.json"
    if stamp.exists() and json.loads(stamp.read_text()).get("files") == wanted():
        return
    for relative, expected in wanted().items():
        target = output / relative
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            continue
        with urllib.request.urlopen(RAW + relative, timeout=120) as response:
            data = response.read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            raise SystemExit(f"{relative} at {COMMIT[:12]} has SHA-256 {digest}, not the pinned {expected}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    stamp.write_text(json.dumps({"repository": CODE["repository"], "commit": COMMIT, "files": wanted()}, indent=1) + "\n")


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
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT)
    ap.add_argument("--no-snapshot", action="store_true", help="fetch only the upstream code and frames")
    args = ap.parse_args()
    fetch_code(args.output)
    print(args.output)
    if not args.no_snapshot:
        print(fetch_snapshot())
    sys.stdout.flush()


if __name__ == "__main__":
    main()

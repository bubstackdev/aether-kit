#!/usr/bin/env python3
"""Fetches the pinned RF-DETR sources: the upstream model code, the nano and seg-nano checkpoints and the reference photos.

    Forge/.venv/bin/python -I Forge/rfdetr/fetch_upstream.py [--output DIR] [--no-checkpoints] [--no-images]

Upstream code (roboflow/rf-detr, Apache-2.0, tag 1.12.0): only the files the harness imports or reads (the LW-DETR
network, its windowed DINOv2 backbone, deformable-attention decoder, segmentation head, postprocess, the checkpoint
loader and the small helpers they import, plus upstream's own Core AI decomposition table) and LICENSE, each fetched
from the pinned commit and checked against `source-pins.json`. They are written under the output directory (default:
~/.cache/aether-kit/rfdetr/upstream/rf-detr-<commit12>) with an UNPACKED.json stamp. The package is never installed:
`rfdetr_graphs.load_upstream` imports these modules read-only, so none of upstream's other dependencies (supervision,
pyDeprecate, ...) is needed.

Checkpoints (Roboflow's own URLs from upstream's `assets/model_weights.py`, Apache-2.0) go to
~/.cache/aether-kit/rfdetr/weights under upstream's file names; reference photos (Wikimedia Commons, CC0) to
~/.cache/aether-kit/rfdetr/images. Every file is verified against its pinned SHA-256. Idempotent: matching files are
left alone.
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
CACHE = Path.home() / ".cache/aether-kit/rfdetr"
DEFAULT = CACHE / f"upstream/rf-detr-{COMMIT[:12]}"
WEIGHTS, IMAGES = CACHE / "weights", CACHE / "images"
RAW = f"https://raw.githubusercontent.com/{CODE['repository']}/{COMMIT}/"
USER_AGENT = "aether-kit-forge/0.1 (https://github.com/bubstackdev/aether-kit)"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, target: Path, expected: str) -> None:
    """Streams `url` to `target` through a `.part` file and keeps it only if its SHA-256 is `expected`."""
    if target.is_file() and sha256_file(target) == expected:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    h = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response, part.open("wb") as out:
        for chunk in iter(lambda: response.read(1 << 22), b""):
            h.update(chunk)
            out.write(chunk)
    if h.hexdigest() != expected:
        part.unlink()
        raise SystemExit(f"{url} has SHA-256 {h.hexdigest()}, not the pinned {expected}")
    part.rename(target)


def fetch_code(output: Path) -> None:
    stamp = output / "UNPACKED.json"
    if stamp.exists() and json.loads(stamp.read_text()).get("files") == CODE["files"]:
        return
    for relative, expected in CODE["files"].items():
        download(RAW + relative, output / relative, expected)
    stamp.write_text(json.dumps({"repository": CODE["repository"], "commit": COMMIT, "files": CODE["files"]},
                                indent=1) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT)
    ap.add_argument("--no-checkpoints", action="store_true")
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()
    fetch_code(args.output)
    print(args.output)
    if not args.no_checkpoints:
        for pin in PINS["checkpoints"].values():
            download(pin["url"], WEIGHTS / pin["file"], pin["sha256"])
        print(WEIGHTS)
    if not args.no_images:
        for name, pin in PINS["referenceImages"]["files"].items():
            download(pin["url"], IMAGES / name, pin["sha256"])
        print(IMAGES)
    sys.stdout.flush()


if __name__ == "__main__":
    main()

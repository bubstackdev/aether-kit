#!/usr/bin/env python3
"""Fetches the pinned YOLOX sources: the upstream model code and demo images, and the yolox_s.pth release checkpoint.

    Forge/.venv/bin/python -I Forge/yolox/fetch_upstream.py [--output DIR] [--weights FILE] [--no-weights]

Upstream code (Megvii-BaseDetection/YOLOX, Apache-2.0): only the files the harness imports or reads (the network, the
experiment recipe, the COCO class names, the letterbox `preproc` it documents, the two box helpers the head imports)
plus LICENSE and the two demo images the reference cases use, each fetched from the pinned commit and checked against
`source-pins.json`. They are written under the output directory (default:
~/.cache/aether-kit/yolox/upstream/yolox-<commit12>) with an UNPACKED.json stamp. The package is never installed:
`yolox_graphs.load_upstream` imports the model modules read-only, so none of upstream's other dependencies (opencv,
loguru, pycocotools, tabulate, ...) is needed.

Checkpoint (the 0.1.1rc0 GitHub release asset yolox_s.pth, Apache-2.0): downloaded to
~/.cache/aether-kit/yolox/weights/0.1.1rc0/yolox_s.pth, then verified. Idempotent: an existing, matching unpack or
checkpoint is left alone.
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
WEIGHTS = PINS["weights"]
DEFAULT = Path.home() / f".cache/aether-kit/yolox/upstream/yolox-{COMMIT[:12]}"
DEFAULT_WEIGHTS = Path.home() / f".cache/aether-kit/yolox/weights/{WEIGHTS['release']}/{WEIGHTS['file']}"
RAW = f"https://raw.githubusercontent.com/{CODE['repository']}/{COMMIT}/"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def wanted() -> dict[str, str]:
    return {**CODE["files"], **PINS["referenceImages"]["files"]}


def fetch_code(output: Path) -> None:
    stamp = output / "UNPACKED.json"
    if stamp.exists() and json.loads(stamp.read_text()).get("files") == wanted():
        return
    for relative, expected in wanted().items():
        target = output / relative
        if target.is_file() and sha256_file(target) == expected:
            continue
        with urllib.request.urlopen(RAW + relative, timeout=120) as response:
            data = response.read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            raise SystemExit(f"{relative} at {COMMIT[:12]} has SHA-256 {digest}, not the pinned {expected}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    stamp.write_text(json.dumps({"repository": CODE["repository"], "commit": COMMIT, "files": wanted()},
                                indent=1) + "\n")


def fetch_weights(target: Path) -> Path:
    if target.is_file() and sha256_file(target) == WEIGHTS["sha256"]:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial")
    with urllib.request.urlopen(WEIGHTS["url"], timeout=600) as response, partial.open("wb") as f:
        for chunk in iter(lambda: response.read(1 << 22), b""):
            f.write(chunk)
    digest = sha256_file(partial)
    if digest != WEIGHTS["sha256"]:
        partial.unlink()
        raise SystemExit(f"{WEIGHTS['url']} has SHA-256 {digest}, not the pinned {WEIGHTS['sha256']}")
    partial.rename(target)
    return target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--no-weights", action="store_true", help="fetch only the upstream code and images")
    args = ap.parse_args()
    fetch_code(args.output)
    print(args.output)
    if not args.no_weights:
        print(fetch_weights(args.weights))
    sys.stdout.flush()


if __name__ == "__main__":
    main()

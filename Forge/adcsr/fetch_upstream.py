#!/usr/bin/env python3
"""Fetches the pinned AdcSR sources: the authors' snapshot (code and weights), the SD 2.1 base U-Net config and license
text, and the reference photos.

    Forge/.venv/bin/python -I Forge/adcsr/fetch_upstream.py [--no-snapshot] [--no-images]

Snapshot (Guaishou74851/AdcSR, Apache-2.0 code; weights CreativeML Open RAIL++-M as SD 2.1 base derivatives):
LICENSE, README.md, the inference code test.py runs (model.py, forward.py, test.py) and the two checkpoints it loads
(weight/net_params_200.pkl, weight/pretrained/halfDecoder.ckpt), downloaded into the Hugging Face cache at the pinned
revision, then verified. Only these files: none of the training weights (osediff.pkl, DAPE.pth, RAM), datasets or
assets. The code is imported read-only from the snapshot (`adcsr_graphs.load_upstream`); nothing is installed.

SD 2.1 base (stabilityai/stable-diffusion-2-1-base is no longer on the Hub; source-pins.json `base`): only
`unet/config.json` (the U-Net's architecture; every weight comes from net_params_200.pkl) and `LICENSE-MODEL`, each from
its pinned mirror revision. No SD 2.1 weights are downloaded.

Reference photos (Wikimedia Commons, CC0) go to ~/.cache/aether-kit/adcsr/images. Every file is verified against its
pinned SHA-256. Idempotent: matching files are left alone.
"""
import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "source-pins.json").read_text())
IMAGES = Path.home() / ".cache/aether-kit/adcsr/images"
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


def snapshot(repository: str, revision: str, files: dict[str, str]) -> Path:
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(repository, revision=revision, allow_patterns=list(files)))
    for relative, expected in files.items():
        if sha256_file(path / relative) != expected:
            raise SystemExit(f"{relative} in {path} differs from the pinned {repository}@{revision[:12]} file")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-snapshot", action="store_true", help="skip the Hugging Face files (checkpoints, SD 2.1 base)")
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()
    if not args.no_snapshot:
        print(snapshot(PINS["repository"], PINS["revision"], PINS["files"]))
        base = PINS["base"]
        for part in ("config", "licenseText"):
            print(snapshot(base[part]["repository"], base[part]["revision"], base[part]["files"]))
    if not args.no_images:
        for name, pin in PINS["referenceImages"]["files"].items():
            download(pin["url"], IMAGES / name, pin["sha256"])
        print(IMAGES)
    sys.stdout.flush()


if __name__ == "__main__":
    main()

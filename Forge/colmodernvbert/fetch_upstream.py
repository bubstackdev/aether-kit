#!/usr/bin/env python3
"""Fetches the pinned ColModernVBERT sources: the checkpoint snapshot, the authors' MIT text and README, the reference
pages and the zoo bundle's sidecar references.

    Forge/.venv/bin/python -I Forge/colmodernvbert/fetch_upstream.py [--output DIR] [--no-snapshot]

Checkpoint (ModernVBERT/colmodernvbert-merged, MIT, not gated): downloaded into the Hugging Face cache at the pinned
revision, only the files `source-pins.json` lists, each verified by SHA-256. The model code is transformers'
ModernVBert / ColModernVBert at the version Forge/uv.lock pins, so no upstream code is unpacked.

Into the output directory (default ~/.cache/aether-kit/colmodernvbert), each verified:
    upstream/   illuin-tech/modernvbert at the pinned commit: LICENSE (bundled), README.md, assets/imgs/table.png
    images/     the Wikimedia Commons document pages (US government works, public domain)
    zoo/        the zoo bundle's reference_query.json, reference_doc.json and test_doc.png (a cross-check only)
Idempotent: matching files are left alone.
"""
import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "source-pins.json").read_text())
DEFAULT = Path.home() / ".cache/aether-kit/colmodernvbert"
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


def wanted() -> dict[str, tuple[str, str]]:
    """{path under the output directory: (url, sha256)}."""
    code, zoo = PINS["upstreamCode"], PINS["zooCrossCheck"]
    out = {}
    raw = f"https://raw.githubusercontent.com/{code['repository']}/{code['commit']}/"
    for relative, expected in code["files"].items():
        out[f"upstream/{relative}"] = (raw + relative, expected)
    for name, meta in PINS["referenceImages"]["files"].items():
        out[f"images/{name}"] = (meta["url"], meta["sha256"])
    hub = f"https://huggingface.co/{zoo['repository']}/resolve/{zoo['revision']}/"
    for relative, expected in zoo["files"].items():
        out[f"zoo/{relative}"] = (hub + relative, expected)
    return out


def fetch_extras(output: Path) -> None:
    for relative, (url, expected) in wanted().items():
        fetch(url, output / relative, expected)


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
    ap.add_argument("--no-snapshot", action="store_true", help="fetch only the license, pages and zoo references")
    args = ap.parse_args()
    fetch_extras(args.output)
    print(args.output)
    if not args.no_snapshot:
        print(fetch_snapshot())
    sys.stdout.flush()


if __name__ == "__main__":
    main()

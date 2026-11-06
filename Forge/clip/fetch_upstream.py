#!/usr/bin/env python3
"""Fetches the pinned CLIP ViT-B/32 sources: openai/CLIP's LICENSE and model card, the reference images, and the
openai/clip-vit-base-patch32 snapshot.

    Forge/.venv/bin/python -I Forge/clip/fetch_upstream.py [--output DIR] [--no-snapshot]

From openai/CLIP (MIT) only LICENSE (bundled) and model-card.md; no code is imported from it, since the model code is
transformers' `CLIPModel` at the version Forge/uv.lock pins. The reference images are scikit-image's sample images at
the pinned commit (public domain or CC0), used only as reference-case inputs. Each is fetched from its pinned commit and
checked against `source-pins.json`, under the output directory (default:
~/.cache/aether-kit/clip/upstream/clip-<commit12>, `upstream/` and `images/`) with an UNPACKED.json stamp.

Checkpoint (openai/clip-vit-base-patch32, MIT): downloaded into the Hugging Face cache at the pinned revision, then
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
CODE, IMAGES = PINS["upstreamCode"], PINS["referenceImages"]
COMMIT = CODE["commit"]
DEFAULT = Path.home() / f".cache/aether-kit/clip/upstream/clip-{COMMIT[:12]}"


def wanted() -> dict[str, tuple[str, str]]:
    """{local path: (url, sha256)}."""
    out = {}
    for prefix, group in (("upstream", CODE), ("images", IMAGES)):
        raw = f"https://raw.githubusercontent.com/{group['repository']}/{group['commit']}/"
        for relative, expected in group["files"].items():
            out[f"{prefix}/{relative}"] = (raw + relative, expected)
    return out


def fetch_code(output: Path) -> None:
    stamp = output / "UNPACKED.json"
    pins = {k: v[1] for k, v in wanted().items()}
    if stamp.exists() and json.loads(stamp.read_text()).get("files") == pins:
        return
    for relative, (url, expected) in wanted().items():
        target = output / relative
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            continue
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            raise SystemExit(f"{url} has SHA-256 {digest}, not the pinned {expected}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    stamp.write_text(json.dumps({"repository": CODE["repository"], "commit": COMMIT,
                                 "images": {"repository": IMAGES["repository"], "commit": IMAGES["commit"]},
                                 "files": pins}, indent=1) + "\n")


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
    ap.add_argument("--no-snapshot", action="store_true", help="fetch only the license, model card and images")
    args = ap.parse_args()
    fetch_code(args.output)
    print(args.output)
    if not args.no_snapshot:
        print(fetch_snapshot())
    sys.stdout.flush()


if __name__ == "__main__":
    main()

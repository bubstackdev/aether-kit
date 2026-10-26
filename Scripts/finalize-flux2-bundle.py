#!/usr/bin/env python3
"""Finalize a Forge FLUX.2 [klein] export into the bundle layout AetherFlux2 opens.

Forge's diffusion export writes `bundle/` (Core AI assets and tokenizer), `manifest-fragment.json`
and `apple-metadata.json`. The prototype's Swift finalizer is not part of AetherKit, so this script
adds the per-target fields AetherFlux2 validates and pins every bundle file by SHA-256:

    python3 Scripts/finalize-flux2-bundle.py EXPORT_DIR CHECKPOINT_DIR OUTPUT_BUNDLE

It streams every `main.mlirb` and refuses a stream whose hash differs from its `main.hash`
declaration. The checkpoint's LICENSE.md is copied into the bundle as provenance. OUTPUT_BUNDLE
must not exist; `EXPORT_DIR/bundle` is moved there (same volume) after the manifest is written.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(export: Path, checkpoint: Path, output: Path) -> None:
    bundle = export / "bundle"
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    fragment = json.loads((export / "manifest-fragment.json").read_text())
    apple = json.loads((export / "apple-metadata.json").read_text())
    license_text = checkpoint / "LICENSE.md"
    if not license_text.is_file():
        raise SystemExit("checkpoint LICENSE.md is missing")
    shutil.copy2(license_text, bundle / "LICENSE.md")

    files: dict[str, str] = {}
    for path in sorted(p for p in bundle.rglob("*") if p.is_file()):
        relative = path.relative_to(bundle).as_posix()
        if relative in ("aether.json", "metadata.json"):
            continue
        files[relative] = "sha256:" + sha256(path)
        if path.name == "main.mlirb":
            declared = (path.parent / "main.hash").read_bytes().hex()
            if files[relative] != "sha256:" + declared:
                raise SystemExit(f"{relative} differs from its main.hash declaration")

    manifest = dict(fragment)
    manifest["target"] = {"backend": "coreai", "platform": "macos", "arch": "any", "compiled": False,
                          "compute": "gpu", "minOS": "27.0"}
    manifest["text"] = {"tokenizer": "tokenizer/", "chatTemplate": "embedded"}
    diffusion = dict(manifest["diffusion"])
    # Per-asset placement (prototype F-P3-48/68): only the shape-dynamic text encoder and denoiser
    # specialize with frequent reshapes; the VAE decoder and encoder plan static memory. This is the
    # single source of placement: AetherFlux2 reads assetReshapes over the Forge fragment's
    # bundle-wide expectFrequentReshapes default.
    reshapes = {"vaeDecoder": False}
    if "vaeEncoder" in apple["assets"]:
        reshapes["vaeEncoder"] = False
    diffusion["assetReshapes"] = reshapes
    manifest["diffusion"] = diffusion
    provenance = dict(manifest["provenance"])
    provenance["files"] = files
    provenance["license"] = {"spdx": "Apache-2.0", "file": "LICENSE.md"}
    provenance["finalizer"] = "aether-kit Scripts/finalize-flux2-bundle.py"
    manifest["provenance"] = provenance

    (bundle / "aether.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (bundle / "metadata.json").write_text(json.dumps(apple, indent=2, sort_keys=True) + "\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle.rename(output)
    record = {"bundle": str(output), "assets": apple["assets"], "files": files,
              "aether.json": "sha256:" + sha256(output / "aether.json"),
              "metadata.json": "sha256:" + sha256(output / "metadata.json"),
              "bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file())}
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve())

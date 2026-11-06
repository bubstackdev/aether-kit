#!/usr/bin/env python3
"""The iOS variant manifest of a published JIT export: its macOS `.aether/aether.json` with `variant.platform: ios`.

The M4 exports (depth-anything-3-small, embeddinggemma-300m, clip-vit-b32, yolox-s, rf-detr-nano, rf-detr-seg-nano)
are JIT `.aimodel` packages with nothing per architecture, so one export serves macOS and iOS. Each exporter's
`--platform ios` writes this manifest too, but only as part of a fresh export, whose graph bytes are not
reproducible; this derives it from the export that was published and device-qualified, leaving every bundle file as
it is. The output is the form the device gate staged and ran (Tools/DeviceGate/run-on-device.sh): the same JSON, keys
sorted, two-space indent, trailing newline.

    python3 -I Forge/ios_manifest.py <bundle>/.aether/aether.json <out>.json

The output is uploaded as `.aether/aether.json` on the model repo's `ios` branch (branched from the macOS pin, so
every other file is the same blob), and the catalog's `ios` variant pins that commit as its main asset's revision
(docs/M4.md, "iOS variants"). Standard library only.
"""
import json
import sys


def ios_manifest(macos: dict) -> dict:
    variant = macos.get("variant")
    if variant != {"platform": "macos"}:
        raise SystemExit(f"expected variant {{'platform': 'macos'}} (a JIT export), found {variant!r}")
    manifest = json.loads(json.dumps(macos))
    manifest["variant"]["platform"] = "ios"
    return manifest


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: ios_manifest.py <macOS aether.json> <iOS aether.json>")
    with open(sys.argv[1], encoding="utf-8") as f:
        manifest = ios_manifest(json.load(f))
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()

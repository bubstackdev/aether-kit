#!/usr/bin/env python3
"""Writes the PNG fixtures for Kit's alpha policy and what upstream decodes from them (development tooling only).

    Forge/.venv/bin/python -I Forge/language/write_alpha_fixtures.py Tests/AetherVisionTests/Fixtures/alpha

Each fixture is a small PNG with transparent pixels whose stored colours differ from black, in the layouts an app may
pass: 8-bit RGBA, 8-bit grey plus alpha, a palette with a transparency table, and 16-bit RGBA. `expected.json` records,
for each, the bytes Transformers' `load_image` produces (`ImageOps.exif_transpose`, then `convert("RGB")`, which drops
alpha and keeps the stored colour), so the Swift CPU suite compares against upstream without running Python.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

W, H = 4, 3


def rgba_pixels() -> np.ndarray:
    """Distinct stored colours under alpha 0, 1, 64, 128, 254 and 255."""
    alphas = [0, 1, 64, 128, 254, 255]
    px = np.zeros((H, W, 4), dtype=np.uint8)
    for i in range(W * H):
        y, x = divmod(i, W)
        px[y, x] = [(37 * i + 11) % 256, (91 * i + 7) % 256, (53 * i + 200) % 256, alphas[i % len(alphas)]]
    return px


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    px = rgba_pixels()
    images = {
        "rgba8": Image.fromarray(px, "RGBA"),
        "gray-alpha8": Image.fromarray(np.stack([px[..., 0], px[..., 3]], -1), "LA"),
        "rgba16": None,
    }
    # A palette image whose second and fourth entries are transparent (tRNS), with non-black stored colours.
    palette = Image.new("P", (W, H))
    palette.putpalette([200, 30, 40, 10, 220, 90, 60, 70, 230, 250, 250, 10] + [0] * (256 * 3 - 12))
    palette.putdata([i % 4 for i in range(W * H)])
    palette.info["transparency"] = bytes([255, 0, 255, 0] + [255] * 252)
    images["palette-trns"] = palette
    record = {"kind": "vision-alpha-fixtures", "rule": "Transformers load_image: ImageOps.exif_transpose(image).convert('RGB')",
              "size": [W, H], "fixtures": {}}
    for name, image in images.items():
        path = out / f"{name}.png"
        if name == "rgba16":
            # 16-bit RGBA: high byte the 8-bit value, low byte different, so the 8-bit reduction is checked.
            import zlib, struct

            wide = px.astype(np.uint16) * 256 + (255 - px.astype(np.uint16))
            rows = b"".join(b"\x00" + wide[y].astype(">u2").tobytes() for y in range(H))
            def chunk(kind: bytes, data: bytes) -> bytes:
                return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 16, 6, 0, 0, 0))
                             + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
        else:
            image.save(path)
        decoded = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        record["fixtures"][name] = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                    "mode": Image.open(path).mode,
                                    "upstreamRGB": np.asarray(decoded).reshape(-1).tolist()}
    (out / "expected.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v["mode"] for k, v in record["fixtures"].items()}))


if __name__ == "__main__":
    main()

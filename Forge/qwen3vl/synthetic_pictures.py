#!/usr/bin/env python3
"""Fixed synthetic pictures for the Qwen3-VL describer gate (docs/VLM.md §6, G4), and the gate's cases.

    Forge/.venv/bin/python -I Forge/qwen3vl/synthetic_pictures.py --output <new dir>

Draws simple, non-graphic scenes (flat shapes, stripes, gradients, a ball moving across three frames) with numpy alone,
so every pixel is reproducible, and writes them as 8-bit RGB PNGs with `cases.json`: each case's instructions,
question, pictures (in order) and token limit, and every file's SHA-256.

Each picture is drawn at one of the five patch grids an earlier fixed-grid describer snapped every picture to
(44x44, 36x48, 48x36, 32x56 and 56x32 patches of 16 pixels): at those sizes the smart resize is the identity, so a
driver with a grid-static tower and one with fixed grids see exactly the same pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from pathlib import Path

import numpy as np

SIZES = {"square": (704, 704), "landscape": (576, 768), "portrait": (768, 576), "wide": (512, 896),
         "tall": (896, 512)}           # (height, width)


def canvas(size: str, rgb) -> np.ndarray:
    h, w = SIZES[size]
    out = np.empty((h, w, 3), np.float64)
    out[:] = rgb
    return out


def disc(img: np.ndarray, cy: float, cx: float, r: float, rgb) -> None:
    y, x = np.ogrid[:img.shape[0], :img.shape[1]]
    img[(y - cy) ** 2 + (x - cx) ** 2 <= r * r] = rgb


def rect(img: np.ndarray, y0: int, x0: int, y1: int, x1: int, rgb) -> None:
    img[y0:y1, x0:x1] = rgb


def triangle(img: np.ndarray, apex: tuple[float, float], left: tuple[float, float], right: tuple[float, float], rgb) -> None:
    y, x = np.mgrid[:img.shape[0], :img.shape[1]]

    def side(a, b):
        return (b[1] - a[1]) * (y - a[0]) - (b[0] - a[0]) * (x - a[1])
    s1, s2, s3 = side(apex, left), side(left, right), side(right, apex)
    inside = ((s1 >= 0) & (s2 >= 0) & (s3 >= 0)) | ((s1 <= 0) & (s2 <= 0) & (s3 <= 0))
    img[inside] = rgb


def house() -> np.ndarray:
    img = canvas("square", (135, 196, 235))
    rect(img, 470, 0, 704, 704, (76, 153, 60))                   # grass
    disc(img, 130, 560, 70, (250, 210, 55))                       # sun
    rect(img, 300, 200, 470, 420, (170, 110, 70))                 # walls
    triangle(img, (190, 310), (300, 180), (300, 440), (190, 40, 40))   # roof
    rect(img, 370, 285, 470, 335, (90, 55, 30))                   # door
    rect(img, 330, 360, 380, 405, (210, 235, 250))                # window
    return img


def stripes() -> np.ndarray:
    img = canvas("landscape", (0, 0, 0))
    colours = [(220, 40, 40), (240, 140, 30), (245, 220, 50), (60, 170, 70), (50, 110, 210), (130, 70, 180)]
    band = img.shape[0] / len(colours)
    for i, c in enumerate(colours):
        img[int(i * band):int((i + 1) * band)] = c
    return img


def checker() -> np.ndarray:
    img = canvas("portrait", (245, 245, 245))
    cell = 64
    y, x = np.mgrid[:img.shape[0], :img.shape[1]]
    img[((y // cell) + (x // cell)) % 2 == 1] = (30, 30, 30)
    disc(img, 384, 288, 120, (40, 90, 220))
    return img


def blocks() -> np.ndarray:
    img = canvas("wide", (250, 250, 250))
    for i, c in enumerate([(210, 40, 40), (40, 170, 70), (40, 80, 210)]):
        x0 = 110 + i * 250
        rect(img, 156, x0, 356, x0 + 200, c)
    return img


def balloons() -> np.ndarray:
    h, w = SIZES["tall"]
    t = np.linspace(0, 1, h)[:, None]
    img = np.empty((h, w, 3), np.float64)
    img[:] = ((1 - t) * np.array([250, 160, 60]) + t * np.array([90, 50, 140]))[:, None, :]
    for cy, cx, r, c in [(220, 160, 70, (230, 50, 60)), (330, 360, 80, (250, 230, 70)), (520, 220, 75, (70, 160, 230))]:
        disc(img, cy, cx, r, c)
        rect(img, int(cy + r), int(cx) - 1, int(cy + r + 160), int(cx) + 2, (240, 240, 240))
    return img


def frame(k: int) -> np.ndarray:
    img = canvas("wide", (150, 150, 150))
    rect(img, 400, 0, 512, 896, (110, 80, 50))                    # floor
    disc(img, 330, 170 + 280 * k, 70, (220, 40, 40))              # a red ball rolling right
    return img


PICTURES = {"house": house, "stripes": stripes, "checker": checker, "blocks": blocks, "balloons": balloons,
            "frame-1": lambda: frame(0), "frame-2": lambda: frame(1), "frame-3": lambda: frame(2)}

DESCRIBE = "You describe pictures accurately and briefly."
CASES = [
    {"name": "house", "instructions": DESCRIBE, "question": "Describe this picture in two sentences.",
     "pictures": ["house"], "maximumTokens": 96},
    {"name": "stripes", "instructions": DESCRIBE, "question": "List the colours you see, from top to bottom.",
     "pictures": ["stripes"], "maximumTokens": 96},
    {"name": "checker", "instructions": DESCRIBE, "question": "What is in the middle of this pattern?",
     "pictures": ["checker"], "maximumTokens": 96},
    {"name": "blocks", "instructions": "", "question": "How many squares are there, and what colours are they?",
     "pictures": ["blocks"], "maximumTokens": 96},
    {"name": "balloons", "instructions": DESCRIBE, "question": "Describe the scene.",
     "pictures": ["balloons"], "maximumTokens": 96},
    {"name": "frames", "instructions": DESCRIBE,
     "question": "These are three frames of a short clip, in order. What moves, and in which direction?",
     "pictures": ["frame-1", "frame-2", "frame-3"], "maximumTokens": 96},
    {"name": "pair", "instructions": DESCRIBE, "question": "Compare the two pictures in one or two sentences.",
     "pictures": ["house", "blocks"], "maximumTokens": 96},
    {"name": "text-only", "instructions": DESCRIBE, "question": "Name the three primary colours of paint.",
     "pictures": [], "maximumTokens": 48},
]


def png(path: Path, img: np.ndarray) -> None:
    """An 8-bit RGB PNG (no ancillary chunks), written by hand so the bytes depend on nothing but the pixels."""
    rgb = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return len(data).to_bytes(4, "big") + kind + data + zlib.crc32(kind + data).to_bytes(4, "big")
    header = w.to_bytes(4, "big") + h.to_bytes(4, "big") + bytes([8, 2, 0, 0, 0])
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9))
                     + chunk(b"IEND", b""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    args.output.mkdir(parents=True)
    files = {}
    for name, draw in PICTURES.items():
        path = args.output / f"{name}.png"
        png(path, draw())
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    cases = [{**c, "pictures": [f"{p}.png" for p in c["pictures"]]} for c in CASES]
    (args.output / "cases.json").write_text(json.dumps({"schema": 1, "files": files, "cases": cases}, indent=1) + "\n")
    print(json.dumps(files, indent=1))


if __name__ == "__main__":
    main()

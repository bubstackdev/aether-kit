"""Turn foreground-masked RGBA images (foreground_mask.swift) into TripoSR's 512×512 conditioning input.

usage: preprocess.py <out-dir> <masked.png>…
For each image: upstream's resize_foreground(0.85), composite over 0.5 grey, then the model's own resize to 512 (bilinear,
antialiased). Writes <stem>.npy ([1, 3, 512, 512] float32 in [0, 1], the export's input layout) and <stem>-input.png.
The eager reference and the Core AI run read the same .npy, so preprocessing never differs between them.

resize_foreground and the composite follow TripoSR's tsr/utils.py and run.py (VAST-AI-Research/TripoSR @ 107cefdc244c),
Copyright (c) 2024 Tripo AI & Stability AI, used under the MIT License; see LICENSE-TripoSR.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

FOREGROUND_RATIO = 0.85
SIZE = 512


def resize_foreground(image: np.ndarray, ratio: float) -> np.ndarray:
    alpha = np.where(image[..., 3] > 0)
    y1, y2, x1, x2 = alpha[0].min(), alpha[0].max(), alpha[1].min(), alpha[1].max()
    fg = image[y1:y2, x1:x2]
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    square = np.pad(fg, ((ph0, ph1), (pw0, pw1), (0, 0)))
    new_size = int(square.shape[0] / ratio)
    ph0 = pw0 = (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    return np.pad(square, ((ph0, ph1), (pw0, pw1), (0, 0)))


def main(out: Path, paths: list[Path]):
    out.mkdir(parents=True, exist_ok=True)
    for path in paths:
        rgba = np.array(Image.open(path).convert("RGBA"))
        image = resize_foreground(rgba, FOREGROUND_RATIO).astype(np.float32) / 255.0
        image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4]) * 0.5
        image = (image * 255.0).astype(np.uint8).astype(np.float32) / 255.0   # upstream round-trips through uint8
        tensor = F.interpolate(torch.from_numpy(image).permute(2, 0, 1)[None], (SIZE, SIZE), mode="bilinear",
                               align_corners=False, antialias=True)
        np.save(out / f"{path.stem}.npy", tensor.numpy().astype(np.float32))
        Image.fromarray((tensor[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)).save(
            out / f"{path.stem}-input.png")
        print(path.stem, tuple(tensor.shape))


if __name__ == "__main__":
    main(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]])

#!/usr/bin/env python3
"""Independent CPU reference for the Qwen3-VL vision encoder (development tooling only).

Upstream Transformers' `Qwen3VLVisionModel` is built from the checkpoint's vision config and loaded with its
`model.visual.*` tensors in float32 on the CPU. For deterministic synthetic images (seeded shapes on a flat
background), one per fixed patch grid at that grid's exact pixel size (no resize, so upstream and Kit see the same
pixels), it records the pixel tensor, the grid, the image
embeddings and the DeepStack features. It also checks this directory's eager tower (`qwen3_vl_vision.Qwen3VLVision`,
float32 and fp16-storage) against them, separating porting errors from Core AI conversion errors.

Outputs under the new `--output` directory: `reference.json`, and per image `<name>.png`, `<name>-pixels.f32`
([n, 1536]), `<name>-embeds.f32` ([n/4, 2048]) and `<name>-deepstack.f32` ([3, n/4, 2048]), all little-endian float32.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen3_vl_vision import IMAGE_GRIDS, Qwen3VLVision, host_inputs, load_weights, patchify  # noqa: E402

# One scene per fixed grid (h, w patches of 16 px): name, width, height, seed.
IMAGES = tuple((f"scene-{w * 16}x{h * 16}", w * 16, h * 16, 11 + i) for i, (h, w) in enumerate(IMAGE_GRIDS))
COLORS = ((220, 40, 40), (40, 80, 220), (40, 170, 70), (240, 200, 30), (140, 60, 180), (245, 140, 30))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def scene(width: int, height: int, seed: int):
    """Two or three filled shapes in distinct colours on white, from a seed."""
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    img = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(img)
    count = rng.choice((2, 3))
    col = width // count
    for i, rgb in enumerate(rng.sample(COLORS, count)):
        size = rng.randint(min(col, height) // 3, min(col, height) * 2 // 3)
        left, top = i * col + rng.randint(0, col - size), rng.randint(0, height - size)
        box = (left, top, left + size, top + size)
        shape = rng.choice(("circle", "square", "triangle"))
        if shape == "circle":
            d.ellipse(box, fill=rgb)
        elif shape == "square":
            d.rectangle(box, fill=rgb)
        else:
            d.polygon([((box[0] + box[2]) // 2, box[1]), (box[2], box[3]), (box[0], box[3])], fill=rgb)
    return img


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def upstream_tower(checkpoint: Path, config: dict):
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    vision_config = AutoConfig.from_pretrained(checkpoint).vision_config
    tower = Qwen3VLVisionModel._from_config(vision_config, torch_dtype=torch.float32).eval()
    state = {}
    for f in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(f, framework="pt") as st:
            for key in st.keys():
                if key.startswith("model.visual."):
                    state[key.removeprefix("model.visual.")] = st.get_tensor(key).float()
    missing, unexpected = tower.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"upstream tower load: missing {missing[:4]}, unexpected {unexpected[:4]}")
    return tower


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    config = json.loads((args.checkpoint / "config.json").read_text())
    tower = upstream_tower(args.checkpoint, config)
    ours32 = Qwen3VLVision(dict(config["vision_config"]))
    load_weights(ours32, args.checkpoint)
    ours32 = ours32.float().eval()
    ours16 = Qwen3VLVision(dict(config["vision_config"]))
    load_weights(ours16, args.checkpoint)
    ours16 = ours16.half().eval()

    args.output.mkdir(parents=True)
    records = []
    with torch.no_grad():
        for name, width, height, seed in IMAGES:
            img = scene(width, height, seed)
            img.save(args.output / f"{name}.png")
            pixels, grid = patchify(np.asarray(img))
            embeds, deepstack = tower(torch.from_numpy(pixels), grid_thw=torch.tensor([grid]))
            embeds, deepstack = embeds.numpy(), torch.stack(deepstack).numpy()
            tables = {k: torch.from_numpy(v) for k, v in host_inputs(*grid).items()}
            port = {}
            for label, module, dtype in (("float32", ours32, torch.float32), ("fp16Storage", ours16, torch.float16)):
                e, s = module(torch.from_numpy(pixels).to(dtype), tables["pos_index"], tables["pos_weight"], tables["patch_rc"])
                port[label] = {"imageEmbedsCosine": round(cosine(e.float().numpy(), embeds), 7),
                               "deepstackCosine": [round(cosine(s[i].float().numpy(), deepstack[i]), 7) for i in range(3)]}
            files = {}
            for suffix, array in (("pixels", pixels), ("embeds", embeds), ("deepstack", deepstack)):
                path = args.output / f"{name}-{suffix}.f32"
                array.astype("<f4").tofile(path)
                files[suffix] = {"file": path.name, "shape": list(array.shape), "sha256": sha256(path)}
            files["png"] = {"file": f"{name}.png", "sha256": sha256(args.output / f"{name}.png")}
            records.append({"name": name, "width": width, "height": height, "seed": seed, "grid": list(grid),
                            "patches": int(pixels.shape[0]), "tokens": int(embeds.shape[0]), "files": files,
                            "eagerPortVsReference": port})
    record = {
        "schema": 1,
        "kind": "qwen3-vl-vision-cpu-reference",
        "oracle": "transformers Qwen3VLVisionModel, float32, CPU, weights model.visual.*",
        "source": {"repository": args.source_repository, "revision": args.source_revision, "license": "apache-2.0"},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "pillow")},
        "scriptSHA256": {p.name: sha256(p) for p in sorted(Path(__file__).resolve().parent.glob("*.py"))},
        "checkpointSHA256": {p.name: sha256(p) for p in sorted(args.checkpoint.glob("*.safetensors"))},
        "preprocessing": "numpy patchify of an image at a fixed grid's exact pixel size: (x/255 - 0.5)/0.5, frame "
                         "repeated twice, rows in merge-window order (no resize)",
        "imageGrids": [list(g) for g in IMAGE_GRIDS],
        "images": records,
    }
    (args.output / "reference.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps([{r["name"]: r["eagerPortVsReference"]} for r in records], indent=1))


if __name__ == "__main__":
    main()

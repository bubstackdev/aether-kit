#!/usr/bin/env python3
"""Independent upstream reference for Qwen3.5 image + text prompts (development tooling only).

Runs in the Qwen3.5 reference environment (Forge/reference-qwen35), like reference_qwen35.py:

    Forge/reference-qwen35/.venv/bin/python -I Forge/language/reference_qwen35_image.py --checkpoint <snapshot> \
        --source-repository Qwen/Qwen3.5-4B --source-revision <sha> --license apache-2.0 --output <new dir> \
        --case orion=<image>

Upstream `Qwen3_5ForConditionalGeneration` and its own processor and chat template, float32 on the CPU. Each image
snaps to Kit's fixed grid nearest its aspect ratio (`IMAGE_GRIDS`, log space, first listed wins a tie) and the
processor resizes it to exactly that grid: its `smart_resize` is replaced by the grid's pixel size, as
reference_qwen3_vl.py does for Qwen3-VL. The prompt is the describer's whole-picture outfit reading (system
instructions, then the image and the request), rendered with `enable_thinking: false`.

Per case it records the rendered token ids, the processor's pixels [n, 1536] and grid, upstream's rope index, the
vision tower's image embeddings [n/4, 2560], a greedy continuation (`--greedy-tokens`, end-of-sequence ignored so the
run has a fixed length) and teacher-forced logits for the prompt's last row and the first `--forced-tokens`
continuation tokens. Outputs under the new `--output` directory: `reference.json` and per case `<name>-pixels.f32`,
`<name>-embeds.f32`, `forced-<name>.f32` (little-endian float32) and a copy of the image.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
IMAGE_GRIDS = ((44, 44), (36, 48), (48, 36), (32, 56), (56, 32))   # qwen3_vl_vision.IMAGE_GRIDS (patches h, w)
SLOTS = [
    ("top", "shirt, blouse, corset, bra, top, or a dress or bodysuit"),
    ("outer", "jacket, coat, cape, vest"),
    ("bottom", "skirt, shorts, trousers"),
    ("waist", "belt, sash, buckle, waist cincher"),
    ("arms", "gloves, sleeves worn separately, bracers, bracelets"),
    ("neck", "scarves, collars, chokers, necklaces, ties"),
    ("headwear", "hats, hoods, caps, masks, goggles, hair accessories"),
    ("carried", "anything held or hanging from the belt or hip: bags, coiled ropes, weapons"),
    ("legwear", "stockings, tights, leggings, socks"),
    ("footwear", "the shoes or boots on the feet"),
]
KEYS = ", ".join(f'"{k}" ({h})' for k, h in SLOTS)
INSTRUCTIONS = (
    "You list the outfit in a picture for an image generator, one body region at a time. Answer with only a JSON "
    f"object with these keys: {KEYS}. Each value is a list of 0 to 2 short phrases, at most 7 words each, naming "
    "a garment or accessory worn or carried there by its own name (corset, gloves, cape, goggles, lasso), with its "
    "colors, material or pattern and cut, like \"red leather corset with gold trim\". Mention an emblem or trim "
    "only on the garment that has it. Use [] when that region shows nothing worn or isn't in the picture. Name each "
    "garment once, under the region where it is worn; carried means part of the costume, not a prop. The person wearing it "
    "doesn't matter: never mention their face, hair, eyes, skin, body, shape, age, pose or expression, and never "
    "name them, a character or a franchise. Only what the picture clearly shows."
)
REQUEST = "This picture is the whole outfit. List the outfit in it as JSON, and nothing about the person wearing it."


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def environment() -> dict:
    project = Path(sys.prefix).resolve().parent
    lock = project / "uv.lock"
    return {"project": str(project.relative_to(ROOT)) if project.is_relative_to(ROOT) else str(project),
            "uvLockSHA256": sha256(lock) if lock.exists() else None, "python": sys.version.split()[0]}


def nearest_grid(width: int, height: int) -> tuple[int, int]:
    aspect, best, distance = math.log(width / height), IMAGE_GRIDS[0], math.inf
    for g in IMAGE_GRIDS:
        d = abs(aspect - math.log(g[1] / g[0]))
        if d < distance:
            best, distance = g, d
    return best


def save(path: Path, array: np.ndarray) -> dict:
    array.astype("<f4").tofile(path)
    return {"file": path.name, "dtype": "float32-le", "shape": list(array.shape), "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True, help="name=<image path>")
    parser.add_argument("--greedy-tokens", type=int, default=128)
    parser.add_argument("--forced-tokens", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    import transformers.models.qwen2_vl.image_processing_qwen2_vl as qwen2_vl_images
    from transformers import AutoModelForImageTextToText, AutoProcessor

    config = json.loads((args.checkpoint / "config.json").read_text())
    patch = int(config["vision_config"]["patch_size"])
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    model = AutoModelForImageTextToText.from_pretrained(args.checkpoint, dtype=torch.float32).eval()
    args.output.mkdir(parents=True)
    started, cases = time.time(), []
    for spec in args.case:
        name, image_path = spec.split("=", 1)
        image_path = Path(image_path)
        copied = args.output / f"{name}{image_path.suffix.lower()}"
        shutil.copyfile(image_path, copied)
        image = Image.open(image_path).convert("RGB")
        grid = nearest_grid(*image.size)
        qwen2_vl_images.smart_resize = lambda *a, **k: (grid[0] * patch, grid[1] * patch)
        messages = [{"role": "system", "content": [{"type": "text", "text": INSTRUCTIONS}]},
                    {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": REQUEST}]}]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt", enable_thinking=False)
        ids = inputs["input_ids"][0].tolist()
        thw = inputs["image_grid_thw"].tolist()
        if [tuple(g[1:]) for g in thw] != [grid]:
            raise SystemExit(f"{name}: processor grid {thw} is not the snapped grid {grid}")
        pixels = inputs["pixel_values"].float()
        with torch.no_grad():
            embeds = model.get_image_features(pixels, inputs["image_grid_thw"]).pooler_output[0].double().numpy()
            rope, _ = model.model.get_rope_index(inputs["input_ids"], inputs["mm_token_type_ids"], inputs["image_grid_thw"])
            out = model.generate(**inputs, max_new_tokens=args.greedy_tokens, do_sample=False, temperature=None,
                                 top_p=None, top_k=None, eos_token_id=None, pad_token_id=processor.tokenizer.pad_token_id)
            greedy = out[0, len(ids):].tolist()
            print(f"[reference] {name}: {len(ids)} prompt tokens, grid {thw[0]}, greedy "
                  f"{processor.decode(greedy[:40])!r} ({time.time() - started:.0f} s)", flush=True)
            forced = greedy[:args.forced_tokens]
            full = dict(inputs)
            full["input_ids"] = torch.tensor([ids + forced])
            full["attention_mask"] = torch.ones_like(full["input_ids"])
            full["mm_token_type_ids"] = torch.cat((inputs["mm_token_type_ids"],
                                                   torch.zeros(1, len(forced), dtype=inputs["mm_token_type_ids"].dtype)), 1)
            rows = list(range(len(ids) - 1, len(ids) + len(forced) - 1))
            reference = model(**full, logits_to_keep=torch.tensor(rows)).logits[0].double().numpy()
        agree = sum(int(np.argmax(r)) == t for r, t in zip(reference, forced))
        print(f"[reference] {name}: forced rows argmax = greedy on {agree}/{len(forced)} ({time.time() - started:.0f} s)",
              flush=True)
        cases.append({
            "name": name, "image": {"file": copied.name, "sha256": sha256(copied), "size": list(image.size),
                                    "source": str(image_path)},
            "grid": thw[0], "promptTokens": ids, "imageTokens": int(np.prod(thw[0]) // 4),
            "ropePositions": rope[:, 0].tolist(), "greedyTokens": greedy, "greedyText": processor.decode(greedy),
            "forcedTokens": forced,
            "pixels": save(args.output / f"{name}-pixels.f32", pixels.numpy()),
            "imageEmbeds": save(args.output / f"{name}-embeds.f32", embeds),
            "forcedLogits": {**save(args.output / f"forced-{name}.f32", reference),
                             "rowMeaning": "row r = logits after prompt + forcedTokens[:r]",
                             "argmaxEqualsGreedy": f"{agree}/{len(forced)}"},
        })
    record = {
        "schema": 1,
        "kind": "qwen35-image-reference",
        "oracle": "transformers Qwen3_5ForConditionalGeneration + AutoProcessor, float32, CPU",
        "environment": environment(),
        "source": {"repository": args.source_repository, "revision": args.source_revision, "license": args.license},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "pillow")},
        "prompt": {"system": INSTRUCTIONS, "user": REQUEST, "enableThinking": False},
        "imageGrids": [list(g) for g in IMAGE_GRIDS],
        "gridSnapping": "nearest aspect ratio in log space, first listed wins a tie; smart_resize replaced by the grid's pixel size",
        "scriptSHA256": {Path(__file__).name: sha256(Path(__file__).resolve())},
        "checkpointSHA256": {p.name: sha256(p) for p in sorted(args.checkpoint.iterdir()) if p.is_file()},
        "cases": cases,
        "seconds": round(time.time() - started, 1),
    }
    (args.output / "reference.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()

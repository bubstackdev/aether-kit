#!/usr/bin/env python3
"""Upstream-PyTorch Qwen3-VL 2B references for the Core AI gate (check_qwen3vl.py) and the Swift driver gate.

    Forge/.venv/bin/python -I Forge/qwen3vl/reference_qwen3vl.py --output <new reference dir>

Runs transformers' unmodified Qwen3VLForConditionalGeneration (pinned weights, SDPA attention) on the CPU in float32,
with upstream's own processor and chat template. The processor's pixel budget is the bundle's (max_pixels =
PATCH_CAPACITY * 256, min_pixels its default 65536), so upstream sizes each image exactly as the bundle's driver must
(smart resize); the decoded image is upstream's own (`load_image`-style PIL decode, `convert("RGB")`). For each case:

    <case>/case.json              image, sizes, grid, prompt, token counts, image start, rope shift, greedy text,
                                  per-step top-2 margins, SHA-256 of every file
    <case>/pixel_values.f32       the processor's patches [h*w, 1536] (block-major, C,T,py,px), unpadded
    <case>/input_ids.i32          the processor's prompt [1, L] (one <|image_pad|> per image token)
    <case>/image_embeds.f32       upstream's tower output [h*w/4, H]
    <case>/deepstack_embeds.f32   upstream's DeepStack features [3, h*w/4, H]
    <case>/rope_positions.i32     upstream's get_rope_index for the prompt [3, L] (the host M-RoPE contract)
    <case>/greedy.i32             upstream's greedy continuation [K] (K = STEPS, EOS not stopping it; its first
                                  CROSSCHECK tokens equal upstream's own generate())
    <case>/logits.f32             upstream's logits for each greedy step [K, 151936]; row 0 is the first token's

and an index `references.json`. Cases: four images with different patch grids (portrait, wide, 4:3, small square) and
a text-only prompt. The pixel tensors, not the image files, are what the graph gate feeds; preprocessing is gated
separately (the Swift side against `pixel_values.f32`). CPU only; a few minutes and about 10 GB.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qwen3vl_graphs as qg  # noqa: E402

IMAGES = Path.home() / ".cache/aether-kit/qwen3vl/images"
STEPS = 48
CROSSCHECK = 12              # tokens of upstream generate() that must equal the step loop

# (name, image file or None, crop/resize applied with Pillow before the processor, question)
CASES = [
    ("market-4x3", "photo-fruit-market-4x3.jpg", None,
     "Describe this photo in detail: the place, the people, and what is being sold."),
    ("dog-portrait", "photo-dog-portrait.jpg", None,
     "Describe the dog in this picture in detail, including its pose, colours and the background."),
    ("stall-wide", "photo-fruit-stall-wide.jpg", None,
     "List the fruits you can see on this stall and describe how they are displayed."),
    ("market-small-square", "photo-fruit-market-4x3.jpg", {"centerCropSquare": True, "resize": [256, 256]},
     "Describe this picture in detail."),
    ("text-only", None, None,
     "Explain in a short paragraph why the sky looks blue during the day and red at sunset."),
]


def load_image(name: str, edit: dict | None):
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(Image.open(IMAGES / name)).convert("RGB")
    if edit:
        if edit.get("centerCropSquare"):
            side = min(image.size)
            left, top = (image.width - side) // 2, (image.height - side) // 2
            image = image.crop((left, top, left + side, top + side))
        if "resize" in edit:
            image = image.resize(tuple(edit["resize"]), Image.Resampling.BICUBIC)
    return image


def greedy(model, inputs, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Upstream's own forward, step by step with its KV cache: the prompt (image included), then `steps` greedy tokens
    fed back one at a time (decode positions from upstream's stored rope deltas, as `generate` computes them). EOS
    does not stop it, so every case has `steps` comparable rows. Returns tokens int32 [steps], logits f32 [steps, V]."""
    model.model.rope_deltas = None                          # upstream caches it on the module; never across cases
    out = model(**inputs, use_cache=True)
    past, row = out.past_key_values, out.logits[0, -1].float()
    tokens, rows = [], []
    for _ in range(steps):
        rows.append(row.numpy().copy())
        token = int(row.argmax())
        tokens.append(token)
        out = model(input_ids=torch.tensor([[token]]), past_key_values=past, use_cache=True)
        past, row = out.past_key_values, out.logits[0, -1].float()
    model.model.rope_deltas = None
    return np.array(tokens, dtype=np.int32), np.stack(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    qg.pins_argument(ap)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--snapshot", type=Path)
    ap.add_argument("--cases", help="comma-separated subset")
    args = ap.parse_args()
    qg.use_pins(args.pins)
    args.snapshot = args.snapshot or qg.DEFAULT_SNAPSHOT
    if "referenceImages" not in qg.PINS:
        raise SystemExit("these pins carry no reference photographs (the 2B pins do)")
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    verified = qg.verify_sources(args.snapshot)
    for name, meta in qg.PINS["referenceImages"]["files"].items():
        if qg.sha256_file(IMAGES / name) != meta["sha256"]:
            raise SystemExit(f"{name} differs from its pin; run Forge/qwen3vl/fetch_upstream.py")
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    started = time.monotonic()
    model = qg.upstream_model(args.snapshot)
    proc = qg.processor(args.snapshot)
    vocab = model.config.text_config.vocab_size
    args.output.mkdir(parents=True)
    index = {"schema": 1, "repository": qg.REPO, "revision": qg.REVISION, "verified": verified,
             "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "tokenizers", "pillow",
                                                                     "numpy")},
             "host": {"machine": platform.machine(), "python": platform.python_version()},
             "dtype": "float32", "device": "cpu", "attention": "sdpa", "steps": STEPS,
             "patchCapacity": qg.PATCH_CAPACITY, "maxPixels": qg.PATCH_CAPACITY * 256, "vocab": vocab, "cases": []}
    wanted = set(args.cases.split(",")) if args.cases else None
    for name, image_name, edit, question in CASES:
        if wanted and name not in wanted:
            continue
        t0 = time.monotonic()
        folder = args.output / name
        folder.mkdir()
        content = ([{"type": "image"}] if image_name else []) + [{"type": "text", "text": question}]
        text = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                        tokenize=False)
        case = {"name": name, "question": question, "prompt": text}
        if image_name:
            image = load_image(image_name, edit)
            inputs = proc(text=[text], images=[image], return_tensors="pt",
                          images_kwargs={"size": {"longest_edge": qg.PATCH_CAPACITY * 256,
                                                   "shortest_edge": 65536}})
            grid = [int(v) for v in inputs["image_grid_thw"][0]]
            ids = inputs["input_ids"][0].numpy()
            pads = np.nonzero(ids == qg.IMAGE_PAD_ID)[0]
            start = int(pads[0])
            ss, sa = qg.rope_shift(start, grid[1], grid[2])
            case.update({"image": image_name, "edit": edit, "decodedSize": list(load_image(image_name, None).size),
                         "imageSize": list(image.size), "resized": [grid[2] * 16, grid[1] * 16],
                         "gridTHW": grid, "patches": grid[1] * grid[2], "imageTokens": len(pads),
                         "imageStart": start, "ropeShiftStart": ss, "ropeShiftAmount": sa})
            pv = inputs["pixel_values"].numpy().astype(np.float32)
            qg.write_f32(folder / "pixel_values.f32", pv)
            vis = model.model.visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
            qg.write_f32(folder / "image_embeds.f32", vis.pooler_output.numpy())
            qg.write_f32(folder / "deepstack_embeds.f32", torch.stack(list(vis.deepstack_features)).numpy())
            rope, _ = model.model.get_rope_index(inputs["input_ids"], inputs["mm_token_type_ids"],
                                                 image_grid_thw=inputs["image_grid_thw"])
            qg.write_i32(folder / "rope_positions.i32", rope[:, 0].numpy())
        else:
            inputs = proc(text=[text], return_tensors="pt")
            ids = inputs["input_ids"][0].numpy()
            case.update({"image": None, "imageTokens": 0})
        qg.write_i32(folder / "input_ids.i32", ids[None])
        case["promptTokens"] = int(len(ids))
        tokens, logits = greedy(model, inputs, STEPS)
        model.model.rope_deltas = None
        check = model.generate(**inputs, max_new_tokens=CROSSCHECK, do_sample=False, top_p=None, top_k=None,
                               temperature=None)[0, len(ids):].numpy()
        model.model.rope_deltas = None
        if not np.array_equal(check, tokens[:len(check)]):
            raise SystemExit(f"{name}: the step loop disagrees with upstream's generate: {check} vs {tokens[:16]}")
        case["generateCrossCheckTokens"] = int(len(check))
        qg.write_i32(folder / "greedy.i32", tokens)
        qg.write_f32(folder / "logits.f32", logits)
        top2 = np.sort(logits, axis=-1)[:, -2:]
        probs = torch.softmax(torch.from_numpy(logits).double(), -1).numpy()
        ptop = np.sort(probs, axis=-1)[:, -2:]
        eos = [i for i, t in enumerate(tokens) if int(t) in (151645, 151643)]
        case.update({"greedy": tokens.tolist(), "greedyText": proc.decode(tokens),
                     "firstEOS": eos[0] if eos else None,
                     "logitMargins": np.round(top2[:, 1] - top2[:, 0], 5).tolist(),
                     "probabilityMargins": np.round(ptop[:, 1] - ptop[:, 0], 5).tolist(),
                     "seconds": round(time.monotonic() - t0, 1)})
        case["files"] = {p.name: qg.sha256_file(p) for p in sorted(folder.iterdir())}
        (folder / "case.json").write_text(json.dumps(case, indent=1) + "\n")
        index["cases"].append({"name": name, "grid": case.get("gridTHW"), "promptTokens": case["promptTokens"],
                               "imageTokens": case["imageTokens"], "firstEOS": case["firstEOS"]})
        print(f"[reference] {name}: grid {case.get('gridTHW')} prompt {len(ids)} tokens, {case['seconds']} s: "
              f"{case['greedyText'][:100]!r}", file=sys.stderr, flush=True)
    index["seconds"] = round(time.monotonic() - started, 1)
    index["peakRSSGB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
    index["scriptSHA256"] = {p.name: qg.sha256_file(p) for p in sorted(HERE.glob("*.py"))}
    (args.output / "references.json").write_text(json.dumps(index, indent=1) + "\n")


if __name__ == "__main__":
    main()

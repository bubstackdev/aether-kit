#!/usr/bin/env python3
"""Independent CPU reference for Qwen3-VL image + text generation (development tooling only).

Upstream Transformers' `Qwen3VLForConditionalGeneration` (float32, CPU) with its own processor and chat template. For
each case (an image and a question) it records the rendered token ids, the processor's pixel tensor and grid, the
model's own 3-axis rope positions, a greedy continuation (`generate`, no sampling) and teacher-forced logits for the
prompt's last row and each forced continuation token.

With `--grids` (Kit's fixed-grid encoder), each image snaps to the grid nearest its aspect ratio (`nearest_grid`) and
the processor resizes it to exactly that grid: its `smart_resize` is replaced by the grid's pixel size for the case, so
upstream receives the same bucket-resized image Kit encodes.

It also runs this directory's eager port (vision tower + bucketed text tower, float32) on the same pixels and ids:
prefill through `withImage`, forced decode through `main`, with host M-RoPE positions (`mrope_positions`), and checks
the positions and logits against the reference.

Outputs under the new `--output` directory: `reference.json`, and per case `<name>-pixels.f32` [n, 1536] and
`<name>-forced-logits.f32` [rows, vocab] (little-endian float32), plus the input image as given.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen3_vl_text import Qwen3VLText, load_weights as load_text, mrope_positions  # noqa: E402
from qwen3_vl_vision import Qwen3VLVision, host_inputs, load_weights as load_vision, nearest_grid  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--case", action="append", required=True, metavar="NAME=IMAGE::QUESTION",
                        help="a case: name, image path and question")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--license", required=True, help="the checkpoint's model-card licence, e.g. apache-2.0")
    parser.add_argument("--base-model", help="the upstream model a fine-tune derives from")
    parser.add_argument("--greedy-tokens", type=int, default=48)
    parser.add_argument("--forced-tokens", type=int, default=32)
    parser.add_argument("--max-pixels", type=int, default=None,
                        help="processor pixel cap (size.longest_edge), mirroring Kit's maximumPixels")
    parser.add_argument("--exif-transpose", action="store_true",
                        help="decode as Transformers' load_image does: ImageOps.exif_transpose, then convert('RGB')")
    parser.add_argument("--grids", default=None,
                        help="fixed patch grids HxW, comma-separated: snap each image to the nearest aspect grid")
    args = parser.parse_args()
    grids = [tuple(int(v) for v in g.split("x")) for g in args.grids.split(",")] if args.grids else None
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    from PIL import Image, ImageOps
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    config = json.loads((args.checkpoint / "config.json").read_text())
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    if args.max_pixels:
        size = dict(processor.image_processor.size)
        size["longest_edge"] = args.max_pixels
        processor.image_processor.size = size
        processor.image_processor.max_pixels = args.max_pixels
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.checkpoint, torch_dtype=torch.float32).eval()
    image_token, merge = int(config["image_token_id"]), int(config["vision_config"]["spatial_merge_size"])

    # Two passes, so the float32 upstream model and the float32 eager port are never resident together (at 4B the
    # pair needs about 36 GB): upstream for every case first, then the port on the same inputs.
    args.output.mkdir(parents=True)
    cases, pending = [], []
    for spec in args.case:
        name, rest = spec.split("=", 1)
        image_path, question = rest.split("::", 1)
        image_path = Path(image_path)
        copied = args.output / f"{name}{image_path.suffix.lower()}"
        shutil.copyfile(image_path, copied)
        image = Image.open(image_path)
        image = (ImageOps.exif_transpose(image) if args.exif_transpose else image).convert("RGB")
        snapped = None
        if grids:
            import transformers.models.qwen2_vl.image_processing_qwen2_vl_fast as fast

            snapped = nearest_grid(image.size[0], image.size[1], grids)
            patch = int(config["vision_config"]["patch_size"])
            fast.smart_resize = lambda *a, _g=snapped, **k: (_g[0] * patch, _g[1] * patch)
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt")
        ids = inputs["input_ids"][0].tolist()
        grid = inputs["image_grid_thw"][0].tolist()
        if snapped and tuple(grid[1:]) != snapped:
            raise SystemExit(f"{name}: processor grid {grid} is not the snapped grid {snapped}")
        pixels = inputs["pixel_values"].float()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.greedy_tokens, do_sample=False, temperature=None,
                                 top_p=None, top_k=None)
            greedy = out[0, len(ids):].tolist()
            forced = greedy[:args.forced_tokens]
            full = {k: v for k, v in inputs.items()}
            full["input_ids"] = torch.tensor([ids + forced])
            full["attention_mask"] = torch.ones_like(full["input_ids"])
            logits = model(**full).logits[0].double().numpy()
            reference_rows = logits[len(ids) - 1:len(ids) + len(forced) - 1]
            rope_ref, _ = model.model.get_rope_index(full["input_ids"], inputs["image_grid_thw"])
        pending.append((name, question, copied, image.size, ids, grid, pixels, greedy, forced, reference_rows, rope_ref))
    del model
    gc.collect()

    vision = Qwen3VLVision(dict(config["vision_config"]))
    load_vision(vision, args.checkpoint)
    vision = vision.float().eval()
    text_cfg = dict(config["text_config"])
    text_cfg["tie_word_embeddings"] = bool(config.get("tie_word_embeddings", text_cfg.get("tie_word_embeddings")))
    text = Qwen3VLText(text_cfg, image_token, len(config["vision_config"]["deepstack_visual_indexes"]))
    load_text(text, args.checkpoint)
    text = text.float().eval()
    for name, question, copied, image_size, ids, grid, pixels, greedy, forced, reference_rows, rope_ref in pending:
        with torch.no_grad():
            # Eager port: vision tower, then withImage prefill and main forced decode with host M-RoPE positions.
            rope_host = np.array(mrope_positions(ids + forced, [tuple(grid)], image_token, merge), dtype=np.int64).T
            positions_match = bool(np.array_equal(rope_host, rope_ref[:, 0].numpy()))
            tables = {k: torch.from_numpy(v) for k, v in host_inputs(*grid).items()}
            embeds, deepstack = vision(pixels, tables["pos_index"], tables["pos_weight"], tables["patch_rc"])
            kc, vc = text.empty_state(len(ids) + len(forced) + 8)
            rope = torch.from_numpy(rope_host.astype(np.int32))
            rows = [text.step(torch.tensor([ids], dtype=torch.int32), torch.arange(len(ids), dtype=torch.int32)[None],
                              rope[:, :len(ids)], kc, vc, embeds, deepstack).double().numpy()]
            for i, tok in enumerate(forced[:-1]):
                p = len(ids) + i
                rows.append(text.step(torch.tensor([[tok]], dtype=torch.int32), torch.tensor([[p]], dtype=torch.int32),
                                      rope[:, p:p + 1], kc, vc).double().numpy())
            port = np.stack(rows)
        cos = [cosine(a, b) for a, b in zip(port, reference_rows)]
        pixels_path = args.output / f"{name}-pixels.f32"
        pixels.numpy().astype("<f4").tofile(pixels_path)
        logits_path = args.output / f"{name}-forced-logits.f32"
        reference_rows.astype("<f4").tofile(logits_path)
        cases.append({
            "name": name, "question": question, "image": {"file": copied.name, "sha256": sha256(copied),
                                                          "size": list(image_size)},
            "promptTokens": ids, "grid": grid, "imageTokens": int(np.prod(grid) // merge ** 2),
            "ropePositions": rope_ref[:, 0, :len(ids)].tolist(),
            "greedyTokens": greedy, "greedyText": processor.decode(greedy), "forcedTokens": forced,
            "pixels": {"file": pixels_path.name, "shape": list(pixels.shape), "sha256": sha256(pixels_path)},
            "forcedLogits": {"file": logits_path.name, "rows": int(reference_rows.shape[0]),
                             "vocab": int(reference_rows.shape[1]), "sha256": sha256(logits_path),
                             "rowMeaning": "row r = logits after prompt + forcedTokens[:r]"},
            "eagerPortVsReference": {"hostRopePositionsMatch": positions_match, "minCosine": round(min(cos), 8),
                                     "argmaxEqualRows": int(sum(int(np.argmax(a)) == int(np.argmax(b))
                                                                for a, b in zip(port, reference_rows))),
                                     "rows": len(cos)},
        })
        print(json.dumps({name: cases[-1]["eagerPortVsReference"], "text": cases[-1]["greedyText"]}), flush=True)
    record = {
        "schema": 1, "kind": "qwen3-vl-cpu-reference",
        "oracle": "transformers Qwen3VLForConditionalGeneration + AutoProcessor (fast image processor), float32, CPU",
        "source": {"repository": args.source_repository, "revision": args.source_revision, "license": args.license,
                   **({"baseModel": args.base_model} if args.base_model else {})},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "torchvision", "pillow")},
        "scriptSHA256": {p.name: sha256(p) for p in sorted(Path(__file__).resolve().parent.glob("*.py"))},
        "checkpointSHA256": {p.name: sha256(p) for p in sorted(args.checkpoint.glob("*.safetensors"))},
        "maximumPixels": args.max_pixels,
        "imageGrids": [list(g) for g in grids] if grids else None,
        "gridSnapping": ("nearest aspect ratio in log space, first listed wins a tie; the processor's smart_resize "
                         "replaced by the grid's pixel size") if grids else None,
        "imageDecoding": ("PIL Image.open(...), ImageOps.exif_transpose, convert('RGB') (Transformers' load_image)"
                          if args.exif_transpose else
                          "PIL Image.open(...).convert('RGB'), no EXIF transpose (the photos carry orientation 1 or none)"),
        "cases": cases,
    }
    (args.output / "reference.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""An MLX reference for the 27B VLM check (no CPU float32 reference fits: 27B is 108 GB in float32): mlx-vlm's own
Qwen3.5 implementation on an 8-bit conversion of the same checkpoint, from ref_vlm.py --prep-only's inputs (the
same prompt ids, pixel patches and grids). Saves the vision tower's merged embeddings, MLX's own M-RoPE positions (to
check against transformers'), and greedy generation with each step's log-probabilities. Run with a venv that has
mlx-vlm; holds /tmp/atelier-gpu.lock.

    python -I ref_mlx.py --model <mlx dir> --prep prep.npz --tokens 48 --output ref.npz
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

LOCK = "/tmp/atelier-gpu.lock"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--prep", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--force", type=Path, help="a reference npz: feed its gen_ids (teacher forcing) instead of greedy")
    args = ap.parse_args()
    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.generate import generate_step
    prep = dict(np.load(args.prep))
    while True:
        try:
            os.mkdir(LOCK)
            open(os.path.join(LOCK, "pid"), "w").write(str(os.getpid()))
            break
        except FileExistsError:
            time.sleep(5)
    try:
        t = time.monotonic()
        model, processor = load(str(args.model))
        load_s = time.monotonic() - t
        ids = mx.array(prep["input_ids"][None].astype(np.int32))
        pixels = mx.array(prep["pixel_values"])
        grid = mx.array(prep["image_grid_thw"])
        dtype = model.vision_tower.patch_embed.proj.weight.dtype
        image, _ = model.vision_tower(pixels.astype(dtype), grid)
        image = np.array(image.astype(mx.float32))
        positions, delta = model.language_model.get_rope_index(ids, grid, None, None)
        positions = np.array(positions)[:, 0].astype(np.int32)
        tokens, logprobs = [], []
        t = time.monotonic()
        forced = list(np.load(args.force)["gen_ids"][:args.tokens]) if args.force else None
        sampler = None
        if forced:
            def sampler(logprobs, _n=[0]):
                token = int(forced[min(_n[0], len(forced) - 1)])
                _n[0] += 1
                return mx.array([token])
        for (token, lp), _ in zip(generate_step(ids, model, pixels, None, max_tokens=args.tokens, temperature=0.0,
                                                sampler=sampler, image_grid_thw=grid), range(args.tokens)):
            tokens.append(int(token))
            logprobs.append(np.array(lp.astype(mx.float32)).reshape(-1))
        gen_s = time.monotonic() - t
    finally:
        if open(os.path.join(LOCK, "pid")).read().strip() == str(os.getpid()):
            shutil.rmtree(LOCK)
    np.savez(args.output, **prep, image_embeds=image, mlx_position_ids=positions,
             mlx_rope_delta=np.array(delta).astype(np.int32).reshape(-1), gen_ids=np.array(tokens, dtype=np.int32),
             gen_logits=np.stack(logprobs).astype(np.float16), reference=np.array("mlx-8bit-logprobs"))
    text = processor.tokenizer.decode(tokens, skip_special_tokens=True)
    same = bool(np.array_equal(positions, prep["position_ids"]))
    meta = {"model": str(args.model), "loadSeconds": round(load_s, 1), "generateSeconds": round(gen_s, 1),
            "tokens": len(tokens), "positionsMatchTransformers": same,
            "ropeDelta": {"mlx": np.array(delta).reshape(-1).tolist(), "transformers": prep["rope_delta"].tolist()},
            "imageTokens": int(image.shape[0]), "text": text}
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()

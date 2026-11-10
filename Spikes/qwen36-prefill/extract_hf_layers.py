#!/usr/bin/env python3
"""Qwen3.6-27B spike, step 0 from the bf16 HF checkpoint (the abliterated pin): the first N decoder layers as fp16
under their HF names, plus the embeddings of the describer-like prompt. Same outputs as extract_mlx_layers.py, so
spike.py reads either. Run with Forge's venv (torch, safetensors, transformers).

    python -I extract_hf_layers.py --snapshot <hf snapshot> --layers 8 --tokens 320 --output <dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_mlx_layers import PLUS_ONE, PROMPT  # noqa: E402  (the same prompt and norm names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=320)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.snapshot / "config.json"))
    index = json.load(open(args.snapshot / "model.safetensors.index.json"))["weight_map"]
    prefixes = tuple(f"model.language_model.layers.{i}." for i in range(args.layers))
    from safetensors.numpy import save_file
    args.output.mkdir(parents=True, exist_ok=True)
    means, count = {}, 0
    # Written 8 layers to a file, so only one group is ever in memory (64 layers are 51 GB in fp16).
    for lo in range(0, args.layers, 8):
        group = tuple(f"model.language_model.layers.{i}." for i in range(lo, min(lo + 8, args.layers)))
        by_file, out = {}, {}
        for k in sorted(k for k in index if k.startswith(group)):
            by_file.setdefault(index[k], []).append(k)
        for f, keys in sorted(by_file.items()):
            with safe_open(str(args.snapshot / f), framework="pt") as h:
                for k in keys:
                    t = h.get_tensor(k).float()
                    if k.endswith(".weight") and any(p in k for p in PLUS_ONE):
                        means[k] = float(t.mean())
                    out[k] = t.to(torch.float16).numpy()
        save_file(out, str(args.output / f"layers-{lo:03d}.safetensors"))
        count += len(out)
        del out
    # HF keeps the (1 + w) norms raw (mean near 0); MLX had them shifted. Refuse a shifted checkpoint here.
    # (Some deep q/k norms do sit near 1 raw, so only a checkpoint where all of them do counts as shifted, as in MLX's.)
    shifted = [k for k, m in means.items() if abs(m - 1) < abs(m)]
    if shifted and len(shifted) == len(means):
        raise SystemExit("every norm looks pre-shifted (mean near 1): not a raw HF checkpoint")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(args.snapshot))
    ids = tok(PROMPT)["input_ids"][:args.tokens]
    if len(ids) < args.tokens:
        raise SystemExit(f"prompt has {len(ids)} tokens")
    with safe_open(str(args.snapshot / index["model.language_model.embed_tokens.weight"]), framework="pt") as h:
        table = h.get_slice("model.language_model.embed_tokens.weight")
        emb = np.stack([table[i:i + 1].float().numpy()[0] for i in ids])
    np.save(args.output / "embeds.npy", emb.astype(np.float16))
    json.dump({"snapshot": str(args.snapshot), "layers": args.layers, "tokens": len(ids), "tensors": count,
               "textConfig": cfg["text_config"], "normMeans": means, "normsShiftedByMLX": False},
              open(args.output / "extract.json", "w"), indent=1)
    print(f"{count} tensors, {len(ids)} tokens, |embeds| max {np.abs(emb).max():.3g}")


if __name__ == "__main__":
    main()

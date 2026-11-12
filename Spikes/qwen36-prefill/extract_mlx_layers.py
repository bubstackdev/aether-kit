#!/usr/bin/env python3
"""Qwen3.6-27B hybrid-prefill spike, step 0: the first N decoder layers of the MLX 4-bit snapshot, dequantized to
fp16 under the HF names, plus the embeddings of a describer-like prompt. Realistic activation ranges for the spike
before the bf16 checkpoint is local (the spike's numerics are Core AI against torch on the SAME weights, so the 4-bit
origin doesn't matter). Needs mlx (run with a venv that has it).

    python -I extract_mlx_layers.py --snapshot <mlx snapshot> --layers 8 --tokens 320 --output <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np

PLUS_ONE = ("input_layernorm", "post_attention_layernorm", "q_norm", "k_norm")
PROMPT = ("You write the costume description an image generator will use to recreate a character's outfit. The pictures "
          "all show the same costume; some may be drawings, renders, statues or cosplay photos, and they may disagree. "
          "Look at all of them and write what the costume really is: where the pictures disagree, follow what most of "
          "them show, and leave out anything only one picture shows unless it is clearly part of the costume. Write one "
          "comma-separated description of at most 90 words, no full sentences, in this order: the main garment first "
          "(what it is, its length, cut, fabric and color); then what makes it distinctive and how it is built, stated "
          "concretely; then the neckline and any openings; then the sleeves; then anything worn on only one arm or "
          "shoulder, without saying left or right; then the smaller pieces (collar, gloves, belts, legwear, footwear). "
          "Give every piece its color. Only describe what you can see in the pictures. ") * 3


def main():
    import mlx.core as mx  # here, so extract_hf_layers can share PROMPT without mlx
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=320)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.snapshot / "config.json"))
    q = cfg.get("quantization", {"group_size": 64, "bits": 4})
    index = json.load(open(args.snapshot / "model.safetensors.index.json"))["weight_map"]
    shards = {}

    def tensor(name):
        f = index[name]
        if f not in shards:
            shards[f] = mx.load(str(args.snapshot / f))
        return shards[f][name]

    def weight(base):
        w = tensor(base + ".weight")
        if base + ".scales" in index:
            w = mx.dequantize(w, tensor(base + ".scales"), tensor(base + ".biases"), group_size=q["group_size"], bits=q["bits"])
        return w

    out, notes = {}, {}
    prefixes = [f"language_model.model.layers.{i}." for i in range(args.layers)]
    bases = sorted({k.rsplit(".", 1)[0] for k in index if k.startswith(tuple(prefixes))
                    and not k.endswith((".scales", ".biases"))} | {k[:-len(".weight")] for k in index if k.startswith(tuple(prefixes)) and k.endswith(".weight")})
    for name in sorted(k for k in index if k.startswith(tuple(prefixes)) and not k.endswith((".scales", ".biases"))):
        base = name[:-len(".weight")] if name.endswith(".weight") else None
        w = weight(base) if base and base + ".scales" in index else tensor(name)
        a = np.array(w.astype(mx.float32))
        hf = "model.language_model." + name[len("language_model.model."):]
        if name.endswith("conv1d.weight") and a.ndim == 3 and a.shape[1] > a.shape[2]:
            a = np.ascontiguousarray(np.moveaxis(a, 2, 1))  # MLX [out, k, in] -> torch [out, in, k]
        if name.endswith(".weight") and any(p in name for p in PLUS_ONE):
            notes.setdefault("normMeans", {})[hf] = float(a.mean())
        out[hf] = a.astype(np.float16)
    # MLX's sanitize stores the (1 + w) norms already shifted; the authoring RMSNormPlusOne wants HF's raw w.
    shifted = all(abs(m - 1) < abs(m) for m in notes.get("normMeans", {}).values())
    if shifted:
        for k in notes["normMeans"]:
            out[k] = (out[k].astype(np.float32) - 1).astype(np.float16)
    notes["normsShiftedByMLX"] = shifted

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(args.snapshot))
    ids = tok(PROMPT)["input_ids"][:args.tokens]
    if len(ids) < args.tokens:
        raise SystemExit(f"prompt has {len(ids)} tokens")
    emb = np.array(mx.dequantize(tensor("language_model.model.embed_tokens.weight")[mx.array(ids)],
                                 tensor("language_model.model.embed_tokens.scales")[mx.array(ids)],
                                 tensor("language_model.model.embed_tokens.biases")[mx.array(ids)],
                                 group_size=q["group_size"], bits=q["bits"]).astype(mx.float32))
    args.output.mkdir(parents=True, exist_ok=True)
    from safetensors.numpy import save_file
    save_file(out, str(args.output / "layers.safetensors"))
    np.save(args.output / "embeds.npy", emb.astype(np.float16))
    json.dump({"snapshot": str(args.snapshot), "layers": args.layers, "tokens": len(ids), "tensors": len(out),
               "textConfig": cfg["text_config"], **notes}, open(args.output / "extract.json", "w"), indent=1)
    print(f"{len(out)} tensors, {len(ids)} tokens, norms shifted by MLX: {shifted}")


if __name__ == "__main__":
    main()

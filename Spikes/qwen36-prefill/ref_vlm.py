#!/usr/bin/env python3
"""The CPU reference for the VLM stage: transformers' unmodified Qwen3.5/3.6 (Qwen3_5ForConditionalGeneration) in
float32 on a describer prompt with real pictures. Saves everything the Core AI side needs to start from the same inputs
and be compared stage by stage: the prompt ids, the processor's pixel patches and grids, the 3-axis M-RoPE positions,
the vision tower's merged embeddings, and greedy generation with each step's logits.

    Forge/.venv/bin/python -I ref_vlm.py --snapshot <hf snapshot> --pictures a.jpg b.jpg --tokens 48 --output ref.npz
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

PATCH_CAPACITY = 2048  # the Core AI tower's rows (Forge's qwen3vl default): 512 merged tokens, 524,288 pixels a picture
INSTRUCTIONS = (
    "You write the costume description an image generator will use to recreate a character's outfit. The pictures all "
    "show the same costume; some may be drawings, renders, statues or cosplay photos, and they may disagree. Look at all "
    "of them and write what the costume really is: where the pictures disagree, follow what most of them show, and leave "
    "out anything only one picture shows unless it is clearly part of the costume.\n"
    "Write one comma-separated description of at most 90 words, no full sentences, in this order: the main garment first "
    "(what it is, its length, cut, fabric and color); then what makes it distinctive and how it is built (for example "
    "what the skirt is made of, how panels are joined, armor plates), stated concretely; then the neckline and any "
    "openings; then the sleeves; then anything worn on only one arm or shoulder, without saying left or right; then the "
    "smaller pieces (collar, gloves, belts, legwear, footwear). Give every piece its color. Only describe what you can "
    "see in the pictures. You may say where on the body a garment sits or what its cut leaves bare, but never describe "
    "the body itself. Never describe the face, hair, skin, eyes or anything that belongs to the person, never name the "
    "character or the franchise, and leave out weapons and props held in the hands.")
USER = "Write the costume description."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--pictures", type=Path, nargs="+", required=True)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--prep-only", action="store_true",
                    help="the inputs and positions only (a meta-device model: no weights), for a reference run elsewhere (ref_mlx.py)")
    args = ap.parse_args()
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    torch.set_num_threads(max(1, (torch.get_num_threads())))
    processor = AutoProcessor.from_pretrained(str(args.snapshot))
    images = [Image.open(p).convert("RGB") for p in args.pictures]
    messages = [{"role": "system", "content": [{"type": "text", "text": INSTRUCTIONS}]},
                {"role": "user", "content": [*({"type": "image"} for _ in images), {"type": "text", "text": USER}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    inputs = processor(text=[prompt], images=images, return_tensors="pt",
                       size={"shortest_edge": 65536, "longest_edge": PATCH_CAPACITY * 256})
    grids = inputs["image_grid_thw"]
    if int(grids.prod(-1).max()) > PATCH_CAPACITY:
        raise SystemExit(f"a picture has more than {PATCH_CAPACITY} patches: {grids.tolist()}")
    if args.prep_only:
        from transformers import AutoConfig
        with torch.device("meta"):
            shell = Qwen3_5ForConditionalGeneration(AutoConfig.from_pretrained(str(args.snapshot)))
        positions, delta = shell.model.get_rope_index(inputs["input_ids"], inputs["mm_token_type_ids"], grids)
        np.savez(args.output, input_ids=inputs["input_ids"][0].numpy().astype(np.int32),
                 pixel_values=inputs["pixel_values"].float().numpy(), image_grid_thw=grids.numpy().astype(np.int32),
                 position_ids=positions[:, 0].numpy().astype(np.int32), rope_delta=delta.numpy().astype(np.int32))
        print(json.dumps({"promptTokens": int(inputs["input_ids"].shape[1]), "grids": grids.tolist(),
                          "positionsMax": int(positions.max()), "ropeDelta": int(delta.ravel()[0])}))
        return
    t = time.monotonic()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(str(args.snapshot), dtype=torch.float32,
                                                            attn_implementation="sdpa").eval()
    load = time.monotonic() - t
    with torch.no_grad():
        embeds = torch.cat(model.model.get_image_features(inputs["pixel_values"], grids).pooler_output, dim=0)
        positions, delta = model.model.get_rope_index(inputs["input_ids"], inputs["mm_token_type_ids"], grids)
        t = time.monotonic()
        out = model.generate(**inputs, max_new_tokens=args.tokens, do_sample=False, output_logits=True,
                             return_dict_in_generate=True)
        gen_seconds = time.monotonic() - t
    prompt_len = inputs["input_ids"].shape[1]
    gen = out.sequences[0, prompt_len:].numpy()
    logits = torch.stack(out.logits, dim=0)[:, 0].float().numpy()            # [N, V]: step i predicts gen[i]
    text = processor.tokenizer.decode(gen, skip_special_tokens=True)
    np.savez(args.output, input_ids=inputs["input_ids"][0].numpy().astype(np.int32),
             mm_token_type_ids=inputs["mm_token_type_ids"][0].numpy().astype(np.int32),
             pixel_values=inputs["pixel_values"].float().numpy(), image_grid_thw=grids.numpy().astype(np.int32),
             position_ids=positions[:, 0].numpy().astype(np.int32), rope_delta=delta.numpy().astype(np.int32),
             image_embeds=embeds.float().numpy(), gen_ids=gen.astype(np.int32), gen_logits=logits.astype(np.float16))
    meta = {"snapshot": str(args.snapshot), "pictures": [str(p) for p in args.pictures], "promptTokens": prompt_len,
            "imageTokens": int(embeds.shape[0]), "grids": grids.tolist(), "loadSeconds": round(load, 1),
            "generateSeconds": round(gen_seconds, 1), "text": text}
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()

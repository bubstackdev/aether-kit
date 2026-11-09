#!/usr/bin/env python3
"""Hermetic CPU test of the export graphs' contract on a tiny random Qwen3-VL (no checkpoint, no Core AI).

    Forge/.venv/bin/python -I Forge/qwen3vl/test_qwen3vl_graphs.py

Builds a randomly initialised Qwen3VLForConditionalGeneration with a small config (the real architecture, every size
shrunk) and checks, in float32:
- `plan_images` + `mrope_positions` (the host's static inputs and the graph's position rule) equal upstream's
  `get_rope_index` for every token of prompts with zero, one, two and three images of different grids;
- `VisionTower` on the zero-padded patches equals upstream's tower per image (cosine > 0.99999), padding rows zero;
- `Decoder`, fed upstream's tower outputs packed into the slots, gives upstream's last-row logits for a multi-image
  prompt (rel L2 <= 1e-4), in one prefill and in chunks that split an image, and when greedy tokens are fed one at a
  time after it.
Exits nonzero on a failure. A few seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qwen3vl_graphs as qg  # noqa: E402

VOCAB = 151936
SLOTS, IMAGES = 96, 4
CAPACITY = 128                                    # tower patch rows: up to 32 tokens an image


def tiny_model():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    config = Qwen3VLConfig(
        text_config={"vocab_size": VOCAB, "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 4,
                     "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "rms_norm_eps": 1e-6,
                     "tie_word_embeddings": True,
                     "rope_parameters": {"rope_type": "default", "rope_theta": 5_000_000.0,
                                         "mrope_section": [4, 2, 2], "mrope_interleaved": True}},
        vision_config={"depth": 6, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2, "patch_size": 16,
                       "spatial_merge_size": 2, "temporal_patch_size": 2, "out_hidden_size": 64,
                       "num_position_embeddings": 64, "deepstack_visual_indexes": [1, 3, 4]},
        image_token_id=qg.IMAGE_PAD_ID, vision_start_token_id=qg.VISION_START_ID, vision_end_token_id=qg.VISION_END_ID,
        tie_word_embeddings=True)
    torch.manual_seed(0)
    model = Qwen3VLForConditionalGeneration(config).eval()
    with torch.no_grad():                         # visible logits: random init is near-flat otherwise
        model.model.language_model.embed_tokens.weight.mul_(20)
    return model


def prompt(grids: list[tuple[int, int]], rng: np.random.Generator) -> tuple[np.ndarray, list[int]]:
    """A user turn of random text with each image's expanded pads (vision start, count pads, vision end) between
    text runs; returns the ids and each image's first-pad position."""
    ids = [151644, 872, 198]
    starts = []
    for h, w in grids:
        ids += rng.integers(100, 5000, size=int(rng.integers(2, 6))).tolist()
        ids.append(qg.VISION_START_ID)
        starts.append(len(ids))
        ids += [qg.IMAGE_PAD_ID] * (h * w // 4)
        ids.append(qg.VISION_END_ID)
    ids += rng.integers(100, 5000, size=7).tolist() + [151645, 198, 151644, 77091, 198]
    return np.array(ids, dtype=np.int64), starts


def main() -> None:
    torch.set_grad_enabled(False)
    failures = []

    def check(name: str, ok: bool, detail="") -> None:
        print(f"[test] {'PASS' if ok else 'FAIL'} {name} {detail}", file=sys.stderr, flush=True)
        if not ok:
            failures.append(name)

    model = tiny_model()
    rng = np.random.default_rng(0)
    cases = {"text-only": [], "one": [(8, 12)], "two": [(8, 12), (10, 6)], "three": [(6, 6), (4, 16), (8, 8)]}
    for name, grids in cases.items():
        ids, starts = prompt(grids, rng)
        thw = torch.tensor([[1, h, w] for h, w in grids], dtype=torch.long) if grids else None
        mm = torch.from_numpy((ids == qg.IMAGE_PAD_ID).astype(np.int32))[None]
        want, _ = model.model.get_rope_index(torch.from_numpy(ids)[None], mm, image_grid_thw=thw)
        positions, start, amount = qg.plan_images(starts, grids, slots=SLOTS, images=IMAGES)
        ext = qg.extension_ids(ids, VOCAB)
        got = qg.mrope_positions(torch.from_numpy(ext), torch.arange(len(ids), dtype=torch.int32),
                                 torch.from_numpy(positions), torch.from_numpy(start), torch.from_numpy(amount),
                                 VOCAB, SLOTS)
        check(f"{name}.mrope", np.array_equal(torch.stack(got).numpy(), want[:, 0].numpy()))

    tower = qg.VisionTower(model.model.visual, capacity=CAPACITY, dtype=torch.float32, storage=torch.float32).eval()
    decoder = qg.Decoder(model, slots=SLOTS, images=IMAGES, dtype=torch.float32, attention=torch.float32,
                         storage=torch.float32, statics=torch.float32).eval()
    grids = cases["three"]
    ids, starts = prompt(grids, rng)
    pixels = [torch.randn(h * w, 1536) for h, w in grids]
    image = torch.zeros(SLOTS, 64)
    deep = torch.zeros(3 * SLOTS, 64)
    base = 0
    for (h, w), pv in zip(grids, pixels):
        thw = torch.tensor([[1, h, w]])
        up = model.model.visual(pv, grid_thw=thw)
        n = h * w // 4
        padded = torch.zeros(CAPACITY, 1536)
        padded[:h * w] = pv
        gi, gd = tower(padded, torch.tensor([h, w], dtype=torch.int32))
        check(f"tower.{h}x{w}.embeds", qg.cosine(gi[:n].numpy(), up.pooler_output.numpy()) > 0.99999)
        tn = CAPACITY // 4
        for k in range(3):
            check(f"tower.{h}x{w}.deepstack{k}",
                  qg.cosine(gd[k * tn:k * tn + n].numpy(), up.deepstack_features[k].numpy()) > 0.99999)
        check(f"tower.{h}x{w}.paddingZero", not gi[n:].any() and not any(gd[k * tn + n:(k + 1) * tn].any()
                                                                         for k in range(3)))
        image[base:base + n] = up.pooler_output
        for k in range(3):
            deep[k * SLOTS + base:k * SLOTS + base + n] = up.deepstack_features[k]
        base += n

    inputs = {"input_ids": torch.from_numpy(ids)[None], "pixel_values": torch.cat(pixels),
              "image_grid_thw": torch.tensor([[1, h, w] for h, w in grids]),
              "mm_token_type_ids": torch.from_numpy((ids == qg.IMAGE_PAD_ID).astype(np.int32))[None]}
    out = model(**inputs, use_cache=True)
    want = out.logits[0, -1].numpy()
    positions, start, amount = qg.plan_images(starts, grids, slots=SLOTS, images=IMAGES)
    statics = (image, deep, torch.from_numpy(positions), torch.from_numpy(start), torch.from_numpy(amount))
    ext = torch.from_numpy(qg.extension_ids(ids, VOCAB))[None]
    length = ext.shape[1]
    kc, vc = decoder.empty_state(length + 8)
    got = decoder(ext, torch.arange(length, dtype=torch.int32)[None], *statics, kc, vc)[0, 0].numpy()
    check("decoder.prefill.relL2", qg.rel_l2(got, want) <= 1e-4, qg.rel_l2(got, want))
    kc, vc = decoder.empty_state(length + 8)
    cut = starts[1] + 3                            # inside the second image
    decoder(ext[:, :cut], torch.arange(cut, dtype=torch.int32)[None], *statics, kc, vc)
    chunked = decoder(ext[:, cut:], torch.arange(length, dtype=torch.int32)[None], *statics, kc, vc)[0, 0].numpy()
    check("decoder.chunked.relL2", qg.rel_l2(chunked, want) <= 1e-4, qg.rel_l2(chunked, want))
    past, row = out.past_key_values, out.logits[0, -1]
    for step in range(4):
        token = int(row.argmax())
        upstream = model(input_ids=torch.tensor([[token]]), past_key_values=past, use_cache=True)
        past, row = upstream.past_key_values, upstream.logits[0, -1]
        t = length + step + 1
        mine = decoder(torch.tensor([[token]], dtype=torch.int32), torch.arange(t, dtype=torch.int32)[None],
                       *statics, kc, vc)[0, 0].numpy()
        check(f"decoder.step{step}.relL2", qg.rel_l2(mine, row.numpy()) <= 1e-4, qg.rel_l2(mine, row.numpy()))
    if failures:
        raise SystemExit(f"{len(failures)} check(s) failed: {failures}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Hermetic CPU test of the export graphs' contract on a tiny random Qwen3.5 (no checkpoint, no Core AI).

    Forge/.venv/bin/python -I Forge/qwen3_5/test_qwen3_5_graphs.py

Builds a randomly initialised Qwen3_5ForConditionalGeneration with a small config (the real architecture: three Gated
DeltaNet layers to a gated full-attention layer, partial interleaved M-RoPE, untied head; every size shrunk) and
checks, in float32:
- `delta_scan` on a block equals the one-token recurrence, piece by piece, on large values (the case that broke a
  64-token block in float32), and padded to CALL_TOKENS rows its padding leaves the state exactly as it was;
- `plan_images` + `mrope_positions` equal upstream's `get_rope_index` for prompts with zero to three images;
- `VisionTower` on zero-padded patches equals upstream's tower per image, padding rows zero;
- `Decoder`, fed upstream's tower outputs packed into the slots, gives upstream's last-row logits for a five-image
  prompt as the runtime feeds it (calls of CALL_TOKENS, the last one shorter, a call boundary inside an image), and in
  calls of irregular sizes, and then greedy tokens one at a time (rel L2 <= 1e-4).
Exits nonzero on a failure. A few seconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qwen3_5_graphs as qg  # noqa: E402

VOCAB = 248320
PAD, START, END = 248056, 248053, 248054
SLOTS, IMAGES = 96, 8
CAPACITY = 128                                    # tower patch rows: up to 32 tokens an image
HIDDEN = 64


def tiny_model():
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    config = Qwen3_5Config(
        text_config={"vocab_size": VOCAB, "hidden_size": HIDDEN, "intermediate_size": 96, "num_hidden_layers": 8,
                     "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32, "rms_norm_eps": 1e-6,
                     "layer_types": ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3
                     + ["full_attention"],
                     "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 16,
                     "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4,
                     "rope_parameters": {"rope_type": "default", "rope_theta": 10_000_000.0,
                                         "partial_rotary_factor": 0.25, "mrope_section": [2, 1, 1],
                                         "mrope_interleaved": True}},
        vision_config={"depth": 3, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2, "patch_size": 16,
                       "spatial_merge_size": 2, "temporal_patch_size": 2, "out_hidden_size": HIDDEN,
                       "num_position_embeddings": 64},
        image_token_id=PAD, vision_start_token_id=START, vision_end_token_id=END, tie_word_embeddings=False)
    torch.manual_seed(0)
    model = Qwen3_5ForConditionalGeneration(config).eval()
    with torch.no_grad():                         # visible logits: random init is near-flat otherwise
        model.model.language_model.embed_tokens.weight.mul_(20)
    return model


def prompt(grids, rng):
    ids = [248045, 846, 198]
    starts = []
    for h, w in grids:
        ids += rng.integers(100, 5000, size=int(rng.integers(2, 6))).tolist()
        ids.append(START)
        starts.append(len(ids))
        ids += [PAD] * (h * w // 4)
        ids.append(END)
    ids += rng.integers(100, 5000, size=7).tolist() + [248046, 198, 248045, 74455, 198]
    return np.array(ids, dtype=np.int64), starts


def main() -> None:
    torch.set_grad_enabled(False)
    failures = []

    def check(name: str, ok: bool, detail="") -> None:
        print(f"[test] {'PASS' if ok else 'FAIL'} {name} {detail}", file=sys.stderr, flush=True)
        if not ok:
            failures.append(name)

    # The block scan against the recurrence, on values like Qwen3.6-27B's layer 0 (|v| ~ 35, slow decay).
    torch.manual_seed(1)
    hv, dk, dv, n = 4, 16, 16, 4 * qg.SCAN_PIECE + 5
    q = qg.l2norm(torch.randn(1, hv, n, dk)) * dk ** -0.5
    k = qg.l2norm(torch.randn(1, hv, n, dk) + 3.0)           # near-parallel keys
    v = torch.randn(1, hv, n, dv) * 35
    g = -torch.rand(1, hv, n) * 0.05
    beta = torch.sigmoid(torch.randn(1, hv, n) + 2)
    state0 = torch.randn(1, hv, dk, dv)
    state, outs = state0, []
    for t in range(n):
        o, state = qg.delta_step(q[:, :, t:t + 1], k[:, :, t:t + 1], v[:, :, t:t + 1], g[:, :, t:t + 1],
                                 beta[:, :, t:t + 1], state)
        outs.append(o)
    want_out, want_state = torch.cat(outs, 2), state
    got_out, got_state = qg.delta_scan(q, k, v, g, beta, state0)
    check("scan.out", qg.rel_l2(got_out.numpy(), want_out.numpy()) <= 1e-4,
          qg.rel_l2(got_out.numpy(), want_out.numpy()))
    check("scan.state", qg.rel_l2(got_state.numpy(), want_state.numpy()) <= 1e-4,
          qg.rel_l2(got_state.numpy(), want_state.numpy()))
    # The same data as one 64-token block (the size that failed on the 27B): the reason for SCAN_PIECE.
    _, whole = qg.delta_block(q[:, :, :64], k[:, :, :64], v[:, :, :64], g[:, :, :64], beta[:, :, :64], state0)
    _, pieces = qg.delta_scan(q[:, :, :64], k[:, :, :64], v[:, :, :64], g[:, :, :64], beta[:, :, :64], state0)
    # Padded: the first 20 tokens in a CALL_TOKENS-row call equal their recurrence, state included.
    state, outs = state0, []
    for t in range(20):
        o, state = qg.delta_step(q[:, :, t:t + 1], k[:, :, t:t + 1], v[:, :, t:t + 1], g[:, :, t:t + 1],
                                 beta[:, :, t:t + 1], state)
        outs.append(o)
    got_out, got_state = qg.delta_scan(q[:, :, :20], k[:, :, :20], v[:, :, :20], g[:, :, :20], beta[:, :, :20], state0,
                                       rows=qg.CALL_TOKENS)
    check("scan.padded.out", qg.rel_l2(got_out.numpy(), torch.cat(outs, 2).numpy()) <= 1e-4,
          qg.rel_l2(got_out.numpy(), torch.cat(outs, 2).numpy()))
    check("scan.padded.state", qg.rel_l2(got_state.numpy(), state.numpy()) <= 1e-4,
          qg.rel_l2(got_state.numpy(), state.numpy()))
    print(f"[test] info scan: one 64-token block rel L2 {qg.rel_l2(whole.numpy(), pieces.numpy()):.3g} against "
          f"{qg.SCAN_PIECE}-token pieces", file=sys.stderr)

    model = tiny_model()
    rng = np.random.default_rng(0)
    cases = {"text-only": [], "one": [(8, 12)], "two": [(8, 12), (10, 6)], "three": [(6, 6), (4, 16), (8, 8)]}
    for name, grids in cases.items():
        ids, starts = prompt(grids, rng)
        thw = torch.tensor([[1, h, w] for h, w in grids], dtype=torch.long) if grids else None
        mm = torch.from_numpy((ids == PAD).astype(np.int32))[None]
        want, _ = model.model.get_rope_index(torch.from_numpy(ids)[None], mm, image_grid_thw=thw)
        positions, start, amount = qg.plan_images(starts, grids, slots=SLOTS, images=IMAGES)
        ext = qg.extension_ids(ids, VOCAB, PAD)
        got = qg.mrope_positions(torch.from_numpy(ext), torch.arange(len(ids), dtype=torch.int32),
                                 torch.from_numpy(positions), torch.from_numpy(start), torch.from_numpy(amount),
                                 VOCAB, SLOTS)
        check(f"{name}.mrope", np.array_equal(torch.stack(got).numpy(), want[:, 0].numpy()))

    tower = qg.VisionTower(model.model.visual, capacity=CAPACITY, dtype=torch.float32, storage=torch.float32,
                           output=torch.float32).eval()
    decoder = qg.Decoder(model, slots=SLOTS, images=IMAGES, dtype=torch.float32, attention=torch.float32,
                         storage=torch.float32).eval()
    grids = [(6, 6), (4, 16), (8, 8), (12, 8), (10, 10)]
    while True:                                    # a prompt whose first prefill boundary falls inside an image
        ids, starts = prompt(grids, rng)
        if any(st < qg.CALL_TOKENS < st + h * w // 4 for st, (h, w) in zip(starts, grids)):
            break
    pixels = [torch.randn(h * w, 1536) for h, w in grids]
    image = torch.zeros(SLOTS, HIDDEN)
    base = 0
    for (h, w), pv in zip(grids, pixels):
        up = model.model.visual(pv, grid_thw=torch.tensor([[1, h, w]])).pooler_output
        tokens = h * w // 4
        padded = torch.zeros(CAPACITY, 1536)
        padded[:h * w] = pv
        got = tower(padded, torch.tensor([h, w], dtype=torch.int32))
        check(f"tower.{h}x{w}", qg.cosine(got[:tokens].numpy(), up.numpy()) > 0.99999)
        check(f"tower.{h}x{w}.paddingZero", not got[tokens:].any())
        image[base:base + tokens] = up
        base += tokens

    inputs = {"input_ids": torch.from_numpy(ids)[None], "pixel_values": torch.cat(pixels),
              "image_grid_thw": torch.tensor([[1, h, w] for h, w in grids]),
              "mm_token_type_ids": torch.from_numpy((ids == PAD).astype(np.int32))[None]}
    out = model(**inputs, use_cache=True)
    want = out.logits[0, -1].numpy()
    positions, start, amount = qg.plan_images(starts, grids, slots=SLOTS, images=IMAGES)
    statics = (image, torch.from_numpy(positions), torch.from_numpy(start), torch.from_numpy(amount))
    ext = torch.from_numpy(qg.extension_ids(ids, VOCAB, PAD))[None]
    length = ext.shape[1]
    if length <= 2 * qg.CALL_TOKENS:
        raise SystemExit(f"the test prompt ({length}) must take several calls")

    def feed(sizes, state):
        i, out = 0, None
        for n in sizes:
            out = decoder(ext[:, i:i + n], torch.arange(i + n, dtype=torch.int32)[None], *statics, *state)
            i += n
        assert i == length
        return out[0, 0].numpy()

    state = decoder.empty_state(length + 8)
    runtime = [qg.CALL_TOKENS] * (length // qg.CALL_TOKENS) + ([length % qg.CALL_TOKENS] if length % qg.CALL_TOKENS else [])
    got = feed(runtime, state)
    check("decoder.prompt.relL2 (calls of CALL_TOKENS, a boundary inside an image)", qg.rel_l2(got, want) <= 1e-4,
          qg.rel_l2(got, want))
    irregular, left = [], length
    for n in [1, 7, qg.CALL_TOKENS, 3, 19, qg.CALL_TOKENS, 2] * 20:
        if left == 0:
            break
        irregular.append(min(n, left))
        left -= irregular[-1]
    other = feed(irregular, decoder.empty_state(length + 8))
    check("decoder.irregularCalls.relL2", qg.rel_l2(other, want) <= 1e-4, qg.rel_l2(other, want))
    past, row = out.past_key_values, out.logits[0, -1]
    for step in range(4):
        token = int(row.argmax())
        upstream = model(input_ids=torch.tensor([[token]]), past_key_values=past, use_cache=True)
        past, row = upstream.past_key_values, upstream.logits[0, -1]
        t = length + step + 1
        mine = decoder(torch.tensor([[token]], dtype=torch.int32), torch.arange(t, dtype=torch.int32)[None],
                       *statics, *state)[0, 0].numpy()
        check(f"decoder.step{step}.relL2", qg.rel_l2(mine, row.numpy()) <= 1e-4, qg.rel_l2(mine, row.numpy()))
    if failures:
        raise SystemExit(f"{len(failures)} check(s) failed: {failures}")
    print("all checks passed", file=sys.stderr)


if __name__ == "__main__":
    main()

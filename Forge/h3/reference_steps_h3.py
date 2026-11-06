"""The H3 driver gate's reference: the bundle's own graphs on the GPU (Python's Core AI runtime, the path check_h3.py
holds to the original modules on the CPU at 0.02 relative L2), fed by Diffusers' own host recipe, so the Swift driver's
host side (tokens, embedding gather, rotary prefix, packed layout, gather, row timesteps, AdaLN tags, modulation and
padding, rotary, the scheduler update, Kit's noise in 0.0.x's order) is checked against an independent one.

    Forge/.venv/bin/python -I Forge/h3/reference_steps_h3.py --output <export dir> [--steps 1] [--seed 20261106]

Writes <export dir>/driver-reference/: spec.json, ids, the tap (bf16 bits), and per step the head velocities and the
advanced latents (float32). The modulation is bound unpadded here (`[6, 3k, D]`), the Swift driver pads it to the
recipe's cardinality: equal results show the padding exact. Reads real MiniMax H3 weights (see LICENSE).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h3_graphs as hg  # noqa: E402
from check_h3 import PROMPT, Graph, Weights, wait_for_gpu  # noqa: E402


def seeded_noise(seed: int, count: int) -> np.ndarray:
    """Kit's SeededGaussianNoise: SplitMix64 and double Box-Muller, float32 (vectorised over the stream)."""
    import math
    mask = (1 << 64) - 1
    pairs = (count + 1) // 2
    states = (np.arange(1, 2 * pairs + 1, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(seed & mask))
    z = states
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    a, b = z[0::2], z[1::2]
    u1 = ((a >> np.uint64(11)).astype(np.float64) + 1) / 9007199254740992.0
    u2 = (b >> np.uint64(11)).astype(np.float64) / 9007199254740992.0
    radius = np.sqrt(-2.0 * np.log(u1))
    angle = 2.0 * math.pi * u2
    out = np.empty(2 * pairs, dtype=np.float32)
    out[0::2] = (radius * np.cos(angle)).astype(np.float32)
    out[1::2] = (radius * np.sin(angle)).astype(np.float32)
    return out[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    bundle = args.output / "bundle"
    out = args.output / "driver-reference"
    out.mkdir(exist_ok=True)
    manifest = json.loads((bundle / ".aether/aether.json").read_text())
    h3 = manifest["h3"]
    profile = h3["profiles"][0]
    recipe = h3["recipes"][h3["defaultRecipe"] + "/t2va"]
    wait_for_gpu(args.wait)
    torch.set_num_threads(8)

    from transformers import AutoTokenizer
    ids = AutoTokenizer.from_pretrained(bundle / "tokenizer")(PROMPT, add_special_tokens=False)["input_ids"]
    T = len(ids)
    c = h3["conditioner"]
    rows = torch.frombuffer(bytearray((bundle / "weights/conditioner/embedding.bin").read_bytes()), dtype=torch.bfloat16).view(c["embedding"]["rows"], -1)
    cos_t = torch.frombuffer(bytearray((bundle / "tables/conditioner/rope-cos.raw").read_bytes()), dtype=torch.bfloat16).view(-1, 128)[:T][None]
    sin_t = torch.frombuffer(bytearray((bundle / "tables/conditioner/rope-sin.raw").read_bytes()), dtype=torch.bfloat16).view(-1, 128)[:T][None]
    specs = json.loads((args.output / "work/build.json").read_text())
    layer, layer_weights = Graph(bundle / "models/conditioner-layer.aimodel"), Weights(bundle, "conditioner", specs["conditioner"]["inputs"])
    x = rows[torch.tensor(ids)][None]
    for index in range(c["layers"]):
        x = layer(hidden=x, cos=cos_t, sin=sin_t, **layer_weights.block(index))["hidden_out"]
    tap = x
    del layer
    (out / "tap.raw").write_bytes(tap.contiguous().view(torch.int16).numpy().tobytes())
    refined = Graph(bundle / "models/prepare-text.aimodel")(text=tap, **Weights(bundle, "prepare-text", specs["prepare"]["textInputs"]).block(0))["refined"]

    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep, MiniMaxH3SetTimestepsStep
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (MINIMAX_H3_AUDIO_CHANNELS, MINIMAX_H3_AUDIO_TAG,
                                                                         MINIMAX_H3_TEXT_TAG, MINIMAX_H3_VIDEO_TAG)
    lf, lh, lw, na = profile["latentFrames"], profile["height"] // 16, profile["width"] // 16, profile["audioLatents"]
    positions, tags, video_idx, audio_idx, text_idx, cv, ca = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        torch.full((T,), MINIMAX_H3_TEXT_TAG, dtype=torch.long), lf, lh, lw, na, (1, 2, 2), MINIMAX_H3_AUDIO_CHANNELS,
        MINIMAX_H3_AUDIO_TAG, MINIMAX_H3_VIDEO_TAG)
    nv, nar = video_idx.numel(), audio_idx.numel()
    R = T + nv + nar
    gather = torch.empty(R, dtype=torch.int32)
    gather[text_idx] = torch.arange(T, dtype=torch.int32)
    gather[video_idx] = T + torch.arange(nv, dtype=torch.int32)
    gather[audio_idx] = T + nv + torch.arange(nar, dtype=torch.int32)
    cos_r, sin_r = MiniMaxH3RotaryPosEmbed()(positions.to(torch.float32))

    # Kit's stream in 0.0.x's order: the source [24, T, H, W] video tensor (then patchified), then the audio rows.
    noise = seeded_noise(args.seed, nv * 96 + nar * 32)
    source = torch.from_numpy(noise[:nv * 96].copy()).view(24, lf, lh, lw)
    video = source.view(24, lf, lh // 2, 2, lw // 2, 2).permute(1, 2, 4, 0, 3, 5).reshape(1, nv, 96).contiguous()
    audio = torch.from_numpy(noise[nv * 96:].copy()).view(1, nar, 32)

    pack, body, heads = (Graph(bundle / f"models/{name}.aimodel") for name in ("prepare-pack", "body", "heads"))
    body_weights = Weights(bundle, "body", specs["body-weights"]["inputs"])
    vi, ai = video_idx.to(torch.int32), audio_idx.to(torch.int32)
    for step in range(args.steps):
        entry = recipe["steps"][step]
        vt, at = recipe["video"]["timesteps"][step], recipe["audio"]["timesteps"][step]
        unique, tindex = MiniMaxH3SetTimestepsStep.build_row_timesteps(video_idx, audio_idx, cv, ca, T, float(vt), float(at), 0.0, 0.0)
        k = unique.numel()
        assert [float(v) for v in unique.to(torch.float32)] == [float(np.float32(v)) for v in entry["timesteps"]], (unique, entry["timesteps"])
        adaln = (tindex * 3 + tags).to(torch.int32)
        temb = torch.frombuffer(bytearray((bundle / entry["temb"].split(":", 1)[1]).read_bytes()), dtype=torch.float32).view(k, -1)
        x = pack(video=video, audio=audio, text=refined, gather=gather)["hidden"]
        for b in range(hg.BLOCKS):
            table = torch.frombuffer(bytearray((bundle / entry["blocks"].split(":", 1)[1].format(block=b)).read_bytes()),
                                     dtype=torch.bfloat16).view(6, 3 * k, hg.HIDDEN)
            x = body(hidden=x, table=table, tags=adaln, cos=cos_r.float(), sin=sin_r.float(), **body_weights.block(b))["hidden_out"]
        o = heads(hidden=x, temb=temb, indices=tindex.to(torch.int32), video_indices=vi, audio_indices=ai)
        vv, va = o["video"].float().numpy(), o["audio"].float().numpy()

        def advance(arm, sample, velocity):
            sigmas = np.asarray(arm["sigmas"], dtype=np.float32)
            sigma = np.float32(1) - np.float32(arm["timesteps"][step])
            ratio = np.float32(sigmas[step + 1] / sigmas[step])
            denoised = (sample + (sigma * velocity).astype(np.float32)).astype(np.float32)
            return ((ratio * sample).astype(np.float32) + ((np.float32(1) - ratio) * denoised).astype(np.float32)).astype(np.float32)
        video = torch.from_numpy(advance(recipe["video"], video.numpy(), vv))
        audio = torch.from_numpy(advance(recipe["audio"], audio.numpy(), va))
        for name, value in (("velocity-video", vv), ("velocity-audio", va), ("latents-video", video.numpy()), ("latents-audio", audio.numpy())):
            (out / f"step{step}-{name}.raw").write_bytes(np.ascontiguousarray(value, dtype=np.float32).tobytes())
        print(json.dumps({"step": step, "k": k}), flush=True)
    (out / "spec.json").write_text(json.dumps({"prompt": PROMPT, "seed": args.seed, "steps": args.steps, "ids": ids,
                                               "profile": profile["key"], "recipe": h3["defaultRecipe"], "rows": R}, indent=1))
    print(json.dumps({"done": str(out)}), flush=True)


if __name__ == "__main__":
    main()

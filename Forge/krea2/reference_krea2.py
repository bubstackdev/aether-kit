#!/usr/bin/env python3
"""CPU float32 Diffusers reference for Krea 2 Turbo (development tooling only).

    Forge/.venv/bin/python -I Forge/krea2/reference_krea2.py --variant vanilla --output <new dir>
    Forge/.venv/bin/python -I Forge/krea2/reference_krea2.py --variant realism --output <new dir>

Runs the unmodified `Krea2Pipeline` (pinned Diffusers) on CPU in float32 at 512x512, 8 steps, guidance 0, with the
initial noise drawn from Kit's seeded stream (`seeded_noise`, packed-token order) so a Swift driver can replay it from
the seed alone. The realism variant applies the exact merge the exporter uses (`krea2_graphs.merge`).

Memory: the transformer is 12.8 B parameters (25.6 GB bf16; 51 GB as float32, too much beside everything else on a
64 GB Mac). It stays resident in its stored bf16, and each of the 28 blocks is widened to float32 (merged, for
realism) just before it runs and released after, so every product is float32 with the exact checkpoint weights. The
text encoder runs first in float32 and is freed before the transformer loads.

Records (all float32 `.npy` unless noted) in `--output`:
    reference.json                 prompt, seed, size, schedule, token ids, shapes, timings, peak RSS, file hashes
    tokens.json                    padded ids and mask as the pipeline builds them, and the unpadded real ids
    text_embeds.npy                [T, 30720] conditioning, real rows only
    noise.npy                      [N, 64] initial packed latents
    step{s}.npz                    latents_in, timestep, cond [8, 6144], image_{1,2,3} [N, 6144], text_{1,2,3}
                                   [T, 6144] (outputs of blocks 6/13/20), velocity [N, 64], latents_out
    image.npy                      [3, H, W] decoded in [-1, 1]; image.png
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import krea2_graphs as kg  # noqa: E402
from export_krea2 import (ADAPTER_SHA256, CHECKPOINT_SHA256, DEFAULT_SNAPSHOT, DIFFUSERS_COMMIT,  # noqa: E402
                          FORGE_INPUTS, ADAPTER_FILE, REPO, REVISION, sha256)

PROMPT = "a tall clock tower in a snowy old town at night, lanterns glowing"
SEED = 17017
SIZE = 512


def peak_rss_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)


def stream_float32(transformer, adapter) -> None:
    """Small modules widen (and merge) once; each transformer block widens and merges just for its own forward."""
    for name, p in transformer.named_parameters():
        if not name.startswith("transformer_blocks."):
            p.data = kg.merge(name, p.data, adapter)
    for i, block in enumerate(transformer.transformer_blocks):
        prefix = f"transformer_blocks.{i}."

        def widen(module, args, prefix=prefix):
            module._stored = {}
            for n, p in module.named_parameters():
                module._stored[n] = p.data
                p.data = kg.merge(prefix + n, p.data, adapter)

        def restore(module, args, output):
            for n, p in module.named_parameters():
                p.data = module._stored[n]
            module._stored = None

        block.register_forward_pre_hook(widen)
        block.register_forward_hook(restore)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("vanilla", "realism"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--adapter", type=Path, default=FORGE_INPUTS / ADAPTER_FILE)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--steps", type=int, default=kg.STEPS)
    parser.add_argument("--allow-unpinned", action="store_true", help="random-weight test snapshots only")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    args.output.mkdir(parents=True)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, Krea2Pipeline, Krea2Transformer2DModel
    from transformers import AutoTokenizer, Qwen3VLModel

    root = args.snapshot.resolve()
    started = time.monotonic()
    timings = {}
    for relative, expected in CHECKPOINT_SHA256.items():
        if args.allow_unpinned:
            break
        if not relative.endswith(".safetensors") and sha256(root / relative) != expected:
            raise SystemExit(f"{relative} differs from the pinned file")
    adapter = None
    if args.variant == "realism":
        if sha256(args.adapter) != ADAPTER_SHA256:
            raise SystemExit("not the pinned realism adapter")
        adapter = kg.LoKr(args.adapter)

    # 1. Text: the pipeline's own padded encode, float32.
    t0 = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(root / "scheduler")
    text_encoder = Qwen3VLModel.from_pretrained(root / "text_encoder", torch_dtype=torch.float32).eval()
    pipe = Krea2Pipeline(scheduler=scheduler, vae=None, text_encoder=text_encoder, tokenizer=tokenizer,
                         transformer=None, text_encoder_select_layers=kg.TEXT_TAPS, is_distilled=True, patch_size=2)
    prefix = pipe.prompt_template_encode_start_idx
    max_len = 512 + prefix - pipe.prompt_template_encode_num_suffix_tokens
    padded = tokenizer([pipe.prompt_template_encode_prefix + args.prompt], truncation=True, padding="max_length",
                       max_length=max_len, return_tensors="pt")
    suffix = tokenizer([pipe.prompt_template_encode_suffix], return_tensors="pt")
    ids = torch.cat([padded.input_ids, suffix.input_ids], 1)[0]
    mask = torch.cat([padded.attention_mask, suffix.attention_mask], 1)[0].bool()
    real_ids = ids[mask].tolist()
    prompt_embeds, prompt_mask = pipe.encode_prompt(args.prompt, device=torch.device("cpu"))
    text_rows = int(prompt_mask.sum())
    text_embeds = prompt_embeds[0][prompt_mask[0]].flatten(1).contiguous()
    np.save(args.output / "text_embeds.npy", text_embeds.numpy())
    (args.output / "tokens.json").write_text(json.dumps({"padded": ids.tolist(), "mask": mask.int().tolist(),
                                                         "real": real_ids}) + "\n")
    del pipe, text_encoder
    timings["text"] = round(time.monotonic() - t0, 1)
    rss_text = peak_rss_gb()
    print(f"[reference] text rows {text_rows}, {timings['text']} s, peak {rss_text} GB", file=sys.stderr, flush=True)

    # 2. Denoise: the pipeline with prompt embeddings, Kit's noise and per-block float32 streaming.
    t0 = time.monotonic()
    transformer = Krea2Transformer2DModel.from_pretrained(root / "transformer", torch_dtype=torch.bfloat16).eval()
    stream_float32(transformer, adapter)
    vae = AutoencoderKLQwenImage.from_pretrained(root / "vae", torch_dtype=torch.float32).eval()
    timings["loadTransformer"] = round(time.monotonic() - t0, 1)
    rows = columns = args.size // kg.PIXELS_PER_TOKEN
    tokens = rows * columns
    noise = torch.from_numpy(kg.seeded_noise(args.seed, tokens * kg.LATENT_CHANNELS)).reshape(tokens, kg.LATENT_CHANNELS)
    np.save(args.output / "noise.npy", noise.numpy())

    steps: list[dict] = []
    padded_text = prompt_embeds.shape[1]
    real_rows = prompt_mask[0]

    def split(h: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        h = h[0].float()
        return h[padded_text:].numpy().copy(), h[:padded_text][real_rows].numpy().copy()

    def pre(module, args_, kwargs):
        steps.append({"latents_in": kwargs["hidden_states"][0].float().numpy().copy(),
                      "timestep": kwargs["timestep"].float().numpy().copy()})

    def post(module, args_, kwargs, output):
        steps[-1]["velocity"] = output[0][0].float().numpy().copy()

    transformer.register_forward_pre_hook(pre, with_kwargs=True)
    transformer.register_forward_hook(post, with_kwargs=True)
    for k, block in ((1, 6), (2, 13), (3, 20)):
        def carry(module, args_, output, k=k):
            steps[-1][f"image_{k}"], steps[-1][f"text_{k}"] = split(output)
        transformer.transformer_blocks[block].register_forward_hook(carry)
    captured = {}
    transformer.time_embed.register_forward_hook(lambda m, a, o: captured.__setitem__("temb", o.float()))
    transformer.time_mod_proj.register_forward_hook(lambda m, a, o: captured.__setitem__("mod", o.float()))

    def on_step(pipeline, index, timestep, kwargs):
        temb, mod = captured["temb"].reshape(1, -1), captured["mod"].reshape(6, -1)
        steps[index]["cond"] = torch.cat([mod, temb + transformer.final_layer.scale_shift_table.float()]).numpy().copy()
        steps[index]["latents_out"] = kwargs["latents"][0].float().numpy().copy()
        print(f"[reference] step {index} t={float(timestep):.3f} {round(time.monotonic() - t0, 1)} s "
              f"peak {peak_rss_gb()} GB", file=sys.stderr, flush=True)
        return {}

    pipe = Krea2Pipeline(scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=tokenizer, transformer=transformer,
                         text_encoder_select_layers=kg.TEXT_TAPS, is_distilled=True, patch_size=2)
    t0 = time.monotonic()
    image = pipe(prompt_embeds=prompt_embeds, prompt_embeds_mask=prompt_mask, latents=noise[None],
                 height=args.size, width=args.size, num_inference_steps=args.steps, guidance_scale=0.0,
                 output_type="pt", callback_on_step_end=on_step, callback_on_step_end_tensor_inputs=["latents"],
                 return_dict=False)[0]
    timings["denoiseAndDecode"] = round(time.monotonic() - t0, 1)
    sigmas = pipe.scheduler.sigmas.float().tolist()
    for s, record in enumerate(steps):
        np.savez(args.output / f"step{s}.npz", **record)
    decoded = (image[0].float() * 2 - 1).numpy()           # postprocess is x / 2 + 0.5 of the clamped decode
    np.save(args.output / "image.npy", decoded)
    from PIL import Image
    Image.fromarray((image[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(args.output / "image.png")

    kit_sigmas, kit_timesteps = kg.schedule(args.steps)
    record = {
        "schema": 1, "variant": args.variant, "source": {"repository": REPO, "revision": REVISION},
        "diffusersCommit": DIFFUSERS_COMMIT,
        "adapter": None if adapter is None else {"sha256": ADAPTER_SHA256, "scale": 1.0, "multiplier": 1.0},
        "prompt": args.prompt, "seed": args.seed, "noise": "splitmix64BoxMuller packedToken float32",
        "height": args.size, "width": args.size, "steps": args.steps, "guidance": 0.0, "mu": kg.MU,
        "imageTokens": tokens, "textRows": text_rows, "renderedTokens": len(real_ids),
        "sigmas": sigmas, "timesteps": [float(s["timestep"][0]) for s in steps],
        "kitSigmas": kit_sigmas, "kitTimesteps": kit_timesteps,
        "kitSigmaMaxAbsDiff": max(abs(a - b) for a, b in zip(sigmas, kit_sigmas)),
        "precision": "float32 CPU; transformer stored bf16, widened per block (exact weights)",
        "torchThreads": torch.get_num_threads(),
        "timingsSeconds": {**timings, "total": round(time.monotonic() - started, 1)},
        "peakRSSGB": {"text": rss_text, "total": peak_rss_gb()},
    }
    record["files"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir())
                       if p.is_file()}
    (args.output / "reference.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("variant", "textRows", "imageTokens", "kitSigmaMaxAbsDiff",
                                             "timingsSeconds", "peakRSSGB")}, indent=1))


if __name__ == "__main__":
    main()

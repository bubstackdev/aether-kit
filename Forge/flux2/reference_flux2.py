#!/usr/bin/env python3
"""CPU float32 Diffusers reference for FLUX.2 [klein] 4B (development tooling only).

    Forge/.venv/bin/python -I Forge/flux2/reference_flux2.py --output <new dir>

Runs the unmodified `Flux2KleinPipeline` (pinned Diffusers) on CPU in float32 at 512x512, 4 steps, no guidance, with
the initial noise drawn from Kit's seeded stream (`seeded_noise`, packed-token order) so a Swift driver can replay it
from the seed alone. The text encoder runs first in float32 and is freed before the transformer loads. Then the VAE
encoder (the reference-image path) encodes the decoded image as `_encode_vae_image` does.

Records (all float32 `.npy` unless noted) in `--output`:
    reference.json                 prompt, seed, size, schedule, shapes, timings, peak RSS, file hashes
    tokens.json                    the pipeline's padded ids and mask, and the real ids
    text_embeds.npy                [512, 7680] conditioning, every row (padding rows condition the denoiser too)
    noise.npy                      [N, 128] initial packed latents
    step{s}.npz                    latents_in, timestep, cond [17, 3072], image_{1,2} [N, 3072], text_{1,2} [512, 3072]
                                   (outputs of blocks 8 and 16 of the 25), velocity [N, 128], latents_out
    image.npy                      [3, H, W] decoded in [-1, 1]; image.png
    encoder_latents.npy            [H/16, W/16, 128] `_encode_vae_image` of image.npy, batch-normalized
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
import flux2_graphs as fg  # noqa: E402
from export_flux2 import CHECKPOINT_SHA256, DEFAULT_SNAPSHOT, DIFFUSERS_COMMIT, REPO, REVISION, sha256  # noqa: E402

PROMPT = "a red fox sitting in a snowy birch forest at dawn, soft light, detailed fur"
SEED = 20260
SIZE = 512
# Segment boundaries as block indices over the 5 double + 20 single blocks: carries k = 1, 2 are the outputs of
# block SEGMENTS[k][0] - 1 (single blocks 2 and 10).
CARRY_SINGLE_BLOCKS = {1: fg.SEGMENTS[1][0] - 1 - fg.DOUBLE_BLOCKS, 2: fg.SEGMENTS[2][0] - 1 - fg.DOUBLE_BLOCKS}


def peak_rss_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--steps", type=int, default=fg.STEPS)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    args.output.mkdir(parents=True)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline, Flux2Transformer2DModel
    from diffusers.pipelines.flux2.pipeline_flux2_klein import retrieve_latents
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    root = args.snapshot.resolve()
    started = time.monotonic()
    timings = {}
    for relative, expected in CHECKPOINT_SHA256.items():
        if not relative.endswith(".safetensors") and sha256(root / relative) != expected:
            raise SystemExit(f"{relative} differs from the pinned file")

    # 1. Text: the pipeline's own padded encode, float32.
    t0 = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(root / "scheduler")
    text_encoder = Qwen3ForCausalLM.from_pretrained(root / "text_encoder", torch_dtype=torch.float32).eval()
    pipe = Flux2KleinPipeline(scheduler=scheduler, vae=None, text_encoder=text_encoder, tokenizer=tokenizer,
                              transformer=None, is_distilled=True)
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    padded = tokenizer(rendered, return_tensors="pt", padding="max_length", truncation=True, max_length=fg.MAX_TOKENS)
    ids, mask = padded.input_ids[0], padded.attention_mask[0]
    prompt_embeds, text_ids = pipe.encode_prompt(prompt=args.prompt, device=torch.device("cpu"),
                                                 max_sequence_length=fg.MAX_TOKENS)
    np.save(args.output / "text_embeds.npy", prompt_embeds[0].float().numpy())
    (args.output / "tokens.json").write_text(json.dumps({"padded": ids.tolist(), "mask": mask.tolist(),
                                                         "real": ids[mask.bool()].tolist(),
                                                         "rendered": rendered}) + "\n")
    del pipe, text_encoder
    timings["text"] = round(time.monotonic() - t0, 1)
    rss_text = peak_rss_gb()
    print(f"[reference] real tokens {int(mask.sum())}, {timings['text']} s, peak {rss_text} GB", file=sys.stderr,
          flush=True)

    # 2. Denoise: the pipeline with prompt embeddings and Kit's noise.
    t0 = time.monotonic()
    transformer = Flux2Transformer2DModel.from_pretrained(root / "transformer", torch_dtype=torch.float32).eval()
    vae = AutoencoderKLFlux2.from_pretrained(root / "vae", torch_dtype=torch.float32).eval()
    timings["loadTransformer"] = round(time.monotonic() - t0, 1)
    rows = columns = args.size // fg.PIXELS_PER_TOKEN
    tokens = rows * columns
    noise = torch.from_numpy(fg.seeded_noise(args.seed, tokens * fg.LATENT_CHANNELS)).reshape(tokens, fg.LATENT_CHANNELS)
    np.save(args.output / "noise.npy", noise.numpy())
    spatial = noise.T.reshape(fg.LATENT_CHANNELS, rows, columns)[None].contiguous()    # what `_pack_latents` undoes

    steps: list[dict] = []
    text_rows = prompt_embeds.shape[1]
    captured: dict[str, torch.Tensor] = {}

    def pre(module, args_, kwargs):
        steps.append({"latents_in": kwargs["hidden_states"][0].float().numpy().copy(),
                      "timestep": kwargs["timestep"].float().numpy().copy()})

    def post(module, args_, kwargs, output):
        steps[-1]["velocity"] = output[0][0].float().numpy().copy()
        hidden = fg.HIDDEN
        steps[-1]["cond"] = torch.cat([captured["img"].reshape(6, hidden), captured["txt"].reshape(6, hidden),
                                       captured["single"].reshape(3, hidden), captured["out"].reshape(2, hidden)]
                                      ).float().numpy().copy()

    transformer.register_forward_pre_hook(pre, with_kwargs=True)
    transformer.register_forward_hook(post, with_kwargs=True)
    for k, block in CARRY_SINGLE_BLOCKS.items():
        def carry(module, args_, output, k=k):
            h = (output if isinstance(output, torch.Tensor) else output[-1])[0].float()
            steps[-1][f"text_{k}"], steps[-1][f"image_{k}"] = h[:text_rows].numpy().copy(), h[text_rows:].numpy().copy()
        transformer.single_transformer_blocks[block].register_forward_hook(carry)
    for key, module in (("img", transformer.double_stream_modulation_img), ("txt", transformer.double_stream_modulation_txt),
                        ("single", transformer.single_stream_modulation), ("out", transformer.norm_out.linear)):
        module.register_forward_hook(lambda m, a, o, key=key: captured.__setitem__(key, o.float()))

    def on_step(pipeline, index, timestep, kwargs):
        steps[index]["latents_out"] = kwargs["latents"][0].float().numpy().copy()
        print(f"[reference] step {index} t={float(timestep):.3f} {round(time.monotonic() - t0, 1)} s "
              f"peak {peak_rss_gb()} GB", file=sys.stderr, flush=True)
        return {}

    pipe = Flux2KleinPipeline(scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=tokenizer,
                              transformer=transformer, is_distilled=True)
    t0 = time.monotonic()
    image = pipe(prompt_embeds=prompt_embeds, latents=spatial, height=args.size, width=args.size,
                 num_inference_steps=args.steps, guidance_scale=1.0, output_type="pt", callback_on_step_end=on_step,
                 callback_on_step_end_tensor_inputs=["latents"], return_dict=False)[0]
    timings["denoiseAndDecode"] = round(time.monotonic() - t0, 1)
    sigmas = pipe.scheduler.sigmas.float().tolist()
    for s, record in enumerate(steps):
        np.savez(args.output / f"step{s}.npz", **record)
    decoded = (image[0].float() * 2 - 1).numpy()           # postprocess is x / 2 + 0.5 of the clamped decode
    np.save(args.output / "image.npy", decoded)
    from PIL import Image
    Image.fromarray((image[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(args.output / "image.png")

    # 3. The reference-image encoder: `_encode_vae_image` on the decoded picture.
    t0 = time.monotonic()
    pixels = torch.from_numpy(decoded)[None]
    latents = Flux2KleinPipeline._patchify_latents(retrieve_latents(vae.encode(pixels), sample_mode="argmax"))
    mean = vae.bn.running_mean.view(1, -1, 1, 1)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    np.save(args.output / "encoder_latents.npy", ((latents - mean) / std)[0].permute(1, 2, 0).float().numpy())
    timings["encode"] = round(time.monotonic() - t0, 1)

    kit_sigmas, kit_timesteps, mu = fg.schedule(args.steps, tokens)
    record = {
        "schema": 1, "source": {"repository": REPO, "revision": REVISION}, "diffusersCommit": DIFFUSERS_COMMIT,
        "prompt": args.prompt, "seed": args.seed, "noise": "splitmix64BoxMuller packedToken float32",
        "height": args.size, "width": args.size, "steps": args.steps, "guidance": None, "mu": mu,
        "imageTokens": tokens, "textRows": text_rows, "realTokens": int(mask.sum()),
        "sigmas": sigmas, "timesteps": [float(s["timestep"][0]) for s in steps],
        "kitSigmas": kit_sigmas, "kitTimesteps": kit_timesteps,
        "kitSigmaMaxAbsDiff": max(abs(a - b) for a, b in zip(sigmas, kit_sigmas)),
        "precision": "float32 CPU", "torchThreads": torch.get_num_threads(),
        "timingsSeconds": {**timings, "total": round(time.monotonic() - started, 1)},
        "peakRSSGB": {"text": rss_text, "total": peak_rss_gb()},
    }
    record["files"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir())
                       if p.is_file()}
    (args.output / "reference.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("realTokens", "imageTokens", "mu", "kitSigmaMaxAbsDiff",
                                             "timingsSeconds", "peakRSSGB")}, indent=1))


if __name__ == "__main__":
    main()

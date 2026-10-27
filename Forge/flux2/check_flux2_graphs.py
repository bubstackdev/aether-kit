#!/usr/bin/env python3
"""Check the FLUX.2 export graphs against the unmodified upstream modules in float32 on CPU (development tooling only).

    <venv>/bin/python -I Forge/flux2/check_flux2_graphs.py --workdir <new dir>                 # random weights
    <venv>/bin/python -I Forge/flux2/check_flux2_graphs.py --workdir <new dir> --checkpoint <snapshot>

Without `--checkpoint` it first writes a random-weight snapshot into `<workdir>/checkpoint` from the upstream classes
(real layer counts and widths, except 2 denoiser heads and a 256-wide text encoder; random norm gains and batch-norm
statistics, so the rotary permutation and every normalization are exercised). That snapshot also serves as an
`export_flux2.py --allow-unpinned` input.

Compared, upstream first so only one copy of each component is resident:
- the three denoiser segments chained against `Flux2Transformer2DModel.forward`, text-to-image and with a reference
  grid at t = 10 appended (velocity rows of every image token);
- the text encoder against `Qwen3ForCausalLM` hidden states 9/18/27 as `Flux2KleinPipeline._get_qwen3_prompt_embeds`
  stacks them, with a right-padded mask;
- the VAE decoder graph against the pipeline's decode tail and the encoder graph against `_encode_vae_image`, both
  with float32 and with the exported fp16 weight storage.

Writes `<workdir>/graph-check.json`; exits nonzero when a float32 comparison falls below 0.99999 cosine.
"""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flux2_graphs import (SEGMENTS, Checkpoint, DenoiserSegment, TextEncoderGraph, VAEDecoderGraph,  # noqa: E402
                          VAEEncoderGraph, grid_ids, load_vae, rotary_table, text_ids)

BAR = 0.99999


def metrics(expected: torch.Tensor, got: torch.Tensor) -> dict:
    e, g = expected.double().ravel(), got.double().ravel()
    cos = float(e @ g / (e.norm() * g.norm() + 1e-30))
    return {"cosine": round(cos, 8), "maxAbs": float((e - g).abs().max()), "scale": float(e.abs().max()),
            "shape": list(got.shape)}


def write_random_checkpoint(root: Path, bundle_tokenizer: Path | None) -> None:
    from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    transformer = Flux2Transformer2DModel(num_layers=5, num_single_layers=20, num_attention_heads=2,
                                          joint_attention_dim=7680, guidance_embeds=False)
    text = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=256, intermediate_size=512, num_hidden_layers=28,
                                        num_attention_heads=2, num_key_value_heads=1, head_dim=128, rms_norm_eps=1e-6,
                                        rope_theta=1_000_000.0, max_position_embeddings=1024, tie_word_embeddings=False))
    vae = AutoencoderKLFlux2()
    with torch.no_grad():
        for model in (transformer, text, vae):
            for name, p in model.named_parameters():
                if "norm" in name and p.ndim == 1:
                    p.copy_(1 + 0.3 * torch.randn_like(p))
                elif p.ndim >= 2:
                    p.mul_(0.5)
        vae.bn.running_mean.copy_(0.2 * torch.randn_like(vae.bn.running_mean))
        vae.bn.running_var.copy_(0.5 + torch.rand_like(vae.bn.running_var))
    transformer.save_pretrained(root / "transformer")
    text.save_pretrained(root / "text_encoder")
    vae.save_pretrained(root / "vae")
    if bundle_tokenizer is not None:
        shutil.copytree(bundle_tokenizer / "tokenizer", root / "tokenizer")
        shutil.copyfile(bundle_tokenizer / "LICENSE.md", root / "LICENSE.md")


def check_denoiser(checkpoint: Checkpoint) -> dict:
    from diffusers import Flux2Transformer2DModel

    g = torch.Generator().manual_seed(1)
    text, rows, columns = 24, 4, 6
    cases = {"text-to-image": [grid_ids(rows, columns)],
             "reference": [grid_ids(rows, columns), grid_ids(3, 4, t=10)]}
    cfg = checkpoint.config("transformer")
    inputs = {}
    for name, grids in cases.items():
        ids = torch.cat(grids)
        inputs[name] = (torch.randn(ids.shape[0], 128, generator=g), torch.randn(text, cfg["joint_attention_dim"], generator=g),
                        torch.tensor([0.82], dtype=torch.float32), ids)
    model = Flux2Transformer2DModel.from_pretrained(checkpoint.root / "transformer", torch_dtype=torch.float32).eval()
    upstream = {}
    with torch.no_grad():
        for name, (latents, embeds, sigma, ids) in inputs.items():
            upstream[name] = model(hidden_states=latents[None], encoder_hidden_states=embeds[None], timestep=sigma,
                                   img_ids=ids[None], txt_ids=text_ids(text)[None], guidance=None,
                                   return_dict=False)[0][0]
    del model
    carries = {name: (latents, embeds, sigma, rotary_table(text_ids(text)), rotary_table(ids))
               for name, (latents, embeds, sigma, ids) in inputs.items()}
    cond = {}
    with torch.no_grad():
        for index in range(len(SEGMENTS)):
            segment = DenoiserSegment.load(checkpoint, index, dtype=torch.float32)
            for name, args in carries.items():
                out = segment(*args)
                if index == 0:
                    cond[name] = out[2]
                carries[name] = out if index == len(SEGMENTS) - 1 else (out[0], out[1], cond[name], args[3], args[4])
            del segment
    return {name: metrics(upstream[name], carries[name]) for name in cases}


def check_text(checkpoint: Checkpoint) -> dict:
    from transformers import Qwen3ForCausalLM

    length, real = 48, 19
    ids = torch.full((1, length), 151643, dtype=torch.int32)
    ids[0, :real] = torch.tensor([151644, 872, 198] + [3000 + (131 * i) % 90000 for i in range(real - 3)])
    mask = torch.zeros((1, length), dtype=torch.int32)
    mask[0, :real] = 1
    model = Qwen3ForCausalLM.from_pretrained(checkpoint.root / "text_encoder", torch_dtype=torch.float32).eval()
    with torch.no_grad():
        out = model(input_ids=ids.long(), attention_mask=mask.long(), output_hidden_states=True, use_cache=False)
        stacked = torch.stack([out.hidden_states[k] for k in (9, 18, 27)], dim=1)
        upstream = stacked.permute(0, 2, 1, 3).reshape(1, length, -1)[0]
    del model, out
    graph = TextEncoderGraph.load(checkpoint, dtype=torch.float32)
    with torch.no_grad():
        got = graph(ids, torch.arange(length, dtype=torch.int32)[None], mask)
    return {"allRows": metrics(upstream, got), "realRows": metrics(upstream[:real], got[:real])}


def check_vae(checkpoint: Checkpoint) -> dict:
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline, retrieve_latents

    vae = load_vae(checkpoint)
    g = torch.Generator().manual_seed(2)
    rows, columns = 4, 6
    latents = torch.randn(1, 128, rows, columns, generator=g).half().float()
    pixels = torch.rand(1, 3, 16 * rows, 16 * columns, generator=g) * 2 - 1
    mean = vae.bn.running_mean.view(1, -1, 1, 1)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    with torch.no_grad():
        decoded = vae.decode(Flux2KleinPipeline._unpatchify_latents(latents * std + mean), return_dict=False)[0][0]
        encoded = Flux2KleinPipeline._patchify_latents(retrieve_latents(vae.encode(pixels), sample_mode="argmax"))
        encoded = ((encoded - mean) / std)[0].permute(1, 2, 0)
        result = {}
        for storage in ("float32", "float16"):
            half = storage == "float16"
            result[f"decoder-{storage}"] = metrics(decoded, VAEDecoderGraph(vae, half)(latents[0].permute(1, 2, 0).half()))
            result[f"encoder-{storage}"] = metrics(encoded, VAEEncoderGraph(vae, half)(pixels[0]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="real snapshot (float32 copies; about 20 GB peak RSS)")
    parser.add_argument("--tokenizer-from", type=Path, help="bundle whose tokenizer/ and LICENSE.md the random snapshot copies")
    args = parser.parse_args()
    if args.workdir.exists():
        raise SystemExit("workdir already exists; choose a new directory")
    args.workdir.mkdir(parents=True)
    torch.set_grad_enabled(False)
    root = args.checkpoint
    if root is None:
        root = args.workdir / "checkpoint"
        write_random_checkpoint(root, args.tokenizer_from)
    checkpoint = Checkpoint(root.resolve())
    report = {"checkpoint": str(checkpoint.root), "random": args.checkpoint is None, "bar": BAR,
              "denoiser": check_denoiser(checkpoint), "text": check_text(checkpoint), "vae": check_vae(checkpoint)}
    report["peakRSSGB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
    gated = [v for group in ("denoiser", "text") for v in report[group].values()]
    gated += [v for k, v in report["vae"].items() if k.endswith("float32")]
    report["passed"] = all(v["cosine"] >= BAR for v in gated)
    (args.workdir / "graph-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))
    if not report["passed"]:
        raise SystemExit("graph check failed")


if __name__ == "__main__":
    np.set_printoptions(precision=6)
    main()

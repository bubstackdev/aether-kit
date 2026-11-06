#!/usr/bin/env python3
"""Random-weight structural check of the Krea 2 export graphs against the unmodified upstream modules (CPU, fp32).

    Forge/.venv/bin/python -I Forge/krea2/selftest_krea2_graphs.py --workdir <new dir> [--convert]

Writes a small random snapshot (the real layer counts, 128-wide heads with GQA, narrow widths; random norm gains so
the rotary permutation and every normalization are exercised), then compares in float32:
- the four chained denoiser segments on *unpadded* text rows against `Krea2Transformer2DModel.forward` on the
  pipeline's padded `[real | PAD]` rows with its key-padding mask (the equivalence the graphs rely on);
- the text encoder graph against `Qwen3VLModel` hidden states at Krea's taps (rows 34..);
- the VAE decoder graph against the pipeline's decode tail (`_unpack_latents`, denormalize, `vae.decode`).
`--convert` also exports each tiny graph through coreai-torch and runs it cpu_only. A development aid, not the gate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import krea2_graphs as kg  # noqa: E402

BAR = 0.99999


def metrics(expected, got) -> dict:
    e, g = expected.double().ravel(), got.double().ravel()
    return {"cosine": round(float(e @ g / (e.norm() * g.norm() + 1e-30)), 8), "maxAbs": float((e - g).abs().max()),
            "scale": float(e.abs().max())}


def randomize(model) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 1 or "scale_shift_table" in name:
                p.copy_(0.3 * torch.randn_like(p))
            elif p.ndim >= 2:
                p.mul_(0.7)


def write_checkpoint(root: Path) -> None:
    from diffusers import AutoencoderKLQwenImage, Krea2Transformer2DModel
    from transformers import Qwen3VLConfig, Qwen3VLModel

    torch.manual_seed(0)
    transformer = Krea2Transformer2DModel(num_attention_heads=2, num_key_value_heads=1, intermediate_size=384,
                                          text_hidden_dim=256, text_num_attention_heads=2, text_num_key_value_heads=2,
                                          text_intermediate_size=320)
    text_cfg = dict(hidden_size=256, intermediate_size=512, num_hidden_layers=36, num_attention_heads=2,
                    num_key_value_heads=1, head_dim=128, vocab_size=151936, rms_norm_eps=1e-6,
                    rope_parameters={"rope_type": "default", "rope_theta": 5000000, "mrope_section": [24, 20, 20],
                                     "mrope_interleaved": True}, tie_word_embeddings=True)
    vision_cfg = dict(depth=2, hidden_size=64, intermediate_size=128, num_heads=2, out_hidden_size=256,
                      deepstack_visual_indexes=[0, 1])
    text = Qwen3VLModel(Qwen3VLConfig(text_config=text_cfg, vision_config=vision_cfg))
    vae = AutoencoderKLQwenImage()
    for model in (transformer, text):
        randomize(model)
    transformer.save_pretrained(root / "transformer")
    text.save_pretrained(root / "text_encoder")
    vae.save_pretrained(root / "vae")


def check_denoiser(checkpoint: kg.Checkpoint, convert_dir: Path | None) -> dict:
    from diffusers import Krea2Pipeline, Krea2Transformer2DModel

    g = torch.Generator().manual_seed(1)
    cfg = checkpoint.config("transformer")
    rows, columns, real, padded = 4, 6, 9, 16
    n = rows * columns
    latents = torch.randn(n, kg.LATENT_CHANNELS, generator=g)
    embeds = torch.randn(padded, len(kg.TEXT_TAPS), cfg["text_hidden_dim"], generator=g)
    mask = torch.zeros(padded, dtype=torch.bool)
    mask[:real] = True
    sigma = torch.tensor([0.71])
    model = Krea2Transformer2DModel.from_pretrained(checkpoint.root / "transformer", torch_dtype=torch.float32).eval()
    with torch.no_grad():
        ids = Krea2Pipeline.prepare_position_ids(padded, rows, columns, "cpu")
        upstream = model(hidden_states=latents[None], encoder_hidden_states=embeds[None], timestep=sigma,
                         position_ids=ids, encoder_attention_mask=mask[None], return_dict=False)[0][0]
    del model
    rope = kg.rotary_table(rows, columns)
    carry = (latents, embeds[:real].flatten(1), sigma, rope)
    cond = None
    with torch.no_grad():
        for index in range(len(kg.SEGMENTS)):
            seg = kg.DenoiserSegment.load(checkpoint, index, compute=torch.float32, strict=False)
            out = seg(*carry)
            if convert_dir is not None:
                convert_and_run(seg, carry, seg.input_names, seg.output_names, convert_dir / f"denoiser{index}.aimodel",
                                out)
            if index == 0:
                cond = out[2]
            carry = out if index == len(kg.SEGMENTS) - 1 else (out[0], out[1], cond, rope)
    return metrics(upstream, carry)


def check_text(checkpoint: kg.Checkpoint, convert_dir: Path | None) -> dict:
    from transformers import Qwen3VLModel

    length = 48
    ids = torch.arange(3000, 3000 + length, dtype=torch.int32)[None]
    positions = torch.arange(length, dtype=torch.int32)[None]
    model = Qwen3VLModel.from_pretrained(checkpoint.root / "text_encoder", torch_dtype=torch.float32).eval()
    with torch.no_grad():
        out = model(input_ids=ids.long(), attention_mask=torch.ones_like(ids, dtype=torch.bool),
                    position_ids=positions.long().unsqueeze(0).expand(3, -1, -1), output_hidden_states=True)
        upstream = torch.stack([out.hidden_states[i] for i in kg.TEXT_TAPS], dim=2)[0, kg.DROP_ROWS:].flatten(1)
    del model
    graph = kg.TextEncoderGraph.load(checkpoint, compute=torch.float32, strict=False)
    with torch.no_grad():
        got = graph(ids, positions).float()
    if convert_dir is not None:
        convert_and_run(graph, (ids, positions), ["input_ids", "position_ids"], ["text_embeds"],
                        convert_dir / "text.aimodel", got, dynamic=kg.TextEncoderGraph.example()[1])
    return metrics(upstream, got)


def check_vae(checkpoint: kg.Checkpoint, convert_dir: Path | None) -> dict:
    from diffusers import Krea2Pipeline

    vae = kg.load_vae(checkpoint)
    g = torch.Generator().manual_seed(2)
    rows, columns = 4, 6
    packed = torch.randn(rows * columns, kg.LATENT_CHANNELS, generator=g).half().float()
    pipe = Krea2Pipeline.__new__(Krea2Pipeline)
    pipe.patch_size, pipe.vae_scale_factor = 2, 8
    with torch.no_grad():
        x = pipe._unpack_latents(packed[None], rows * 16, columns * 16)
        mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        inv_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
        upstream = vae.decode(x / inv_std + mean, return_dict=False)[0][0, :, 0]
        result = {}
        for half in (False, True):
            graph = kg.VAEDecoderGraph(vae, half).eval()
            got = graph(packed.reshape(rows, columns, -1).half())
            result["float16" if half else "float32"] = metrics(upstream, got)
            if half and convert_dir is not None:
                convert_and_run(graph, (packed.reshape(rows, columns, -1).half(),), ["latents"], ["image"],
                                convert_dir / "vae.aimodel", got, dynamic=kg.VAEDecoderGraph.example()[1], run=False)
    return result


def convert_and_run(module, example, inputs, outputs, path, expected, dynamic=None, run=True) -> None:
    import coreai_torch as ct
    import coreai.runtime as rt

    if dynamic is None:
        dynamic = module.example()[1]
    with torch.no_grad():
        ep = torch.export.export(module, tuple(example), dynamic_shapes=dynamic, strict=False)
        ep = ep.run_decompositions(ct.get_decomp_table())
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    converter.add_exported_program(ep, input_names=inputs, output_names=outputs, state_names=[],
                                   entrypoint_name="main")
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(path)
    print(f"converted {path.name}", file=sys.stderr)
    if not run:
        return

    async def go():
        model = await rt.AIModel.load(path, rt.SpecializationOptions.cpu_only())
        fn = model.load_function("main")
        feed = {k: rt.NDArray(np.ascontiguousarray(v.detach().numpy())) for k, v in zip(inputs, example)}
        return {k: v.numpy().copy() for k, v in (await fn(feed, {})).items()}

    got = asyncio.run(go())
    expected = expected if isinstance(expected, tuple) else (expected,)
    for name, value in zip(outputs, expected):
        m = metrics(value.float(), torch.from_numpy(got[name].astype(np.float32)))
        print(f"  cpu_only {path.stem}.{name}: {m}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--convert", action="store_true")
    args = parser.parse_args()
    if args.workdir.exists():
        raise SystemExit("workdir already exists; choose a new directory")
    args.workdir.mkdir(parents=True)
    torch.set_grad_enabled(False)
    root = args.workdir / "checkpoint"
    write_checkpoint(root)
    checkpoint = kg.Checkpoint(root)
    convert_dir = None
    if args.convert:
        convert_dir = args.workdir / "converted"
        convert_dir.mkdir()
    report = {"denoiser": check_denoiser(checkpoint, convert_dir), "text": check_text(checkpoint, convert_dir),
              "vae": check_vae(checkpoint, convert_dir)}
    gated = [report["denoiser"], report["text"], report["vae"]["float32"]]
    report["passed"] = all(v["cosine"] >= BAR for v in gated)
    print(json.dumps(report, indent=1))
    if not report["passed"]:
        raise SystemExit("graph self-test failed")


if __name__ == "__main__":
    main()

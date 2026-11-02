"""Export real SD-Turbo stages and independent Diffusers trajectory fixtures (CPU only)."""

import argparse, gc, hashlib, importlib.metadata, json, platform, shutil, sys
from pathlib import Path
import numpy as np
import torch
from diffusers import UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Forge/diffusion"))
from graphs import TextEncoder, EpsilonDenoiser, LatentDecoder

REPO = "stabilityai/sd-turbo"
REVISION = "b261bac6fd2cf515557d5d0707481eafa0485ec2"


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert(module, example, inputs, output, entry, path):
    import coreai_torch as ct

    with torch.no_grad():
        ep = torch.export.export(module, example, strict=False).run_decompositions(
            ct.get_decomp_table()
        )
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    converter.add_exported_program(
        ep,
        input_names=inputs,
        output_names=[output],
        state_names=[],
        entrypoint_name=entry,
    )
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(path)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--reference-only", action="store_true")
    args = p.parse_args()
    root = args.checkpoint.resolve()
    out = args.output.resolve()
    if out.exists():
        raise SystemExit("Choose a new output directory")
    if args.size < 64 or args.size > 512 or args.size % 64:
        raise SystemExit("size must be a multiple of 64 in 64...512")
    expected = json.loads(
        Path(__file__).with_name("checkpoint-sha256.json").read_text()
    )
    for name, sha in expected.items():
        if digest(root / name) != sha:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    out.mkdir(parents=True)
    bundle = out / "bundle"
    bundle.mkdir()
    fixtures = out / "reference"
    fixtures.mkdir()
    torch.set_num_threads(8)
    torch.manual_seed(17)
    report = {
        "source": {"repository": REPO, "revision": REVISION},
        "host": platform.platform(),
        "versions": {
            n: importlib.metadata.version(n)
            for n in ["torch", "transformers", "diffusers", "coreai-torch"]
        },
        "checkpointSHA256": {
            str(f.relative_to(root)): digest(f)
            for f in root.rglob("*")
            if f.is_file() and ".cache" not in f.parts
        },
        "scriptSHA256": {
            str(f.relative_to(Path(__file__).resolve().parents[2])): digest(f)
            for f in [
                Path(__file__).resolve(),
                Path(__file__).resolve().parents[2] / "Forge/diffusion/graphs.py",
            ]
        },
        "stages": {},
    }

    def save():
        (out / "prepare-report.json").write_text(json.dumps(report, indent=2) + "\n")

    def stage(name, module, example, names, result, entry):
        print("Export " + name, flush=True)
        if not args.reference_only:
            convert(
                module.eval(),
                example,
                names,
                result,
                entry,
                bundle / (name + ".aimodel"),
            )
        report["stages"][name] = (
            "reference only" if args.reference_only else "converted"
        )
        save()

    tokenizer = CLIPTokenizerFast.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    tokenizer.save_pretrained(bundle / "tokenizer")
    tokenizer_prompts = [
        "",
        "Hello WORLD!",
        "A cat's blue eyes; we're here.",
        "café naïve – 雪 🦊",
        "line one\n\tline two  1234",
        "cafe\u0301",
        " ".join(["mountain"] * 100),
        "<|startoftext|>cat<|endoftext|>",
    ]
    tokenizer_cases = [
        {
            "prompt": p,
            "ids": tokenizer(
                p, padding="max_length", max_length=77, truncation=True
            ).input_ids,
        }
        for p in tokenizer_prompts
    ]
    (fixtures / "tokenizer-cases.json").write_text(
        json.dumps(tokenizer_cases, indent=2) + "\n"
    )
    prompts = [
        "A small red wooden cabin beside a lake, mountains, daylight.",
        "A blue ceramic teapot on a wooden table.",
        "A lighthouse on a rocky coast at sunrise.",
        "A watercolor painting of an orange fox in a snowy forest.",
    ]
    cases = []
    text = CLIPTextModel.from_pretrained(
        root / "text_encoder",
        variant="fp16",
        torch_dtype=torch.float32,
        local_files_only=True,
        attn_implementation="eager",
    ).eval()
    for i, prompt in enumerate(prompts):
        ids = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(torch.int32)
        embeds = text(ids, return_dict=False)[0]
        case = fixtures / str(i)
        case.mkdir()
        ids.numpy().tofile(case / "ids.i32")
        embeds.numpy().tofile(case / "text.f32")
        cases.append(
            {
                "id": str(i),
                "prompt": prompt,
                "steps": [1, 4, 2, 3][i],
                "ids": ids.flatten().tolist(),
            }
        )
    stage(
        "textEncoder", TextEncoder(text), (ids,), ["input_ids"], "text_embeds", "encode"
    )
    del text
    gc.collect()
    unet = UNet2DConditionModel.from_pretrained(
        root / "unet", variant="fp16", torch_dtype=torch.float32, local_files_only=True
    ).eval()
    from diffusers.models.attention_processor import AttnProcessor2_0

    unet.set_attn_processor(AttnProcessor2_0())
    side = args.size // 8
    for c in cases:
        case = fixtures / c["id"]
        scheduler = EulerDiscreteScheduler.from_pretrained(
            root / "scheduler", local_files_only=True
        )
        if (
            scheduler.config.prediction_type != "epsilon"
            or scheduler.config.timestep_spacing != "trailing"
        ):
            raise ValueError("Unsupported scheduler")
        scheduler.set_timesteps(c["steps"])
        noise = torch.randn(
            1, 4, side, side, generator=torch.Generator().manual_seed(42)
        )
        noise.numpy().tofile(case / "noise.f32")
        x = noise * scheduler.init_noise_sigma
        embeds = torch.from_numpy(
            np.fromfile(case / "text.f32", dtype=np.float32).reshape(1, 77, 1024)
        )
        c["timesteps"] = scheduler.timesteps.tolist()
        c["sigmas"] = scheduler.sigmas.tolist()
        c["initialNoiseSigma"] = float(scheduler.init_noise_sigma)
        for k, t in enumerate(scheduler.timesteps):
            x.numpy().tofile(case / f"latent-{k}.f32")
            sample = scheduler.scale_model_input(x, t)
            epsilon = unet(sample, t, encoder_hidden_states=embeds, return_dict=False)[
                0
            ]
            epsilon.numpy().tofile(case / f"epsilon-{k}.f32")
            x = scheduler.step(epsilon, t, x, return_dict=False)[0]
        x.numpy().tofile(case / "final.f32")
        print("Reference denoise " + c["id"], flush=True)
    stage(
        "denoiser",
        EpsilonDenoiser(unet),
        (sample, torch.tensor([999.0]), embeds),
        ["sample", "timestep", "text_embeds"],
        "epsilon",
        "denoise",
    )
    del unet
    gc.collect()
    vae = AutoencoderKL.from_pretrained(
        root / "vae", variant="fp16", torch_dtype=torch.float32, local_files_only=True
    ).eval()
    for c in cases:
        case = fixtures / c["id"]
        x = torch.from_numpy(
            np.fromfile(case / "final.f32", dtype=np.float32).reshape(1, 4, side, side)
        )
        image = vae.decode(x / vae.config.scaling_factor, return_dict=False)[0][0]
        image.numpy().tofile(case / "image.f32")
        from PIL import Image

        rgb = (
            ((image / 2 + 0.5).clamp(0, 1) * 255)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .numpy()
        )
        Image.fromarray(rgb).save(case / "reference.png")
    stage("decoder", LatentDecoder(vae), (x,), ["latents"], "image", "decode")
    del vae
    gc.collect()
    manifest = {
        "schema": 1,
        "family": "sd-turbo",
        "source": report["source"],
        "width": args.size,
        "height": args.size,
        "latentChannels": 4,
        "latentScale": 8,
        "textLength": 77,
        "textWidth": 1024,
        "scalarType": "float32",
        "schedules": {
            str(c["steps"]): {
                "timesteps": c["timesteps"],
                "sigmas": c["sigmas"],
                "initialNoiseSigma": c["initialNoiseSigma"],
            }
            for c in cases
        },
    }
    # Capture all supported schedules directly from upstream; the runtime never guesses scheduler semantics.
    for steps in (1, 2, 3, 4):
        s = EulerDiscreteScheduler.from_pretrained(
            root / "scheduler", local_files_only=True
        )
        s.set_timesteps(steps)
        manifest["schedules"][str(steps)] = {
            "timesteps": s.timesteps.tolist(),
            "sigmas": s.sigmas.tolist(),
            "initialNoiseSigma": float(s.init_noise_sigma),
        }
    shutil.copyfile(root / "LICENSE.md", bundle / "LICENSE.md")
    manifest["files"] = {
        str(f.relative_to(bundle)): digest(f) for f in bundle.rglob("*") if f.is_file()
    }
    (bundle / "diffusion.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (fixtures / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    save()


if __name__ == "__main__":
    main()

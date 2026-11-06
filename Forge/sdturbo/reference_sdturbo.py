#!/usr/bin/env python3
"""CPU float32 Diffusers reference for SD-Turbo (development tooling only).

    Forge/.venv/bin/python -I Forge/sdturbo/reference_sdturbo.py --output <new dir>

Runs the unmodified `StableDiffusionPipeline` (pinned Diffusers) on CPU in float32 at 512x512, guidance 0, with the
initial latents drawn from Kit's seeded stream (`seeded_noise`, NCHW order) so the Swift driver can replay a case from
its seed alone. A forward hook on the U-Net records each step's inputs and epsilon for the teacher-forced graph check.

Records in `--output`:
    reference.json        cases (prompt, seed, steps, ids, schedule table), tokenizer cases, versions, file hashes
    <case>/text_embeds.npy [1, 77, 1024]   noise.npy [1, 4, 64, 64] (unscaled)
    <case>/step{k}.npz     sample (scaled model input), timestep, epsilon, latents_in, latents_out
    <case>/final.npy       final latents     image.npy [3, H, W] decoded on [-1, 1]     image.png
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sdturbo_graphs as sg  # noqa: E402

CASES = [
    {"id": "teapot-1", "prompt": "A blue ceramic teapot on a wooden table.", "seed": 42, "steps": 1},
    {"id": "fox-4", "prompt": "A watercolor painting of an orange fox in a snowy forest.", "seed": 7, "steps": 4},
]
TOKENIZER_PROMPTS = [
    "", "Hello WORLD!", "A cat's blue eyes; we're here.", "café naïve – 雪 🦊", "line one\n\tline two  1234",
    "café", " ".join(["mountain"] * 100), "<|startoftext|>cat<|endoftext|>",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, default=sg.DEFAULT_SNAPSHOT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    root = args.snapshot.resolve()
    for relative, expected in sg.CHECKPOINT_SHA256.items():
        if sha256(root / relative) != expected:
            raise SystemExit(f"{relative} differs from the pinned {sg.REVISION[:12]} file")
    args.output.mkdir(parents=True)
    torch.set_num_threads(8)
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(root, variant="fp16", torch_dtype=torch.float32,
                                                   local_files_only=True, safety_checker=None,
                                                   requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    record = {"versions": {p: importlib.metadata.version(p) for p in ("torch", "diffusers", "transformers")},
              "source": {"repository": sg.REPO, "revision": sg.REVISION}, "size": sg.SIZE, "cases": [],
              "tokenizerCases": [{"prompt": p, "ids": sg.tokenize(pipe.tokenizer, p)} for p in TOKENIZER_PROMPTS]}
    tables = sg.schedule_tables(root)
    shape = sg.latent_shape(sg.SIZE)
    for case in CASES:
        started = time.monotonic()
        out = args.output / case["id"]
        out.mkdir()
        ids = sg.tokenize(pipe.tokenizer, case["prompt"])
        embeds = pipe.text_encoder(torch.tensor([ids]), return_dict=False)[0]
        np.save(out / "text_embeds.npy", embeds.numpy())
        noise = torch.from_numpy(sg.seeded_noise(case["seed"], int(np.prod(shape))).reshape(shape))
        np.save(out / "noise.npy", noise.numpy())
        steps = []

        def hook(module, inputs, kwargs, output):
            steps.append({"sample": inputs[0].numpy().copy(), "timestep": np.float32(inputs[1]),
                          "epsilon": output[0].numpy().copy()})

        handle = pipe.unet.register_forward_hook(hook, with_kwargs=True)
        trajectory = []

        def on_step(pipeline, i, t, kwargs):
            trajectory.append(kwargs["latents"].numpy().copy())
            return kwargs

        final = pipe(case["prompt"], height=sg.SIZE, width=sg.SIZE, num_inference_steps=case["steps"],
                     guidance_scale=0.0, latents=noise.clone(), output_type="latent",
                     callback_on_step_end=on_step, callback_on_step_end_tensor_inputs=["latents"]).images
        handle.remove()
        if len(steps) != case["steps"] or len(trajectory) != case["steps"]:
            raise SystemExit(f"expected {case['steps']} U-Net calls, saw {len(steps)}")
        latents_in = noise.numpy() * np.float32(tables[str(case["steps"])]["initNoiseSigma"])
        for k, s in enumerate(steps):
            np.savez(out / f"step{k}.npz", latents_in=latents_in, latents_out=trajectory[k], **s)
            latents_in = trajectory[k]
        np.save(out / "final.npy", final.numpy())
        image = pipe.vae.decode(final / pipe.vae.config.scaling_factor, return_dict=False)[0][0]
        np.save(out / "image.npy", image.numpy())
        from PIL import Image
        rgb = ((image / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
        Image.fromarray(rgb).save(out / "image.png")
        record["cases"].append({**case, "ids": ids, "schedule": tables[str(case["steps"])],
                                "seconds": round(time.monotonic() - started, 1)})
        print(f"[reference] {case['id']} done in {record['cases'][-1]['seconds']} s", file=sys.stderr, flush=True)
    record["peakRSSGB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
    record["files"] = {str(p.relative_to(args.output)): sha256(p) for p in sorted(args.output.rglob("*"))
                       if p.is_file()}
    (args.output / "reference.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

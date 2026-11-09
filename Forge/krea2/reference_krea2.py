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
  refine (unless --no-refine): `image.npy` encoded by the source VAE, then the remaining steps from a start step:
    encoder_latents.npy            [N, 64] `vae.encode(image).latent_dist.mode()`, normalized, packed (`_pack_latents`)
    refine_noise.npy               [N, 64] Kit's seeded noise for the refine seed
    refine_start.npy               [N, 64] `(1 - sigma) * x0 + sigma * noise` at the start step's sigma
    refine_image.npy               [3, H, W] the refined decode in [-1, 1]; refine_image.png
  references (`--references <0.0.x fixture dir>`, refine skipped): the reference cases, each through transformers'
  unmodified `Qwen3VLModel` in float32 with its pixels (upstream Krea 2 has no image conditioning: the images are
  extra text rows, the 0.0.x reference contract), then the pipeline from one case's conditioning (`--reference-case`):
    cases.json                     per case: prompt, grids, rows, the fixture checks (ids, rope, pixels sha256 and
                                   the oracle files' sha256 matches), the tap-order finding
    cases/<id>/patches.npy         [P, 1536] the processor's patch rows (block-major), grids.json (t, h, w) per image
    cases/<id>/ids.npy, graph_ids.npy, rope.npy   int32 [L], [L] (pads as extension ids), [3, L]
    cases/<id>/image_embeds.npy    [n, 2560] merged tokens of all images in order; deepstack_embeds.npy [3, n, 2560]
    cases/<id>/conditioning.npy    [L - 34, 30720] the 12 taps (rows 34..)
    text_embeds.npy, step*.npz, image.npy   as above, for `--reference-case`
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


def refine_reference(pipe, transformer, vae, prompt_embeds, prompt_mask, decoded: np.ndarray, rows: int, columns: int,
                     args) -> dict:
    """Image-to-image from `image.npy`: the source VAE's posterior mode, normalized (the inverse of the decode tail's
    `x * std + mean`) and packed by `_pack_latents`; `(1 - sigma) * x0 + sigma * noise` at the start step's sigma
    (flow matching); then the pipeline's own transformer and scheduler steps from there, and the pipeline's decode."""
    from diffusers.pipelines.krea2.pipeline_krea2 import retrieve_timesteps

    seed = args.seed + 1 if args.refine_seed is None else args.refine_seed
    tokens = rows * columns
    z_dim = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1)
    picture = torch.from_numpy(decoded).float()
    mode = vae.encode(picture[None, :, None]).latent_dist.mode()[:, :, 0]
    x0 = pipe._pack_latents((mode - mean) / std, 1, z_dim, 2 * rows, 2 * columns)[0].float()          # [N, 64]
    noise = torch.from_numpy(kg.seeded_noise(seed, tokens * kg.LATENT_CHANNELS)).reshape(tokens, kg.LATENT_CHANNELS)
    sigmas = np.linspace(1.0, 1 / args.steps, args.steps)
    timesteps, _ = retrieve_timesteps(pipe.scheduler, args.steps, torch.device("cpu"), sigmas=sigmas, mu=kg.MU)
    sigma = float(pipe.scheduler.sigmas[args.refine_step])
    kit_sigma = kg.schedule(args.steps)[0][args.refine_step]
    if abs(sigma - kit_sigma) > 1e-6:
        raise SystemExit(f"the pipeline's sigma at step {args.refine_step} is {sigma}, Kit's schedule {kit_sigma}")
    latents = ((1 - sigma) * x0 + sigma * noise)[None]
    np.save(args.output / "encoder_latents.npy", x0.numpy())
    np.save(args.output / "refine_noise.npy", noise.numpy())
    np.save(args.output / "refine_start.npy", latents[0].numpy())
    pipe.scheduler.set_begin_index(args.refine_step)
    position_ids = pipe.prepare_position_ids(prompt_embeds.shape[1], rows, columns, torch.device("cpu"))
    for i in range(args.refine_step, args.steps):
        t = timesteps[i]
        timestep = (t / pipe.scheduler.config.num_train_timesteps).expand(1).to(latents.dtype)
        velocity = transformer(hidden_states=latents, encoder_hidden_states=prompt_embeds, timestep=timestep,
                               position_ids=position_ids, encoder_attention_mask=prompt_mask, return_dict=False)[0]
        latents = pipe.scheduler.step(velocity, t, latents, return_dict=False)[0]
    unpacked = pipe._unpack_latents(latents, rows * kg.PIXELS_PER_TOKEN, columns * kg.PIXELS_PER_TOKEN)
    picture = vae.decode(unpacked * std[..., None] + mean[..., None], return_dict=False)[0][:, :, 0]
    picture = pipe.image_processor.postprocess(picture, output_type="pt")[0].float()
    np.save(args.output / "refine_image.npy", (picture * 2 - 1).numpy())
    from PIL import Image
    Image.fromarray((picture.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(
        args.output / "refine_image.png")
    print(f"[reference] refine from step {args.refine_step} (sigma {sigma:.4f}), seed {seed}", file=sys.stderr, flush=True)
    return {"source": "image.npy", "startStep": args.refine_step, "startSigma": sigma, "seed": seed,
            "steps": args.steps, "start": "(1 - sigma) * x0 + sigma * noise", "encoder": "posterior mode"}


def reference_cases(model, tokenizer, args) -> dict:
    """Each 0.0.x fixture case through `Qwen3VLModel` (float32, with pixels): the oracle tensors, checked against the
    fixture's ids, rope and the recorded sha256s of the oracle it was made from."""
    from PIL import Image
    from transformers import Qwen2VLImageProcessorFast

    fixture = json.loads((args.references / "fixture.json").read_text())
    recorded = fixture["sha256"]
    processor = Qwen2VLImageProcessorFast(size={"shortest_edge": 65536, "longest_edge": 16777216}, patch_size=16,
                                          temporal_patch_size=2, merge_size=2, image_mean=[0.5] * 3,
                                          image_std=[0.5] * 3)
    sha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()  # noqa: E731
    out = {}
    for line in (args.references / fixture["cases"]).read_text().splitlines():
        case = json.loads(line)
        cid, files = case["id"], case["files"]
        t0 = time.monotonic()
        images = [Image.open(args.references / f).convert("RGB") for f in files["images"]]
        batch = processor(images=images, return_tensors="pt")
        pixels, thw = batch["pixel_values"].float(), batch["image_grid_thw"]
        grids = [tuple(int(v) for v in g) for g in thw.tolist()]
        prompt = kg.reference_prompt(tokenizer, case["prompt"], grids)
        ids = np.array(prompt["ids"], dtype=np.int32)
        rope = np.array(prompt["rope"], dtype=np.int32)
        fixture_ids = np.fromfile(args.references / files["ids"], dtype="<i4")
        fixture_rope = np.fromfile(args.references / files["rope"], dtype="<i4")
        input_ids = torch.from_numpy(ids.astype(np.int64))[None]
        types = (input_ids == kg.IMAGE_PAD_ID).int()
        upstream_rope = model.get_rope_index(input_ids, types, thw)[0][:, 0].numpy().astype(np.int32)
        run = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), pixel_values=pixels,
                    image_grid_thw=thw, mm_token_type_ids=types, output_hidden_states=True)
        hidden = run.hidden_states
        taps = torch.stack([hidden[i] for i in kg.TEXT_TAPS], dim=2)[0].flatten(1).float()     # [L, 30720]
        features = model.get_image_features(pixels, thw)
        image_embeds = torch.cat(features.pooler_output).float()
        deepstack = torch.stack([torch.cat(level) for level in features.deepstack_features]).float()
        # The other tap order (DeepStack added before a layer's output is recorded): only tap 2 (hidden_states[2], the
        # second layer's output) can differ, by level 1 at the image rows.
        image_rows = torch.from_numpy(ids == kg.IMAGE_PAD_ID)
        other = taps.clone()
        width = image_embeds.shape[1]
        tap2 = kg.TEXT_TAPS.index(2)
        other[image_rows, tap2 * width:(tap2 + 1) * width] += deepstack[1]
        directory = args.output / "cases" / cid
        directory.mkdir(parents=True)
        arrays = {"patches": pixels.numpy(), "ids": ids, "graph_ids": np.array(prompt["graphIds"], dtype=np.int32),
                  "rope": rope.reshape(3, -1), "image_embeds": image_embeds.numpy(),
                  "deepstack_embeds": deepstack.numpy(), "conditioning": taps[kg.DROP_ROWS:].numpy()}
        for name, value in arrays.items():
            np.save(directory / f"{name}.npy", value)
        (directory / "grids.json").write_text(json.dumps(grids) + "\n")

        def matches(key: str, *candidates) -> str | None:
            want = recorded.get(f"{cid}/{key}")
            return next((label for label, value in candidates if sha(value) == want), None)

        def cosine(a, b):
            a, b = a.double().ravel(), b.double().ravel()
            return round(float(a @ b / (a.norm() * b.norm())), 8)

        record = {
            "prompt": case["prompt"], "grids": grids, "rows": len(ids) - kg.DROP_ROWS, "ids": prompt["ids"],
            "truncated": prompt["truncated"], "imageTokens": [s[1] - s[0] for s in prompt["spans"]],
            "fixtureGrids": case["grids"] == [list(g) for g in grids],
            "idsExact": bool(np.array_equal(ids, fixture_ids)),
            "ropeExact": bool(np.array_equal(rope.ravel(), fixture_rope)),          # the fixture is flat [3 * L]
            "ropeMatchesUpstream": bool(np.array_equal(rope.reshape(3, -1), upstream_rope)),
            "pixelsSHA256Match": sha(pixels.numpy().astype("<f4")) == recorded.get(f"{cid}/pixels.f32"),
            # Which recorded oracle file this regeneration reproduces bit for bit (None: none; versions differ).
            "oracleSHA256Match": {
                "text": matches("text.f32", ("taps", taps.numpy().astype("<f4")),
                                ("tapsFromRow34", taps[kg.DROP_ROWS:].numpy().astype("<f4")),
                                ("otherOrder", other.numpy().astype("<f4")),
                                ("otherOrderFromRow34", other[kg.DROP_ROWS:].numpy().astype("<f4"))),
                "image": matches("image.f32", ("imageEmbeds", image_embeds.numpy().astype("<f4"))),
                "deepstack": matches("deepstack.f32", ("deepstack", deepstack.numpy().astype("<f4")),
                                     ("deepstackTokenMajor", deepstack.permute(1, 0, 2).numpy().astype("<f4"))),
            },
            "tapOrder": {"recorded": "hidden_states[k] before DeepStack level k is added (transformers)",
                         "otherOrderTap2Cosine": cosine(taps[image_rows, tap2 * width:(tap2 + 1) * width],
                                                        other[image_rows, tap2 * width:(tap2 + 1) * width])},
            "seconds": round(time.monotonic() - t0, 1),
        }
        out[cid] = record
        print(f"[reference] case {cid}: rows {record['rows']} ids {record['idsExact']} rope {record['ropeExact']} "
              f"pixels {record['pixelsSHA256Match']} oracle {record['oracleSHA256Match']} "
              f"({record['seconds']} s, peak {peak_rss_gb()} GB)", file=sys.stderr, flush=True)
    (args.output / "cases.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


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
    parser.add_argument("--refine-step", type=int, default=kg.STEPS - 1,
                        help="the step a refine of image.npy starts from (default the last: sigma about 0.31)")
    parser.add_argument("--refine-seed", type=int, help="the refine noise's seed (default seed + 1)")
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--references", type=Path,
                        help="the 0.0.x reference fixture (cases.jsonl, fixture.json, <case>/image*.png, ids, rope)")
    parser.add_argument("--reference-case", default="one-square", help="the case the image is rendered from")
    parser.add_argument("--allow-unpinned", action="store_true", help="random-weight test snapshots only")
    args = parser.parse_args()
    if args.references is not None:
        args.no_refine = True
    if not args.no_refine and not 1 <= args.refine_step < args.steps:
        parser.error("--refine-step must be in 1 ..< steps (step 0 would ignore the picture)")
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
    cases = None
    if args.references is not None:
        cases = reference_cases(text_encoder, tokenizer, args)
        chosen = cases[args.reference_case]
        args.prompt = chosen["prompt"]
        cond = torch.from_numpy(np.load(args.output / "cases" / args.reference_case / "conditioning.npy"))
        prompt_embeds = cond.reshape(1, cond.shape[0], len(kg.TEXT_TAPS), -1)
        prompt_mask = torch.ones(1, cond.shape[0], dtype=torch.bool)
        real_ids = chosen["ids"]
        ids, mask = torch.tensor(real_ids), torch.ones(len(real_ids), dtype=torch.bool)
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
    main_timesteps = [float(s["timestep"][0]) for s in steps]
    decoded = (image[0].float() * 2 - 1).numpy()           # postprocess is x / 2 + 0.5 of the clamped decode
    np.save(args.output / "image.npy", decoded)
    from PIL import Image
    Image.fromarray((image[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(args.output / "image.png")

    refine = None
    if not args.no_refine:
        t0 = time.monotonic()
        refine = refine_reference(pipe, transformer, vae, prompt_embeds, prompt_mask, decoded, rows, columns, args)
        timings["refine"] = round(time.monotonic() - t0, 1)

    kit_sigmas, kit_timesteps = kg.schedule(args.steps)
    record = {
        "schema": 1, "variant": args.variant, "source": {"repository": REPO, "revision": REVISION},
        "diffusersCommit": DIFFUSERS_COMMIT,
        "adapter": None if adapter is None else {"sha256": ADAPTER_SHA256, "scale": 1.0, "multiplier": 1.0},
        "prompt": args.prompt, "seed": args.seed, "noise": "splitmix64BoxMuller packedToken float32",
        "height": args.size, "width": args.size, "steps": args.steps, "guidance": 0.0, "mu": kg.MU,
        "imageTokens": tokens, "textRows": text_rows, "renderedTokens": len(real_ids),
        **({"references": {"fixture": str(args.references), "case": args.reference_case,
                           "cases": {k: {x: v[x] for x in v if x != "ids"} for k, v in cases.items()}}}
           if cases is not None else {}),
        "sigmas": sigmas, "timesteps": main_timesteps, "refine": refine,
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

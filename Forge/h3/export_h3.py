"""Export MiniMax H3 (FL2VA + the Turbo 8-step adapter) as an Aether-Kit bundle (MANIFEST.md §5.8, §5.9 proposed).

    Forge/.venv/bin/python -I Forge/h3/export_h3.py --output <dir> --stage <stage> [...]
    Forge/.venv/bin/python -I Forge/h3/export_h3.py --sample-manifest > Tests/AetherCoreTests/Fixtures/h3/minimax-h3.json

Stages write into OUTPUT/bundle (the installed-revision layout: `.aether/aether.json`, models/, weights/, tables/,
tokenizer/, the license files) and record what they made in OUTPUT/work/build.json; OUTPUT/work also holds the
converted VAEs. Each stage checks its graph on the CPU against the original modules before converting it, as the
0.0.x exporter did, and refuses to start when free disk is below --minimum-free-gb.

  convert-vae      Convert the video and audio VAEs to Diffusers layout (the pinned converter) under OUTPUT/work.
  body             ONE transformer-block graph for all 50 blocks: weights w0…w21 are inputs (the Turbo LoRA slots
                   included), rows and the timestep cardinality (k = 1…3, table [6, 3k, D]) dynamic. CPU check on 128
                   rows of real block-0 weights with zero LoRA: byte-identical to the PEFT block.
  body-weights     Each block's 22 inputs (base from the converter, LoRA from the adapter through PEFT, in the body
                   graph's input order) written to weights/body/block-NN.bin; the table is weights/body.json.
  tables           Per recipe, task and step: the timestep embedding [k, 2688] f32 and the 50 blocks' AdaLN tables
                   [6, 3k, 5376] bf16, from the original modules (the AdaLN branches carry no LoRA).
  heads            The output norm and both heads (rows, k, video and audio row counts dynamic).
  prepare          The per-render text graph (context_embedder + the token refiner with its LoRA, weights lifted to
                   weights/prepare-text.bin) and the per-step pack graph.
  conditioner      ONE Qwen3-VL decoder-layer graph (11 weights lifted) for layers 0…49, their weights in
                   weights/conditioner/layer-NN.bin, the token-embedding table copied verbatim as a row source, and the
                   text rotary tables cos/sin [4096, 128] bf16. Layers 0 and 49 run from their weight files on the CPU.
  audio-decoder    The audio decoder at each profile's fixed latent count (the dynamic one aborts in MPSGraph).
  video-decoder    The bf16 window-tile decoder: packed rows [448, 96] f32 → 22 frames [1, 3, 22, 256, 256] f32.
  video-encoder    The first-frame encoder as one 256 px tile graph and a tail per canvas (fl2va keyframes).
  vision           The Qwen3-VL vision tower per canvas (fl2va keyframe presentation), weights lifted.
  package          Tokenizer, license files, NOTICE, `.aether/aether.json`, SHA256SUMS and export-report.json.

--random (body, body-weights only): seeded random weights in the real shapes, for the device-chain gate; nothing is
read from the checkpoint and the result is not a model.
"""
from __future__ import annotations

import argparse
import json
import resource
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h3_graphs as hg  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots" / hg.REVISION
PROFILES = [hg.Geometry(960, 544, 120), hg.Geometry(544, 960, 120)]
CANVASES = sorted({f"{p.width}x{p.height}" for p in PROFILES})
NOTICE_LINE = "MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, Copyright © 2026 MiniMax. All Rights Reserved."
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja")


def peak_rss_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)


class Build:
    """OUTPUT/work/build.json: what each stage made, merged stage by stage."""

    def __init__(self, output: Path):
        self.output = output
        self.bundle, self.work = output / "bundle", output / "work"
        self.bundle.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)
        self.path = self.work / "build.json"
        self.meta = json.loads(self.path.read_text()) if self.path.exists() else {
            "schema": 1, "source": hg.REPO, "revision": hg.REVISION, "diffusersCommit": hg.DIFFUSERS_COMMIT}

    def record(self, stage: str, **fields) -> None:
        self.meta[stage] = {**fields, "peakRSSGB": peak_rss_gb()}
        self.path.write_text(json.dumps(self.meta, indent=1, sort_keys=True))
        print(json.dumps({"stage": stage, "peakRSSGB": peak_rss_gb()}), flush=True)

    def room(self, gigabytes: float, floor: float) -> None:
        free = shutil.disk_usage(self.output).free / 1e9
        if free - gigabytes < floor:
            raise SystemExit(f"stopping: {free:.1f} GB free, this stage needs ~{gigabytes:.1f} GB, floor {floor} GB")


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def random_block(seed: int):
    """A block in the real shapes with seeded random weights (and LoRA slots at 1e-3), for the chain gate only."""
    model = hg.meta_model(num_layers=1)
    block = model.transformer_blocks[0].to_empty(device="cpu")
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in block.named_parameters():
            if "norm" in name:
                p.copy_(1 + 0.05 * torch.randn(p.shape, generator=generator))
            else:
                p.copy_(torch.randn(p.shape, generator=generator) / p.shape[-1] ** 0.5)
    block = block.to(torch.bfloat16)
    hg.zero_turbo(model)
    with torch.no_grad():
        for name, p in block.named_parameters():
            if ".lora_" in name:
                p.copy_((1e-3 * torch.randn(p.shape, generator=generator)).to(p.dtype))
    block.adaln_proj = hg.CachedModulation()
    return model, block


def real_block(args, b: int, adapter_live: bool):
    converter, _ = hg.load_converter(args.converter)
    checkpoint = hg.Checkpoint(args.snapshot, converter)
    model = hg.meta_model()
    model.load_state_dict(checkpoint.converted(lambda n: n.startswith(f"blocks.{b}.")), strict=False, assign=True)
    if adapter_live:
        hg.load_turbo(model, hg.adapter_path(), lambda k: k.startswith(f"transformer_blocks.{b}."))
    else:
        hg.zero_turbo(model, b)
    return model, model.transformer_blocks[b]


def stage_body(args, build: Build) -> None:
    from torch.export import Dim
    model, block = random_block(1043) if args.random else real_block(args, 0, adapter_live=False)
    torch.manual_seed(1043)
    hidden, temb = torch.randn(1, 128, hg.HIDDEN, dtype=torch.bfloat16), torch.randn(2, hg.TEMB_DIM)
    tags, cos, sin = torch.randint(0, 6, (128,)), torch.randn(128, 96), torch.randn(128, 96)
    if args.random:
        table = torch.randn(6, 6, hg.HIDDEN, dtype=torch.bfloat16) * 0.1
        with torch.inference_mode():
            expected = block(hidden, table, tags, (cos, sin))
    else:
        with torch.inference_mode():
            table = torch.stack(block.adaln_proj(temb))
            expected = block(hidden, temb, tags, (cos, sin))
        block.adaln_proj = hg.CachedModulation()
    graph, params = hg.lift(hg.BlockCall(block).eval())
    with torch.inference_mode():
        actual = graph(hidden, table, tags.to(torch.int32), cos, sin, *params.values())
    names = list(params)
    check = {"rows": 128, "cardinality": 2, "byteIdentical": bool(torch.equal(actual, expected)), "bindings": names}
    assert check["byteIdentical"] and len(names) == 22, check
    rows, k = Dim("rows", min=hg.ROWS[0], max=hg.ROWS[1]), Dim("k", min=hg.CARDINALITY[0], max=hg.CARDINALITY[1])
    n = PROFILES[0].video_rows + 463
    example = (torch.zeros(1, n, hg.HIDDEN, dtype=torch.bfloat16), torch.zeros(6, 6, hg.HIDDEN, dtype=torch.bfloat16),
               torch.zeros(n, dtype=torch.int32), torch.zeros(n, 96), torch.zeros(n, 96))
    dynamic = [{1: rows}, {1: 3 * k}, {0: rows}, {0: rows}, {0: rows}] + [None] * len(names)
    inputs = ["hidden", "table", "tags", "cos", "sin"] + [f"w{i}" for i in range(len(names))]
    asset = hg.convert(graph, (*example, *params.values()), (tuple(dynamic),), inputs, ["hidden_out"],
                       build.bundle / "models/body.aimodel")
    build.record("body", asset=asset, check=check, random=args.random, rows=list(hg.ROWS), cardinality=list(hg.CARDINALITY))


def stage_body_weights(args, build: Build) -> None:
    expected = build.meta["body"]["check"]["bindings"]
    count = args.random_blocks if args.random else hg.BLOCKS
    build.room(0.75 * count, args.minimum_free_gb)
    blocks, specs = [], None
    for b in range(count):
        started = time.monotonic()
        _, block = random_block(2000 + b) if args.random else real_block(args, b, adapter_live=True)
        if not args.random:
            block.adaln_proj = hg.CachedModulation()   # the tables replace it, as in the body graph
        params = dict(hg.BlockCall(block).named_parameters())
        assert list(params) == expected, f"block {b}'s binding order differs from the body graph"
        extents, block_specs = hg.WeightFile(build.bundle / f"weights/body/block-{b:02d}.bin",
                                             f"main:weights/body/block-{b:02d}.bin").write(params, torch.bfloat16)
        assert specs is None or [s["shape"] for s in block_specs] == [s["shape"] for s in specs]
        specs = specs or block_specs
        blocks.append(extents)
        print(json.dumps({"block": b, "seconds": round(time.monotonic() - started, 1)}), flush=True)
    table = {"inputs": [f"w{i}" for i in range(len(expected))], "blocks": blocks}
    (build.bundle / "weights/body.json").write_text(json.dumps(table, indent=1))
    build.record("body-weights", blocks=count, inputs=specs, random=args.random,
                 adapter={"repo": hg.ADAPTER_REPO, "revision": hg.ADAPTER_REVISION, "sha256": hg.ADAPTER_SHA256,
                          "execution": "dynamicLora", "rank": hg.LORA_RANK, "alpha": hg.LORA_ALPHA})


def schedule_plans() -> dict:
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
    plans = {}
    for name, (points, video_shift, audio_shift) in hg.RECIPES.items():
        arms = []
        for shift in (video_shift, audio_shift):
            scheduler = MiniMaxH3Scheduler(shift=shift)
            scheduler.set_timesteps(points)
            arms.append({"shift": shift, "timesteps": scheduler.timesteps.tolist(), "sigmas": scheduler.sigmas.tolist()})
        for task, condition in (("t2va", None), ("fl2va", hg.FL2VA_CONDITION_TIMESTEP)):
            steps = [sorted({tv, ta} | ({max(tv, condition)} if condition is not None else set()))
                     for tv, ta in zip(arms[0]["timesteps"], arms[1]["timesteps"])]
            plans[(name, task)] = (arms, steps)
    return plans


def stage_tables(args, build: Build) -> None:
    converter, _ = hg.load_converter(args.converter)
    checkpoint = hg.Checkpoint(args.snapshot, converter)
    model = hg.meta_model()
    model.load_state_dict(checkpoint.converted(lambda n: n.startswith("time_embedder.")), strict=False, assign=True)
    model.time_proj = type(model.time_proj)(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
    plans, index = schedule_plans(), {}
    with torch.inference_mode():
        embeddings = {}
        for (name, task), (arms, steps) in plans.items():
            entry = index.setdefault(f"{name}/{task}", {"video": arms[0], "audio": arms[1], "steps": []})
            for i, values in enumerate(steps):
                t = torch.tensor(values, dtype=torch.float32)
                embeddings[(name, task, i)] = temb = model.time_embedder(model.time_proj(t).to(torch.float32))
                path = f"tables/{name}/{task}/step-{i}"
                entry["steps"].append({"timesteps": values, "temb": {"file": f"main:{path}/temb.raw",
                                                                     **hg.raw_tensor(build.bundle / path / "temb.raw", temb)}})
        for block in range(hg.BLOCKS):
            weights = checkpoint.converted(lambda n, b=block: n.startswith(f"blocks.{b}.adaln_proj."))
            prefix, module = f"transformer_blocks.{block}.adaln_proj.", model.transformer_blocks[block].adaln_proj
            module.load_state_dict({k.removeprefix(prefix): v for k, v in weights.items()}, strict=True, assign=True)
            for (name, task, i), temb in embeddings.items():
                if block == 0:
                    index[f"{name}/{task}"]["steps"][i]["blocks"] = f"main:tables/{name}/{task}/step-{i}/block-{{block:02d}}.raw"
                hg.raw_tensor(build.bundle / f"tables/{name}/{task}/step-{i}/block-{block:02d}.raw", torch.stack(module(temb)))
            model.transformer_blocks[block].adaln_proj.to("meta")
            print(json.dumps({"tables": block}), flush=True)
    build.record("tables", recipes=index)


def stage_heads(args, build: Build) -> None:
    from torch.export import Dim
    converter, _ = hg.load_converter(args.converter)
    model = hg.meta_model()
    model.load_state_dict(hg.Checkpoint(args.snapshot, converter).converted(lambda n: n.startswith("final_layer.")),
                          strict=False, assign=True)
    heads = hg.Heads(model).eval()
    torch.manual_seed(1043)
    r, nv, na = 640, 510, 128
    hidden, temb = torch.randn(1, r, hg.HIDDEN, dtype=torch.bfloat16), torch.randn(2, hg.TEMB_DIM)
    indices, vi, ai = torch.randint(0, 2, (r,)), torch.arange(r - nv, r), torch.arange(r - nv - na, r - nv)
    with torch.inference_mode():
        states = model.norm_out(hidden, temb, indices).to(model.proj_out.weight.dtype)
        expected = (model.proj_out(states).index_select(1, vi), model.audio_proj_out(states).index_select(1, ai))
        actual = heads(hidden, temb, indices.to(torch.int32), vi.to(torch.int32), ai.to(torch.int32))
    check = {"byteIdentical": all(torch.equal(a, e) for a, e in zip(actual, expected))}
    assert check["byteIdentical"], check
    rows, k = Dim("rows", min=hg.ROWS[0], max=hg.ROWS[1]), Dim("k", min=1, max=3)
    video, audio = Dim("video", min=32, max=hg.ROWS[1]), Dim("audio", min=2, max=2048)
    n = 19333
    example = (torch.zeros(1, n, hg.HIDDEN, dtype=torch.bfloat16), torch.zeros(2, hg.TEMB_DIM), torch.zeros(n, dtype=torch.int32),
               torch.arange(463, n, dtype=torch.int32), torch.arange(49, 463, dtype=torch.int32))
    asset = hg.convert(heads, example, ({1: rows}, {0: k}, {0: rows}, {0: video}, {0: audio}),
                       ["hidden", "temb", "indices", "video_indices", "audio_indices"], ["video", "audio"],
                       build.bundle / "models/heads.aimodel")
    build.record("heads", asset=asset, check=check)


def stage_prepare(args, build: Build) -> None:
    from torch.export import Dim
    converter, _ = hg.load_converter(args.converter)
    model = hg.meta_model()
    model.load_state_dict(hg.Checkpoint(args.snapshot, converter).converted(
        lambda n: n.startswith(("token_refiner.", "condition_proj.", "video_patch_proj.", "audio_patch_proj."))),
        strict=False, assign=True)
    hg.load_turbo(model, hg.adapter_path(), lambda k: k.startswith("token_refiner."))
    text_module, pack = hg.PrepareText(model).eval(), hg.Pack(model).eval()
    graph, params = hg.lift(text_module)
    torch.manual_seed(1043)
    T, nv, na = 49, 510, 128
    text, video, audio = torch.randn(1, T, hg.TEXT_DIM, dtype=torch.bfloat16), torch.randn(1, nv, 96), torch.randn(1, na, 32)
    order = torch.randperm(T + nv + na)
    ti, vi, ai = order[:T].sort().values, order[T:T + nv].sort().values, order[T + nv:].sort().values
    extents, specs = hg.WeightFile(build.bundle / "weights/prepare-text.bin", "main:weights/prepare-text.bin").write(params)
    lifted_from_file = [hg.read_extent(build.bundle, extents[s["name"]], getattr(torch, s["dtype"]), s["shape"]) for s in specs]
    with torch.inference_mode():
        refined = model.token_refiner(model.context_embedder(text.to(model.context_embedder.weight.dtype)))
        hidden = refined.new_zeros((1, T + nv + na, refined.shape[-1]))
        hidden = hidden.index_copy(1, ti, refined)
        hidden = hidden.index_copy(1, vi, model.proj_in(video.to(model.proj_in.weight.dtype)).to(refined.dtype))
        expected = hidden.index_copy(1, ai, model.audio_proj_in(audio.to(model.audio_proj_in.weight.dtype)).to(refined.dtype))
        actual_text = graph(text, *lifted_from_file)
        gather = torch.empty(T + nv + na, dtype=torch.int32)
        gather[torch.cat((ti, vi, ai))] = torch.arange(T + nv + na, dtype=torch.int32)
        actual = pack(video, audio, actual_text, gather)
    check = {"textByteIdentical": bool(torch.equal(actual_text, refined)), "packByteIdentical": bool(torch.equal(actual, expected))}
    assert all(check.values()), check
    (build.bundle / "weights/prepare-text.json").write_text(json.dumps({"inputs": list(extents), "blocks": [extents]}, indent=1))
    tokens, rows = Dim("tokens", min=hg.TEXT_TOKENS[0], max=hg.TEXT_TOKENS[1]), Dim("rows", min=hg.ROWS[0], max=hg.ROWS[1])
    video_rows, audio_rows = Dim("video", min=32, max=hg.ROWS[1]), Dim("audio", min=2, max=2048)
    text_asset = hg.convert(graph, (text, *params.values()), (({1: tokens},) + (None,) * len(params),),
                            ["text"] + list(extents), ["refined"], build.bundle / "models/prepare-text.aimodel")
    pack_asset = hg.convert(pack, (video, audio, actual_text, gather), ({1: video_rows}, {1: audio_rows}, {1: tokens}, {0: rows}),
                            ["video", "audio", "text", "gather"], ["hidden"], build.bundle / "models/prepare-pack.aimodel")
    build.record("prepare", text=text_asset, pack=pack_asset, textInputs=specs, check=check)


def stage_conditioner(args, build: Build) -> None:
    from torch.export import Dim
    from transformers import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer, Qwen3VLTextRotaryEmbedding
    directory = args.snapshot / "FL2VA/text_encoder"
    config = Qwen3VLConfig.from_pretrained(directory).text_config
    config._attn_implementation = "sdpa"
    build.room(52, args.minimum_free_gb)
    rotary = Qwen3VLTextRotaryEmbedding(config)
    torch.manual_seed(1043)
    probe = torch.randn(1, 16, config.hidden_size, dtype=torch.bfloat16)
    cos, sin = rotary(probe, torch.arange(16).view(1, 1, -1).expand(3, 1, -1))
    causal = torch.ones(16, 16, dtype=torch.bool).tril().view(1, 1, 16, 16)
    blocks, checks, graph, names, specs, first = [], [], None, None, None, None
    for index in range(hg.CONDITIONER_LAYERS):
        with torch.device("meta"):
            layer = Qwen3VLTextDecoderLayer(config, index)
        layer.load_state_dict(hg.sharded(directory, f"model.language_model.layers.{index}."), strict=True, assign=True)
        layer.eval()
        lifted, params = hg.lift(hg.ConditionerLayer(layer).eval())
        if names is None:
            graph, names = lifted, list(params)
        assert list(params) == names, f"layer {index}'s binding order differs"
        extents, layer_specs = hg.WeightFile(build.bundle / f"weights/conditioner/layer-{index:02d}.bin",
                                             f"main:weights/conditioner/layer-{index:02d}.bin").write(params, torch.bfloat16)
        specs = specs or layer_specs
        blocks.append(extents)
        if index in (0, hg.CONDITIONER_LAYERS - 1):
            from_file = [hg.read_extent(build.bundle, extents[s["name"]], torch.bfloat16, s["shape"]) for s in layer_specs]
            first = first or from_file
            with torch.inference_mode():
                actual = graph(probe, cos, sin, *from_file)
                expected = layer(probe, position_embeddings=(cos, sin), attention_mask=causal)
            check = {"layer": index, "byteIdentical": bool(torch.equal(actual, expected)), "relativeL2": rel_l2(actual, expected)}
            assert check["relativeL2"] < 1e-3, check
            checks.append(check)
        print(json.dumps({"conditioner": index}), flush=True)
        if index:
            del layer, lifted, params
    (build.bundle / "weights/conditioner.json").write_text(json.dumps({"inputs": [s["name"] for s in specs], "blocks": blocks}, indent=1))
    T = Dim("tokens", min=hg.TEXT_TOKENS[0], max=hg.TEXT_TOKENS[1])
    example = tuple(torch.zeros(1, 49, d, dtype=torch.bfloat16) for d in (config.hidden_size, config.head_dim, config.head_dim))
    asset = hg.convert(graph, (*example, *first), (({1: T}, {1: T}, {1: T}) + (None,) * len(names),),
                       ["hidden", "cos", "sin"] + [s["name"] for s in specs], ["hidden_out"], build.bundle / "models/conditioner-layer.aimodel")
    # The token-embedding table as a row source, copied verbatim (the host gathers a prompt's rows).
    index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    key = "model.language_model.embed_tokens.weight"
    offset, length, dtype, shape = hg.safetensors_span(directory / index[key], key)
    assert dtype == "BF16" and shape == [config.vocab_size, config.hidden_size], (dtype, shape)
    target = build.bundle / "weights/conditioner/embedding.bin"
    with open(directory / index[key], "rb") as source, open(target, "wb") as out:
        source.seek(offset)
        remaining = length
        while remaining:
            chunk = source.read(min(remaining, 1 << 26))
            out.write(chunk)
            remaining -= len(chunk)
    with torch.inference_mode():
        cos_all, sin_all = rotary(torch.zeros(1, 1, config.hidden_size, dtype=torch.bfloat16),
                                  torch.arange(hg.TEXT_TOKENS[1]).view(1, 1, -1).expand(3, 1, -1))
    rope = {"cos": hg.raw_tensor(build.bundle / "tables/conditioner/rope-cos.raw", cos_all[0]),
            "sin": hg.raw_tensor(build.bundle / "tables/conditioner/rope-sin.raw", sin_all[0])}
    build.record("conditioner", asset=asset, inputs=specs, checks=checks, layers=hg.CONDITIONER_LAYERS, tap=hg.TAP,
                 embedding={"file": "main:weights/conditioner/embedding.bin", "rows": shape[0], "width": shape[1], "dtype": "bfloat16"},
                 rope={"theta": config.rope_parameters["rope_theta"], "mropeSection": list(rotary.mrope_section),
                       "headDim": config.head_dim, "dtype": "bfloat16", **rope})


def converted_vae(build: Build, audio: bool = False):
    from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
    if audio:
        return AutoencoderKLMiniMaxH3Audio.from_pretrained(build.work / "converted/audio-vae", torch_dtype=torch.float32).eval()
    return AutoencoderKLMiniMaxH3.from_pretrained(build.work / "converted/vae", torch_dtype=torch.float32).eval()


def stage_convert_vae(args, build: Build) -> None:
    import diffusers
    converter, path = hg.load_converter(args.converter)
    video, audio = build.work / "converted/vae", build.work / "converted/audio-vae"
    if not video.exists():
        converter.convert_video_vae(str(args.snapshot / "FL2VA"), str(video), converter.MINIMAX_H3_VIDEO_VAE_CONFIG,
                                    diffusers.__version__, 5 << 30)
    if not audio.exists():
        converter.convert_audio_vae(str(args.snapshot / "FL2VA"), str(audio), diffusers.__version__)
    build.record("convert-vae", converter=str(path), converterSHA256=hg.sha256_file(path),
                 video={p.name: hg.sha256_file(p) for p in sorted(video.iterdir()) if p.is_file()},
                 audio={p.name: hg.sha256_file(p) for p in sorted(audio.iterdir()) if p.is_file()})


def stage_audio_decoder(args, build: Build) -> None:
    vae = converted_vae(build, audio=True)
    module, fixed = hg.AudioDecoder(vae).eval(), {}
    for na in sorted({p.audio_latents for p in PROFILES}):
        torch.manual_seed(1043)
        rows = torch.randn(2, na, 32)
        with torch.inference_mode():
            actual = module(rows)
            mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1)
            std = torch.tensor(vae.config.latents_std).view(1, -1, 1)
            reference = vae.decode(rows.permute(0, 2, 1).contiguous() * std + mean, return_dict=False)[0].float().squeeze(1)
        check = {"latents": na, "relativeL2": rel_l2(actual, reference), "shape": list(actual.shape)}
        assert check["relativeL2"] < 1e-5 and check["shape"] == [2, hg.AUDIO_HOP * na], check
        asset = hg.convert(module, (rows,), None, ["rows"], ["pcm"], build.bundle / f"models/audio-decoder-na{na}.aimodel")
        fixed[str(na)] = {**asset, "check": check}
    build.record("audio-decoder", fixed=fixed, sampleRate=vae.config.sampling_rate, hop=hg.AUDIO_HOP, channels=hg.AUDIO_CHANNELS)


def stage_video_decoder(args, build: Build) -> None:
    from diffusers import AutoencoderKLMiniMaxH3
    reference_vae = converted_vae(build)
    vae = AutoencoderKLMiniMaxH3.from_pretrained(build.work / "converted/vae", torch_dtype=torch.bfloat16).eval()
    reference_vae.use_tiling = vae.use_tiling = False
    module = hg.WindowTileDecoder(vae).eval()
    torch.manual_seed(1043)
    rows = torch.randn(448, 96)
    z = module.latents(rows)
    with torch.inference_mode():
        actual = module(rows)
        reference = vae._decode(z.to(torch.bfloat16)).float()
        full = reference_vae._decode(z)
    check = {"byteIdentical": bool(torch.equal(actual, reference)), "relativeL2VsFloat32": rel_l2(actual, full)}
    assert check["byteIdentical"], check
    asset = hg.convert(module, (rows,), None, ["rows"], ["pixels"], build.bundle / "models/video-decoder-tile256-bf16.aimodel")
    build.record("video-decoder", asset=asset, check=check, tile=hg.VIDEO_TILE, windowLatents=hg.WINDOW_LATENTS,
                 windowFrames=hg.WINDOW_FRAMES, windowStride=hg.WINDOW_STRIDE, precision="bfloat16")


def stage_video_encoder(args, build: Build) -> None:
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
    from diffusers.utils.torch_utils import randn_tensor
    vae = converted_vae(build)
    assert vae.use_tiling and vae.tile_sample_min_height == vae.tile_sample_min_width == hg.VIDEO_TILE
    tile_module, side, tails, checks = hg.TileEncoder(vae).eval(), hg.VIDEO_TILE, {}, {}
    for key in CANVASES:
        width, height = (int(v) for v in key.split("x"))
        ys, _, yo = vae._split_tiles(height, side, 64)
        xs, _, xo = vae._split_tiles(width, side, 64)
        torch.manual_seed(1043)
        image = torch.randint(0, 256, (1, 3, 1, height, width), dtype=torch.uint8)
        with torch.inference_mode():
            moments = torch.cat([tile_module(image[..., y:y + side, x:x + side].float()) for y in ys for x in xs])
            eps = randn_tensor((1, vae.config.latent_channels, 1, height // 16, width // 16),
                               generator=torch.Generator().manual_seed(hg.ENCODE_SEED), device=torch.device("cpu"), dtype=moments.dtype)
            tail = hg.TileTail(vae, eps, len(ys), len(xs), yo, xo).eval()
            actual = tail(moments)
            reference = encode_vae_condition(vae, image, hg.PIXEL_MEAN, hg.PIXEL_STD, hg.ENCODE_SEED)
            normalized = (image.float().div(255.0) - torch.tensor(hg.PIXEL_MEAN).view(1, -1, 1, 1, 1)) / torch.tensor(hg.PIXEL_STD).view(1, -1, 1, 1, 1)
            assert list(DiagonalGaussianDistribution(vae._encode(normalized)).mean.shape) == list(eps.shape)
        check = {"canvas": key, "tiles": len(ys) * len(xs), "yStarts": ys, "xStarts": xs, "byteIdentical": bool(torch.equal(actual, reference))}
        assert check["byteIdentical"], check
        tails[key] = hg.convert(tail, (moments,), None, ["tiles"], ["condition"], build.bundle / f"models/video-encoder-tail-{key}.aimodel")
        checks[key] = check
    tile = hg.convert(tile_module, (torch.zeros(1, 3, 1, side, side),), None, ["pixels"], ["moments"],
                      build.bundle / "models/video-encoder-tile256.aimodel")
    build.record("video-encoder", tile=tile, tails=tails, checks=checks, tileSize=side, seed=hg.ENCODE_SEED,
                 momentsChannels=2 * vae.config.latent_channels)


def stage_vision(args, build: Build) -> None:
    from transformers import Qwen3VLConfig
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (Qwen3VLVisionModel, Qwen3VLVisionRotaryEmbedding,
                                                               apply_rotary_pos_emb_vision)
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids
    directory = args.snapshot / "FL2VA/text_encoder"
    config = Qwen3VLConfig.from_pretrained(directory)
    vc = config.vision_config
    vc._attn_implementation = "sdpa"
    with torch.device("meta"):
        visual = Qwen3VLVisionModel(vc)
    visual.load_state_dict(hg.sharded(directory, "model.visual."), strict=True, assign=True)
    visual.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(vc)
    visual.eval()
    merge, patch, side = vc.spatial_merge_size, vc.patch_size, visual.num_grid_per_side

    class VisionTower(torch.nn.Module):
        """pixels [n, 1536] bf16, pos_index [n, 4] i32, pos_weight [n, 4] f32, patch_rc [n, 2] i32 (merge-window order,
        one image) → image_embeds [n/4, 5120] bf16, deepstack [3, n/4, 5120] bf16."""

        def __init__(self):
            super().__init__()
            self.visual = visual

        def forward(self, pixels, pos_index, pos_weight, patch_rc):
            v = self.visual
            x = v.patch_embed(pixels)
            x = x + (v.pos_embed(pos_index) * pos_weight[:, :, None]).sum(1).to(x.dtype)
            cos, sin = v.rotary_pos_emb(x, patch_rc)
            n, levels = x.shape[0], []
            for index, block in enumerate(v.blocks):
                attn = block.attn
                q, k, w = attn.qkv(block.norm1(x)).reshape(n, 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
                q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
                o, _ = sdpa_attention_forward(attn, q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
                                              w.transpose(0, 1).unsqueeze(0), attention_mask=None, scaling=attn.scaling,
                                              dropout=0.0, is_causal=False)
                x = x + attn.proj(o.reshape(n, -1).contiguous())
                x = x + block.mlp(block.norm2(x))
                if index in v.deepstack_visual_indexes:
                    levels.append(v.deepstack_merger_list[v.deepstack_visual_indexes.index(index)](x))
            return v.merger(x), torch.stack(levels)

    graph, params = hg.lift(VisionTower().eval())
    extents, specs = hg.WeightFile(build.bundle / "weights/vision.bin", "main:weights/vision.bin").write(params, torch.bfloat16)
    (build.bundle / "weights/vision.json").write_text(json.dumps({"inputs": list(extents), "blocks": [extents]}, indent=1))
    functions, checks = {}, {}
    for key in CANVASES:
        width, height = (int(v) for v in key.split("x"))
        assert width % (patch * merge) == 0 and height % (patch * merge) == 0, key
        grid_h, grid_w = height // patch, width // patch
        grid = torch.tensor([[1, grid_h, grid_w]])
        pos_index, pos_weight = get_vision_interpolation_indices_and_weights(
            grid, num_grid_per_side=side, mode=visual.interpolation_mode, align_corners=visual.interpolation_align_corners,
            spatial_merge_size=merge)
        pos_index, patch_rc = pos_index.to(torch.int32), get_vision_position_ids(grid, merge).to(torch.int32)
        torch.manual_seed(1043)
        pixels = (torch.randint(0, 256, (grid_h * grid_w, 3 * vc.temporal_patch_size * patch * patch)).float() / 127.5 - 1).to(torch.bfloat16)
        with torch.inference_mode():
            reference = visual(pixels, grid_thw=grid, return_dict=True)
            embeds, deepstack = graph(pixels, pos_index, pos_weight, patch_rc, *params.values())
        expected = (reference.pooler_output, torch.stack(reference.deepstack_features))
        check = {"canvas": key, "grid": [grid_h, grid_w], "relativeL2": [rel_l2(a, b) for a, b in ((embeds, expected[0]), *zip(deepstack, expected[1]))]}
        assert max(check["relativeL2"]) < 1e-3, check
        functions[key] = hg.convert(graph, (pixels, pos_index, pos_weight, patch_rc, *params.values()), None,
                                    ["pixels", "pos_index", "pos_weight", "patch_rc"] + list(extents), ["image_embeds", "deepstack"],
                                    build.bundle / f"models/conditioner-vision-{key}.aimodel")
        checks[key] = check
    build.record("vision", functions=functions, checks=checks, inputs=specs, patchSize=patch, mergeSize=merge,
                 temporalPatchSize=vc.temporal_patch_size, tableSide=side, deepstackLevels=len(vc.deepstack_visual_indexes),
                 presentation={"label": "<Picture {index}>: ", "visionStart": config.vision_start_token_id,
                               "imagePad": config.image_token_id, "visionEnd": config.vision_end_token_id})


# MARK: the manifest (MANIFEST.md §5.8 normative; driver `minimax-h3@1` and its `h3` block proposed until G4 step 2b)

def manifest(meta: dict, files: dict[str, dict]) -> dict:
    gpu = {"compute": "gpu", "expectFrequentReshapes": True}
    fixed = {"compute": "gpu", "expectFrequentReshapes": False}
    from_ = lambda ref: {"name": ref.split(".", 1)[1], "source": {"from": ref}}  # noqa: E731

    def weights(table: str, prefix: str, count: int, blocks: int) -> dict:
        return {"table": f"main:weights/{table}.json", "inputs": [f"w{i}" for i in range(count)], "blocks": blocks}

    body_inputs = len(meta["body"]["check"]["bindings"])
    text_inputs = len(meta["prepare"]["textInputs"])
    layer_inputs = len(meta["conditioner"]["inputs"])
    graphs = {
        "conditionerLayer": {"file": "main:models/conditioner-layer.aimodel", "function": "main",
                             "inputs": {"hidden": "hidden", "cos": "cos", "sin": "sin"}, "outputs": {"hidden": "hidden_out"},
                             "weights": weights("conditioner", "w", layer_inputs, hg.CONDITIONER_LAYERS), "residency": "perPrompt", **gpu},
        "prepareText": {"file": "main:models/prepare-text.aimodel", "function": "main", "inputs": {"text": "text"},
                        "outputs": {"refined": "refined"}, "weights": weights("prepare-text", "w", text_inputs, 1),
                        "residency": "perPrompt", **gpu},
        "preparePack": {"file": "main:models/prepare-pack.aimodel", "function": "main",
                        "inputs": {"video": "video", "audio": "audio", "text": from_("prepareText.refined"), "gather": "gather"},
                        "outputs": {"hidden": "hidden"}, "residency": "perStep", **gpu},
        "body": {"file": "main:models/body.aimodel", "function": "main",
                 "inputs": {"hidden": from_("preparePack.hidden"), "table": "table", "tags": "tags", "cos": "cos", "sin": "sin"},
                 "outputs": {"hidden": "hidden_out"}, "weights": weights("body", "w", body_inputs, hg.BLOCKS),
                 "residency": "perStep", **gpu},
        "heads": {"file": "main:models/heads.aimodel", "function": "main",
                  "inputs": {"hidden": from_("body.hidden"), "temb": "temb", "indices": "indices",
                             "videoIndices": "video_indices", "audioIndices": "audio_indices"},
                  "outputs": {"video": "video", "audio": "audio"}, "residency": "perStep", **gpu},
        "videoDecoder": {"file": "main:models/video-decoder-tile256-bf16.aimodel", "function": "main", "inputs": {"rows": "rows"},
                         "outputs": {"pixels": "pixels"}, "residency": "perImage", **fixed},
        "videoEncoderTile": {"file": "main:models/video-encoder-tile256.aimodel", "function": "main", "inputs": {"pixels": "pixels"},
                             "outputs": {"moments": "moments"}, "residency": "perPrompt", **fixed},
    }
    for key in CANVASES:
        graphs[f"videoEncoderTail{key}"] = {"file": f"main:models/video-encoder-tail-{key}.aimodel", "function": "main",
                                            "inputs": {"tiles": "tiles"}, "outputs": {"condition": "condition"},
                                            "residency": "perPrompt", **fixed}
        if "vision" in meta:
            graphs[f"conditionerVision{key}"] = {
                "file": f"main:models/conditioner-vision-{key}.aimodel", "function": "main",
                "inputs": {"pixels": "pixels", "posIndex": "pos_index", "posWeight": "pos_weight", "patchRC": "patch_rc"},
                "outputs": {"imageEmbeds": "image_embeds", "deepstack": "deepstack"},
                "weights": weights("vision", "w", len(meta["vision"]["inputs"]), 1), "residency": "perPrompt", **fixed}
    for na in sorted(meta["audio-decoder"]["fixed"], key=int):
        graphs[f"audioDecoder{na}"] = {"file": f"main:models/audio-decoder-na{na}.aimodel", "function": "main",
                                       "inputs": {"rows": "rows"}, "outputs": {"pcm": "pcm"}, "residency": "perImage", **fixed}
    c, v, a = meta["conditioner"], meta["video-decoder"], meta["audio-decoder"]
    h3 = {
        "tasks": ["t2va", "fl2va"],
        "profiles": [{"key": p.key, **p.dict(), "audioDecoder": f"audioDecoder{p.audio_latents}",
                      "keyframeEncoder": f"videoEncoderTail{p.width}x{p.height}"} for p in PROFILES],
        "limits": {"rows": list(hg.ROWS), "cardinality": list(hg.CARDINALITY), "textTokens": list(hg.TEXT_TOKENS)},
        "conditioner": {"layers": hg.CONDITIONER_LAYERS, "tap": hg.TAP, "embedding": c["embedding"],
                        "rope": {k: c["rope"][k] for k in ("theta", "mropeSection", "headDim", "dtype")},
                        "ropeTables": {"cos": "main:tables/conditioner/rope-cos.raw", "sin": "main:tables/conditioner/rope-sin.raw",
                                       "rows": hg.TEXT_TOKENS[1]}},
        "recipes": {k: {"video": r["video"], "audio": r["audio"], "steps": [{"timesteps": s["timesteps"], "temb": s["temb"]["file"],
                                                                            "blocks": s["blocks"]} for s in r["steps"]]}
                    for k, r in meta["tables"]["recipes"].items()},
        "defaultRecipe": "turbo8",
        "adapter": meta["body-weights"]["adapter"],
        "video": {"tile": v["tile"], "windowLatents": v["windowLatents"], "windowFrames": v["windowFrames"],
                  "windowStride": v["windowStride"], "precision": v["precision"], "packing": "patch2x2ChannelMajor",
                  "rowChannels": 96, "noise": {"kind": "splitmix64BoxMuller", "order": "packedToken", "dtype": "float32"}},
        "keyframe": {"tileSize": meta["video-encoder"]["tileSize"], "seed": meta["video-encoder"]["seed"],
                     "pixelMean": list(hg.PIXEL_MEAN), "pixelStd": list(hg.PIXEL_STD), "conditionTimestep": hg.FL2VA_CONDITION_TIMESTEP},
        "audio": {"sampleRate": a["sampleRate"], "hop": a["hop"], "channels": a["channels"], "rowChannels": 32},
        "platforms": ["macos"],
    }
    if "vision" in meta:
        vm = meta["vision"]
        h3["vision"] = {k: vm[k] for k in ("patchSize", "mergeSize", "temporalPatchSize", "tableSide", "deepstackLevels", "presentation")}
    return {
        "schema": 1,
        "id": "minimax-h3",
        "task": "text-to-video-audio",
        "driver": "minimax-h3@1",
        "variant": {"platform": "macos"},
        "assets": {"main": {"subtree": "", "files": [{"path": p, **m} for p, m in sorted(files.items())]}},
        "graphs": graphs,
        "tokenizer": {"source": {"dir": "main:tokenizer"}, "chatTemplate": "main:tokenizer/chat_template.jinja",
                      "renderer": "minimax-h3-prompt", "addSpecialTokens": False},
        "h3": h3,
        "precision": {"weights": "bf16 (transformer, conditioner, refiner, Turbo LoRA unmerged)", "activations": "bf16",
                      "components": {"heads": "float32", "videoDecoder": "bf16 weights and compute, float32 in/out",
                                     "audioDecoder": "float32", "videoEncoder": "float32"}},
        "license": "LicenseRef-MiniMax-H3-Community-License",
        "x-license": {"files": ["LICENSE", "NOTICE"], "modified": True, "territory": "excludes the EU, UK, Republic of Korea and USA",
                      "commercialRevenueCapUSD": 20_000_000, "uiAttribution": "MiniMax H3",
                      "encoder": "Qwen3-VL-32B (Apache-2.0)"},
        "provenance": {
            "source": hg.REPO, "sourceRevision": hg.REVISION,
            "adapter": f"{hg.ADAPTER_REPO}@{hg.ADAPTER_REVISION}:{hg.ADAPTER_FILE}",
            "producer": "aether-kit Forge/h3/export_h3.py",
            "refs": {"diffusersCommit": hg.DIFFUSERS_COMMIT, "forgeLockSHA256": meta.get("forgeLockSHA256", "0" * 64),
                     "exportReport": "export-report.json"},
        },
    }


def sample_meta() -> dict:
    """`build.json` with placeholder values: the manifest's shape without an export (the Swift decoding fixture)."""
    plans = {("turbo8", "t2va"): 9, ("turbo8", "fl2va"): 9, ("turbo4", "t2va"): 5, ("turbo4", "fl2va"): 5}
    recipes = {}
    for (name, task), points in plans.items():
        arm = {"shift": 0.0, "timesteps": [0.0] * points, "sigmas": [0.0] * points}
        recipes[f"{name}/{task}"] = {"video": arm, "audio": arm, "steps": [
            {"timesteps": [0.5], "temb": {"file": f"main:tables/{name}/{task}/step-{i}/temb.raw"},
             "blocks": f"main:tables/{name}/{task}/step-{i}/block-{{block:02d}}.raw"} for i in range(points - 1)]}
    return {
        "body": {"check": {"bindings": [f"b{i}" for i in range(22)]}},
        "body-weights": {"adapter": {"repo": hg.ADAPTER_REPO, "revision": hg.ADAPTER_REVISION, "sha256": hg.ADAPTER_SHA256,
                                     "execution": "dynamicLora", "rank": hg.LORA_RANK, "alpha": hg.LORA_ALPHA}},
        "prepare": {"textInputs": [{"name": f"w{i}"} for i in range(20)]},
        "conditioner": {"inputs": [{"name": f"w{i}"} for i in range(11)],
                        "embedding": {"file": "main:weights/conditioner/embedding.bin", "rows": 151936, "width": 5120, "dtype": "bfloat16"},
                        "rope": {"theta": 5000000, "mropeSection": [24, 20, 20], "headDim": 128, "dtype": "bfloat16"}},
        "tables": {"recipes": recipes},
        "video-decoder": {"tile": hg.VIDEO_TILE, "windowLatents": hg.WINDOW_LATENTS, "windowFrames": hg.WINDOW_FRAMES,
                          "windowStride": hg.WINDOW_STRIDE, "precision": "bfloat16"},
        "video-encoder": {"tileSize": hg.VIDEO_TILE, "seed": hg.ENCODE_SEED},
        "audio-decoder": {"fixed": {str(p.audio_latents): {} for p in PROFILES}, "sampleRate": hg.SAMPLE_RATE,
                          "hop": hg.AUDIO_HOP, "channels": hg.AUDIO_CHANNELS},
    }


def sample_manifest() -> dict:
    return manifest(sample_meta(), {".aether/placeholder": {"bytes": 1, "sha256": "0" * 64}})


def notice() -> str:
    return "\n".join([
        "MiniMax H3 (Aether-Kit Core AI export)",
        "",
        NOTICE_LINE,
        "",
        f"This bundle is a modified version of {hg.REPO} at revision {hg.REVISION} (the FL2VA checkpoint) with the",
        f"Turbo 8-step LoRA {hg.ADAPTER_REPO}@{hg.ADAPTER_REVISION} ({hg.ADAPTER_FILE}, sha256 {hg.ADAPTER_SHA256}).",
        "Aether-Kit's Forge exporter re-expressed its modules as Core AI graphs (the transformer block, the Qwen3-VL",
        "text decoder layer, the text refiner and the vision tower with their weights lifted to inputs and stored per",
        "block; the heads, pack, video and audio decoders and the first-frame encoder with their weights baked in) and",
        "precomputed its AdaLN and rotary tables. The weights are otherwise unchanged; the LoRA stays unmerged.",
        "",
        "Use, reproduction and distribution are subject to the MiniMax H3 Community License Agreement (LICENSE),",
        "including its territorial restriction (it excludes the European Union, the United Kingdom, the Republic of",
        "Korea and the United States of America) and its Acceptable Use Policy. Commercial products must display",
        "\"MiniMax H3\"; above USD 20,000,000 yearly revenue a separate authorization from MiniMax is required.",
        "The text encoder is Qwen3-VL-32B, licensed under the Apache License 2.0.",
        "",
    ])


def stage_package(args, build: Build) -> None:
    bundle, meta = build.bundle, build.meta
    for name in ("body", "body-weights", "tables", "heads", "prepare", "conditioner", "audio-decoder", "video-decoder", "video-encoder"):
        assert name in meta and not meta[name].get("random"), f"stage {name} is missing (or random); the bundle is incomplete"
    tokenizer = bundle / "tokenizer"
    tokenizer.mkdir(exist_ok=True)
    for name in TOKENIZER_FILES:
        source = args.snapshot / "FL2VA/tokenizer" / name
        if source.exists():
            shutil.copyfile(source, tokenizer / name)
    shutil.copyfile(args.snapshot / "LICENSE", bundle / "LICENSE")
    (bundle / "NOTICE").write_text(notice())
    meta["forgeLockSHA256"] = hg.sha256_file(HERE.parent / "uv.lock")
    (bundle / ".aether").mkdir(exist_ok=True)
    skip = {".aether/aether.json", "SHA256SUMS", "export-report.json"}
    files = {}
    for p in sorted(q for q in bundle.rglob("*") if q.is_file()):
        rel = p.relative_to(bundle).as_posix()
        if rel not in skip:
            files[rel] = {"bytes": p.stat().st_size, "sha256": hg.sha256_file(p)}
    document = manifest(meta, files)
    (bundle / ".aether/aether.json").write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    report = {"schema": 1, "source": hg.REPO, "revision": hg.REVISION, "diffusersCommit": hg.DIFFUSERS_COMMIT,
              "forgeLockSHA256": meta["forgeLockSHA256"], "scriptSHA256": {p.name: hg.sha256_file(p) for p in sorted(HERE.glob("*.py"))},
              "stages": {k: v for k, v in meta.items() if isinstance(v, dict)}, "bundleBytes": sum(m["bytes"] for m in files.values())}
    (bundle / "export-report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    sums = [f"{m['sha256']}  {p}" for p, m in sorted(files.items())]
    sums += [f"{hg.sha256_file(bundle / p)}  {p}" for p in (".aether/aether.json", "export-report.json")]
    (bundle / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    build.record("package", files=len(files), bytes=report["bundleBytes"])


STAGES = {"convert-vae": stage_convert_vae, "body": stage_body, "body-weights": stage_body_weights, "tables": stage_tables,
          "heads": stage_heads, "prepare": stage_prepare, "conditioner": stage_conditioner, "audio-decoder": stage_audio_decoder,
          "video-decoder": stage_video_decoder, "video-encoder": stage_video_encoder, "vision": stage_vision, "package": stage_package}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=list(STAGES))
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--converter", type=Path, help="Diffusers' scripts/convert_minimax_h3_to_diffusers.py (default: uv's checkout)")
    parser.add_argument("--random", action="store_true", help="body, body-weights: seeded random weights (the chain gate only)")
    parser.add_argument("--random-blocks", type=int, default=4)
    parser.add_argument("--minimum-free-gb", type=float, default=150.0)
    parser.add_argument("--sample-manifest", action="store_true", help="print the manifest's shape and exit (the Swift fixture)")
    args = parser.parse_args()
    if args.sample_manifest:
        print(json.dumps(sample_manifest(), indent=2, ensure_ascii=False))
        return
    assert args.output and args.stage, "--output and --stage are required"
    if args.random and args.stage not in ("body", "body-weights"):
        raise SystemExit("--random applies to body and body-weights only")
    if not args.random:
        assert args.snapshot.resolve().name == hg.REVISION, "unexpected checkpoint snapshot"
    torch.set_num_threads(8)
    build = Build(args.output.resolve())
    started = time.monotonic()
    STAGES[args.stage](args, build)
    print(json.dumps({"stage": args.stage, "seconds": round(time.monotonic() - started, 1), "peakRSSGB": peak_rss_gb()}), flush=True)


if __name__ == "__main__":
    main()

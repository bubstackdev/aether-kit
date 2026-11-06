"""The H3 bundle's per-graph oracle gate: every exported graph on the GPU against the original modules on the CPU, at
the qualified canvas (960×544, 124 frames, T2VA; the first-frame encoder for FL2VA), each graph fed the CPU oracle's
own inputs so every comparison isolates one graph. The limit is the 0.0.x prototype's: relative L2 ≤ 0.02 per tensor.

    Forge/.venv/bin/python -I Forge/h3/check_h3.py --output <export dir> --report <new report.json> [--wait]

  tap       conditioner: the bundle's layer graph ×50 from its weight files, the embedding row source and rotary tables,
            against the original Qwen3-VL decoder layers streamed from the checkpoint (causal mask), hidden_states[50]
  refined   prepare-text (weights from its file) on the CPU tap, against context_embedder + token refiner + Turbo LoRA
  packed    prepare-pack on the CPU refined text and seeded latents in the pipeline's own layout ([text | keyframes |
            audio | video], channel-major audio: `build_packed_sequence`) and row timesteps (`build_row_timesteps`)
  block00   the body graph, block 0's weights from the bundle, on the CPU packed sequence with the step-0 AdaLN table
            file, against the original block (PEFT LoRA, live AdaLN from the live time embedding)
  block49   the same for block 49, on the GPU chain's block-48 output (the GPU runs all 50 blocks from their files)
  heads     on the GPU block-49 output (both heads, video and audio rows)
  tile      one window-tile decode; pcm the audio decoder; condition the tiled first-frame encoder (FL2VA)

Reads real MiniMax H3 weights: mind the license (LICENSE; its territory excludes the EU, UK, Republic of Korea and the
USA). Runs one heavy step at a time and refuses to start while Swift tests hold the GPU (or waits with --wait).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h3_graphs as hg  # noqa: E402
import export_h3 as ex  # noqa: E402

LIMIT = 0.02
PROMPT = "A red fox trots through a snowy birch forest at dawn, its breath steaming; soft wind and distant birdsong."


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def to_np(t: torch.Tensor) -> np.ndarray:
    import ml_dtypes
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
    return t.numpy()


class Graph:
    """A bundle graph on the GPU (Python's Core AI runtime). bf16 outputs come back through DLPack as torch tensors."""

    def __init__(self, path: Path, function: str = "main"):
        import coreai.runtime as rt
        self.rt = rt
        options = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
        self.model = asyncio.run(rt.AIModel.load(path, options))
        self.function = self.model.load_function(function)

    def __call__(self, **feeds) -> dict[str, torch.Tensor]:
        rt = self.rt

        async def go():
            out = await self.function({k: rt.NDArray(np.ascontiguousarray(to_np(v) if isinstance(v, torch.Tensor) else v))
                                       for k, v in feeds.items()}, {})
            return {k: torch.from_dlpack(v._tensor.to_dlpack()).clone() for k, v in out.items()}
        return asyncio.run(go())


class Weights:
    """A bundle weight table: block b's inputs as tensors (the bytes the runtime binds)."""

    def __init__(self, bundle: Path, table: str, specs: list[dict]):
        self.bundle, self.doc, self.specs = bundle, json.loads((bundle / f"weights/{table}.json").read_text()), specs

    def block(self, b: int) -> dict[str, torch.Tensor]:
        return {s["name"]: hg.read_extent(self.bundle, self.doc["blocks"][b][s["name"]], getattr(torch, s["dtype"]), s["shape"])
                for s in self.specs}


def wait_for_gpu(wait: bool) -> None:
    while True:
        busy = subprocess.run(["pgrep", "-fl", "xctest|swiftpm-testing-helper"], capture_output=True, text=True).stdout.strip()
        if not busy:
            return
        if not wait:
            raise SystemExit(f"Swift tests hold the GPU; not loading graphs:\n{busy}")
        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="the export directory (bundle/ and work/)")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, default=ex.DEFAULT_SNAPSHOT)
    parser.add_argument("--converter", type=Path)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    assert not args.report.exists(), "report exists; choose a new path"
    torch.set_num_threads(8)
    bundle, meta = args.output / "bundle", json.loads((args.output / "work/build.json").read_text())
    profile = ex.PROFILES[0]
    results, started = {}, time.monotonic()

    def record(name, actual, expected, **extra):
        value = rel_l2(actual.float(), expected.float())
        results[name] = {"relativeL2": value, "pass": bool(value <= LIMIT and torch.isfinite(actual.float()).all()),
                         "shape": list(actual.shape), **extra}
        print(json.dumps({name: results[name]}), flush=True)

    wait_for_gpu(args.wait)
    # tap: the conditioner.
    from transformers import AutoTokenizer, Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer, Qwen3VLTextRotaryEmbedding
    ids = AutoTokenizer.from_pretrained(bundle / "tokenizer")(PROMPT, add_special_tokens=False)["input_ids"]
    T = len(ids)
    directory = args.snapshot / "FL2VA/text_encoder"
    config = Qwen3VLConfig.from_pretrained(directory).text_config
    config._attn_implementation = "sdpa"
    c = meta["conditioner"]
    rows = torch.frombuffer(bytearray((bundle / "weights/conditioner/embedding.bin").read_bytes()), dtype=torch.bfloat16).view(c["embedding"]["rows"], -1)
    cos_t = torch.frombuffer(bytearray((bundle / "tables/conditioner/rope-cos.raw").read_bytes()), dtype=torch.bfloat16).view(-1, config.head_dim)[:T][None]
    sin_t = torch.frombuffer(bytearray((bundle / "tables/conditioner/rope-sin.raw").read_bytes()), dtype=torch.bfloat16).view(-1, config.head_dim)[:T][None]
    rotary = Qwen3VLTextRotaryEmbedding(config)
    original = hg.sharded(directory, "model.language_model.embed_tokens.")["weight"]
    cpu = original[torch.tensor(ids)][None]
    gpu = rows[torch.tensor(ids)][None]
    assert torch.equal(cpu, gpu), "the embedding row source differs from the checkpoint"
    cos, sin = rotary(cpu, torch.arange(T).view(1, 1, -1).expand(3, 1, -1))
    assert torch.equal(cos, cos_t) and torch.equal(sin, sin_t), "the rotary tables differ from the original rotary"
    causal = torch.ones(T, T, dtype=torch.bool).tril().view(1, 1, T, T)
    layer_graph, layer_weights = Graph(bundle / "models/conditioner-layer.aimodel"), Weights(bundle, "conditioner", c["inputs"])
    for index in range(hg.CONDITIONER_LAYERS):
        with torch.device("meta"):
            layer = Qwen3VLTextDecoderLayer(config, index)
        layer.load_state_dict(hg.sharded(directory, f"model.language_model.layers.{index}."), strict=True, assign=True)
        with torch.inference_mode():
            cpu = layer.eval()(cpu, position_embeddings=(cos, sin), attention_mask=causal)
        gpu = layer_graph(hidden=gpu, cos=cos_t, sin=sin_t, **layer_weights.block(index))["hidden_out"]
        del layer
    record("tap", gpu, cpu, tokens=T)
    tap = cpu

    # refined, packed: the prepare graphs.
    converter, _ = hg.load_converter(args.converter)
    checkpoint = hg.Checkpoint(args.snapshot, converter)
    adapter = hg.adapter_path()
    model = hg.meta_model()
    model.load_state_dict(checkpoint.converted(lambda n: n.startswith(("token_refiner.", "condition_proj.", "video_patch_proj.",
                                                                       "audio_patch_proj.", "time_embedder.", "final_layer."))),
                          strict=False, assign=True)
    hg.load_turbo(model, adapter, lambda k: k.startswith("token_refiner."))
    model.time_proj = type(model.time_proj)(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
    with torch.inference_mode():
        refined = model.token_refiner(model.context_embedder(tap.to(model.context_embedder.weight.dtype)))
    text_graph = Graph(bundle / "models/prepare-text.aimodel")
    record("refined", text_graph(text=tap, **Weights(bundle, "prepare-text", meta["prepare"]["textInputs"]).block(0))["refined"], refined)
    # The pipeline's own layout ([text | keyframes | audio | video], channel-major audio, the fp64 rotary grid) and row
    # timesteps, so blocks see real step-0 contexts: the prototype found synthetic positions expose bf16 arithmetic
    # noise at block 49 rather than a graph defect.
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (MINIMAX_H3_AUDIO_CHANNELS, MINIMAX_H3_AUDIO_TAG,
                                                                         MINIMAX_H3_TEXT_TAG, MINIMAX_H3_VIDEO_TAG)
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep, MiniMaxH3SetTimestepsStep
    na = profile.audio_latents
    position_ids, token_tags, video_indices, audio_indices, text_indices, cond_video, cond_audio = \
        MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            torch.full((T,), MINIMAX_H3_TEXT_TAG, dtype=torch.long), profile.latent_frames, profile.latent_height,
            profile.latent_width, na, (1, 2, 2), MINIMAX_H3_AUDIO_CHANNELS, MINIMAX_H3_AUDIO_TAG, MINIMAX_H3_VIDEO_TAG)
    nv, nar = video_indices.numel(), audio_indices.numel()
    R = T + nv + nar
    assert nv == profile.video_rows and cond_video == 0, (nv, cond_video)
    g = torch.Generator().manual_seed(20261106)
    video, audio = torch.randn(1, nv, 96, generator=g), torch.randn(1, nar, 32, generator=g)
    gather = torch.empty(R, dtype=torch.int32)
    gather[text_indices] = torch.arange(T, dtype=torch.int32)
    gather[video_indices] = T + torch.arange(nv, dtype=torch.int32)
    gather[audio_indices] = T + nv + torch.arange(nar, dtype=torch.int32)
    with torch.inference_mode():
        packed = refined.new_zeros((1, R, refined.shape[-1]))
        packed = packed.index_copy(1, text_indices, refined)
        packed = packed.index_copy(1, video_indices, model.proj_in(video).to(refined.dtype))
        packed = packed.index_copy(1, audio_indices, model.audio_proj_in(audio).to(refined.dtype))
    pack_graph = Graph(bundle / "models/prepare-pack.aimodel")
    record("packed", pack_graph(video=video, audio=audio, text=refined, gather=gather)["hidden"], packed, rows=R)

    # block00, block49: the body; step 0 of turbo8/t2va.
    recipe = meta["tables"]["recipes"]["turbo8/t2va"]
    step = recipe["steps"][0]
    video_t, audio_t = recipe["video"]["timesteps"][0], recipe["audio"]["timesteps"][0]
    unique, timestep_indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
        video_indices, audio_indices, cond_video, cond_audio, T, float(video_t), float(audio_t), 0.0, 0.0)
    assert [float(v) for v in unique] == [float(v) for v in step["timesteps"]], (unique, step["timesteps"])
    timesteps = unique.to(torch.float32)
    with torch.inference_mode():
        temb = model.time_embedder(model.time_proj(timesteps).to(torch.float32))
    temb_file = torch.frombuffer(bytearray((bundle / step["temb"]["file"].split(":", 1)[1]).read_bytes()), dtype=torch.float32).view(temb.shape)
    record("temb", temb_file, temb)
    adaln = (timestep_indices * 3 + token_tags).to(torch.int32)
    cos_r, sin_r = MiniMaxH3RotaryPosEmbed()(position_ids.to(torch.float32))   # the model's rotary (built on meta with it)
    body, body_weights = Graph(bundle / "models/body.aimodel"), Weights(bundle, "body", meta["body-weights"]["inputs"])

    def table(b):
        path = bundle / step["blocks"].split(":", 1)[1].format(block=b)
        return torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.bfloat16).view(6, 3 * len(step["timesteps"]), hg.HIDDEN)

    def cpu_block(b, hidden):
        block_model = hg.meta_model()
        block_model.load_state_dict(checkpoint.converted(lambda n: n.startswith(f"blocks.{b}.")), strict=False, assign=True)
        hg.load_turbo(block_model, adapter, lambda k: k.startswith(f"transformer_blocks.{b}."))
        with torch.inference_mode():
            return block_model.transformer_blocks[b](hidden, temb, adaln.long(), (cos_r, sin_r))

    x = packed
    for b in range(hg.BLOCKS):
        y = body(hidden=x, table=table(b), tags=adaln, cos=cos_r.float(), sin=sin_r.float(), **body_weights.block(b))["hidden_out"]
        if b == 0:
            record("block00", y, cpu_block(0, packed))
        if b == hg.BLOCKS - 1:
            record("block49", y, cpu_block(b, x))
        x = y
        print(json.dumps({"gpuBlock": b}), flush=True)

    # heads.
    heads = hg.Heads(model).eval()
    vi, ai = video_indices.to(torch.int32), audio_indices.to(torch.int32)
    with torch.inference_mode():
        ev, ea = heads(x, temb, timestep_indices.to(torch.int32), vi, ai)
    out = Graph(bundle / "models/heads.aimodel")(hidden=x, temb=temb, indices=timestep_indices.to(torch.int32), video_indices=vi, audio_indices=ai)
    record("heads_video", out["video"], ev)
    record("heads_audio", out["audio"], ea)

    # tile, pcm, condition: the decoders and the first-frame encoder.
    from diffusers import AutoencoderKLMiniMaxH3
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
    work = args.output / "work"
    vae16 = AutoencoderKLMiniMaxH3.from_pretrained(work / "converted/vae", torch_dtype=torch.bfloat16).eval()
    vae16.use_tiling = False
    decoder = hg.WindowTileDecoder(vae16).eval()
    rows = torch.randn(448, 96, generator=g)
    with torch.inference_mode():
        record("tile", Graph(bundle / "models/video-decoder-tile256-bf16.aimodel")(rows=rows)["pixels"], decoder(rows))
    audio_vae = ex.converted_vae(type("B", (), {"work": work})(), audio=True)
    audio_rows = torch.randn(MINIMAX_H3_AUDIO_CHANNELS, na, 32, generator=g)
    with torch.inference_mode():
        record("pcm", Graph(bundle / f"models/audio-decoder-na{na}.aimodel")(rows=audio_rows)["pcm"], hg.AudioDecoder(audio_vae).eval()(audio_rows))
    vae = ex.converted_vae(type("B", (), {"work": work})())
    key = f"{profile.width}x{profile.height}"
    image = torch.randint(0, 256, (1, 3, 1, profile.height, profile.width), dtype=torch.uint8, generator=g)
    starts = meta["video-encoder"]["checks"][key]
    tile_graph = Graph(bundle / "models/video-encoder-tile256.aimodel")
    side = hg.VIDEO_TILE
    moments = torch.cat([tile_graph(pixels=image[..., y:y + side, x0:x0 + side].float())["moments"]
                         for y in starts["yStarts"] for x0 in starts["xStarts"]])
    with torch.inference_mode():
        reference = encode_vae_condition(vae, image, hg.PIXEL_MEAN, hg.PIXEL_STD, hg.ENCODE_SEED)
    record("condition", Graph(bundle / f"models/video-encoder-tail-{key}.aimodel")(tiles=moments)["condition"], reference)

    report = {"schema": 1, "profile": profile.dict(), "prompt": PROMPT, "limit": LIMIT, "results": results,
              "pass": all(r["pass"] for r in results.values()), "seconds": round(time.monotonic() - started, 1)}
    args.report.write_text(json.dumps(report, indent=1))
    print(json.dumps({"pass": report["pass"], "seconds": report["seconds"]}), flush=True)
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()

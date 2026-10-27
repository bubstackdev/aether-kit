"""Export a MiniMax H3 profile for AetherKit from the original FL2VA checkpoint (CPU only, lab-free).

Stages (each records its outputs, with SHA-256, in OUTPUT/build.json):
  convert-vae    Convert the video VAE to Diffusers layout under OUTPUT/converted/vae (pinned converter).
  body           Export ONE transformer-block graph for all 50 blocks: weights (w0..w21, Turbo LoRA slots included) are
                 inputs, and both the row count and the timestep cardinality (k = 1...3, table [6,3k,D]) are dynamic.
                 CPU check on 128 rows of real block-0 weights (zero LoRA): byte-identical to the PEFT block.
  tables         Precompute, per schedule and task, every step's timestep embeddings [k,2688] f32 and the 50 blocks'
                 AdaLN tables [6,3k,5376] bf16 with the original modules (the AdaLN branches carry no Turbo LoRA).
  heads          Export the output norm and both heads: hidden [1,R,D] bf16, temb [k,2688] f32, timestep indices [R],
                 video/audio row indices -> video [1,Nv,96] f32, audio [1,Na,32] f32 (all row counts dynamic).
  conditioner    Export the Qwen3-VL-32B text path H3 reads: the token embedding (ids [1,T] i32 -> [1,T,5120] bf16) and
                 decoder layers 0...49 (hidden, cos, sin -> hidden; causal attention inside), T dynamic (1...4096). The
                 tap is hidden_states[50], unnormalised. Each layer is checked on CPU against the same layer with an
                 explicit causal mask. Stops before any export that would leave less than --minimum-free-gb free.
  parameters     Record the 22 bindings (w0..w21) of each of the 50 blocks as byte ranges read in place from the
                 pinned FL2VA shards and the Turbo adapter (no copy). Every binding's bytes are checked against the
                 converter's tensor (base) or the PEFT-loaded adapter tensor (LoRA), in the body graph's input order.
  prepare        Export the per-render text graph (context_embedder + the 2-layer token refiner with its Turbo LoRA; every
                 weight lifted to an input and bound in place) and the per-step pack graph (video/audio patch projections,
                 gathered with the refined text into the packed sequence). Both are byte-checked on CPU.
  conditioner-layer  Export ONE Qwen3-VL decoder-layer graph with its 11 weights lifted, bound in place for layers
                 0...49 (replacing 50 baked layer assets), and the token-embedding table as an in-place row source.
                 Layers 0 and 49 run with their in-place weights against the original layer on CPU.
  conditioner-rope  Write the Qwen3-VL text rotary tables cos/sin [4096, 128] bf16 for text positions 0...4095 with
                 the original rotary module (text-only presentations: the three M-RoPE axes coincide). A prompt of T
                 tokens uses the first T rows, so Swift never recomputes an angle.
  audio-decoder  Convert the audio VAE (identity mapping, pinned converter) and export its decoder: channel-major
                 normalized rows [2, Na, 32] f32 -> stereo planar PCM [2, 800 Na] f32 at 32 kHz, Na dynamic, checked
                 on CPU against the pipeline's denormalize + `decode`.
  audio-decoder-fixed  Export the same audio decoder with a STATIC latent count for each --audio-latents value (the
                 dynamic asset aborts in MPSGraph on the GPU: "Failed to resolve dynamic dimensions for memref.alloc",
                 E1 2026-10-19). Each is checked on CPU against the pipeline's denormalize + `decode` at that count.
  video-decoder-bf16  The bundle's default video decoder once exported (float32 stays as `video_decoder.float32`): the same window-tile decoder with the VAE loaded at
                 torch_dtype=bfloat16, the reduced precision Diffusers itself supports (`_keep_in_fp32_modules` keeps the
                 norms and scales in float32; `decode` casts z to the decoder's dtype). Checked on CPU against `_decode`
                 at that dtype, and its distance from the float32 decoder is recorded.
  video-encoder-first  Export the first-frame (fl2va keyframe) encoder, one STATIC graph per canvas orientation
                 (960x544, 544x960): pixels [1,3,1,H,W] f32 in 0...255 -> normalized condition latents [1,24,1,H/16,W/16]
                 f32. It is the pipeline's `encode_vae_condition` exactly: ImageNet normalize, the (tiled) encoder, the
                 posterior SAMPLED under the fixed keyframe seed 42 (its draw baked in as a constant), float16 rounding,
                 latent normalization. Checked on CPU against `encode_vae_condition` (byte-identical). Needs no
                 download: the FL2VA video_vae weights are in the local snapshot (converted under converted/vae).
  finalize       Write OUTPUT/bundle.json (schema 3) from build.json, re-hashing every referenced asset stream, table
                 file and in-place weight range. `complete` is true only when every required part exists.
  video-decoder  Export the 256 px spatial-tile window decoder: packed rows [448,96] f32 -> pixels [1,3,22,256,256] f32
                 (latent denormalisation, decoder, source pre-padding removal), then check it on CPU against the
                 original `_decode` of a single-tile, single-window clip.
No Core AI runtime, specialisation or GPU is used. Inputs are read-only; outputs stay under OUTPUT.
Usage: python -I -B export-h3-profile.py --snapshot SNAP --converter DIR --output OUT --stage STAGE
"""
import argparse, json, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import h3_export_lib as h


def record(output: Path, **fields):
    path = output / "build.json"
    meta = json.loads(path.read_text()) if path.exists() else dict(schema=1, snapshot=h.SNAPSHOT, diffusers_commit=h.COMMIT)
    meta.update(fields)
    path.write_text(json.dumps(meta, indent=1, sort_keys=True))
    return meta


class WindowTileDecoder(torch.nn.Module):
    """Packed tile rows of one temporal window -> the 22 frames the source keeps from that window."""

    def __init__(self, vae, tile=256):
        super().__init__()
        self.vae, self.side = vae, tile // 16
        self.register_buffer("mean", torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(1, 24, 1, 1, 1))
        self.register_buffer("std", torch.tensor(vae.config.latents_std, dtype=torch.float32).view(1, 24, 1, 1, 1))

    def forward(self, rows):
        s = self.side
        # rows[(f*s/2 + y)*s/2 + x, c*4 + dy*2 + dx] -> z[0, c, f, 2y+dy, 2x+dx]
        z = rows.view(7, s // 2, s // 2, 24, 2, 2).permute(3, 0, 1, 4, 2, 5).reshape(1, 24, 7, s, s)
        z = z * self.std + self.mean
        clip = self.vae.decoder(self.vae.post_quant_conv(z))
        return torch.cat((clip[:, :, 3:20], clip[:, :, 23:28]), dim=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--converter", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--stage", required=True, choices=["convert-vae", "video-decoder", "body", "tables", "heads", "conditioner", "parameters", "prepare", "conditioner-layer", "conditioner-rope", "audio-decoder", "audio-decoder-fixed", "video-decoder-bf16", "video-encoder-first", "video-encoder-first-tiled", "finalize"])
    ap.add_argument("--canvases", default="960x544,544x960", help="video-encoder-first(-tiled): WIDTHxHEIGHT canvases")
    ap.add_argument("--converted", type=Path, help="video-encoder-first-tiled: the converted VAE directory (default OUTPUT/converted)")
    ap.add_argument("--base-bundle", type=Path, help="video-encoder-first-tiled: write OUTPUT/bundle.json = this bundle plus the tiled entries")
    ap.add_argument("--audio-latents", default="207", help="comma-separated per-channel audio latent counts (audio-decoder-fixed)")
    ap.add_argument("--minimum-free-gb", type=float, default=150.0)
    args = ap.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    assert args.snapshot.resolve().name == h.SNAPSHOT, "unexpected checkpoint snapshot"
    torch.set_num_threads(8)
    converter = h.load_converter(args.converter)
    if args.stage == "convert-vae":
        import diffusers
        target = out / "converted/vae"
        assert not target.exists(), "remove the superseded conversion first"
        converter.convert_video_vae(str(args.snapshot / "FL2VA"), str(target), converter.MINIMAX_H3_VIDEO_VAE_CONFIG,
                                    diffusers.__version__, 5 << 30)
        files = {p.name: h.sha256_file(p) for p in sorted(target.iterdir()) if p.is_file()}
        record(out, converted_vae=dict(path=str(target), files=files))
    elif args.stage == "body":
        from torch.export import Dim
        checkpoint = h.Checkpoint(args.snapshot, converter)
        model = h.meta_model()
        model.load_state_dict(checkpoint.converted(lambda n: n.startswith("blocks.0.")), strict=False, assign=True)
        block = model.transformer_blocks[0]
        adapter, alphas = {}, {}
        for t in ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2"]:
            linear = block.get_submodule(t)
            adapter[f"transformer.transformer_blocks.0.{t}.lora_A.weight"] = torch.zeros(128, linear.in_features, dtype=torch.bfloat16)
            adapter[f"transformer.transformer_blocks.0.{t}.lora_B.weight"] = torch.zeros(linear.out_features, 128, dtype=torch.bfloat16)
            alphas[f"transformer.transformer_blocks.0.{t}.alpha"] = 8.0
        model.load_lora_adapter(adapter, prefix="transformer", adapter_name="turbo", network_alphas=alphas)
        torch.manual_seed(1043)
        hidden, temb = torch.randn(1, 128, 5376, dtype=torch.bfloat16), torch.randn(2, 2688)
        tags, cos, sin = torch.randint(0, 6, (128,)), torch.randn(128, 96), torch.randn(128, 96)
        with torch.inference_mode():
            table = torch.stack(block.adaln_proj(temb))
            expected = block(hidden, temb, tags, (cos, sin))
        block.adaln_proj = h.CachedModulation()
        graph, params = h.lift(h.BlockCall(block).eval())
        with torch.inference_mode():
            actual = graph(hidden, table, tags.to(torch.int32), cos, sin, *params.values())
        names = list(params)
        check = dict(rows=128, cardinality=2, byte_identical=bool(torch.equal(actual, expected)), bindings=names)
        print(json.dumps(dict(check, bindings=len(names))), flush=True)
        assert check["byte_identical"] and len(names) == 22
        rows, k = Dim("rows", min=64, max=40960), Dim("k", min=1, max=3)
        example = (torch.zeros(1, 19333, 5376, dtype=torch.bfloat16), torch.zeros(6, 6, 5376, dtype=torch.bfloat16),
                   torch.zeros(19333, dtype=torch.int32), torch.zeros(19333, 96), torch.zeros(19333, 96))
        dynamic = [{1: rows}, {1: 3 * k}, {0: rows}, {0: rows}, {0: rows}] + [None] * len(names)
        started = time.monotonic()
        asset = h.export_asset(out / "models/body.aimodel", graph, (*example, *params.values()),
                               ["hidden", "temb", "tags", "cos", "sin"] + [f"w{i}" for i in range(22)], ["hidden_out"],
                               dynamic_shapes=(tuple(dynamic),))
        asset["seconds"] = round(time.monotonic() - started, 1)
        record(out, body=asset, body_cpu_check=check, body_dynamic=dict(rows=[64, 40960], cardinality=[1, 3]))
        print(json.dumps(asset), flush=True)
    elif args.stage == "tables":
        from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
        checkpoint = h.Checkpoint(args.snapshot, converter)
        model = h.meta_model()
        model.load_state_dict(checkpoint.converted(lambda n: n.startswith("time_embedder.")), strict=False, assign=True)
        model.time_proj = type(model.time_proj)(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        recipes = {"turbo8": (9, 12.0, 3.0), "turbo4": (5, 12.0, 3.0)}   # grid points, video/audio shift (544p adapter)
        plans = {}
        for name, (points, video_shift, audio_shift) in recipes.items():
            arms = []
            for shift in (video_shift, audio_shift):
                scheduler = MiniMaxH3Scheduler(shift=shift); scheduler.set_timesteps(points)
                arms.append(dict(shift=shift, timesteps=scheduler.timesteps.tolist(), sigmas=scheduler.sigmas.tolist()))
            for task, condition in (("t2va", None), ("fl2va", 0.999)):
                steps = []
                for tv, ta in zip(arms[0]["timesteps"], arms[1]["timesteps"]):
                    values = {tv, ta} | ({max(tv, condition)} if condition is not None else set())
                    steps.append(sorted(values))
                plans[(name, task)] = (arms, steps)
        tables = out / "tables"
        index = {}
        with torch.inference_mode():
            embeddings = {}
            for (name, task), (arms, steps) in plans.items():
                for i, values in enumerate(steps):
                    t = torch.tensor(values, dtype=torch.float32)
                    embeddings[(name, task, i)] = model.time_embedder(model.time_proj(t).to(torch.float32))
            for block in range(50):
                weights = checkpoint.converted(lambda n, b=block: n.startswith(f"blocks.{b}.adaln_proj."))
                prefix = f"transformer_blocks.{block}.adaln_proj."
                module = model.transformer_blocks[block].adaln_proj
                module.load_state_dict({k.removeprefix(prefix): v for k, v in weights.items()}, strict=True, assign=True)
                for (name, task, i), temb in embeddings.items():
                    entry = index.setdefault(f"{name}/{task}", dict(schedule=plans[(name, task)][0], steps=[]))
                    while len(entry["steps"]) <= i:
                        entry["steps"].append(dict(timesteps=plans[(name, task)][1][len(entry["steps"])], blocks=[]))
                    if block == 0:
                        entry["steps"][i]["temb"] = h.raw_fixture(tables / name / task / f"step-{i}" / "temb.raw", "temb", temb)
                    table = torch.stack(module(temb))
                    entry["steps"][i]["blocks"].append(h.raw_fixture(tables / name / task / f"step-{i}" / f"block-{block:02d}.raw", "temb", table))
                model.transformer_blocks[block].adaln_proj.to("meta")
                print("tables block", block, flush=True)
        record(out, tables=index)
    elif args.stage == "heads":
        from torch.export import Dim
        checkpoint = h.Checkpoint(args.snapshot, converter)
        model = h.meta_model()
        model.load_state_dict(checkpoint.converted(lambda n: n.startswith("final_layer.")), strict=False, assign=True)

        class Heads(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm_out, self.proj_out, self.audio_proj_out = model.norm_out, model.proj_out, model.audio_proj_out

            def forward(self, hidden, temb, indices, video_indices, audio_indices):
                hidden = self.norm_out(hidden, temb, indices).to(self.proj_out.weight.dtype)
                return self.proj_out(hidden).index_select(1, video_indices), self.audio_proj_out(hidden).index_select(1, audio_indices)
        heads = Heads().eval()
        torch.manual_seed(1043)
        r, nv, na = 640, 510, 128
        hidden, temb = torch.randn(1, r, 5376, dtype=torch.bfloat16), torch.randn(2, 2688)
        indices = torch.randint(0, 2, (r,)); vi = torch.arange(r - nv, r); ai = torch.arange(r - nv - na, r - nv)
        with torch.inference_mode():
            states = model.norm_out(hidden, temb, indices).to(model.proj_out.weight.dtype)
            expected = (model.proj_out(states).index_select(1, vi), model.audio_proj_out(states).index_select(1, ai))
            actual = heads(hidden, temb, indices.to(torch.int32), vi.to(torch.int32), ai.to(torch.int32))
        check = dict(byte_identical=all(torch.equal(a, e) for a, e in zip(actual, expected)), dtypes=[str(a.dtype) for a in actual])
        print(json.dumps(check), flush=True)
        assert check["byte_identical"]
        rows, k, video, audio = Dim("rows", min=64, max=40960), Dim("k", min=1, max=3), Dim("video", min=32, max=40960), Dim("audio", min=2, max=2048)
        example = (torch.zeros(1, 19333, 5376, dtype=torch.bfloat16), torch.zeros(2, 2688), torch.zeros(19333, dtype=torch.int32),
                   torch.arange(463, 19333, dtype=torch.int32), torch.arange(49, 463, dtype=torch.int32))
        started = time.monotonic()
        asset = h.export_asset(out / "models/heads.aimodel", heads, example, ["hidden", "temb", "indices", "video_indices", "audio_indices"],
                               ["video", "audio"], dynamic_shapes=({1: rows}, {0: k}, {0: rows}, {0: video}, {0: audio}))
        asset["seconds"] = round(time.monotonic() - started, 1)
        record(out, heads=asset, heads_cpu_check=check)
        print(json.dumps(asset), flush=True)
    elif args.stage == "conditioner":
        import shutil
        from torch.export import Dim
        from safetensors import safe_open
        from transformers import Qwen3VLConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer, Qwen3VLTextRotaryEmbedding
        directory = args.snapshot / "FL2VA/text_encoder"
        config = Qwen3VLConfig.from_pretrained(directory).text_config
        config._attn_implementation = "sdpa"
        index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]

        def weights(prefix):
            result = {}
            for name in sorted(n for n in index if n.startswith(prefix)):
                with safe_open(directory / index[name], framework="pt", device="cpu") as f:
                    result[name.removeprefix(prefix)] = f.get_tensor(name)
            return result

        def room(gigabytes):
            free = shutil.disk_usage(out).free / 1e9
            if free - gigabytes < args.minimum_free_gb:
                raise SystemExit(f"stopping: {free:.1f} GB free, export needs ~{gigabytes:.1f} GB, floor {args.minimum_free_gb} GB")

        T = Dim("tokens", min=2, max=4096)
        meta = json.loads((out / "build.json").read_text()) if (out / "build.json").exists() else {}
        conditioner = meta.get("conditioner", dict(layers=[], checks=[]))
        rotary = Qwen3VLTextRotaryEmbedding(config)
        conditioner["rope"] = dict(theta=config.rope_parameters["rope_theta"], mrope_section=list(rotary.mrope_section),
                                   head_dim=config.head_dim, dtype="bfloat16")
        if "embedding" not in conditioner:
            room(1.7)
            table = weights("model.language_model.embed_tokens.")["weight"]
            embedding = torch.nn.Embedding(table.shape[0], table.shape[1], _weight=table).eval()

            class Embed(torch.nn.Module):
                def __init__(self):
                    super().__init__(); self.embedding = embedding
                def forward(self, ids):
                    return self.embedding(ids)
            conditioner["embedding"] = h.export_asset(out / "models/conditioner-embedding.aimodel", Embed(),
                (torch.zeros(1, 49, dtype=torch.int32),), ["ids"], ["hidden"], dynamic_shapes=({1: T},))
            del embedding, table
            record(out, conditioner=conditioner)
            print("exported embedding", flush=True)
        torch.manual_seed(1043)
        probe = torch.randn(1, 16, config.hidden_size, dtype=torch.bfloat16)
        positions = torch.arange(16).view(1, 1, -1).expand(3, 1, -1)
        cos, sin = rotary(probe, positions)
        causal = torch.ones(16, 16, dtype=torch.bool).tril().view(1, 1, 16, 16)
        for layer_index in range(len(conditioner["layers"]), 50):
            with torch.device("meta"):
                layer = Qwen3VLTextDecoderLayer(config, layer_index)
            layer.load_state_dict(weights(f"model.language_model.layers.{layer_index}."), strict=True, assign=True)
            layer.eval()

            class Layer(torch.nn.Module):
                def __init__(self, layer):
                    super().__init__(); self.layer = layer
                def forward(self, hidden, cos, sin):
                    return self.layer(hidden, position_embeddings=(cos, sin), attention_mask=None)
            wrapped = Layer(layer).eval()
            with torch.inference_mode():
                actual = wrapped(probe, cos, sin)
                expected = layer(probe, position_embeddings=(cos, sin), attention_mask=causal)
            check = dict(layer=layer_index, byte_identical=bool(torch.equal(actual, expected)),
                         relative_l2=float((actual.double() - expected.double()).norm() / expected.double().norm()))
            assert check["relative_l2"] < 1e-3, check
            room(1.1)
            started = time.monotonic()
            asset = h.export_asset(out / f"models/conditioner-layer-{layer_index:02d}.aimodel", wrapped,
                (torch.zeros(1, 49, config.hidden_size, dtype=torch.bfloat16), torch.zeros(1, 49, config.head_dim, dtype=torch.bfloat16),
                 torch.zeros(1, 49, config.head_dim, dtype=torch.bfloat16)), ["hidden", "cos", "sin"], ["hidden_out"],
                dynamic_shapes=({1: T}, {1: T}, {1: T}))
            asset["seconds"] = round(time.monotonic() - started, 1)
            conditioner["layers"].append(asset); conditioner["checks"].append(check)
            record(out, conditioner=conditioner)
            print(json.dumps(dict(check, seconds=asset["seconds"])), flush=True)
            del layer, wrapped
    elif args.stage == "parameters":
        checkpoint = h.Checkpoint(args.snapshot, converter)
        spans = h.Spans.sharded(checkpoint.directory)
        adapter = h.adapter_path()
        adapter_spans = h.Spans([adapter])
        expected = json.loads((out / "build.json").read_text())["body_cpu_check"]["bindings"]
        parameters = []
        started = time.monotonic()
        for b in range(50):
            model = h.meta_model()
            model.load_state_dict(checkpoint.converted(lambda n, b=b: n.startswith(f"blocks.{b}.") and ".adaln_proj." not in n),
                                  strict=False, assign=True)
            h.load_turbo(model, adapter, lambda k, b=b: k.startswith(f"transformer_blocks.{b}."))
            model.transformer_blocks[b].adaln_proj = h.CachedModulation()   # the tables replace it, as in the body graph
            params = dict(h.BlockCall(model.transformer_blocks[b]).named_parameters())
            assert list(params) == expected, f"block {b} binding order differs from the body graph"
            entries = []
            for i, (name, tensor) in enumerate(params.items()):
                if ".lora_" in name:
                    segments = adapter_spans.segments(h.lora_source_key(f"transformer_blocks.{b}", name))
                else:
                    segments = h.transformer_source_segments(spans, f"transformer_blocks.{b}." + name.removeprefix("block.").replace(".base_layer", ""))
                entries.append(h.bound_tensor(f"w{i}", segments, tensor, "bfloat16"))
            parameters.append(entries)
            del model, params
            print(json.dumps(dict(block=b, bindings=len(entries), seconds=round(time.monotonic() - started, 1))), flush=True)
        record(out, parameters=parameters, adapter=dict(repository=h.ADAPTER_REPOSITORY, revision=h.ADAPTER_REVISION,
                                                        sha256=h.ADAPTER_SHA256, execution="dynamic_lora", rank=128, alpha=8.0))
    elif args.stage == "prepare":
        from torch.export import Dim
        checkpoint = h.Checkpoint(args.snapshot, converter)
        spans = h.Spans.sharded(checkpoint.directory)
        adapter = h.adapter_path()
        adapter_spans = h.Spans([adapter])
        model = h.meta_model()
        model.load_state_dict(checkpoint.converted(lambda n: n.startswith(("token_refiner.", "condition_proj.", "video_patch_proj.", "audio_patch_proj."))),
                              strict=False, assign=True)
        h.load_turbo(model, adapter, lambda k: k.startswith("token_refiner."))

        class Text(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.context_embedder, self.token_refiner = model.context_embedder, model.token_refiner
            def forward(self, text):
                return self.token_refiner(self.context_embedder(text))

        class Pack(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.proj_in, self.audio_proj_in = model.proj_in, model.audio_proj_in
            def forward(self, video, audio, text, gather):
                # The packed rows are a permutation of [text; video; audio]: `gather[r]` names the source of row r.
                rows = torch.cat((text, self.proj_in(video).to(text.dtype), self.audio_proj_in(audio).to(text.dtype)), dim=1)
                return rows.index_select(1, gather)
        text_module, pack = Text().eval(), Pack().eval()
        graph, params = h.lift(text_module)
        torch.manual_seed(1043)
        T, nv, na = 49, 510, 128
        text = torch.randn(1, T, 5120, dtype=torch.bfloat16)
        video, audio = torch.randn(1, nv, 96), torch.randn(1, na, 32)
        order = torch.randperm(T + nv + na)
        ti, vi, ai = order[:T].sort().values, order[T:T + nv].sort().values, order[T + nv:].sort().values
        with torch.inference_mode():
            refined = model.token_refiner(model.context_embedder(text.to(model.context_embedder.weight.dtype)))
            hidden = refined.new_zeros((1, T + nv + na, refined.shape[-1]))
            hidden = hidden.index_copy(1, ti, refined)
            hidden = hidden.index_copy(1, vi, model.proj_in(video.to(model.proj_in.weight.dtype)).to(refined.dtype))
            expected = hidden.index_copy(1, ai, model.audio_proj_in(audio.to(model.audio_proj_in.weight.dtype)).to(refined.dtype))
            actual_text = graph(text, *params.values())
            gather = torch.empty(T + nv + na, dtype=torch.int32)
            gather[torch.cat((ti, vi, ai))] = torch.arange(T + nv + na, dtype=torch.int32)
            actual = pack(video, audio, actual_text, gather)
        check = dict(text_byte_identical=bool(torch.equal(actual_text, refined)), pack_byte_identical=bool(torch.equal(actual, expected)),
                     text_bindings=list(params))
        print(json.dumps(dict(check, text_bindings=len(params))), flush=True)
        assert check["text_byte_identical"] and check["pack_byte_identical"]
        bindings = []
        for i, (name, tensor) in enumerate(params.items()):
            if ".lora_" in name:
                segments = adapter_spans.segments(name.replace(".turbo.weight", ".default.weight"))
            else:
                segments = h.transformer_source_segments(spans, name.replace(".base_layer", ""))
            bindings.append(h.bound_tensor(f"w{i}", segments, tensor))
        tokens, rows, video_rows, audio_rows = Dim("tokens", min=2, max=4096), Dim("rows", min=64, max=40960), Dim("video", min=32, max=40960), Dim("audio", min=2, max=2048)
        started = time.monotonic()
        text_asset = h.export_asset(out / "models/prepare-text.aimodel", graph, (text, *params.values()),
                                    ["text"] + [f"w{i}" for i in range(len(params))], ["refined"],
                                    dynamic_shapes=(({1: tokens},) + (None,) * len(params),))
        text_asset["seconds"] = round(time.monotonic() - started, 1)
        example = (video, audio, actual_text, gather)
        started = time.monotonic()
        pack_asset = h.export_asset(out / "models/prepare-pack.aimodel", pack, example,
                                    ["video", "audio", "text", "gather"], ["hidden"],
                                    dynamic_shapes=({1: video_rows}, {1: audio_rows}, {1: tokens}, {0: rows}))
        pack_asset["seconds"] = round(time.monotonic() - started, 1)
        record(out, prepare=dict(text=text_asset, text_parameters=bindings, pack=pack_asset), prepare_cpu_check=check)
        print(json.dumps(dict(text=text_asset, pack=pack_asset)), flush=True)
    elif args.stage == "conditioner-layer":
        from torch.export import Dim
        from safetensors import safe_open
        from transformers import Qwen3VLConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer, Qwen3VLTextRotaryEmbedding
        directory = args.snapshot / "FL2VA/text_encoder"
        config = Qwen3VLConfig.from_pretrained(directory).text_config
        config._attn_implementation = "sdpa"
        spans = h.Spans.sharded(directory)

        def layer_weights(index):
            prefix = f"model.language_model.layers.{index}."
            names = sorted(k for k in spans.entries if k.startswith(prefix))
            result = {}
            for name in names:
                file, *_ = spans.span(name)
                with safe_open(file, framework="pt", device="cpu") as f:
                    result[name.removeprefix(prefix)] = f.get_tensor(name)
            return result

        class Layer(torch.nn.Module):
            def __init__(self, layer):
                super().__init__(); self.layer = layer
            def forward(self, hidden, cos, sin):
                return self.layer(hidden, position_embeddings=(cos, sin), attention_mask=None)
        rotary = Qwen3VLTextRotaryEmbedding(config)
        torch.manual_seed(1043)
        probe = torch.randn(1, 16, config.hidden_size, dtype=torch.bfloat16)
        cos, sin = rotary(probe, torch.arange(16).view(1, 1, -1).expand(3, 1, -1))
        causal = torch.ones(16, 16, dtype=torch.bool).tril().view(1, 1, 16, 16)
        parameters, checks, graph, names = [], [], None, None
        for index in range(50):
            with torch.device("meta"):
                layer = Qwen3VLTextDecoderLayer(config, index)
            layer.load_state_dict(layer_weights(index), strict=True, assign=True)
            layer.eval()
            lifted, params = h.lift(Layer(layer).eval())
            if names is None:
                graph, names = lifted, list(params)
            assert list(params) == names, f"layer {index} binding order differs"
            entries = [h.bound_tensor(f"w{i}", spans.segments(f"model.language_model.layers.{index}." + n.removeprefix("layer.")), t, "bfloat16")
                       for i, (n, t) in enumerate(params.items())]
            parameters.append(entries)
            if index in (0, 49):
                in_place = [torch.frombuffer(bytearray(h.read_segments(e["segments"])), dtype=torch.bfloat16).view(e["shape"]) for e in entries]
                with torch.inference_mode():
                    actual = graph(probe, cos, sin, *in_place)
                    expected = layer(probe, position_embeddings=(cos, sin), attention_mask=causal)
                check = dict(layer=index, byte_identical=bool(torch.equal(actual, expected)),
                             relative_l2=float((actual.double() - expected.double()).norm() / expected.double().norm()))
                assert check["relative_l2"] < 1e-3, check
                checks.append(check); print(json.dumps(check), flush=True)
            if index:
                del layer, lifted, params
        T = Dim("tokens", min=2, max=4096)
        example = (torch.zeros(1, 49, config.hidden_size, dtype=torch.bfloat16), torch.zeros(1, 49, config.head_dim, dtype=torch.bfloat16),
                   torch.zeros(1, 49, config.head_dim, dtype=torch.bfloat16))
        first = [torch.frombuffer(bytearray(h.read_segments(e["segments"])), dtype=torch.bfloat16).view(e["shape"]) for e in parameters[0]]
        started = time.monotonic()
        asset = h.export_asset(out / "models/conditioner-layer.aimodel", graph, (*example, *first),
                               ["hidden", "cos", "sin"] + [f"w{i}" for i in range(len(names))], ["hidden_out"],
                               dynamic_shapes=(({1: T}, {1: T}, {1: T}) + (None,) * len(names),))
        asset["seconds"] = round(time.monotonic() - started, 1)
        file, offset, length, dtype, shape = spans.span("model.language_model.embed_tokens.weight")
        assert dtype == "BF16" and shape == [config.vocab_size, config.hidden_size]
        embedding = dict(path=str(file), offset=offset, length=length, dtype="bfloat16", shape=shape)
        meta = json.loads((out / "build.json").read_text())
        rope = dict(theta=config.rope_parameters["rope_theta"], mrope_section=list(rotary.mrope_section), head_dim=config.head_dim, dtype="bfloat16")
        conditioner = dict(layer=asset, parameters=parameters, bindings=names, embedding_table=embedding, checks=checks, rope=rope,
                           superseded=dict(embedding=meta.get("conditioner", {}).get("embedding"), layers=len(meta.get("conditioner", {}).get("layers", []))))
        record(out, conditioner=conditioner)
        print(json.dumps(dict(asset=asset, bindings=len(names))), flush=True)
    elif args.stage == "conditioner-rope":
        from transformers import Qwen3VLConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding
        config = Qwen3VLConfig.from_pretrained(args.snapshot / "FL2VA/text_encoder").text_config
        rotary = Qwen3VLTextRotaryEmbedding(config)
        probe = torch.zeros(1, 1, config.hidden_size, dtype=torch.bfloat16)
        with torch.inference_mode():
            cos, sin = rotary(probe, torch.arange(4096).view(1, 1, -1).expand(3, 1, -1))
            short = rotary(probe, torch.arange(49).view(1, 1, -1).expand(3, 1, -1))
        check = dict(prefix_identical=bool(torch.equal(short[0], cos[:, :49]) and torch.equal(short[1], sin[:, :49])), dtype=str(cos.dtype))
        assert check["prefix_identical"] and cos.dtype == torch.bfloat16, check
        tables = out / "tables/conditioner"
        record(out, conditioner_rope=dict(cos=h.raw_fixture(tables / "rope-cos.raw", "cos", cos[0]), sin=h.raw_fixture(tables / "rope-sin.raw", "sin", sin[0]),
                                          check=check))
        print(json.dumps(check), flush=True)
    elif args.stage == "audio-decoder":
        import diffusers
        from diffusers import AutoencoderKLMiniMaxH3Audio
        from torch.export import Dim
        target = out / "converted/audio-vae"
        if not target.exists():
            converter.convert_audio_vae(str(args.snapshot / "FL2VA"), str(target), diffusers.__version__)
        vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(target, torch_dtype=torch.float32).eval()

        class AudioDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dec_in_proj, self.decoder = vae.dec_in_proj, vae.decoder
                self.register_buffer("mean", torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(1, -1, 1))
                self.register_buffer("std", torch.tensor(vae.config.latents_std, dtype=torch.float32).view(1, -1, 1))
            def forward(self, rows):
                latents = rows.permute(0, 2, 1) * self.std + self.mean
                return self.decoder(self.dec_in_proj(latents)).squeeze(1)
        module = AudioDecoder().eval()
        torch.manual_seed(1043)
        rows = torch.randn(2, 207, 32)
        started = time.monotonic()
        with torch.inference_mode():
            actual = module(rows)
            mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1); std = torch.tensor(vae.config.latents_std).view(1, -1, 1)
            reference = vae.decode(rows.permute(0, 2, 1).contiguous() * std + mean, return_dict=False)[0].float().squeeze(1)
        check = dict(shape=list(actual.shape), byte_identical=bool(torch.equal(actual, reference)),
                     relative_l2=float((actual.double() - reference.double()).norm() / reference.double().norm()),
                     seconds=round(time.monotonic() - started, 1))
        print(json.dumps(check), flush=True)
        assert check["relative_l2"] < 1e-5, check
        files = {p.name: h.sha256_file(p) for p in sorted(target.iterdir()) if p.is_file()}
        started = time.monotonic()
        asset = h.export_asset(out / "models/audio-decoder.aimodel", module, (rows,), ["rows"], ["pcm"],
                               dynamic_shapes=({1: Dim("latents", min=2, max=2048)},))
        asset["seconds"] = round(time.monotonic() - started, 1)
        record(out, audio_decoder=dict(asset, sample_rate=vae.config.sampling_rate, hop=800, channels=2), audio_decoder_cpu_check=check,
               converted_audio_vae=dict(path=str(target), files=files))
        print(json.dumps(asset), flush=True)
    elif args.stage == "audio-decoder-fixed":
        from diffusers import AutoencoderKLMiniMaxH3Audio
        target = out / "converted/audio-vae"
        vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(target, torch_dtype=torch.float32).eval()

        class AudioDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dec_in_proj, self.decoder = vae.dec_in_proj, vae.decoder
                self.register_buffer("mean", torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(1, -1, 1))
                self.register_buffer("std", torch.tensor(vae.config.latents_std, dtype=torch.float32).view(1, -1, 1))
            def forward(self, rows):
                latents = rows.permute(0, 2, 1) * self.std + self.mean
                return self.decoder(self.dec_in_proj(latents)).squeeze(1)
        module = AudioDecoder().eval()
        meta = json.loads((out / "build.json").read_text())
        fixed = meta.get("audio_decoder_fixed", {})
        for na in sorted({int(v) for v in args.audio_latents.split(",")}):
            assert 2 <= na <= 2048
            torch.manual_seed(1043)
            rows = torch.randn(2, na, 32)
            with torch.inference_mode():
                actual = module(rows)
                mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1); std = torch.tensor(vae.config.latents_std).view(1, -1, 1)
                reference = vae.decode(rows.permute(0, 2, 1).contiguous() * std + mean, return_dict=False)[0].float().squeeze(1)
            check = dict(latents=na, shape=list(actual.shape), byte_identical=bool(torch.equal(actual, reference)),
                         relative_l2=float((actual.double() - reference.double()).norm() / reference.double().norm()))
            print(json.dumps(check), flush=True)
            assert check["relative_l2"] < 1e-5 and check["shape"] == [2, 800 * na], check
            started = time.monotonic()
            asset = h.export_asset(out / f"models/audio-decoder-na{na}.aimodel", module, (rows,), ["rows"], ["pcm"])
            asset["seconds"] = round(time.monotonic() - started, 1)
            fixed[str(na)] = dict(asset, check=check)
            record(out, audio_decoder_fixed=fixed)
            print(json.dumps(asset), flush=True)
    elif args.stage == "finalize":
        meta = json.loads((out / "build.json").read_text())
        def asset(item):
            assert h.sha256_file(Path(item["path"]) / "main.mlirb") == item["sha256"], item["path"]
            return dict(path=item["path"], function=item["function"], sha256=item["sha256"])
        def table(item):
            assert h.sha256_file(Path(item["path"])) == item["sha256"], item["path"]
            return {k: item[k] for k in ("name", "path", "dtype", "shape", "sha256")}
        def bound(item):   # in-place weights: re-read every range, so the manifest only ever names verified bytes
            assert h.sha256_bytes(h.read_segments(item["segments"])) == item["sha256"], item["name"]
            return {k: item[k] for k in ("name", "segments", "dtype", "shape", "sha256")}
        recipes = {}
        for key, entry in meta["tables"].items():
            recipe, task = key.split("/")
            arms = entry["schedule"]
            recipes.setdefault(recipe, dict(schedule=dict(video=arms[0], audio=arms[1]), tasks={}))
            recipes[recipe]["tasks"][task] = [dict(timesteps=step["timesteps"], temb=table(step["temb"]),
                                                   blocks=[table(b) for b in step["blocks"]]) for step in entry["steps"]]
        c = meta.get("conditioner", {})
        lifted = "layer" in c and len(c.get("parameters", [])) == 50 and "conditioner_rope" in meta
        missing = [name for name, ok in (("prepare", "prepare" in meta), ("parameters", len(meta.get("parameters", [])) == 50),
                   ("conditioner", lifted and "conditioner_rope" in meta), ("adapter", "adapter" in meta), ("audio_decoder", "audio_decoder" in meta),
                   ("audio_decoder_fixed", bool(meta.get("audio_decoder_fixed")))) if not ok]
        def table_range(item):
            data = h.read_segments([item])
            return dict(item, sha256=h.sha256_bytes(data))
        conditioner = dict(embedding_table=table_range(c["embedding_table"]), layer=asset(c["layer"]), parameters=[[bound(t) for t in layer] for layer in c["parameters"]],
                           tap=50, rope=c["rope"], rope_tables=dict(cos=table(meta["conditioner_rope"]["cos"]), sin=table(meta["conditioner_rope"]["sin"]))) if lifted else None
        prepare = meta.get("prepare")
        prepare = dict(text=asset(prepare["text"]), text_parameters=[bound(t) for t in prepare["text_parameters"]], pack=asset(prepare["pack"])) if prepare else None
        audio = meta.get("audio_decoder")
        audio = dict(asset(audio), sample_rate=audio["sample_rate"], hop=audio["hop"], channels=audio["channels"],
                     fixed={na: asset(item) for na, item in sorted(meta.get("audio_decoder_fixed", {}).items())}) if audio else None
        bundle = dict(schema=3, complete=not missing, missing=missing, snapshot=h.SNAPSHOT, diffusers_commit=h.COMMIT,
                      transformer_weight_layout=h.CANONICAL_QKV_LAYOUT, adapter=meta.get("adapter"), tasks=["t2va", "fl2va"],
                      limits=dict(rows=meta["body_dynamic"]["rows"], cardinality=meta["body_dynamic"]["cardinality"], text_tokens=[2, 4096]),
                      tokenizer=str(args.snapshot / "FL2VA/tokenizer"), conditioner=conditioner, prepare=prepare,
                      body=asset(meta["body"]), heads=asset(meta["heads"]),
                      parameters=[[bound(t) for t in block] for block in meta.get("parameters", [])], recipes=recipes,
                      # Default: the bf16 tile decoder (E1: 2.4x faster, half the memory, PSNR 55 dB vs float32); the float32
                      # one stays listed as a parity/diagnostic decoder.
                      video_decoder=dict(asset(meta.get("video_decoder_bf16", meta["video_decoder"])), tile=meta["video_tile"], window_latents=7,
                                         window_frames=22, window_stride=5,
                                         precision="bfloat16" if "video_decoder_bf16" in meta else "float32",
                                         float32=asset(meta["video_decoder"]) if "video_decoder_bf16" in meta else None), audio_decoder=audio,
                      video_encoder_first={key: asset(item) for key, item in sorted(meta.get("video_encoder_first", {}).items())})
        (out / "bundle.json").write_text(json.dumps(bundle, indent=1))
        sizes = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        print(json.dumps(dict(complete=bundle["complete"], missing=missing, bytes_on_disk=sizes)), flush=True)
    elif args.stage == "video-decoder-bf16":
        from diffusers import AutoencoderKLMiniMaxH3
        reference_vae = AutoencoderKLMiniMaxH3.from_pretrained(out / "converted/vae", torch_dtype=torch.float32).eval()
        vae = AutoencoderKLMiniMaxH3.from_pretrained(out / "converted/vae", torch_dtype=torch.bfloat16).eval()
        reference_vae.use_tiling = vae.use_tiling = False

        class Bf16WindowTileDecoder(WindowTileDecoder):
            def forward(self, rows):
                s = self.side
                z = rows.view(7, s // 2, s // 2, 24, 2, 2).permute(3, 0, 1, 4, 2, 5).reshape(1, 24, 7, s, s)
                z = (z * self.std + self.mean).to(self.vae.post_quant_conv.weight.dtype)
                clip = self.vae.decoder(self.vae.post_quant_conv(z))
                return torch.cat((clip[:, :, 3:20], clip[:, :, 23:28]), dim=2).float()
        module = Bf16WindowTileDecoder(vae).eval()
        torch.manual_seed(1043)
        rows = torch.randn(448, 96)
        normalized = rows.view(7, 8, 8, 24, 2, 2).permute(3, 0, 1, 4, 2, 5).reshape(1, 24, 7, 16, 16)
        z = normalized * module.std + module.mean
        started = time.monotonic()
        with torch.inference_mode():
            actual = module(rows)
            reference = vae._decode(z.to(torch.bfloat16)).float()
            full = reference_vae._decode(z)
        check = dict(shape=list(actual.shape), byte_identical=bool(torch.equal(actual, reference)),
                     relative_l2=float((actual.double() - reference.double()).norm() / reference.double().norm()),
                     relative_l2_vs_float32=float((actual.double() - full.double()).norm() / full.double().norm()),
                     dtypes=sorted({str(p.dtype) for p in vae.parameters()}), seconds=round(time.monotonic() - started, 1))
        print(json.dumps(check), flush=True)
        assert check["byte_identical"], check
        started = time.monotonic()
        asset = h.export_asset(out / "models/video-decoder-tile256-bf16.aimodel", module, (rows,), ["rows"], ["pixels"])
        asset["seconds"] = round(time.monotonic() - started, 1)
        record(out, video_decoder_bf16=dict(asset, check=check))
        print(json.dumps(asset), flush=True)
    elif args.stage == "video-encoder-first":
        from diffusers import AutoencoderKLMiniMaxH3
        from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
        from diffusers.utils.torch_utils import randn_tensor
        vae = AutoencoderKLMiniMaxH3.from_pretrained(out / "converted/vae", torch_dtype=torch.float32).eval()
        pixel_mean, pixel_std, seed = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 42

        class FirstFrameEncoder(torch.nn.Module):
            def __init__(self, eps):
                super().__init__()
                self.vae = vae
                self.register_buffer("eps", eps)
                self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(1, -1, 1, 1, 1))
                self.register_buffer("pixel_std", torch.tensor(pixel_std).view(1, -1, 1, 1, 1))
                self.register_buffer("mean", torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1))
                self.register_buffer("std", torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1))

            def forward(self, pixels):
                posterior = self.vae.encode((pixels.div(255.0) - self.pixel_mean) / self.pixel_std, return_dict=False)[0]
                latents = (posterior.mean + posterior.std * self.eps).to(torch.float16).float()
                return (latents - self.mean) / self.std
        meta = json.loads((out / "build.json").read_text())
        encoders = meta.get("video_encoder_first", {})
        for width, height in (tuple(int(v) for v in c.split("x")) for c in args.canvases.split(",")):
            key = f"{width}x{height}"
            torch.manual_seed(1043)
            image = torch.randint(0, 256, (1, 3, 1, height, width), dtype=torch.uint8)
            started = time.monotonic()
            with torch.inference_mode():
                moments = vae.encode((image.float().div(255.0) - torch.tensor(pixel_mean).view(1, -1, 1, 1, 1)) / torch.tensor(pixel_std).view(1, -1, 1, 1, 1),
                                     return_dict=False)[0]
                eps = randn_tensor(moments.mean.shape, generator=torch.Generator().manual_seed(seed), device=torch.device("cpu"), dtype=moments.mean.dtype)
                module = FirstFrameEncoder(eps).eval()
                actual = module(image.float())
                reference = encode_vae_condition(vae, image, pixel_mean, pixel_std, seed)
            check = dict(canvas=key, shape=list(actual.shape), byte_identical=bool(torch.equal(actual, reference)),
                         relative_l2=float((actual.double() - reference.double()).norm() / reference.double().norm()),
                         seconds=round(time.monotonic() - started, 1))
            print(json.dumps(check), flush=True)
            assert check["byte_identical"], check
            started = time.monotonic()
            asset = h.export_asset(out / f"models/video-encoder-first-{key}.aimodel", module, (image.float(),), ["pixels"], ["condition"])
            asset["seconds"] = round(time.monotonic() - started, 1)
            encoders[key] = dict(asset, check=check, encode_seed=seed)
            record(out, video_encoder_first=encoders)
            print(json.dumps(asset), flush=True)
    elif args.stage == "video-encoder-first-tiled":
        # The same encode_vae_condition, split where the pipeline already splits it: `_encode_clip` runs the encoder
        # and quant_conv per 256 px source tile. The whole-frame graph unrolls all tiles into one graph whose native
        # scratch grows with the tile count (about 1.8 GiB per tile; 27 GiB at 960x544). Here: ONE tile graph
        # (normalized pixels -> moments, run per tile) and one small TAIL graph per canvas (`_stitch_tiles` in latent
        # space, the posterior with the fixed seed-42 draw baked in, float16 rounding, latent normalization).
        from diffusers import AutoencoderKLMiniMaxH3
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
        from diffusers.utils.torch_utils import randn_tensor
        # The overlay never writes into the base export: its bundle.json, build.json and models stay untouched.
        if args.base_bundle and out == args.base_bundle.resolve().parent:
            raise SystemExit("--output must not be the base bundle's directory; write the overlay elsewhere")
        converted = (args.converted or out / "converted").resolve()
        vae = AutoencoderKLMiniMaxH3.from_pretrained(converted / "vae", torch_dtype=torch.float32).eval()
        assert vae.use_tiling and vae.tile_sample_min_height == vae.tile_sample_min_width == 256
        assert vae.tile_sample_min_overlap_height == vae.tile_sample_min_overlap_width == 64 and vae.spatial_compression_ratio == 16
        pixel_mean, pixel_std, seed, side = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 42, 256

        class TileEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vae = vae
                self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(1, -1, 1, 1, 1))
                self.register_buffer("pixel_std", torch.tensor(pixel_std).view(1, -1, 1, 1, 1))

            def forward(self, pixels):   # [1,3,1,256,256] in 0...255 -> moments [1,48,1,16,16]
                return self.vae.quant_conv(self.vae.encoder((pixels.div(255.0) - self.pixel_mean) / self.pixel_std))

        class TileTail(torch.nn.Module):
            def __init__(self, eps, rows, columns, y_overlaps, x_overlaps):
                super().__init__()
                self.vae, self.rows, self.columns = vae, rows, columns
                self.y_overlaps = [o // vae.spatial_compression_ratio for o in y_overlaps]
                self.x_overlaps = [o // vae.spatial_compression_ratio for o in x_overlaps]
                self.register_buffer("eps", eps)
                self.register_buffer("mean", torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1))
                self.register_buffer("std", torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1))

            def forward(self, tiles):   # [rows*columns, 48, 1, 16, 16], row-major -> condition [1,24,1,H/16,W/16]
                grid = [[tiles[i * self.columns + j:i * self.columns + j + 1] for j in range(self.columns)] for i in range(self.rows)]
                posterior = DiagonalGaussianDistribution(self.vae._stitch_tiles(grid, self.y_overlaps, self.x_overlaps))
                latents = (posterior.mean + posterior.std * self.eps).to(torch.float16).float()
                return (latents - self.mean) / self.std

        meta_path = out / "build.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else dict(schema=1, snapshot=h.SNAPSHOT, diffusers_commit=h.COMMIT)
        tile_module = TileEncoder().eval()
        tails, checks = {}, {}
        for width, height in (tuple(int(v) for v in c.split("x")) for c in args.canvases.split(",")):
            key = f"{width}x{height}"
            ys, _, yo = vae._split_tiles(height, side, 64)
            xs, _, xo = vae._split_tiles(width, side, 64)
            torch.manual_seed(1043)
            image = torch.randint(0, 256, (1, 3, 1, height, width), dtype=torch.uint8)
            started = time.monotonic()
            with torch.inference_mode():
                moments = torch.cat([tile_module(image[..., y:y + side, x:x + side].float()) for y in ys for x in xs])
                eps = randn_tensor((1, vae.config.latent_channels, 1, height // 16, width // 16), generator=torch.Generator().manual_seed(seed),
                                   device=torch.device("cpu"), dtype=moments.dtype)
                tail = TileTail(eps, len(ys), len(xs), yo, xo).eval()
                actual = tail(moments)
                reference = encode_vae_condition(vae, image, pixel_mean, pixel_std, seed)
                # The draw must be the one encode_vae_condition takes (posterior.sample on the stitched mean's shape).
                stitched = vae._encode(((image.float().div(255.0) - torch.tensor(pixel_mean).view(1, -1, 1, 1, 1)) / torch.tensor(pixel_std).view(1, -1, 1, 1, 1)))
                assert list(DiagonalGaussianDistribution(stitched).mean.shape) == list(eps.shape)
            check = dict(canvas=key, tiles=len(ys) * len(xs), y_starts=ys, x_starts=xs, y_overlaps=yo, x_overlaps=xo, shape=list(actual.shape),
                         byte_identical=bool(torch.equal(actual, reference)),
                         relative_l2=float((actual.double() - reference.double()).norm() / reference.double().norm()),
                         seconds=round(time.monotonic() - started, 1))
            print(json.dumps(check), flush=True)
            assert check["byte_identical"], check
            started = time.monotonic()
            asset = h.export_asset(out / f"models/video-encoder-first-tail-{key}.aimodel", tail, (moments,), ["tiles"], ["condition"])
            asset["seconds"] = round(time.monotonic() - started, 1)
            tails[key], checks[key] = asset, check
            print(json.dumps(asset), flush=True)
        started = time.monotonic()
        tile_asset = h.export_asset(out / "models/video-encoder-first-tile256.aimodel", tile_module, (torch.zeros(1, 3, 1, side, side),), ["pixels"], ["moments"])
        tile_asset["seconds"] = round(time.monotonic() - started, 1)
        tiled = dict(tile=tile_asset, tile_size=side, moments_channels=2 * vae.config.latent_channels, tails=tails, checks=checks, encode_seed=seed,
                     converted_vae=str(converted / "vae"), recipe_sha256=h.sha256_file(Path(__file__)))
        record(out, video_encoder_first_tiled=tiled)
        if args.base_bundle:
            base_path = args.base_bundle.resolve()
            bundle = json.loads(base_path.read_text())
            assert bundle["schema"] == 3 and "video_encoder_first_tiled" not in bundle
            entry = lambda a: dict(path=a["path"], function=a["function"], sha256=a["sha256"])
            bundle["video_encoder_first_tiled"] = dict(tile=entry(tile_asset), tile_size=side, moments_channels=2 * vae.config.latent_channels,
                                                       tails={k: entry(v) for k, v in sorted(tails.items())})
            bundle["lineage"] = dict(base_bundle=str(base_path), base_bundle_sha256=h.sha256_file(base_path),
                                     change="added video_encoder_first_tiled (tile graph + per-canvas tails); every other entry is the base bundle's")
            (out / "bundle.json").write_text(json.dumps(bundle, indent=1))
            print(json.dumps(dict(bundle=str(out / "bundle.json"), sha256=h.sha256_file(out / "bundle.json"))), flush=True)
    elif args.stage == "video-decoder":
        from diffusers import AutoencoderKLMiniMaxH3
        vae = AutoencoderKLMiniMaxH3.from_pretrained(out / "converted/vae", torch_dtype=torch.float32).eval()
        vae.use_tiling = False
        module = WindowTileDecoder(vae).eval()
        torch.manual_seed(1043)
        rows = torch.randn(448, 96)
        normalized = rows.view(7, 8, 8, 24, 2, 2).permute(3, 0, 1, 4, 2, 5).reshape(1, 24, 7, 16, 16)
        z = normalized * module.std + module.mean   # what a pipeline hands `_decode`
        started = time.monotonic()
        with torch.inference_mode():
            actual = module(rows)
            reference = vae._decode(z)          # the original single-window, untiled control flow
        check = dict(shape=list(actual.shape), byte_identical=bool(torch.equal(actual, reference)),
                     relative_l2=float((actual.double() - reference.double()).norm() / reference.double().norm()),
                     seconds=round(time.monotonic() - started, 1))
        print(json.dumps(check), flush=True)
        assert check["byte_identical"], check
        started = time.monotonic()
        asset = h.export_asset(out / "models/video-decoder-tile256.aimodel", module, (rows,), ["rows"], ["pixels"])
        asset["seconds"] = round(time.monotonic() - started, 1)
        record(out, video_decoder=asset, video_decoder_cpu_check=check, video_window_latents=7, video_window_frames=22,
               video_window_stride=5, video_tile=256)
        print(json.dumps(asset), flush=True)


if __name__ == "__main__":
    main()

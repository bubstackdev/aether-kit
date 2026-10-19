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
  finalize       Write OUTPUT/bundle.json (schema 2) from build.json, re-hashing every referenced asset stream and
                 table file. `complete` is true only when every required part (prepare, LoRA weights) exists.
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
    ap.add_argument("--stage", required=True, choices=["convert-vae", "video-decoder", "body", "tables", "heads", "conditioner", "finalize"])
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
    elif args.stage == "finalize":
        meta = json.loads((out / "build.json").read_text())
        def asset(item):
            assert h.sha256_file(Path(item["path"]) / "main.mlirb") == item["sha256"], item["path"]
            return dict(path=item["path"], function=item["function"], sha256=item["sha256"])
        def table(item):
            assert h.sha256_file(Path(item["path"])) == item["sha256"], item["path"]
            return {k: item[k] for k in ("name", "path", "dtype", "shape", "sha256")}
        recipes = {}
        for key, entry in meta["tables"].items():
            recipe, task = key.split("/")
            arms = entry["schedule"]
            recipes.setdefault(recipe, dict(schedule=dict(video=arms[0], audio=arms[1]), tasks={}))
            recipes[recipe]["tasks"][task] = [dict(timesteps=step["timesteps"], temb=table(step["temb"]),
                                                   blocks=[table(b) for b in step["blocks"]]) for step in entry["steps"]]
        conditioner = meta.get("conditioner", {})
        missing = [name for name, ok in (("prepare", "prepare" in meta), ("parameters", "parameters" in meta),
                   ("conditioner", len(conditioner.get("layers", [])) == 50 and "embedding" in conditioner),
                   ("adapter", "adapter" in meta)) if not ok]
        bundle = dict(schema=2, complete=not missing, missing=missing, snapshot=h.SNAPSHOT, diffusers_commit=h.COMMIT,
                      transformer_weight_layout=h.CANONICAL_QKV_LAYOUT, adapter=meta.get("adapter"), tasks=["t2va", "fl2va"],
                      limits=dict(rows=meta["body_dynamic"]["rows"], cardinality=meta["body_dynamic"]["cardinality"], text_tokens=[2, 4096]),
                      tokenizer=str(args.snapshot / "FL2VA/tokenizer"),
                      conditioner=dict(embedding=asset(conditioner["embedding"]) if "embedding" in conditioner else None,
                                       layers=[asset(a) for a in conditioner.get("layers", [])], tap=50, rope=conditioner.get("rope")),
                      prepare=asset(meta["prepare"]) if "prepare" in meta else None, body=asset(meta["body"]), heads=asset(meta["heads"]),
                      parameters=meta.get("parameters", []), recipes=recipes,
                      video_decoder=dict(asset(meta["video_decoder"]), tile=meta["video_tile"], window_latents=7, window_frames=22,
                                         window_stride=5), audio_decoder=None)
        (out / "bundle.json").write_text(json.dumps(bundle, indent=1))
        sizes = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        print(json.dumps(dict(complete=bundle["complete"], missing=missing, bytes_on_disk=sizes)), flush=True)
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

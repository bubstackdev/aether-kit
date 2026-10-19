"""Export a MiniMax H3 profile for AetherKit from the original FL2VA checkpoint (CPU only, lab-free).

Stages (each records its outputs, with SHA-256, in OUTPUT/build.json):
  convert-vae    Convert the video VAE to Diffusers layout under OUTPUT/converted/vae (pinned converter).
  body           Export ONE transformer-block graph for all 50 blocks: weights (w0..w21, Turbo LoRA slots included) are
                 inputs, and both the row count and the timestep cardinality (k = 1...3, table [6,3k,D]) are dynamic.
                 CPU check on 128 rows of real block-0 weights (zero LoRA): byte-identical to the PEFT block.
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
    ap.add_argument("--stage", required=True, choices=["convert-vae", "video-decoder", "body"])
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

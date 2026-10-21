"""CPU-only original H3 video decoder export with explicit weight inputs.

Avoids load-time constant-weight optimization. Keeps the original VAE decode,
spatial tiling and temporal window math. No dependencies/lab/checkpoint changes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

from h3_export import COMMIT, SNAPSHOT, digest, save


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lab", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--diagnostic-latents", type=Path, required=True)
    ap.add_argument("--spatial-tiles", action="store_true", help="Export one original 256px tile; Rust retains source grid/blending")
    args = ap.parse_args()
    lab, out = args.lab.resolve(), args.output.resolve()
    if out.is_relative_to(lab) or out == args.bundle.resolve().parent:
        raise ValueError("output must be separate from lab and input bundle")
    out.mkdir(parents=True, exist_ok=True)
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(lab / "forge"), str(lab / "forge/vendor/python"), str(lab / "forge/vendor/diffusers/src")]
    import qualify_block as q
    from h3_fx import parameter_input_graph
    import torch
    import coreai_torch as ct
    from safetensors import safe_open
    from diffusers import AutoencoderKLMiniMaxH3
    torch.set_num_threads(4)
    environment = json.loads((lab / "records/environment.json").read_text())
    assert environment["unmodifiedDependencies"] and environment["diffusers"]["commit"] == COMMIT
    meta = json.loads(args.bundle.read_text())
    assert meta["complete"] and meta["snapshot"] == SNAPSHOT and meta["diffusers_commit"] == COMMIT
    assert (meta["width"], meta["height"], meta["frames"]) == (960, 544, 124)
    with torch.device("meta"):
        vae = AutoencoderKLMiniMaxH3.from_config(q.SNAPSHOT / "vae/config.json")
    converter = q.converter()
    weights = {}
    source = q.SNAPSHOT / "FL2VA/video_vae/source/model.safetensors"
    with safe_open(source, framework="pt", device="cpu") as f:
        for name in f.keys():
            weights.update(converter.convert_video_vae_key(name, f.get_tensor(name), dict(vae.config)))
    vae.load_state_dict(weights, strict=True, assign=True)
    vae.eval()
    vae.enable_tiling()
    vae.decoder.rope = type(vae.decoder.rope)(dim=int(vae.config.decoder_rope_dim_ratio * vae.config.decoder_attention_head_dim), theta=vae.config.decoder_rope_theta)

    class Video(torch.nn.Module):
        def __init__(self, height, width):
            super().__init__()
            self.vae = vae
            self.h, self.w = height // 16, width // 16
        def forward(self, rows):
            z = rows.reshape(1, 7, self.h // 2, self.w // 2, 24, 1, 2, 2).permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(1, 24, 7, self.h, self.w)
            z = z * torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1) + torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
            return self.vae.decode(z, return_dict=False)[0]

    def fixture(name, tensor):
        tensor = tensor.detach().contiguous().cpu()
        path = out / "fixtures" / (name + ".raw")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = tensor.view(torch.uint8).numpy().tobytes()
        path.write_bytes(data)
        return dict(name=Path(name).name, path=str(path), shape=list(tensor.shape),
                    dtype=str(tensor.dtype).removeprefix("torch."), sha256=hashlib.sha256(data).hexdigest())

    def export(name, graph, rows, parameters):
        print("CPU export", name, "parameters", len(parameters), flush=True)
        begin = time.monotonic()
        path = out / "models" / (name + ".aimodel")
        path.parent.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode():
            ep = torch.export.export(graph, (rows, *parameters.values())).run_decompositions(ct.get_decomp_table())
        assert all(x.kind.name != "PARAMETER" for x in ep.graph_signature.input_specs), "large parameters must stay explicit"
        # Normalization constants and derived rotary buffers remain embedded;
        # the 9 GiB of trained weights must all be function arguments.
        for key, value in dict(ep.state_dict, **ep.constants).items():
            if isinstance(value, torch.Tensor):
                assert value.numel() * value.element_size() <= (1 << 20), (key, value.shape)
        cv = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
        cv.add_exported_program(ep, input_names=["rows"] + [f"w{i}" for i in range(len(parameters))], output_names=["pixels"], state_names=[], entrypoint_name="main")
        program = cv.to_coreai()
        program.optimize()
        program.save_asset(path)
        checksum = digest(path / "main.mlirb")
        assert (path / "main.hash").read_bytes().hex() == checksum
        print("exported", name, round(time.monotonic() - begin, 2), "seconds", (path / "main.mlirb").stat().st_size, "bytes", flush=True)
        return dict(path=str(path), function="main", sha256=checksum)

    diagnostic_size = 256 if args.spatial_tiles else 128
    original = Video(diagnostic_size, diagnostic_size).eval()
    raw = args.diagnostic_latents.read_bytes()
    if args.spatial_tiles:
        torch.manual_seed(1043)
        rows = torch.randn(7 * 8 * 8, 96)
    else:
        rows = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(7 * 4 * 4, 96)
    print("CPU original VAE reference", diagnostic_size, flush=True)
    with torch.inference_mode():
        expected = original(rows)
    small, small_parameters = parameter_input_graph(original, (rows,))
    with torch.inference_mode():
        actual = small(rows, *small_parameters.values())
    assert torch.equal(actual, expected), "parameter lifting changed original CPU decode"
    small_model = export(f"video-functional-{diagnostic_size}", small, rows, small_parameters)
    probe = dict(scope=f"original {diagnostic_size}px/22-frame source-precision VAE; explicit weights versus original CPU module",
                 input=fixture("probe/rows", rows), reference=fixture("probe/pixels", expected), model=small_model)
    if args.spatial_tiles:
        model, parameters = small_model, small_parameters
        ys, _, yo = vae._split_tiles(544, 256, 64)
        xs, _, xo = vae._split_tiles(960, 256, 64)
        tiling = dict(tile_height=256, tile_width=256, y_starts=ys, x_starts=xs, y_overlaps=yo, x_overlaps=xo)
        meta["video_spatial_tiles"] = tiling
        # Independent source stitching and packed-latent extraction oracles.
        from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
        packed = torch.arange(7 * 17 * 30 * 96, dtype=torch.float32).reshape(1, 7 * 17 * 30, 96) * 0.00317
        z = packed.reshape(1, 7, 17, 30, 24, 1, 2, 2).permute(0,4,1,5,2,6,3,7).reshape(1,24,7,34,60)
        tile_records, decoded_rows = [], []
        for iy, y in enumerate(ys):
            decoded_row=[]
            for ix, x in enumerate(xs):
                number=iy*len(xs)+ix
                pixels = ((torch.arange(3*22*256*256,dtype=torch.float32).reshape(1,3,22,256,256) % 541) * 0.0137 + number * 0.31)
                inputs=patchify_video_latents(z[:,:,:,y//16:y//16+16,x//16:x//16+16],(1,2,2))
                tile_records.append(dict(pixels=fixture(f"spatial/pixels-{number}",pixels),rows=fixture(f"spatial/rows-{number}",inputs)))
                decoded_row.append(pixels)
            decoded_rows.append(decoded_row)
        reference=vae._stitch_tiles(decoded_rows,yo,xo)
        save(out/"spatial-oracle.json",dict(width=960,height=544,frames=22,tiling=tiling,
            packed_rows=fixture("spatial/packed",packed),tiles=tile_records,
            reference_sha256=hashlib.sha256(reference.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
            scope="original source spatial stitching with deterministic decoder outputs, plus source patchify on odd latent offsets; no decoder stub used for numerical VAE oracle"))
    else:
        full_rows = torch.zeros(7 * 17 * 30, 96)
        graph, parameters = parameter_input_graph(Video(544, 960).eval(), (full_rows,))
        assert list(parameters) == list(small_parameters), "VAE parameter binding order differs"
        model = export("video-functional-960x544", graph, full_rows, parameters)
    packet = []
    for i, (name, tensor) in enumerate(parameters.items()):
        item = fixture(f"parameters/w{i}", tensor)
        packet.append(dict(item, parameter=name))
    probe["parameters"] = packet
    save(out / "probe.json", probe)
    meta["video_decoder"] = model
    meta["video_parameters"] = packet
    meta["video_decoder_oracle"] = str(out / "probe.json")
    meta["video_decoder_recipe"] = dict(kind="original_f32_explicit_parameters", recipe_sha256=digest(__file__), source=str(source))
    save(out / "bundle.json", meta)
    save(out / "cpu-checks.json", dict(passed=True, gpu_work=False, original_diagnostic_decode_byte_identical=True,
         diagnostic_size=diagnostic_size, spatial_tiles=args.spatial_tiles,
         scope=f"{diagnostic_size}px original VAE versus lifted CPU graph; no full-profile or native decode result",
         parameters=len(packet), parameter_bytes=sum(t.numel()*t.element_size() for t in parameters.values()),
         source_snapshot=SNAPSHOT, diffusers_commit=COMMIT, recipe_sha256=digest(__file__),
         bundle_sha256=digest(out / "bundle.json")))
    print("Functional decoder CPU preparation complete", flush=True)


if __name__ == "__main__":
    main()

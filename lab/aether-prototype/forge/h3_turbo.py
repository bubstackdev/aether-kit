"""Export the pinned 544p eight-step H3 recipe on CPU; never run GPU inference.

Keeps PEFT's dynamic low-rank branches instead of merging updates into BF16
weights. Reuses the base bundle's unchanged conditioner, AdaLN and decoders.
The checkpoint, dependencies and lab are read-only; all outputs go to --output.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import sys
import time

from h3_export import COMMIT, SNAPSHOT, digest, save
from h3_weights import CANONICAL_QKV_LAYOUT, convert_raw_transformer_key

ADAPTER_REVISION = "3ec17a324ced54151364f24f8b5fb6bf7e26414f"
ADAPTER_SHA256 = "e16ac20824d6e6649b193806f8fb095639bd9946c97b1bb84b4248eab1cc807f"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lab, out = args.lab.resolve(), args.output.resolve()
    if out.is_relative_to(lab) or out == args.base.resolve().parent:
        raise ValueError("output must be separate from the read-only lab/base bundle")
    out.mkdir(parents=True, exist_ok=True)
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(lab / "forge"), str(lab / "forge/vendor/python"), str(lab / "forge/vendor/diffusers/src")]
    import qualify_block as q
    from export_functional import parameter_input_graph
    import torch
    import coreai_torch as ct
    from safetensors import safe_open
    torch.set_num_threads(4)
    torch.manual_seed(1043)
    environment = json.loads((lab / "records/environment.json").read_text())
    assert environment["unmodifiedDependencies"] and environment["diffusers"]["commit"] == COMMIT
    assert q.SNAPSHOT.name == SNAPSHOT
    base = json.loads(args.base.read_text())
    assert base["complete"] and base["snapshot"] == SNAPSHOT and base["diffusers_commit"] == COMMIT
    assert base.get("recipe", "base50") == "base50" and not base.get("adapter")
    assert base.get("transformer_weight_layout") == CANONICAL_QKV_LAYOUT, "base must use the canonical raw-shard QKV contract"
    assert (base["width"], base["height"], base["frames"]) == (960, 544, 124)
    adapter_path = lab / "forge/vendor/weights/minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors"
    assert digest(adapter_path) == ADAPTER_SHA256
    adapter = dict(repository="lightx2v/Minimax-h3-Turbo", revision=ADAPTER_REVISION,
                   sha256=ADAPTER_SHA256, execution="dynamic_lora", rank=128, alpha=8.0)
    meta_path = out / "build.json"
    meta = dict(base, complete=False, recipe="turbo8", adapter=adapter,
                base_bundle_sha256=digest(args.base), recipe_sha256=digest(__file__))
    save(meta_path, meta)

    def fixture(name, tensor):
        tensor = tensor.detach().contiguous().cpu()
        path = out / "fixtures" / (name + ".raw")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = tensor.view(torch.uint8).numpy().tobytes()
        path.write_bytes(data)
        return dict(name=Path(name).name, path=str(path), dtype=str(tensor.dtype).removeprefix("torch."),
                    shape=list(tensor.shape), sha256=hashlib.sha256(data).hexdigest())

    def read(item):
        data = Path(item["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
        return torch.frombuffer(bytearray(data), dtype=getattr(torch, item["dtype"])).reshape(item["shape"])

    def export(name, module, example, inputs, outputs):
        path = out / "models" / (name + ".aimodel")
        path.parent.mkdir(parents=True, exist_ok=True)
        print("CPU export", name, flush=True)
        begin = time.monotonic()
        with torch.inference_mode():
            ep = torch.export.export(module.eval(), example).run_decompositions(ct.get_decomp_table())
        cv = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
        cv.add_exported_program(ep, input_names=inputs, output_names=outputs, state_names=[], entrypoint_name="main")
        program = cv.to_coreai()
        program.optimize()
        program.save_asset(path)
        checksum = digest(path / "main.mlirb")
        assert (path / "main.hash").read_bytes().hex() == checksum
        print("exported", name, round(time.monotonic() - begin, 2), flush=True)
        return dict(path=str(path), function="main", sha256=checksum)

    # Call the original scheduler, retaining its exact F32 sigma/timestep bits.
    arms, schedulers = [], []
    for shift in (12.0, 3.0):
        scheduler = q.MiniMaxH3Scheduler(shift=shift)
        scheduler.set_timesteps(9)
        assert len(scheduler.timesteps) == 8 and len(scheduler.sigmas) == 9
        arms.append(dict(shift=shift, timesteps=scheduler.timesteps.tolist(), sigmas=scheduler.sigmas.tolist()))
        schedulers.append(scheduler)
    schedule_path = out / "schedule.json"
    save(schedule_path, dict(gridPoints=9, expectedForwards=8, schedules=arms))
    meta["schedule"] = str(schedule_path)
    torch.manual_seed(8271)
    video, audio = torch.randn(32), torch.randn(32)
    sha = lambda t: hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    numerics = dict(snapshot=SNAPSHOT, diffusers_commit=COMMIT, adapter=adapter,
                    video=video.tolist(), audio=audio.tolist(), steps=[])
    for i in range(8):
        vv, av = torch.randn(32), torch.randn(32)
        video = schedulers[0].step(vv, schedulers[0].timesteps[i], video, return_dict=False)[0]
        audio = schedulers[1].step(av, schedulers[1].timesteps[i], audio, return_dict=False)[0]
        numerics["steps"].append(dict(video_velocity=vv.tolist(), audio_velocity=av.tolist(),
                                      video_sha256=sha(video), audio_sha256=sha(audio)))
    save(out / "schedule-oracle.json", numerics)

    with torch.device("meta"):
        model = q.MiniMaxH3Transformer3DModel()
    cfg, converter = dict(model.config), q.converter()
    index = json.loads((q.SNAPSHOT / "FL2VA/transformer/model.safetensors.index.json").read_text())["weight_map"]
    weights = {}
    for name, shard in sorted(index.items()):
        if name.startswith("blocks.") and not name.startswith("blocks.0."):
            continue
        with safe_open(q.SNAPSHOT / "FL2VA/transformer" / shard, framework="pt", device="cpu") as f:
            value = f.get_tensor(name)
        weights.update(convert_raw_transformer_key(converter, name, value, cfg))
    model.load_state_dict(weights, strict=False, assign=True)
    model.eval()
    model.rope = type(model.rope)(rope_freq_dim=cfg["rope_freq_dim"], rope_theta=cfg["rope_theta"])
    subset, alphas = {}, {}
    with safe_open(adapter_path, framework="pt", device="cpu") as source:
        assert float(source.metadata()["alpha"]) == 8.0
        targets = []
        for key in source.keys():
            if not key.endswith(".lora_A.default.weight"):
                continue
            target = key.removesuffix(".lora_A.default.weight")
            a = source.get_slice(key).get_shape()
            bkey = target + ".lora_B.default.weight"
            b = source.get_slice(bkey).get_shape()
            assert a[0] == b[1] == 128 and tuple(model.get_submodule(target).weight.shape) == (b[0], a[1])
            targets.append(target)
            if target.startswith(("token_refiner.", "transformer_blocks.0.")):
                subset["transformer." + key.replace(".default.weight", ".weight")] = source.get_tensor(key)
                subset["transformer." + bkey.replace(".default.weight", ".weight")] = source.get_tensor(bkey)
                alphas["transformer." + target + ".alpha"] = 8.0
        assert len(source.keys()) == 624 and len(targets) == 312
    # Preserve upstream PEFT casts, rank scaling and residual additions in FX.
    model.load_lora_adapter(subset, prefix="transformer", adapter_name="turbo", network_alphas=alphas)
    for target in targets:
        if target.startswith(("token_refiner.", "transformer_blocks.0.")):
            assert model.get_submodule(target).scaling["turbo"] == 8.0 / 128

    nt = base["text_rows"]
    nv = 37 * 17 * 30
    na = round(124 / 24 * 40) * 2
    rows = nt + nv + na
    class Prepare(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.video, self.audio = model.proj_in, model.audio_proj_in
            self.text, self.refine = model.context_embedder, model.token_refiner
        def forward(self, video, audio, text):
            text = self.refine(self.text(text))
            return torch.cat((text, self.audio(audio).to(text.dtype), self.video(video).to(text.dtype)), dim=1)
    original = base["oracle"]
    text, video, audio = (read(original[k]) for k in ("text", "video", "audio"))
    captured = {}
    class Stop(Exception):
        pass
    def capture(_module, values):
        captured["hidden"] = values[0].detach()
        raise Stop()
    handle = model.transformer_blocks[0].register_forward_pre_hook(capture)
    with torch.inference_mode():
        try:
            # Same source outer path, with real conditioning and the adapter active.
            from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep
            pos, tags, vi, ai, ti, _, _ = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
                torch.ones(nt, dtype=torch.int64), 37, 34, 60, na // 2, (1, 2, 2), 2, 2, 0)
            model(video, audio, text, torch.zeros(1), tags * 0, tags, pos, vi, ai, ti, return_dict=False)
        except Stop:
            pass
        hidden = captured["hidden"]
        assert torch.equal(hidden, Prepare()(video, audio, text))
    handle.remove()
    oracle = dict(original, hidden=fixture("oracle/hidden", hidden))
    meta["oracle"] = oracle
    meta["prepare"] = export("prepare-turbo8", Prepare(), (video, audio, text), ["video", "audio", "text"], ["hidden"])

    cos, sin = read(base["rotary_cos"]), read(base["rotary_sin"])
    temb = model.time_embedder(model.time_proj(torch.zeros(1)))
    block = model.transformer_blocks[0]
    with torch.inference_mode():
        table = torch.stack(block.adaln_proj(temb))
        small = hidden[:, :128], table, tags[:128], cos[:128], sin[:128]
        expected = block(small[0], temb, small[2], (small[3], small[4]))
    block.adaln_proj = q.CachedModulation()
    wrapped = q.BlockCall(block).eval()
    graph, parameters = parameter_input_graph(wrapped, small)
    with torch.inference_mode():
        actual = graph(*small, *parameters.values())
        assert torch.equal(actual, expected)
    assert len(parameters) == 22, list(parameters)
    probe = dict(inputs=[fixture("probe/" + n, t) for n, t in zip(("hidden", "temb", "tags", "cos", "sin"), small) if n != "tags"],
                 reference=fixture("probe/reference", expected), rewrite_byte_identical=True,
                 scope="128-row original CPU adapter block versus mechanically lifted graph; not full trajectory")
    # Native signatures use checked I32 tags; the numerical values are unchanged.
    probe["inputs"].insert(2, fixture("probe/tags", tags[:128].to(torch.int32)))
    bindings = ["hidden", "temb", "tags", "cos", "sin"] + [f"w{i}" for i in range(22)]
    probe["model"] = export("body-turbo8-probe", graph, (*small, *parameters.values()), bindings, ["hidden_out"])
    bodies = []
    for cardinality in (1, 2):
        inputs = (torch.zeros(1, rows, 5376, dtype=torch.bfloat16),
                  torch.zeros(6, 3 * cardinality, 5376, dtype=torch.bfloat16), tags, cos, sin)
        variant, params = parameter_input_graph(wrapped, inputs)
        assert list(params) == list(parameters)
        bodies.append(export(f"body-turbo8-{cardinality}", variant, (*inputs, *params.values()), bindings, ["hidden_out"]))
    meta["bodies"] = bodies
    production_cases = []
    with torch.inference_mode():
        for i, body in enumerate(bodies):
            values = torch.tensor(sorted(set((arms[0]["timesteps"][i], arms[1]["timesteps"][i]))))
            embedding = model.time_embedder(model.time_proj(values))
            # T2VA has text/video at video time, audio at audio time; no anchors.
            adaln = tags if i == 0 else tags + (tags == 2) * 3
            # block.adaln_proj was replaced above; use the unchanged source
            # modulation export's original checkpoint module for this probe.
            with torch.device("meta"):
                original_block = q.MiniMaxH3Transformer3DModel().transformer_blocks[0]
            prefix = "transformer_blocks.0.adaln_proj."
            original_block.adaln_proj.load_state_dict({k.removeprefix(prefix):v for k,v in weights.items() if k.startswith(prefix)}, strict=True, assign=True)
            packet = [dict(oracle["hidden"], name="hidden"), fixture(f"production-{i}/temb", torch.stack(original_block.adaln_proj(embedding)))]
            packet += [fixture(f"production-{i}/tags", adaln.to(torch.int32)), dict(base["rotary_cos"],name="cos"), dict(base["rotary_sin"],name="sin")]
            production_cases.append(dict(model=body,inputs=packet,scope="full-geometry single block; finite output only, no CPU trajectory oracle"))

    # Reuse all 500 base tensor files; save only the twelve low-rank tensors per block.
    # Parameter names/dtypes come directly from the exported upstream PEFT graph.
    blocks = []
    with safe_open(adapter_path, framework="pt", device="cpu") as source:
        for i in range(50):
            original_params = {p["parameter"]: p for p in base["parameters"][i]}
            packet = []
            for j, (name, example) in enumerate(parameters.items()):
                plain = name.replace(".base_layer.", ".")
                if ".lora_" in name:
                    target = name.removeprefix("block.").replace(".turbo.weight", ".default.weight")
                    value = source.get_tensor(f"transformer_blocks.{i}." + target).to(example.dtype)
                    assert tuple(value.shape) == tuple(example.shape)
                    if i == 0:
                        assert torch.equal(value, example)
                    item = fixture(f"adapter-{i}/w{j}", value)
                else:
                    item = dict(original_params[plain])
                    assert item["shape"] == list(example.shape) and item["dtype"] == str(example.dtype).removeprefix("torch.")
                    if i == 0:
                        assert sha(example.detach()) == item["sha256"]
                packet.append(dict(item, name=f"w{j}", parameter=name))
            blocks.append(packet)
            print("CPU parameter mapping", i, flush=True)
    meta["parameters"] = blocks
    probe["parameters"] = blocks[0]
    probe["production_cases"] = production_cases
    save(out / "probe.json", probe)
    save(meta_path, meta)
    meta["complete"] = True
    save(out / "bundle.json", meta)
    save(out / "cpu-checks.json", dict(passed=True, gpu_work=False, adapter=adapter,
         target_count=len(targets), block_count=len(blocks), bindings_per_block=22,
         source_schedule_evaluations=8, original_prepare_byte_identical=True,
         original_small_block_byte_identical=True, full_generation_checked=False,
         recipe_sha256=digest(__file__), bundle_sha256=digest(out / "bundle.json")))
    print("CPU preparation complete:", out / "bundle.json", flush=True)
    del model, weights, parameters, graph, wrapped
    gc.collect()


if __name__ == "__main__":
    main()

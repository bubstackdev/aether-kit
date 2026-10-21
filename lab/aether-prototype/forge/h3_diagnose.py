"""CPU-only phase oracles for the rejected Turbo8 production render.

Calls the pinned upstream modules. Prepares identical inputs for owned tensors,
automatic Metal outputs and caller-bound Metal outputs; performs no inference
through Core AI and never initializes a GPU device.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

from h3_export import COMMIT, SNAPSHOT, digest, save
from h3_weights import CANONICAL_QKV_LAYOUT, convert_raw_transformer_key


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lab", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--adapter-probe", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    lab, out = args.lab.resolve(), args.output.resolve()
    if out.is_relative_to(lab):
        raise ValueError("outputs must stay outside the read-only lab")
    out.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(lab / "forge"), str(lab / "forge/vendor/python"),
                    str(lab / "forge/vendor/diffusers/src")]
    import qualify_block as q
    import torch
    from safetensors import safe_open
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep, MiniMaxH3SetTimestepsStep,
    )
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    base = json.loads(args.bundle.read_text())
    pilot = json.loads(args.adapter_probe.read_text())
    assert base["snapshot"] == SNAPSHOT and base["diffusers_commit"] == COMMIT
    assert base["recipe"] == "turbo8"
    assert base.get("transformer_weight_layout") == CANONICAL_QKV_LAYOUT
    assert (base["width"], base["height"], base["frames"]) == (960, 544, 124)
    report = dict(schema=1, gpu_work=False, passed=False, source_recipe_sha256=digest(__file__),
                  bundle_sha256=digest(args.bundle), cases=[], cache_comparisons=[])
    save(out / "cpu-checks.json", report)

    def read(item):
        raw = Path(item["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
        return torch.frombuffer(bytearray(raw), dtype=getattr(torch, item["dtype"])).reshape(item["shape"])

    def fixture(name, value):
        value = value.detach().contiguous().cpu()
        path = out / "fixtures" / (name + ".raw")
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = value.view(torch.uint8).numpy().tobytes()
        path.write_bytes(raw)
        return dict(name=name.rsplit("/", 1)[-1], path=str(path), dtype=str(value.dtype).removeprefix("torch."),
                    shape=list(value.shape), sha256=hashlib.sha256(raw).hexdigest())

    def case(name, model, inputs, expected, row_indices=None):
        record = dict(name=name, model=model, inputs=inputs,
                      outputs=[fixture(name + "/" + key, value) for key, value in expected.items()])
        if row_indices is not None:
            for item in record["outputs"]:
                item.update(native_shape=list(hidden.shape), row_indices=row_indices.tolist())
        report["cases"].append(record)
        save(out / "probe.json", dict(schema=1, predefined_relative_l2_limit=0.02,
                                      scope="original CPU phase comparisons at production row geometry; no full-trajectory/quality claim",
                                      cases=report["cases"]))
        save(out / "cpu-checks.json", report)

    with torch.device("meta"):
        model = q.MiniMaxH3Transformer3DModel()
    cfg, converter = dict(model.config), q.converter()
    index = json.loads((q.SNAPSHOT / "FL2VA/transformer/model.safetensors.index.json").read_text())["weight_map"]

    def weights_for(predicate):
        weights = {}
        for name, shard in sorted(index.items()):
            if predicate(name):
                with safe_open(q.SNAPSHOT / "FL2VA/transformer" / shard, framework="pt", device="cpu") as f:
                    weights.update(convert_raw_transformer_key(converter, name, f.get_tensor(name), cfg))
        return weights

    print("CPU load original outer modules and block 0", flush=True)
    model.load_state_dict(weights_for(lambda n: not n.startswith("blocks.") or n.startswith("blocks.0.")),
                          strict=False, assign=True)
    model.eval()
    adapter_path = lab / "forge/vendor/weights/minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors"
    assert digest(adapter_path) == base["adapter"]["sha256"]
    adapter, alphas = {}, {}
    with safe_open(adapter_path, framework="pt", device="cpu") as source:
        for key in source.keys():
            if key.startswith(("token_refiner.", "transformer_blocks.0.")):
                adapter["transformer." + key.replace(".default.weight", ".weight")] = source.get_tensor(key)
                if ".lora_A." in key:
                    alphas["transformer." + key.split(".lora_A.")[0] + ".alpha"] = 8.0
    model.load_lora_adapter(adapter, prefix="transformer", adapter_name="turbo", network_alphas=alphas)
    original = base["oracle"]
    hidden, video, audio, text = (read(original[k]) for k in ("hidden", "video", "audio", "text"))
    expected = torch.cat((model.token_refiner(model.context_embedder(text)),
                          model.audio_proj_in(audio).to(text.dtype), model.proj_in(video).to(text.dtype)), dim=1)
    assert torch.equal(hidden, expected)
    case("prepare", base["prepare"], [dict(original[k], name=k) for k in ("video", "audio", "text")],
         dict(hidden=expected))
    del expected
    nt = base["text_rows"]
    _, tags, vi, ai, _, _, _ = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        torch.ones(nt, dtype=torch.int64), 37, 34, 60, audio.shape[1] // 2, (1, 2, 2), 2, 2, 0)
    schedule = json.loads(Path(base["schedule"]).read_text())
    plans = []
    for i in range(8):
        vt, at = (arm["timesteps"][i] for arm in schedule["schedules"])
        values, indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(vi, ai, 0, 0, nt, vt, at, 0.999, 1.0)
        embedding = model.time_embedder(model.time_proj(values))
        plans.append((values, indices, embedding))
        if i < 2:
            case(f"time-{i}", base["time"], [fixture(f"time-{i}/timestep", values)], dict(temb=embedding))
            head_hidden = model.norm_out(hidden, embedding, indices).float()
            case(f"heads-{i}", base["heads"],
                 [dict(original["hidden"], name="hidden"), fixture(f"heads-{i}/temb", embedding),
                  fixture(f"heads-{i}/indices", indices.to(torch.int32))],
                 dict(video=model.proj_out(head_hidden).index_select(1, vi),
                      audio=model.audio_proj_out(head_hidden).index_select(1, ai)))
            del head_hidden

    # Keep all keys/values but compute selected queries and row-wise tails.
    # This is a diagnostic CPU oracle, not an exported production rewrite.
    import diffusers.models.transformers.transformer_minimax_h3 as implementation
    def selected_reference(block, hidden, embedding, adaln, rotary, selected):
        dispatch = implementation.dispatch_attention_fn
        def attention(query, key, value, *positional, **keywords):
            result = dispatch(query.index_select(1, selected), key, value, *positional, **keywords)
            return torch.zeros_like(query).index_copy(1, selected, result)
        def before(_module, inputs):
            return (inputs[0].index_select(1, selected), *inputs[1:])
        def after(_module, _inputs, result):
            shape = list(result.shape)
            shape[1] = hidden.shape[1]
            return result.new_zeros(shape).index_copy(1, selected, result)
        hooks = []
        try:
            for module in (block.attn.to_out[0], block.ff):
                hooks += [module.register_forward_pre_hook(before), module.register_forward_hook(after)]
            implementation.dispatch_attention_fn = attention
            return block(hidden, embedding, adaln, rotary).index_select(1, selected)
        finally:
            implementation.dispatch_attention_fn = dispatch
            for hook in hooks:
                hook.remove()

    cos, sin = read(base["rotary_cos"]), read(base["rotary_sin"])
    small = pilot["inputs"]
    small_hidden, small_tags = read(small[0]), read(small[2]).long()
    small_rotary = read(small[3]), read(small[4])
    expected = read(pilot["reference"])
    assert torch.equal(model.transformer_blocks[0](small_hidden, plans[0][2], small_tags, small_rotary), expected)
    selected_small = torch.arange(0, 128, 4)
    actual = selected_reference(model.transformer_blocks[0], small_hidden, plans[0][2], small_tags,
                                small_rotary, selected_small)
    assert torch.equal(actual, expected.index_select(1, selected_small)), "selected-query method changed CPU output"
    report["selected_query_validation"] = dict(cpu_full_rows=128, selected_rows=32, byte_identical=True,
                                               scope="source full block versus selected-query/row-wise-tail hooks")
    selected = torch.tensor(sorted(set(torch.linspace(0, nt-1, 8).long().tolist()
        + ai[torch.linspace(0, len(ai)-1, 16).long()].tolist()
        + vi[torch.linspace(0, len(vi)-1, 40).long()].tolist())), dtype=torch.int64)
    for i in range(2):
        print(f"CPU block 0 with complete production context, {len(selected)} selected queries, cardinality {i + 1}", flush=True)
        started = time.monotonic()
        values, indices, embedding = plans[i]
        adaln = indices * 3 + tags
        actual = selected_reference(model.transformer_blocks[0], hidden, embedding, adaln, (cos, sin), selected)
        packet = pilot["production_cases"][i]
        assert torch.equal(read(packet["inputs"][0]), hidden)
        assert torch.equal(read(packet["inputs"][2]).to(torch.int64), adaln)
        case(f"body-{i}", base["bodies"][i],
             packet["inputs"] + base["parameters"][0], dict(hidden_out=actual), row_indices=selected)
        report.setdefault("body_cpu_seconds", []).append(time.monotonic() - started)
        del actual
        gc.collect()

    cache_record = next(args.cache.glob("*/cache.json"))
    cached = json.loads(cache_record.read_text())
    report["native_cache"] = dict(path=str(cache_record), key=cached["key"], record_sha256=digest(cache_record))
    assert len(cached["embeddings"]) == 8 and len(cached["tables"]) == 50

    def compare(name, actual, item):
        expected = read(dict(item, path=str(cache_record.parent / item["path"])))
        assert expected.shape == actual.shape and expected.dtype == actual.dtype
        difference = (actual.double() - expected.double()).norm()
        l2 = float(difference / actual.double().norm().clamp_min(1e-30))
        check = dict(name=name, relative_l2=l2, passed=bool(torch.isfinite(expected).all()) and l2 <= 0.02,
                     expected_sha256=fixture("cache-reference/" + name, actual)["sha256"],
                     native_cached_sha256=item["sha256"])
        report["cache_comparisons"].append(check)

    for i, (_, _, embedding) in enumerate(plans):
        compare(f"time-{i}", embedding, cached["embeddings"][i])
    for b in range(50):
        print("CPU modulation comparison", b, flush=True)
        block = model.transformer_blocks[b]
        prefix = f"transformer_blocks.{b}.adaln_proj."
        if b != 0:
            weights = weights_for(lambda n: n.startswith(f"blocks.{b}.adaln_proj."))
            block.adaln_proj.load_state_dict({k.removeprefix(prefix): v for k, v in weights.items()}, strict=True, assign=True)
        for i, (_, _, embedding) in enumerate(plans):
            compare(f"block-{b}-step-{i}", torch.stack(block.adaln_proj(embedding)), cached["tables"][b][i])
        if b != 0:
            block.adaln_proj.to_empty(device="meta")
            del weights
        gc.collect()
        save(out / "cpu-checks.json", report)
    report["passed"] = all(c["passed"] for c in report["cache_comparisons"])
    report["scope"] = "Original CPU outer phases, sampled-query block oracle retaining complete production context, and native-produced preparation cache comparison; no complete CPU trajectory or quality result"
    save(out / "cpu-checks.json", report)
    print("CPU diagnosis preparation complete", report["passed"], flush=True)


if __name__ == "__main__":
    main()

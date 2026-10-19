"""Convert block 0 from the FL2VA raw shards with the pinned Diffusers converter and compare the
SHA-256 of each base binding with AetherKit's corrected block-bundle fixture. CPU only, no model run.
Usage: python -I -B check_block0_weights.py SNAPSHOT FIXTURE_BUNDLE_JSON CONVERTER_DIR"""
import hashlib, importlib.util, json, sys
from pathlib import Path
import torch
from safetensors import safe_open
snapshot, fixture, converter_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
spec = importlib.util.spec_from_file_location("conv", converter_dir / "convert_minimax_h3_to_diffusers.py")
conv = importlib.util.module_from_spec(spec); spec.loader.exec_module(conv)
directory = snapshot / "FL2VA/transformer"
config = json.loads((directory / "config.json").read_text())
index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
converted = {}
for name in sorted(k for k in index if k.startswith("blocks.0.")):
    with safe_open(directory / index[name], framework="pt", device="cpu") as f: value = f.get_tensor(name)
    if name.endswith(".attn.qkv_proj.weight"):
        value = conv.reorder_interleaved_qkv(value, config["num_attention_heads"], config["attention_head_dim"])
    for key, tensor in conv.convert_transformer_key(name, value, config): converted[key] = tensor
expected = json.loads(fixture.read_text())["parameters"][0]
results = []
for item in expected:
    if ".lora_" in item["parameter"]: continue
    key = "transformer_blocks.0." + item["parameter"].removeprefix("block.").replace(".base_layer.", ".")
    value = converted[key].contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    ok = hashlib.sha256(raw).hexdigest() == item["sha256"] and list(value.shape) == item["shape"]
    results.append(dict(name=item["name"], parameter=key, dtype=str(value.dtype), ok=ok))
print(json.dumps(dict(all_equal=all(r["ok"] for r in results), checked=len(results), results=results), indent=1))

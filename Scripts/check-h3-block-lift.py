"""CPU self-check of the lifting machinery on real block-0 weights with a SYNTHETIC rank-128 adapter
(random A/B; the real Turbo adapter is not on disk). The lifted graph must equal the PEFT block exactly."""
import sys, json
from pathlib import Path
import torch
sys.path.insert(0, sys.argv[3])
import h3_export_lib as h
snapshot, converter_dir = Path(sys.argv[1]), Path(sys.argv[2])
torch.manual_seed(1043)
checkpoint = h.Checkpoint(snapshot, h.load_converter(converter_dir))
model = h.meta_model()
weights = checkpoint.converted(lambda n: n.startswith("blocks.0.") or n.startswith("time_") or n.startswith("timestep"))
missing = model.load_state_dict(weights, strict=False, assign=True)
block = model.transformer_blocks[0]
targets = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2"]
adapter, alphas = {}, {}
for t in targets:
    linear = block.get_submodule(t)
    adapter[f"transformer.transformer_blocks.0.{t}.lora_A.weight"] = (torch.randn(128, linear.in_features) * 0.01).to(torch.bfloat16)
    adapter[f"transformer.transformer_blocks.0.{t}.lora_B.weight"] = (torch.randn(linear.out_features, 128) * 0.01).to(torch.bfloat16)
    alphas[f"transformer.transformer_blocks.0.{t}.alpha"] = 8.0
model.load_lora_adapter(adapter, prefix="transformer", adapter_name="turbo", network_alphas=alphas)
rows = 128
hidden = torch.randn(1, rows, 5376, dtype=torch.bfloat16)
temb = torch.randn(1, 2688, dtype=torch.float32)
tags = torch.randint(0, 3, (rows,))
cos, sin = torch.randn(rows, 96), torch.randn(rows, 96)
with torch.inference_mode():
    table = torch.stack(block.adaln_proj(temb))
    expected = block(hidden, temb, tags, (cos, sin))
block.adaln_proj = h.CachedModulation()
wrapped = h.BlockCall(block).eval()
graph, params = h.lift(wrapped)
with torch.inference_mode():
    actual = graph(hidden, table, tags, cos, sin, *params.values())
names = list(params)
print(json.dumps(dict(byte_identical=bool(torch.equal(actual, expected)), bindings=len(names), names=names), indent=1))
if len(sys.argv) > 4:
    import time
    started = time.monotonic()
    record = h.export_asset(Path(sys.argv[4]) / "body-probe.aimodel", graph, (hidden, table, tags.to(torch.int32), cos, sin, *params.values()),
                            ["hidden", "temb", "tags", "cos", "sin"] + [f"w{i}" for i in range(22)], ["hidden_out"])
    print(json.dumps(dict(record, seconds=round(time.monotonic() - started, 1))))

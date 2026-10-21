"""CPU-only independent audit of every exported H3 body/adapter binding."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

from h3_export import digest, save
from h3_weights import CANONICAL_QKV_LAYOUT, convert_raw_transformer_key


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lab", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    lab = args.lab.resolve()
    if args.output.resolve().is_relative_to(lab):
        raise ValueError("output must stay outside the read-only lab")
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(lab / "forge"), str(lab / "forge/vendor/python"), str(lab / "forge/vendor/diffusers/src")]
    import qualify_block as q
    import torch
    from safetensors import safe_open
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    bundle = json.loads(args.bundle.read_text())
    assert bundle.get("transformer_weight_layout") == CANONICAL_QKV_LAYOUT
    with torch.device("meta"):
        model = q.MiniMaxH3Transformer3DModel()
    cfg, converter = dict(model.config), q.converter()
    config = json.loads((q.SNAPSHOT / "transformer/config.json").read_text())
    normalized = json.loads(json.dumps(cfg))
    differences = {k: [v, normalized.get(k)] for k, v in config.items()
                   if not k.startswith("_") and v != normalized.get(k)}
    assert not differences, differences
    index = json.loads((q.SNAPSHOT / "FL2VA/transformer/model.safetensors.index.json").read_text())["weight_map"]
    adapter = lab / "forge/vendor/weights/minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors"
    assert digest(adapter) == bundle["adapter"]["sha256"]
    report = dict(schema=1, passed=False, gpu_work=False, bundle_sha256=digest(args.bundle),
                  source_recipe_sha256=digest(__file__), config_differences=differences, blocks=[])
    with safe_open(adapter, framework="pt", device="cpu") as lora:
        for b, bindings in enumerate(bundle["parameters"]):
            weights = {}
            for name, shard in sorted(index.items()):
                if name.startswith(f"blocks.{b}.") and not name.startswith(f"blocks.{b}.adaln_proj."):
                    with safe_open(q.SNAPSHOT / "FL2VA/transformer" / shard, framework="pt", device="cpu") as source:
                        weights.update(convert_raw_transformer_key(converter, name, source.get_tensor(name), cfg))
            checks = []
            for item in bindings:
                name = item["parameter"].removeprefix("block.")
                if ".lora_" in name:
                    value = lora.get_tensor(f"transformer_blocks.{b}." + name.replace(".turbo.weight", ".default.weight"))
                else:
                    value = weights[f"transformer_blocks.{b}." + name.replace(".base_layer.", ".")]
                raw = value.contiguous().view(torch.uint8).numpy().tobytes()
                checksum = hashlib.sha256(raw).hexdigest()
                passed = (checksum == item["sha256"] and list(value.shape) == item["shape"]
                          and str(value.dtype).removeprefix("torch.") == item["dtype"])
                checks.append(dict(name=item["name"], parameter=item["parameter"], passed=passed,
                                   source_sha256=checksum, exported_sha256=item["sha256"]))
            report["blocks"].append(dict(block=b, passed=all(c["passed"] for c in checks), checks=checks))
            save(args.output, report)
            print("CPU weight audit", b, report["blocks"][-1]["passed"], flush=True)
            del weights, value, raw
            gc.collect()
    report["passed"] = len(report["blocks"]) == 50 and all(b["passed"] for b in report["blocks"])
    report["scope"] = "All 500 base and 600 low-rank body tensor bindings compared against independently reread checkpoint tensors and the pinned original converter; no inference/trajectory claim"
    save(args.output, report)


if __name__ == "__main__":
    main()

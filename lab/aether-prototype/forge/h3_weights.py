"""Respect the pinned converter's raw-shard versus state-dict QKV contract."""
import hashlib
import json
from pathlib import Path

CANONICAL_QKV_LAYOUT = "raw_per_head_qkv_to_contiguous_qkv_v1"


def convert_raw_transformer_key(converter, name, tensor, config):
    # convert_transformer_key expects the reference model's *loaded* state
    # dictionary. The upstream shard streamer performs this normalization first.
    # Keep both operations in the original converter; do not transcribe them.
    if name.endswith(".attn.qkv_proj.weight"):
        tensor = converter.reorder_interleaved_qkv(
            tensor, config["num_attention_heads"], config["attention_head_dim"])
    return converter.convert_transformer_key(name, tensor, config)


def canonical_body_parameters(blocks, snapshot, converter, config, output):
    """Replace the three raw-shard QKV bindings per block, retaining other files."""
    import torch
    from safetensors import safe_open
    directory = Path(snapshot) / "FL2VA/transformer"
    index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    result = []
    for b, original in enumerate(blocks):
        source_name = f"blocks.{b}.attn.qkv_proj.weight"
        with safe_open(directory / index[source_name], framework="pt", device="cpu") as source:
            canonical = dict(convert_raw_transformer_key(converter, source_name, source.get_tensor(source_name), config))
        packet = []
        for item in original:
            name = item["parameter"].removeprefix("block.").replace(".base_layer.", ".")
            if name in {"attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"}:
                value = canonical[f"transformer_blocks.{b}." + name].contiguous()
                assert list(value.shape) == item["shape"] and str(value.dtype).removeprefix("torch.") == item["dtype"]
                path = Path(output) / "canonical-qkv" / f"block-{b}" / (item["name"] + ".raw")
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = value.view(torch.uint8).numpy().tobytes()
                path.write_bytes(raw)
                item = dict(item, path=str(path), sha256=hashlib.sha256(raw).hexdigest())
            packet.append(item)
        result.append(packet)
        print("canonical raw QKV bindings", b, flush=True)
    return result

"""Small CPU fixtures for Kit's FL2VA presentation (`H3Presentation`, `H3VisionInputs`) from upstream code alone.

No model weights are loaded: the processor, tokenizer, the vision tower's table helpers and the text rotary module need
none. The presentation cases come from `prepare-h3-profile-oracle.py --part presentation` runs (upstream's FL2VA text
encoder step, verbatim): pass their output directories with --presentation, and the small one is embedded whole, larger
ones as SHA-256 of their int32 bytes.

  vision   For synthetic RGB8 keyframes (value at byte i = (7 i + 3) mod 256), the bf16 pixels the step feeds the tower
           (the torchvision processor's float32 output, rounded to bf16) and the position-table taps, weights and patch
           (row, column) `get_vision_interpolation_indices_and_weights` / `get_vision_position_ids` derive, as SHA-256.
  rotary   The text rotary module's cos [N, 128] bf16 for positions 0..N-1 (the profile's 1-D table layout), embedded,
           and its cos at the embedded presentation's M-RoPE positions, as SHA-256: Kit gathers the second from the first.

Usage (Forge's H3 oracle environment):
       Forge/h3-oracle/.venv/bin/python -I -B Scripts/prepare-h3-presentation-oracle.py --presentation DIR [DIR ...] --output FILE
"""
import argparse, base64, hashlib, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import h3_export_lib as h

SNAPSHOT_ROOT = Path.home() / ".cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots" / h.SNAPSHOT


def digest(tensor) -> str:
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def raw(directory: Path, name: str):
    manifest = json.loads((directory / "manifest.json").read_text())
    entry = manifest["tensors"][name]
    data = (directory / entry["file"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["sha256"], name
    return torch.frombuffer(bytearray(data), dtype=getattr(torch, entry["dtype"])).view(entry["shape"]).clone(), manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--presentation", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    from PIL import Image
    from transformers import AutoTokenizer, Qwen3VLConfig, Qwen3VLProcessor
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids
    import transformers

    config = Qwen3VLConfig.from_pretrained(SNAPSHOT_ROOT / "FL2VA/text_encoder")
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT_ROOT / "FL2VA/tokenizer")
    processor = Qwen3VLProcessor.from_pretrained(SNAPSHOT_ROOT / "FL2VA/processor")
    vc = config.vision_config
    side, merge = int(vc.num_position_embeddings ** 0.5), vc.spatial_merge_size

    presentations, embedded = [], None
    for directory in args.presentation:
        ids, manifest = raw(directory, "token_ids")
        tags, _ = raw(directory, "text_token_tags")
        positions, _ = raw(directory, "position_ids")
        grids, _ = raw(directory, "image_grid_thw")
        fields = manifest["fields"]
        prompt = ids[sum(int(g.prod()) // merge ** 2 + 2 for g in grids) + 6 * len(grids):]
        case = dict(canvas=fields["canvas"], prompt_sha256=fields["prompt_sha256"], keyframes=len(grids),
                    anchors=manifest["report"]["keyframe_anchors"],
                    grids=grids[:, 1:].tolist(), count=int(ids.numel()),
                    ids_sha256=digest(ids), tags_sha256=digest(tags), positions_sha256=digest(positions),
                    prompt_ids_sha256=digest(prompt))
        if embedded is None or ids.numel() < len(embedded["ids"]):
            embedded = dict(canvas=fields["canvas"], prompt=prompt.tolist(), ids=ids.tolist(), tags=tags.tolist(), positions=positions.tolist())
        presentations.append(case)

    labels = {f"<Picture {i}>: ": tokenizer(f"<Picture {i}>: ", add_special_tokens=False)["input_ids"] for i in (1, 2)}

    vision = []
    for width, height in ((320, 224), (288, 288)):
        rgb = ((np.arange(height * width * 3) * 7 + 3) % 256).astype(np.uint8).reshape(height, width, 3)
        out = processor.image_processor(images=[Image.fromarray(rgb)], return_tensors="pt")
        grid = out["image_grid_thw"]
        index, weight = get_vision_interpolation_indices_and_weights(grid, num_grid_per_side=side, mode="bilinear",
                                                                   align_corners=True, spatial_merge_size=merge)
        rc = get_vision_position_ids(grid, merge)
        vision.append(dict(width=width, height=height, grid=grid[0, 1:].tolist(),
                           pixels_sha256=digest(out["pixel_values"].to(torch.bfloat16)),
                           pos_index_sha256=digest(index.to(torch.int32)), pos_weight_sha256=digest(weight.to(torch.float32)),
                           patch_rc_sha256=digest(rc.to(torch.int32)), pos_weight_dtype=str(weight.dtype)))

    rotary = Qwen3VLTextRotaryEmbedding(config.text_config)
    positions = torch.tensor(embedded["positions"])
    rows = int(positions.max()) + 1
    probe = torch.zeros(1, 1, config.text_config.hidden_size, dtype=torch.bfloat16)
    with torch.inference_mode():
        table, _ = rotary(probe, torch.arange(rows).view(1, 1, -1).expand(3, 1, -1))
        gathered, _ = rotary(probe, positions[:, None, :])
    oracle = dict(schema=1, generator="Scripts/prepare-h3-presentation-oracle.py", transformers=transformers.__version__,
                  snapshot=h.SNAPSHOT, diffusers_commit=h.COMMIT,
                  presentation_source="prepare-h3-profile-oracle.py --part presentation (MiniMaxH3FL2VATextEncoderStep, verbatim)",
                  labels=labels, presentations=presentations, embedded=embedded, vision=vision,
                  vision_pixels="synthetic RGB8: byte i = (7 i + 3) mod 256",
                  rotary=dict(rows=rows, mrope_section=list(config.text_config.rope_parameters["mrope_section"]),
                              cos_table_base64=base64.b64encode(table[0].contiguous().view(torch.uint8).numpy().tobytes()).decode(),
                              cos_sha256=digest(gathered[0])))
    args.output.write_text(json.dumps(oracle, indent=1) + "\n")


if __name__ == "__main__":
    main()

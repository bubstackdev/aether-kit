"""The keyframe vision tower's fp32 reference for an FL2VA presentation oracle run (CPU only).

Runs upstream's `Qwen3VLVisionModel` on the oracle run's own pixels and grid twice: in bf16, as the pipeline does
(asserted byte-identical to the oracle's merger rows), and with the same bf16 weights upcast to fp32 compute, the
reference the tower is gated against (user decision 2026-10-29: relative L2 <= 0.03). Writes both, plus the
bf16-vs-fp32 spread as a diagnostic, in the oracle's raw-tensor format.

Usage (Forge's H3 oracle environment):
       Forge/h3-oracle/.venv/bin/python -I -B Scripts/prepare-h3-vision-reference.py --oracle ORACLE_DIR --output NEW_DIR
"""
import argparse, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import h3_export_lib as h

SNAPSHOT_ROOT = Path.home() / ".cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots" / h.SNAPSHOT


def raw(directory: Path, name: str):
    entry = json.loads((directory / "manifest.json").read_text())["tensors"][name]
    data = (directory / entry["file"]).read_bytes()
    assert h.sha256_bytes(data) == entry["sha256"], f"{name} changed on disk"
    return torch.frombuffer(bytearray(data), dtype=getattr(torch, entry["dtype"])).view(entry["shape"]).clone()


def distance(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return dict(relative_l2=float((a - b).norm() / b.norm()), cosine=float(a @ b / (a.norm() * b.norm())), max_abs=float((a - b).abs().max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from safetensors import safe_open
    from transformers import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel, Qwen3VLVisionRotaryEmbedding
    args.output.mkdir(parents=True, exist_ok=False)
    directory = SNAPSHOT_ROOT / "FL2VA/text_encoder"
    config = Qwen3VLConfig.from_pretrained(directory)
    vc = config.vision_config
    vc._attn_implementation = "sdpa"
    spans = h.Spans.sharded(directory)
    state = {}
    for name in sorted(k for k in spans.entries if k.startswith("model.visual.")):
        with safe_open(spans.span(name)[0], framework="pt", device="cpu") as f:
            state[name.removeprefix("model.visual.")] = f.get_tensor(name)
    with torch.device("meta"):
        visual = Qwen3VLVisionModel(vc)
    visual.load_state_dict(state, strict=True, assign=True)
    visual.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(vc)
    visual.eval()
    pixels, grid = raw(args.oracle, "pixel_values"), raw(args.oracle, "image_grid_thw").long()

    def run(model, values):
        with torch.inference_mode():
            out = model(values, grid_thw=grid, return_dict=True)
        return out.pooler_output, torch.stack(out.deepstack_features)
    bf16 = run(visual, pixels)
    assert torch.equal(bf16[0], raw(args.oracle, "image_embeds")) and torch.equal(bf16[1], raw(args.oracle, "deepstack")), \
        "the bf16 tower differs from the oracle's"
    fp32 = run(visual.float(), pixels.float())
    manifest = dict(tensors={}, fields=dict(oracle=str(args.oracle.resolve()), oracle_manifest_sha256=h.sha256_file(args.oracle / "manifest.json"),
                                            script_sha256=h.sha256_file(Path(__file__)), snapshot=h.SNAPSHOT, torch=torch.__version__,
                                            reference="bf16 weights upcast, fp32 compute"), report={})
    for name, tensor in (("image_embeds", fp32[0]), ("deepstack", fp32[1])):
        entry = h.raw_fixture(args.output / f"{name}.raw", name, tensor.contiguous())
        manifest["tensors"][name] = dict(file=f"{name}.raw", dtype=entry["dtype"], shape=entry["shape"], sha256=entry["sha256"])
    names = ["image_embeds"] + [f"deepstack[{k}]" for k in range(fp32[1].shape[0])]
    manifest["report"]["bf16_vs_fp32"] = {n: distance(a, b) for n, a, b in zip(names, [bf16[0], *bf16[1]], [fp32[0], *fp32[1]])}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(json.dumps(manifest["report"], indent=1))


if __name__ == "__main__":
    main()

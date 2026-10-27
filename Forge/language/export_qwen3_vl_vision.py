#!/usr/bin/env python3
"""Export the Qwen3-VL vision tower as an Aether Kit image-encoder asset (development tooling only).

    Forge/.venv/bin/python -I Forge/language/export_qwen3_vl_vision.py --checkpoint <hf snapshot> --output <new dir> \
        --source-repository Qwen/Qwen3-VL-2B-Instruct --source-revision <sha>

Writes only under the new `--output` directory: `vision.aimodel/` (one static-shape function `encode_<h>x<w>` per
fixed patch grid, over shared fp16 weights with float32 compute) and `export-report.json` with the grids, versions,
hashes and a cpu_only self-check of every function against the eager tower. No dimension is dynamic: a dynamic-shape
export intermittently computed wrong outputs on the GPU (see docs/qwen3-vl.md). GPU behaviour is qualified natively
in Swift (specialized without expectFrequentReshapes).
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen3_vl_vision import IMAGE_GRIDS, Qwen3VLVision, function_name, host_inputs, load_weights  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def feed(grid: tuple[int, int], patch_values: int, seed: int) -> dict[str, np.ndarray]:
    """Pixels drawn in the normalized range [-1, 1] with the host tables for the grid."""
    rng = np.random.default_rng(seed)
    h, w = grid
    tables = host_inputs(1, h, w)
    return {"pixels": rng.uniform(-1, 1, (h * w, patch_values)).astype(np.float16), **tables}


async def asset_outputs(asset: Path, grids: list[tuple[int, int]], feeds: list[dict[str, np.ndarray]]):
    import coreai.runtime as rt

    model = await rt.AIModel.load(asset, rt.SpecializationOptions.cpu_only())
    out = []
    for g, f in zip(grids, feeds):
        fn = model.load_function(function_name(g))
        r = await fn({k: rt.NDArray(v) for k, v in f.items()}, {})
        out.append((r["image_embeds"].numpy().astype(np.float32), r["deepstack"].numpy().astype(np.float32)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--grids", default=",".join(f"{h}x{w}" for h, w in IMAGE_GRIDS),
                        help="fixed patch grids HxW, comma-separated (default: IMAGE_GRIDS)")
    args = parser.parse_args()
    grids = [tuple(int(v) for v in g.split("x")) for g in args.grids.split(",")]
    merge = 2
    if len(set(grids)) != len(grids) or any(h % merge or w % merge for h, w in grids):
        raise SystemExit("grids must be distinct multiples of the merge size")
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    import coreai_torch as ct

    config = json.loads((args.checkpoint / "config.json").read_text())
    module = Qwen3VLVision(dict(config["vision_config"]))
    load_weights(module, args.checkpoint)
    module = module.to(torch.float16).eval()
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    for g in grids:
        with torch.no_grad():
            ep = torch.export.export(module, module.example(g)).run_decompositions(ct.get_decomp_table())
        converter.add_exported_program(ep, input_names=["pixels", "pos_index", "pos_weight", "patch_rc"],
                                       output_names=["image_embeds", "deepstack"], state_names=[],
                                       entrypoint_name=function_name(g))
    program = converter.to_coreai()
    program.optimize()
    args.output.mkdir(parents=True)
    asset = args.output / "vision.aimodel"
    program.save_asset(asset)

    feeds = [feed(g, module.patch_values, i) for i, g in enumerate(grids)]
    got = asyncio.run(asset_outputs(asset, grids, feeds))
    cases = []
    with torch.no_grad():
        for g, f, (emb, ds) in zip(grids, feeds, got):
            e_emb, e_ds = module(*(torch.from_numpy(f[k]) for k in ("pixels", "pos_index", "pos_weight", "patch_rc")))
            e_emb, e_ds = e_emb.float().numpy(), e_ds.float().numpy()
            cases.append({"function": function_name(g), "grid": list(g), "patches": g[0] * g[1],
                          "imageEmbedsShape": list(emb.shape),
                          "imageEmbedsCosine": round(cosine(emb, e_emb), 7),
                          "deepstackCosine": [round(cosine(ds[i], e_ds[i]), 7) for i in range(ds.shape[0])]})
    worst = min(min([c["imageEmbedsCosine"], *c["deepstackCosine"]]) for c in cases)
    report = {
        "schema": 1,
        "recipe": "qwen3-vl-vision-fp16-storage-fp32-compute-static-grids-v2",
        "imageGrids": [{"height": h, "width": w, "function": function_name((h, w)), "tokens": h * w // merge ** 2}
                       for h, w in grids],
        "exportedAt": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hostOS": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "coreai-torch", "coreai-core")},
        "scriptSHA256": {p.name: sha256(p) for p in sorted(Path(__file__).resolve().parent.glob("*.py"))},
        "source": {"repository": args.source_repository, "revision": args.source_revision, "license": "apache-2.0",
                   "checkpoint": str(args.checkpoint)},
        "visionConfig": config["vision_config"],
        "selfCheck": {"passed": worst >= 0.999, "worstCosine": worst, "bar": 0.999, "cases": cases,
                      "reference": "eager fp16-storage fp32-compute tower on random normalized pixels"},
        "modelFiles": {str(p.relative_to(asset)): sha256(p) for p in sorted(asset.rglob("*")) if p.is_file()},
    }
    (args.output / "export-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["selfCheck"], indent=1))
    if not report["selfCheck"]["passed"]:
        raise SystemExit("self-check failed")


if __name__ == "__main__":
    main()

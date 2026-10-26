#!/usr/bin/env python3
"""Export the Qwen3-VL vision tower as an Aether Kit image-encoder asset (development tooling only).

    <venv>/bin/python -I Forge/language/export_qwen3_vl_vision.py --checkpoint <hf snapshot> --output <new dir> \
        --source-repository Qwen/Qwen3-VL-2B-Instruct --source-revision <sha>

Writes only under the new `--output` directory: `vision.aimodel/` (function `encode`, fp16 weights with float32
compute) and `export-report.json` with versions, hashes and a cpu_only self-check against the eager tower on three
grids. GPU behaviour is qualified natively in Swift (specialized without expectFrequentReshapes).
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
from qwen3_vl_vision import Qwen3VLVision, host_inputs, load_weights  # noqa: E402

GRIDS = ((14, 14), (16, 20), (26, 16))          # patch grids (h, w): 196, 320 and 416 patches


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


async def asset_outputs(asset: Path, feeds: list[dict[str, np.ndarray]]):
    import coreai.runtime as rt

    model = await rt.AIModel.load(asset, rt.SpecializationOptions.cpu_only())
    fn = model.load_function("encode")
    out = []
    for f in feeds:
        r = await fn({k: rt.NDArray(v) for k, v in f.items()}, {})
        out.append((r["image_embeds"].numpy().astype(np.float32), r["deepstack"].numpy().astype(np.float32)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    import coreai_torch as ct

    config = json.loads((args.checkpoint / "config.json").read_text())
    module = Qwen3VLVision(dict(config["vision_config"]))
    load_weights(module, args.checkpoint)
    module = module.to(torch.float16).eval()
    example, dynamic = module.example()
    with torch.no_grad():
        ep = torch.export.export(module, example, dynamic_shapes=dynamic)
        ep = ep.run_decompositions(ct.get_decomp_table())
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    converter.add_exported_program(ep, input_names=["pixels", "pos_index", "pos_weight", "patch_rc"],
                                   output_names=["image_embeds", "deepstack"], state_names=[], entrypoint_name="encode")
    program = converter.to_coreai()
    program.optimize()
    args.output.mkdir(parents=True)
    asset = args.output / "vision.aimodel"
    program.save_asset(asset)

    feeds = [feed(g, module.patch_values, i) for i, g in enumerate(GRIDS)]
    got = asyncio.run(asset_outputs(asset, feeds))
    cases = []
    with torch.no_grad():
        for g, f, (emb, ds) in zip(GRIDS, feeds, got):
            e_emb, e_ds = module(*(torch.from_numpy(f[k]) for k in ("pixels", "pos_index", "pos_weight", "patch_rc")))
            e_emb, e_ds = e_emb.float().numpy(), e_ds.float().numpy()
            cases.append({"grid": list(g), "patches": g[0] * g[1], "imageEmbedsShape": list(emb.shape),
                          "imageEmbedsCosine": round(cosine(emb, e_emb), 7),
                          "deepstackCosine": [round(cosine(ds[i], e_ds[i]), 7) for i in range(ds.shape[0])]})
    worst = min(min([c["imageEmbedsCosine"], *c["deepstackCosine"]]) for c in cases)
    report = {
        "schema": 1,
        "recipe": "qwen3-vl-vision-fp16-storage-fp32-compute-v1",
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

"""Eager fp32 CPU references for preprocessed images: scene codes, the 256³ density grid through upstream's
renderer.query_triplane (grid_sample), and the marching-cubes mesh at threshold 25.

usage: reference_images.py <tsr-source-dir> <inputs-dir> <out-dir> [--res 256]
<inputs-dir> holds preprocess.py's <stem>.npy files. For each stem it writes <stem>-ref-codes.npy, <stem>-ref-grid.npy
(float32) and <stem>-ref-mesh.npz (vertices, faces), plus references.json with timings.
"""
import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("inputs")
ap.add_argument("out")
ap.add_argument("--res", type=int, default=256)
ap.add_argument("--threshold", type=float, default=25.0)
a = ap.parse_args()

sys.modules["torchmcubes"] = types.SimpleNamespace(marching_cubes=None)
sys.modules.setdefault("rembg", types.ModuleType("rembg"))
sys.path.insert(0, a.src)
sys.path.insert(0, str(Path(__file__).parent))
from tsr.system import TSR  # noqa: E402

from tsr_graphs import Scene  # noqa: E402

RADIUS = 0.87
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
snap = next((hub / "models--stabilityai--TripoSR/snapshots").iterdir())
model = TSR.from_pretrained(str(snap), config_name="config.yaml", weight_name="model.ckpt").eval()
scene = Scene(model)
lin = torch.linspace(0, 1, a.res) * 2 * RADIUS - RADIUS
gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
grid_points = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], -1)

import mcubes  # noqa: E402

report = {"snapshot": snap.name, "res": a.res, "threshold": a.threshold, "images": {}}
for path in sorted(Path(a.inputs).glob("*.npy")):
    stem = path.stem
    image = torch.from_numpy(np.load(path))
    with torch.no_grad():
        t0 = time.time()
        codes = scene(image)
        scene_s = time.time() - t0
        density = torch.empty(len(grid_points))
        t0 = time.time()
        for s in range(0, len(grid_points), 262144):
            chunk = grid_points[s:s + 262144]
            density[s:s + 262144] = model.renderer.query_triplane(model.decoder, chunk, codes)["density_act"][:, 0]
        grid_s = time.time() - t0
    np.save(out / f"{stem}-ref-codes.npy", codes.numpy())
    dens = density.numpy()
    np.save(out / f"{stem}-ref-grid.npy", dens)
    v, f = mcubes.marching_cubes((dens - a.threshold).reshape(a.res, a.res, a.res), 0.0)
    v = v[:, [2, 1, 0]] / (a.res - 1.0) * 2 * RADIUS - RADIUS      # upstream's axis swap and scaling
    f = f[:, [2, 1, 0]]
    np.savez(out / f"{stem}-ref-mesh.npz", vertices=v.astype(np.float32), faces=f.astype(np.int32))
    report["images"][stem] = {"sceneSeconds": scene_s, "gridSeconds": grid_s, "vertices": int(len(v)),
                              "faces": int(len(f)), "occupied": int((dens > a.threshold).sum())}
    print(stem, json.dumps(report["images"][stem]), flush=True)
(out / "references.json").write_text(json.dumps(report, indent=1))

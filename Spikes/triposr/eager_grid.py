"""Eager fp32 (CPU) density grid for the reference scene codes, via upstream renderer.query_triplane (grid_sample).

usage: eager_grid.py <tsr-source-dir> <export-dir> <results-dir> [res]
"""
import sys
import os
import time
import types
from pathlib import Path

import numpy as np
import torch

sys.modules["torchmcubes"] = types.SimpleNamespace(marching_cubes=None)
sys.modules.setdefault("rembg", types.ModuleType("rembg"))
sys.path.insert(0, sys.argv[1])
from tsr.system import TSR  # noqa: E402

E, R = Path(sys.argv[2]), Path(sys.argv[3])
res = int(sys.argv[4]) if len(sys.argv) > 4 else 256
R.mkdir(parents=True, exist_ok=True)
hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
snap = next((hub / "models--stabilityai--TripoSR/snapshots").iterdir())
model = TSR.from_pretrained(str(snap), config_name="config.yaml", weight_name="model.ckpt").eval()
codes = torch.from_numpy(np.load(E / "ref-scene-codes.npy"))
lin = torch.linspace(0, 1, res) * 2 * 0.87 - 0.87
gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
grid = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], -1)
out = torch.empty(len(grid))
t = time.time()
with torch.no_grad():
    for s in range(0, len(grid), 262144):
        out[s:s + 262144] = model.renderer.query_triplane(model.decoder, grid[s:s + 262144], codes)["density_act"][:, 0]
print("eager grid %d^3 on CPU: %.1fs" % (res, time.time() - t))
np.save(R / f"density-grid-{res}-eager.npy", out.numpy())

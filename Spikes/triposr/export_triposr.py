"""Throwaway spike: export TripoSR (MIT, stabilityai/TripoSR) to Core AI as two functions-in-separate-assets.

  scene.aimodel  `main`: image [1, 3, 512, 512] (RGB in [0, 1], background already composited) → scene codes
                 [3, 40, 64, 64]  (DINO ViT-B/16 → 16-layer triplane transformer → ConvTranspose upsample)
  field.aimodel  `main`: scene codes [3, 40, 64, 64] + points [N, 3] in (-r, r) → density [N, 1], color [N, 3]
                 (triplane bilinear sampling as an explicit gather: Core AI has no grid_sampler_2d lowering)
  grid.aimodel   `main` (0.0.3, --grid-slab S): scene codes + grid coordinates xs [R] (x and y) and zs [S] → density
                 [R, R, S] for one z-slab of the regular extraction grid, with no gather: each plane is sampled by
                 hat-function interpolation matrices, so its features for every grid point are two small matmuls

The upstream code (TripoSR-main, MIT) is imported read-only from the downloaded tarball. torchmcubes is not
needed for export and is stubbed. Also writes eager fp32 reference outputs for a fixed synthetic image.

usage: export_triposr.py <tsr-source-dir> <out-dir> [--dtype float16|float32] [--points N] [--grid-slab S --res R]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import coreai_torch as ct

sys.path.insert(0, str(Path(__file__).parent))
from tsr_graphs import Field, GridField, Scene  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--dtype", default="float16")
ap.add_argument("--points", type=int, default=65536)
ap.add_argument("--only", default="scene,field")
ap.add_argument("--grid-slab", type=int, default=0, help="also export grid.aimodel for z-slabs of S planes")
ap.add_argument("--res", type=int, default=256, help="grid resolution R for grid.aimodel")
args = ap.parse_args()

sys.modules["torchmcubes"] = types.SimpleNamespace(marching_cubes=None)   # export never runs marching cubes
sys.modules.setdefault("rembg", types.ModuleType("rembg"))
sys.path.insert(0, args.src)
from tsr.system import TSR  # noqa: E402

hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
snap = next((hub / "models--stabilityai--TripoSR/snapshots").iterdir())
model = TSR.from_pretrained(str(snap), config_name="config.yaml", weight_name="model.ckpt").eval()
DT = torch.float16 if args.dtype == "float16" else torch.float32
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def convert(module, example, inputs, outputs, path):
    t0 = time.time()
    with torch.no_grad():
        ep = torch.export.export(module, example).run_decompositions(ct.get_decomp_table())
    c = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    c.add_exported_program(ep, input_names=inputs, output_names=outputs)
    prog = c.to_coreai()
    prog.optimize()
    if path.exists():
        import shutil
        shutil.rmtree(path)
    prog.save_asset(path)
    print(path.name, "exported in %.1fs" % (time.time() - t0), flush=True)


# Fixed synthetic input: a shaded sphere-ish blob on gray (deterministic; real images come later from the host).
yy, xx = np.mgrid[0:512, 0:512] / 511.0
blob = ((xx - 0.5) ** 2 + (yy - 0.5) ** 2) < 0.12
img = np.full((512, 512, 3), 0.5, np.float32)
img[blob] = np.stack([0.8 * (1 - yy[blob]), 0.3 + 0.4 * xx[blob], 0.2 + 0.0 * xx[blob]], -1)
image = torch.from_numpy(img).permute(2, 0, 1)[None]
np.save(out / "input-image.npy", image.numpy())

scene_ref = Scene(model)
with torch.no_grad():
    t0 = time.time()
    codes32 = scene_ref(image)
    print("eager fp32 scene (CPU) %.2fs" % (time.time() - t0))
np.save(out / "ref-scene-codes.npy", codes32.numpy())
rng = np.random.default_rng(0)
pts = torch.from_numpy(rng.uniform(-0.87, 0.87, (args.points, 3)).astype(np.float32))
np.save(out / "ref-points.npy", pts.numpy())
field_ref = Field(model)
with torch.no_grad():
    d32, c32 = field_ref(codes32, pts)
    # Cross-check the explicit sampler against upstream grid_sample + decoder on the same points.
    up = model.renderer.query_triplane(model.decoder, pts, codes32)
    print("explicit vs grid_sample: density rel err %.2e, color max err %.2e" % (
        float((d32 - up["density_act"]).abs().max() / up["density_act"].abs().max()),
        float((c32 - up["color"]).abs().max())))
np.save(out / "ref-density.npy", d32.numpy())
np.save(out / "ref-color.npy", c32.numpy())

model = model.to(DT)
tag = "fp16" if DT == torch.float16 else "fp32"
if "scene" in args.only:
    convert(Scene(model).to(DT), (image.to(DT),), ["image"], ["scene_codes"], out / f"scene-{tag}.aimodel")
if "field" in args.only:
    name = f"field-{tag}" if args.points == 65536 else f"field-{args.points}-{tag}"
    convert(Field(model).to(DT), (codes32.to(DT), pts.to(DT)), ["scene_codes", "points"], ["density", "color"],
            out / f"{name}.aimodel")
if args.grid_slab:
    lin = torch.linspace(0, 1, args.res) * 2 * 0.87 - 0.87
    grid = GridField(model).to(DT)
    convert(grid, (codes32.to(DT), lin.to(DT), lin[:args.grid_slab].to(DT)), ["scene_codes", "xs", "zs"], ["density"],
            out / f"grid-{args.res}x{args.grid_slab}-{tag}.aimodel")

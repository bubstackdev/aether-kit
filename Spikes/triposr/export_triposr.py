"""Throwaway spike: export TripoSR (MIT, stabilityai/TripoSR) to Core AI as two functions-in-separate-assets.

  scene.aimodel  `main`: image [1, 3, 512, 512] (RGB in [0, 1], background already composited) → scene codes
                 [3, 40, 64, 64]  (DINO ViT-B/16 → 16-layer triplane transformer → ConvTranspose upsample)
  field.aimodel  `main`: scene codes [3, 40, 64, 64] + points [N, 3] in (-r, r) → density [N, 1], color [N, 3]
                 (triplane bilinear sampling as an explicit gather: Core AI has no grid_sampler_2d lowering)

The upstream code (TripoSR-main, MIT) is imported read-only from the downloaded tarball. torchmcubes is not
needed for export and is stubbed. Also writes eager fp32 reference outputs for a fixed synthetic image.

usage: export_triposr.py <tsr-source-dir> <out-dir> [--dtype float16|float32] [--points N]
"""
from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import coreai_torch as ct

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--dtype", default="float16")
ap.add_argument("--points", type=int, default=65536)
ap.add_argument("--only", default="scene,field")
args = ap.parse_args()

sys.modules["torchmcubes"] = types.SimpleNamespace(marching_cubes=None)   # export never runs marching cubes
sys.modules.setdefault("rembg", types.ModuleType("rembg"))
sys.path.insert(0, args.src)
from tsr.system import TSR  # noqa: E402

snap = next((Path.home() / ".cache/huggingface/hub/models--stabilityai--TripoSR/snapshots").iterdir())
model = TSR.from_pretrained(str(snap), config_name="config.yaml", weight_name="model.ckpt").eval()
DT = torch.float16 if args.dtype == "float16" else torch.float32
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


class Scene(torch.nn.Module):
    """Image → scene codes with the 32×32 interpolated position embedding precomputed (fixed 512 input)."""

    def __init__(self, m: TSR):
        super().__init__()
        self.m = m
        vit = m.image_tokenizer.model
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 512, 512)
            pos = vit.embeddings.interpolate_pos_encoding(vit.embeddings.position_embeddings.new_zeros(1, 1025, 768),
                                                          512, 512)
        self.register_buffer("pos", pos.clone())
        vit.embeddings.interpolate_pos_encoding = lambda embeddings, h, w: self.pos.to(embeddings.dtype)

    def forward(self, image):
        tok = self.m.image_tokenizer
        x = (image[:, None] - tok.image_mean.to(image.dtype)) / tok.image_std.to(image.dtype)
        o = tok.model(x[:, 0], interpolate_pos_encoding=True)
        img_tokens = o.last_hidden_state                                   # [1, 1025, 768]
        tokens = self.m.tokenizer(1)                                        # [1, 1024, 3072]
        tokens = self.m.backbone(tokens, encoder_hidden_states=img_tokens)
        return self.m.post_processor(self.m.tokenizer.detokenize(tokens))[0]   # [3, 40, 64, 64]


class Field(torch.nn.Module):
    """Triplane density/color query with explicit bilinear sampling (grid_sample, align_corners=False, zero pad)."""

    def __init__(self, m: TSR):
        super().__init__()
        self.decoder = m.decoder
        self.radius = float(m.renderer.cfg.radius)
        self.density_bias = float(m.renderer.cfg.density_bias)

    def forward(self, planes, points):
        P, C, H, W = planes.shape                                   # 3, 40, 64, 64
        padded = F.pad(planes, (1, 1, 1, 1))                         # zero border = grid_sample zeros padding
        Hp, Wp = H + 2, W + 2
        flat = padded.permute(0, 2, 3, 1).reshape(P * Hp * Wp, C)    # [3·66·66, 40]
        x = points / self.radius                                     # (-1, 1)
        feats = []
        for p, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):        # grid (x → W, y → H) = (coord a, coord b)
            u = ((x[:, a] + 1) * W - 1) * 0.5 + 1                    # padded pixel coordinates
            v = ((x[:, b] + 1) * H - 1) * 0.5 + 1
            u0, v0 = torch.floor(u), torch.floor(v)
            wu, wv = (u - u0)[:, None], (v - v0)[:, None]
            u0i, v0i = u0.to(torch.int32), v0.to(torch.int32)
            base = p * Hp * Wp
            def g(vv, uu):
                return flat.index_select(0, (base + vv * Wp + uu).reshape(-1))
            f = (g(v0i, u0i) * (1 - wu) * (1 - wv) + g(v0i, u0i + 1) * wu * (1 - wv)
                 + g(v0i + 1, u0i) * (1 - wu) * wv + g(v0i + 1, u0i + 1) * wu * wv)
            feats.append(f)
        h = torch.cat(feats, dim=-1)                                 # [N, 120], plane-major like "Np Cp -> (Np Cp)"
        o = self.decoder.layers(h)
        density = torch.exp(o[:, 0:1] + self.density_bias)
        color = torch.sigmoid(o[:, 1:4])
        return density, color


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
    convert(Field(model).to(DT), (codes32.to(DT), pts.to(DT)), ["scene_codes", "points"], ["density", "color"],
            out / f"field-{tag}.aimodel")

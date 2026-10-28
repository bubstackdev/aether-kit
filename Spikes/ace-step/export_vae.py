"""Throwaway spike: export the ACE-Step 1.5 audio VAE decoder (diffusers AutoencoderOobleck, 48 kHz stereo,
1920× upsampling from 25 Hz × 64-channel latents) to Core AI at a fixed latent window.

usage: export_vae.py <out-dir> [--frames 250] [--dtype float16|float32]
Writes vae-dec-<frames>-<tag>.aimodel plus an eager fp32 reference (input latents + output audio) for the window.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderOobleck

import coreai_torch as ct

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--frames", type=int, default=250)
ap.add_argument("--dtype", default="float16")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
snap = next((hub / "models--ACE-Step--Ace-Step1.5/snapshots").iterdir())
vae = AutoencoderOobleck.from_pretrained(snap / "vae", torch_dtype=torch.float32).eval()
# Fold weight norm (g · v/‖v‖) into plain weights: the export should see ordinary convolutions.
for m in vae.modules():
    try:
        torch.nn.utils.remove_weight_norm(m)
    except ValueError:
        pass


class Decode(torch.nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, latents):           # [1, 64, frames] → [1, 2, frames · 1920]
        return self.dec(latents)


rng = np.random.default_rng(0)
lat = torch.from_numpy(rng.standard_normal((1, 64, a.frames)).astype(np.float32))
ref_path = out / f"vae-ref-latents-{a.frames}.npy"
if not ref_path.exists():
    np.save(ref_path, lat.numpy())
    with torch.no_grad():
        t = time.time()
        audio = Decode(vae.decoder)(lat)
        print("eager fp32 CPU decode of %d frames: %.2fs -> %s" % (a.frames, time.time() - t, tuple(audio.shape)))
    np.save(out / f"vae-ref-audio-{a.frames}.npy", audio.numpy())
lat = torch.from_numpy(np.load(ref_path))
DT = torch.float16 if a.dtype == "float16" else torch.float32
tag = "fp16" if DT == torch.float16 else "fp32"
m = Decode(vae.decoder).to(DT)
t = time.time()
with torch.no_grad():
    ep = torch.export.export(m, (lat.to(DT),)).run_decompositions(ct.get_decomp_table())
c = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
c.add_exported_program(ep, input_names=["latents"], output_names=["audio"])
prog = c.to_coreai()
prog.optimize()
path = out / f"vae-dec-{a.frames}-{tag}.aimodel"
if path.exists():
    import shutil
    shutil.rmtree(path)
prog.save_asset(path)
print(path.name, "exported in %.1fs" % (time.time() - t))

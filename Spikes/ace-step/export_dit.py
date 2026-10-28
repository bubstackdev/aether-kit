"""Throwaway spike: export one ACE-Step 1.5 turbo DiT velocity step (decoder only, 24 layers, 2048-d) to Core AI.

Inputs (static): xt [1, T, 64] noisy latent, t [1] timestep, context [1, T, 128] (source latents ‖ chunk mask),
encoder [1, L, 2048] packed condition (lyric ‖ timbre ‖ text after the condition encoder), encoder_mask [1, L]
(1 for condition rows, 0 for padding; 0.0.3: real conditions are shorter than L and padded). Output: v [1, T, 64].
The upstream modeling file (MIT, from the model repo) is imported from the snapshot; only decoder.* weights load.
Writes an eager fp32 reference output for fixed random inputs.

usage: export_dit.py <out-dir> [--frames 1500] [--enc 512] [--dtype float16] [--skip-export]
"""
import argparse
import os
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--frames", type=int, default=1500)
ap.add_argument("--enc", type=int, default=512)
ap.add_argument("--dtype", default="float16")
ap.add_argument("--skip-export", action="store_true")
ap.add_argument("--skip-ref", action="store_true")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
snap = next((hub / "models--ACE-Step--Ace-Step1.5/snapshots").iterdir())
mdir = snap / "acestep-v15-turbo"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cfg_mod = load_module("configuration_acestep_v15", mdir / "configuration_acestep_v15.py")
modeling = load_module("modeling_acestep_v15_turbo", mdir / "modeling_acestep_v15_turbo.py")
config = cfg_mod.AceStepConfig(**json.loads((mdir / "config.json").read_text()))
config._attn_implementation = "sdpa"
dit = modeling.AceStepDiTModel(config).eval()
state = {}
with safe_open(mdir / "model.safetensors", framework="pt") as f:
    for k in f.keys():
        if k.startswith("decoder."):
            state[k.removeprefix("decoder.")] = f.get_tensor(k).float()
missing, unexpected = dit.load_state_dict(state, strict=False)
print("missing", missing[:5], "unexpected", unexpected[:5])
del state
print("decoder params %.2fB" % (sum(p.numel() for p in dit.parameters()) / 1e9))


class Step(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, xt, t, context, encoder, encoder_mask):
        ones = torch.ones(xt.shape[:2], dtype=xt.dtype)
        return self.d(hidden_states=xt, timestep=t, timestep_r=t, attention_mask=ones,
                      encoder_hidden_states=encoder, encoder_attention_mask=encoder_mask > 0.5, context_latents=context,
                      use_cache=False)[0]


T, L = a.frames, a.enc
rng = np.random.default_rng(0)
inputs = [rng.standard_normal((1, T, 64)).astype(np.float32), np.array([0.6], np.float32),
          np.concatenate([0.3 * rng.standard_normal((1, T, 64)), np.ones((1, T, 64))], -1).astype(np.float32),
          (0.5 * rng.standard_normal((1, L, 2048))).astype(np.float32), np.ones((1, L), np.float32)]
names = ["xt", "t", "context", "encoder", "encoder_mask"]
for n, x in zip(names, inputs):
    np.save(out / f"dit-ref-{n}-{T}x{L}.npy", x)
step = Step(dit)
if not a.skip_ref:
    with torch.no_grad():
        t0 = time.time()
        v = step(*[torch.from_numpy(x) for x in inputs])
        print("eager fp32 CPU step T=%d L=%d: %.1fs" % (T, L, time.time() - t0))
    np.save(out / f"dit-ref-v-{T}x{L}.npy", v.numpy())


def export():

    import coreai_torch as ct  # noqa: E402

    DT = torch.float16 if a.dtype == "float16" else torch.float32
    tag = "fp16" if DT == torch.float16 else "fp32"
    half = step.to(DT)
    t0 = time.time()
    with torch.no_grad():
        ep = torch.export.export(half, tuple(torch.from_numpy(x).to(DT) for x in inputs))
        ep = ep.run_decompositions(ct.get_decomp_table())
    print("torch.export %.1fs" % (time.time() - t0))
    c = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    c.add_exported_program(ep, input_names=names, output_names=["v"])
    prog = c.to_coreai()
    prog.optimize()
    path = out / f"dit-{T}x{L}-{tag}.aimodel"
    if path.exists():
        import shutil
        shutil.rmtree(path)
    prog.save_asset(path)
    print(path.name, "exported in %.1fs" % (time.time() - t0))


if not a.skip_export:
    export()

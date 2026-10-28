"""Why does the Core AI DiT step miss eager fp32 on real conditions (step-0 velocity cosine about 0.66)? Checks the
padded condition sequence and its mask, which the 0.0.2 single-step check (all-ones mask, random inputs) never used.

usage: diag_dit_mask.py <export-dir> <reference.npz> <out-dir> --eager      (CPU: eager fp32 step 0, three ways)
       USE_OS_COREAI=1 diag_dit_mask.py <export-dir> <reference.npz> <out-dir> --run   (GPU, under the lock)

--eager writes, for step 0 of the reference's trajectory:
  unpadded   the condition rows only (L = valid rows), as reference_trajectory.py runs it;
  padded     zero-padded to the export's L with the boolean mask, as the export is called;
  unmasked   zero-padded with an all-ones mask: what the step computes if the mask is ignored;
  masked     zero-padded through dit_step.Step, which applies the mask to cross-attention (the upstream DiT discards
             its encoder_attention_mask argument, so `padded` equals `unmasked`).
--run feeds the padded inputs to the Core AI step (exported through dit_step.Step) and compares it with all four, and
also runs the export's own fixed random inputs (all-ones mask) against their eager output, as the 0.0.2 check did.
"""
import argparse
import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("exports")
ap.add_argument("reference")
ap.add_argument("out")
ap.add_argument("--eager", action="store_true")
ap.add_argument("--run", action="store_true")
ap.add_argument("--dit", default="dit-1500x512-fp16")
a = ap.parse_args()
E, OUT = Path(a.exports), Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
ref = np.load(a.reference)
n = int(ref["encoder_mask"].sum())
T0 = 1.0                                                       # the turbo schedule's first timestep


def rel_l2(x, y):
    return float(np.linalg.norm((x - y).ravel()) / np.linalg.norm(y.ravel()))


def cosine(x, y):
    x, y = x.ravel().astype(np.float64), y.ravel().astype(np.float64)
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


if a.eager:
    import torch
    from safetensors import safe_open

    torch.set_grad_enabled(False)
    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    mdir = next((hub / "models--ACE-Step--Ace-Step1.5/snapshots").iterdir()) / "acestep-v15-turbo"

    def load_module(name, path):
        s = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(s)
        sys.modules[name] = m
        s.loader.exec_module(m)
        return m

    cfg_mod = load_module("configuration_acestep_v15", mdir / "configuration_acestep_v15.py")
    modeling = load_module("modeling_acestep_v15_turbo", mdir / "modeling_acestep_v15_turbo.py")
    config = cfg_mod.AceStepConfig(**json.loads((mdir / "config.json").read_text()))
    config._attn_implementation = "sdpa"
    dit = modeling.AceStepDiTModel(config).eval()
    with safe_open(mdir / "model.safetensors", framework="pt") as f:
        dit.load_state_dict({k.removeprefix("decoder."): f.get_tensor(k).float() for k in f.keys()
                             if k.startswith("decoder.")})
    xt, context = torch.from_numpy(ref["noise"]), torch.from_numpy(ref["context"])
    encoder, mask = torch.from_numpy(ref["encoder"]), torch.from_numpy(ref["encoder_mask"])
    t = torch.full((1,), T0)

    def step(enc, enc_mask):
        return dit(hidden_states=xt, timestep=t, timestep_r=t, attention_mask=torch.ones(xt.shape[:2]),
                   encoder_hidden_states=enc, encoder_attention_mask=enc_mask, context_latents=context,
                   use_cache=False)[0].numpy()

    sys.path.insert(0, str(Path(__file__).parent))
    from dit_step import Step

    ways = {"unpadded": step(encoder[:, :n], mask[:, :n] > 0.5), "padded": step(encoder, mask > 0.5),
            "unmasked": step(encoder, torch.ones_like(mask) > 0.5),
            "masked": Step(dit)(xt, t, context, encoder, mask).numpy()}
    for name, v in ways.items():
        np.save(OUT / f"eager-v0-{name}.npy", v)
    report = {"validRows": n, "L": int(mask.shape[1]),
              "referenceStep0VsUnpadded": rel_l2(ways["unpadded"], ref["fp32_vt"][0]),
              "paddedVsUnpadded": {"relL2": rel_l2(ways["padded"], ways["unpadded"]),
                                   "cosine": cosine(ways["padded"], ways["unpadded"])},
              "unmaskedVsUnpadded": {"relL2": rel_l2(ways["unmasked"], ways["unpadded"]),
                                     "cosine": cosine(ways["unmasked"], ways["unpadded"])},
              "maskedVsUnpadded": {"relL2": rel_l2(ways["masked"], ways["unpadded"]),
                                   "cosine": cosine(ways["masked"], ways["unpadded"])}}
    (OUT / "eager.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))

if a.run:
    import coreai.runtime as rt

    async def main():
        opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
        fn = (await rt.AIModel.load(E / f"{a.dit}.aimodel", opts)).load_function("main")
        dt = np.float16
        feed = {"xt": rt.NDArray(ref["noise"].astype(dt)), "t": rt.NDArray(np.array([T0], dt)),
                "context": rt.NDArray(ref["context"].astype(dt)), "encoder": rt.NDArray(ref["encoder"].astype(dt)),
                "encoder_mask": rt.NDArray(ref["encoder_mask"].astype(dt))}
        v = (await fn(feed))["v"].numpy().astype(np.float32)
        report = {}
        for name in ("unpadded", "padded", "unmasked", "masked"):
            eager = np.load(OUT / f"eager-v0-{name}.npy")
            report[f"coreaiVsEager{name.capitalize()}"] = {"relL2": rel_l2(v, eager), "cosine": cosine(v, eager)}
        # The export's own fixed random inputs (all-ones mask), as the 0.0.2 check ran them.
        tag = a.dit.split("-")[1]
        names = ["xt", "t", "context", "encoder", "encoder_mask"]
        rand = {k: np.load(E / f"dit-ref-{k}-{tag}.npy") for k in names}
        vr = (await fn({k: rt.NDArray(x.astype(dt)) for k, x in rand.items()}))["v"].numpy().astype(np.float32)
        eager_r = np.load(E / f"dit-ref-v-{tag}.npy")
        report["randomInputsAllOnesMask"] = {"relL2": rel_l2(vr, eager_r), "cosine": cosine(vr, eager_r)}
        (OUT / "coreai.json").write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps(report))

    asyncio.run(main())

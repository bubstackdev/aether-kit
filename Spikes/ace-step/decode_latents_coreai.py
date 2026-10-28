"""Decode every trajectory's final latents with the same VAE, so the gate can judge the DiT and the VAE separately.

usage: USE_OS_COREAI=1 decode_latents_coreai.py <export-dir> <references-dir> <coreai-dir> [--vae32 NAME] [--vae16 NAME]
Runs under the GPU lock. reference_trajectory.py decodes fp32 and bf16 latents with the eager fp32 VAE, while the Core
AI runs decode with the fp16 VAE export, so a straight audio comparison mixes the DiT's error with the fp16 VAE's. This
decodes, with the Core AI fp32 VAE (default vae-dec-1500-fp32): the fp32 and bf16 reference latents and every Core AI
candidate's latents (<id>-coreai.npz, <id>-e2e.npz), and, with the fp16 VAE (default vae-dec-1500-fp16), the fp32
latents. Writes <id>-decoded.npz in <coreai-dir> (fp32, bf16, coreai, e2e: fp32-VAE audio; fp32_vae16: the fp32
latents through the fp16 VAE) and decoded.json (the fp32 VAE export against the eager fp32 VAE's reference audio).
"""
import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import coreai.runtime as rt

ap = argparse.ArgumentParser()
ap.add_argument("exports")
ap.add_argument("references")
ap.add_argument("coreai")
ap.add_argument("--vae32", default="vae-dec-1500-fp32")
ap.add_argument("--vae16", default="vae-dec-1500-fp16")
a = ap.parse_args()
E, REF, C = Path(a.exports), Path(a.references), Path(a.coreai)


def rel_l2(x, y):
    return float(np.linalg.norm((x - y).ravel()) / np.linalg.norm(y.ravel()))


async def main():
    opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    vae32 = (await rt.AIModel.load(E / f"{a.vae32}.aimodel", opts)).load_function("main")
    vae16 = (await rt.AIModel.load(E / f"{a.vae16}.aimodel", opts)).load_function("main")

    async def decode(fn, x0, dt):
        latents = x0.astype(np.float32).transpose(0, 2, 1).astype(dt)            # [1, T, 64] → [1, 64, T]
        return (await fn({"latents": rt.NDArray(latents)}))["audio"].numpy()[0].astype(np.float32)

    report = {"vae32": a.vae32, "vae16": a.vae16, "prompts": {}}
    for path in sorted(REF.glob("*.npz")):
        ref = np.load(path)
        out = {"fp32": await decode(vae32, ref["fp32_x0"], np.float32),
               "fp32_vae16": await decode(vae16, ref["fp32_x0"], np.float16)}
        if "bf16_x0" in ref.files:
            out["bf16"] = await decode(vae32, ref["bf16_x0"], np.float32)
        for suffix in ("coreai", "e2e"):
            candidate = C / f"{path.stem}-{suffix}.npz"
            if candidate.exists():
                out[suffix] = await decode(vae32, np.load(candidate)["coreai_x0"], np.float32)
        np.savez(C / f"{path.stem}-decoded.npz", **out)
        report["prompts"][path.stem] = {"decoded": sorted(out),
                                        "vae32VsEagerFp32Audio": rel_l2(out["fp32"], ref["fp32_audio"])}
        print(path.stem, json.dumps(report["prompts"][path.stem]), flush=True)
    (C / "decoded.json").write_text(json.dumps(report, indent=1) + "\n")


asyncio.run(main())

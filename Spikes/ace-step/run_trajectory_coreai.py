"""The ACE-Step 1.5 turbo trajectory through Core AI, from the eager references' conditions and noise.

usage: USE_OS_COREAI=1 run_trajectory_coreai.py <export-dir> <references-dir> <out-dir> [--dit NAME] [--vae NAME]
Runs under the GPU lock. For each <id>.npz from reference_trajectory.py: 8 turbo steps (shift 3) through the exported
DiT velocity step (default dit-1500x512-fp16), with the latent state and the Euler update kept in fp32 on the host and
cast to the asset's precision at each call, then the exported VAE decoder (default vae-dec-1500-fp16). Feeding the
reference conditions judges the DiT and the VAE alone; the encoder exports are judged separately.

Writes <id>-coreai.npz (coreai_xt, coreai_vt [8, 1, T, 64], coreai_x0, coreai_audio) and <id>-coreai.wav, plus
coreai-run.json with timings and the process's footprint per stage.
"""
import argparse
import asyncio
import json
import resource
import time
import wave
from pathlib import Path

import numpy as np
import coreai.runtime as rt

# acestep_port.SHIFT_TIMESTEPS[3.0], inlined so this runner needs only the Core AI runtime and numpy.
SHIFT_TIMESTEPS_3 = [1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3]

ap = argparse.ArgumentParser()
ap.add_argument("exports")
ap.add_argument("references")
ap.add_argument("out")
ap.add_argument("--dit", default="dit-1500x512-fp16")
ap.add_argument("--vae", default="vae-dec-1500-fp16")
a = ap.parse_args()
E, REF, OUT = Path(a.exports), Path(a.references), Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)


def footprint():
    """Peak resident set so far (bytes); the gate's memory figure is the kit-side measurement, this is a hint."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def write_wav(path, audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").T.copy()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(audio.shape[0])
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(pcm.tobytes())


async def main():
    opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    report = {"dit": a.dit, "vae": a.vae, "prompts": {}}
    t0 = time.perf_counter()
    dit_model = await rt.AIModel.load(E / f"{a.dit}.aimodel", opts)
    dit = dit_model.load_function("main")
    report["ditLoadSeconds"], report["afterDitLoadMaxRSS"] = time.perf_counter() - t0, footprint()
    t0 = time.perf_counter()
    vae_model = await rt.AIModel.load(E / f"{a.vae}.aimodel", opts)
    vae = vae_model.load_function("main")
    report["vaeLoadSeconds"], report["afterVaeLoadMaxRSS"] = time.perf_counter() - t0, footprint()
    dt = np.float16 if a.dit.endswith("fp16") else np.float32
    vae_dt = np.float16 if a.vae.endswith("fp16") else np.float32
    for path in sorted(REF.glob("*.npz")):
        if path.stem.endswith("-coreai"):
            continue
        ref = np.load(path)
        static = {"context": rt.NDArray(ref["context"].astype(dt)), "encoder": rt.NDArray(ref["encoder"].astype(dt)),
                  "encoder_mask": rt.NDArray(ref["encoder_mask"].astype(dt))}
        xt = ref["noise"].astype(np.float32)
        xs, vs, step_seconds = [], [], []
        for i, t in enumerate(SHIFT_TIMESTEPS_3):
            feed = dict(static, xt=rt.NDArray(xt.astype(dt)), t=rt.NDArray(np.array([t], dt)))
            s = time.perf_counter()
            vt = (await dit(feed))["v"].numpy().astype(np.float32)
            step_seconds.append(time.perf_counter() - s)
            xs.append(xt)
            vs.append(vt)
            xt = xt - vt * t if i == len(SHIFT_TIMESTEPS_3) - 1 else xt - vt * (t - SHIFT_TIMESTEPS_3[i + 1])
        s = time.perf_counter()
        audio = (await vae({"latents": rt.NDArray(xt.transpose(0, 2, 1).astype(vae_dt))}))["audio"].numpy()[0]
        vae_seconds = time.perf_counter() - s
        audio = audio.astype(np.float32)
        np.savez(OUT / f"{path.stem}-coreai.npz", coreai_xt=np.stack(xs), coreai_vt=np.stack(vs), coreai_x0=xt,
                 coreai_audio=audio)
        write_wav(OUT / f"{path.stem}-coreai.wav", audio)
        report["prompts"][path.stem] = {"stepSeconds": step_seconds, "ditSeconds": sum(step_seconds),
                                        "vaeSeconds": vae_seconds, "maxRSS": footprint()}
        print(path.stem, json.dumps(report["prompts"][path.stem]), flush=True)
    (OUT / "coreai-run.json").write_text(json.dumps(report, indent=1))


asyncio.run(main())

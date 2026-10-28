"""The go configuration in one process, for its own time and footprint: fp16 text encoder, fp16 condition encoder,
fp16 masked DiT and the fp32 VAE in 10 s tiles (32-frame overlap), with nothing else loaded.

usage: USE_OS_COREAI=1 go_pipeline_coreai.py <export-dir> <references-dir> <coreai-dir> <out.json>
           [--text 256] [--lyric 512] [--dit dit-1500x512-fp16] [--window vae-dec-250-fp32] [--overlap 32]
Runs under the GPU lock and /usr/bin/time -l (the process's peak footprint is the go configuration's). For each
<id>.npz it runs the front end and trajectory exactly as run_encoders_coreai.py does, then tile-decodes, and checks
the final latents and audio against that earlier end-to-end run decoded by the one-window fp32 VAE
(<coreai-dir>/<id>-e2e.npz and <id>-decoded.npz), so the timings belong to the same computation.
The host packing follows pack_sequences in the pinned model file (ACE-Step/Ace-Step1.5 @ 19671f4,
modeling_acestep_v15_turbo.py), Copyright (c) 2026 ACEStep, used under the MIT License; see LICENSE-ACE-Step-1.5.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import coreai.runtime as rt

sys.path.insert(0, str(Path(__file__).parent))
from tiled_vae_coreai import tile_decode  # noqa: E402

SHIFT_TIMESTEPS_3 = [1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3]
PAD_ID = 151643

ap = argparse.ArgumentParser()
ap.add_argument("exports")
ap.add_argument("references")
ap.add_argument("coreai")
ap.add_argument("out")
ap.add_argument("--text", type=int, default=256)
ap.add_argument("--lyric", type=int, default=512)
ap.add_argument("--dit", default="dit-1500x512-fp16")
ap.add_argument("--window", default="vae-dec-250-fp32")
ap.add_argument("--overlap", type=int, default=32)
a = ap.parse_args()
E, REF, C = Path(a.exports), Path(a.references), Path(a.coreai)


def snr_db(x, ref):
    x, ref = x.astype(np.float64), ref.astype(np.float64)
    return float(10 * np.log10(np.sum(ref ** 2) / max(np.sum((x - ref) ** 2), 1e-30)))


async def main():
    opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    report = {"configuration": {"text": f"text-encoder-{a.text}-fp16", "condition": f"condition-{a.text}x{a.lyric}-fp16",
                                "dit": a.dit, "vae": a.window, "overlap": a.overlap}, "load": {}, "prompts": {}}
    fns = {}
    for key, name in report["configuration"].items():
        if key == "overlap":
            continue
        t0 = time.perf_counter()
        fns[key] = (await rt.AIModel.load(E / f"{name}.aimodel", opts)).load_function("main")
        report["load"][name] = time.perf_counter() - t0
    table = np.load(E / "lyric-embedding-table.npy")
    timbre = np.load(E / "timbre-silence.npy").astype(np.float32)
    window = int(a.window.split("-")[2])
    vae_dt = np.float16 if a.window.endswith("fp16") else np.float32

    for path in sorted(REF.glob("*.npz")):
        ref = np.load(path)
        rep, t_start = {}, time.perf_counter()
        text_ids, lyric_ids = ref["text_ids"][0], ref["lyric_ids"][0]
        lt, ll = len(text_ids), len(lyric_ids)
        ids = np.full((1, a.text), PAD_ID, np.int32)
        ids[0, :lt] = text_ids
        s = time.perf_counter()
        hidden = (await fns["text"]({"input_ids": rt.NDArray(ids)}))["hidden"].numpy()
        rep["textSeconds"] = time.perf_counter() - s
        lyric = np.zeros((1, a.lyric, table.shape[1]), np.float16)
        lyric[0, :ll] = table[lyric_ids]
        mask = np.zeros((1, a.lyric), np.float16)
        mask[0, :ll] = 1
        s = time.perf_counter()
        o = await fns["condition"]({"text_hidden": rt.NDArray(hidden.astype(np.float16)),
                                     "lyric_embeds": rt.NDArray(lyric), "lyric_mask": rt.NDArray(mask)})
        rep["conditionSeconds"] = time.perf_counter() - s
        packed = np.concatenate([o["lyric_encoded"].numpy().astype(np.float32)[:, :ll], timbre,
                                 o["text_projected"].numpy().astype(np.float32)[:, :lt]], axis=1)
        L, n = ref["encoder"].shape[1], packed.shape[1]
        encoder = np.zeros((1, L, packed.shape[2]), np.float16)
        encoder[:, :n] = packed
        encoder_mask = np.zeros((1, L), np.float16)
        encoder_mask[:, :n] = 1
        static = {"context": rt.NDArray(ref["context"].astype(np.float16)), "encoder": rt.NDArray(encoder),
                  "encoder_mask": rt.NDArray(encoder_mask)}
        xt = ref["noise"].astype(np.float32)
        steps = []
        for i, t in enumerate(SHIFT_TIMESTEPS_3):
            s = time.perf_counter()
            vt = (await fns["dit"](dict(static, xt=rt.NDArray(xt.astype(np.float16)),
                                        t=rt.NDArray(np.array([t], np.float16)))))["v"].numpy().astype(np.float32)
            steps.append(time.perf_counter() - s)
            xt = xt - vt * t if i == len(SHIFT_TIMESTEPS_3) - 1 else xt - vt * (t - SHIFT_TIMESTEPS_3[i + 1])
        rep["stepSeconds"], rep["ditSeconds"] = steps, sum(steps)
        s = time.perf_counter()
        audio, windows = await tile_decode(fns["vae"], xt.transpose(0, 2, 1), window, a.overlap, vae_dt)
        rep["vaeSeconds"], rep["vaeWindows"] = time.perf_counter() - s, windows
        rep["totalSeconds"] = time.perf_counter() - t_start
        earlier = np.load(C / f"{path.stem}-e2e.npz")["coreai_x0"]
        rep["finalLatentsEqualEarlierE2E"] = bool(np.array_equal(xt, earlier))
        rep["audioSNRdBVsEarlierE2EOneWindowFp32VAE"] = snr_db(audio, np.load(C / f"{path.stem}-decoded.npz")["e2e"])
        report["prompts"][path.stem] = rep
        print(path.stem, json.dumps(rep), flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1) + "\n")


asyncio.run(main())

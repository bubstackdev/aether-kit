"""The ACE-Step 1.5 text-to-music front end through Core AI, then the trajectory end to end (the 0.0.2 report's open
item 2), and the VAE decoded in tiles against one window (open item 4).

usage: USE_OS_COREAI=1 run_encoders_coreai.py <export-dir> <references-dir> <out-dir> [--text 256] [--lyric 512]
           [--dit NAME] [--vae NAME] [--tile-vae NAME] [--overlaps 0,8,16,32,64]
Runs under the GPU lock. For each <id>.npz from reference_trajectory.py:
  1. text-encoder-<text>: the caption prompt's token IDs, right-padded with <|endoftext|> (the model is causal, so the
     first rows don't depend on the padding) → hidden states, compared with the eager fp32 `text_hidden`;
  2. the lyric token embeddings by row lookup in the exported embedding table, padded with a mask;
  3. condition-<text>x<lyric>: text projector and lyric encoder;
  4. packing on the host, as upstream's pack_sequences does for text-to-music: the valid lyric rows, the constant
     silence-timbre row, the valid caption rows, then zero padding to the DiT's L; compared with the eager `encoder`;
  5. the 8-step trajectory through the DiT with those conditions and the VAE, written as <id>-e2e.npz/wav for
     gate_trajectory.py (--candidate e2e);
  6. tiled decoding: the end-to-end final latents decoded by the window asset (default vae-dec-250-fp16) with each
     overlap, a linear crossfade across the overlap, against one decode of the whole 60 s by the long window asset.
Writes encoders-run.json.
"""
import argparse
import asyncio
import json
import time
import wave
from pathlib import Path

import numpy as np
import coreai.runtime as rt

SHIFT_TIMESTEPS_3 = [1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3]
PAD_ID = 151643                     # <|endoftext|>, Qwen's padding token

ap = argparse.ArgumentParser()
ap.add_argument("exports")
ap.add_argument("references")
ap.add_argument("out")
ap.add_argument("--text", type=int, default=256)
ap.add_argument("--lyric", type=int, default=512)
ap.add_argument("--dit", default="dit-1500x512-fp16")
ap.add_argument("--vae", default="vae-dec-1500-fp16")
ap.add_argument("--tile-vae", default="vae-dec-250-fp16")
ap.add_argument("--overlaps", default="0,8,16,32,64")
a = ap.parse_args()
E, REF, OUT = Path(a.exports), Path(a.references), Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)


def rel_l2(x, ref):
    return float(np.linalg.norm((x - ref).ravel()) / np.linalg.norm(ref.ravel()))


def cosine(x, ref):
    x, ref = x.ravel().astype(np.float64), ref.ravel().astype(np.float64)
    return float(x @ ref / (np.linalg.norm(x) * np.linalg.norm(ref)))


def snr_db(x, ref):
    return float(10 * np.log10(np.sum(ref.astype(np.float64) ** 2) / max(np.sum((x - ref).astype(np.float64) ** 2), 1e-20)))


def write_wav(path, audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").T.copy()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(audio.shape[0])
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(pcm.tobytes())


async def load(name, opts):
    model = await rt.AIModel.load(E / f"{name}.aimodel", opts)
    return model, model.load_function("main")


async def main():
    opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    report = {"text": a.text, "lyric": a.lyric, "dit": a.dit, "vae": a.vae, "tileVae": a.tile_vae, "load": {},
              "prompts": {}}
    models = {}
    for key, name in (("text", f"text-encoder-{a.text}-fp16"), ("condition", f"condition-{a.text}x{a.lyric}-fp16"),
                      ("dit", a.dit), ("vae", a.vae), ("tile", a.tile_vae)):
        t0 = time.perf_counter()
        models[key] = await load(name, opts)
        report["load"][name] = time.perf_counter() - t0
    table = np.load(E / "lyric-embedding-table.npy")                 # [vocab, 1024] fp16
    timbre = np.load(E / "timbre-silence.npy").astype(np.float32)    # [1, rows, 2048]
    window = int(a.tile_vae.split("-")[2])
    hop = 1920                                                       # audio samples per latent frame

    for path in sorted(REF.glob("*.npz")):
        if path.stem.endswith(("-coreai", "-e2e")):
            continue
        ref = np.load(path)
        rep = {}
        text_ids, lyric_ids = ref["text_ids"][0], ref["lyric_ids"][0]
        lt, ll = len(text_ids), len(lyric_ids)
        if lt > a.text or ll > a.lyric:
            raise SystemExit(f"{path.stem}: {lt} caption / {ll} lyric tokens exceed the exported {a.text} / {a.lyric}")
        ids = np.full((1, a.text), PAD_ID, np.int32)
        ids[0, :lt] = text_ids
        s = time.perf_counter()
        hidden = (await models["text"][1]({"input_ids": rt.NDArray(ids)}))["hidden"].numpy().astype(np.float32)
        rep["textSeconds"] = time.perf_counter() - s
        rep["textHidden"] = {"cosine": cosine(hidden[:, :lt], ref["text_hidden"]),
                             "relL2": rel_l2(hidden[:, :lt], ref["text_hidden"])}
        lyric = np.zeros((1, a.lyric, table.shape[1]), np.float16)
        lyric[0, :ll] = table[lyric_ids]
        mask = np.zeros((1, a.lyric), np.float16)
        mask[0, :ll] = 1
        rep["lyricEmbeds"] = {"relL2": rel_l2(lyric[:, :ll].astype(np.float32), ref["lyric_hidden"])}
        s = time.perf_counter()
        o = await models["condition"][1]({"text_hidden": rt.NDArray(hidden.astype(np.float16)),
                                           "lyric_embeds": rt.NDArray(lyric), "lyric_mask": rt.NDArray(mask)})
        rep["conditionSeconds"] = time.perf_counter() - s
        text_projected = o["text_projected"].numpy().astype(np.float32)[:, :lt]
        lyric_encoded = o["lyric_encoded"].numpy().astype(np.float32)[:, :ll]
        packed = np.concatenate([lyric_encoded, timbre, text_projected], axis=1)
        n = packed.shape[1]
        L = ref["encoder"].shape[1]
        encoder = np.zeros((1, L, packed.shape[2]), np.float32)
        encoder[:, :n] = packed
        encoder_mask = np.zeros((1, L), np.float32)
        encoder_mask[:, :n] = 1
        rep["conditionRows"] = {"ours": n, "reference": int(ref["encoder_mask"].sum())}
        rep["encoder"] = {"cosine": cosine(encoder[:, :n], ref["encoder"][:, :n]),
                          "relL2": rel_l2(encoder[:, :n], ref["encoder"][:, :n]),
                          "maskEqual": bool(np.array_equal(encoder_mask, ref["encoder_mask"].astype(np.float32)))}

        dt = np.float16 if a.dit.endswith("fp16") else np.float32
        static = {"context": rt.NDArray(ref["context"].astype(dt)), "encoder": rt.NDArray(encoder.astype(dt)),
                  "encoder_mask": rt.NDArray(encoder_mask.astype(dt))}
        xt = ref["noise"].astype(np.float32)
        xs, vs, steps = [], [], []
        for i, t in enumerate(SHIFT_TIMESTEPS_3):
            s = time.perf_counter()
            vt = (await models["dit"][1](dict(static, xt=rt.NDArray(xt.astype(dt)), t=rt.NDArray(np.array([t], dt)))))[
                "v"].numpy().astype(np.float32)
            steps.append(time.perf_counter() - s)
            xs.append(xt)
            vs.append(vt)
            xt = xt - vt * t if i == len(SHIFT_TIMESTEPS_3) - 1 else xt - vt * (t - SHIFT_TIMESTEPS_3[i + 1])
        rep["ditSeconds"] = sum(steps)
        latents = xt.transpose(0, 2, 1)                                   # [1, 64, T]
        s = time.perf_counter()
        audio = (await models["vae"][1]({"latents": rt.NDArray(latents.astype(np.float16))}))["audio"].numpy()[0]
        rep["vaeSeconds"] = time.perf_counter() - s
        audio = audio.astype(np.float32)
        np.savez(OUT / f"{path.stem}-e2e.npz", coreai_xt=np.stack(xs), coreai_vt=np.stack(vs), coreai_x0=xt,
                 coreai_audio=audio)
        write_wav(OUT / f"{path.stem}-e2e.wav", audio)

        # Tiled decode: windows of `window` frames stepping by window − overlap; within each overlap the earlier
        # window fades out and the later one fades in linearly. The last window is right-aligned to the end.
        frames = latents.shape[2]
        rep["tiled"] = {}
        for overlap in [int(v) for v in a.overlaps.split(",")]:
            stride = window - overlap
            starts = list(range(0, frames - window + 1, stride))
            if starts[-1] + window < frames:
                starts.append(frames - window)
            out = np.zeros((2, frames * hop), np.float64)
            weight = np.zeros(frames * hop, np.float64)
            s = time.perf_counter()
            for k, start in enumerate(starts):
                piece = (await models["tile"][1]({"latents": rt.NDArray(latents[:, :, start:start + window].astype(
                    np.float16))}))["audio"].numpy()[0].astype(np.float64)
                ramp = np.ones(window * hop)
                lead = (starts[k - 1] + window - start) * hop if k else 0          # samples shared with the previous
                if lead > 0:
                    ramp[:lead] = np.linspace(0, 1, lead + 2)[1:-1]
                trail = (start + window - starts[k + 1]) * hop if k + 1 < len(starts) else 0
                if trail > 0:
                    ramp[-trail:] = np.minimum(ramp[-trail:], np.linspace(1, 0, trail + 2)[1:-1])
                out[:, start * hop:(start + window) * hop] += piece * ramp
                weight[start * hop:(start + window) * hop] += ramp
            tiled = (out / np.maximum(weight, 1e-12)).astype(np.float32)
            rep["tiled"][str(overlap)] = {"windows": len(starts), "seconds": time.perf_counter() - s,
                                          "snrDBVsOneWindow": snr_db(tiled, audio),
                                          "maxAbsDiff": float(np.abs(tiled - audio).max())}
            if overlap == max(int(v) for v in a.overlaps.split(",")):
                write_wav(OUT / f"{path.stem}-e2e-tiled{overlap}.wav", tiled)
        report["prompts"][path.stem] = rep
        print(path.stem, json.dumps(rep), flush=True)
    (OUT / "encoders-run.json").write_text(json.dumps(report, indent=1))


asyncio.run(main())

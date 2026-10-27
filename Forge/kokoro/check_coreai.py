#!/usr/bin/env python3
"""Runs exported Kokoro Core AI graphs from Python and compares them with upstream PyTorch.

A development probe only (run under the shared GPU lock); AetherSpeech's native
qualification test is the SDK acceptance path.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kokoro_graphs import HARMONICS, SAMPLES_PER_FRAME, align, build_kmodel, round_durations, splitmix_normal, tokenize
from upstream_reference import run_upstream


async def run_coreai(durations_asset, decoder_asset, compute, ids, style, seed):
    import coreai.runtime as rt

    if compute == "cpu":
        options = rt.SpecializationOptions.cpu_only()
    elif compute == "gpu":
        options = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    else:
        options = rt.SpecializationOptions.default()
    t0 = time.monotonic()
    dmodel = await rt.AIModel.load(durations_asset, options)
    xmodel = await rt.AIModel.load(decoder_asset, options)
    dfn, xfn = dmodel.load_function("durations"), xmodel.load_function("decode")
    t1 = time.monotonic()
    out = await dfn({"input_ids": rt.NDArray(np.array([ids], np.int32)), "style": rt.NDArray(style.numpy())})
    sums = torch.from_numpy(out["duration"].numpy().copy())
    d = torch.from_numpy(out["prosody"].numpy().copy())
    t_en = torch.from_numpy(out["text"].numpy().copy())
    t2 = time.monotonic()
    print("durations ran", flush=True)
    dur = round_durations(sums, 1.0)
    en, asr = align(d, t_en, dur)
    frames = en.shape[-1]
    noise = splitmix_normal(seed, frames * SAMPLES_PER_FRAME * HARMONICS).reshape(1, frames, SAMPLES_PER_FRAME, HARMONICS)
    t3 = time.monotonic()
    res = await xfn({"prosody": rt.NDArray(en.numpy().copy()), "text": rt.NDArray(asr.numpy().copy()),
                     "style": rt.NDArray(style.numpy()), "noise": rt.NDArray(noise)})
    audio = torch.from_numpy(res["audio"].numpy().copy()).reshape(-1)
    t4 = time.monotonic()
    return sums, dur, audio, {"load_s": t1 - t0, "durations_s": t2 - t1, "decode_s": t4 - t3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--durations", type=Path, required=True)
    ap.add_argument("--decoder", type=Path, required=True)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--compute", choices=["cpu", "gpu", "default"], default="cpu")
    ap.add_argument("--phonemes", default="həlˈO wˈɜɹld, ðɪs ɪz kˈOkəɹO.")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--seed", type=int, default=24000)
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()
    kmodel = build_kmodel(args.upstream, args.snapshot)
    ids = tokenize(kmodel.vocab, args.phonemes)
    style = torch.load(args.snapshot / "voices" / f"{args.voice}.pt", weights_only=True)[len(args.phonemes) - 1]
    ref_audio, ref_dur = run_upstream(kmodel, args.upstream, ids, style, 1.0, args.seed)
    with torch.no_grad():
        ref_sums = torch.sigmoid(kmodel.predictor.duration_proj(kmodel.predictor.lstm(kmodel.predictor.text_encoder(
            kmodel.bert_encoder(kmodel.bert(torch.LongTensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long))).transpose(-1, -2),
            style[:, 128:], torch.tensor([len(ids)]), torch.zeros(1, len(ids), dtype=torch.bool)))[0])).sum(-1)
    for _ in range(args.repeat):
        sums, dur, audio, timing = asyncio.run(run_coreai(args.durations, args.decoder, args.compute, ids, style, args.seed))
    report = {"compute": args.compute, "tokens": len(ids), "frames": int(dur.sum()),
              "duration_sum_max_abs": float((sums - ref_sums).abs().max()),
              "durations_equal": bool(torch.equal(dur, ref_dur)), "timing": timing,
              "audio_seconds": audio.numel() / 24000}
    if audio.numel() == ref_audio.numel():
        err = audio - ref_audio
        report.update(max_abs=float(err.abs().max()), rel_l2=float(err.norm() / ref_audio.norm()),
                      finite=bool(torch.isfinite(audio).all()))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

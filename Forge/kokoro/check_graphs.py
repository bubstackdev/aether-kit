#!/usr/bin/env python3
"""Eager CPU check: export graphs + host steps versus the upstream KModel."""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kokoro_graphs import (DecoderGraph, DurationsGraph, SAMPLES_PER_FRAME, HARMONICS, align, build_kmodel,
                           round_durations, splitmix_normal, tokenize)
from upstream_reference import run_upstream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--phonemes", default="həlˈO wˈɜɹld, ðɪs ɪz kˈOkəɹO.")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--seed", type=int, default=24000)
    args = ap.parse_args()
    torch.manual_seed(0)
    reference = build_kmodel(args.upstream, args.snapshot)
    export_copy = build_kmodel(args.upstream, args.snapshot)
    ids = tokenize(reference.vocab, args.phonemes)
    pack = torch.load(args.snapshot / "voices" / f"{args.voice}.pt", weights_only=True)
    ref_s = pack[len(args.phonemes) - 1]
    audio_ref, dur_ref = run_upstream(reference, args.upstream, ids, ref_s, 1.0, args.seed)
    audio_ref2, _ = run_upstream(reference, args.upstream, ids, ref_s, 1.0, args.seed)

    durations = DurationsGraph(export_copy).eval()
    decoder = DecoderGraph(export_copy).eval()
    with torch.no_grad():
        sums, d, t_en = durations(torch.tensor([ids], dtype=torch.int32), ref_s)
        dur = round_durations(sums, 1.0)
        en, asr = align(d, t_en, dur)
        frames = en.shape[-1]
        noise = torch.from_numpy(splitmix_normal(args.seed, frames * SAMPLES_PER_FRAME * HARMONICS))
        audio = decoder(en, asr, ref_s, noise.reshape(1, frames, SAMPLES_PER_FRAME, HARMONICS)).reshape(-1)
    report = {
        "tokens": len(ids), "frames": frames,
        "durations_equal": bool(torch.equal(dur, dur_ref)),
        "upstream_repeat_max_abs": float((audio_ref - audio_ref2).abs().max()),
        "samples": [audio.numel(), audio_ref.numel()],
    }
    if audio.numel() == audio_ref.numel():
        err = audio - audio_ref
        report["max_abs"] = float(err.abs().max())
        report["rel_l2"] = float(err.norm() / audio_ref.norm())
        report["ref_rms"] = float(audio_ref.pow(2).mean().sqrt())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

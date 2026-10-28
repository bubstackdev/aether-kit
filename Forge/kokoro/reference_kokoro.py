#!/usr/bin/env python3
"""Generates independent upstream-PyTorch Kokoro references for native qualification.

Runs the unmodified kokoro 0.9.4 KModel on CPU (float32) for fixed phoneme inputs with
seeded SineGen noise (see upstream_reference.py). For each case it writes:

    <case>/case.json       inputs, frames, duration sums, hashes, sensitivity floor
    <case>/audio.f32       upstream waveform, little-endian float32
    <case>/perturbed.f32   upstream waveform with a 1e-6 relative perturbation of the
                           harmonic source, establishing the model's intrinsic sensitivity

and an index `references.json`. No G2P runs: phonemes are fixed strings in Kokoro's
alphabet. Writes only the new output directory. CPU-only; no GPU lock needed.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kokoro_graphs import (HARMONICS, SAMPLES_PER_FRAME, DecoderGraph, DurationsGraph, align, build_kmodel,
                           round_durations, splitmix_normal, tokenize)
from upstream_reference import run_upstream
import speech_metrics

CASES = [
    {"name": "short-af_heart", "voice": "af_heart", "speed": 1.0, "seed": 24000,
     "phonemes": "həlˈO wˈɜɹld, ðɪs ɪz kˈOkəɹO."},
    {"name": "sentence-am_michael", "voice": "am_michael", "speed": 1.0, "seed": 7,
     "phonemes": "ðə kwˈɪk bɹˈWn fˈɑks ʤˈʌmps ˌOvəɹ ðə lˈAzi dˈɔɡ, ənd ðˈɛn ɪt ɹˈʌnz əwˈA ˌɪntu ðə fˈɔɹəst."},
    {"name": "speed-bf_emma", "voice": "bf_emma", "speed": 1.2, "seed": 99,
     "phonemes": "ɪt wəz ðə bˈɛst ɒv tˈImz, ɪt wəz ðə wˈɜːst ɒv tˈImz; ɪt wəz ðɪ ˈAʤ ɒv wˈɪzdəm."},
    {"name": "long-af_bella", "voice": "af_bella", "speed": 0.9, "seed": 2026,
     "phonemes": ("kˈOkəɹO ɪz ən ˈOpən wˈAt tˌitˌiˈɛs mˈɑdᵊl wɪð ˈATi tˈu mˈɪljən pəɹˈæməTəɹz. "
                  "dəspˈIt ɪts lˈItwˌAt ˈɑɹkətˌɛkʧəɹ, ɪt dəlˈɪvəɹz kˈɑmpəɹəbᵊl kwˈɑləTi tə lˈɑɹʤəɹ mˈɑdᵊlz "
                  "wˌIl bˈiɪŋ səɡnˈɪfəkəntli fˈæstəɹ ənd mˈɔɹ kˈɔst əfˈɪʃənt. wɪð əpˈæʧi lˈIsᵊnst wˈAts, "
                  "kˈOkəɹO kæn bi dəplˈYd ˈɛniwˌɛɹ fɹʌm pɹədˈʌkʃən ənvˈIɹənmənts tə pˈɜɹsᵊnᵊl pɹˈɑʤɛkts.")},
    # Lands in bucket 512 (257-512 frames), the iOS frame cap, which the other cases skip.
    {"name": "medium-af_bella", "voice": "af_bella", "speed": 1.0, "seed": 512,
     "phonemes": ("kˈOkəɹO ɪz ən ˈOpən wˈAt tˌitˌiˈɛs mˈɑdᵊl wɪð ˈATi tˈu mˈɪljən pəɹˈæməTəɹz. "
                  "dəspˈIt ɪts lˈItwˌAt ˈɑɹkətˌɛkʧəɹ, ɪt dəlˈɪvəɹz kˈɑmpəɹəbᵊl kwˈɑləTi tə lˈɑɹʤəɹ mˈɑdᵊlz.")},
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=Path)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--upstream", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    torch.manual_seed(0)
    reference = build_kmodel(args.upstream, args.snapshot)
    reference64 = build_kmodel(args.upstream, args.snapshot).double()
    reference64.decoder.generator.stft.window = reference64.decoder.generator.stft.window.double()
    graphs_model = build_kmodel(args.upstream, args.snapshot)
    durations_graph, decoder_graph = DurationsGraph(graphs_model).eval(), DecoderGraph(graphs_model).eval()
    args.output.mkdir(parents=True)
    index = {"schema": 1, "generator": "Forge/kokoro/reference_kokoro.py",
             "scripts": {p.name: sha256(p.read_bytes()) for p in sorted(Path(__file__).resolve().parent.glob("*.py"))},
             "upstream": {"package": "kokoro 0.9.4", "wheel_sha256": "a129dc6364a286bd6a92c396e9862459d3d3e45f2c15596ed5a94dcee5789efd",
                          "checkpoint_sha256": hashlib.sha256((args.snapshot / "kokoro-v1_0.pth").read_bytes()).hexdigest(),
                          "revision": args.snapshot.name},
             "tools": {"torch": torch.__version__, "python": platform.python_version(), "device": "cpu", "dtype": "float32"},
             "metrics": speech_metrics.DEFINITION, "cases": []}
    for case in CASES:
        ids = tokenize(reference.vocab, case["phonemes"])
        pack = torch.load(args.snapshot / "voices" / f"{case['voice']}.pt", weights_only=True)
        style = pack[len(case["phonemes"]) - 1]
        t0 = time.monotonic()
        audio, pred_dur = run_upstream(reference, args.upstream, ids, style, case["speed"], case["seed"])
        upstream_s = time.monotonic() - t0
        repeat, _ = run_upstream(reference, args.upstream, ids, style, case["speed"], case["seed"])
        if not torch.equal(audio, repeat):
            raise SystemExit(f"upstream reference is not deterministic for {case['name']}")
        # Intrinsic sensitivity: perturb the harmonic source by 1e-6 relative noise.
        source = reference.decoder.generator.m_source
        original = source.forward
        gen = torch.Generator().manual_seed(1)

        def perturbed(x):
            s, n, uv = original(x)
            return s * (1 + 1e-6 * torch.randn(s.shape, generator=gen)), n, uv

        source.forward = perturbed
        try:
            perturbed_audio, _ = run_upstream(reference, args.upstream, ids, style, case["speed"], case["seed"])
        finally:
            source.forward = original
        # Precision floor: the same upstream code in float64 (same noise, same frames).
        torch.set_default_dtype(torch.float64)  # upstream allocates its alignment matrix with the default dtype
        try:
            audio64, dur64 = run_upstream(reference64, args.upstream, ids, style.double(), case["speed"], case["seed"])
        finally:
            torch.set_default_dtype(torch.float32)
        with torch.no_grad():
            sums, d, t_en = durations_graph(torch.tensor([ids], dtype=torch.int32), style)
            frames = round_durations(sums, case["speed"])
            en, asr = align(d, t_en, frames)
            noise = torch.from_numpy(splitmix_normal(case["seed"], int(frames.sum()) * SAMPLES_PER_FRAME * HARMONICS))
            graph_audio = decoder_graph(en, asr, style, noise.reshape(1, -1, SAMPLES_PER_FRAME, HARMONICS)).reshape(-1)
        a = audio.numpy().astype("<f4")
        p = perturbed_audio.numpy().astype("<f4")
        directory = args.output / case["name"]
        directory.mkdir()
        (directory / "audio.f32").write_bytes(a.tobytes())
        (directory / "perturbed.f32").write_bytes(p.tobytes())
        record = {
            **case, "input_ids": ids, "frames": pred_dur.tolist(), "duration_sums": sums.reshape(-1).tolist(),
            "frames_match_export_graph": bool(torch.equal(frames, pred_dur)),
            "samples": int(a.size), "seconds": a.size / 24000,
            "audio_sha256": sha256(a.tobytes()), "perturbed_sha256": sha256(p.tobytes()),
            "upstream_cpu_seconds": round(upstream_s, 3),
            "sensitivity_floor": speech_metrics.compare(p, a),
            "float64_upstream": (speech_metrics.compare(a, audio64.numpy()) if torch.equal(dur64, pred_dur) else "frames differ"),
            "eager_export_graph": speech_metrics.compare(graph_audio.numpy(), a) if graph_audio.numel() == a.size else None,
        }
        (directory / "case.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        index["cases"].append({"name": case["name"], "case_sha256": sha256((directory / "case.json").read_bytes())})
        print(json.dumps({k: record[k] for k in ("name", "samples", "frames_match_export_graph", "sensitivity_floor", "float64_upstream", "eager_export_graph")}))
    (args.output / "references.json").write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    main()

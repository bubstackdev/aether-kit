#!/usr/bin/env python3
"""Eager CPU self-test of the export graphs, before any Core AI conversion.

    Forge/.venv/bin/python -I Forge/kokoro/selftest_kokoro_graphs.py --reference <reference dir> [--report <json>]

For each reference case: the durations graph's frames equal upstream's exactly, and the bucketed decoder padded to its
bucket stays within the gate's bars against the upstream audio (check_kokoro.py's bars). At bucket edges (lengths
63/64/65 and 127/128/129, cut from the longest case): an utterance that exactly fills its bucket decodes bit-identically
padded and unpadded, and the others differ only by float32 reduction order (recorded, as in 0.0.x: 0.4-1.8%). Exits
nonzero when a check fails. CPU only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import speech_metrics  # noqa: E402
from check_kokoro import FLOOR_MULTIPLE, MAX_LOG_MEL, MAX_REL_L2  # noqa: E402
from kokoro_graphs import (DEFAULT_SNAPSHOT, DEFAULT_UPSTREAM, HARMONICS, SAMPLES_PER_FRAME,  # noqa: E402
                           BucketedDecoderGraph, DurationsGraph, align, build_kmodel, round_durations,
                           splitmix_normal, tokenize, verify_sources)
from export_kokoro import DECODER_BUCKETS  # noqa: E402


def decode(graph, en, asr, style, seed, frames, bucket):
    pad = bucket - frames
    en = torch.nn.functional.pad(en[..., :frames], (0, pad))
    asr = torch.nn.functional.pad(asr[..., :frames], (0, pad))
    noise = torch.from_numpy(splitmix_normal(seed, bucket * SAMPLES_PER_FRAME * HARMONICS)).reshape(
        1, bucket, SAMPLES_PER_FRAME, HARMONICS)
    with torch.no_grad():
        out = graph(en, asr, style, noise, torch.tensor([frames], dtype=torch.int32)).reshape(-1)
    return out[:frames * SAMPLES_PER_FRAME].numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    ap.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    verify_sources(args.snapshot, args.upstream)
    torch.manual_seed(0)
    kmodel = build_kmodel(args.upstream, args.snapshot)
    durations = DurationsGraph(kmodel).eval()
    decoder = BucketedDecoderGraph(build_kmodel(args.upstream, args.snapshot)).eval()
    bucket_for = lambda f: next(b for b in DECODER_BUCKETS if b >= f)  # noqa: E731
    index = json.loads((args.reference / "references.json").read_text())
    results, failures, aligned = {"cases": [], "edges": []}, [], {}
    for entry in index["cases"]:
        case = json.loads((args.reference / entry["name"] / "case.json").read_text())
        expected = np.frombuffer((args.reference / entry["name"] / "audio.f32").read_bytes(), dtype="<f4")
        style = torch.load(args.snapshot / "voices" / f"{case['voice']}.pt", weights_only=True)[len(case["phonemes"]) - 1]
        ids = tokenize(kmodel.vocab, case["phonemes"])
        with torch.no_grad():
            sums, d, t_en = durations(torch.tensor([ids], dtype=torch.int32), style)
            per_token = round_durations(sums, case["speed"])
            en, asr = align(d, t_en, per_token)
        frames = en.shape[-1]
        aligned[entry["name"]] = (en, asr, style, case["seed"])
        padded = decode(decoder, en, asr, style, case["seed"], frames, bucket_for(frames))
        value = speech_metrics.compare(padded, expected)
        bound = min(MAX_REL_L2, FLOOR_MULTIPLE * case["float64_upstream"]["rel_l2"])
        ok = per_token.tolist() == case["frames"] and value["rel_l2"] <= bound and value["log_mel_mean_abs"] <= MAX_LOG_MEL
        results["cases"].append({"name": entry["name"], "frames": frames, "bucket": bucket_for(frames),
                                 "framesExact": per_token.tolist() == case["frames"], "vsUpstream": value,
                                 "relL2Bound": bound, "passed": ok})
        if not ok:
            failures.append(entry["name"])
        print(json.dumps({"name": entry["name"], "rel_l2": round(value["rel_l2"], 5), "passed": ok}), flush=True)
    longest = max(aligned.values(), key=lambda a: a[0].shape[-1])
    en, asr, style, seed = longest
    for edge in (64, 128):
        for frames in (edge - 1, edge, edge + 1):
            unpadded = decode(decoder, en, asr, style, seed, frames, frames)
            padded = decode(decoder, en, asr, style, seed, frames, bucket_for(frames))
            identical = bool(np.array_equal(padded, unpadded))
            record = {"frames": frames, "bucket": bucket_for(frames), "identical": identical,
                      "paddedVsUnpadded": speech_metrics.compare(padded, unpadded)}
            results["edges"].append(record)
            if frames == bucket_for(frames) and not identical:
                failures.append(f"edge {frames}: a full bucket must decode bit-identically")
            print(json.dumps({"frames": frames, "bucket": record["bucket"], "identical": identical,
                              "rel_l2": round(record["paddedVsUnpadded"]["rel_l2"], 5)}), flush=True)
    results["failures"] = failures
    if args.report:
        args.report.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"failures": failures}))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

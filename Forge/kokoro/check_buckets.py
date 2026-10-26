#!/usr/bin/env python3
"""Eager CPU check of the bucketed decoder: padded-to-bucket runs versus unpadded runs.

For each reference case, and for frame counts at and next to bucket edges (built from a
case's aligned features), runs BucketedDecoderGraph unpadded (F_b = frames) and padded to the
next bucket, and compares the valid prefix. Also compares the unpadded bucketed graph with the
original DecoderGraph and the upstream reference audio. CPU only.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kokoro_graphs import (HARMONICS, SAMPLES_PER_FRAME, BucketedDecoderGraph, DecoderGraph, DurationsGraph, align,
                           build_kmodel, round_durations, splitmix_normal, tokenize)
import speech_metrics


def decode(graph, en, asr, style, seed, frames, bucket):
    pad = bucket - frames
    en = torch.nn.functional.pad(en[..., :frames], (0, pad))
    asr = torch.nn.functional.pad(asr[..., :frames], (0, pad))
    noise = torch.from_numpy(splitmix_normal(seed, bucket * SAMPLES_PER_FRAME * HARMONICS)).reshape(1, bucket, SAMPLES_PER_FRAME, HARMONICS)
    with torch.no_grad():
        out = graph(en, asr, style, noise, torch.tensor([frames], dtype=torch.int32)).reshape(-1)
    return out[:frames * SAMPLES_PER_FRAME].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--references", type=Path, required=True)
    ap.add_argument("--buckets", default="64,96,128,192,256,384,512,768,1024,1536,2048,3072,4096")
    ap.add_argument("--edges", default="64,128,256,512")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    buckets = [int(b) for b in args.buckets.split(",")]

    def bucket_for(f):
        return next(b for b in buckets if b >= f)

    kmodel = build_kmodel(args.upstream, args.snapshot)
    durations = DurationsGraph(kmodel).eval()
    plain = DecoderGraph(build_kmodel(args.upstream, args.snapshot)).eval()
    bucketed = BucketedDecoderGraph(build_kmodel(args.upstream, args.snapshot)).eval()
    index = json.loads((args.references / "references.json").read_text())
    results = {"buckets": buckets, "cases": [], "edges": []}
    aligned = {}
    for entry in index["cases"]:
        case = json.loads((args.references / entry["name"] / "case.json").read_text())
        reference = np.frombuffer((args.references / entry["name"] / "audio.f32").read_bytes(), dtype="<f4")
        pack = torch.load(args.snapshot / "voices" / f"{case['voice']}.pt", weights_only=True)
        style = pack[len(case["phonemes"]) - 1]
        ids = tokenize(kmodel.vocab, case["phonemes"])
        with torch.no_grad():
            sums, d, t_en = durations(torch.tensor([ids], dtype=torch.int32), style)
            en, asr = align(d, t_en, round_durations(sums, case["speed"]))
            noise = torch.from_numpy(splitmix_normal(case["seed"], en.shape[-1] * SAMPLES_PER_FRAME * HARMONICS))
            original = plain(en, asr, style, noise.reshape(1, -1, SAMPLES_PER_FRAME, HARMONICS)).reshape(-1).numpy()
        frames = en.shape[-1]
        aligned[entry["name"]] = (en, asr, style, case["seed"])
        unpadded = decode(bucketed, en, asr, style, case["seed"], frames, frames)
        padded = decode(bucketed, en, asr, style, case["seed"], frames, bucket_for(frames))
        record = {"name": entry["name"], "frames": frames, "bucket": bucket_for(frames),
                  "padded_vs_unpadded": speech_metrics.compare(padded, unpadded),
                  "padded_identical": bool(np.array_equal(padded, unpadded)),
                  "unpadded_vs_original_graph": speech_metrics.compare(unpadded, original),
                  "padded_vs_upstream": speech_metrics.compare(padded, reference),
                  "original_graph_vs_upstream": speech_metrics.compare(original, reference)}
        results["cases"].append(record)
        print(json.dumps({k: (round(v["rel_l2"], 6) if isinstance(v, dict) else v) for k, v in record.items()}), flush=True)
    en, asr, style, seed = aligned["long-af_bella"]  # 913 frames: long enough for every edge below 913
    for edge in [int(e) for e in args.edges.split(",")]:
        for frames in (edge - 1, edge, edge + 1):
            if frames > en.shape[-1] or frames < 8:
                continue
            unpadded = decode(bucketed, en, asr, style, seed, frames, frames)
            padded = decode(bucketed, en, asr, style, seed, frames, bucket_for(frames))
            record = {"frames": frames, "bucket": bucket_for(frames),
                      "padded_vs_unpadded": speech_metrics.compare(padded, unpadded),
                      "padded_identical": bool(np.array_equal(padded, unpadded))}
            results["edges"].append(record)
            print(json.dumps({"frames": frames, "bucket": record["bucket"], "rel_l2": record["padded_vs_unpadded"]["rel_l2"],
                              "log_mel": record["padded_vs_unpadded"]["log_mel_mean_abs"], "identical": record["padded_identical"]}), flush=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()

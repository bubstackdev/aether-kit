#!/usr/bin/env python3
"""Export Kokoro-82M (hexgrad/Kokoro-82M, Apache-2.0) to a float32 Core AI speech bundle.

Runs in Forge's locked environment (`uv sync --frozen --project Forge`) with
the pinned Hugging Face snapshot and the unpacked kokoro 0.9.4 wheel. No G2P code is
imported. Writes only the explicitly supplied new output directory:

    OUTPUT/
      kokoro.json              execution contract read by AetherSpeech
      durations.aimodel/       function "durations"
      decoder.aimodel/         function "decode"
      voices/<id>.f32          510 x 256 little-endian float32 style rows
      export-report.json       provenance, eager self-check, conversion timings

Run Core AI conversion under the shared GPU lock (it is CPU-bound but memory heavy).
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kokoro_graphs import (HARMONICS, MAX_PHONEMES, SAMPLE_RATE, SAMPLES_PER_FRAME, BucketedDecoderGraph, DecoderGraph, DurationsGraph,
                           reduce_decoder_precision,
                           align, build_kmodel, round_durations, splitmix_normal, tokenize)

REPO = "hexgrad/Kokoro-82M"
REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
CHECKPOINT_SHA256 = "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4"
MAX_FRAMES = 4096  # ~102 s of audio per utterance; longer text must be chunked by the caller
# Decoder frame buckets: each specializes once on the GPU; utterances pad to the next bucket.
# Powers of two bound padding to < 2x decode time and keep the specialization set small (see
# docs/kokoro-speech.md#length-buckets for the measured compile, latency and memory trade-off).
DECODER_BUCKETS = [64, 128, 256, 512, 1024, 2048, 4096]
LANGUAGES = {"a": "en-US", "b": "en-GB", "e": "es", "f": "fr", "h": "hi", "i": "it", "j": "ja", "p": "pt-BR", "z": "zh"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_hashes(root: Path) -> dict:
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def convert(module, example, dynamic, inputs, outputs, entry, path: Path, ct):
    t0 = time.monotonic()
    with torch.no_grad():
        ep = torch.export.export(module, example, dynamic_shapes=dynamic, strict=False)
        ep = ep.run_decompositions(ct.get_decomp_table())
    t1 = time.monotonic()
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    converter.add_exported_program(ep, input_names=inputs, output_names=outputs, state_names=[], entrypoint_name=entry)
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(path)
    return {"export_s": round(t1 - t0, 2), "convert_s": round(time.monotonic() - t1, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=Path)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--upstream", type=Path, required=True, help="unpacked kokoro-0.9.4 wheel")
    ap.add_argument("--only", choices=["durations", "decoder"], default=None, help="probe a single graph")
    ap.add_argument("--decoder", choices=["bucketed", "plain"], default="bucketed",
                    help="bucketed adds a `frames` input so inputs can be padded to fixed frame buckets")
    ap.add_argument("--decoder-precision", choices=["float32", "float16", "float16-storage", "int8"], default="float32",
                    help="EXPERIMENTAL decoder weights: float16 (with float16 compute), float16-storage or weight-only "
                         "int8. These fail qualification, and AetherSpeech refuses to open such bundles")
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    if sha256(args.snapshot / "kokoro-v1_0.pth") != CHECKPOINT_SHA256:
        raise SystemExit("checkpoint hash differs from the pinned Kokoro v1.0 release")
    import coreai_torch as ct

    torch.manual_seed(0)
    kmodel = build_kmodel(args.upstream, args.snapshot)
    config = json.loads((args.snapshot / "config.json").read_text())
    durations = DurationsGraph(kmodel).eval()
    bucketed = args.decoder == "bucketed"
    decoder = (BucketedDecoderGraph(kmodel) if bucketed else DecoderGraph(kmodel)).eval()

    # Example shapes for tracing (dynamic dimensions are declared below).
    phonemes = "həlˈO wˈɜɹld, ðɪs ɪz kˈOkəɹO."
    ids = tokenize(config["vocab"], phonemes)
    pack = torch.load(args.snapshot / "voices" / "af_heart.pt", weights_only=True)
    style = pack[len(phonemes) - 1]
    with torch.no_grad():
        sums, d, t_en = durations(torch.tensor([ids], dtype=torch.int32), style)
        en, asr = align(d, t_en, round_durations(sums, 1.0))
    frames = en.shape[-1]
    noise = torch.from_numpy(splitmix_normal(1, frames * SAMPLES_PER_FRAME * HARMONICS)).reshape(
        1, frames, SAMPLES_PER_FRAME, HARMONICS)

    args.output.mkdir(parents=True)
    report = {"schema": 1, "source": {"repository": REPO, "revision": REVISION, "checkpoint_sha256": CHECKPOINT_SHA256,
                                      "license": "Apache-2.0"},
              "tools": {"torch": torch.__version__, "coreai-torch": importlib.metadata.version("coreai-torch"),
                        "python": platform.python_version()},
              "scripts": {p.name: sha256(p) for p in sorted(Path(__file__).resolve().parent.glob("*.py"))},
              "conversion": {}}
    tokens = torch.export.Dim("tokens", min=3, max=MAX_PHONEMES + 2)
    frame_dim = torch.export.Dim("frames", min=2, max=MAX_FRAMES)
    if args.only in (None, "durations"):
        report["conversion"]["durations"] = convert(
            durations, (torch.tensor([ids], dtype=torch.int32), style), ({1: tokens}, None),
            ["input_ids", "style"], ["duration", "prosody", "text"], "durations",
            args.output / "durations.aimodel", ct)
    if args.only in (None, "decoder"):
        example = (en, asr, style, noise) + ((torch.tensor([frames], dtype=torch.int32),) if bucketed else ())
        decoder = reduce_decoder_precision(decoder, args.decoder_precision, example)
        if bucketed:
            valid = torch.tensor([frames], dtype=torch.int32)
            report["conversion"]["decoder"] = convert(
                decoder, (en, asr, style, noise, valid), ({2: frame_dim}, {2: frame_dim}, None, {1: frame_dim}, None),
                ["prosody", "text", "style", "noise", "frames"], ["audio"], "decode", args.output / "decoder.aimodel", ct)
        else:
            report["conversion"]["decoder"] = convert(
                decoder, (en, asr, style, noise), ({2: frame_dim}, {2: frame_dim}, None, {1: frame_dim}),
                ["prosody", "text", "style", "noise"], ["audio"], "decode", args.output / "decoder.aimodel", ct)
    if args.only is None:
        (args.output / "voices").mkdir()
        voices = []
        for path in sorted((args.snapshot / "voices").glob("*.pt")):
            pack = torch.load(path, weights_only=True)
            if tuple(pack.shape) != (MAX_PHONEMES, 1, 256):
                raise SystemExit(f"unexpected voice shape {tuple(pack.shape)} for {path.name}")
            out = args.output / "voices" / (path.stem + ".f32")
            out.write_bytes(pack.reshape(MAX_PHONEMES, 256).to(torch.float32).numpy().astype("<f4").tobytes())
            voices.append({"id": path.stem, "language": LANGUAGES[path.stem[0]],
                           "gender": {"f": "female", "m": "male"}[path.stem[1]],
                           "file": f"voices/{out.name}", "sha256": sha256(out), "source_sha256": sha256(path)})
        contract = {
            "schema": 1, "family": "kokoro-82m", "precision": args.decoder_precision,
            "source": report["source"],
            "sampleRate": SAMPLE_RATE, "samplesPerFrame": SAMPLES_PER_FRAME, "harmonics": HARMONICS,
            "maximumPhonemes": MAX_PHONEMES, "maximumFrames": MAX_FRAMES, "styleDimension": 256,
            **({"decoderBuckets": DECODER_BUCKETS} if bucketed else {}),
            **({"decoderPrecision": args.decoder_precision} if args.decoder_precision != "float32" else {}),
            "boundaryToken": 0, "vocabulary": config["vocab"],
            "models": {
                "durations": {"path": "durations.aimodel", "function": "durations",
                              "sha256": tree_hashes(args.output / "durations.aimodel")},
                "decoder": {"path": "decoder.aimodel", "function": "decode",
                            "sha256": tree_hashes(args.output / "decoder.aimodel")},
            },
            "voices": voices,
        }
        (args.output / "kokoro.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n")
    (args.output / "export-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["conversion"]))


if __name__ == "__main__":
    main()

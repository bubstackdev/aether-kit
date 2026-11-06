#!/usr/bin/env python3
"""Export Kokoro-82M (hexgrad/Kokoro-82M, Apache-2.0) as an Aether-Kit `kokoro@1` speech bundle.

    uv sync --frozen --project Forge
    Forge/.venv/bin/python -I Forge/kokoro/fetch_upstream.py          # the pinned kokoro 0.9.4 model code, once
    Forge/.venv/bin/python -I Forge/kokoro/export_kokoro.py --output <new bundle dir>
    Forge/.venv/bin/python -I Forge/kokoro/export_kokoro.py --sample-manifest   # the Swift decoding fixture

Nothing is downloaded: the snapshot must be in the Hugging Face cache (or `--snapshot`), and every file read is
verified against `source-pins.json`. No grapheme-to-phoneme code is imported (see kokoro_graphs.py). Conversion runs on
the CPU, in float32. The output directory is laid out as an installed revision root:

    OUTPUT/
      .aether/aether.json               the Aether manifest (MANIFEST.md §5.8, driver `kokoro@1`, proposed)
      kokoro_82m-durations.aimodel      token ids + style -> per-token duration sums, prosody and text features
      kokoro_82m-decoder.aimodel        aligned features + style + noise + valid frame count -> 24 kHz audio
      voices/<id>.f32                   the 54 v1.0 voice packs, 510 x 256 little-endian float32
      LICENSE, NOTICE                   Apache-2.0 and the modification notice
      SHA256SUMS                        every bundle file except itself and the log
      export-report.json                provenance, versions, timings, peak RSS
      forge-log.jsonl                   one event per stage

Parity: `reference_kokoro.py` (the unmodified upstream KModel on the CPU) records the references; `check_kokoro.py`
runs the exported graphs through Core AI with the host recipe and gates them against those references.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import importlib.metadata
import json
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kokoro_graphs as kg  # noqa: E402

ID = "kokoro-82m"
NAME = "kokoro_82m"
ASSETS = {"durations": f"{NAME}-durations.aimodel", "decoder": f"{NAME}-decoder.aimodel"}
NOT_BUNDLE = {".aether/aether.json", "SHA256SUMS", "export-report.json", "forge-log.jsonl"}
MAX_FRAMES = 4096   # about 102 s per utterance; longer text is split by the caller
# Decoder frame buckets: each specializes once on the GPU and utterances pad to the smallest that fits. Powers of two
# bound padding to under 2x and keep the set small; Metal memory per bucket grows faster than the bucket (0.0.x: 0.3,
# 0.4, 0.6, 1.6, 2.3 GB for 64...1024, 8.3 GB for 2048), so the default caps are 1024 frames on macOS, 512 on iOS.
DECODER_BUCKETS = [64, 128, 256, 512, 1024, 2048, 4096]
FRAME_CAP = {"macos": 1024, "ios": 512}
LANGUAGES = {"a": "en-US", "b": "en-GB", "e": "es", "f": "fr", "h": "hi", "i": "it", "j": "ja", "p": "pt-BR", "z": "zh"}
GENDERS = {"f": "female", "m": "male"}
# Placement. The decoder runs on the GPU without the reshape hint (0.0.x: with it, CoreAIRuntime aborted) and specializes
# once per bucket. The durations graph runs on the CPU, where its dynamic token axis takes the hint and no per-length
# GPU specialization happens: it is about 30 ms per utterance there and its frames are the ones qualified. (0.0.x saw
# it abort on the GPU, macOS 27.0.1; on macOS 27.2 `check_kokoro.py --compute durations=gpu` passes the same gate, with
# duration sums within 1e-5 of the CPU's, so the GPU is an option once its per-length specialization is measured.)
COMPUTE = {"durations": ("cpu", True), "decoder": ("gpu", False)}


class Log:
    def __init__(self, path: Path) -> None:
        self.path, self.started = path, {}

    def event(self, stage: str, message: str, **fields) -> None:
        record = {"message": message, "stage": stage, "t": round(time.time(), 3), **fields}
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"[kokoro:{stage}] {message}" + (f" {fields}" if fields else ""), file=sys.stderr, flush=True)

    def start(self, stage: str) -> None:
        self.started[stage] = time.monotonic()
        self.event(stage, "start")

    def done(self, stage: str, **fields) -> float:
        seconds = round(time.monotonic() - self.started.pop(stage), 2)
        self.event(stage, "done", seconds=seconds, peakRSSGB=peak_rss_gb(), **fields)
        return seconds


def peak_rss_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)    # macOS reports bytes


def tree_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def tree_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in tree_files(root))


def convert(module, example: tuple, dynamic, inputs: list[str], outputs: list[str], entry: str, path: Path) -> dict:
    import coreai_torch as ct

    t0 = time.monotonic()
    with torch.no_grad():
        ep = torch.export.export(module, example, dynamic_shapes=dynamic, strict=False)
        ep = ep.run_decompositions(ct.get_decomp_table())
    t1 = time.monotonic()
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    converter.add_exported_program(ep, input_names=inputs, output_names=outputs, state_names=[], entrypoint_name=entry)
    del ep
    program = converter.to_coreai()
    del converter
    program.optimize()
    program.save_asset(path)
    return {"exportSeconds": round(t1 - t0, 1), "convertSeconds": round(time.monotonic() - t1, 1),
            "bytes": tree_bytes(path), "inputs": inputs, "outputs": outputs, "function": entry}


# MARK: the manifest (MANIFEST.md §5.8 `kokoro@1`, proposed until G5)

def voice_entry(voice: str) -> dict:
    return {"table": f"voice.{voice}", "language": LANGUAGES[voice[0]], "gender": GENDERS[voice[1]]}


def manifest(files: dict[str, dict], vocabulary: dict[str, int], voices: list[str], report: dict) -> dict:
    graph = lambda role, function, inputs, outputs: {  # noqa: E731
        "file": f"main:{ASSETS[role]}", "function": function, "inputs": inputs, "outputs": outputs,
        "compute": COMPUTE[role][0], "expectFrequentReshapes": COMPUTE[role][1]}
    return {
        "schema": 1,
        "id": ID,
        "task": "text-to-speech",
        "driver": "kokoro@1",
        "x-status": "draft: driver kokoro@1 and its kokoro block are proposed in MANIFEST.md §5.8 (G5)",
        "variant": {"platform": "macos"},
        "assets": {"main": {"subtree": "", "files": [{"path": p, **meta} for p, meta in sorted(files.items())]}},
        "graphs": {
            "durations": graph("durations", "durations", {"tokens": "input_ids", "style": "style"},
                               {"durations": "duration", "prosody": "prosody", "text": "text"}),
            "decoder": graph("decoder", "decode",
                             {"prosody": "prosody", "text": "text", "style": "style", "noise": "noise",
                              "frames": "frames"},
                             {"audio": "audio"}),
        },
        "tables": {f"voice.{v}": {"asset": "main", "file": f"voices/{v}.f32", "dtype": "float32",
                                  "shape": [kg.MAX_PHONEMES, 256], "layout": "rowMajor littleEndian"} for v in voices},
        "kokoro": {
            "sampleRate": kg.SAMPLE_RATE, "samplesPerFrame": kg.SAMPLES_PER_FRAME, "harmonics": kg.HARMONICS,
            "phonemes": {"vocabulary": vocabulary, "boundaryToken": 0, "maxPhonemes": kg.MAX_PHONEMES,
                         "normalization": "none", "unknown": "refuse"},
            "style": {"dimension": 256, "row": "phonemeCount - 1", "decoder": "style[0..<128]",
                      "prosody": "style[128..<256]"},
            "durations": {"frames": "max(1, roundHalfToEven(duration / speed))", "alignment": "repeat"},
            "frames": {"buckets": DECODER_BUCKETS, "maxFrames": MAX_FRAMES, "cap": FRAME_CAP,
                       "padding": "zeros to the bucket; `frames` is the valid count; audio past 600*frames is dropped"},
            "noise": {"kind": "splitmix64BoxMuller", "shape": "[1, bucketFrames, 600, 9]", "order": "rowMajor",
                      "dtype": "float32"},
            "voices": {v: voice_entry(v) for v in voices},
        },
        "precision": {"weights": "fp32", "activations": "fp32",
                      "components": {"decoder": "float16 and int8 decoders failed 0.0.x qualification; not offered"}},
        "license": "Apache-2.0",
        "x-license": {"files": ["LICENSE", "NOTICE"], "modified": True},
        "provenance": {
            "source": kg.REPO, "sourceRevision": kg.REVISION,
            "producer": "aether-kit Forge/kokoro/export_kokoro.py",
            "exportScript": report["scriptSHA256"]["export_kokoro.py"],
            "refs": {"upstreamCode": "kokoro 0.9.4", "upstreamWheelSHA256": kg.UPSTREAM_WHEEL_SHA256,
                     "forgeLockSHA256": report["forgeLockSHA256"], "exportReport": "export-report.json"},
        },
    }


def sample_manifest() -> dict:
    """`manifest()` for placeholder graph files and report values, with the real vocabulary and voices (from the pinned
    config). Regenerate the Swift fixture with `--sample-manifest > Tests/AetherCoreTests/Fixtures/kokoro/aether.json`."""
    snapshot = kg.DEFAULT_SNAPSHOT
    vocabulary = json.loads((snapshot / "config.json").read_text())["vocab"]
    voices = kg.voice_ids(snapshot)
    files = {f"{name}/weights.bin": {"bytes": 1, "sha256": "0" * 64} for name in ASSETS.values()}
    files.update({f"voices/{v}.f32": {"bytes": kg.MAX_PHONEMES * 256 * 4, "sha256": "0" * 64} for v in voices})
    report = {"scriptSHA256": {"export_kokoro.py": "0" * 64}, "forgeLockSHA256": "0" * 64}
    return manifest(files, vocabulary, voices, report)


def notice() -> str:
    return "\n".join([
        "Kokoro-82M (Aether-Kit Core AI export)",
        "",
        f"This bundle is a modified version of {kg.REPO} at revision {kg.REVISION}, licensed under the",
        "Apache License, Version 2.0 (LICENSE). Aether-Kit's Forge exporter re-expressed the model (kokoro 0.9.4,",
        "Apache-2.0, by hexgrad) as two float32 Core AI graphs (durations; decoder with an explicit noise input and",
        "frame buckets) and converted the voice packs to raw float32. It is not an official release of Kokoro.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="new bundle directory")
    parser.add_argument("--sample-manifest", action="store_true",
                        help="print the manifest for placeholder files and exit (the Swift decoding fixture)")
    parser.add_argument("--snapshot", type=Path, default=kg.DEFAULT_SNAPSHOT)
    parser.add_argument("--upstream", type=Path, default=kg.DEFAULT_UPSTREAM, help="fetch_upstream.py's unpack")
    args = parser.parse_args()
    if args.sample_manifest:
        print(json.dumps(sample_manifest(), indent=2, ensure_ascii=False))
        return
    if args.output is None:
        parser.error("--output is required")
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    started = time.monotonic()
    args.output.mkdir(parents=True)
    log = Log(args.output / "forge-log.jsonl")
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    root = args.snapshot.resolve()
    lock = HERE.parent / "uv.lock"
    report = {
        "schema": 1, "id": ID, "command": sys.argv,
        "exportedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {"os": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
                 "machine": platform.machine(), "python": platform.python_version()},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "coreai-torch", "coreai-core", "transformers",
                                                                "numpy")},
        "forgeLockSHA256": kg.sha256_file(lock), "forgePyprojectSHA256": kg.sha256_file(HERE.parent / "pyproject.toml"),
        "scriptSHA256": {p.name: kg.sha256_file(p) for p in sorted(HERE.glob("*.py"))},
        "source": {"repository": kg.REPO, "revision": kg.REVISION, "snapshot": str(root), "verified": {},
                   "upstreamCode": {"package": "kokoro 0.9.4", "wheelSHA256": kg.UPSTREAM_WHEEL_SHA256}},
        "precision": "float32 weights and activations; no quantization",
        "decoderBuckets": DECODER_BUCKETS, "compute": {r: c for r, (c, _) in COMPUTE.items()},
        "conversion": {}, "stages": {},
    }
    report_path = args.output / "export-report.json"

    def save() -> None:
        report["peakRSSGB"] = peak_rss_gb()
        report["wallSeconds"] = round(time.monotonic() - started, 1)
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    log.start("verify-sources")
    report["source"]["verified"] = kg.verify_sources(root, args.upstream)
    report["stages"]["verify-sources"] = log.done("verify-sources")
    save()

    kmodel = kg.build_kmodel(args.upstream, root)
    vocabulary = kmodel.vocab
    # Example inputs for tracing (the dynamic axes are declared below).
    phonemes = "həlˈO wˈɜɹld, ðɪs ɪz kˈOkəɹO."
    ids = torch.tensor([kg.tokenize(vocabulary, phonemes)], dtype=torch.int32)
    style = torch.load(root / "voices" / "af_heart.pt", weights_only=True)[len(phonemes) - 1]
    durations = kg.DurationsGraph(kmodel).eval()
    decoder = kg.BucketedDecoderGraph(kmodel).eval()
    sums, d, t_en = durations(ids, style)
    en, asr = kg.align(d, t_en, kg.round_durations(sums, 1.0))
    frames = en.shape[-1]
    noise = torch.from_numpy(kg.splitmix_normal(1, frames * kg.SAMPLES_PER_FRAME * kg.HARMONICS)).reshape(
        1, frames, kg.SAMPLES_PER_FRAME, kg.HARMONICS)
    tokens = torch.export.Dim("tokens", min=3, max=kg.MAX_PHONEMES + 2)
    frame_dim = torch.export.Dim("frames", min=2, max=MAX_FRAMES)

    log.start("export-durations")
    report["conversion"]["durations"] = convert(
        durations, (ids, style), ({1: tokens}, None), ["input_ids", "style"], ["duration", "prosody", "text"],
        "durations", args.output / ASSETS["durations"])
    report["stages"]["export-durations"] = log.done("export-durations")
    save()
    log.start("export-decoder")
    report["conversion"]["decoder"] = convert(
        decoder, (en, asr, style, noise, torch.tensor([frames], dtype=torch.int32)),
        ({2: frame_dim}, {2: frame_dim}, None, {1: frame_dim}, None),
        ["prosody", "text", "style", "noise", "frames"], ["audio"], "decode", args.output / ASSETS["decoder"])
    report["stages"]["export-decoder"] = log.done("export-decoder")
    del durations, decoder, kmodel
    gc.collect()
    save()

    log.start("package")
    (args.output / "voices").mkdir()
    voices = kg.voice_ids(root)
    for voice in voices:
        pack = torch.load(root / "voices" / f"{voice}.pt", weights_only=True)
        if tuple(pack.shape) != (kg.MAX_PHONEMES, 1, 256):
            raise SystemExit(f"unexpected voice shape {tuple(pack.shape)} for {voice}")
        if voice[0] not in LANGUAGES or voice[1] not in GENDERS:
            raise SystemExit(f"voice {voice} has no known language/gender prefix")
        data = pack.reshape(kg.MAX_PHONEMES, 256).to(torch.float32).numpy().astype("<f4").tobytes()
        (args.output / "voices" / f"{voice}.f32").write_bytes(data)
    shutil.copyfile(args.upstream / "kokoro-0.9.4.dist-info/licenses/LICENSE", args.output / "LICENSE")
    (args.output / "NOTICE").write_text(notice())
    files = {str(p.relative_to(args.output)): {"bytes": p.stat().st_size, "sha256": kg.sha256_file(p)}
             for p in tree_files(args.output) if str(p.relative_to(args.output)) not in NOT_BUNDLE}
    report["voices"] = len(voices)
    report["bundleBytes"] = sum(f["bytes"] for f in files.values())
    report["assetBytes"] = {role: tree_bytes(args.output / name) for role, name in ASSETS.items()}
    report["stages"]["package"] = log.done("package")
    save()
    (args.output / ".aether").mkdir()
    manifest_path = args.output / ".aether/aether.json"
    manifest_path.write_text(json.dumps(manifest(files, vocabulary, voices, report), indent=2, ensure_ascii=False) + "\n")
    report["aetherJSONSHA256"] = kg.sha256_file(manifest_path)
    save()
    sums_lines = [f"{kg.sha256_file(p)}  {p.relative_to(args.output)}" for p in tree_files(args.output)
                  if p.name not in ("SHA256SUMS", "forge-log.jsonl")]
    (args.output / "SHA256SUMS").write_text("\n".join(sums_lines) + "\n")
    print(json.dumps({"conversion": report["conversion"], "bundleBytes": report["bundleBytes"],
                      "peakRSSGB": report["peakRSSGB"], "wallSeconds": report["wallSeconds"]}, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stages a trimmed Kokoro bundle and the reference cases for the on-device test host.

Copies the two model directories, only the voices the reference cases use (rewriting kokoro.json's
voice list), and each case's case.json and audio.f32 into
Examples/SpeechDevice/Assets, which is git-ignored apart from .gitkeep. Remove it after the run.
Standard library only.
"""
import argparse
import json
import shutil
from pathlib import Path

MARKER = ".kokoro-device-staging"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", type=Path)
    ap.add_argument("references", type=Path)
    ap.add_argument("--frame-cap", type=int, default=1024)
    ap.add_argument("--decoder-only", type=Path, metavar="REDUCED_BUNDLE",
                    help="stage only BUNDLE's float32 decoder and REDUCED_BUNDLE's decoder for the decoder-only probe mode")
    ap.add_argument("--decoder-frames", type=int, default=200, help="decoder-only: real frames (padded to their bucket)")
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).resolve().parents[2] / "Examples/SpeechDevice/Assets")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / MARKER
    existing = [c for c in args.output.iterdir() if c.name not in (".gitkeep", MARKER)]
    if existing and not marker.exists():
        raise SystemExit(f"{args.output} is not empty and was not staged by this script; refusing to delete its contents")
    for child in existing:  # only a previous staging of this script; the tracked .gitkeep stays
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    marker.write_text("staged by Forge/kokoro/stage_device_assets.py; safe to delete\n")
    if args.decoder_only:
        float32 = json.loads((args.bundle / "kokoro.json").read_text())
        reduced = json.loads((args.decoder_only / "kokoro.json").read_text())
        if float32.get("precision") != "float32" or reduced.get("decoderPrecision", reduced.get("precision")) == "float32":
            raise SystemExit("--decoder-only needs a float32 BUNDLE and a reduced-precision REDUCED_BUNDLE")
        bucket = next(b for b in float32["decoderBuckets"] if b >= args.decoder_frames)
        (args.output / "decoder-only").mkdir()
        shutil.copytree(args.bundle / float32["models"]["decoder"]["path"], args.output / "decoder-only" / "float32.aimodel")
        shutil.copytree(args.decoder_only / reduced["models"]["decoder"]["path"], args.output / "decoder-only" / "reduced.aimodel")
        (args.output / "decoder-only.json").write_text(json.dumps({"frames": args.decoder_frames, "bucket": bucket}) + "\n")
        print(args.output, "decoder-only", bucket)
        return
    (args.output / "bundle" / "voices").mkdir(parents=True)
    contract = json.loads((args.bundle / "kokoro.json").read_text())
    index = json.loads((args.references / "references.json").read_text())
    names = [c["name"] for c in index["cases"]]
    voices = {json.loads((args.references / n / "case.json").read_text())["voice"] for n in names}
    for model in contract["models"].values():
        shutil.copytree(args.bundle / model["path"], args.output / "bundle" / model["path"])
    contract["voices"] = [v for v in contract["voices"] if v["id"] in voices]
    for voice in contract["voices"]:
        shutil.copy2(args.bundle / voice["file"], args.output / "bundle" / voice["file"])
    (args.output / "bundle" / "kokoro.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n")
    for name in names:
        target = args.output / "references" / name
        target.mkdir(parents=True)
        for file in ("case.json", "audio.f32"):
            shutil.copy2(args.references / name / file, target / file)
    (args.output / "frame-cap.txt").write_text(f"{args.frame_cap}\n")
    print(args.output, sorted(voices))


if __name__ == "__main__":
    main()

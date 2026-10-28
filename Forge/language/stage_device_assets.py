#!/usr/bin/env python3
"""Stages an iOS-registered language bundle and its reference for the on-device host (Examples/LanguageDevice).

    python3 -I Forge/language/stage_device_assets.py text BUNDLE REFERENCE
    python3 -I Forge/language/stage_device_assets.py vision BUNDLE REFERENCE [--case NAME ...]

BUNDLE is a registered macOS bundle (`language-model.json`). Its model asset(s) and tokenizer are copied into
Examples/LanguageDevice/Assets/bundle and registered again for iOS with the same registration script and source
(`register_qwen3_bundle.py`, or `register_qwen3_vl_bundle.py` for a bundle with a vision registration): the asset is
not re-exported, only the registration's platform changes. REFERENCE is the float32 Transformers reference directory
(`reference.json` plus its files); vision mode copies only the selected cases (default: the first). The Assets
directory is git-ignored apart from .gitkeep; remove its contents after the run. Standard library only.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

MARKER = ".language-device-staging"
HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("text", "vision"))
    ap.add_argument("bundle", type=Path)
    ap.add_argument("reference", type=Path)
    ap.add_argument("--case", action="append", dest="cases", help="vision: a reference case to stage (repeatable)")
    ap.add_argument("--output", type=Path, default=HERE.parents[1] / "Examples/LanguageDevice/Assets")
    args = ap.parse_args()
    registration = json.loads((args.bundle / "language-model.json").read_text())
    if (registration.get("vision") is not None) != (args.mode == "vision"):
        raise SystemExit(f"{args.bundle} is not a {args.mode} bundle")

    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / MARKER
    existing = [c for c in args.output.iterdir() if c.name not in (".gitkeep", MARKER)]
    if existing and not marker.exists():
        raise SystemExit(f"{args.output} is not empty and was not staged by this script; refusing to delete its contents")
    for child in existing:  # only a previous staging of this script; the tracked .gitkeep stays
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    marker.touch()

    bundle = args.output / "bundle"
    bundle.mkdir()
    names = [registration["modelDirectory"], registration["tokenizerDirectory"]]
    if registration.get("vision"):
        names.append(registration["vision"]["modelDirectory"])
    names += [p.name for p in args.bundle.glob("*export-report.json")]
    for name in names:
        source = args.bundle / name
        (shutil.copytree if source.is_dir() else shutil.copy2)(source, bundle / name)
    script = HERE / ("register_qwen3_vl_bundle.py" if args.mode == "vision" else "register_qwen3_bundle.py")
    command = [sys.executable, "-I", str(script), "--bundle", str(bundle),
               "--source-repository", registration["sourceRepository"], "--source-revision", registration["sourceRevision"],
               "--platform", "ios", "--maximum-context-length", str(registration["maximumContextLength"])]
    if registration.get("baseModel"):
        command += ["--base-model", registration["baseModel"]]
    if args.mode == "vision":
        command += ["--maximum-image-tokens", str(registration["vision"]["maximumImageTokens"])]
    elif "supportsReasoning" in registration:
        command += ["--supports-reasoning", "true" if registration["supportsReasoning"] else "false"]
    subprocess.run(command, check=True)
    ios = json.loads((bundle / "language-model.json").read_text())
    if ios.get("platform") != "ios":
        raise SystemExit("the iOS registration does not name the ios platform")
    # Only the platform may differ: the model and vision hashes, tokenizer files, stop tokens, bounds and every other
    # field must match the macOS registration exactly.
    differences = sorted(k for k in set(ios) | set(registration)
                         if k != "platform" and ios.get(k) != registration.get(k))
    if differences:
        raise SystemExit(f"the iOS registration differs from the macOS one beyond platform: {differences}")

    reference = args.output / "reference"
    reference.mkdir()
    document = json.loads((args.reference / "reference.json").read_text())
    files = []
    if args.mode == "vision":
        cases = [c for c in document["cases"] if c["name"] in args.cases] if args.cases else document["cases"][:1]
        if not cases:
            raise SystemExit(f"no reference case named {args.cases}")
        document["cases"] = cases
        for c in cases:
            files += [c["image"]["file"], c["forcedLogits"]["file"]]
    else:
        files.append(document["forcedLogits"]["file"])
    for name in files:
        shutil.copy2(args.reference / name, reference / name)
    (reference / "reference.json").write_text(json.dumps(document, indent=2) + "\n")
    probe = {"mode": args.mode}
    if args.mode == "vision":
        probe["cases"] = [c["name"] for c in document["cases"]]
    (args.output / "probe.json").write_text(json.dumps(probe) + "\n")
    total = sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())
    print(f"staged {args.mode}: {total / 1e9:.2f} GB in {args.output}")


if __name__ == "__main__":
    main()

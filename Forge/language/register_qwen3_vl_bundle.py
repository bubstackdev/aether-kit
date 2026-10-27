#!/usr/bin/env python3
"""Write `language-model.json` for a Qwen3-VL bundle (development tooling only; standard library only).

    python3 Forge/language/register_qwen3_vl_bundle.py --bundle <dir with model.aimodel, vision.aimodel, tokenizer/> \
        --source-repository Qwen/Qwen3-VL-2B-Instruct --source-revision <sha> --platform macos [--maximum-image-tokens 512]

Reads the text export's `export-report.json` (bounds), the vision export's `vision-export-report.json` (its fixed
image grids, one static-shape function each) and the tokenizer copy's `config.json`, `preprocessor_config.json` and
`generation_config.json`; hashes the assets the way `LanguageModelManifest.verify` does, and records every tokenizer
file (`tokenizerFiles`, by the same rules as `register_qwen3_bundle.py`) so `verifyTokenizer` checks the exact file set. The vision registration never
requests `expectFrequentReshapes` (engine F-P3-26) and always lists its grids (dynamic-shape encoders are refused).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def tokenizer_files(root: Path) -> dict[str, str]:
    """Every regular tokenizer file by relative path, as `verifyTokenizer` sees them (`register_qwen3_bundle.py`'s
    rules). Any symbolic link refuses registration first, hidden or not (the verifier refuses non-regular entries
    before it ignores dotfiles); then dotfiles (a Finder .DS_Store) are skipped."""
    files = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_symlink():
            sys.exit(f"tokenizer entry {rel} is a symbolic link; register real files only")
        if any(part.startswith(".") for part in rel.parts):
            continue
        if p.is_file():
            files[str(rel)] = sha256(p)
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--platform", choices=("macos", "ios"), required=True)
    parser.add_argument("--maximum-context-length", type=int, default=4096)
    parser.add_argument("--maximum-image-tokens", type=int, default=512)
    parser.add_argument("--base-model", help="provenance: the upstream model a fine-tune derives from")
    args = parser.parse_args()
    b = args.bundle
    report = json.loads((b / "export-report.json").read_text())
    tok = b / "tokenizer"
    config = json.loads((tok / "config.json").read_text())
    pre = json.loads((tok / "preprocessor_config.json").read_text())
    generation = json.loads((tok / "generation_config.json").read_text())
    eos = generation.get("eos_token_id", config["text_config"]["eos_token_id"])
    stop = sorted(set(eos if isinstance(eos, list) else [eos]))
    vc = config["vision_config"]
    bounds = report["exportBounds"]
    vision_report = json.loads((b / "vision-export-report.json").read_text())
    grids = [{"height": g["height"], "width": g["width"]} for g in vision_report["imageGrids"]]
    budget = min(args.maximum_image_tokens, bounds["maximumImageChunk"])
    merge = vc["spatial_merge_size"]
    if any(g["height"] * g["width"] // merge ** 2 > budget for g in grids):
        raise SystemExit("an image grid exceeds the image-token budget")
    manifest = {
        "schemaVersion": 1,
        "modelDirectory": "model.aimodel",
        "tokenizerDirectory": "tokenizer",
        "sourceRepository": args.source_repository,
        "sourceRevision": args.source_revision,
        "modelSHA256": sha256(b / "model.aimodel" / "main.mlirb"),
        "tokenizerSHA256": sha256(tok / "tokenizer.json"),
        "tokenizerFiles": tokenizer_files(tok),
        "maximumContextLength": args.maximum_context_length,
        "stopTokenIDs": stop,
        "platform": args.platform,
        "quantization": "fp16",
        "exportBounds": bounds,
        "vision": {
            "modelDirectory": "vision.aimodel",
            "modelSHA256": sha256(b / "vision.aimodel" / "main.mlirb"),
            "imageTokenID": config["image_token_id"],
            "visionStartTokenID": config["vision_start_token_id"],
            "visionEndTokenID": config["vision_end_token_id"],
            "patchSize": vc["patch_size"],
            "mergeSize": vc["spatial_merge_size"],
            "temporalPatchSize": vc["temporal_patch_size"],
            "minimumPixels": pre["size"]["shortest_edge"],
            "maximumPixels": pre["size"]["longest_edge"],
            "imageMean": pre["image_mean"],
            "imageStandardDeviation": pre["image_std"],
            "positionTableSide": int(round(vc["num_position_embeddings"] ** 0.5)),
            "maximumImageTokens": budget,
            "imageGrids": grids,
        },
    }
    if args.base_model:
        manifest["baseModel"] = args.base_model
    (b / "language-model.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("modelSHA256", "stopTokenIDs", "exportBounds")}))


if __name__ == "__main__":
    main()

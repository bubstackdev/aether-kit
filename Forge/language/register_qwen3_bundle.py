#!/usr/bin/env python3
"""Write `language-model.json` for a dense Qwen3 bundle (development tooling only; standard library only).

    python3 Forge/language/register_qwen3_bundle.py --bundle <export dir with model.aimodel, tokenizer/> \
        --source-repository Qwen/Qwen3-0.6B --source-revision <sha> --platform macos [--maximum-context-length 4096]

The bundle is an `export_qwen3.py` output. Reads its `export-report.json` (bounds, source) and the tokenizer copy's
`generation_config.json` (stop IDs); hashes the assets the way `LanguageModelManifest.verify` does, and registers every
tokenizer file (`tokenizerFiles`) so `verifyTokenizer` covers the chat template too. Refuses a source
that differs from the export's, and a context beyond the exported cache capacity.
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
    """Every regular tokenizer file by relative path, as `verifyTokenizer` sees them. Any symbolic link refuses
    registration first, hidden or not (the verifier refuses non-regular entries before it ignores dotfiles); then
    dotfiles (a Finder .DS_Store) are skipped."""
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
    parser.add_argument("--base-model", help="provenance: the upstream model a fine-tune derives from")
    parser.add_argument("--supports-reasoning", choices=("true", "false"),
                        help="consumer metadata: whether the model has a thinking mode (hybrid Qwen3 yes, 2507 Instruct no)")
    args = parser.parse_args()
    b = args.bundle
    report = json.loads((b / "export-report.json").read_text())
    source = report["source"]
    if (source["repository"], source["revision"]) != (args.source_repository, args.source_revision):
        sys.exit(f"export source {source['repository']}@{source['revision']} is not the one registered")
    bounds = report["exportBounds"]
    if not 0 < args.maximum_context_length <= min(bounds["maximumCacheCapacity"], 16384):
        sys.exit("the context must be positive and within the exported cache capacity")
    generation = json.loads((b / "tokenizer" / "generation_config.json").read_text())
    eos = generation["eos_token_id"]
    manifest = {
        "schemaVersion": 1,
        "modelDirectory": "model.aimodel",
        "tokenizerDirectory": "tokenizer",
        "sourceRepository": args.source_repository,
        "sourceRevision": args.source_revision,
        "modelSHA256": sha256(b / "model.aimodel" / "main.mlirb"),
        "tokenizerSHA256": sha256(b / "tokenizer" / "tokenizer.json"),
        "maximumContextLength": args.maximum_context_length,
        "stopTokenIDs": sorted(set(eos if isinstance(eos, list) else [eos])),
        "platform": args.platform,
        "quantization": {"float16": "fp16", "float32": "fp32"}.get(report["dtype"], report["dtype"]),
        "exportBounds": bounds,
        "tokenizerFiles": tokenizer_files(b / "tokenizer"),
    }
    if args.base_model:
        manifest["baseModel"] = args.base_model
    if args.supports_reasoning:
        manifest["supportsReasoning"] = args.supports_reasoning == "true"
    (b / "language-model.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("modelSHA256", "stopTokenIDs", "quantization", "exportBounds")}))


if __name__ == "__main__":
    main()

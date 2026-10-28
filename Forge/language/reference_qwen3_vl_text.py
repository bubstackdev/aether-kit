#!/usr/bin/env python3
"""Independent CPU reference for a text-only prompt on a Qwen3-VL checkpoint (development tooling only).

Upstream Transformers' `Qwen3VLForConditionalGeneration` (float32, CPU) with its own processor and chat template, on
one user message without images: the rendered token ids, a greedy continuation (`generate`, no sampling) and the
teacher-forced logits for the prompt's last row and each forced continuation token. The record has the fields Kit's
plan gate reads for its near-tie rule (`languageOverlappedPrefillMatchesOtherPlans`, AETHER_OVERLAP_REFERENCES):
`promptTokens`, `forcedTokens` and `forcedLogits`.

    Forge/.venv/bin/python -I Forge/language/reference_qwen3_vl_text.py --checkpoint <hf snapshot> \
        --source-repository <repo> --source-revision <rev> --license <licence> \
        --text <user message> [--repeat N] --output <new dir>

Outputs under the new `--output` directory: `reference.json` and `forced-logits.f32` [rows, vocab] (little-endian
float32; row r = logits after prompt + forcedTokens[:r]).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--license", required=True, help="the checkpoint's model-card licence, e.g. apache-2.0")
    parser.add_argument("--base-model", help="the upstream model a fine-tune derives from")
    parser.add_argument("--text", required=True, help="the user message")
    parser.add_argument("--repeat", type=int, default=1, help="repeat the message text this many times")
    parser.add_argument("--greedy-tokens", type=int, default=48)
    parser.add_argument("--forced-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.checkpoint, torch_dtype=torch.float32).eval()
    text = args.text * args.repeat
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt")
    ids = inputs["input_ids"][0].tolist()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.greedy_tokens, do_sample=False, temperature=None,
                             top_p=None, top_k=None)
        greedy = out[0, len(ids):].tolist()
        forced = greedy[:args.forced_tokens]
        full = torch.tensor([ids + forced])
        rows = model(input_ids=full, attention_mask=torch.ones_like(full)).logits[0, len(ids) - 1:len(ids) + len(forced) - 1]
        rows = rows.double().numpy()
    args.output.mkdir(parents=True)
    logits_path = args.output / "forced-logits.f32"
    rows.astype("<f4").tofile(logits_path)
    record = {
        "schema": 1, "kind": "qwen3-vl-text-cpu-reference",
        "oracle": "transformers Qwen3VLForConditionalGeneration + AutoProcessor chat template, float32, CPU, no images",
        "source": {"repository": args.source_repository, "revision": args.source_revision, "license": args.license,
                   **({"baseModel": args.base_model} if args.base_model else {})},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers")},
        "scriptSHA256": {Path(__file__).name: sha256(Path(__file__).resolve())},
        "checkpointSHA256": {p.name: sha256(p) for p in sorted(args.checkpoint.glob("*.safetensors"))},
        "text": {"message": args.text, "repeat": args.repeat},
        "promptTokens": ids, "greedyTokens": greedy, "greedyText": processor.decode(greedy), "forcedTokens": forced,
        "forcedLogits": {"file": logits_path.name, "rows": int(rows.shape[0]), "vocab": int(rows.shape[1]),
                         "sha256": sha256(logits_path), "rowMeaning": "row r = logits after prompt + forcedTokens[:r]"},
    }
    (args.output / "reference.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"promptTokens": len(ids), "greedyText": record["greedyText"]}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Independent upstream reference for the Qwen3.5 hybrid decoder, text only (development tooling only).

Runs in the Qwen3.5 reference environment, not Forge's main one (Forge's transformers 4.57.3 has no `qwen3_5`):

    uv sync --frozen --project Forge/reference-qwen35
    Forge/reference-qwen35/.venv/bin/python -I Forge/language/reference_qwen35.py --checkpoint <snapshot> \
        --source-repository Qwen/Qwen3.5-4B --source-revision <sha> --license apache-2.0 --output <new dir>

Upstream Transformers (`Qwen3_5ForConditionalGeneration`) loads the pinned checkpoint in float32, read in place.

- Chat cases: each prompt rendered with the checkpoint's chat template (`enable_thinking: false`, as the describer
  renders it); the greedy continuation (`generate`, no sampling, `--greedy-tokens`) and teacher-forced logits for the
  prompt's last row and each of the first `--forced-tokens` continuation tokens (one forward pass over prompt +
  forced tokens).
- Long context: a document of `--long-tokens` tokens (this repository's docs, concatenated in sorted order) in one
  forward pass, recording logits every `--long-every` rows and at the end. Upstream's multi-token path is the chunked
  gated delta rule over the whole sequence, so this is the drift reference for a host that feeds the same tokens in
  fixed-width submissions.

`--device cpu` (default) or `mps` (float32 either way). Teacher-forced rows always come from `--forced-device`
(default cpu); greedy runs may go on MPS (upstream's torch DeltaNet fallback decodes slowly on the CPU). The record
names the device of each.

Outputs, under the new `--output` directory: `reference.json` (token ids, row metadata, versions, the environment
that produced it, checkpoint identity, hashes) and `forced-<case>.f32`, `long.f32` (little-endian float32
[rows, vocab]).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

CASES = {
    "sky": "Explain, in a few sentences, why the sky is blue during the day and red at sunset.",
    "code": "Write a Python function that returns the n-th Fibonacci number iteratively, with a short docstring.",
    "list": "List five practical tips for keeping houseplants healthy in winter, one line each.",
}
ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def environment() -> dict:
    """Which locked environment produced the record (the interpreter's project and its lockfile hash)."""
    project = Path(sys.prefix).resolve().parent
    lock = project / "uv.lock"
    return {"project": str(project.relative_to(ROOT)) if project.is_relative_to(ROOT) else str(project),
            "uvLockSHA256": sha256(lock) if lock.exists() else None,
            "python": sys.version.split()[0]}


def long_document(root: Path) -> tuple[str, list[str]]:
    files = sorted(p for p in (root / "docs").rglob("*.md"))
    return "\n\n".join(p.read_text() for p in files), [str(p.relative_to(root)) for p in files]


def load(checkpoint: Path, device: str):
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText.from_pretrained(checkpoint, dtype=torch.float32).to(device).eval()


def logits_rows(model, tokens: list[int], rows: list[int]) -> np.ndarray:
    """Logits at `rows` (row r = after tokens[:r + 1]) from one forward pass over `tokens`; the head runs on those rows
    only (a 4,096-row float32 logits tensor would be 4 GB)."""
    device = next(model.parameters()).device
    with torch.no_grad():
        out = model(input_ids=torch.tensor([tokens], device=device), use_cache=False,
                    logits_to_keep=torch.tensor(rows, device=device))
    return out.logits[0].double().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"], help="greedy runs and the long pass")
    parser.add_argument("--forced-device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--greedy-tokens", type=int, default=128)
    parser.add_argument("--forced-tokens", type=int, default=32)
    parser.add_argument("--long-tokens", type=int, default=4096)
    parser.add_argument("--long-every", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    eos = json.loads((args.checkpoint / "config.json").read_text())["text_config"]["eos_token_id"]
    args.output.mkdir(parents=True)
    started = time.time()

    # Greedy continuations (and the long pass) on --device.
    model = load(args.checkpoint, args.device)
    cases = []
    for name, message in CASES.items():
        text = tokenizer.apply_chat_template([{"role": "user", "content": message}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        prompt = tokenizer(text, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            out = model.generate(input_ids=torch.tensor([prompt], device=args.device), max_new_tokens=args.greedy_tokens,
                                 do_sample=False, temperature=None, top_p=None, top_k=None, eos_token_id=None,
                                 pad_token_id=tokenizer.pad_token_id)
        greedy = out[0, len(prompt):].tolist()
        cases.append({"name": name, "prompt": message, "renderedPrompt": text, "promptTokens": prompt,
                      "greedyTokens": greedy, "greedyText": tokenizer.decode(greedy), "greedyDevice": args.device,
                      "forcedTokens": greedy[:args.forced_tokens]})
        print(f"[reference] {name}: {len(prompt)} prompt tokens, greedy {tokenizer.decode(greedy[:24])!r} "
              f"({time.time() - started:.0f} s)", flush=True)

    document, files = long_document(ROOT)
    tokens = tokenizer(document, add_special_tokens=False)["input_ids"][:args.long_tokens]
    rows = sorted(set(list(range(args.long_every - 1, len(tokens), args.long_every)) + [len(tokens) - 1]))
    long = logits_rows(model, tokens, rows)
    long_path = args.output / "long.f32"
    long.astype("<f4").tofile(long_path)
    print(f"[reference] long: {len(tokens)} tokens, {len(rows)} rows ({time.time() - started:.0f} s)", flush=True)

    # Teacher-forced rows on --forced-device.
    if args.forced_device != args.device:
        del model
        model = load(args.checkpoint, args.forced_device)
    for case in cases:
        prompt, forced = case["promptTokens"], case["forcedTokens"]
        reference = logits_rows(model, prompt + forced, list(range(len(prompt) - 1, len(prompt) + len(forced) - 1)))
        path = args.output / f"forced-{case['name']}.f32"
        reference.astype("<f4").tofile(path)
        agree = sum(int(np.argmax(r)) == t for r, t in zip(reference, forced))
        case["forcedLogits"] = {"file": path.name, "dtype": "float32-le", "rows": int(reference.shape[0]),
                                "vocab": int(reference.shape[1]), "sha256": sha256(path), "device": args.forced_device,
                                "rowMeaning": "row r = logits after prompt + forcedTokens[:r]",
                                "argmaxEqualsGreedy": f"{agree}/{len(forced)}"}
        print(f"[reference] {case['name']}: forced rows argmax = greedy on {agree}/{len(forced)}", flush=True)

    record = {
        "schema": 1,
        "kind": "qwen35-hybrid-text-reference",
        "oracle": "transformers Qwen3_5ForConditionalGeneration, float32",
        "environment": environment(),
        "source": {"repository": args.source_repository, "revision": args.source_revision, "license": args.license},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers")},
        "stopTokenIDs": [eos] if isinstance(eos, int) else eos,
        "scriptSHA256": {Path(__file__).name: sha256(Path(__file__).resolve())},
        "checkpointSHA256": {p.name: sha256(p) for p in sorted(args.checkpoint.iterdir()) if p.is_file()},
        "cases": cases,
        "long": {"files": files, "tokens": tokens, "rows": rows, "file": long_path.name, "dtype": "float32-le",
                 "vocab": int(long.shape[1]), "sha256": sha256(long_path), "device": args.device,
                 "path": "one forward pass over the whole document (upstream chunked gated delta rule)",
                 "rowMeaning": "row r = logits after tokens[:r + 1]"},
        "seconds": round(time.time() - started, 1),
    }
    (args.output / "reference.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()

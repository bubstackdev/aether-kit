#!/usr/bin/env python3
"""Independent CPU reference for Qwen3 language qualification (development tooling only).

Upstream Transformers loads the pinned checkpoint in float32 on the CPU. For a chat-rendered prompt it records:
- the greedy continuation (`generate`, no sampling) of `--greedy-tokens` tokens;
- teacher-forced logits for the prompt's last row and each of the first `--forced-tokens` continuation tokens.

It also checks this directory's export module (`qwen3_decode.Qwen3Decoder`, both contracts, eager float32) against
the upstream logits, which separates porting errors from Core AI conversion errors.

Outputs, under the new `--output` directory:
- `reference.json`: prompt and continuation token ids, row metadata, versions, checkpoint identity, hashes;
- `forced-logits.f32`: little-endian float32 `[rows, vocab]` teacher-forced logits.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen3_decode import Qwen3Decoder, load_weights, read_config  # noqa: E402

PROMPT = "Explain, in a few sentences, why the sky is blue during the day and red at sunset."


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def eager_rows(module: Qwen3Decoder, prompt: list[int], forced: list[int], chunk: int) -> np.ndarray:
    """Last-row logits after a chunked prefill, then after each forced token."""
    kc, vc = module.empty_state(len(prompt) + len(forced) + 8)
    rows = []

    def step(tokens: list[int], start: int) -> np.ndarray:
        ids = torch.tensor([tokens], dtype=torch.int32)
        if module.contract == "running":
            pos = torch.arange(start + len(tokens), dtype=torch.int32)[None]
        else:
            pos = torch.arange(start, start + len(tokens), dtype=torch.int32)[None]
        return module.step(ids, pos, kc, vc).double().numpy()

    with torch.no_grad():
        for start in range(0, len(prompt), chunk):
            last = step(prompt[start:start + chunk], start)
        rows.append(last)
        for i, tok in enumerate(forced[:-1]):
            rows.append(step([tok], len(prompt) + i))
    return np.stack(rows)


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    cos = [float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))) for x, y in zip(a, b)]
    return {"minCosine": round(min(cos), 8), "maxAbs": round(float(np.abs(a - b).max()), 6),
            "argmaxEqualRows": int(sum(int(np.argmax(x)) == int(np.argmax(y)) for x, y in zip(a, b))), "rows": len(a)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--license", required=True, help="the checkpoint's model-card licence, e.g. apache-2.0")
    parser.add_argument("--base-model", help="the upstream model a fine-tune derives from, e.g. Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--greedy-tokens", type=int, default=64)
    parser.add_argument("--forced-tokens", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(max(1, torch.get_num_threads()))
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    text = tokenizer.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    prompt = tokenizer(text, add_special_tokens=False)["input_ids"]
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        out = model.generate(torch.tensor([prompt]), max_new_tokens=args.greedy_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None, eos_token_id=None, pad_token_id=0)
    greedy = out[0, len(prompt):].tolist()
    forced = greedy[:args.forced_tokens]
    with torch.no_grad():
        full = model(torch.tensor([prompt + forced])).logits[0].double().numpy()
    # Row r predicts token r + 1: the prompt's last row, then one row per forced token except the last.
    reference = full[len(prompt) - 1:len(prompt) + len(forced) - 1]
    # Free the upstream model before loading the port: at 4B, two float32 copies would need about 32 GB.
    del model, out, full
    gc.collect()

    cfg = read_config(args.checkpoint)
    port = {}
    for contract, chunk in (("running", 7), ("bucketed", 8)):
        module = Qwen3Decoder(cfg, contract)
        load_weights(module, args.checkpoint)
        module = module.float().eval()
        port[contract] = compare(eager_rows(module, prompt, forced, chunk), reference)
        del module

    args.output.mkdir(parents=True)
    logits_path = args.output / "forced-logits.f32"
    reference.astype("<f4").tofile(logits_path)
    record = {
        "schema": 1,
        "kind": "qwen3-cpu-reference",
        "oracle": "transformers AutoModelForCausalLM, float32, CPU",
        "source": {"repository": args.source_repository, "revision": args.source_revision, "license": args.license,
                   **({"baseModel": args.base_model} if args.base_model else {})},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers")},
        "scriptSHA256": {p.name: sha256(p) for p in sorted(Path(__file__).resolve().parent.glob("*.py"))},
        "checkpointSHA256": {p.name: sha256(p) for p in sorted(args.checkpoint.glob("*.safetensors"))},
        "prompt": PROMPT,
        "promptTokens": prompt,
        "greedyTokens": greedy,
        "greedyText": tokenizer.decode(greedy),
        "forcedTokens": forced,
        "forcedLogits": {"file": logits_path.name, "dtype": "float32-le", "rows": int(reference.shape[0]),
                         "vocab": int(reference.shape[1]), "sha256": sha256(logits_path),
                         "rowMeaning": "row r = logits after prompt + forcedTokens[:r]"},
        "eagerPortVsReference": port,
    }
    (args.output / "reference.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"greedyText": record["greedyText"], "eagerPortVsReference": port}, indent=1))


if __name__ == "__main__":
    main()

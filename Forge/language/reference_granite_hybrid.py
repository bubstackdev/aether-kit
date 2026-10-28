#!/usr/bin/env python3
"""Independent CPU reference for Granite 4.0-H hybrid qualification (development tooling only).

Upstream Transformers (`GraniteMoeHybridForCausalLM`) loads the pinned checkpoint in float32 on the CPU, read in place.

- Chat cases: for each chat-rendered prompt, the greedy continuation (`generate`, no sampling) and teacher-forced
  logits for the prompt's last row and each of the first `--forced-tokens` continuation tokens.
- Long context: a document of `--long-tokens` tokens (this repository's docs, concatenated in sorted order) fed one
  token at a time through upstream's recurrent decode path, recording logits every `--long-every` rows and at the
  end. This is the drift reference for the fp32 SSM state: Transformers' multi-token path materialises
  [chunks, 256, 256, heads, d_state] and cannot continue a chunk from a cached conv state, so long contexts use its
  one-token path.

It also checks this directory's port (`granite_hybrid.GraniteHybridDecoder`, eager float32) against both, with the
prompt fed in the hybrid plan (64-token chunks, then the remainder in powers of two, never stepping back).

Outputs, under the new `--output` directory: `reference.json` (token ids, row metadata, versions, checkpoint identity,
hashes, the port check), `forced-<case>.f32` and `long.f32` (little-endian float32 [rows, vocab]).
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from granite_hybrid import GraniteHybridDecoder, load_weights, read_config  # noqa: E402

CASES = {
    "sky": "Explain, in a few sentences, why the sky is blue during the day and red at sunset.",
    "code": "Write a Python function that returns the n-th Fibonacci number iteratively, with a short docstring.",
    "list": "List five practical tips for keeping houseplants healthy in winter, one line each.",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def hybrid_plan(count: int, chunk: int = 64) -> list[int]:
    """Submission widths for `count` prompt tokens: full chunks, then the remainder in descending powers of two."""
    widths = [chunk] * (count // chunk)
    rest = count % chunk
    bit = chunk // 2
    while rest:
        if rest >= bit:
            widths.append(bit)
            rest -= bit
        bit //= 2
    return widths


def port_rows(module: GraniteHybridDecoder, tokens: list[int], rows: list[int]) -> np.ndarray:
    """Eager port logits at `rows` (row r = logits after tokens[:r + 1]), feeding tokens in the hybrid plan up to the
    first requested row and one at a time after it."""
    kc, vc, cs, ss = module.empty_state(len(tokens) + 8)
    out, start = {}, 0
    want = set(rows)
    with torch.no_grad():
        for n in hybrid_plan(rows[0] + 1) + [1] * (len(tokens) - rows[0] - 1):
            ids = torch.tensor([tokens[start:start + n]], dtype=torch.int32)
            pos = torch.arange(start, start + n, dtype=torch.int32)[None]
            logits = module.step(ids, pos, kc, vc, cs, ss, recurrent=n == 1)
            start += n
            if start - 1 in want:
                out[start - 1] = logits.double().numpy()
            if start - 1 >= rows[-1]:
                break
    return np.stack([out[r] for r in rows])


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    cos = [float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))) for x, y in zip(a, b)]
    return {"minCosine": round(min(cos), 8), "maxAbs": round(float(np.abs(a - b).max()), 6),
            "argmaxEqualRows": int(sum(int(np.argmax(x)) == int(np.argmax(y)) for x, y in zip(a, b))), "rows": len(a)}


def long_document(root: Path) -> tuple[str, list[str]]:
    files = sorted(p for p in (root / "docs").rglob("*.md"))
    return "\n\n".join(p.read_text() for p in files), [str(p.relative_to(root)) for p in files]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--greedy-tokens", type=int, default=128)
    parser.add_argument("--forced-tokens", type=int, default=32)
    parser.add_argument("--long-tokens", type=int, default=8192)
    parser.add_argument("--long-every", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.float32).eval()
    args.output.mkdir(parents=True)
    cases, logits_by_case = [], {}
    for name, message in CASES.items():
        text = tokenizer.apply_chat_template([{"role": "user", "content": message}], tokenize=False,
                                             add_generation_prompt=True)
        prompt = tokenizer(text, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            out = model.generate(torch.tensor([prompt]), max_new_tokens=args.greedy_tokens, do_sample=False,
                                 temperature=None, top_p=None, top_k=None, eos_token_id=None,
                                 pad_token_id=tokenizer.pad_token_id)
        greedy = out[0, len(prompt):].tolist()
        forced = greedy[:args.forced_tokens]
        with torch.no_grad():
            full = model(torch.tensor([prompt + forced])).logits[0].double().numpy()
        reference = full[len(prompt) - 1:len(prompt) + len(forced) - 1]
        path = args.output / f"forced-{name}.f32"
        reference.astype("<f4").tofile(path)
        logits_by_case[name] = reference
        cases.append({"name": name, "prompt": message, "promptTokens": prompt, "greedyTokens": greedy,
                      "greedyText": tokenizer.decode(greedy), "forcedTokens": forced,
                      "forcedLogits": {"file": path.name, "dtype": "float32-le", "rows": int(reference.shape[0]),
                                       "vocab": int(reference.shape[1]), "sha256": sha256(path),
                                       "rowMeaning": "row r = logits after prompt + forcedTokens[:r]"}})
        print(f"[reference] {name}: {len(prompt)} prompt tokens, greedy {tokenizer.decode(greedy[:24])!r}", flush=True)

    # Long context through upstream's one-token recurrent path.
    document, files = long_document(Path(__file__).resolve().parents[2])
    tokens = tokenizer(document, add_special_tokens=False)["input_ids"][:args.long_tokens]
    rows = sorted(set(list(range(args.long_every - 1, len(tokens), args.long_every)) + [len(tokens) - 1]))
    first = 16
    long_rows, started = {}, time.time()
    from transformers.models.granitemoehybrid.modeling_granitemoehybrid import HybridMambaAttentionDynamicCache

    # Upstream builds this cache only inside `generate`; without it each call would see no context.
    cache = HybridMambaAttentionDynamicCache(model.config, 1, dtype=torch.float32)
    with torch.no_grad():
        out = model(torch.tensor([tokens[:first]]), past_key_values=cache, use_cache=True,
                    cache_position=torch.arange(first))
        cache = out.past_key_values
        for i in range(first, len(tokens)):
            out = model(torch.tensor([[tokens[i]]]), past_key_values=cache, use_cache=True,
                        cache_position=torch.tensor([i]))
            cache = out.past_key_values
            if i in rows:
                long_rows[i] = out.logits[0, -1].double().numpy()
                print(f"[reference] long row {i}: {time.time() - started:.0f} s", flush=True)
    long = np.stack([long_rows[r] for r in rows])
    long_path = args.output / "long.f32"
    long.astype("<f4").tofile(long_path)
    print(f"[reference] long: {len(tokens)} tokens, {len(rows)} rows, {time.time() - started:.0f} s", flush=True)
    del model, cache, out

    cfg = read_config(args.checkpoint)
    port = GraniteHybridDecoder(cfg)
    load_weights(port, args.checkpoint)
    port = port.float().eval()
    port_check = {}
    for case in cases:
        seq = case["promptTokens"] + case["forcedTokens"]
        r = list(range(len(case["promptTokens"]) - 1, len(seq) - 1))
        port_check[case["name"]] = compare(port_rows(port, seq, r), logits_by_case[case["name"]])
    port_check["long"] = compare(port_rows(port, tokens, rows), long)
    print(json.dumps(port_check, indent=1), flush=True)

    record = {
        "schema": 1,
        "kind": "granite-hybrid-cpu-reference",
        "oracle": "transformers GraniteMoeHybridForCausalLM, float32, CPU",
        "source": {"repository": args.source_repository, "revision": args.source_revision, "license": args.license},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers")},
        "scriptSHA256": {p.name: sha256(p) for p in sorted(Path(__file__).resolve().parent.glob("*.py"))},
        "checkpointSHA256": {p.name: sha256(p) for p in sorted(args.checkpoint.iterdir()) if p.is_file()},
        "cases": cases,
        "long": {"files": files, "tokens": tokens, "rows": rows, "file": long_path.name, "dtype": "float32-le",
                 "vocab": int(long.shape[1]), "sha256": sha256(long_path),
                 "path": "prompt tokens[:16] in one call, then one token per call (upstream recurrent decode)",
                 "rowMeaning": "row r = logits after tokens[:r + 1]"},
        "eagerPortVsReference": port_check,
        "portPlan": "64-token chunks, then the remainder in powers of two, then one token per call (recurrent form)",
    }
    (args.output / "reference.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()

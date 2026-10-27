#!/usr/bin/env python3
"""Independent references for AetherLanguageRuntime's sampler.

`random-generator.json`: known-answer outputs of SplitMix64 and of xoshiro256** seeded by it, from this file's own
implementation of the published algorithms, so the Swift generator is checked against code it doesn't share.

`transformers-warpers.json`: what upstream Transformers' own logits processors and warpers make of seeded synthetic
logits rows, in `generate`'s order: `RepetitionPenaltyLogitsProcessor` (over the whole request, prompt included),
`PrefixConstrainedLogitsProcessor` (the allowed set), then `TemperatureLogitsWarper`, `TopKLogitsWarper` and
`TopPLogitsWarper`, then the softmax a sampled step draws from. Each case records the greedy token after the penalty
and the constraint, and the kept tokens with their probabilities. Rows are redrawn until no token's preceding mass lies
within 2e-5 of top-p (well above float32 summation error over these vocabularies), so float32 and float64 summation
agree on the kept set; deliberate ties sit at the top-k boundary
and at the greedy maximum, where both implementations' tie rules are defined.

Writes only into `--output-dir`. Development tooling: Python is absent from shipping targets.

    Forge/.venv/bin/python -I Forge/language/reference_sampling.py --output-dir Tests/AetherLanguageRuntimeTests/Fixtures
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from transformers.generation.logits_process import (
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

MASK = (1 << 64) - 1
SEEDS = [0, 1, 42, 0x0123456789ABCDEF, MASK]


def splitmix64(state: int):
    while True:
        state = (state + 0x9E3779B97F4A7C15) & MASK
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        yield z ^ (z >> 31)


def rotl(x: int, k: int) -> int:
    return ((x << k) | (x >> (64 - k))) & MASK


def xoshiro256starstar(seed: int):
    mix = splitmix64(seed)
    s = [next(mix) for _ in range(4)]
    while True:
        result = (rotl((s[1] * 5) & MASK, 7) * 9) & MASK
        t = (s[1] << 17) & MASK
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = rotl(s[3], 45)
        yield result


def generator_vectors() -> dict:
    cases = []
    for seed in SEEDS:
        mix, xo = splitmix64(seed), xoshiro256starstar(seed)
        cases.append({
            "seed": str(seed),
            "splitmix64": [str(next(mix)) for _ in range(8)],
            "xoshiro256starstar": [str(next(xo)) for _ in range(8)],
        })
    return {"note": "UInt64 values as decimal strings; xoshiro256** state filled by four SplitMix64 outputs", "cases": cases}


# (name, vocabulary, repetition penalty, allowed fraction or None, temperature, top-k, top-p, tie)
WARPER_CASES = [
    ("plain-softmax", 64, 1.0, None, 1.0, None, None, None),
    ("temperature-low", 1000, 1.0, None, 0.7, None, None, None),
    ("temperature-high", 1000, 1.0, None, 1.5, None, None, None),
    ("top-k-1", 1000, 1.0, None, 1.0, 1, None, None),
    ("top-k-40", 1000, 1.0, None, 0.8, 40, None, None),
    ("top-k-tied-boundary", 1000, 1.0, None, 1.0, 5, None, "top-k"),
    ("top-p-0.5", 1000, 1.0, None, 1.0, None, 0.5, None),
    ("top-p-0.9", 2000, 1.0, None, 0.7, None, 0.9, None),
    ("top-p-0.95-flat", 2000, 1.0, None, 2.0, None, 0.95, None),
    ("top-k-then-top-p", 2000, 1.0, None, 0.7, 50, 0.9, None),
    ("penalty", 1000, 1.3, None, 1.0, None, None, None),
    ("penalty-strong-greedy-shift", 1000, 3.0, None, 1.0, None, None, None),
    ("penalty-top-k-top-p", 2000, 1.2, None, 0.8, 40, 0.95, None),
    ("allowed", 1000, 1.0, 0.05, 1.0, None, None, None),
    ("allowed-top-p", 2000, 1.0, 0.02, 0.7, None, 0.9, None),
    ("allowed-penalty-top-k", 2000, 1.1, 0.1, 1.0, 20, None, None),
    ("allowed-single", 1000, 1.0, 0.001, 1.0, None, None, None),
    ("greedy-tie", 1000, 1.0, None, 1.0, None, None, "greedy"),
]


def compact(x: np.float32) -> float:
    """The float32 value through its shortest decimal (which reads back as the same float32)."""
    return float(repr(np.float32(x)).removeprefix("np.float32(").removesuffix(")"))


def warper_case(name, vocab, penalty, allowed_fraction, temperature, top_k, top_p, tie, seed) -> dict:
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        row = rng.normal(0.0, 2.5, vocab).astype(np.float32)
        row[rng.choice(vocab, 3, replace=False)] += 6.0
        history = rng.integers(0, vocab, 48).tolist()
        history += history[:8]  # repeated tokens count once
        if tie == "top-k":
            order = np.argsort(-row, kind="stable")
            row[order[top_k - 1 : top_k + 2]] = row[order[top_k - 1]]  # three tokens tied at the k-th logit
        if tie == "greedy":
            top = int(np.argmax(row))
            twin = (top + 17) % vocab
            row[twin] = row[top]  # the lower index must win
        allowed = None
        if allowed_fraction is not None:
            allowed = sorted(rng.choice(vocab, max(1, int(vocab * allowed_fraction)), replace=False).tolist())
        scores = torch.from_numpy(row.copy())[None]
        input_ids = torch.tensor(history)[None]
        if penalty != 1.0:
            scores = RepetitionPenaltyLogitsProcessor(penalty)(input_ids, scores)
        if allowed is not None:
            scores = PrefixConstrainedLogitsProcessor(lambda batch, ids: allowed, num_beams=1)(input_ids, scores)
        greedy = int(torch.argmax(scores))
        scores = TemperatureLogitsWarper(temperature)(input_ids, scores)
        if top_k is not None:
            scores = TopKLogitsWarper(top_k)(input_ids, scores)
        if top_p is not None:
            before = scores.double().softmax(-1)[0].numpy()
            descending = np.sort(before[before > 0])[::-1]
            above = np.concatenate([[0.0], np.cumsum(descending)[:-1]])
            if np.min(np.abs(above - top_p)) < 2e-5:
                continue
            scores = TopPLogitsWarper(top_p)(input_ids, scores)
        probabilities = torch.softmax(scores, -1)[0].numpy()
        kept = np.nonzero(probabilities > 0)[0]
        return {
            "name": name, "seed": seed, "logits": [compact(x) for x in row], "history": history,
            "repetitionPenalty": penalty, "allowed": allowed, "temperature": temperature, "topK": top_k, "topP": top_p,
            "greedy": greedy, "kept": kept.tolist(), "probabilities": [compact(probabilities[i]) for i in kept],
        }
    raise RuntimeError(f"{name}: no row clear of the top-p boundary")


def warper_cases() -> dict:
    return {
        "transformers": importlib.metadata.version("transformers"), "torch": importlib.metadata.version("torch"),
        "note": "float32 logits; probabilities are torch.softmax of the processed float32 scores",
        "cases": [warper_case(*case, seed=1000 + index) for index, case in enumerate(WARPER_CASES)],
    }


def write(path: Path, payload: dict, indent: int | None = 1) -> None:
    separators = None if indent else (",", ":")
    path.write_text(json.dumps(payload, indent=indent, separators=separators, sort_keys=True) + "\n")
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write(args.output_dir / "random-generator.json", generator_vectors())
    write(args.output_dir / "transformers-warpers.json", warper_cases(), indent=None)


if __name__ == "__main__":
    main()

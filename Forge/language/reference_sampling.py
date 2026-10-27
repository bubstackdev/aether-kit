#!/usr/bin/env python3
"""Independent references for AetherLanguageRuntime's sampler.

`random-generator.json`: known-answer outputs of SplitMix64 and of xoshiro256** seeded by it, from this file's own
implementation of the published algorithms, so the Swift generator is checked against code it doesn't share.

Writes only into `--output-dir`. Development tooling: Python is absent from shipping targets.

    Forge/.venv/bin/python -I Forge/language/reference_sampling.py --output-dir Tests/AetherLanguageRuntimeTests/Fixtures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write(args.output_dir / "random-generator.json", generator_vectors())


if __name__ == "__main__":
    main()

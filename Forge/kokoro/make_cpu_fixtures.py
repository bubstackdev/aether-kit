#!/usr/bin/env python3
"""Writes the small CPU fixture used by AetherSpeechTests (no model execution)."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kokoro_graphs import HARMONICS, SAMPLES_PER_FRAME, splitmix_normal, tokenize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=Path)
    ap.add_argument("--snapshot", type=Path, required=True)
    args = ap.parse_args()
    vocab = json.loads((args.snapshot / "config.json").read_text())["vocab"]
    phonemes = ["həlˈO wˈɜɹld, ðɪs ɪz kˈOkəɹO.", "ðə kwˈɪk bɹˈWn fˈɑks ʤˈʌmps.", "a\u0303", "ʦ↗"]
    noise_count = 98 * SAMPLES_PER_FRAME * HARMONICS
    noise = splitmix_normal(24000, noise_count)
    rng = np.random.default_rng(7)
    sums = (rng.random(40, dtype=np.float32) * 12).astype(np.float32)
    sums[:4] = np.array([0.2, 2.5, 3.5, 1.5000001], dtype=np.float32)
    cases = []
    for speed in (1.0, 0.8, 1.3):
        frames = torch.round(torch.from_numpy(sums) / speed).clamp(min=1).long().tolist()
        cases.append({"speed": speed, "frames": frames})
    fixture = {
        "source": "Forge/kokoro/make_cpu_fixtures.py",
        "vocabulary": vocab,
        "tokenization": [{"phonemes": p, "ids": tokenize(vocab, p)} for p in phonemes],
        "noise": {"seed": 24000, "count": noise_count, "first": [float(v) for v in noise[:8]],
                  "sha256": hashlib.sha256(noise.astype("<f4").tobytes()).hexdigest()},
        "durations": {"sums": [float(v) for v in sums], "cases": cases},
    }
    args.output.write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

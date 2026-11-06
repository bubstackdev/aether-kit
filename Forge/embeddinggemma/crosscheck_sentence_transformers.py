#!/usr/bin/env python3
"""Cross-checks the references against sentence-transformers itself (optional; not in the Forge lock).

    uv run --project Forge --frozen --with sentence-transformers==6.2.0 \
        python -I Forge/embeddinggemma/crosscheck_sentence_transformers.py --reference <reference dir> [--report <json>]

reference_embeddinggemma.py re-expresses the sentence-transformers head (pooling, dense, normalize) and its tokenization
rather than importing the package, which would add scikit-learn, scipy and more to the shared Forge environment. This
script runs the real `SentenceTransformer` from the pinned snapshot (CPU, float32, SDPA, max_seq_length 256) in an
ephemeral uv overlay and requires, per case: its tokenization of prompt + text equals the reference's `input_ids`
(through the last real token), and `encode(text, prompt_name=role)` has cosine >= 0.999999 with the reference
embedding. Exits nonzero when a case fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import embeddinggemma_graphs as eg  # noqa: E402

BAR = 0.999999


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--snapshot", type=Path, default=eg.DEFAULT_SNAPSHOT)
    args = ap.parse_args()
    eg.verify_sources(args.snapshot)
    import importlib.metadata

    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(args.snapshot), device="cpu",
                                model_kwargs={"attn_implementation": "sdpa", "dtype": torch.float32})
    model.max_seq_length = eg.SEQUENCE
    index = json.loads((args.reference / "references.json").read_text())
    shapes = index["shapes"]
    results = []
    for entry in index["cases"]:
        folder = args.reference / entry["name"]
        case = json.loads((folder / "case.json").read_text())
        want = eg.read_f32(folder / "embedding.f32", shapes["embedding"])
        ids = eg.read_i32(folder / "input_ids.i32", shapes["input_ids"])[0, : case["keptTokens"]]
        tokens = model.tokenize([case["prompt"] + case["text"]])["input_ids"][0].numpy()
        got = model.encode([case["text"]], prompt_name=case["role"], convert_to_numpy=True)
        c = eg.cosine(got, want)
        same = bool(np.array_equal(tokens, ids))
        passed = same and c >= BAR
        results.append({"name": case["name"], "idsEqual": same, "cosine": c, "oneMinusCosine": 1 - c,
                        "maxAbs": float(np.abs(got - want).max()), "passed": passed})
        print(f"[crosscheck] {'PASS' if passed else 'FAIL'} {case['name']}: ids equal {same}, 1 - cosine {1 - c:.3e}",
              file=sys.stderr, flush=True)
    passed = all(r["passed"] for r in results)
    if args.report:
        args.report.write_text(json.dumps({
            "passed": passed, "bar": BAR, "reference": str(args.reference),
            "versions": {p: importlib.metadata.version(p) for p in ("sentence-transformers", "transformers", "torch")},
            "cases": results}, indent=2) + "\n")
    if not passed:
        raise SystemExit("sentence-transformers cross-check failed")


if __name__ == "__main__":
    main()

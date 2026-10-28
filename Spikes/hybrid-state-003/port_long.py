#!/usr/bin/env python3
"""Hybrid-state 0.0.3 spike: an 8,192-token drift reference from Forge's eager float32 port.

Upstream's one-token path takes about 0.07-0.15 s per token on the CPU, so the upstream long reference
(reference_granite_hybrid.py) stops at 4,096 tokens. This feeds the same document further through the port in 64-token
chunks; the port matches upstream to 4.9e-5 (maxAbs) over the upstream rows (results/reference-port-check.json).
Writes a reference directory with only the `long` section (no chat cases), which the harness reads with
`--phases long`.

    Forge/.venv/bin/python -I Spikes/hybrid-state-003/port_long.py --checkpoint <snapshot> --output <new dir> [--tokens 8192]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Forge" / "language"))
from granite_hybrid import GraniteHybridDecoder, load_weights, read_config  # noqa: E402
from reference_granite_hybrid import long_document, sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--every", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output directory already exists")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    document, files = long_document(ROOT)
    tokens = tokenizer(document, add_special_tokens=False)["input_ids"][:args.tokens]
    rows = sorted(set(list(range(args.every - 1, len(tokens), args.every)) + [len(tokens) - 1]))
    module = GraniteHybridDecoder(read_config(args.checkpoint))
    load_weights(module, args.checkpoint)
    module = module.float().eval()
    kc, vc, cs, ss = module.empty_state(len(tokens) + 64)
    got = {}
    with torch.no_grad():
        for start in range(0, len(tokens), 64):
            chunk = tokens[start:start + 64]
            logits = module.step(torch.tensor([chunk], dtype=torch.int32),
                                 torch.arange(start, start + len(chunk), dtype=torch.int32)[None], kc, vc, cs, ss)
            if start + len(chunk) - 1 in rows:
                got[start + len(chunk) - 1] = logits.double().numpy()
    long = np.stack([got[r] for r in rows])
    args.output.mkdir(parents=True)
    path = args.output / "long.f32"
    long.astype("<f4").tofile(path)
    record = {"schema": 1, "kind": "granite-hybrid-port-long-reference",
              "oracle": "Forge granite_hybrid eager float32, 64-token chunks", "cases": [],
              "long": {"files": files, "tokens": tokens, "rows": rows, "file": path.name, "dtype": "float32-le",
                       "vocab": int(long.shape[1]), "sha256": sha256(path), "rowMeaning": "row r = logits after tokens[:r + 1]"}}
    (args.output / "reference.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"{len(tokens)} tokens, rows {rows}")


if __name__ == "__main__":
    main()

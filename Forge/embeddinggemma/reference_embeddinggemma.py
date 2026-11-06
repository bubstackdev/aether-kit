#!/usr/bin/env python3
"""Upstream-PyTorch EmbeddingGemma references for the Core AI gate (check_embeddinggemma.py) and the Swift driver gate.

    Forge/.venv/bin/python -I Forge/embeddinggemma/reference_embeddinggemma.py --output <new reference dir>

Runs transformers' unmodified Gemma3TextModel (pinned weights, SDPA attention) and the sentence-transformers head
(mean pooling, Dense, Dense, Normalize) on the CPU in float32, on exactly the tensors the exported graph is fed: the
prompted text tokenized as sentence-transformers does at max_seq_length 256, right-padded with id 0 to 256. Tokenization
is not under test here: each case's ids and mask are written out, and gates feed them as is (the host tokenizer is gated
separately against `input_ids.i32`). Each case also runs upstream on the unpadded ids (sentence-transformers' own path
for one text), which must agree with the padded run to cosine >= 0.999999. For each case:

    <case>/case.json            role, prompt, text, token count, truncation, SHA-256 of each file
    <case>/input_ids.i32        [1,256] little-endian int32 (<bos> ... <eos>, then 0s)
    <case>/attention_mask.i32   [1,256] little-endian int32 (1 for real tokens, 0 for padding)
    <case>/embedding.f32        upstream embedding [1,768] little-endian float32, unit length

and an index `references.json`, with the query x document cosine matrix. Cases: three queries and four documents from
a few tokens up to 252 tokens, one non-English, and one document over 256 tokens (truncated). CPU only.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import embeddinggemma_graphs as eg  # noqa: E402

LIGHTHOUSE = (
    "The lighthouse keeper's log for the winter of the great storm runs to nearly forty pages. Each entry begins "
    "with the barometer reading, the direction and strength of the wind, and the state of the sea, recorded in a "
    "small and careful hand. On the third night the glass fell faster than the keeper had ever seen it fall, and he"
    " wrote that the swell was breaking over the lower gallery, more than twenty metres above the rocks. He trimmed"
    " the wicks every two hours, wound the clockwork that turned the great lens, and carried paraffin up the "
    "hundred and twelve steps of the tower by lantern light. Twice the windows of the lamp room cracked under the "
    "spray, and twice he patched them with canvas and tar. A coastal steamer, blown far off its course, sighted the"
    " beam just before dawn on the fourth day and turned away from the reef in time; her captain later sent a "
    "letter of thanks, which the keeper pasted into the back of the log. When the weather finally cleared, he "
    "walked the shore and counted the timbers, crates and barrels that the storm had thrown up on the shingle, and "
    "noted them all, one by one, before sitting down to his first hot meal in a week.")
RELIEF = (
    " The relief boat arrived on the ninth day with fresh water, flour, oil and a sack of letters, and the keeper's "
    "assistant, who had been stranded ashore for the whole of the storm, climbed the ladder to the landing stage "
    "looking, the keeper wrote, as though he had aged ten years in as many days.")

# (name, role, text, expected token range incl. prompt and special tokens)
CASES = [
    ("query-short", "query", "Which planet is known as the Red Planet?", (10, 30)),
    ("query-long", "query", "How did lighthouse keepers keep the lamp burning and warn passing ships during a long "
                            "winter storm, before electric lights and radio beacons were common?", (30, 60)),
    ("query-french", "query", "Quelle est la meilleure façon de conserver le pain frais pendant plusieurs jours ?",
     (15, 40)),
    ("document-short", "document", "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
     (15, 40)),
    ("document-medium", "document",
     "Bread goes stale mostly because its starch recrystallizes, not because it dries out. Keeping a loaf at room "
     "temperature in a paper or cloth bag preserves the crust for a day or two; for longer storage, slice it and "
     "freeze it in an airtight bag, then toast the slices straight from the freezer. The refrigerator is the worst "
     "place for bread, since cold speeds up the recrystallization.", (70, 110)),
    ("document-near-256", "document", LIGHTHOUSE, (245, 256)),
    ("document-truncated", "document", LIGHTHOUSE + RELIEF, (257, 4096)),
]
SHAPES = {"input_ids": [1, eg.SEQUENCE], "attention_mask": [1, eg.SEQUENCE], "embedding": [1, eg.DIMENSION]}
PADDED_VS_UNPADDED_BAR = 0.999999


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--snapshot", type=Path, default=eg.DEFAULT_SNAPSHOT)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    verified = eg.verify_sources(args.snapshot)
    args.output.mkdir(parents=True)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    upstream = eg.Upstream(args.snapshot)
    tok = eg.tokenizer(args.snapshot)
    prompts = eg.prompts(args.snapshot)
    index = {
        "schema": 1, "generator": "Forge/embeddinggemma/reference_embeddinggemma.py",
        "scripts": {p.name: eg.sha256_file(p) for p in sorted(HERE.glob("*.py"))},
        "source": {"repository": eg.REPO, "revision": eg.REVISION, "verified": verified},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "tokenizers", "numpy")},
        "host": {"machine": platform.machine(), "python": platform.python_version()},
        "precision": "float32 on the CPU (transformers Gemma3TextModel, SDPA; sentence-transformers head re-expressed)",
        "shapes": SHAPES, "layout": "row-major, little-endian (int32 ids and mask, float32 embedding)",
        "prompts": prompts,
        "tokenization": ("(prompt + text).strip(), special tokens added, truncation='longest_first' at 256, "
                         "padding='max_length' (right, id 0)"),
        "cases": [],
    }
    embeddings = {}
    for name, role, text, (low, high) in CASES:
        started = time.monotonic()
        out = args.output / name
        out.mkdir()
        prompted = prompts[role] + text
        full, _ = eg.encode(tok, prompted, sequence=1 << 16, pad=False)
        ids, mask = eg.encode(tok, prompted)
        ids_unpadded, mask_unpadded = eg.encode(tok, prompted, pad=False)
        tokens = int(full.shape[1])
        if not low <= tokens <= high:
            raise SystemExit(f"{name}: {tokens} tokens, outside the case's intended range {low}..{high}")
        kept = int(mask.sum())
        if ids[0, 0] != 2 or ids[0, kept - 1] != 1:
            raise SystemExit(f"{name}: the encoded sequence does not start with <bos> and end with <eos>")
        embedding = upstream(torch.from_numpy(ids), torch.from_numpy(mask)).numpy()
        unpadded = upstream(torch.from_numpy(ids_unpadded), torch.from_numpy(mask_unpadded)).numpy()
        agreement = eg.cosine(embedding, unpadded)
        if not (np.isfinite(embedding).all() and abs(float(np.linalg.norm(embedding)) - 1) < 1e-5):
            raise SystemExit(f"{name}: upstream embedding is not finite and unit length")
        if agreement < PADDED_VS_UNPADDED_BAR:
            raise SystemExit(f"{name}: padded vs unpadded upstream cosine {agreement} < {PADDED_VS_UNPADDED_BAR}")
        eg.write_i32(out / "input_ids.i32", ids)
        eg.write_i32(out / "attention_mask.i32", mask)
        eg.write_f32(out / "embedding.f32", embedding)
        embeddings[name] = embedding
        files = ("input_ids.i32", "attention_mask.i32", "embedding.f32")
        record = {"name": name, "role": role, "prompt": prompts[role], "text": text,
                  "tokens": tokens, "keptTokens": kept, "truncated": tokens > eg.SEQUENCE, "shapes": SHAPES,
                  "files": {f: eg.sha256_file(out / f) for f in files},
                  "paddedVsUnpaddedCosine": agreement, "norm": float(np.linalg.norm(embedding)),
                  "seconds": round(time.monotonic() - started, 2)}
        (out / "case.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        index["cases"].append({"name": name, "caseJSONSHA256": eg.sha256_file(out / "case.json")})
        print(f"[reference] {name}: {tokens} tokens (kept {kept}), padded/unpadded cosine {agreement:.9f}, "
              f"{record['seconds']} s", file=sys.stderr, flush=True)
    queries = [n for n, r, *_ in CASES if r == "query"]
    documents = [n for n, r, *_ in CASES if r == "document"]
    matrix = {q: {d: round(eg.cosine(embeddings[q], embeddings[d]), 6) for d in documents} for q in queries}
    # Retrieval sanity: each query's best document is one it was written for (the two lighthouse documents share
    # their first 254 tokens).
    expected = {"query-short": ["document-short"], "query-long": ["document-near-256", "document-truncated"],
                "query-french": ["document-medium"]}
    for q, ds in expected.items():
        best = max(matrix[q], key=matrix[q].get)
        if best not in ds:
            raise SystemExit(f"{q}: best document {best}, expected one of {ds}; check the prompts and pooling")
    index["similarity"] = {"queries": queries, "documents": documents, "cosine": matrix, "expectedBest": expected}
    index["peakRSSGB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
    (args.output / "references.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

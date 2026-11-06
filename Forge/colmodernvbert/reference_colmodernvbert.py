#!/usr/bin/env python3
"""Upstream-PyTorch ColModernVBERT references for the Core AI gate (check_colmodernvbert.py) and the Swift driver gate.

    Forge/.venv/bin/python -I Forge/colmodernvbert/reference_colmodernvbert.py --output <new reference dir>

Runs the unmodified transformers `ColModernVBertForRetrieval` (pinned weights, loaded as sentence-transformers loads
them) on the CPU in float32, on exactly the tensors the exported graphs are fed:
- pages: upstream's own single-tile preprocessing (`ColModernVBertProcessor.process_images(do_image_splitting=False)`:
  Pillow LANCZOS to longest edge 2048, LANCZOS stretch to 512 x 512, normalized to [-1, 1]), with the fixed document
  prompt; the doc graph gets the same float32 tensor;
- queries: the query graph's row (`colmodernvbert_graphs.tokenize_query`: upstream's `process_queries` ids, cut to fit
  64 if longer, right-padded); upstream runs on the real ids, unpadded.
Preprocessing is not under test here: each input is written out, and gates feed it as is. Files:

    pages/<case>/case.json        source, preprocessing, SHA-256 of each file, the expected query
    pages/<case>/input.f32        the doc graph input [1,1,3,512,512], little-endian float32
    pages/<case>/embedding.f32    upstream unit doc vectors [89,128], little-endian float32
    queries/tokens.i32            the query graph rows [N,64], little-endian int32 ([PAD] 50283 after the real ids)
    queries/mask.i32              their attention masks [N,64] (1 for every real id)
    queries/embedding.f32         upstream unit query vectors [N,64,128], zero rows after each query's real count
    references.json               the index: cases, queries (text, count, cut), upstream's MaxSim score matrix
                                  [queries x pages] and each query's ranking; the zoo cross-check; and the
                                  single-stage resize measurement (what a host stretch to 512 with one filter changes)

Cases: six document pages (five public-domain US government pages from Wikimedia Commons and the authors' own results
table), the zoo bundle's test_doc.png, and a seeded synthetic page. Queries: one per real page (upstream must rank its
page first) plus edge cases (empty, digits and punctuation, French, a long query cut to fit). CPU only; no GPU lease.
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
import colmodernvbert_graphs as cg  # noqa: E402

PAGE_CASES = [
    {"name": "nasa-eclipse", "image": "images/nasa-eclipse-fact-sheet.jpg"},
    {"name": "cdc-germs", "image": "images/cdc-stop-the-spread.png"},
    {"name": "cdc-antibiotics", "image": "images/cdc-antibiotic-resistance.jpg"},
    {"name": "fda-nutrition", "image": "images/fda-nutrition-facts.png"},
    {"name": "nasa-pext", "image": "images/nasa-pext-fact-sheet.jpg"},
    {"name": "vidore-table", "image": "upstream/assets/imgs/table.png"},
    {"name": "zoo-test-doc", "image": "zoo/test_doc.png"},
    {"name": "synthetic-7", "synthetic": 7},
]
QUERIES = [
    {"text": "When can the annular solar eclipse be seen in October 2023?", "page": "nasa-eclipse"},
    {"text": "How many feet apart should people stay to stop the spread of germs?", "page": "cdc-germs"},
    {"text": "How do antibiotic-resistant bacteria spread from farm animals to people?", "page": "cdc-antibiotics"},
    {"text": "How many calories and how much sodium are in one serving?", "page": "fda-nutrition"},
    {"text": "NASA polylingual experimental terminal satellite", "page": "nasa-pext"},
    {"text": "Which retriever has the best ViDoRe score among models under one billion parameters?",
     "page": "vidore-table"},
    {"text": ""},
    {"text": "COVID-19: 6 ft, 20 seconds, 2,000 calories?"},
    {"text": "Qu'est-ce qu'une éclipse solaire annulaire ?"},
    {"text": " ".join(["Describe every nutrient, vitamin and mineral listed on the label, with its daily value"] * 4)},
]
# Pillow's filters for the single-stage measurement (Image.Resampling values).
SINGLE_STAGE = {"pillowBicubic": 3, "pillowLanczos": 1, "pillowBilinear": 2}
SHAPES = {"image": [1, 1, 3, cg.SIZE, cg.SIZE], "docEmbedding": [cg.DOC_LENGTH, cg.EMBED_DIM]}


def upstream_ids(model, ids: np.ndarray) -> np.ndarray:
    """Upstream's forward on one query's real ids (unpadded, mask of ones): [n, 128]."""
    t = torch.from_numpy(ids.astype(np.int64))[None]
    with torch.no_grad():
        return model(input_ids=t, attention_mask=torch.ones_like(t)).embeddings[0].numpy()


def zoo_cross_check(model, proc, upstream: Path) -> dict:
    """The zoo's sidecar references against this checkpoint in float32: its four queries with its own tokenization (no
    augmentation suffix, right-padded with [PAD] to 32) and test_doc.png with its prompt ids. Reported, never gated:
    the zoo's numbers are fp16 from colpali_engine at an earlier revision of ModernVBERT/colmodernvbert, and its page
    preprocessing (letterbox into 512 x 512) is not upstream's, so the doc numbers are approximate."""
    from PIL import Image

    zq = json.loads((upstream / "zoo/reference_query.json").read_text())
    zd = json.loads((upstream / "zoo/reference_doc.json").read_text())
    out = {"repository": cg.ZOO_REPO, "revision": cg.ZOO_REVISION, "queries": {}}
    for key, text in zq["queries"].items():
        n = zq["real_lens"][key]
        ids = np.asarray(proc.tokenizer(text).input_ids, dtype=np.int64)[: zq["seq_len"]]
        row = np.full((1, zq["seq_len"]), zq["pad_token_id"], dtype=np.int64)
        row[0, :len(ids)] = ids
        mask = (np.arange(zq["seq_len"]) < len(ids)).astype(np.int64)[None]
        with torch.no_grad():
            ours = model(input_ids=torch.from_numpy(row), attention_mask=torch.from_numpy(mask)).embeddings[0].numpy()
        theirs = np.asarray(zq["query_embeddings"][key], dtype=np.float32)
        c = cg.cosine(ours[:n], theirs[:n])
        out["queries"][key] = {"realTokens": n, "idsLength": int(len(ids)), "cosineMin": float(c.min()),
                               "cosineMean": float(c.mean())}
    ids = zd["input_ids"]
    with Image.open(upstream / "zoo/test_doc.png") as im:
        rgb = im.convert("RGB")
    # The zoo README: longest edge to 512 (aspect kept), padded to 512 x 512; padding is 0 after normalization.
    w, h = rgb.size
    scale = cg.SIZE / max(w, h)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    small = np.asarray(rgb.resize(size, 1), dtype=np.float32).transpose(2, 0, 1) / 255
    x = np.zeros((1, 1, 3, cg.SIZE, cg.SIZE), dtype=np.float32)
    x[0, 0, :, :size[1], :size[0]] = (small - 0.5) / 0.5
    with torch.no_grad():
        t = torch.tensor([ids])
        ours = model(input_ids=t, attention_mask=torch.ones_like(t), pixel_values=torch.from_numpy(x)).embeddings[0]
    theirs = np.asarray(zd["doc_embeddings"], dtype=np.float32)
    c = cg.cosine(ours.numpy(), theirs)
    out["doc"] = {"promptIdsEqual": ids == cg.DOC_IDS, "letterboxSize": list(size), "cosineMin": float(c.min()),
                  "cosineMean": float(c.mean()),
                  "imageRowsCosineMean": float(c[cg.IMAGE_SPAN[0]:cg.IMAGE_SPAN[1]].mean())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--snapshot", type=Path, default=cg.DEFAULT_SNAPSHOT)
    ap.add_argument("--upstream", type=Path, default=cg.DEFAULT_UPSTREAM)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    verified = cg.verify_sources(args.snapshot, args.upstream)
    args.output.mkdir(parents=True)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    model = cg.build_model(args.snapshot)
    proc = cg.processor(args.snapshot)

    # MARK: queries
    started = time.monotonic()
    n = len(QUERIES)
    tokens = np.zeros((n, cg.QUERY_LENGTH), dtype=np.int32)
    masks = np.zeros((n, cg.QUERY_LENGTH), dtype=np.int32)
    query_emb = np.zeros((n, cg.QUERY_LENGTH, cg.EMBED_DIM), dtype=np.float32)
    query_meta = []
    for i, q in enumerate(QUERIES):
        ids, mask, count, cut = cg.tokenize_query(proc.tokenizer, q["text"])
        upstream_row = proc.process_queries([q["text"]])["input_ids"][0].tolist()
        if not cut and upstream_row != ids[0, :count].tolist():
            raise SystemExit(f"query {i}: tokenize_query differs from upstream process_queries")
        if cut and (len(upstream_row) <= cg.QUERY_LENGTH or upstream_row[:count - 11] != ids[0, :count - 11].tolist()):
            raise SystemExit(f"query {i}: the cut row is not upstream's prefix")
        emb = upstream_ids(model, ids[0, :count])
        if not np.isfinite(emb).all():
            raise SystemExit(f"query {i}: non-finite upstream output")
        tokens[i], masks[i] = ids[0], mask[0]
        query_emb[i, :count] = emb
        query_meta.append({"text": q["text"], "count": count, "cut": cut, "upstreamLength": len(upstream_row),
                           "expectedPage": q.get("page")})
    (args.output / "queries").mkdir(parents=True)
    cg.write_i32(args.output / "queries/tokens.i32", tokens)
    cg.write_i32(args.output / "queries/mask.i32", masks)
    cg.write_f32(args.output / "queries/embedding.f32", query_emb)
    query_seconds = round(time.monotonic() - started, 2)

    # MARK: pages
    page_emb, cases, single_stage = {}, [], {}
    for case in PAGE_CASES:
        started = time.monotonic()
        out = args.output / "pages" / case["name"]
        out.mkdir(parents=True)
        if "image" in case:
            path = args.upstream / case["image"]
            x = cg.page_pixels(path, proc)
            group, name = case["image"].split("/", 1)
            source = {"kind": "image", "file": case["image"], "sha256": cg.sha256_file(path),
                      "origin": {"images": "Wikimedia Commons (public domain)",
                                 "upstream": f"{cg.CODE_REPO}@{cg.CODE_COMMIT}",
                                 "zoo": f"{cg.ZOO_REPO}@{cg.ZOO_REVISION}"}[group],
                      "preprocess": "ColModernVBertProcessor.process_images(do_image_splitting=False): Pillow "
                                    "convert('RGB'), LANCZOS to longest edge 2048 (short side even), LANCZOS "
                                    "stretch to 512 x 512, x/255, (x-0.5)/0.5"}
        else:
            path = None
            x = cg.synthetic_pixels(case["synthetic"])
            source = {"kind": "synthetic", "seed": case["synthetic"],
                      "recipe": "colmodernvbert_graphs.synthetic_pixels"}
        emb = cg.upstream_doc(model, x)
        if not np.isfinite(emb).all():
            raise SystemExit(f"{case['name']}: non-finite upstream output")
        if path is not None:
            # Upstream ignores the pixel mask: a mask of zeros gives the same vectors.
            zeros = np.zeros((1, 1, cg.SIZE, cg.SIZE), dtype=np.int64)
            if not np.array_equal(cg.upstream_doc(model, x, zeros), emb):
                raise SystemExit(f"{case['name']}: upstream's output depends on pixel_attention_mask")
            single_stage[case["name"]] = {}
            for filter_name, resample in SINGLE_STAGE.items():
                y = cg.stretch_pixels(path, resample)
                alt = cg.upstream_doc(model, y)
                c = cg.cosine(alt, emb)
                single_stage[case["name"]][filter_name] = {
                    "inputMaxAbs": float(np.abs(y - x).max()), "inputMeanAbs": float(np.abs(y - x).mean()),
                    "cosineMin": float(c.min()), "cosineMean": float(c.mean()), "embedding": alt}
        cg.write_f32(out / "input.f32", x)
        cg.write_f32(out / "embedding.f32", emb)
        page_emb[case["name"]] = emb
        record = {"name": case["name"], "source": source, "shapes": SHAPES,
                  "files": {f: cg.sha256_file(out / f) for f in ("input.f32", "embedding.f32")},
                  "expectedQueries": [i for i, q in enumerate(QUERIES) if q.get("page") == case["name"]],
                  "seconds": round(time.monotonic() - started, 2)}
        (out / "case.json").write_text(json.dumps(record, indent=2) + "\n")
        cases.append({"name": case["name"], "caseJSONSHA256": cg.sha256_file(out / "case.json")})
        print(f"[reference] {case['name']}: {record['seconds']} s", file=sys.stderr, flush=True)

    # MARK: MaxSim
    names = [c["name"] for c in PAGE_CASES]
    scores = np.array([[cg.maxsim(query_emb[i], page_emb[p]) for p in names] for i in range(n)])
    rankings = [[names[j] for j in np.argsort(-scores[i], kind="stable")] for i in range(n)]
    for i, q in enumerate(QUERIES):
        if q.get("page") and rankings[i][0] != q["page"]:
            raise SystemExit(f"upstream ranks {rankings[i][0]!r} first for {q['text']!r}, not {q['page']!r}")
    real = [c["name"] for c in PAGE_CASES if "image" in c]
    measured = {}
    for filter_name in SINGLE_STAGE:
        alt = {p: single_stage[p][filter_name].pop("embedding") for p in real}
        alt_scores = np.array([[cg.maxsim(query_emb[i], alt[p]) for p in real] for i in range(n)])
        ref_scores = scores[:, [names.index(p) for p in real]]
        top_same = [bool(np.argmax(alt_scores[i]) == np.argmax(ref_scores[i])) for i in range(n)]
        measured[filter_name] = {
            "cosineMin": min(single_stage[p][filter_name]["cosineMin"] for p in real),
            "cosineMeanOfMeans": float(np.mean([single_stage[p][filter_name]["cosineMean"] for p in real])),
            "scoreMaxAbsDelta": float(np.abs(alt_scores - ref_scores).max()),
            "top1SameAsUpstream": f"{sum(top_same)}/{n}",
            "perPage": {p: single_stage[p][filter_name] for p in real}}

    index = {
        "schema": 1, "generator": "Forge/colmodernvbert/reference_colmodernvbert.py",
        "scripts": {p.name: cg.sha256_file(p) for p in sorted(HERE.glob("*.py"))},
        "source": {"repository": cg.REPO, "revision": cg.REVISION, "verified": verified},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "transformers", "numpy",
                                                                "pillow")},
        "host": {"machine": platform.machine(), "python": platform.python_version()},
        "precision": "float32 on the CPU (transformers ColModernVBertForRetrieval.forward, unmodified, sdpa attention)",
        "shapes": {**SHAPES, "tokens": [n, cg.QUERY_LENGTH], "queryEmbedding": [n, cg.QUERY_LENGTH, cg.EMBED_DIM]},
        "layout": "row-major, little-endian float32 (.f32) and int32 (.i32)",
        "graphInput": {"image": "pages/<case>/input.f32 as is (float32)",
                       "tokens": "queries/tokens.i32 rows with queries/mask.i32 as attention_mask"},
        "docPrompt": {"ids": cg.DOC_IDS, "imageTokenSpan": list(cg.IMAGE_SPAN), "patchGrid": list(cg.PATCH_GRID)},
        "queries": {"items": query_meta, "seconds": query_seconds,
                    "files": {f: cg.sha256_file(args.output / "queries" / f)
                              for f in ("tokens.i32", "mask.i32", "embedding.f32")},
                    "tokenizer": "transformers ColModernVBertProcessor.process_queries ids (query + 10 "
                                 "<end_of_utterance>, [CLS] ... [SEP]); cut to fit 64 keeping the suffix; "
                                 "right-padded with [PAD] 50283"},
        "maxSim": {"pages": names, "scores": scores.tolist(), "rankings": rankings,
                   "rule": "sum over query rows of max over the 89 doc rows of the dot product (float64)"},
        "zooCrossCheck": zoo_cross_check(model, proc, args.upstream),
        "singleStageResize": {"comment": "Each real page stretched to 512 x 512 in one Pillow resize (as a host "
                                         "ImagePreprocessor stretch does) instead of upstream's two LANCZOS passes; "
                                         "upstream's vectors from that input against the reference's.",
                              "filters": measured},
        "cases": cases,
    }
    index["peakRSSGB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
    (args.output / "references.json").write_text(json.dumps(index, indent=2) + "\n")
    for i, q in enumerate(QUERIES):
        print(f"[reference] {q['text'][:50]!r}: {rankings[i][:3]} {np.round(np.sort(scores[i])[::-1][:3], 3)}",
              file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()

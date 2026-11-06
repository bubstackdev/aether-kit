#!/usr/bin/env python3
"""Eager CPU self-test of the export graphs and the host contract, before any Core AI conversion.

    Forge/.venv/bin/python -I Forge/colmodernvbert/selftest_colmodernvbert.py [--reference <reference dir>] \
        [--report <json>]

- `DocGraph` in float32 equals upstream's `ColModernVBertForRetrieval` (the fixed prompt, `masked_scatter` of the
  image features) bit for bit; upstream's `pixel_attention_mask` changes nothing (a mask of zeros gives the same
  vectors), which is why the doc graph takes none.
- `QueryGraph` in float32 on the right-padded 64-id row matches upstream on the unpadded ids (cosine >= 0.9999999 per
  row: the padded attention changes only the reduction order) and its padding rows are exactly zero.
- The shipped precision (`mixed`: float16 vision tower, float32 ModernBERT), run eagerly, is within the gate's bar;
  the all-float16 alternative is measured (informational: query rows fall below the bar, see `check_colmodernvbert`).
- The host contract the Swift driver implements: `tokenize_query` is upstream's `process_queries` row (and a long query
  keeps the suffix and both specials); the processor's document prompt is `DOC_IDS`, image rows 13..76; the processor
  normalizes as `(x / 255 - 0.5) / 0.5` in float32 and the kit's `(x * (1/255) - 0.5) / 0.5` is within one float32
  ulp (2^-23) of it; and the pixel shuffle puts the page's 64 x 64-pixel cell (r, c) in image row 8 r + c (patchGrid
  [8, 8] row-major).
- The token ids the manifest names (`[CLS]`, `[SEP]`, `[PAD]`, `<end_of_utterance>`, `<image>`) and the sample
  manifest's `late-interaction-retriever@1` roles (MANIFEST.md §1.3).
With `--reference`, the cases are reference_colmodernvbert.py's (upstream's float32 output must reproduce them exactly);
otherwise one seeded synthetic page and two queries. Exits nonzero when a check fails. CPU only; about a minute.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import colmodernvbert_graphs as cg  # noqa: E402
import export_colmodernvbert  # noqa: E402
import reference_colmodernvbert  # noqa: E402
from check_colmodernvbert import COSINE_BAR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--snapshot", type=Path, default=cg.DEFAULT_SNAPSHOT)
    ap.add_argument("--upstream", type=Path, default=cg.DEFAULT_UPSTREAM)
    args = ap.parse_args()
    cg.verify_sources(args.snapshot, args.upstream)
    torch.set_grad_enabled(False)
    checks: list[dict] = []

    def record(name: str, value: float, bar: float, passed: bool) -> None:
        checks.append({"check": name, "value": value, "bar": bar, "passed": bool(passed)})
        print(f"[selftest] {'PASS' if passed else 'FAIL'} {name}: {value} (bar {bar})", file=sys.stderr, flush=True)

    upstream = cg.build_model(args.snapshot)
    proc = cg.processor(args.snapshot)
    tok = proc.tokenizer

    # MARK: cases
    pages, queries = [], []
    if args.reference:
        index = json.loads((args.reference / "references.json").read_text())
        shapes = index["shapes"]
        for entry in index["cases"]:
            folder = args.reference / "pages" / entry["name"]
            pages.append((entry["name"], cg.read_f32(folder / "input.f32", shapes["image"]),
                          cg.read_f32(folder / "embedding.f32", shapes["docEmbedding"])))
        ids = cg.read_i32(args.reference / "queries/tokens.i32", shapes["tokens"])
        masks = cg.read_i32(args.reference / "queries/mask.i32", shapes["tokens"])
        recorded = cg.read_f32(args.reference / "queries/embedding.f32", shapes["queryEmbedding"])
        for i, item in enumerate(index["queries"]["items"]):
            row, mask, count, cut = cg.tokenize_query(tok, item["text"])
            same = (np.array_equal(row[0], ids[i]) and np.array_equal(mask[0], masks[i]) and count == item["count"]
                    and cut == item["cut"])
            record(f"query.{i}.tokens.reproduced", float(same), 1.0, same)
            queries.append((i, row, mask, count, recorded[i]))
        # The reference tensors are upstream's own processor output.
        for name, x, _ in pages:
            case = next(c for c in reference_colmodernvbert.PAGE_CASES if c["name"] == name)
            if "image" in case:
                again = cg.page_pixels(args.upstream / case["image"], proc)
                record(f"page.{name}.input.reproduced", float(np.array_equal(again, x)), 1.0, np.array_equal(again, x))
    else:
        x = cg.synthetic_pixels(3)
        pages.append(("synthetic-3", x, None))
        for i, text in enumerate(["How many calories are in one serving?", "COVID-19: 6 ft, 20 seconds"]):
            row, mask, count, _ = cg.tokenize_query(tok, text)
            queries.append((i, row, mask, count, None))

    # MARK: graphs against upstream
    exact_doc = cg.DocGraph(cg.build_model(args.snapshot), torch.float32, torch.float32).eval()
    exact_query = cg.QueryGraph(cg.build_model(args.snapshot), torch.float32).eval()
    variants = {name: (cg.DocGraph(cg.build_model(args.snapshot), *cg.PRECISIONS[name]).eval(),
                       cg.QueryGraph(cg.build_model(args.snapshot), cg.PRECISIONS[name][1]).eval())
                for name in ("mixed", "float16")}
    for name, x, recorded in pages:
        want = cg.upstream_doc(upstream, x)
        if recorded is not None:
            same = np.array_equal(want, recorded)
            record(f"doc.{name}.reference.reproduced", float(same), 1.0, same)
        zeros = np.zeros((1, 1, cg.SIZE, cg.SIZE), dtype=np.int64)
        same = np.array_equal(cg.upstream_doc(upstream, x, zeros), want)
        record(f"doc.{name}.pixelMaskIgnored", float(same), 1.0, same)
        got = exact_doc(torch.from_numpy(x))[0].numpy()
        record(f"doc.{name}.float32.exact", float(np.array_equal(got, want)), 1.0, np.array_equal(got, want))
        for variant, (doc, _) in variants.items():
            c = float(cg.cosine(doc(torch.from_numpy(x))[0].numpy(), want).min())
            bar = COSINE_BAR if variant == "mixed" else 0.0
            record(f"doc.{name}.{variant}.cosine.min", round(c, 7), bar, c >= bar)
    for i, row, mask, count, recorded in queries:
        want = reference_colmodernvbert.upstream_ids(upstream, row[0, :count])
        if recorded is not None:
            same = np.array_equal(want, recorded[:count]) and not recorded[count:].any()
            record(f"query.{i}.reference.reproduced", float(same), 1.0, same)
        got = exact_query(torch.from_numpy(row), torch.from_numpy(mask))[0].numpy()
        c = float(cg.cosine(got[:count], want).min())
        record(f"query.{i}.float32.cosine.min", round(c, 9), 0.9999999, c >= 0.9999999)
        zero = bool((got[count:] == 0).all())
        record(f"query.{i}.float32.padding.zero", float(zero), 1.0, zero)
        for variant, (_, query) in variants.items():
            c = float(cg.cosine(query(torch.from_numpy(row), torch.from_numpy(mask))[0].numpy()[:count], want).min())
            bar = COSINE_BAR if variant == "mixed" else 0.0
            record(f"query.{i}.{variant}.cosine.min", round(c, 7), bar, c >= bar)

    # MARK: host contract
    text = "How far apart should people stay?"
    row, mask, count, cut = cg.tokenize_query(tok, text)
    want = proc.process_queries([text])["input_ids"][0].tolist()
    same = row[0, :count].tolist() == want and not cut and (row[0, count:] == cg.PAD).all() and mask.sum() == count
    record("tokenizer.processQueries", float(same), 1.0, bool(same))
    long = " ".join(["alpha beta gamma delta"] * 30)
    row, mask, count, cut = cg.tokenize_query(tok, long)
    full = proc.process_queries([long])["input_ids"][0].tolist()
    same = (cut and count == cg.QUERY_LENGTH and row[0, :count - 11].tolist() == full[:count - 11]
            and row[0, count - 11:count].tolist() == full[-11:])
    record("tokenizer.bodyOnlyTruncation", float(same), 1.0, bool(same))
    facts = (tok.convert_tokens_to_ids(["[CLS]", "[SEP]", "[PAD]", "<end_of_utterance>", "<image>"])
             == [cg.CLS, cg.SEP, cg.PAD, cg.END_OF_UTTERANCE, cg.IMAGE_TOKEN]
             and upstream.vlm.config.image_token_id == cg.IMAGE_TOKEN
             and tok(cg.AUGMENTATION, add_special_tokens=False).input_ids == [cg.END_OF_UTTERANCE] * 10)
    record("tokenizer.ids", float(facts), 1.0, facts)

    from PIL import Image
    with Image.open(args.upstream / "images/fda-nutrition-facts.png") as im:
        page = im.convert("RGB").resize((cg.SIZE, cg.SIZE), Image.Resampling.BICUBIC)
    batch = proc.process_images([page], do_image_splitting=False, do_resize=False)
    prompt = batch["input_ids"][0].tolist() == cg.DOC_IDS
    record("docPrompt.ids", float(prompt), 1.0, prompt)
    pixels = np.asarray(page, dtype=np.float32).transpose(2, 0, 1)[None, None]
    upstream_rule = (pixels / np.float32(255) - np.float32(0.5)) / np.float32(0.5)
    same = np.array_equal(upstream_rule, batch["pixel_values"].numpy())
    record("preprocess.normalization.upstreamRule.exact", float(same), 1.0, same)
    # The kit multiplies by scale (1/255) where upstream divides by 255: at most one float32 ulp of 1 apart.
    host = (pixels * np.float32(1 / 255) - np.float32(0.5)) / np.float32(0.5)
    gap = float(np.abs(host - batch["pixel_values"].numpy()).max())
    record("preprocess.normalization.kitRule.maxAbs", gap, 2 ** -23, gap <= 2 ** -23)

    connector = upstream.vlm.connector
    side = cg.SIZE // cg.PATCH
    hits = []
    for r, c in [(0, 0), (0, 7), (3, 5), (7, 0), (7, 7)]:
        vision = torch.zeros(1, side * side, connector.modality_projection.in_features // cg.SHUFFLE ** 2)
        grid = vision.view(1, side, side, -1)
        grid[0, r * cg.SHUFFLE:(r + 1) * cg.SHUFFLE, c * cg.SHUFFLE:(c + 1) * cg.SHUFFLE] = 1.0
        rows = connector(vision)[0].abs().sum(-1).nonzero().flatten().tolist()
        hits.append(rows == [r * cg.PATCH_GRID[1] + c])
    record("patchGrid.rowMajor", float(all(hits)), 1.0, all(hits))

    sample = export_colmodernvbert.sample_manifest("macos")
    q_spec, d_spec = sample["graphs"]["queryEncoder"], sample["graphs"]["docEncoder"]
    retrieval = sample["postprocess"]["retrieval"]
    roles = (sample["driver"] == "late-interaction-retriever@1"
             and q_spec["inputs"] == {"tokens": "tokens", "attentionMask": "attention_mask"}
             and q_spec["outputs"] == {"tokenEmbeddings": "token_embeddings"}
             and d_spec["inputs"] == {"image": "image"} and d_spec["outputs"] == {"tokenEmbeddings": "token_embeddings"}
             and export_colmodernvbert.QUERY_IO == (["tokens", "attention_mask"], ["token_embeddings"])
             and export_colmodernvbert.DOC_IO == (["image"], ["token_embeddings"])
             and sample["postprocess"]["kind"] == "maxSim"
             and retrieval == {"imageTokenSpan": list(cg.IMAGE_SPAN), "patchGrid": list(cg.PATCH_GRID)}
             and retrieval["patchGrid"][0] * retrieval["patchGrid"][1] == cg.IMAGE_SPAN[1] - cg.IMAGE_SPAN[0]
             and sample["tokenizer"]["text"]["padTo"] == q_spec["limits"]["staticSequence"] == cg.QUERY_LENGTH
             and sample["preprocess"]["image"]["layout"] == "NVCHW" and sample["preprocess"]["image"]["views"] == 1)
    record("manifest.roles", float(roles), 1.0, roles)
    passed = all(c["passed"] for c in checks)
    if args.report:
        args.report.write_text(json.dumps({"passed": passed, "checks": checks}, indent=2) + "\n")
    if not passed:
        raise SystemExit("self-test failed")


if __name__ == "__main__":
    main()

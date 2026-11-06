#!/usr/bin/env python3
"""Eager CPU self-test of the export graph, before any Core AI conversion.

    Forge/.venv/bin/python -I Forge/gliner/selftest_gliner.py [--reference <reference dir>] [--report <json>]

- In float32, `ClassifierGraph` (static S, the precomputed relative-position table, the Python twins of DeBERTa's
  TorchScript helpers, int32 I/O) equals upstream's own forward (transformers DebertaV2Model on the same padded inputs,
  the marker gather, gliner2's classifier) bit for bit, at both buckets.
- The shipped recipe (float16 weights and compute), run eagerly, is within the gate's bars against the float32
  reference (probabilities within 0.01, same argmax and decision); its deltas are recorded. This is the CPU half of the
  precision decision; check_gliner.py measures the GPU.
- The helpers the export substitutes give the TorchScript originals' values; the manifest's compact word pattern splits
  like gliner2's WhitespaceTokenSplitter.
- `--sample-manifest`'s manifest names the `gliner-classifier@1` roles (MANIFEST.md §1.3), the graph's I/O names, the
  sequence buckets, the `tokenizer.text` contract, upstream's marker ids and `postprocess.kind: gliner`.
With `--reference`, the cases are reference_gliner.py's model-card, synthetic and longest cases (and upstream's padded
float32 logits must reproduce the recorded ones within 1e-4); otherwise one case laid out by upstream here. Exits
nonzero when a check fails. CPU only; a few minutes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gliner_graphs as gg  # noqa: E402
import export_gliner  # noqa: E402
from check_gliner import PROB_BAR  # noqa: E402

SPLIT_SAMPLES = ["Email me at Jane.Doe+news@example.co.uk or see https://ex.am/ple?q=1 now!",
                 "@handle_42 said: state-of-the-art re_use ¥5,400 (approx.) www.example.org/x",
                 "Ünïcödé İstanbul naïve café — 3.14 vs 2,718; tab\tand\nnewline"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--snapshot", type=Path, default=gg.DEFAULT_SNAPSHOT)
    ap.add_argument("--code", type=Path, default=gg.DEFAULT_CODE)
    ap.add_argument("--long", type=int, default=4, help="the longest N fast-decisions cases of each bucket")
    args = ap.parse_args()
    gg.verify_sources(args.snapshot, args.code)
    torch.set_grad_enabled(False)
    checks: list[dict] = []

    def record(name: str, value: float, bar: float | None, passed: bool) -> None:
        checks.append({"check": name, "value": value, "bar": bar, "passed": bool(passed)})
        print(f"[selftest] {'PASS' if passed else 'FAIL'} {name}: {value} (bar {bar})", file=sys.stderr, flush=True)

    upstream = gg.load_upstream(args.snapshot, args.code)
    cases = []
    if args.reference:
        index = json.loads((args.reference / "references.json").read_text())
        picked = [c for c in index["cases"] if c["set"] in ("readme21", "synthetic")]
        for s in gg.SEQUENCES:
            longest = sorted((c for c in index["cases"] if c["bucket"] == s and c["set"].startswith("fast")),
                             key=lambda c: -c["seqLen"])
            picked += longest[:args.long]
        cases = [(c["name"], np.asarray(c["inputIds"], np.int32), np.asarray(c["labelPositions"], np.int32),
                  c["bucket"], c["results"]) for c in picked]
    else:
        run = gg.run_upstream(upstream, "The parcel arrived two weeks late and the box was crushed.",
                              {"sentiment": ["positive", "negative", "neutral"],
                               "issue": {"labels": ["late", "damaged", "wrong item"], "multi_label": True}})
        cases.append(("local/parcel", run["input_ids"], run["labelPositions"], gg.bucket_for(len(run["input_ids"])),
                      run["tasks"]))

    exact = {s: gg.ClassifierGraph(upstream, s, torch.float32) for s in gg.SEQUENCES}
    half = {gg.SEQUENCES[0]: gg.ClassifierGraph(upstream, gg.SEQUENCES[0], gg.DTYPES[export_gliner.DTYPE])}
    for s in gg.SEQUENCES[1:]:
        half[s] = gg.ClassifierGraph(upstream, s, gg.DTYPES[export_gliner.DTYPE], share=half[gg.SEQUENCES[0]])
    worst_half = {"prob": 0.0, "logit": 0.0}
    with gg.traceable_deberta():
        for name, ids, positions, bucket, results in cases:
            m = len(positions)
            want = np.concatenate([np.asarray(t["logits"], np.float32) for t in results])
            for s in [x for x in gg.SEQUENCES if x >= bucket]:
                feeds = gg.graph_inputs(ids, positions, s)
                reference = gg.upstream_logits(upstream, *feeds)
                tensors = tuple(torch.from_numpy(x) for x in feeds)
                same = np.array_equal(exact[s](*tensors).numpy(), reference)
                record(f"{name}@s{s}.float32.exact", float(same), 1.0, same)
                if args.reference:
                    d = float(np.abs(reference[0, :m] - want).max())
                    record(f"{name}@s{s}.reference.reproduced", d, 1e-4, d <= 1e-4)
                if s != bucket:
                    continue
                got = half[s](*tensors).numpy()
                metrics = gg.task_metrics(got, results)
                p = max(x["maxAbsProb"] for x in metrics)
                worst_half["prob"] = max(worst_half["prob"], p)
                worst_half["logit"] = max(worst_half["logit"], max(x["maxAbsLogit"] for x in metrics))
                ok = bool(np.isfinite(got).all()) and p <= PROB_BAR and all(
                    x["sameArgmax"] and x["sameDecision"] for x in metrics)
                record(f"{name}@s{s}.{export_gliner.DTYPE}.maxAbsProb", round(p, 7), PROB_BAR, ok)
    record(f"{export_gliner.DTYPE}.worstMaxAbsLogit (informational)", round(worst_half["logit"], 6), None, True)

    from transformers.models.deberta_v2 import modeling_deberta_v2 as mdv
    q = torch.zeros(16, 7, 64)
    same = bool(torch.equal(gg._scaled_size_sqrt(q, 3), mdv.scaled_size_sqrt(q, 3)))
    rel = torch.arange(49).reshape(1, 7, 7)
    same &= bool(torch.equal(gg._build_rpos(q, q, rel, 256, 512), mdv.build_rpos(q, q, rel, 256, 512)))
    record("helpers.equal", float(same), 1.0, same)

    gliner2 = gg.import_gliner2(args.code)
    from gliner2.processing.word_splitter import WhitespaceTokenSplitter   # noqa: E402 (the pinned unpack)
    del gliner2
    compact = re.compile(export_gliner.WORD_PATTERN, re.IGNORECASE)
    splitter = WhitespaceTokenSplitter()
    same = all([(m.group().lower(), m.start(), m.end()) for m in compact.finditer(t)] == list(splitter(t))
               for t in SPLIT_SAMPLES)
    record("manifest.wordPattern", float(same), 1.0, same)

    sample = export_gliner.sample_manifest("macos")
    graph = sample["graphs"]["classifier"]
    roles = (sample["driver"] == "gliner-classifier@1"
             and graph["inputs"] == {"tokens": "input_ids", "attentionMask": "attention_mask",
                                     "labelPositions": "label_idx"}
             and graph["outputs"] == {"labelLogits": "logits"}
             and export_gliner.INPUTS == ["input_ids", "attention_mask", "label_idx"]
             and export_gliner.OUTPUTS == ["logits"]
             and graph["buckets"]["axis"] == "sequence" and graph["buckets"]["rule"] == "smallestFitting"
             and [e["value"] for e in graph["buckets"]["entries"]] == list(gg.SEQUENCES)
             and graph["limits"]["staticSequence"] == list(gg.SEQUENCES)
             and graph["function"] == graph["buckets"]["entries"][0]["function"]
             and sample["postprocess"] == {**sample["postprocess"], "kind": "gliner",
                                           "gliner": {"threshold": gg.THRESHOLD}})
    record("manifest.roles", float(roles), 1.0, roles)
    text = sample["tokenizer"]["text"]
    contract = (text["padId"] == gg.PAD_ID and text["padSide"] == "right" and "padTo" not in text
                and text["truncation"] == "word" and sample["tokenizer"]["addSpecialTokens"] is False
                and sample["tokenizer"]["markerIds"] == gg.marker_ids(upstream))
    record("manifest.tokenizer", float(contract), 1.0, contract)
    passed = all(c["passed"] for c in checks)
    if args.report:
        args.report.write_text(json.dumps({"passed": passed, "cases": len(cases), "float16Worst": worst_half,
                                           "checks": checks}, indent=2) + "\n")
    if not passed:
        raise SystemExit("self-test failed")


if __name__ == "__main__":
    main()

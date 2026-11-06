#!/usr/bin/env python3
"""Eager CPU self-test of the export graph, before any Core AI conversion.

    Forge/.venv/bin/python -I Forge/embeddinggemma/selftest_embeddinggemma.py [--reference <reference dir>] \
        [--report <json>]

- In float32 with float32 weights, `EmbeddingGraph` (static 256 sequence, precomputed bidirectional masks, int32 I/O)
  equals upstream's own forward (transformers Gemma3TextModel with the 2-D mask, then the sentence-transformers head)
  bit for bit, so the mask substitution changes nothing.
- The shipped recipe (float16-stored weights widened to float32 compute), run eagerly, is within the gate's cosine bar
  (check_embeddinggemma.py); its 1 - cosine is recorded.
- The float16-compute alternative is measured and recorded (informational, never failing): whether its output is
  finite and, if so, its 1 - cosine. The export's choice of float32 compute rests on this measurement.
- `--sample-manifest`'s manifest names the `text-embedder@1` roles (MANIFEST.md §1.3), the graph's I/O names, the
  `tokenizer.text` contract (pad 0, right, 256, both prompts from upstream's config) and `postprocess.kind`.
With `--reference`, the cases are reference_embeddinggemma.py's (and upstream's float32 output must reproduce them
exactly); otherwise one tokenized query. Exits nonzero when a check fails. CPU only; under a minute.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import embeddinggemma_graphs as eg  # noqa: E402
import export_embeddinggemma  # noqa: E402
from check_embeddinggemma import COSINE_BAR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--snapshot", type=Path, default=eg.DEFAULT_SNAPSHOT)
    args = ap.parse_args()
    eg.verify_sources(args.snapshot)
    torch.set_grad_enabled(False)
    checks: list[dict] = []

    def record(name: str, value: float, bar: float, passed: bool) -> None:
        checks.append({"check": name, "value": value, "bar": bar, "passed": bool(passed)})
        print(f"[selftest] {'PASS' if passed else 'FAIL'} {name}: {value} (bar {bar})", file=sys.stderr, flush=True)

    prompts = eg.prompts(args.snapshot)
    cases = []
    if args.reference:
        index = json.loads((args.reference / "references.json").read_text())
        for entry in index["cases"]:
            folder = args.reference / entry["name"]
            cases.append((entry["name"], eg.read_i32(folder / "input_ids.i32", index["shapes"]["input_ids"]),
                          eg.read_i32(folder / "attention_mask.i32", index["shapes"]["attention_mask"]),
                          eg.read_f32(folder / "embedding.f32", index["shapes"]["embedding"])))
    else:
        ids, mask = eg.encode(eg.tokenizer(args.snapshot), prompts["query"] + "How do tides form?")
        cases.append(("query-tides", ids, mask, None))

    upstream = eg.Upstream(args.snapshot)
    exact = eg.EmbeddingGraph(copy.deepcopy(upstream), torch.float32, torch.float32).eval()
    shipped = eg.EmbeddingGraph(copy.deepcopy(upstream), eg.DTYPES[export_embeddinggemma.COMPUTE_DTYPE],
                                eg.DTYPES[export_embeddinggemma.WEIGHTS]).eval()
    half = eg.EmbeddingGraph(copy.deepcopy(upstream), torch.float16).eval()
    for name, ids, mask, recorded in cases:
        feeds = (torch.from_numpy(ids), torch.from_numpy(mask))
        want = upstream(*feeds).numpy()
        same = np.array_equal(exact(*feeds).numpy(), want)
        record(f"{name}.float32.exact", float(same), 1.0, same)
        if recorded is not None:
            again = np.array_equal(want, recorded)
            record(f"{name}.reference.reproduced", float(again), 1.0, again)
        c = eg.cosine(shipped(*feeds).numpy(), want)
        record(f"{name}.fp16weights.oneMinusCosine", float(1 - c), round(1 - COSINE_BAR, 6), c >= COSINE_BAR)
        h = half(*feeds).numpy()
        finite = bool(np.isfinite(h).all())
        record(f"{name}.float16.finite", float(finite), 0.0, True)
        if finite:
            ch = eg.cosine(h, want)
            record(f"{name}.float16.oneMinusCosine", float(1 - ch), round(1 - COSINE_BAR, 6), True)

    sample = export_embeddinggemma.sample_manifest("macos")
    graph = sample["graphs"]["encoder"]
    text = sample["tokenizer"]["text"]
    roles = (sample["driver"] == "text-embedder@1"
             and graph["inputs"] == {"tokens": "input_ids", "attentionMask": "attention_mask"}
             and graph["outputs"] == {"embedding": "embedding"}
             and export_embeddinggemma.INPUTS == ["input_ids", "attention_mask"]
             and export_embeddinggemma.OUTPUTS == ["embedding"]
             and graph["limits"]["staticSequence"] == eg.SEQUENCE
             and sample["postprocess"]["kind"] == "embedding"
             and sample["postprocess"]["embedding"]["pooling"] == "inGraph")
    record("manifest.roles", float(roles), 1.0, roles)
    contract = (text["padId"] == eg.PAD_ID and text["padSide"] == "right" and text["padTo"] == eg.SEQUENCE
                and export_embeddinggemma.SAMPLE_PROMPTS == prompts == text["prompts"]
                and sample["tokenizer"]["addSpecialTokens"] is True)
    record("manifest.tokenizerText", float(contract), 1.0, contract)
    passed = all(c["passed"] for c in checks)
    if args.report:
        args.report.write_text(json.dumps({"passed": passed, "checks": checks}, indent=2) + "\n")
    if not passed:
        raise SystemExit("self-test failed")


if __name__ == "__main__":
    main()

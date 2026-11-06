#!/usr/bin/env python3
"""Eager CPU self-test of the export graph and the precision ladder, before any Core AI conversion.

    Forge/.venv/bin/python -I Forge/shieldstral/selftest_shieldstral.py --reference <reference dir> \
        [--report <json>] [--recipes float16/float16 float16/int8 ...] [--full-model]

- The vocabulary's yes/no variants are YES_IDS/NO_IDS (upstream's rule over all 131072 ids), and the canonical
  `yes`/`no` are among them.
- In float32, `ClassifierGraph` (static 512 grid, in-graph causal + padding mask, last-token gather, variant head,
  softmax, int32 I/O) reproduces the reference probabilities (upstream's padded forward and the graph's rule in float64)
  within 5e-6 on every case (the float32 head's rounding), so the mask, gather and head substitutions change nothing.
- With `--full-model`, transformers' whole `Mistral3ForConditionalGeneration` (vision tower included, loaded by
  `from_pretrained` from the pinned snapshot, float32) gives the same last-position logits as the harness's text-only
  `Upstream` on one case (max |delta| <= 1e-4), so loading only `language_model.model.*` is upstream's text path.
- Each recipe in `--recipes` (`compute/weights`; default: the shipped one) is measured eagerly on every case: P(yes)
  within the gate's 0.01 and the same decision as the reference (the shipped recipe must pass; the others are
  recorded, never failing), whether its output is finite, and its worst |delta P(yes)|.
- `--sample-manifest`'s manifest names the `yes-no-classifier@1` roles (MANIFEST.md §1.3), the graph's I/O names, the
  `tokenizer.text` contract (pad 11, right, 512, bodyOnly, the template, no added special tokens) and `postprocess`.
Exits nonzero when a check fails. CPU only; a few minutes per recipe, up to ~30 GB of memory.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_shieldstral  # noqa: E402
import shieldstral_graphs as sg  # noqa: E402
from check_shieldstral import PROB_BAR  # noqa: E402

# The graph's head and softmax are float32 (a ~2e-6 logit rounding against the float64 reference head); measured
# max 1.1e-6 in probability over the 25 cases. The hidden state itself is bit-identical to upstream's padded forward.
EXACT_BAR = 5e-6
FULL_MODEL_BAR = 1e-4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--snapshot", type=Path, default=sg.DEFAULT_SNAPSHOT)
    shipped = f"{export_shieldstral.COMPUTE_DTYPE}/{export_shieldstral.WEIGHTS}"
    ap.add_argument("--recipes", nargs="*", default=[shipped])
    ap.add_argument("--full-model", action="store_true")
    ap.add_argument("--skip-exact", action="store_true", help="skip the float32 graph check (recipe runs only)")
    args = ap.parse_args()
    sg.verify_sources(args.snapshot)
    torch.set_grad_enabled(False)
    checks: list[dict] = []

    def record(name: str, value: float, bar: float, passed: bool, **extra) -> None:
        checks.append({"check": name, "value": value, "bar": bar, "passed": bool(passed), **extra})
        print(f"[selftest] {'PASS' if passed else 'FAIL'} {name}: {value} (bar {bar})", file=sys.stderr, flush=True)

    tok = sg.tokenizer(args.snapshot)
    labels = sg.label_ids(tok)
    ok = labels == (sg.YES_IDS, sg.NO_IDS) and 13059 in sg.YES_IDS and 2649 in sg.NO_IDS
    record("labels.variants", float(ok), 1.0, ok, yes=[tok.convert_ids_to_tokens(i) for i in labels[0]],
           no=[tok.convert_ids_to_tokens(i) for i in labels[1]])

    index = json.loads((args.reference / "references.json").read_text())
    cases = []
    for entry in index["cases"]:
        folder = args.reference / entry["name"]
        case = json.loads((folder / "case.json").read_text())
        cases.append((entry["name"], torch.from_numpy(sg.read_i32(folder / "input_ids.i32", index["shapes"]["input_ids"])),
                      torch.from_numpy(sg.read_i32(folder / "attention_mask.i32",
                                                   index["shapes"]["attention_mask"])),
                      np.array(case["probs"], dtype=np.float64)))

    upstream = sg.Upstream(args.snapshot)
    if args.full_model:
        from transformers import Mistral3ForConditionalGeneration
        full = Mistral3ForConditionalGeneration.from_pretrained(str(args.snapshot), dtype=torch.float32,
                                                                attn_implementation="sdpa").eval()
        name, ids, mask, _ = cases[0]
        n = int(mask.sum())
        want = full(input_ids=ids[:, :n].long()).logits[0, -1]
        got = upstream.logits(ids[:, :n])[-1]
        delta = float((want - got).abs().max())
        record(f"{name}.fullModel.maxAbsLogit", delta, FULL_MODEL_BAR, delta <= FULL_MODEL_BAR)
        del full
        gc.collect()

    if not args.skip_exact:
        exact = sg.ClassifierGraph(copy.deepcopy(upstream), sg.YES_IDS, sg.NO_IDS, torch.float32).eval()
        for name, ids, mask, want in cases:
            got = exact(ids, mask).double().numpy()[0]
            delta = float(np.abs(got - want).max())
            record(f"{name}.float32.graphVsReference", delta, EXACT_BAR, delta <= EXACT_BAR)
        del exact
        gc.collect()

    example = (cases[0][1], cases[0][2])
    recipes = {}
    for recipe in args.recipes:
        compute, weights = recipe.split("/")
        started = time.monotonic()
        dtype, storage = sg.DTYPES[compute], sg.DTYPES.get(weights)
        graph = sg.ClassifierGraph(copy.deepcopy(upstream), sg.YES_IDS, sg.NO_IDS, dtype,
                                   storage if storage is not None and storage != dtype else None).eval()
        if weights in export_shieldstral.QUANT_SCHEMES:
            graph, _ = export_shieldstral.quantize(graph, example, weights)
        gating = recipe == shipped
        worst, rows = 0.0, []
        for name, ids, mask, want in cases:
            got = graph(ids, mask).double().numpy()[0]
            finite = bool(np.isfinite(got).all())
            delta = float(np.abs(got - want).max()) if finite else float("inf")
            same = bool(finite and (got[1] > sg.THRESHOLD) == (want[1] > sg.THRESHOLD))
            worst = max(worst, delta)
            rows.append({"case": name, "pYes": float(got[1]), "pYesReference": float(want[1]), "delta": delta})
            record(f"{name}.{recipe}.maxAbs", delta, PROB_BAR, (delta <= PROB_BAR and same) or not gating,
                   sameDecision=same, finite=finite, gating=gating)
        recipes[recipe] = {"worstAbsDelta": worst, "seconds": round(time.monotonic() - started, 1), "cases": rows}
        print(f"[selftest] recipe {recipe}: worst |dP| {worst:.6f} in {recipes[recipe]['seconds']} s",
              file=sys.stderr, flush=True)
        del graph
        gc.collect()

    sample = export_shieldstral.sample_manifest("macos")
    graph = sample["graphs"]["classifier"]
    text = sample["tokenizer"]["text"]
    roles = (sample["driver"] == "yes-no-classifier@1"
             and graph["inputs"] == {"tokens": "input_ids", "attentionMask": "attention_mask"}
             and graph["outputs"] == {"probs": "probs"}
             and export_shieldstral.INPUTS == ["input_ids", "attention_mask"]
             and export_shieldstral.OUTPUTS == ["probs"]
             and graph["limits"]["staticSequence"] == sg.SEQUENCE
             and sample["postprocess"]["kind"] == "yesNo"
             and sample["postprocess"]["probs"] == {"positiveIndex": 1, "threshold": 0.5})
    record("manifest.roles", float(roles), 1.0, roles)
    reference_template = {k: index["template"][k] for k in ("prefix", "body", "suffix")}
    contract = (text["padId"] == sg.PAD_ID == 11 and text["padSide"] == "right" and text["padTo"] == sg.SEQUENCE
                and text["truncation"] == "bodyOnly" and text["maskRule"] == "realTokens"
                and text["template"] == reference_template
                and sample["tokenizer"]["addSpecialTokens"] is False and sample["tokenizer"]["chatTemplate"] is None)
    record("manifest.tokenizerText", float(contract), 1.0, contract)
    passed = all(c["passed"] for c in checks)
    if args.report:
        args.report.write_text(json.dumps({"passed": passed, "recipes": recipes, "checks": checks}, indent=2) + "\n")
    if not passed:
        raise SystemExit("self-test failed")


if __name__ == "__main__":
    main()

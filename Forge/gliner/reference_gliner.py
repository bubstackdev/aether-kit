#!/usr/bin/env python3
"""Upstream GLiNER2.5-Decide references for the Core AI gate (check_gliner.py) and the Swift driver gate.

    Forge/.venv/bin/python -I Forge/gliner/reference_gliner.py --output <new reference dir> [--zoo <zoo snapshot>]

Runs gliner2 2.0.0's own `classify_text` (the pinned wheel, imported read-only; transformers' DebertaV2Model) on the
CPU in float32 for every case, and captures what its classification path computed: the laid-out input ids, the [L]
marker positions and the classifier's logits per task (`gliner_graphs.run_upstream`). Its decisions must equal
classify_text's own answer. Each case is also replayed through upstream's modules on the graph's padded inputs at the
case's bucket (ids right-padded with 0, label_idx padded to 32), which must give the same logits within 1e-4; the
model-card and synthetic cases are replayed at 512 too.

Cases: the zoo bundle's fixtures (mlboydaisuke/GLiNER2.5-Decide-CoreAI `gate/`: the 21 model-card examples, 340
fastino/fast-decisions rows that fit 256 ids and 93 that need 257-512; and its per-bucket reference_s256/512.json), each
cross-checked against the zoo's own gliner2 2.0.0 fp32 oracle (ids and marker positions equal; logits and decisions
reported), plus synthetic cases this export adds: all 32 label slots, label descriptions with a prompt, an empty text
and a text near 512 ids. Writes `references.json`: every case's text, tasks, ids, positions, bucket, per-task logits,
probabilities and decisions, and the zoo comparison. CPU only; about ten minutes.
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
import gliner_graphs as gg  # noqa: E402

PADDED_BAR = 1e-4          # max |logit| between upstream on the padded graph inputs and on the unpadded ids
ZOO_SETS = ("gate/readme21.json", "gate/fast_decisions_s256.json", "gate/fast_decisions_long.json")
ZOO_REFERENCES = ("macos/reference_s256.json", "macos/reference_s512.json")

COLORS = ["red", "orange", "yellow", "green", "blue", "indigo", "violet", "black", "white", "gray", "brown", "pink",
          "purple", "cyan", "magenta", "teal", "maroon", "navy", "olive", "silver", "gold", "beige", "coral",
          "crimson", "lavender", "lime", "mint", "peach", "plum", "rust", "salmon", "turquoise"]
REPORT = ("The quarterly maintenance report for the northern water treatment plant covers the replacement of two "
          "intake pumps, a recalibration of the chlorine dosing system, and an inspection of the sedimentation "
          "basins after the spring floods. Crews found minor cracking in the east basin wall, which was sealed and "
          "will be monitored monthly. The new pumps reduced energy use by about twelve percent compared with the "
          "same quarter last year, and turbidity at the outlet stayed below the regulatory limit on every day of the "
          "period. Two unplanned outages occurred: a power failure during a storm in April, covered by the backup "
          "generators within ninety seconds, and a sensor fault in May that triggered a false alarm and a brief "
          "manual shutdown. ")
CLOSING = ("Next quarter the team will replace the remaining two intake pumps, extend the monitoring of the east basin "
           "to weekly visual checks, and test the new remote telemetry link to the regional control centre. The budget "
           "for these works was approved in March, and no further outages are planned during the summer peak.")
SYNTHETIC = [
    ("synthetic/full-32-labels", "I painted the fence a deep shade of blue with a thin white trim along the top.",
     {"colors": {"labels": COLORS, "multi_label": True}}),
    ("synthetic/descriptions-prompt", "My card was swallowed by the ATM this morning and I need a replacement.",
     {"request": {"labels": {"card_replacement": "The customer needs a new physical card",
                             "pin_change": "The customer wants to change the card PIN",
                             "balance": "The customer asks for the account balance"},
                  "prompt": "What does the customer want?"},
      "urgency": ["low", "medium", "high"]}),
    ("synthetic/empty-text", "", {"sentiment": ["positive", "negative", "neutral"]}),
    ("synthetic/near-512", REPORT * 3 + CLOSING,
     {"topic": ["maintenance", "finance", "sports", "politics", "weather"],
      "outcome": {"labels": ["incident", "routine", "upgrade"], "multi_label": True}}),
]


def zoo_cases(zoo: Path) -> list[dict]:
    """The zoo fixtures as (name, set, text, tasks, zoo oracle) records."""
    cases = []
    for relative in ZOO_SETS:
        fixture = json.loads((zoo / relative).read_text())
        for c in fixture["cases"]:
            positions = [p for s in c["schema_special_indices"] for p in s[1:]]
            oracle = [{"task": t["task"], "logits": t["logits"], "decision": t["decision"]} for t in c["task_results"]]
            cases.append({"name": c["id"], "set": fixture["header"]["set"], "text": c["text"], "tasks": c["tasks"],
                          "zoo": {"inputIds": c["input_ids"], "labelPositions": positions, "tasks": oracle}})
    for relative in ZOO_REFERENCES:
        r = json.loads((zoo / relative).read_text())
        n = int(r["seq_len"])
        positions = [int(p) for t in r["task_slices"] for p in r["label_idx"][t["start"]:t["end"]]]
        oracle = [{"task": t["task"], "logits": t["logits"], "decision": t["decision"]} for t in r["oracle_fp32"]]
        cases.append({"name": f"zoo-reference-s{r['S']}/{r['case_id']}", "set": f"zoo_reference_s{r['S']}",
                      "text": r["text"], "tasks": r["tasks"],
                      "zoo": {"inputIds": r["input_ids"][:n], "labelPositions": positions, "tasks": oracle}})
    return cases


def compare_zoo(run: dict, zoo: dict) -> dict:
    ids_equal = run["input_ids"].tolist() == list(zoo["inputIds"])
    positions_equal = run["labelPositions"].tolist() == list(zoo["labelPositions"])
    by_task = {t["task"]: t for t in zoo["tasks"]}
    worst, decisions = 0.0, True
    for t in run["tasks"]:
        z = by_task[t["task"]]
        worst = max(worst, float(np.abs(np.asarray(t["logits"], np.float64) - np.asarray(z["logits"])).max()))
        decisions &= t["decision"] == z["decision"]
    return {"idsEqual": ids_equal, "positionsEqual": positions_equal, "maxAbsLogit": worst,
            "decisionsEqual": bool(decisions)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--snapshot", type=Path, default=gg.DEFAULT_SNAPSHOT)
    ap.add_argument("--code", type=Path, default=gg.DEFAULT_CODE)
    ap.add_argument("--zoo", type=Path, default=gg.DEFAULT_ZOO)
    ap.add_argument("--limit", type=int, help="at most this many zoo cases per set (a quick run; not for gating)")
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    verified = gg.verify_sources(args.snapshot, args.code)
    zoo_verified = gg.verify_zoo(args.zoo)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    model = gg.load_upstream(args.snapshot, args.code)
    markers = gg.marker_ids(model)
    classifier = json.loads((args.zoo / "macos/classifier.json").read_text())
    if markers != classifier["marker_ids"]:
        raise SystemExit(f"upstream marker ids {markers} differ from the zoo's classifier.json")
    cases = zoo_cases(args.zoo)
    if args.limit:
        kept, seen = [], {}
        for c in cases:
            seen[c["set"]] = seen.get(c["set"], 0) + 1
            if seen[c["set"]] <= args.limit:
                kept.append(c)
        cases = kept
    cases += [{"name": n, "set": "synthetic", "text": t, "tasks": k, "zoo": None} for n, t, k in SYNTHETIC]
    args.output.mkdir(parents=True)
    index = {
        "schema": 1, "generator": "Forge/gliner/reference_gliner.py",
        "scripts": {p.name: gg.sha256_file(p) for p in sorted(HERE.glob("*.py"))},
        "source": {"repository": gg.REPO, "revision": gg.REVISION, "verified": verified},
        "zoo": {"repository": gg.ZOO["repository"], "revision": gg.ZOO["revision"], "verified": zoo_verified},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "tokenizers", "numpy")}
        | {"gliner2": gg.CODE["version"]},
        "host": {"machine": platform.machine(), "python": platform.python_version(), "threads": torch.get_num_threads()},
        "precision": "float32 on the CPU (gliner2 classify_text; transformers DebertaV2Model)",
        "sequences": list(gg.SEQUENCES), "maxLabels": gg.MAX_LABELS, "padId": gg.PAD_ID, "markerIds": markers,
        "inputs": ("graph inputs at a bucket S: inputIds right-padded with padId to S, attention mask 1 for the ids "
                   "and 0 after, label_idx = labelPositions padded to maxLabels with labelPositions[0]; a task's "
                   "logits are slots start ..< start + len(labels)"),
        "cases": [],
    }
    started_all = time.monotonic()
    for i, case in enumerate(cases):
        started = time.monotonic()
        run = gg.run_upstream(model, case["text"], case["tasks"])
        n = int(run["input_ids"].shape[0])
        bucket = gg.bucket_for(n)
        want = np.concatenate([np.asarray(t["logits"], np.float32) for t in run["tasks"]])
        m = len(want)
        padded = {}
        for s in sorted({bucket, *([512] if case["set"] in ("readme21", "synthetic") else [])}):
            got = gg.upstream_logits(model, *gg.graph_inputs(run["input_ids"], run["labelPositions"], s))[0, :m]
            padded[str(s)] = float(np.abs(got - want).max())
            if padded[str(s)] > PADDED_BAR:
                raise SystemExit(f"{case['name']}: padded vs unpadded upstream logits differ by {padded[str(s)]} at S={s}")
        record = {"name": case["name"], "set": case["set"], "text": case["text"], "tasks": case["tasks"],
                  "seqLen": n, "bucket": bucket, "labels": m, "inputIds": run["input_ids"].tolist(),
                  "labelPositions": run["labelPositions"].tolist(), "results": run["tasks"],
                  "classifyText": run["classifyText"], "paddedVsUnpaddedMaxAbsLogit": padded,
                  "seconds": round(time.monotonic() - started, 2)}
        if case["zoo"] is not None:
            record["zooOracle"] = compare_zoo(run, case["zoo"])
        index["cases"].append(record)
        z = record.get("zooOracle")
        print(f"[reference] {i + 1}/{len(cases)} {case['name']}: {n} ids (s{bucket}), {m} labels"
              + (f", zoo ids {z['idsEqual']} positions {z['positionsEqual']} |dlogit| {z['maxAbsLogit']:.2e} "
                 f"decisions {z['decisionsEqual']}" if z else "") + f", {record['seconds']} s",
              file=sys.stderr, flush=True)
    zoo_records = [c["zooOracle"] for c in index["cases"] if "zooOracle" in c]
    index["summary"] = {
        "cases": len(index["cases"]),
        "byBucket": {str(s): sum(c["bucket"] == s for c in index["cases"]) for s in gg.SEQUENCES},
        "maxLabels": max(c["labels"] for c in index["cases"]),
        "maxSeqLen": max(c["seqLen"] for c in index["cases"]),
        "paddedVsUnpaddedMaxAbsLogit": max(v for c in index["cases"] for v in c["paddedVsUnpaddedMaxAbsLogit"].values()),
        "zoo": {"cases": len(zoo_records), "idsEqual": sum(z["idsEqual"] for z in zoo_records),
                "positionsEqual": sum(z["positionsEqual"] for z in zoo_records),
                "decisionsEqual": sum(z["decisionsEqual"] for z in zoo_records),
                "maxAbsLogit": max((z["maxAbsLogit"] for z in zoo_records), default=0.0)},
        "seconds": round(time.monotonic() - started_all, 1),
        "peakRSSGB": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),
    }
    (args.output / "references.json").write_text(json.dumps(index, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(index["summary"], indent=1), file=sys.stderr)
    bad = [c["name"] for c in index["cases"] if "zooOracle" in c
           and not (c["zooOracle"]["idsEqual"] and c["zooOracle"]["positionsEqual"])]
    if bad:
        raise SystemExit(f"{len(bad)} cases do not reproduce the zoo's ids or positions: {bad[:5]}")


if __name__ == "__main__":
    main()

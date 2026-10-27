"""Upstream reference probabilities at a greedy divergence past the reference's forced rows.

Upstream Qwen3VLForConditionalGeneration, float32, CPU, on the reference record's own pixels, grid and prompt ids: one
forward pass over prompt + the shared greedy tokens, softmax of the last row.

usage: divergence_margin.py CHECKPOINT REFERENCE_DIR CASE DIVERGENCE KIT_TOKEN KIT_REPORT OUTPUT
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

checkpoint, reference_dir, case_name, divergence, kit_token, kit_report, output = sys.argv[1:]
divergence, kit_token = int(divergence), int(kit_token)
reference_dir = Path(reference_dir)
record = json.loads((reference_dir / "reference.json").read_text())
case = next(c for c in record["cases"] if c["name"] == case_name)
ids, greedy, grid = case["promptTokens"], case["greedyTokens"], case["grid"]
pixels = torch.from_numpy(np.fromfile(reference_dir / case["pixels"]["file"], dtype="<f4")
                          .reshape(case["pixels"]["shape"]).copy())
kit = json.loads(Path(kit_report).read_text())
kit_case = next(c for c in kit["cases"] if c["name"] == case_name)
shared = greedy[:divergence]
assert kit_case["greedyTokens"][:divergence] == shared, "Kit and reference do not share the prefix"
assert kit_case["greedyTokens"][divergence] == kit_token

processor = AutoProcessor.from_pretrained(checkpoint)
model = Qwen3VLForConditionalGeneration.from_pretrained(checkpoint, dtype=torch.float32).eval()
input_ids = torch.tensor([ids + shared])
with torch.no_grad():
    logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), pixel_values=pixels,
                   image_grid_thw=torch.tensor([grid])).logits[0, -1].double()
probs = torch.softmax(logits, -1)
order = torch.argsort(probs, descending=True).tolist()
ref_token = greedy[divergence]
assert order[0] == ref_token, f"upstream argmax {order[0]} is not the reference greedy token {ref_token}"


def entry(t):
    return [t, round(float(probs[t]), 5), processor.decode([t])]


margin = float(probs[order[0]] - probs[order[1]])
rank = order.index(kit_token)
result = {
    "schema": 1,
    "kind": "qwen3-vl-greedy-divergence-margin",
    "case": case_name,
    "why": (f"Kit's greedy output diverges from upstream at token {divergence}, past the reference's "
            f"{case['forcedLogits']['rows']} forced rows, so the Kit report has no greedyDivergenceReferenceMargin for it"),
    "method": ("upstream Qwen3VLForConditionalGeneration, float32, CPU, on the reference record's pixels, grid and "
               f"prompt ids; one forward pass over prompt + the shared {divergence} greedy tokens; softmax of the last row"),
    "reference": f"{reference_dir.name}/reference.json (case {case_name})",
    "kitReport": Path(kit_report).name,
    "divergence": divergence,
    "referenceToken": {"id": ref_token, "text": processor.decode([ref_token]), "probability": round(float(probs[ref_token]), 5)},
    "kitToken": {"id": kit_token, "text": processor.decode([kit_token]), "probability": round(float(probs[kit_token]), 5),
                 "rankInReference": rank},
    "top5": [entry(t) for t in order[:5]],
    "referenceTop1MinusTop2": round(margin, 5),
    "nearTie": margin < 0.1 and rank < 5,
    "nearTieRule": ("reference top-1 minus top-2 probability < 0.1 and Kit's token in the reference top-5 "
                    "(the image + text gate's rule)"),
}
Path(output).write_text(json.dumps(result, indent=1) + "\n")
print(json.dumps(result, indent=1))

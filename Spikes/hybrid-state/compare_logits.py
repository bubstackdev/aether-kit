"""Teacher-forced comparison of dumped asset logits against the authors' fp32 Transformers model on CPU.

usage: compare_logits.py <results-dir> <out.json>
Each <cell>.json/.logits pair holds 16 rows: the last prompt step, then 15 decode steps. Row i is the logits after
prompt + generated[:i]; the reference is one full-recompute forward over prompt + generated[:15].
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

ckpt = next((Path.home() / "Library/Caches/aether/checkpoints/models--ibm-granite--granite-4.0-h-350m/snapshots")
            .iterdir())
root = Path(sys.argv[1])
prompt = [int(x) for x in (root.parent / "prompt.txt").read_text().split(",")]
model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float32).eval()
cache: dict[tuple, np.ndarray] = {}
out = {}
for meta in sorted(root.glob("*.json")):
    rep = json.loads(meta.read_text())
    rows = np.fromfile(meta.with_suffix(".logits"), dtype=np.float32).reshape(-1, rep["vocab"])
    gen = rep["generated"][: rows.shape[0] - 1]
    key = tuple(gen)
    if key not in cache:
        with torch.no_grad():
            ids = torch.tensor([prompt + gen])
            logits = model(ids).logits[0].float().numpy()
        cache[key] = logits[len(prompt) - 1: len(prompt) - 1 + rows.shape[0]]
    ref = cache[key]
    cos = [float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) for a, b in zip(rows, ref)]
    agree = [int(np.argmax(a)) == int(np.argmax(b)) for a, b in zip(rows, ref)]
    srt = np.sort(ref, axis=1)
    out[meta.stem] = {"minCosine": min(cos), "maxAbs": float(np.abs(rows - ref).max()),
                      "argmaxAgree": f"{sum(agree)}/{len(agree)}",
                      "disagreeMargins": [float(srt[i, -1] - srt[i, -2]) for i, a in enumerate(agree) if not a],
                      "tokPerSecMedian": rep["tokPerSecMedian"], "warmMedianMs": rep["warmMedianMs"],
                      "warmP90Ms": rep["warmP90Ms"], "loadSeconds": rep["loadSeconds"]}
    print(meta.stem, out[meta.stem])
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))

#!/usr/bin/env python3
"""CPU check of the eager Qwen3.5 port (float32, real weights) against the upstream reference, in Forge's environment:

    Forge/.venv/bin/python -I Spikes/qwen35-vl/check_port.py --checkpoint <snapshot> --reference <dir> --out <json>

The port is fed as a host feeds the asset: the prompt in padded W-wide submissions, then one token per padded call.
Compares the teacher-forced rows of every chat case and the long document's rows (cosine, argmax).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Forge" / "language"))
from qwen35_hybrid import build, load_weights, read_config  # noqa: E402


def compare(a, b) -> dict:
    cos = [float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))) for x, y in zip(a, b)]
    return {"minCosine": round(min(cos), 8), "argmaxEqualRows": int(sum(int(np.argmax(x)) == int(np.argmax(y)) for x, y in zip(a, b))),
            "rows": len(a)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--delta-rule", default="chunk")
    a = p.parse_args()
    torch.set_grad_enabled(False)
    ref = json.loads((a.reference / "reference.json").read_text())
    cfg, image_token = read_config(a.checkpoint)
    port = build(cfg, image_token, torch.float32, max_capacity=8192, delta_rule=a.delta_rule, max_width=a.width)
    load_weights(port, a.checkpoint)
    port.eval()
    vocab, w, out, t0 = cfg["vocab_size"], a.width, {}, time.time()

    def feed(tokens, rows, prompt):
        """Logits at `rows`: tokens[:prompt] in W-wide chunks, the rest one per call."""
        state, got, start = port.empty_state(len(tokens) + w), {}, 0
        plan = [tokens[s:min(s + w, prompt)] for s in range(0, prompt, w)] + [[t] for t in tokens[prompt:]]
        for chunk in plan:
            logits = port.step(chunk, start, w, state)
            start += len(chunk)
            if start - 1 in rows:
                got[start - 1] = logits.double().numpy()
            if start - 1 >= max(rows):
                break
        return np.stack([got[r] for r in rows])

    for c in ref["cases"]:
        prompt, forced = c["promptTokens"], c["forcedTokens"]
        want = np.fromfile(a.reference / c["forcedLogits"]["file"], dtype="<f4").reshape(len(forced), vocab)
        rows = list(range(len(prompt) - 1, len(prompt) + len(forced) - 1))
        out[c["name"]] = compare(feed(prompt + forced, rows, len(prompt)), want)
        print(c["name"], out[c["name"]], f"{time.time() - t0:.0f} s", flush=True)
    long = ref["long"]
    want = np.fromfile(a.reference / long["file"], dtype="<f4").reshape(len(long["rows"]), vocab)
    out["long"] = compare(feed(long["tokens"], long["rows"], len(long["tokens"])), want)
    print("long", out["long"], f"{time.time() - t0:.0f} s", flush=True)
    a.out.write_text(json.dumps({"width": w, "deltaRule": a.delta_rule, "dtype": "float32", "results": out}, indent=1) + "\n")


if __name__ == "__main__":
    main()

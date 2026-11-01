#!/usr/bin/env python3
"""Native checks of a Qwen3.5 hybrid text asset (Forge/language/export_qwen35_text.py), in Forge's environment, through
Core AI's Python runtime on the GPU. Run under the GPU lock:

    ~/.cache/aether-kit/with-gpu-lock.sh q35 Forge/.venv/bin/python -I Spikes/qwen35-vl/gpu_check.py tiny <asset> \
        --delta-rule composite --width 8
    ~/.cache/aether-kit/with-gpu-lock.sh q35 Forge/.venv/bin/python -I Spikes/qwen35-vl/gpu_check.py reference <asset> \
        <reference dir> --width 64 --capacity 4608 --out <json>

`tiny`: the random tiny model (rebuilt by the exporter's `tiny_module`) against its eager float32 port, stepping
alike: a 21-token prompt in padded W-wide chunks, then one-token padded steps; logits, KV rows and both states.

`reference`: the real asset against the upstream fp32 reference (reference_qwen35.py): per chat case, teacher-forced
rows (cosine; argmax equal or Kit's near tie: reference top-1 minus top-2 probability < 0.1 and the asset's token in
the reference top 5) and a greedy run as long as the reference's; then the long document's rows (drift).

The host plays Kit's part: KV caches are host arrays at a fixed capacity, passed as read-only inputs, and the valid
keyUpdate/valueUpdate rows are written after each call; the two recurrent states are bound as states, zeroed per
request. This runtime copies the KV arrays per call, so its timings are not Kit's (the Swift session measures those).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Forge" / "language"))


def cosine(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def near_tie(judge: np.ndarray, token: int) -> tuple[bool, float]:
    p = np.exp(judge.astype(np.float64) - judge.max())
    p /= p.sum()
    order = np.argsort(-p)
    margin = float(p[order[0]] - p[order[1]])
    return margin < 0.1 and token in order[:5].tolist(), margin


class Runner:
    """One loaded asset; `reset()` starts a request (zeroed states and KV)."""

    def __init__(self, model, width: int, capacity: int, layout: str = "S2"):
        import coreai.runtime as rt

        self.layout = layout
        self.rt, self.fn, self.width, self.capacity = rt, model.load_function("main"), width, capacity

    def configure(self, kv_shape, conv_shape, rec_shape, hidden: int):
        self.kv_shape, self.conv_shape, self.rec_shape, self.hidden = kv_shape, conv_shape, rec_shape, hidden

    def reset(self):
        rt = self.rt
        self.kc = np.zeros(self.kv_shape, np.float16)
        self.vc = np.zeros(self.kv_shape, np.float16)
        self.states = {"convState": rt.NDArray(np.zeros(self.conv_shape, np.float16)),
                       "recurrentState": rt.NDArray(np.zeros(self.rec_shape, np.float32))}
        self.position = 0

    async def step(self, tokens, rope=None, image=None) -> np.ndarray:
        rt, n, w, start = self.rt, len(tokens), self.width, self.position
        if self.layout == "D1" and n == 1 and start > 0:
            w = 1                                                    # D1's decode width; prefill stays padded at W
        ids = np.zeros((1, w), np.int32)
        ids[0, :n] = tokens
        pos = np.full((1, w), -1, np.int32)
        pos[0, :n] = np.arange(start, start + n)
        rp = np.zeros((3, w), np.int32)
        rp[:, :n] = pos[0, :n] if rope is None else rope
        img = np.zeros((w, self.hidden), np.float16)
        if image is not None:
            img[:n] = image
        if self.layout == "D1":
            pos = np.array([[start, n]], np.int32)
        r = await self.fn({"input_ids": rt.NDArray(ids), "cache_positions": rt.NDArray(pos), "rope_positions": rt.NDArray(rp),
                           "image_embeds": rt.NDArray(img), "keyCache": rt.NDArray(self.kc), "valueCache": rt.NDArray(self.vc)},
                          self.states)
        self.kc[:, :, :, start:start + n] = r["keyUpdate"].numpy()[:, :, :, :n]
        self.vc[:, :, :, start:start + n] = r["valueUpdate"].numpy()[:, :, :, :n]
        self.position += n
        return r["logits"].numpy()[0, -1].astype(np.float32)

    async def prefill(self, tokens, every=None) -> np.ndarray:
        last = None
        for s in range(0, len(tokens), self.width):
            last = await self.step(tokens[s:s + self.width])
            if every:
                every(self.position - 1, last)
        return last


async def load(asset: Path):
    import coreai.runtime as rt

    options = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    t0 = time.time()
    model = await rt.AIModel.load(asset, options)
    return model, time.time() - t0


async def tiny(args) -> dict:
    from export_qwen35_text import tiny_module

    module = tiny_module(args.delta_rule)
    eager = tiny_module("loop").float().eval()                      # the same weights, float32, the reference loop
    model, load_s = await load(args.asset)
    runner = Runner(model, args.width, 48, args.layout)
    kc, vc, cs, rs = eager.empty_state(48)
    runner.configure(kc.shape, cs.shape, rs.shape, module.cfg["hidden_size"])
    runner.reset()
    tokens = np.random.default_rng(7).integers(0, 500, 26).tolist()
    state = (kc, vc, cs, rs)
    rows, start = [], 0
    plan = [tokens[s:s + args.width] for s in range(0, 21, args.width)] + [[t] for t in tokens[21:]]
    for chunk in plan:
        got = await runner.step(chunk)
        with torch.no_grad():
            want = eager.step(chunk, start, args.width, state).numpy()
        start += len(chunk)
        rows.append({"tokens": len(chunk), "cosine": round(cosine(got, want), 6),
                     "argmaxEqual": int(np.argmax(got)) == int(np.argmax(want)),
                     "keyCosine": round(cosine(runner.kc[:, :, :, :start], kc[:, :, :, :start].numpy()), 6),
                     "convStateCosine": round(cosine(runner.states["convState"].numpy(), cs.numpy()), 6),
                     "recurrentStateCosine": round(cosine(runner.states["recurrentState"].numpy(), rs.numpy()), 6)})
    passed = all(r["cosine"] >= 0.999 and r["argmaxEqual"] and r["recurrentStateCosine"] >= 0.999 for r in rows)
    return {"asset": str(args.asset), "loadSeconds": round(load_s, 2), "passed": passed, "steps": rows}


async def reference(args) -> dict:
    ref_dir = args.reference
    ref = json.loads((ref_dir / "reference.json").read_text())
    report = json.loads((args.asset.parent / "export-report.json").read_text())
    cfg = report["config"]
    attn_layers = cfg["layer_types"].count("full_attention")
    lin_layers = cfg["layer_types"].count("linear_attention")
    kv = (attn_layers, 1, cfg["num_key_value_heads"], args.capacity, cfg["head_dim"])
    conv = (lin_layers, 1, 2 * cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
            + cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"], cfg["linear_conv_kernel_dim"] - 1)
    rec = (lin_layers, 1, cfg["linear_num_value_heads"], cfg["linear_key_head_dim"], cfg["linear_value_head_dim"])
    vocab = cfg["vocab_size"]
    model, load_s = await load(args.asset)
    runner = Runner(model, args.width, args.capacity, report["layout"])
    runner.configure(kv, conv, rec, cfg["hidden_size"])
    print(f"[qualify] loaded in {load_s:.1f} s", flush=True)

    def rows_of(name, count):
        return np.fromfile(ref_dir / name, dtype="<f4").reshape(count, vocab)

    cases = []
    for c in ref["cases"]:
        prompt, forced, greedy_ref = c["promptTokens"], c["forcedTokens"], c["greedyTokens"]
        want = rows_of(c["forcedLogits"]["file"], len(forced))
        runner.reset()
        t0 = time.time()
        got = [await runner.prefill(prompt)]
        for t in forced[:-1]:
            got.append(await runner.step([t]))
        cos = [cosine(a, b) for a, b in zip(got, want)]
        mism = [i for i, (a, b) in enumerate(zip(got, want)) if int(np.argmax(a)) != int(np.argmax(b))]
        ties = [i for i in mism if near_tie(want[i], int(np.argmax(got[i])))[0]]
        runner.reset()
        logits = await runner.prefill(prompt)
        generated, divergence = [], None
        for i, expected in enumerate(greedy_ref):
            token = int(np.argmax(logits))
            if divergence is None and token != expected:
                tie, margin = near_tie(logits, expected)
                divergence = {"index": i, "nativeMargin": round(margin, 4), "acceptedNearTie": tie}
            generated.append(token)
            if i + 1 < len(greedy_ref):
                logits = await runner.step([token])
        case = {"name": c["name"], "promptTokens": len(prompt), "forcedRows": len(got), "minCosine": round(min(cos), 6),
                "meanCosine": round(float(np.mean(cos)), 6), "argmaxMismatches": len(mism), "acceptedNearTies": len(ties),
                "forcedPassed": min(cos) >= 0.995 and len(mism) == len(ties), "greedyTokens": len(generated),
                "greedyIdentical": divergence is None, "greedyDivergence": divergence,
                "greedyPassed": divergence is None or divergence["acceptedNearTie"], "generated": generated,
                "seconds": round(time.time() - t0, 1)}
        cases.append(case)
        print(f"[qualify] {c['name']}: min cos {case['minCosine']}, mismatches {len(mism)} (near ties {len(ties)}), "
              f"greedy {'identical' if divergence is None else divergence}", flush=True)

    long = ref["long"]
    want_rows = dict(zip(long["rows"], rows_of(long["file"], len(long["rows"]))))
    got_rows = {}
    runner.reset()
    if len(long["tokens"]) <= args.capacity:
        await runner.prefill(long["tokens"], every=lambda r, l: got_rows.__setitem__(r, l) if r in want_rows else None)
    # Rows that fall inside a chunk are not observable at W > 1; the reference's rows sit at multiples of 256 - 1.
    drift = [{"row": r, "cosine": round(cosine(got_rows[r], want_rows[r]), 6),
              "argmaxEqual": int(np.argmax(got_rows[r])) == int(np.argmax(want_rows[r])),
              "nearTie": near_tie(want_rows[r], int(np.argmax(got_rows[r])))[0]} for r in sorted(got_rows)]
    print(f"[qualify] long: {len(drift)} rows, min cos {min(d['cosine'] for d in drift) if drift else None}", flush=True)
    return {"asset": str(args.asset), "recipe": report["recipe"], "loadSeconds": round(load_s, 2), "width": args.width,
            "capacity": args.capacity, "cases": cases, "long": drift,
            "passed": all(c["forcedPassed"] and c["greedyPassed"] for c in cases)
            and all(d["argmaxEqual"] or d["nearTie"] for d in drift)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["tiny", "reference"])
    parser.add_argument("asset", type=Path)
    parser.add_argument("reference", type=Path, nargs="?")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=4608)
    parser.add_argument("--delta-rule", default="composite")
    parser.add_argument("--layout", default="S2", choices=["S2", "D1"], help="tiny mode (reference mode reads the report)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = asyncio.run(tiny(args) if args.mode == "tiny" else reference(args))
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    print(text if args.mode == "tiny" else json.dumps({"passed": result["passed"]}))


if __name__ == "__main__":
    main()

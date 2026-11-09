#!/usr/bin/env python3
"""Hermetic CPU self-test of the Qwen3 export graph, before any Core AI conversion. No checkpoint, no network.

    Forge/.venv/bin/python -I Forge/qwen3/selftest_qwen3.py [--report <json>]

Builds small random Qwen3 checkpoints with transformers' own `Qwen3ForCausalLM` (one tied head, one untied, GQA, a
rope_theta other than the default), saves them as safetensors, and loads them through the exporter's loader into
`Decoder` in float32. Checks, against transformers' float32 logits of the same token sequence:
- one prefill of the whole prompt gives upstream's last-row logits (rel L2 <= 1e-5, same argmax);
- the running contract over chunks (prefill in pieces, then single-token steps, the KV state carried over in place)
  gives upstream's logits at every fed position (rel L2 <= 1e-5) and the cache rows past the fed ones stay zero;
- a 0.0.x-recipe build (float16 storage and compute, float32 attention) stays close (cosine >= 0.999; informational);
- what the graph doesn't implement is refused: sliding window, rope scaling, attention bias, a MoE config, unused or
  missing tensors;
- the sample manifest has the decoder-lm@1 roles (tokens, positions, last-row logits, one KV sequence state on axis 3).
Exits nonzero when a check fails. A few seconds.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_qwen3  # noqa: E402
import qwen3_graphs as g  # noqa: E402

LOGITS_BAR = 1e-5


def tiny(tied: bool, root: Path) -> Path:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(1 if tied else 2)
    cfg = Qwen3Config(vocab_size=320, hidden_size=64, intermediate_size=96, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=32, max_position_embeddings=512,
                      rope_theta=50000.0, rms_norm_eps=1e-6, tie_word_embeddings=tied)
    model = Qwen3ForCausalLM(cfg).eval()
    with torch.no_grad():                       # random norm gains, so a misplaced gain shows
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.uniform_(0.5, 1.5)
    path = root / ("tied" if tied else "untied")
    model.save_pretrained(path, safe_serialization=True)
    return path


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    checks: list[dict] = []

    def record(name: str, value, bar, passed: bool, gated: bool = True) -> None:
        checks.append({"check": name, "value": value, "bar": bar, "passed": bool(passed), "gated": gated})
        tag = ("PASS" if passed else "FAIL") if gated else "INFO"
        print(f"[selftest] {tag} {name}: {value} (bar {bar})", file=sys.stderr, flush=True)

    from transformers import Qwen3ForCausalLM

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for tied in (True, False):
            name = "tied" if tied else "untied"
            path = tiny(tied, root)
            upstream = Qwen3ForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="eager").eval()
            cfg = g.read_config(path)
            decoder = g.Decoder(cfg, g.load_weights(path), dtype=torch.float32, attention=torch.float32,
                                storage=torch.float32, logits=torch.float32, context=512).eval()
            ids = torch.randint(0, cfg["vocab_size"], (1, 41), generator=torch.Generator().manual_seed(7))
            want = upstream(input_ids=ids.long()).logits[0].numpy()                      # [41, V]

            kc, vc = decoder.empty_state(64)
            got = g.step(decoder, ids[0].tolist(), 0, kc, vc).numpy()
            r = rel_l2(got, want[-1])
            record(f"{name}.prefill.relL2", r, LOGITS_BAR, r <= LOGITS_BAR and got.argmax() == want[-1].argmax())

            # Chunks 17 + 9 (prefill), then single tokens; every fed position's last-row logits against upstream's row.
            kc, vc = decoder.empty_state(48)
            worst, pos = 0.0, 0
            for size in [17, 9] + [1] * 15:
                chunk = ids[0, pos:pos + size].tolist()
                out = g.step(decoder, chunk, pos, kc, vc).numpy()
                pos += size
                worst = max(worst, rel_l2(out, want[pos - 1]))
            record(f"{name}.runningContract.worstRelL2", worst, LOGITS_BAR, worst <= LOGITS_BAR)
            tail = float(max(kc[:, :, :, pos:].abs().max(), vc[:, :, :, pos:].abs().max()))
            record(f"{name}.runningContract.rowsPastFedZero", tail, 0.0, tail == 0.0)

            half = g.Decoder(cfg, g.load_weights(path), context=512).eval()                # the shipped recipe
            kc, vc = half.empty_state(64)
            h = g.step(half, ids[0].tolist(), 0, kc, vc).float().numpy()
            c = cosine(h, want[-1])
            record(f"{name}.float16Recipe.cosine", round(c, 6), 0.999, c >= 0.999, gated=False)

        cfg = g.read_config(root / "tied")
        for key, value in [("use_sliding_window", True), ("rope_scaling", {"rope_type": "yarn", "factor": 4.0}),
                           ("attention_bias", True), ("model_type", "qwen3_moe"), ("hidden_act", "gelu")]:
            try:
                g.check_config({**cfg, key: value})
                refused = False
            except SystemExit:
                refused = True
            record(f"refuses.{key}", refused, True, refused)
        weights = g.load_weights(root / "tied")
        weights["extra.weight"] = torch.zeros(1)
        try:
            g.Decoder(cfg, weights, context=512)
            refused = False
        except SystemExit:
            refused = True
        record("refuses.unusedTensor", refused, True, refused)
        weights = g.load_weights(root / "untied")
        del weights["lm_head.weight"]
        try:
            g.Decoder(g.read_config(root / "untied"), weights, context=512)
            refused = False
        except SystemExit:
            refused = True
        record("refuses.missingHead", refused, True, refused)

    doc = export_qwen3.sample_manifest("macos")
    dec = doc["graphs"]["decoder"]
    roles = bool(doc["driver"] == "decoder-lm@1" and dec["inputs"] == {"tokens": "input_ids", "positions": "position_ids"}
             and dec["outputs"]["logits"] == {"name": "logits", "rows": "last"}
             and dec["states"]["sequence"] == [{"tensors": ["keyCache", "valueCache"], "kind": "kv", "seqAxis": 3}]
             and dec["limits"]["prefillChunk"] == g.PREFILL_MAX and doc["tokenizer"]["stop"] == ["<|im_end|>", "<|endoftext|>"])
    record("sampleManifest.roles", roles, True, roles)

    failed = [c["check"] for c in checks if c["gated"] and not c["passed"]]
    if args.report:
        args.report.write_text(json.dumps({"checks": checks, "failed": failed}, indent=2) + "\n")
    print(json.dumps({"checks": len(checks), "failed": failed}), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

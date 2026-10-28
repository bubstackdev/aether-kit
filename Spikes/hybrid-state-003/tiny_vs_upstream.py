#!/usr/bin/env python3
"""Hybrid-state 0.0.3 spike: Forge's Granite port against upstream Transformers on a random tiny model (CPU, fp32).

1. Every prefill plan against `GraniteMoeHybridForCausalLM` on a 37-token sequence: whole prompt, uneven chunks,
   chunk then one-token steps (SSD form and recurrence), all one-token, and padded fixed widths 8 and 16.
2. The padded-source overflow: with large decay rates (A_log = log 60, dt_bias = 4), a padded submission's SSD decay
   exp(acum_t - acum_s) overflows for padding treated as a causal source; its dx is 0, and inf * 0 is NaN. The port's
   chunk is run as written and with the padded-source mask removed (the code before the fix), against the unpadded
   plan.

    Forge/.venv/bin/python -I Spikes/hybrid-state-003/tiny_vs_upstream.py
"""

from __future__ import annotations

import inspect
import sys
import textwrap
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Forge" / "language"))
import granite_hybrid as gh  # noqa: E402

KW = dict(vocab_size=256, hidden_size=64, intermediate_size=128, shared_intermediate_size=96, num_hidden_layers=4,
          layer_types=["mamba", "attention", "mamba", "mamba"], num_attention_heads=4, num_key_value_heads=2,
          mamba_n_heads=8, mamba_d_head=16, mamba_d_state=16, mamba_expand=2, mamba_n_groups=1, mamba_d_conv=4,
          mamba_chunk_size=256, num_local_experts=0, position_embedding_type="nope", embedding_multiplier=12.0,
          residual_multiplier=0.246, logits_scaling=3.0, attention_multiplier=0.015625, tie_word_embeddings=True,
          rms_norm_eps=1e-5, mamba_conv_bias=True, mamba_proj_bias=False, attention_bias=False)
T = 37


def run(m, ids, plan, recurrent_ones=False, padded=None, cap=64):
    """(row, logits) after each submission; `padded` = fixed width with a padded tail."""
    kc, vc, cs, ss = m.empty_state(cap)
    rows, start = [], 0
    with torch.no_grad():
        for n in plan:
            if padded:
                tok = torch.zeros(1, padded, dtype=torch.int32)
                tok[0, :n] = ids[0, start:start + n]
                pos = torch.full((1, padded), -1, dtype=torch.int32)
                pos[0, :n] = torch.arange(start, start + n)
                logits, ku, vu = m(tok, pos, kc, vc, cs, ss, padded=True)
                kc[:, :, :, start:start + n] = ku[:, :, :, :n]
                vc[:, :, :, start:start + n] = vu[:, :, :, :n]
                out = logits[0, -1]
            else:
                out = m.step(ids[:, start:start + n], torch.arange(start, start + n, dtype=torch.int32)[None], kc, vc, cs, ss,
                             recurrent=recurrent_ones and n == 1 and start > 0)
            start += n
            rows.append((start - 1, out.double()))
    return rows


def main() -> None:
    from transformers import GraniteMoeHybridConfig, GraniteMoeHybridForCausalLM

    torch.manual_seed(0)
    hf = GraniteMoeHybridForCausalLM(GraniteMoeHybridConfig(**KW)).eval()
    with torch.no_grad():
        for n, p in hf.named_parameters():
            if "A_log" in n:
                p.copy_(torch.log(torch.rand_like(p) * 8 + 1))
            elif "dt_bias" in n:
                p.normal_(0, 1)
            elif "norm" in n:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(0, 0.08)
    m = gh.GraniteHybridDecoder(dict(KW)).eval()
    gh.load_state_dict(m, hf.state_dict())
    ids = torch.randint(0, 256, (1, T)).to(torch.int32)
    with torch.no_grad():
        ref = hf(ids.long()).logits[0].double()

    print("1. Port against upstream (max |logits difference| over the rows each plan produces)")
    plans = {"whole prompt": dict(plan=[T]), "chunks 8,8,8,8,5": dict(plan=[8, 8, 8, 8, 5]),
             "16 then 21 x 1, SSD form": dict(plan=[16] + [1] * 21),
             "16 then 21 x 1, recurrence": dict(plan=[16] + [1] * 21, recurrent_ones=True),
             "37 x 1, recurrence": dict(plan=[1] * T, recurrent_ones=True),
             "padded width 8 (8,8,8,8,5+3 pad)": dict(plan=[8, 8, 8, 8, 5], padded=8),
             "padded width 16 (16,16,5+11 pad)": dict(plan=[16, 16, 5], padded=16)}
    for name, kw in plans.items():
        rows = run(m, ids, **kw)
        print(f"   {name:36s} rows {len(rows):2d}  max |diff| {max(float((r - ref[i]).abs().max()) for i, r in rows):.2e}")

    print("2. Padded-source overflow (A_log = log 60, dt_bias = 4)")
    with torch.no_grad():
        for layer in m.layers:
            if layer.kind == "mamba":
                layer.mixer.A_log.fill_(float(torch.log(torch.tensor(60.0))))
                layer.mixer.dt_bias.fill_(4.0)
    plain = run(m, ids, [16, 16, 5])
    fixed = run(m, ids, [16, 16, 5], padded=16)
    source = textwrap.dedent(inspect.getsource(gh.Mamba.chunk))
    assert "causal = causal & valid[None, :]" in source
    namespace = dict(vars(gh))
    exec(source.replace("causal = causal & valid[None, :]", "pass"), namespace)    # the chunk before the fix
    original = gh.Mamba.chunk
    gh.Mamba.chunk = namespace["chunk"]
    try:
        unfixed = run(m, ids, [16, 16, 5], padded=16)
    finally:
        gh.Mamba.chunk = original
    for name, rows in (("as written (padded sources masked)", fixed), ("before the fix (padded sources unmasked)", unfixed)):
        finite = [bool(torch.isfinite(r).all()) for _, r in rows]
        if all(finite):
            diff = max(float((r - p).abs().max()) for (_, r), (_, p) in zip(rows, plain))
            print(f"   {name:42s} logits finite in every submission; max |diff| against unpadded {diff:.2e}")
        else:
            print(f"   {name:42s} non-finite logits from submission {finite.index(False) + 1} of {len(rows)} on")


if __name__ == "__main__":
    main()

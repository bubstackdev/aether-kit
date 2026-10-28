#!/usr/bin/env python3
"""Hybrid-state 0.0.3 spike: CPU exactness of the padded fixed-width prefill (eager float32, real weights).

For each reference chat case and a 1,000-token document prefix, the prompt is fed under several plans and the last
logits row plus both recurrent states are compared with a single whole-prompt submission:
  dynamic   64-token chunks, the remainder in descending powers of two (the C1/C2 plan)
  pad64     64-token submissions, the tail padded to 64 (S1/S2)
  pad16     16-token submissions, the tail padded to 16
The chat cases' teacher-forced rows are also compared with the fp32 reference (reference_granite_hybrid.py), after the
padded 64-wide prefill, with each forced token decoded two ways: one at a time through the recurrence (S1's `decode`),
and as a padded 64-wide submission with one valid token (S2, which has only `main`).

    Forge/.venv/bin/python -I Spikes/hybrid-state-003/padded_exactness.py --checkpoint <snapshot> --reference <dir> --output <json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Forge" / "language"))
from granite_hybrid import GraniteHybridDecoder, load_weights, read_config  # noqa: E402
from reference_granite_hybrid import hybrid_plan  # noqa: E402


def feed(m, tokens, widths, padded_width=None, cap=1280):
    kc, vc, cs, ss = m.empty_state(cap)
    start, last = 0, None
    with torch.no_grad():
        for n in widths:
            chunk = tokens[start:start + n]
            if padded_width:
                ids = torch.zeros(1, padded_width, dtype=torch.int32)
                ids[0, :n] = torch.tensor(chunk)
                pos = torch.full((1, padded_width), -1, dtype=torch.int32)
                pos[0, :n] = torch.arange(start, start + n)
                logits, ku, vu = m(ids, pos, kc, vc, cs, ss, padded=True)
                kc[:, :, :, start:start + n] = ku[:, :, :, :n]
                vc[:, :, :, start:start + n] = vu[:, :, :, :n]
                last = logits[0, -1]
            else:
                last = m.step(torch.tensor([chunk], dtype=torch.int32), torch.arange(start, start + n, dtype=torch.int32)[None],
                              kc, vc, cs, ss)
            start += n
    return last.double(), (kc, vc, cs, ss)


def widths_for(count, w):
    return [w] * (count // w) + ([count % w] if count % w else [])


def compare(a, b):
    la, sa = a
    lb, sb = b
    return {"logitsMaxAbs": float((la - lb).abs().max()),
            "convStateMaxAbs": float((sa[2].double() - sb[2].double()).abs().max()),
            "ssmStateMaxAbs": float((sa[3].double() - sb[3].double()).abs().max()),
            "ssmStateRelative": float((sa[3] - sb[3]).norm() / (sb[3].norm() + 1e-30))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    m = GraniteHybridDecoder(read_config(args.checkpoint))
    load_weights(m, args.checkpoint)
    m = m.float().eval()
    ref = json.loads((args.reference / "reference.json").read_text())
    prompts = {c["name"]: c["promptTokens"] for c in ref["cases"]}
    prompts["document-1000"] = ref["long"]["tokens"][:1000]
    out = {"plans": {}, "forcedRowsVsReference": {}}
    for name, toks in prompts.items():
        whole = feed(m, toks, [len(toks)])
        out["plans"][name] = {"tokens": len(toks),
                              "dynamic": compare(feed(m, toks, hybrid_plan(len(toks))), whole),
                              "pad64": compare(feed(m, toks, widths_for(len(toks), 64), 64), whole),
                              "pad16": compare(feed(m, toks, widths_for(len(toks), 16), 16), whole)}
    for case in ref["cases"]:
        prompt, forced = case["promptTokens"], case["forcedTokens"]
        rows = np.fromfile(args.reference / case["forcedLogits"]["file"], dtype="<f4").reshape(len(forced), -1).astype(np.float64)
        for decode in ("recurrent", "padded64"):
            logits, (kc, vc, cs, ss) = feed(m, prompt, widths_for(len(prompt), 64), 64)
            got = [logits.numpy()]
            with torch.no_grad():
                for i, t in enumerate(forced[:-1]):
                    position = len(prompt) + i
                    if decode == "recurrent":
                        row = m.step(torch.tensor([[t]], dtype=torch.int32), torch.tensor([[position]], dtype=torch.int32),
                                     kc, vc, cs, ss, recurrent=True)
                    else:
                        ids = torch.zeros(1, 64, dtype=torch.int32)
                        ids[0, 0] = t
                        pos = torch.full((1, 64), -1, dtype=torch.int32)
                        pos[0, 0] = position
                        lg, ku, vu = m(ids, pos, kc, vc, cs, ss, padded=True)
                        kc[:, :, :, position:position + 1] = ku[:, :, :, :1]
                        vc[:, :, :, position:position + 1] = vu[:, :, :, :1]
                        row = lg[0, -1]
                    got.append(row.double().numpy())
            cos = [float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) for a, b in zip(got, rows)]
            out["forcedRowsVsReference"][f"{case['name']}/{decode}"] = {
                "plan": f"pad64 prefill, then {'recurrent decode (S1)' if decode == 'recurrent' else 'padded 64-wide one-token decode (S2)'}",
                "rows": len(rows), "minCosine": min(cos), "maxAbs": float(max(np.abs(a - b).max() for a, b in zip(got, rows))),
                "argmaxEqual": sum(int(np.argmax(a)) == int(np.argmax(b)) for a, b in zip(got, rows))}
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

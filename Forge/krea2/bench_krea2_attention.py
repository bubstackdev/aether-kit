#!/usr/bin/env python3
"""One Krea 2 joint attention on the GPU, in several mask formulations (development tooling only).

    Forge/.venv/bin/python -I Forge/krea2/bench_krea2_attention.py --workdir <new dir> --report <new .json>

Each variant is a static graph of q [1, 48, S, 128], k and v [1, 12, S, 128] (fp16, GQA) through
`F.scaled_dot_product_attention`, converted with coreai-torch and run on the GPU; its median run time over `--runs`
calls (after one warm-up) is recorded, and each masked variant's output is compared with the unmasked attention
over the real keys alone (keys past the real rows are padding). Variants:
- `exact`: S = image tokens + real rows, no mask (the exact-length export);
- `padded`: S = image tokens + 512, no mask (what padding alone costs);
- `bool`: S = image + 512 with a boolean key mask [1, 1, 1, S] (the current masked export);
- `bias`: the same with an additive fp16 bias [1, 1, 1, S] (0 real, -1e4 padding);
- `biasFull`: the bias broadcast to [1, 1, S, S] before SDPA;
- `zeroKeys`: no mask; padding keys and values zeroed and the result renormalized by a second SDPA over ones
  (out / out_ones), which keeps the maskless kernel twice.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HEADS, KV, D = 48, 12, 128


class Attention(nn.Module):
    def __init__(self, variant: str, length: int) -> None:
        super().__init__()
        self.variant, self.length = variant, length

    def forward(self, q, k, v, valid):
        scale = D ** -0.5
        if self.variant in ("exact", "padded"):
            return F.scaled_dot_product_attention(q, k, v, scale=scale, enable_gqa=True)
        keep = valid > 0.5
        if self.variant == "bool":
            return F.scaled_dot_product_attention(q, k, v, attn_mask=keep[None, None, None, :], scale=scale,
                                                  enable_gqa=True)
        bias = ((valid - 1.0) * 1e4).to(q.dtype)[None, None, None, :]
        if self.variant == "bias":
            return F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=scale, enable_gqa=True)
        if self.variant == "biasFull":
            full = bias.expand(1, 1, q.shape[2], k.shape[2])
            return F.scaled_dot_product_attention(q, k, v, attn_mask=full, scale=scale, enable_gqa=True)
        if self.variant == "zeroKeys":
            m = valid.to(q.dtype)[None, None, :, None]
            out = F.scaled_dot_product_attention(q, k * m, v * m, scale=scale, enable_gqa=True)
            ones = F.scaled_dot_product_attention(q, k * m, m.expand_as(v), scale=scale, enable_gqa=True)
            return out / ones
        raise ValueError(self.variant)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--image", type=int, default=4096)
    parser.add_argument("--rows", type=int, default=27)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--variants", nargs="*", default=["exact", "padded", "bool", "bias", "biasFull", "zeroKeys"])
    args = parser.parse_args()
    if args.workdir.exists() or args.report.exists():
        raise SystemExit("workdir and report must be new")
    args.workdir.mkdir(parents=True)
    import coreai_torch as ct
    import coreai.runtime as rt

    g = torch.Generator().manual_seed(0)
    padded = args.image + 512
    q = torch.randn(1, HEADS, padded, D, generator=g).half()
    k = torch.randn(1, KV, padded, D, generator=g).half()
    v = torch.randn(1, KV, padded, D, generator=g).half()
    # Keys: the real text rows first, then the padding, then the image tokens (the joint layout).
    valid = torch.ones(padded)
    valid[args.rows:512] = 0
    real = torch.cat([torch.arange(args.rows), torch.arange(512, padded)])
    with torch.no_grad():
        want = F.scaled_dot_product_attention(q.float()[:, :, real], k.float()[:, :, real], v.float()[:, :, real], scale=D ** -0.5,
                                              enable_gqa=True)
    results = {}
    for variant in args.variants:
        exact = variant == "exact"
        feed = ((q[:, :, real], k[:, :, real], v[:, :, real], valid[real]) if exact else (q, k, v, valid))
        module = Attention(variant, feed[0].shape[2]).eval()
        t0 = time.monotonic()
        with torch.no_grad():
            ep = torch.export.export(module, feed, strict=False).run_decompositions(ct.get_decomp_table())
        converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
        converter.add_exported_program(ep, input_names=["q", "k", "v", "valid"], output_names=["out"], state_names=[],
                                       entrypoint_name="main")
        program = converter.to_coreai()
        program.optimize()
        path = args.workdir / f"{variant}.aimodel"
        program.save_asset(path)
        convert = round(time.monotonic() - t0, 1)

        async def go():
            options = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
            model = await rt.AIModel.load(path, options)
            fn = model.load_function("main")
            inputs = {n: rt.NDArray(np.ascontiguousarray(t.numpy())) for n, t in zip(["q", "k", "v", "valid"], feed)}
            times, out = [], None
            for i in range(args.runs + 1):
                t = time.monotonic()
                out = (await fn(inputs, {}))["out"].numpy()
                if i:
                    times.append(time.monotonic() - t)
            return times, out
        times, out = asyncio.run(go())
        got = torch.from_numpy(out.astype(np.float32))
        got = got if exact else got[:, :, real]
        e, o = want.double().ravel(), got.double().ravel()
        results[variant] = {"S": int(feed[0].shape[2]), "convertSeconds": convert,
                            "medianMs": round(1e3 * statistics.median(times), 2),
                            "minMs": round(1e3 * min(times), 2),
                            "cosine": round(float(e @ o / (e.norm() * o.norm())), 6),
                            "maxAbs": float((e - o).abs().max())}
        print(f"[attn] {variant}: {results[variant]}", file=sys.stderr, flush=True)
    args.report.write_text(json.dumps({"image": args.image, "rows": args.rows, "runs": args.runs,
                                       "variants": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()

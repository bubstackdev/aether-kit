#!/usr/bin/env python3
"""Split-attention placement probe (docs/spikes/split-attention-2026-10.md). Development tooling only.

Three bucketed `main` variants over the same random weights (2 layers, Qwen3-0.6B-shaped attention):
  fused  today's graph: one SDPA over [cache || chunk] with a boolean mask;
  split  the cache and the chunk scored separately, with one shared max and denominator (no cache copy), at every q;
  cond   `torch.cond` choosing fused at q == 1 and split at q > 1, in one function.

    Forge/.venv/bin/python -I Spikes/split-attention/split_attention_probe.py eager
    Forge/.venv/bin/python -I Spikes/split-attention/split_attention_probe.py export <fused|split|cond> --output <new .aimodel>
    Forge/.venv/bin/python -I Spikes/split-attention/split_attention_probe.py check <fused asset> <variant asset>

`split-attention-probe.swift` runs an asset on the GPU.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Forge" / "language"))
from qwen3_decode import Qwen3Decoder  # noqa: E402

CFG = {"vocab_size": 8192, "hidden_size": 1024, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
       "num_hidden_layers": 2, "intermediate_size": 3072, "rms_norm_eps": 1e-6, "rope_theta": 1_000_000.0,
       "tie_word_embeddings": True}
NEG = -1e30


def masks(key_cache_layer, new_positions):
    key_positions = torch.cumsum(torch.ones_like(key_cache_layer[0, 0, :, 0], dtype=torch.int32), 0, dtype=torch.int32) - 1
    cache_visible = key_positions[None, :] < new_positions[:1, None]           # [1, cap]
    chunk_visible = new_positions[None, :] <= new_positions[:, None]           # [q, q]
    return cache_visible, chunk_visible


def fused(q, k, v, kc, vc, new_positions):
    cache_visible, chunk_visible = masks(kc, new_positions)
    q_len = q.shape[2]
    visible = torch.cat((cache_visible.expand(q_len, -1), chunk_visible), dim=1)
    keys, values = torch.cat((kc, k), dim=2), torch.cat((vc, v), dim=2)
    return F.scaled_dot_product_attention(q, keys, values, attn_mask=visible, enable_gqa=q.shape[1] != k.shape[1]).contiguous()


def split(q, k, v, kc, vc, new_positions):
    """Scores against the read-only cache and against the chunk, one shared max, one shared denominator. Batch-free
    3-D bmm with query heads grouped by KV head ([kv, g*q, D]), so neither the cache nor the chunk is copied."""
    cache_visible, chunk_visible = masks(kc, new_positions)
    heads, q_len, dim = q.shape[1], q.shape[2], q.shape[3]
    kv_heads = k.shape[1]
    groups = heads // kv_heads
    qg = q[0].reshape(kv_heads, groups * q_len, dim)
    scale = dim ** -0.5
    sc = torch.bmm(qg, kc[0].transpose(1, 2)).to(torch.float32) * scale                  # [kv, g*q, cap]
    sc = torch.where(cache_visible, sc, NEG)
    sk = torch.bmm(qg, k[0].transpose(1, 2)).to(torch.float32) * scale                   # [kv, g*q, q]
    sk = torch.where(chunk_visible.repeat(groups, 1), sk, NEG)
    m = torch.maximum(sc.amax(-1, keepdim=True), sk.amax(-1, keepdim=True))
    pc, pk = torch.exp(sc - m), torch.exp(sk - m)
    denom = pc.sum(-1, keepdim=True) + pk.sum(-1, keepdim=True)
    out = (torch.bmm(pc.to(q.dtype), vc[0]).to(torch.float32) + torch.bmm(pk.to(q.dtype), v[0]).to(torch.float32)) / denom
    return out.to(q.dtype).reshape(1, heads, q_len, dim)


class Probe(Qwen3Decoder):
    def __init__(self, variant: str) -> None:
        super().__init__(CFG, "bucketed", max_capacity=16384)
        self.variant = variant

    def forward(self, input_ids, position_ids, key_cache, value_cache):
        q_len = input_ids.shape[1]
        x = self.embed(input_ids)
        new_positions = position_ids[0]
        cos, sin = self.rope(new_positions, x.dtype)
        keys, values = [], []
        for i, block in enumerate(self.blocks):
            kc, vc = key_cache[i], value_cache[i]

            def layer(attention, block=block):
                def run(x, cos, sin, kc, vc, new_positions):
                    q, k, v = block.project(x, cos, sin)
                    k, v = k.contiguous(), v.contiguous()
                    return block.finish(x, attention(q, k, v, kc, vc, new_positions)).contiguous(), k, v
                return run

            if self.variant == "cond":
                # The cond wraps the whole layer: around attention alone, export fails on the branches' symbolic
                # strides. The predicate is a tensor (contiguous positions: last == first iff q == 1), because a size
                # predicate lowers to a Python `eq` that coreai-torch cannot convert.
                x, k, v = torch.cond(new_positions[-1] == new_positions[0], layer(fused), layer(split), (x, cos, sin, kc, vc, new_positions))
            else:
                x, k, v = layer(fused if self.variant == "fused" else split)(x, cos, sin, kc, vc, new_positions)
            keys.append(k)
            values.append(v)
        return self.head(x[:, -1:]), torch.stack(keys), torch.stack(values)


def build(variant: str) -> Probe:
    torch.manual_seed(0)
    module = Probe(variant)
    with torch.no_grad():
        for n, p in module.named_parameters():
            p.normal_(0, 0.05) if "norm" not in n else p.fill_(1.0)
    return module.to(torch.float16).eval()


def export(variant: str, output: Path) -> None:
    import coreai_torch as ct

    module = build(variant)
    example, dynamic = module.example(512)
    with torch.no_grad():
        # Workaround: the split graph's broadcast strides raise guards export cannot prove (`min(128 q, 256 q) == 128 q`)
        # unless they are deferred to runtime asserts.
        ep = torch.export.export(module, example, dynamic_shapes=dynamic, prefer_deferred_runtime_asserts_over_guards=True)
        ep = ep.run_decompositions(ct.get_decomp_table())
    # Workaround: the deferred shape asserts (Python `eq` on sizes) have no Core AI lowering; they are checks only.
    for gm in [ep.graph_module, *(m for m in ep.graph_module.modules() if isinstance(m, torch.fx.GraphModule))]:
        for n in list(gm.graph.nodes):
            if n.op == "call_function" and n.target in (torch.ops.aten._assert_scalar.default,
                                                         torch.ops.aten.sym_constrain_range_for_size.default):
                gm.graph.erase_node(n)
        gm.graph.eliminate_dead_code()
        gm.recompile()
    hops = [n.target for n in ep.graph.nodes if n.op == "call_function" and "cond" in str(n.target)]
    print(f"[probe] exported, cond nodes: {len(hops)}", flush=True)
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    converter.add_exported_program(ep, input_names=["input_ids", "cache_positions", "keyCache", "valueCache"],
                                   output_names=["logits", "keyUpdate", "valueUpdate"], state_names=[], entrypoint_name="main")
    program = converter.to_coreai()
    print("[probe] converted", flush=True)
    program.optimize()
    print("[probe] optimized", flush=True)
    program.save_asset(output)
    print(f"[probe] saved {output}", flush=True)


def eager_check() -> None:
    """Eager fused vs split on the same weights: chunked prefill and decode against a partly filled cache. (Eager
    `torch.cond` over the lifted weights fails in Dynamo; its branches are exactly these two graphs.)"""
    mods = {v: build(v) for v in ("fused", "split")}
    cap, prompt = 96, 70
    ids = torch.randint(0, CFG["vocab_size"], (1, prompt + 4), generator=torch.Generator().manual_seed(1), dtype=torch.int64).to(torch.int32)
    out = {}
    for name, m in mods.items():
        kc, vc = m.empty_state(cap)
        rows = []
        with torch.no_grad():
            for start, n in [(0, 64), (64, 6), (70, 1), (71, 1), (72, 1), (73, 1)]:
                rows.append(m.step(ids[:, start:start + n], torch.arange(start, start + n, dtype=torch.int32)[None], kc, vc).float())
        out[name] = torch.stack(rows)
    for name in ("split",):
        d = (out[name] - out["fused"]).abs().amax(-1)
        cos = F.cosine_similarity(out[name], out["fused"], dim=-1)
        print(f"[eager] {name} vs fused per step: maxAbs {[round(float(x), 5) for x in d]} cosine {[round(float(x), 6) for x in cos]}")


async def asset_logits(asset: Path, steps) -> list[np.ndarray]:
    import coreai.runtime as rt

    model = await rt.AIModel.load(asset, rt.SpecializationOptions.cpu_only())
    fn = model.load_function("main")
    shape = (2, 1, 8, 96, 128)
    kc, vc = np.zeros(shape, np.float16), np.zeros(shape, np.float16)
    rows = []
    for ids, start in steps:
        r = await fn({"input_ids": rt.NDArray(ids), "cache_positions": rt.NDArray(np.arange(start, start + ids.shape[1], dtype=np.int32)[None]),
                      "keyCache": rt.NDArray(kc), "valueCache": rt.NDArray(vc)}, {})
        kc[:, :, :, start:start + ids.shape[1]] = r["keyUpdate"].numpy()
        vc[:, :, :, start:start + ids.shape[1]] = r["valueUpdate"].numpy()
        rows.append(r["logits"].numpy()[0, -1].astype(np.float32))
    return rows


def asset_check(fused_asset: Path, variant_asset: Path) -> None:
    rng = np.random.default_rng(1)
    ids = rng.integers(0, CFG["vocab_size"], (1, 74)).astype(np.int32)
    steps = [(ids[:, s:s + n], s) for s, n in [(0, 64), (64, 6), (70, 1), (71, 1), (72, 1), (73, 1)]]
    a = asyncio.run(asset_logits(fused_asset, steps))
    b = asyncio.run(asset_logits(variant_asset, steps))
    for i, (x, y) in enumerate(zip(a, b)):
        cos = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
        print(f"[cpu_only] step {i} (q {steps[i][0].shape[1]}): cosine {cos:.6f} maxAbs {np.abs(x - y).max():.5f} "
              f"argmaxEqual {int(x.argmax()) == int(y.argmax())} bitIdentical {bool((x == y).all())}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["export", "eager", "check"])
    parser.add_argument("args", nargs="*")
    parser.add_argument("--output", type=Path)
    a = parser.parse_args()
    if a.mode == "export":
        if a.output.exists():
            raise SystemExit("output exists")
        export(a.args[0], a.output)
    elif a.mode == "eager":
        eager_check()
    else:
        asset_check(Path(a.args[0]), Path(a.args[1]))


if __name__ == "__main__":
    main()

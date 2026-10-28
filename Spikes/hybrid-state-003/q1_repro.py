#!/usr/bin/env python3
"""Standalone repro (torch + coreai-torch only, random weights, no checkpoint): Core AI's GPU path aborts at q = 1.

A stack of Mamba-2 layers in SSD chunk form (Granite 4.0-H 350M dimensions: hidden 768, 48 heads x 32, d_state 128,
conv kernel 4), each updating two states in place (conv fp16, SSM fp32), optionally with one NoPE attention layer over
a read-only KV input. One function, dynamic q. On the GPU (Kit's options), q = 4 and q = 2 run; the first q = 1 call
aborts the process:

    MPSNDArrayMatrixMultiplication.mm:212: failed assertion `K dimension is inconsistently defined...'

with 4 or more Mamba layers (3 run). cpu_only runs every q. Run the asset with q1-repro.swift (under the GPU lock).

    Forge/.venv/bin/python -I Spikes/hybrid-state-003/q1_repro.py --layers mmmam --output <new .aimodel>

The Mamba-2 SSD chunk maths re-implements, in reduced form, Transformers' GraniteMoeHybrid (Apache-2.0, copyright the
HuggingFace Team and IBM); see NOTICE.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

HIDDEN, HEADS, HEAD_DIM, D_STATE, KERNEL = 768, 48, 32, 128, 4
INNER, CONV = HEADS * HEAD_DIM, HEADS * HEAD_DIM + 2 * D_STATE
ATT_HEADS, ATT_KV, ATT_DIM = 12, 4, 64


class Mamba(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_proj = torch.nn.Linear(HIDDEN, INNER + CONV + HEADS, bias=False)
        self.conv_w = torch.nn.Parameter(torch.randn(CONV, KERNEL) * 0.1)
        self.dt_bias = torch.nn.Parameter(torch.randn(HEADS) * 0.1)
        self.a_log = torch.nn.Parameter(torch.log(torch.rand(HEADS) * 8 + 1))
        self.out_proj = torch.nn.Linear(INNER, HIDDEN, bias=False)

    def forward(self, x, positions, conv_state, ssm_state, layer: int):
        q = x.shape[1]
        zxbcdt = self.in_proj(x)[0]
        z, xbc, dt = zxbcdt[:, :INNER], zxbcdt[:, INNER:INNER + CONV], zxbcdt[:, INNER + CONV:]
        joined = torch.cat((conv_state[layer, 0], xbc.transpose(0, 1)), dim=1)            # [CONV, q + 3]
        conv_state[layer:layer + 1] = joined[:, -(KERNEL - 1):][None, None]
        conv = sum(joined[:, j:j + q] * self.conv_w[:, j:j + 1] for j in range(KERNEL)).float()
        xbc = F.silu(conv).transpose(0, 1)
        xs = xbc[:, :INNER].reshape(q, HEADS, HEAD_DIM)
        B, C = xbc[:, INNER:INNER + D_STATE], xbc[:, INNER + D_STATE:]
        dt = F.softplus(dt.float() + self.dt_bias.float())
        acum = torch.cumsum(dt * -torch.exp(self.a_log.float()), dim=0)                     # [q, H]
        dx = (dt[:, :, None] * xs).permute(1, 0, 2)                                         # [H, q, P]
        causal = positions[:, None] >= positions[None, :]
        decay = torch.exp(torch.where(causal[None], acum.T[:, :, None] - acum.T[:, None, :], -1e30))
        y = torch.bmm(decay * (C @ B.T)[None], dx)                                          # [H, q, P]
        h0 = ssm_state[layer, 0]
        y = y + torch.exp(acum).T[:, :, None] * torch.matmul(C[None], h0.transpose(1, 2))
        end = acum[-1:]
        h = torch.exp(end[0])[:, None, None] * h0 + torch.matmul((torch.exp(end - acum).T[:, :, None] * dx).transpose(1, 2), B[None])
        ssm_state[layer:layer + 1] = (h + (y.reshape(-1)[:1] * 0))[None, None]
        return x + self.out_proj(y.permute(1, 0, 2).reshape(1, q, INNER).to(x.dtype) * F.silu(z)[None].to(x.dtype))


class Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = torch.nn.Linear(HIDDEN, ATT_HEADS * ATT_DIM, bias=False)
        self.k = torch.nn.Linear(HIDDEN, ATT_KV * ATT_DIM, bias=False)
        self.v = torch.nn.Linear(HIDDEN, ATT_KV * ATT_DIM, bias=False)
        self.o = torch.nn.Linear(ATT_HEADS * ATT_DIM, HIDDEN, bias=False)

    def forward(self, x, positions, key_cache, value_cache):
        q_len = x.shape[1]
        q = self.q(x).view(1, q_len, ATT_HEADS, ATT_DIM).transpose(1, 2)
        k = self.k(x).view(1, q_len, ATT_KV, ATT_DIM).transpose(1, 2).contiguous()
        v = self.v(x).view(1, q_len, ATT_KV, ATT_DIM).transpose(1, 2).contiguous()
        key_pos = torch.cumsum(torch.ones_like(key_cache[0, 0, :, 0], dtype=torch.int32), 0, dtype=torch.int32) - 1
        visible = torch.cat(((key_pos[None, :] < positions[:1, None]).expand(q_len, -1), positions[None, :] <= positions[:, None]), 1)
        mask = torch.where(visible, 0.0, -30000.0).to(x.dtype)[None, None]
        rows, g = key_cache.shape[2] + q_len, ATT_HEADS // ATT_KV
        keys = torch.cat((key_cache, k), 2)[:, :, None].expand(1, ATT_KV, g, rows, ATT_DIM).reshape(1, ATT_HEADS, rows, ATT_DIM)
        values = torch.cat((value_cache, v), 2)[:, :, None].expand(1, ATT_KV, g, rows, ATT_DIM).reshape(1, ATT_HEADS, rows, ATT_DIM)
        out = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask, scale=1 / 64)
        return x + self.o(out.transpose(1, 2).reshape(1, q_len, -1)), k, v


class Stack(torch.nn.Module):
    def __init__(self, pattern: str) -> None:
        super().__init__()
        self.pattern = pattern
        self.layers = torch.nn.ModuleList(Mamba() if c == "m" else Attention() for c in pattern)
        self.slots = [pattern[:i].count(c) for i, c in enumerate(pattern)]

    def forward(self, x, positions, key_cache, value_cache, conv_state, ssm_state):
        pos = positions[0]
        k = v = None
        for kind, layer, slot in zip(self.pattern, self.layers, self.slots):
            if kind == "m":
                x = layer(x, pos, conv_state, ssm_state, slot)
            else:
                x, k, v = layer(x, pos, key_cache, value_cache)
        return x[:, -1:], k, v


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="mmmam", help="m = Mamba, a = attention (exactly one a)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.layers.count("a") != 1 or args.layers.count("m") < 2:
        raise SystemExit("new output path, exactly one attention layer, at least two Mamba layers")
    import coreai_torch as ct

    torch.manual_seed(0)
    module = Stack(args.layers).half().eval()
    mambas = args.layers.count("m")
    q_dim = torch.export.Dim("q", min=1, max=512)
    cap_dim = torch.export.Dim("capacity", min=2, max=16384)
    example = (torch.zeros(1, 8, HIDDEN, dtype=torch.float16), torch.arange(16, 24, dtype=torch.int32)[None],
               torch.zeros(1, ATT_KV, 56, ATT_DIM, dtype=torch.float16), torch.zeros(1, ATT_KV, 56, ATT_DIM, dtype=torch.float16),
               torch.zeros(mambas, 1, CONV, KERNEL - 1, dtype=torch.float16), torch.zeros(mambas, 1, HEADS, HEAD_DIM, D_STATE))
    dynamic = ({1: q_dim}, {1: q_dim}, {2: cap_dim}, {2: cap_dim}, None, None)
    with torch.no_grad():
        ep = torch.export.export(module, example, dynamic_shapes=dynamic, prefer_deferred_runtime_asserts_over_guards=True)
        ep = ep.run_decompositions(ct.get_decomp_table())
    for n in list(ep.graph.nodes):                      # deferred shape asserts: no Core AI lowering, checks only
        if n.op == "call_function" and n.target == torch.ops.aten._assert_scalar.default:
            ep.graph.erase_node(n)
    for n in list(ep.graph.nodes):                      # same-shape copies left by the in-place state writes
        if n.op == "call_function" and n.target == torch.ops.aten.copy.default and len(n.args) == 2:
            dst, src = n.args
            if [str(d) for d in dst.meta["val"].shape] == [str(d) for d in src.meta["val"].shape]:
                n.replace_all_uses_with(src)
                ep.graph.erase_node(n)
    ep.graph.eliminate_dead_code()
    ep.graph_module.recompile()
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    converter.add_exported_program(ep, input_names=["x", "positions", "keyCache", "valueCache"],
                                   output_names=["y", "keyUpdate", "valueUpdate"], state_names=["convState", "ssmState"],
                                   entrypoint_name="main")
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(args.output)
    print(f"saved {args.output}: {args.layers}, states convState [{mambas},1,{CONV},{KERNEL - 1}] fp16, ssmState [{mambas},1,{HEADS},{HEAD_DIM},{D_STATE}] fp32")


if __name__ == "__main__":
    main()

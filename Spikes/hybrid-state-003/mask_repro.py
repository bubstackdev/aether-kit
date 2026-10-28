#!/usr/bin/env python3
"""Minimal repro: Core AI cpu_only SDPA mask handling in a graph that also mutates a state.

One function: q/k/v from an input by fixed random projections, keys and values = [cache input ‖ chunk], SDPA with a
mask built from position data, plus one in-place state update (`state[0:1] = state[0:1] * 0.5 + x.sum()`), which is
what triggers the defects. Each mask form is exported with and without the state and run cpu_only against eager
float32. No checkpoint; seconds of CPU.

    Forge/.venv/bin/python -I Spikes/hybrid-state-003/mask_repro.py <scratch dir> [granite] [--mha] [--gpu] [--save DIR]

Prints, per form: OK (cosine >= 0.999), WRONG (runs, cosine below), or FAIL (the call raises).

`granite` runs the inferenceFailed repro instead, which needs a larger graph: the four-layer random-weight Granite
hybrid from export_granite_hybrid.py (three Mamba layers that mutate both states, one attention layer), one call of
8 tokens, with the attention's mask form swapped. The control passes the same states as plain inputs (no mutation).
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HEADS, KV_HEADS, DIM, HIDDEN, Q, CAP = 12, 4, 64, 768, 8, 64
if "--mha" in sys.argv:                    # plain multi-head attention: the 2-D float mask is wrong with or without enable_gqa
    sys.argv.remove("--mha")
    KV_HEADS = HEADS
GPU = "--gpu" in sys.argv                  # Python runtime on the GPU (aborts in MPSGraph on these dynamic graphs)
if GPU:
    sys.argv.remove("--gpu")
SAVE = None                                # --save DIR: keep each asset with its inputs and eager output for
if "--save" in sys.argv:                   # mask-repro-gpu.swift, which runs them the way Kit does
    i = sys.argv.index("--save")
    SAVE = Path(sys.argv[i + 1])
    del sys.argv[i:i + 2]


def options():
    import coreai.runtime as rt

    if GPU:
        return rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    return rt.SpecializationOptions.cpu_only()


class Attention(torch.nn.Module):
    def __init__(self, form: str, state: bool) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.wq = torch.nn.Parameter(torch.randn(HEADS * DIM, HIDDEN, generator=g) * 0.05)
        self.wk = torch.nn.Parameter(torch.randn(KV_HEADS * DIM, HIDDEN, generator=g) * 0.05)
        self.wv = torch.nn.Parameter(torch.randn(KV_HEADS * DIM, HIDDEN, generator=g) * 0.05)
        self.form, self.state = form, state

    def forward(self, x, positions, key_cache, value_cache, state):
        q_len = x.shape[1]
        q = F.linear(x, self.wq).view(1, q_len, HEADS, DIM).transpose(1, 2)
        k = F.linear(x, self.wk).view(1, q_len, KV_HEADS, DIM).transpose(1, 2)
        v = F.linear(x, self.wv).view(1, q_len, KV_HEADS, DIM).transpose(1, 2)
        keys, values = torch.cat((key_cache, k), dim=2), torch.cat((value_cache, v), dim=2)
        pos = positions[0]
        key_pos = torch.cumsum(torch.ones_like(key_cache[0, 0, :, 0], dtype=torch.int32), 0, dtype=torch.int32) - 1
        visible = torch.cat(((key_pos[None, :] < pos[:1, None]).expand(q_len, -1), pos[None, :] <= pos[:, None]), dim=1)
        boolean, four_d, gqa = self.form.startswith("bool"), "4d" in self.form, self.form.endswith("gqa")
        mask = visible if boolean else torch.where(visible, 0.0, -30000.0).to(x.dtype)
        if four_d:
            mask = mask[None, None]
        if not gqa:
            rows = keys.shape[2]
            keys = keys[:, :, None].expand(1, KV_HEADS, HEADS // KV_HEADS, rows, DIM).reshape(1, HEADS, rows, DIM)
            values = values[:, :, None].expand(1, KV_HEADS, HEADS // KV_HEADS, rows, DIM).reshape(1, HEADS, rows, DIM)
        out = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask, scale=1 / 64, enable_gqa=gqa)
        if self.state:
            state[0:1] = state[0:1] * 0.5 + x.float().sum()
        return out.transpose(1, 2).reshape(1, q_len, HEADS * DIM)


def run(form: str, state: bool, scratch: Path) -> str:
    import coreai.runtime as rt
    import coreai_torch as ct

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Forge" / "language"))
    from export_qwen3 import drop_identity_copies

    module = Attention(form, state).half().eval()
    q_dim = torch.export.Dim("q", min=1, max=64)
    cap_dim = torch.export.Dim("capacity", min=2, max=1024)
    ex = (torch.zeros(1, Q, HIDDEN, dtype=torch.float16), torch.arange(Q, dtype=torch.int32)[None] + 16,
          torch.zeros(1, KV_HEADS, 56, DIM, dtype=torch.float16), torch.zeros(1, KV_HEADS, 56, DIM, dtype=torch.float16),
          torch.zeros(2, 4, dtype=torch.float32))
    dyn = ({1: q_dim}, {1: q_dim}, {2: cap_dim}, {2: cap_dim}, None)
    with torch.no_grad():
        ep = torch.export.export(module, ex, dynamic_shapes=dyn).run_decompositions(ct.get_decomp_table())
    drop_identity_copies(ep)
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    inputs = ["x", "positions", "keyCache", "valueCache"] + ([] if state else ["state"])
    converter.add_exported_program(ep, input_names=inputs, output_names=["y"], state_names=["state"] if state else [],
                                   entrypoint_name="main")
    program = converter.to_coreai()
    program.optimize()
    path = Path(tempfile.mkdtemp(dir=scratch)) / "m.aimodel"
    program.save_asset(path)

    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, Q, HIDDEN)).astype(np.float16)
    positions = np.arange(Q, dtype=np.int32)[None]                               # start 0: every cache row hidden
    kc = rng.standard_normal((1, KV_HEADS, CAP, DIM)).astype(np.float16)          # garbage rows that must stay hidden
    vc = rng.standard_normal((1, KV_HEADS, CAP, DIM)).astype(np.float16)
    with torch.no_grad():
        ref = module.float()(torch.from_numpy(x).float(), torch.from_numpy(positions), torch.from_numpy(kc).float(),
                             torch.from_numpy(vc).float(), torch.zeros(2, 4)).numpy().ravel()

    async def call():
        model = await rt.AIModel.load(path, options())
        fn = model.load_function("main")
        feed = {"x": rt.NDArray(x), "positions": rt.NDArray(positions), "keyCache": rt.NDArray(kc), "valueCache": rt.NDArray(vc)}
        st = {"state": rt.NDArray(np.zeros((2, 4), np.float32))}
        r = await (fn(feed, st) if state else fn({**feed, **st}, {}))
        return r["y"].numpy().astype(np.float64).ravel()

    if SAVE is not None:
        out = SAVE / f"{'mha' if KV_HEADS == HEADS else 'gqa'}-{form}-{'state' if state else 'nostate'}"
        out.mkdir(parents=True)
        shutil.copytree(path, out / "m.aimodel")
        for name, array in (("x", x), ("positions", positions), ("keyCache", kc), ("valueCache", vc)):
            array.tofile(out / f"{name}.bin")
        ref.astype("<f4").tofile(out / "reference.bin")
        (out / "meta.txt").write_text(f"{Q} {HIDDEN} {KV_HEADS} {CAP} {DIM} {int(state)}\n")
    try:
        got = asyncio.run(call())
        cos = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref)))
        verdict = f"{'OK' if cos >= 0.999 else 'WRONG'} (cosine {cos:.4f})"
    except Exception as e:  # noqa: BLE001
        verdict = f"FAIL ({e})"
    finally:
        shutil.rmtree(path.parent)
    return verdict


def granite(form: str, mutate: bool, scratch: Path) -> str:
    import coreai.runtime as rt
    import coreai_torch as ct

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Forge" / "language"))
    import granite_hybrid as gh
    from export_granite_hybrid import TINY
    from export_qwen3 import drop_identity_copies

    def attention(self, x, key_cache, value_cache, layer, visible):
        q_len = x.shape[1]
        q = self.q_proj(x).view(1, q_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(1, q_len, self.kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_proj(x).view(1, q_len, self.kv_heads, self.head_dim).transpose(1, 2).contiguous()
        keys, values = torch.cat((key_cache[layer], k), dim=2), torch.cat((value_cache[layer], v), dim=2)
        mask = visible[0, 0] == 0 if form.startswith("bool") else visible[0, 0]           # visible is 4-D additive
        if "4d" in form:
            mask = mask[None, None]
        gqa = form.endswith("gqa")
        if not gqa:
            g, rows = self.heads // self.kv_heads, keys.shape[2]
            keys = keys[:, :, None].expand(1, self.kv_heads, g, rows, self.head_dim).reshape(1, self.heads, rows, self.head_dim)
            values = values[:, :, None].expand(1, self.kv_heads, g, rows, self.head_dim).reshape(1, self.heads, rows, self.head_dim)
        out = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask, scale=self.scale, enable_gqa=gqa)
        return self.o_proj(out.transpose(1, 2).reshape(1, q_len, -1)), k, v

    def still(self, x, positions, conv_state, ssm_state, layer):                          # control: reads, no writes
        z, xbc, dt = self.project(x)
        return self.out_proj((z + conv_state[layer, 0, :self.inner, 0].float() + ssm_state[layer, 0, 0, 0, 0]).to(x.dtype)[None])

    original = gh.Mamba.__dict__.get("_original_chunk") or gh.Mamba.chunk
    gh.Mamba._original_chunk = original
    gh.Attention.forward = attention
    gh.Mamba.chunk = original if mutate else still
    torch.manual_seed(0)
    m = gh.GraniteHybridDecoder(dict(TINY)).half().eval()
    ex, dyn, _, _ = gh.examples(m, 512)
    with torch.no_grad():
        ep = torch.export.export(gh.Main(m), ex, dynamic_shapes=dyn).run_decompositions(ct.get_decomp_table())
    drop_identity_copies(ep)
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    states = ["convState", "ssmState"] if mutate else []
    converter.add_exported_program(ep, input_names=["input_ids", "cache_positions", "keyCache", "valueCache"] + ([] if mutate else ["convState", "ssmState"]),
                                   output_names=["logits", "keyUpdate", "valueUpdate"], state_names=states, entrypoint_name="main")
    program = converter.to_coreai()
    program.optimize()
    path = Path(tempfile.mkdtemp(dir=scratch)) / "m.aimodel"
    program.save_asset(path)
    shapes = [t.shape for t in m.empty_state(64)]

    async def call():
        model = await rt.AIModel.load(path, options())
        fn = model.load_function("main")
        feed = {"input_ids": rt.NDArray(np.arange(8, dtype=np.int32)[None] + 300),
                "cache_positions": rt.NDArray(np.arange(8, dtype=np.int32)[None]),
                "keyCache": rt.NDArray(np.zeros(shapes[0], np.float16)), "valueCache": rt.NDArray(np.zeros(shapes[1], np.float16))}
        st = {"convState": rt.NDArray(np.zeros(shapes[2], np.float16)), "ssmState": rt.NDArray(np.zeros(shapes[3], np.float32))}
        return await (fn(feed, st) if mutate else fn({**feed, **st}, {}))

    try:
        asyncio.run(call())
        verdict = "runs"
    except Exception as e:  # noqa: BLE001
        verdict = f"FAIL ({e})"
    finally:
        shutil.rmtree(path.parent)
    return verdict


def main() -> None:
    scratch = Path(sys.argv[1])
    scratch.mkdir(parents=True, exist_ok=True)
    if sys.argv[2:] == ["granite"]:
        for form in ("bool2d_gqa", "bool4d_gqa", "float2d_gqa", "float4d_gqa", "bool2d", "bool4d", "float2d", "float4d"):
            print(f"{form:12s} states as inputs: {granite(form, False, scratch):40s} states mutated: {granite(form, True, scratch)}", flush=True)
        return
    for form in ("bool2d_gqa", "bool2d", "float2d_gqa", "float2d", "bool4d_gqa", "float4d"):
        print(f"{form:12s} no state: {run(form, False, scratch):32s} with state: {run(form, True, scratch)}", flush=True)


if __name__ == "__main__":
    main()

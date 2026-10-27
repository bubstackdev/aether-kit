"""Throwaway spike: export Granite 4.0-H 350M decode-graph variants to attribute decode cost.

Uses the prototype engine's forge (read-only, via PYTHONPATH) for the model definition and weight loading, and
coreai-torch for conversion. Each variant is one `.aimodel` under OUT/<variant>.aimodel.

Variants (all fp16 weights/activations, q = 1 semantics):
  ref        forge recipe @5 equivalent: `main` (dynamic SSD step) + `decode` (direct recurrence), stacked states
  nomamba    Mamba mixers removed (attention + MLP + embedding/head floor); KV states only
  split      direct recurrence, one state tensor per Mamba layer (56 recurrent states) instead of stacked [28, …]
  nobarrier  direct recurrence without the read-before-write `after()` barrier (timing only; F-P2-20 hazard)
  fp16state  direct recurrence with the SSM state stored and updated in fp16
  staticq    direct recurrence with a static q = 1 input (total/capacity stay dynamic)

Run:  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<engine>/forge <forge venv>/bin/python export_variants.py <variant>…
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import coreai_torch as ct
from aether_forge.arch.granite_hybrid import GraniteHybridFamily, RecurrentDecode
from aether_forge.arch.qwen3 import read_config
from aether_forge.export import drop_identity_copies
from aether_forge.prim.ssm import after

CKPT = next((Path.home() / "Library/Caches/aether/checkpoints/models--ibm-granite--granite-4.0-h-350m/snapshots")
            .iterdir())
OUT = Path.home() / ".cache/huggingface/aether-kit/spikes/granite"


def recurrent_mixer(m, x, conv_s, ssm_s, slot, *, barrier=True, state_dtype=torch.float32, split=False):
    """Direct Mamba2 recurrence for one token. conv_s/ssm_s are stacked [layers, …] (indexed by slot) or, when
    `split`, this layer's own [1, …] tensors."""
    zxbcdt = m.in_proj(x)
    z = zxbcdt[..., :m.inner]
    xbc = zxbcdt[..., m.inner:m.inner + m.conv_dim]
    dt = zxbcdt[..., m.inner + m.conv_dim:]
    old_conv = conv_s if split else conv_s[slot]
    joined = torch.cat((old_conv, xbc.transpose(1, 2).to(old_conv.dtype)), dim=2)              # [1, C, k]
    q = x.shape[1]
    tail = joined[:, :, -(m.kernel - 1):]
    if split:
        conv_s[0:1] = tail          # slice write (a whole-tensor copy_ loses its state output)
    else:
        conv_s[slot:slot + 1] = tail[None]
    w = m.conv_weight.to(joined.dtype)
    conv = sum(joined[:, :, j:j + q] * w[None, :, j:j + 1] for j in range(m.kernel))          # [1, C, q]
    if m.conv_bias is not None:
        conv = conv + m.conv_bias.to(conv.dtype)[None, :, None]
    xbc = F.silu(conv).transpose(1, 2)
    cdt = torch.float32 if state_dtype == torch.float32 else torch.float16
    xs = xbc[..., :m.inner].reshape(1, q, m.heads, m.head_dim).to(cdt)
    gs = m.groups * m.d_state
    B = xbc[..., m.inner:m.inner + gs].reshape(1, q, m.groups, m.d_state).to(cdt)
    C = xbc[..., m.inner + gs:].reshape(1, q, m.groups, m.d_state).to(cdt)
    per = m.heads // m.groups
    B = B.repeat_interleave(per, dim=2)
    C = C.repeat_interleave(per, dim=2)
    dt = F.softplus(dt.float() + m.dt_bias.float())
    a = dt * (-torch.exp(m.A_log.float()))
    h0 = (ssm_s if split else ssm_s[slot]).to(cdt)
    new_state = (torch.exp(a[:, 0]).to(cdt)[..., None, None] * h0
                 + (dt[:, 0, :, None].to(cdt) * xs[:, 0])[..., None] * B[:, 0, :, None, :])
    y = torch.einsum("bhpn,bqhn->bqhp", new_state, C).float()
    y = y + m.D.float()[None, None, :, None] * xs.float()
    if barrier:
        new_state = after(new_state, y.to(cdt))
    if split:
        ssm_s[0:1] = new_state.to(ssm_s.dtype)
    else:
        ssm_s[slot:slot + 1] = new_state.to(ssm_s.dtype)[None]
    return m._output(y, z, x.dtype)


def lean_mixer(m, x, conv_s, ssm_s, slot):
    """One-token Mamba2 recurrence written for a small op count (groups = 1, kernel-4 conv as one multiply-reduce,
    no einsum/repeat_interleave, precomputed −exp(A_log), fp32 from the in_proj cast on). Same math as the forge
    companion; the F-P2-20 barrier stays."""
    zxbcdt = m.in_proj(x)[0, 0].float()                                     # [3376]
    z = zxbcdt[:m.inner]
    xbc = zxbcdt[m.inner:m.inner + m.conv_dim]
    dt = zxbcdt[m.inner + m.conv_dim:]
    old = conv_s[slot, 0].float()                                          # [C, 3]
    joined = torch.cat((old, xbc[:, None]), dim=1)                         # [C, 4]
    conv_s[slot:slot + 1] = joined[None, None, :, 1:].to(conv_s.dtype)
    conv = (joined * m.conv_weight.float()).sum(-1) + m.conv_bias.float()  # [C]
    xbc = F.silu(conv)
    xs = xbc[:m.inner].view(m.heads, m.head_dim)                           # [H, P]
    B = xbc[m.inner:m.inner + m.d_state]                                   # [N]   (groups = 1)
    C = xbc[m.inner + m.d_state:]                                          # [N]
    dt = F.softplus(dt + m.dt_bias.float())                                # [H]
    decay = torch.exp(dt * m.A_neg)                                        # [H]
    h0 = ssm_s[slot, 0]                                                    # [H, P, N] fp32
    h = decay[:, None, None] * h0 + (dt[:, None] * xs)[:, :, None] * B
    y = (h * C).sum(-1) + m.D.float()[:, None] * xs                        # [H, P]
    h = after(h, y)
    ssm_s[slot:slot + 1] = h[None, None]
    g = y.reshape(1, 1, m.inner) * F.silu(z)
    g = g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + m.eps) * m.norm_weight.float()
    return m.out_proj(g.to(x.dtype))


# Fused Mamba2 one-token update as a Core AI custom Metal kernel (coreai_torch.TorchMetalKernel): conv state shift,
# depthwise conv + SiLU, softplus(dt), decay, state update and the y reduction in one dispatch per layer.
# Metal tensor subscripts run innermost-first (x, y, ...), the reverse of torch's dimension order.
# Granite 4.0-H 350M shapes: inner 1536 = 48 heads x 32, d_state 128, conv channels 1792, kernel 4.
_SSM_SRC = """
    const int INNER = 1536, NS = 128, CH = 1792, P = 32;
    int t = int(gid);
    if (t >= INNER) return;
    int h = t / P, p = t % P;
    float xin = float(zx[INNER + t]);
    float acc = conv_b[t] + float(conv_old[0, t]) * conv_w[0, t] + float(conv_old[1, t]) * conv_w[1, t]
              + float(conv_old[2, t]) * conv_w[2, t] + xin * conv_w[3, t];
    float x = acc / (1.0f + exp(-acc));
    conv_new[0, t] = conv_old[1, t]; conv_new[1, t] = conv_old[2, t]; conv_new[2, t] = half(xin);
    if (t < 2 * NS) {
        int c = INNER + t;
        conv_new[0, c] = conv_old[1, c]; conv_new[1, c] = conv_old[2, c]; conv_new[2, c] = zx[INNER + c];
    }
    float dr = float(zx[INNER + CH + h]) + dt_bias[h];
    float dt = dr > 20.0f ? dr : log(1.0f + exp(dr));
    float decay = exp(dt * a_neg[h]);
    float dx = dt * x;
    float yacc = 0.0f;
    for (int n = 0; n < NS; n++) {
        int cb = INNER + n, cc = INNER + NS + n;
        float b = conv_b[cb] + float(conv_old[0, cb]) * conv_w[0, cb] + float(conv_old[1, cb]) * conv_w[1, cb]
                + float(conv_old[2, cb]) * conv_w[2, cb] + float(zx[INNER + cb]) * conv_w[3, cb];
        float c = conv_b[cc] + float(conv_old[0, cc]) * conv_w[0, cc] + float(conv_old[1, cc]) * conv_w[1, cc]
                + float(conv_old[2, cc]) * conv_w[2, cc] + float(zx[INNER + cc]) * conv_w[3, cc];
        b = b / (1.0f + exp(-b));
        c = c / (1.0f + exp(-c));
        float hn = decay * h0[n, p, h] + dx * b;
        h_new[n, p, h] = hn;
        yacc += hn * c;
    }
    y[p, h] = yacc + d_skip[h] * x;
"""


def _ssm_ref(zx: torch.Tensor, conv_old: torch.Tensor, conv_w: torch.Tensor, conv_b: torch.Tensor,
             dt_bias: torch.Tensor, a_neg: torch.Tensor, d_skip: torch.Tensor,
             h0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inner, ns, ch = 1536, 128, 1792
    xbc_in = zx[inner:inner + ch].float()
    joined = torch.cat((conv_old.float(), xbc_in[:, None]), dim=1)
    xbc = F.silu((joined * conv_w).sum(-1) + conv_b)
    xs = xbc[:inner].view(48, 32)
    B, C = xbc[inner:inner + ns], xbc[inner + ns:]
    dt = F.softplus(zx[inner + ch:].float() + dt_bias)
    h = torch.exp(dt * a_neg)[:, None, None] * h0 + (dt[:, None] * xs)[:, :, None] * B
    y = (h * C).sum(-1) + d_skip[:, None] * xs
    conv_new = torch.cat((conv_old[:, 1:], zx[inner:inner + ch, None]), dim=1)
    return conv_new, h, y


_SSM_KERNEL = None


def ssm_kernel():
    global _SSM_KERNEL
    if _SSM_KERNEL is None:
        from coreai.authoring import MetalParameter
        _SSM_KERNEL = ct.TorchMetalKernel(
            "mamba2_step", ["zx", "conv_old", "conv_w", "conv_b", "dt_bias", "a_neg", "d_skip", "h0"],
            ["conv_new", "h_new", "y"], _SSM_SRC, _ssm_ref,
            metal_params=[MetalParameter("gid", "uint", "thread_position_in_grid")])
    return _SSM_KERNEL


def kernel_mixer(m, x, conv_s, ssm_s, slot):
    zx = m.in_proj(x)[0, 0]                                                # [3376] fp16
    conv_new, h, y = ssm_kernel()(
        zx, conv_s[slot, 0], m.conv_weight.float(), m.conv_bias.float(), m.dt_bias.float(), m.A_neg,
        m.D.float(), ssm_s[slot, 0], threads_per_grid=(1536, 1, 1), threads_per_thread_group=(256, 1, 1),
        result_shapes=[[1792, 3], [48, 32, 128], [48, 32]])
    conv_s[slot:slot + 1] = conv_new[None, None]
    ssm_s[slot:slot + 1] = h[None, None]
    g = y.reshape(1, 1, m.inner) * F.silu(zx[:m.inner].float())
    g = g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + m.eps) * m.norm_weight.float()
    return m.out_proj(g.to(x.dtype))


STATIC_CAPACITY = 1024


def static_attention(attn, x, key_cache, value_cache, slot, pos):
    """Fixed-shape one-token attention: the new row is blended in at the runtime position (one-hot over the fixed
    capacity, data-derived) and the mask is `arange(C) <= pos`. Every read of the old cache feeds the write."""
    q = attn.q_proj(x).view(1, 1, attn.heads, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(x).view(1, 1, attn.kv_heads, attn.head_dim).transpose(1, 2)
    v = attn.v_proj(x).view(1, 1, attn.kv_heads, attn.head_dim).transpose(1, 2)
    idx = torch.arange(key_cache.shape[3], dtype=torch.int32)
    hot = (idx == pos).to(k.dtype)[None, None, :, None]
    keys = key_cache[slot] * (1 - hot) + k * hot
    values = value_cache[slot] * (1 - hot) + v * hot
    key_cache[slot:slot + 1] = keys[None]
    value_cache[slot:slot + 1] = values[None]
    mask = (idx <= pos)[None, :]
    out = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask, scale=attn.scale, enable_gqa=True)
    return attn.o_proj(out.transpose(1, 2).reshape(1, 1, attn.heads * attn.head_dim))


class Decode(torch.nn.Module):
    """One-token decode over the forge module's weights with a selectable Mamba formulation."""

    def __init__(self, inner, mode: str):
        super().__init__()
        self.inner, self.mode = inner, mode

    def forward(self, input_ids, position_ids, key_cache, value_cache, *rec):
        mod, cfg = self.inner, self.inner.cfg
        static = self.mode.startswith("static-")
        q_len, total = input_ids.shape[1], position_ids.shape[1]
        offset = total - q_len
        positions = position_ids[0]
        n = mod.mamba_layers
        x = mod.embed(input_ids) * float(cfg["embedding_multiplier"])
        for block, slot in zip(mod.blocks, mod.slot):
            if block.kind == "mamba":
                if self.mode in ("nomamba", "static-nomamba"):
                    pass
                else:
                    h = block.input_norm(x)
                    if self.mode == "static-kernel":
                        h = kernel_mixer(block.mixer, h, rec[0], rec[1], slot)
                    elif self.mode == "static-lean":
                        h = lean_mixer(block.mixer, h, rec[0], rec[1], slot)
                    elif self.mode == "split":
                        h = recurrent_mixer(block.mixer, h, rec[slot], rec[n + slot], slot, split=True)
                    else:
                        h = recurrent_mixer(block.mixer, h, rec[0], rec[1], slot,
                                            barrier=self.mode != "nobarrier",
                                            state_dtype=torch.float16 if self.mode == "fp16state" else torch.float32)
                    x = x + h * block.residual
            else:
                h = block.input_norm(x)
                if static:
                    h = static_attention(block.mixer, h, key_cache, value_cache, slot, position_ids[0, 0])
                else:
                    h = block.mixer(h, key_cache, value_cache, slot, offset, total, positions)
                x = x + h * block.residual
            x = x + block.mlp(block.post_norm(x)) * block.residual
        x = mod.norm(x)
        return F.linear(x, mod.embed.weight) / float(cfg["logits_scaling"])


def build():
    fam = GraniteHybridFamily()
    cfg = read_config(CKPT)
    module = fam.build(cfg, {"tie_embeddings": True}, torch.float16)
    fam.load_weights(module, CKPT)
    for b in module.blocks:
        if b.kind == "mamba":
            assert b.mixer.groups == 1 and b.mixer.kernel == 4
            b.mixer.register_buffer("A_neg", -torch.exp(b.mixer.A_log.detach().float()))
    return fam, module


def state_examples(fam, module, mode: str, capacity: int):
    kc, vc, cs, ss = fam.empty_state(module, capacity)
    if mode in ("nomamba", "static-nomamba"):
        return [kc, vc], ["keyCache", "valueCache"]
    if mode == "split":
        n = module.mamba_layers
        convs = [torch.zeros_like(cs[0]) for _ in range(n)]
        ssms = [torch.zeros_like(ss[0]) for _ in range(n)]
        return [kc, vc, *convs, *ssms], ["keyCache", "valueCache"] + [f"conv{i}" for i in range(n)] + \
            [f"ssm{i}" for i in range(n)]
    if mode == "fp16state":
        ss = ss.half()
    return [kc, vc, cs, ss], ["keyCache", "valueCache", "convState", "ssmState"]


def export_one(m, example, dyn, drop_copies=True):
    with torch.no_grad():
        ep = torch.export.export(m, example, dynamic_shapes=dyn)
        ep = ep.run_decompositions(ct.get_decomp_table())
        if drop_copies:
            # Whole-tensor state writes (split) become copy nodes the signature names; keep those.
            drop_identity_copies(ep)
    return ep


def main(variants):
    OUT.mkdir(parents=True, exist_ok=True)
    fam, module = build()
    max_len = 4096
    q_dim = torch.export.Dim("q", min=1, max=max_len)
    total_dim = torch.export.Dim("total", min=2, max=max_len)
    cap_dim = torch.export.Dim("capacity", min=2, max=max_len)
    rp = OUT / "export-report.json"
    report = json.loads(rp.read_text()) if rp.exists() else {}
    for v in variants:
        t0 = time.time()
        conv = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
        if v == "ref":
            spec = fam.export_spec(module, {"trace": {"seq": {"min": 1, "max": max_len, "example_q": 8,
                                                              "example_total": 24}}})
            ep = export_one(module, spec.example_inputs, spec.dynamic_shapes)
            dep = export_one(RecurrentDecode(module), spec.example_inputs, spec.dynamic_shapes)
            conv.add_exported_program(ep, input_names=spec.input_names, output_names=spec.output_names,
                                      state_names=spec.state_names)
            conv.add_exported_program(dep, input_names=spec.input_names, output_names=spec.output_names,
                                      state_names=spec.state_names, entrypoint_name="decode")
        elif v.startswith("static-"):
            # Fully fixed shapes: one token, the current position as a [1, 1] input, capacity fixed.
            states, names = state_examples(fam, module, v, STATIC_CAPACITY)
            ids = torch.tensor([[100]], dtype=torch.int32)
            pos = torch.tensor([[23]], dtype=torch.int32)
            ep = export_one(Decode(module, v), (ids, pos, *states), None)
            if len(ep.graph_signature.user_inputs_to_mutate) != len(names):
                raise SystemExit(f"{v}: mutated inputs do not match states")
            if v == "static-kernel":
                conv.register_custom_kernels([ssm_kernel()])
            conv.add_exported_program(ep, input_names=["input_ids", "position_ids"], output_names=["logits"],
                                      state_names=names)
        else:
            states, names = state_examples(fam, module, v, 64)
            static = v == "staticq"
            # Example uses q = 1 (decode); dynamic metadata keeps q symbolic except for `staticq`.
            ids = torch.tensor([[100]], dtype=torch.int32)
            pos = torch.arange(24, dtype=torch.int32)[None]
            if static:
                dyn = (None, {1: total_dim}, {3: cap_dim}, {3: cap_dim}) + ((None,) * (len(states) - 2),)
            else:
                # torch.export specializes a size-1 example dim: trace with q = 2 so q stays symbolic, like the forge
                # decode (the recurrence reads token 0; the graph only ever runs with q = 1).
                ids = torch.tensor([[100, 101]], dtype=torch.int32)
                dyn = ({1: q_dim}, {1: total_dim}, {3: cap_dim}, {3: cap_dim}) + ((None,) * (len(states) - 2),)
            dec = Decode(module, v)
            ep = export_one(dec, (ids, pos, *states), dyn, drop_copies=v != "split")
            mutated = len(ep.graph_signature.user_inputs_to_mutate)
            if mutated != len(names):
                raise SystemExit(f"{v}: {mutated} mutated inputs for {len(names)} states")
            conv.add_exported_program(ep, input_names=["input_ids", "position_ids"], output_names=["logits"],
                                      state_names=names)
        prog = conv.to_coreai()
        prog.optimize()
        path = OUT / f"{v}.aimodel"
        if path.exists():
            import shutil
            shutil.rmtree(path)
        prog.save_asset(path)
        report[v] = {"seconds": round(time.time() - t0, 1), "path": str(path)}
        print(v, report[v], flush=True)
        rp.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:] or ["ref", "nomamba", "split", "nobarrier", "fp16state", "staticq"])

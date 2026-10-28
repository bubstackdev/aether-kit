"""Granite 4.0-H dense hybrid decoder (Mamba-2 + NoPE attention) for Aether Kit language exports.

Development tooling only; never required by SDK consumers. Written for AetherKit from the checkpoint's config and the
GraniteMoeHybrid architecture (upstream reference: Transformers `GraniteMoeHybridForCausalLM`, used only by
`reference_granite_hybrid.py`). Dense checkpoints only (`num_local_experts == 0`).

Step contract (`hybrid`, 0.0.3 spike): the 0.0.2 bucketed KV contract for the attention layers plus in-place
recurrent state for the Mamba layers.

    main(input_ids [1, q] i32, cache_positions [1, q] i32 = start..<start+q,
         keyCache, valueCache [attn layers, 1, kv_heads, capacity, head_dim] fp16 (read-only inputs);
         states convState [mamba layers, 1, conv_dim, kernel - 1] fp16, ssmState [mamba layers, 1, heads, head_dim, d_state] fp32)
      -> logits [1, 1, vocab], keyUpdate, valueUpdate [attn layers, 1, kv_heads, q, head_dim]

- Attention is the bucketed contract unchanged: one SDPA over [cache ‖ chunk] with a mask from position data,
  the chunk's keys and values returned for the host to write at rows start..<start+q.
- Each Mamba layer treats the submission as one SSD chunk: the quadratic intra-chunk term, the carried state's term,
  and the state at the chunk's end, written back in place. Chunked prefill is exact across submissions. Recurrent
  state cannot step back, so a host must never resubmit rows it already sent (no bucketed step-back tail): the
  prompt's remainder goes in smaller widths instead.
- The SSM state is fp32 (fp16 measured 1.1-1.4% state error in the prototype); the conv state holds raw in_proj
  outputs and stays fp16.
- `decode` (optional second function, same names) runs q = 1 with the one-token recurrence, which needs far fewer
  ops than the SSD form. Core AI wires a copy of the weights per function that runs, so it costs one more copy.

Every read of a state's old contents must feed that state's write (F-P2-20): the SSM state's old value also feeds the
output term, so the write is ordered after the output with `after`. `state_order_violations` checks an exported graph.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

NEG = -1e30


def after(t: torch.Tensor, dep: torch.Tensor) -> torch.Tensor:
    """`t`, with a data dependency on `dep` so a state write of `t` is ordered after every read feeding `dep`."""
    return t + (dep.reshape(-1)[:1] * 0).to(t.dtype)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.to(torch.float32)
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight.to(x.dtype)


class Mamba(torch.nn.Module):
    """Mamba-2 mixer (one group). Weights: in_proj -> [z | x B C | dt], depthwise conv over x B C, gated RMSNorm."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden = cfg["hidden_size"]
        self.heads, self.head_dim, self.d_state = cfg["mamba_n_heads"], cfg["mamba_d_head"], cfg["mamba_d_state"]
        self.inner = int(cfg["mamba_expand"] * hidden)
        if cfg.get("mamba_n_groups", 1) != 1 or self.heads * self.head_dim != self.inner:
            raise NotImplementedError("one Mamba group and heads * head_dim == expand * hidden only")
        self.kernel = cfg["mamba_d_conv"]
        self.conv_dim = self.inner + 2 * self.d_state
        self.in_proj = torch.nn.Linear(hidden, self.inner + self.conv_dim + self.heads, bias=bool(cfg.get("mamba_proj_bias")))
        self.conv_weight = torch.nn.Parameter(torch.zeros(self.conv_dim, self.kernel))
        self.conv_bias = torch.nn.Parameter(torch.zeros(self.conv_dim)) if cfg.get("mamba_conv_bias", True) else None
        self.dt_bias = torch.nn.Parameter(torch.zeros(self.heads))
        self.A_log = torch.nn.Parameter(torch.zeros(self.heads))
        self.D = torch.nn.Parameter(torch.ones(self.heads))
        self.norm_weight = torch.nn.Parameter(torch.ones(self.inner))
        self.out_proj = torch.nn.Linear(self.inner, hidden, bias=bool(cfg.get("mamba_proj_bias")))
        self.eps = cfg["rms_norm_eps"]

    def project(self, x):
        zxbcdt = self.in_proj(x)[0]                                                    # [q, inner + conv_dim + heads]
        return (zxbcdt[:, :self.inner], zxbcdt[:, self.inner:self.inner + self.conv_dim],
                zxbcdt[:, self.inner + self.conv_dim:])

    def output(self, y, z, dtype):
        """Gated RMSNorm over the whole inner width (one group), then out_proj. y [q, inner] fp32."""
        g = y * F.silu(z.float())
        g = g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + self.eps) * self.norm_weight.float()
        return self.out_proj(g.to(dtype)[None])

    def chunk(self, x, positions, conv_state, ssm_state, layer: int):
        """The submission as one SSD chunk. x [1, q, hidden]; positions [q] (masks from position data, F-P2-6)."""
        q = x.shape[1]
        z, xbc, dt = self.project(x)
        old = conv_state[layer, 0]                                                     # [conv_dim, kernel - 1]
        joined = torch.cat((old, xbc.transpose(0, 1).to(old.dtype)), dim=1)            # [conv_dim, q + kernel - 1]
        conv_state[layer:layer + 1] = joined[:, -(self.kernel - 1):][None, None]
        w = self.conv_weight.to(joined.dtype)
        conv = sum(joined[:, j:j + q] * w[:, j:j + 1] for j in range(self.kernel)).float()
        if self.conv_bias is not None:
            conv = conv + self.conv_bias.float()[:, None]
        xbc = F.silu(conv).transpose(0, 1)                                             # [q, conv_dim] fp32
        xs = xbc[:, :self.inner].reshape(q, self.heads, self.head_dim)                 # [q, H, P]
        B = xbc[:, self.inner:self.inner + self.d_state]                               # [q, N]
        C = xbc[:, self.inner + self.d_state:]                                         # [q, N]
        dt = F.softplus(dt.float() + self.dt_bias.float())                             # [q, H]
        acum = torch.cumsum(dt * -torch.exp(self.A_log.float()), dim=0)                # [q, H], decreasing
        dx = (dt[:, :, None] * xs).permute(1, 0, 2)                                    # [H, q, P]
        # Intra-chunk: y_t += sum_{s<=t} exp(acum_t - acum_s) (C_t . B_s) dt_s x_s.
        causal = positions[:, None] >= positions[None, :]                              # [t, s]
        decay = torch.exp(torch.where(causal[None], acum.transpose(0, 1)[:, :, None] - acum.transpose(0, 1)[:, None, :], NEG))
        y = torch.bmm(decay * (C @ B.transpose(0, 1))[None], dx)                       # [H, q, P]
        # Carried state: y_t += exp(acum_t) C_t . h0.
        h0 = ssm_state[layer, 0]                                                       # [H, P, N] fp32
        y = y + torch.exp(acum).transpose(0, 1)[:, :, None] * torch.matmul(C[None], h0.transpose(1, 2))
        y = y + self.D.float()[:, None, None] * xs.permute(1, 0, 2)
        # State at the chunk's end: exp(acum_end) h0 + sum_s exp(acum_end - acum_s) dt_s x_s B_s^T.
        end = acum[-1:]                                                                # [1, H]
        carry = torch.exp(end - acum).transpose(0, 1)[:, :, None] * dx                 # [H, q, P]
        h = torch.exp(end[0])[:, None, None] * h0 + torch.matmul(carry.transpose(1, 2), B[None])
        ssm_state[layer:layer + 1] = after(h, y)[None, None]
        return self.output(y.permute(1, 0, 2).reshape(q, self.inner), z, x.dtype)

    def recurrent(self, x, conv_state, ssm_state, layer: int):
        """One token (q = 1) with the direct recurrence: the same maths as `chunk` at q = 1, in fewer ops."""
        z, xbc, dt = self.project(x)
        old = conv_state[layer, 0]                                                     # [conv_dim, kernel - 1]
        joined = torch.cat((old, xbc.transpose(0, 1).to(old.dtype)), dim=1)            # [conv_dim, kernel]
        conv_state[layer:layer + 1] = joined[:, 1:][None, None]
        conv = (joined.float() * self.conv_weight.float()).sum(-1)
        if self.conv_bias is not None:
            conv = conv + self.conv_bias.float()
        xbc = F.silu(conv)                                                             # [conv_dim]
        xs = xbc[:self.inner].view(self.heads, self.head_dim)                          # [H, P]
        B = xbc[self.inner:self.inner + self.d_state]
        C = xbc[self.inner + self.d_state:]
        dt = F.softplus(dt[0].float() + self.dt_bias.float())                          # [H]
        h0 = ssm_state[layer, 0]
        h = torch.exp(dt * -torch.exp(self.A_log.float()))[:, None, None] * h0 + (dt[:, None] * xs)[:, :, None] * B
        y = (h * C).sum(-1) + self.D.float()[:, None] * xs                             # [H, P]
        ssm_state[layer:layer + 1] = after(h, y)[None, None]
        return self.output(y.reshape(1, self.inner), z, x.dtype)


class Attention(torch.nn.Module):
    """NoPE grouped-query attention, scaled by `attention_multiplier`, over the bucketed [cache ‖ chunk]."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden = cfg["hidden_size"]
        self.heads, self.kv_heads = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.head_dim = hidden // self.heads
        self.scale = float(cfg["attention_multiplier"])
        bias = bool(cfg.get("attention_bias"))
        self.q_proj = torch.nn.Linear(hidden, self.heads * self.head_dim, bias=bias)
        self.k_proj = torch.nn.Linear(hidden, self.kv_heads * self.head_dim, bias=bias)
        self.v_proj = torch.nn.Linear(hidden, self.kv_heads * self.head_dim, bias=bias)
        self.o_proj = torch.nn.Linear(hidden, hidden, bias=bias)

    def forward(self, x, key_cache, value_cache, layer: int, visible):
        q_len = x.shape[1]
        q = self.q_proj(x).view(1, q_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(1, q_len, self.kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_proj(x).view(1, q_len, self.kv_heads, self.head_dim).transpose(1, 2).contiguous()
        # Core AI workarounds, not fixes (coreai-core 1.0.0b2, 2026-10-27; repro: Spikes/hybrid-state-003/mask_repro.py):
        # - cpu_only fails (inferenceFailed) on this graph, whose Mamba layers mutate states, for SDPA with
        #   `enable_gqa` or with a boolean mask; the same graph with the states as plain inputs runs.
        # - cpu_only silently attends wrongly with a 2-D float mask over explicitly repeated KV heads, states or not
        #   (logits cosine 0.986 on the 350M, 0.38 on the minimal repro).
        # So: a 4-D float additive mask over KV heads repeated here, which matches eager.
        groups, rows = self.heads // self.kv_heads, key_cache.shape[3] + q_len
        keys = torch.cat((key_cache[layer], k), dim=2)[:, :, None].expand(1, self.kv_heads, groups, rows, self.head_dim)
        values = torch.cat((value_cache[layer], v), dim=2)[:, :, None].expand(1, self.kv_heads, groups, rows, self.head_dim)
        keys, values = keys.reshape(1, self.heads, rows, self.head_dim), values.reshape(1, self.heads, rows, self.head_dim)
        attn = F.scaled_dot_product_attention(q, keys, values, attn_mask=visible, scale=self.scale)
        return self.o_proj(attn.transpose(1, 2).reshape(1, q_len, self.heads * self.head_dim)), k, v


class Layer(torch.nn.Module):
    def __init__(self, cfg: dict[str, Any], kind: str) -> None:
        super().__init__()
        hidden, eps = cfg["hidden_size"], cfg["rms_norm_eps"]
        self.kind = kind
        self.input_norm = RMSNorm(hidden, eps)
        self.post_norm = RMSNorm(hidden, eps)
        self.mixer = Mamba(cfg) if kind == "mamba" else Attention(cfg)
        inter = cfg["shared_intermediate_size"]
        self.mlp_in = torch.nn.Linear(hidden, 2 * inter, bias=False)
        self.mlp_out = torch.nn.Linear(inter, hidden, bias=False)
        self.inter = inter

    def mlp(self, x):
        h = self.mlp_in(x)
        return self.mlp_out(F.silu(h[..., :self.inter]) * h[..., self.inter:])


class GraniteHybridDecoder(torch.nn.Module):
    def __init__(self, cfg: dict[str, Any], max_capacity: int = 16384) -> None:
        super().__init__()
        if cfg.get("num_local_experts", 0) or cfg.get("position_embedding_type", "nope") != "nope":
            raise NotImplementedError("dense NoPE Granite hybrids only")
        self.cfg, self.max_capacity = cfg, max_capacity
        self.kinds = list(cfg["layer_types"])
        self.embed = torch.nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = torch.nn.ModuleList(Layer(cfg, kind) for kind in self.kinds)
        self.norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        if not cfg.get("tie_word_embeddings", False):
            raise NotImplementedError("tied embeddings only")
        # Each layer's row in the stacked state of its kind.
        self.slot = [self.kinds[:i].count(kind) for i, kind in enumerate(self.kinds)]
        self.mamba_layers, self.attention_layers = self.kinds.count("mamba"), self.kinds.count("attention")
        self.embedding_multiplier = float(cfg["embedding_multiplier"])
        self.residual_multiplier = float(cfg["residual_multiplier"])
        self.logits_scaling = float(cfg["logits_scaling"])

    def forward(self, input_ids, cache_positions, key_cache, value_cache, conv_state, ssm_state, recurrent: bool = False):
        q_len = input_ids.shape[1]
        x = self.embed(input_ids) * self.embedding_multiplier
        positions = cache_positions[0]
        key_positions = torch.cumsum(torch.ones_like(key_cache[0, 0, 0, :, 0], dtype=torch.int32), 0, dtype=torch.int32) - 1
        cache_visible = key_positions[None, :] < positions[:1, None]
        chunk_visible = positions[None, :] <= positions[:, None]
        visible = torch.cat((cache_visible.expand(q_len, -1), chunk_visible), dim=1)  # [q, capacity + q]
        visible = torch.where(visible, 0.0, -30000.0).to(x.dtype)[None, None]         # [1, 1, q, capacity + q], additive (see Attention)
        keys, values = [], []
        for layer, slot in zip(self.layers, self.slot):
            h = layer.input_norm(x)
            if layer.kind == "mamba":
                h = (layer.mixer.recurrent(h, conv_state, ssm_state, slot) if recurrent
                     else layer.mixer.chunk(h, positions, conv_state, ssm_state, slot))
            else:
                h, k, v = layer.mixer(h, key_cache, value_cache, slot, visible)
                keys.append(k)
                values.append(v)
            x = x + h * self.residual_multiplier
            x = x + layer.mlp(layer.post_norm(x)) * self.residual_multiplier
        logits = F.linear(self.norm(x[:, -1:]), self.embed.weight) / self.logits_scaling
        return logits, torch.stack(keys), torch.stack(values)

    def empty_state(self, capacity: int):
        """(keyCache, valueCache, convState, ssmState) zeros: KV fp16-or-module dtype, SSM state always fp32."""
        c, dtype = self.cfg, self.embed.weight.dtype
        attn = self.layers[self.kinds.index("attention")].mixer
        mamba = self.layers[self.kinds.index("mamba")].mixer
        kv = (self.attention_layers, 1, attn.kv_heads, capacity, attn.head_dim)
        return (torch.zeros(kv, dtype=dtype), torch.zeros(kv, dtype=dtype),
                torch.zeros(self.mamba_layers, 1, mamba.conv_dim, mamba.kernel - 1, dtype=dtype),
                torch.zeros(self.mamba_layers, 1, mamba.heads, mamba.head_dim, mamba.d_state, dtype=torch.float32))

    def step(self, input_ids, cache_positions, key_cache, value_cache, conv_state, ssm_state, recurrent: bool = False):
        """One host-loop step on torch tensors: last-row logits [vocab]; the returned KV rows are written like the
        host does, the recurrent states are updated by the call itself."""
        logits, ku, vu = self(input_ids, cache_positions, key_cache, value_cache, conv_state, ssm_state, recurrent)
        start, q = int(cache_positions[0, 0]), input_ids.shape[1]
        key_cache[:, :, :, start:start + q] = ku
        value_cache[:, :, :, start:start + q] = vu
        return logits[0, -1]


class Main(torch.nn.Module):
    """`main`: the SSD chunk form for any q."""

    def __init__(self, inner: GraniteHybridDecoder) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, input_ids, cache_positions, key_cache, value_cache, conv_state, ssm_state):
        return self.inner(input_ids, cache_positions, key_cache, value_cache, conv_state, ssm_state)


class Decode(torch.nn.Module):
    """`decode`: q = 1 with the direct recurrence."""

    def __init__(self, inner: GraniteHybridDecoder) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, input_ids, cache_positions, key_cache, value_cache, conv_state, ssm_state):
        return self.inner(input_ids, cache_positions, key_cache, value_cache, conv_state, ssm_state, recurrent=True)


def examples(module: GraniteHybridDecoder, max_len: int):
    """(main example, main dynamic shapes, decode example, decode dynamic shapes) for torch.export."""
    q_dim = torch.export.Dim("q", min=1, max=max_len)
    cap_dim = torch.export.Dim("capacity", min=2, max=module.max_capacity)
    kc, vc, cs, ss = module.empty_state(56)
    q, start = 8, 16
    main = (torch.arange(q, dtype=torch.int32)[None] + 100, torch.arange(start, start + q, dtype=torch.int32)[None],
            kc, vc, cs, ss)
    main_dyn = ({1: q_dim}, {1: q_dim}, {3: cap_dim}, {3: cap_dim}, None, None)
    decode = (torch.tensor([[100]], dtype=torch.int32), torch.tensor([[start]], dtype=torch.int32), kc, vc, cs, ss)
    decode_dyn = (None, None, {3: cap_dim}, {3: cap_dim}, None, None)
    return main, main_dyn, decode, decode_dyn


def state_order_violations(ep) -> list[str]:
    """F-P2-20 lint for an exported program: every node reading a mutated state input must be an ancestor of the
    node that produces that state's new value. Returns one line per violation."""
    sig = ep.graph_signature
    mutated = sig.user_inputs_to_mutate                                            # output name -> input name
    nodes = {n.name: n for n in ep.graph.nodes}
    out = [n for n in ep.graph.nodes if n.op == "output"][0]
    produced = {arg.name: arg for arg in out.args[0] if isinstance(arg, torch.fx.Node)}
    problems = []
    for output_name, input_name in mutated.items():
        write = produced.get(output_name)
        if write is None:
            continue
        ancestors, stack = set(), [write]
        while stack:
            n = stack.pop()
            if n.name in ancestors:
                continue
            ancestors.add(n.name)
            stack.extend(a for a in n.all_input_nodes)
        for user in nodes[input_name].users:
            # A read is any non-write use; it must feed the write, directly or through later reads. Dead nodes
            # (left by slice assignment) read nothing at run time.
            if user.name not in ancestors and user.users:
                problems.append(f"{input_name}: {user.name} ({user.target}) does not feed the write {write.name}")
    return problems


def name_map(module: GraniteHybridDecoder) -> dict[str, str]:
    m = {"model.embed_tokens.weight": "embed.weight", "model.norm.weight": "norm.weight"}
    for i, layer in enumerate(module.layers):
        src, dst = f"model.layers.{i}.", f"layers.{i}."
        m |= {f"{src}input_layernorm.weight": f"{dst}input_norm.weight",
              f"{src}post_attention_layernorm.weight": f"{dst}post_norm.weight",
              f"{src}shared_mlp.input_linear.weight": f"{dst}mlp_in.weight",
              f"{src}shared_mlp.output_linear.weight": f"{dst}mlp_out.weight"}
        if layer.kind == "mamba":
            for a, b in [("A_log", "A_log"), ("D", "D"), ("conv1d.bias", "conv_bias"), ("conv1d.weight", "conv_weight"),
                         ("dt_bias", "dt_bias"), ("in_proj.weight", "in_proj.weight"), ("norm.weight", "norm_weight"),
                         ("out_proj.weight", "out_proj.weight"), ("in_proj.bias", "in_proj.bias"),
                         ("out_proj.bias", "out_proj.bias")]:
                m[f"{src}mamba.{a}"] = f"{dst}mixer.{b}"
        else:
            for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
                m[f"{src}self_attn.{p}.weight"] = f"{dst}mixer.{p}.weight"
                m[f"{src}self_attn.{p}.bias"] = f"{dst}mixer.{p}.bias"
    return m


def load_state_dict(module: GraniteHybridDecoder, tensors: dict[str, torch.Tensor]) -> None:
    """Copy checkpoint-named tensors into the module; every parameter must be filled exactly once."""
    params = dict(module.named_parameters())
    names = name_map(module)
    filled = set()
    with torch.no_grad():
        for key, t in tensors.items():
            if key == "lm_head.weight":
                continue                                                           # tied to the embedding
            if key not in names or names[key] not in params:
                raise KeyError(f"unexpected checkpoint tensor {key}")
            target = params[names[key]]
            t = t.reshape(target.shape) if key.endswith("conv1d.weight") else t    # [C, 1, k] -> [C, k]
            if tuple(t.shape) != tuple(target.shape):
                raise ValueError(f"{key}: {tuple(t.shape)} does not fit {names[key]} {tuple(target.shape)}")
            target.copy_(t.to(target.dtype))
            filled.add(names[key])
    missing = sorted(set(params) - filled)
    if missing:
        raise KeyError(f"checkpoint is missing {missing[:8]}")


def load_weights(module: GraniteHybridDecoder, checkpoint: Path) -> None:
    from safetensors import safe_open

    files = sorted(checkpoint.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {checkpoint}")
    tensors = {}
    for f in files:
        with safe_open(f, framework="pt") as st:
            tensors |= {k: st.get_tensor(k) for k in st.keys()}
    load_state_dict(module, tensors)


def read_config(checkpoint: Path) -> dict[str, Any]:
    return json.loads((checkpoint / "config.json").read_text())

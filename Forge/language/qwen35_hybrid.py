"""Qwen3.5 hybrid decoder (Gated DeltaNet + gated full attention) for Aether Kit language exports.

Development tooling only; never required by SDK consumers. Written for AetherKit from the checkpoint's config and the
Qwen3.5 architecture. The layer structure and the gated delta rule follow Transformers' `Qwen3_5ForConditionalGeneration`
(5.22.1; Apache-2.0, copyright the Qwen team and the HuggingFace Team), which `reference_qwen35.py` runs as the
independent reference in its own environment (Forge/reference-qwen35); see NOTICE. The multi-token prediction head
(`mtp.*`) is not used, as upstream doesn't use it either. The vision tower (`model.visual.*`) is a separate asset.

Step contract (`hybrid`, docs/spikes/qwen3.5-vl-2026-11.md): the bucketed KV contract for the full-attention layers plus
declared recurrent states for the linear-attention layers, at a fixed width W with a padded tail.

    main(input_ids [1, W] i32, cache_positions [1, W] i32 = start..start+n-1 then -1, rope_positions [3, W] i32,
         image_embeds [W, hidden] fp16, keyCache, valueCache [attn layers, 1, kv_heads, capacity, head_dim] fp16
         (read-only inputs); states convState [linear layers, 1, conv_dim, kernel - 1] fp16,
         recurrentState [linear layers, 1, v_heads, k_head_dim, v_head_dim] fp32)
      -> logits [1, 1, vocab] (the last valid row), keyUpdate, valueUpdate [attn layers, 1, kv_heads, W, head_dim]

- Full attention: one SDPA over [cache ‖ chunk] with a boolean mask from position data; the chunk's keys and values
  are returned for the host to write at the valid rows. The query projection carries an output gate
  (output × sigmoid(gate)); rotary covers the first `partial_rotary_factor` of each head with interleaved M-RoPE.
- Gated DeltaNet: depthwise causal conv over [convState ‖ chunk], then the gated delta rule over the chunk's tokens
  from recurrentState, written back in place. Recurrent state can't step back, so a host never resubmits rows: the
  prompt's tail is padded instead.
- Padding (cache position -1) changes no state: its decay is 0 (g = 0, so exp(g) = 1) and its write strength β is 0;
  its q, k and v are zeroed so no NaN can reach the state; the conv state keeps the last kernel - 1 *valid* inputs;
  its keys are masked; logits come from the last valid row. A padded query row attends to itself only, so no row of
  the mask is empty.
- Image rows are aligned with the tokens and used only where the token is the image token (text steps send zeros).
- Norms: the checkpoint's zero-centred RMSNorm gains (x̂ · (1 + w)) are folded to plain gains (1 + w) at load, in
  float32, before any cast. The DeltaNet output norm is a plain gain (upstream `RMSNormGated`).

Every read of a state's old contents must feed that state's write (F-P2-20); `granite_hybrid.state_order_violations`
checks an exported graph.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from granite_hybrid import after  # noqa: F401  (re-exported for the exporter's lint helpers)
from qwen3_vl_text import MRotaryEmbedding, mrope_axes

PREFIX = "model.language_model."
DELTA_RULES = ("chunk", "composite", "loop")


class RMSNorm(torch.nn.Module):
    """Plain gain `w`; float32 statistics."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.to(torch.float32)
        return x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.normalize(x) * self.weight.float()).to(x.dtype)


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)


def delta_rule_loop(q, k, v, g, beta, state):
    """The gated delta rule, one token at a time (upstream's recurrent form). q, k [1, H, s, Dk] (l2-normed and
    scaled here, like the composite), v [1, H, s, Dv], g and beta [1, H, s], state [1, H, Dk, Dv]; all float32.
    Returns (output [1, s, H, Dv], new state)."""
    q, k = l2norm(q), l2norm(k)
    q = q * (q.shape[-1] ** -0.5)
    outs = []
    for t in range(q.shape[2]):
        state = state * g[:, :, t].exp()[..., None, None]
        kv_mem = (state * k[:, :, t, :, None]).sum(dim=-2)
        delta = (v[:, :, t] - kv_mem) * beta[:, :, t, None]
        state = state + k[:, :, t, :, None] * delta[:, :, None, :]
        outs.append((state * q[:, :, t, :, None]).sum(dim=-2))
    return torch.stack(outs, dim=1), state


def unit_lower_inverse(m: torch.Tensor) -> torch.Tensor:
    """Inverse of unit lower-triangular matrices m [..., W, W] (W a power of two) by blocked doubling: from the 1 x 1
    diagonal blocks (all 1), each level joins pairs of inverted diagonal blocks A⁻¹, B⁻¹ with their off-diagonal block
    C into [[A⁻¹, 0], [-B⁻¹ C A⁻¹, B⁻¹]]. log2(W) levels of small batched matmuls, no loop over rows, and no matrix
    powers: the power series (I - N)⁻¹ = Σ Nᵏ cancels catastrophically in float32 on real weights (measured: logits
    cosine 0.96, then NaN, on the 4B), this doesn't."""
    width, lead = m.shape[-1], m.shape[:-2]
    inverse = torch.ones(*lead, width, 1, 1, dtype=m.dtype)
    b = 1
    while b < width:
        nb = width // (2 * b)
        pick = torch.eye(nb, dtype=m.dtype).reshape(nb, 1, nb, 1)
        blocks = (m.reshape(*lead, nb, 2 * b, nb, 2 * b) * pick).sum(dim=-2)          # diagonal 2b blocks [.., nb, 2b, 2b]
        pairs = inverse.reshape(*lead, nb, 2, b, b)
        a_inv, b_inv = pairs[..., 0, :, :], pairs[..., 1, :, :]
        x = -(b_inv @ blocks[..., b:, :b] @ a_inv)
        top = torch.cat((a_inv, torch.zeros_like(a_inv)), dim=-1)
        inverse = torch.cat((top, torch.cat((x, b_inv), dim=-1)), dim=-2)
        b *= 2
    return inverse.reshape(*lead, width, width)


def delta_rule_chunk(q, k, v, g, beta, state, max_width: int):
    """The gated delta rule over the whole submission as one chunk (the WY form of upstream's chunked rule), no loop:
    same arguments and result as `delta_rule_loop`. A submission of s < max_width tokens (D1) is padded to max_width
    tokens with g = 0, β = 0 and zero q, k, v, which change nothing, so every matrix below is a static W x W.

    With G the running sum of g and N the strictly lower matrix -β_i (k_i · k_j) exp(G_i - G_j), T = (I - N)⁻¹
    (`unit_lower_inverse`). From the carried state S: u = T (β v) - T (β exp(G) k) S,
    o = exp(G) q S + (q kᵀ ∘ D) u with D_ij = exp(G_i - G_j) for j <= i, S' = exp(G_last) S + (exp(G_last - G) k)ᵀ u.
    The decay exponents are masked before exp, so no entry above the diagonal can overflow."""
    s = q.shape[2]
    width = 1
    while width < max_width:
        width *= 2
    if isinstance(s, int) and s == width:
        place = None                                                                   # S2: already W wide
    else:
        # D1: a symbolic s. coreai-torch can't lower a pad by a symbolic amount, so the rows are placed with a
        # [W, s] selection matrix instead (and taken back with its transpose).
        rows = torch.cumsum(torch.ones_like(g[0, 0], dtype=torch.int32), 0, dtype=torch.int32) - 1
        place = (torch.arange(width, dtype=torch.int32)[:, None] == rows[None, :]).to(q.dtype)
        q, k, v = (place @ t for t in (q, k, v))
        g, beta = (place @ g[..., None])[..., 0], (place @ beta[..., None])[..., 0]
    q, k = l2norm(q), l2norm(k)
    q = q * (q.shape[-1] ** -0.5)
    G = torch.cumsum(g, dim=-1)                                                        # [1, H, W]
    idx = torch.arange(width, dtype=torch.int32)
    lower_incl = idx[None, :] <= idx[:, None]                                          # [W, W], j <= i
    strict = idx[None, :] < idx[:, None]
    decay = torch.exp(torch.where(lower_incl, G[..., :, None] - G[..., None, :], -1e4))
    kb = k * beta[..., None]
    n = torch.where(strict, -(kb @ k.transpose(-1, -2)) * decay, 0.0)
    t = unit_lower_inverse(torch.eye(width, dtype=q.dtype) - n)
    eg = torch.exp(G)[..., None]                                                       # [1, H, W, 1]
    u = t @ (v * beta[..., None]) - (t @ (kb * eg)) @ state
    attn = torch.where(lower_incl, (q @ k.transpose(-1, -2)) * decay, 0.0)
    out = (q * eg) @ state + attn @ u                                                  # [1, H, W, Dv]
    last = G[..., -1:]                                                                 # [1, H, 1]
    new = torch.exp(last)[..., None] * state + (k * torch.exp(last - G)[..., None]).transpose(-1, -2) @ u
    if place is not None:
        out = place.transpose(0, 1) @ out
    return out.transpose(1, 2), new


class GatedDeltaNet(torch.nn.Module):
    def __init__(self, cfg: dict[str, Any], delta_rule: str, max_width: int = 64) -> None:
        super().__init__()
        hidden = cfg["hidden_size"]
        self.max_width = max_width
        self.k_heads, self.v_heads = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
        self.k_dim, self.v_dim = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
        self.key_width, self.value_width = self.k_heads * self.k_dim, self.v_heads * self.v_dim
        self.conv_dim = 2 * self.key_width + self.value_width
        self.kernel = cfg["linear_conv_kernel_dim"]
        self.in_proj_qkv = torch.nn.Linear(hidden, self.conv_dim, bias=False)
        self.in_proj_z = torch.nn.Linear(hidden, self.value_width, bias=False)
        self.in_proj_b = torch.nn.Linear(hidden, self.v_heads, bias=False)
        self.in_proj_a = torch.nn.Linear(hidden, self.v_heads, bias=False)
        self.conv_weight = torch.nn.Parameter(torch.zeros(self.conv_dim, self.kernel))
        self.dt_bias = torch.nn.Parameter(torch.ones(self.v_heads))
        self.A_log = torch.nn.Parameter(torch.zeros(self.v_heads))
        self.norm = RMSNorm(self.v_dim, cfg["rms_norm_eps"])
        self.out_proj = torch.nn.Linear(self.value_width, hidden, bias=False)
        if delta_rule not in DELTA_RULES:
            raise ValueError(f"delta rule {delta_rule!r} is not one of {DELTA_RULES}")
        self.delta_rule = delta_rule
        if delta_rule == "composite":
            from coreai_torch.composite_ops import GatedDeltaUpdate

            self.update = GatedDeltaUpdate(use_qk_l2_norm=True)

    def forward(self, x, conv_state, recurrent_state, slot: int, valid):
        """x [1, q, hidden]; valid bool [q] (a valid prefix, then padding)."""
        q_len = x.shape[1]
        qkv = self.in_proj_qkv(x)[0]                                                   # [q, conv_dim]
        z = self.in_proj_z(x)[0].reshape(q_len, self.v_heads, self.v_dim)
        b, a = self.in_proj_b(x)[0], self.in_proj_a(x)[0]                              # [q, v_heads]

        old = conv_state[slot, 0]                                                      # [conv_dim, kernel - 1]
        joined = torch.cat((old, qkv.transpose(0, 1).to(old.dtype)), dim=1)           # [conv_dim, q + kernel - 1]
        # The last kernel - 1 valid inputs sit at joined columns n..n + kernel - 2 (n valid tokens).
        n = valid.to(torch.int32).sum()
        tail = n + torch.cumsum(torch.ones(self.kernel - 1, dtype=torch.int32), 0, dtype=torch.int32) - 1
        conv_state[slot:slot + 1] = torch.index_select(joined, 1, tail)[None, None]
        w = self.conv_weight.float()
        conv = sum(joined[:, j:j + q_len].float() * w[:, j:j + 1] for j in range(self.kernel))
        mixed = torch.where(valid[:, None], F.silu(conv).transpose(0, 1), 0.0)         # [q, conv_dim] fp32

        query = mixed[:, :self.key_width].reshape(q_len, self.k_heads, self.k_dim)
        key = mixed[:, self.key_width:2 * self.key_width].reshape(q_len, self.k_heads, self.k_dim)
        value = mixed[:, 2 * self.key_width:].reshape(q_len, self.v_heads, self.v_dim)
        repeat = self.v_heads // self.k_heads
        if repeat > 1:                                                                 # key head i serves v heads i*r..
            query = query[:, :, None].expand(q_len, self.k_heads, repeat, self.k_dim).reshape(q_len, self.v_heads, self.k_dim)
            key = key[:, :, None].expand(q_len, self.k_heads, repeat, self.k_dim).reshape(q_len, self.v_heads, self.k_dim)
        beta = torch.where(valid[:, None], torch.sigmoid(b.float()), 0.0)              # [q, v_heads]
        g = torch.where(valid[:, None], -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float()), 0.0)

        h0 = recurrent_state[slot:slot + 1, 0]                                         # [1, H, Dk, Dv] fp32
        args = (query.transpose(0, 1)[None], key.transpose(0, 1)[None], value.transpose(0, 1)[None],
                g.transpose(0, 1)[None], beta.transpose(0, 1)[None], h0)
        if self.delta_rule == "composite":
            out, h = self.update(*args)
        elif self.delta_rule == "chunk":
            out, h = delta_rule_chunk(*args, max_width=self.max_width)
        else:
            out, h = delta_rule_loop(*args)
        recurrent_state[slot:slot + 1] = after(h.to(torch.float32), out)[None]
        y = self.norm.normalize(out[0]) * self.norm.weight.float() * F.silu(z.float())  # [q, H, Dv]
        return self.out_proj(y.reshape(1, q_len, self.value_width).to(x.dtype))


class Attention(torch.nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden = cfg["hidden_size"]
        self.heads, self.kv_heads, self.head_dim = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.q_proj = torch.nn.Linear(hidden, 2 * self.heads * self.head_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = torch.nn.Linear(self.heads * self.head_dim, hidden, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg["rms_norm_eps"])
        self.k_norm = RMSNorm(self.head_dim, cfg["rms_norm_eps"])

    @staticmethod
    def rotate(x, cos, sin):
        """Rotary on the first cos.shape[-1] dims (split halves), the rest passed through. x [1, heads, q, D]."""
        r = cos.shape[-1]
        rot, rest = x[..., :r], x[..., r:]
        half = r // 2
        rot = rot * cos + torch.cat((-rot[..., half:], rot[..., :half]), dim=-1) * sin
        return torch.cat((rot, rest), dim=-1)

    def forward(self, x, cos, sin, key_cache, value_cache, slot: int, visible):
        q_len = x.shape[1]
        qg = self.q_proj(x).view(1, q_len, self.heads, 2 * self.head_dim)
        query, gate = qg[..., :self.head_dim], qg[..., self.head_dim:]
        query = self.q_norm(query).transpose(1, 2)                                     # [1, H, q, D]
        key = self.k_norm(self.k_proj(x).view(1, q_len, self.kv_heads, self.head_dim)).transpose(1, 2)
        value = self.v_proj(x).view(1, q_len, self.kv_heads, self.head_dim).transpose(1, 2).contiguous()
        cos, sin = cos[None, None], sin[None, None]
        query, key = self.rotate(query, cos, sin), self.rotate(key, cos, sin).contiguous()
        keys = torch.cat((key_cache[slot], key), dim=2)
        values = torch.cat((value_cache[slot], value), dim=2)
        attn = F.scaled_dot_product_attention(query, keys, values, attn_mask=visible, enable_gqa=self.heads != self.kv_heads)
        attn = attn.transpose(1, 2).reshape(1, q_len, self.heads * self.head_dim)
        attn = attn * torch.sigmoid(gate.reshape(1, q_len, self.heads * self.head_dim))
        return self.o_proj(attn), key, value


class Layer(torch.nn.Module):
    def __init__(self, cfg: dict[str, Any], kind: str, delta_rule: str, max_width: int) -> None:
        super().__init__()
        hidden, eps = cfg["hidden_size"], cfg["rms_norm_eps"]
        self.kind = kind
        self.input_norm = RMSNorm(hidden, eps)
        self.post_norm = RMSNorm(hidden, eps)
        self.mixer = GatedDeltaNet(cfg, delta_rule, max_width) if kind == "linear_attention" else Attention(cfg)
        inter = cfg["intermediate_size"]
        self.gate_proj = torch.nn.Linear(hidden, inter, bias=False)
        self.up_proj = torch.nn.Linear(hidden, inter, bias=False)
        self.down_proj = torch.nn.Linear(inter, hidden, bias=False)

    def mlp(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen35HybridDecoder(torch.nn.Module):
    def __init__(self, cfg: dict[str, Any], image_token: int, max_capacity: int = 16384, delta_rule: str = "chunk",
                 max_width: int = 64) -> None:
        super().__init__()
        self.cfg, self.image_token, self.max_capacity = cfg, image_token, max_capacity
        self.kinds = list(cfg["layer_types"])
        if set(self.kinds) != {"linear_attention", "full_attention"} or not cfg.get("tie_word_embeddings", False):
            raise NotImplementedError("Qwen3.5 hybrids with tied embeddings only")
        if not cfg.get("attn_output_gate", False) or cfg.get("attention_bias"):
            raise NotImplementedError("gated attention without bias only")
        self.embed = torch.nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = torch.nn.ModuleList(Layer(cfg, kind, delta_rule, max_width) for kind in self.kinds)
        self.norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.slot = [self.kinds[:i].count(kind) for i, kind in enumerate(self.kinds)]
        self.linear_layers, self.attention_layers = self.kinds.count("linear_attention"), self.kinds.count("full_attention")
        self.rope = self.make_rope(cfg)
        self.start_scalar = False   # D1: `cache_positions` is [1, 2] = (start, valid count)

    @staticmethod
    def make_rope(cfg: dict[str, Any]) -> MRotaryEmbedding:
        rope = cfg["rope_parameters"]
        rotary = int(cfg["head_dim"] * rope.get("partial_rotary_factor", 1.0))
        return MRotaryEmbedding(rotary, float(rope["rope_theta"]),
                                mrope_axes(list(rope["mrope_section"]), bool(rope.get("mrope_interleaved", False))))

    def forward(self, input_ids, cache_positions, rope_positions, image_embeds, key_cache, value_cache, conv_state,
                recurrent_state):
        """One submission. Cache position -1 marks padding after a valid prefix of at least one token; padding changes
        no state, and its keyUpdate/valueUpdate rows are returned but must not be written."""
        q_len = input_ids.shape[1]
        x = self.embed(input_ids)
        x = torch.where((input_ids[0] == self.image_token)[None, :, None], image_embeds[None].to(x.dtype), x)
        rows = torch.cumsum(torch.ones_like(input_ids[0]), 0, dtype=torch.int32) - 1      # 0..q-1, from input_ids
        if self.start_scalar:
            # D1: (start, valid count) built into positions in the graph, so q has one carrier (0.0.3 defect 2).
            valid = rows < cache_positions[0, 1]
            positions = torch.where(valid, cache_positions[0, 0] + rows, -1)
        else:
            positions = cache_positions[0]
            valid = positions >= 0
        key_positions = torch.cumsum(torch.ones_like(key_cache[0, 0, 0, :, 0], dtype=torch.int32), 0, dtype=torch.int32) - 1
        cache_visible = key_positions[None, :] < positions[:1, None]
        diagonal = rows[None, :] == rows[:, None]
        chunk_visible = ((positions[None, :] <= positions[:, None]) & valid[None, :]) | (diagonal & ~valid[:, None])
        visible = torch.cat((cache_visible.expand(q_len, -1), chunk_visible), dim=1)  # bool [q, capacity + q]
        cos, sin = self.rope(rope_positions, x.dtype)
        keys, values = [], []
        for layer, slot in zip(self.layers, self.slot):
            h = layer.input_norm(x)
            if layer.kind == "linear_attention":
                h = layer.mixer(h, conv_state, recurrent_state, slot, valid)
            else:
                h, k, v = layer.mixer(h, cos, sin, key_cache, value_cache, slot, visible)
                keys.append(k)
                values.append(v)
            x = x + h
            x = x + layer.mlp(layer.post_norm(x))
        last = torch.index_select(x, 1, valid.to(torch.int32).sum().reshape(1) - 1)
        logits = F.linear(self.norm(last), self.embed.weight)
        return logits, torch.stack(keys), torch.stack(values)

    def empty_state(self, capacity: int):
        """(keyCache, valueCache, convState, recurrentState) zeros: KV and conv in the module dtype, recurrent fp32."""
        dtype = self.embed.weight.dtype
        attn = self.layers[self.kinds.index("full_attention")].mixer
        lin = self.layers[self.kinds.index("linear_attention")].mixer
        kv = (self.attention_layers, 1, attn.kv_heads, capacity, attn.head_dim)
        return (torch.zeros(kv, dtype=dtype), torch.zeros(kv, dtype=dtype),
                torch.zeros(self.linear_layers, 1, lin.conv_dim, lin.kernel - 1, dtype=dtype),
                torch.zeros(self.linear_layers, 1, lin.v_heads, lin.k_dim, lin.v_dim, dtype=torch.float32))

    def step(self, tokens, start: int, width: int, state, rope=None, image_rows=None):
        """One host-loop submission on torch tensors: `tokens` (n <= width valid ids at cache rows start..start+n-1)
        padded to `width`; `rope` [3, n] (default: text positions), `image_rows` [n, hidden] (default zeros). Writes the
        valid KV rows like the host; the recurrent states are updated by the call. Returns last-valid-row logits."""
        n = len(tokens)
        kc, vc, cs, rs = state
        ids = torch.zeros(1, width, dtype=torch.int32)
        ids[0, :n] = torch.as_tensor(tokens, dtype=torch.int32)
        pos = torch.full((1, width), -1, dtype=torch.int32)
        pos[0, :n] = torch.arange(start, start + n, dtype=torch.int32)
        rp = torch.zeros(3, width, dtype=torch.int32)
        rp[:, :n] = pos[0, :n] if rope is None else torch.as_tensor(rope, dtype=torch.int32)
        hidden = self.cfg["hidden_size"]
        img = torch.zeros(width, hidden, dtype=self.embed.weight.dtype)
        if image_rows is not None:
            img[:n] = image_rows
        positions = torch.tensor([[start, n]], dtype=torch.int32) if self.start_scalar else pos
        logits, ku, vu = self(ids, positions, rp, img, kc, vc, cs, rs)
        kc[:, :, :, start:start + n] = ku[:, :, :, :n]
        vc[:, :, :, start:start + n] = vu[:, :, :, :n]
        return logits[0, -1]


class PaddedMain(torch.nn.Module):
    """`main` at a fixed width W (S2): the inputs in the contract's order."""

    def __init__(self, inner: Qwen35HybridDecoder) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, input_ids, cache_positions, rope_positions, image_embeds, key_cache, value_cache, conv_state,
                recurrent_state):
        return self.inner(input_ids, cache_positions, rope_positions, image_embeds, key_cache, value_cache, conv_state,
                          recurrent_state)


def padded_example(module: Qwen35HybridDecoder, width: int):
    """(example, dynamic shapes) for `PaddedMain`: q fixed at `width`, a padded tail in the example, capacity dynamic."""
    cap_dim = torch.export.Dim("capacity", min=2, max=module.max_capacity)
    kc, vc, cs, rs = module.empty_state(56)
    start, n = 16, max(1, width - 3)
    positions = torch.full((1, width), -1, dtype=torch.int32)
    positions[0, :n] = torch.arange(start, start + n, dtype=torch.int32)
    ids = torch.arange(width, dtype=torch.int32)[None] + 100
    ids[0, 2:min(6, n)] = module.image_token
    rope = torch.zeros(3, width, dtype=torch.int32)
    rope[:, :n] = positions[0, :n]
    image = torch.zeros(width, module.cfg["hidden_size"], dtype=module.embed.weight.dtype)
    example = (ids, positions, rope, image, kc, vc, cs, rs)
    return example, (None, None, None, None, {3: cap_dim}, {3: cap_dim}, None, None)


def start_scalar_example(module: Qwen35HybridDecoder, max_width: int):
    """(example, dynamic shapes) for D1: q dynamic (1..max_width) on input_ids, rope_positions and image_embeds,
    `cache_positions` [1, 2] = (start, valid count), capacity dynamic."""
    q_dim = torch.export.Dim("q", min=1, max=max_width)
    cap_dim = torch.export.Dim("capacity", min=2, max=module.max_capacity)
    kc, vc, cs, rs = module.empty_state(56)
    width, start = max(3, min(12, max_width - 1)), 16
    n = width - 1
    ids = torch.arange(width, dtype=torch.int32)[None] + 100
    ids[0, 1:n] = module.image_token
    rope = torch.zeros(3, width, dtype=torch.int32)
    rope[:, :n] = torch.arange(start, start + n, dtype=torch.int32)
    image = torch.zeros(width, module.cfg["hidden_size"], dtype=module.embed.weight.dtype)
    example = (ids, torch.tensor([[start, n]], dtype=torch.int32), rope, image, kc, vc, cs, rs)
    return example, ({1: q_dim}, None, {1: q_dim}, {0: q_dim}, {3: cap_dim}, {3: cap_dim}, None, None)


def name_map(module: Qwen35HybridDecoder) -> dict[str, tuple[str, str]]:
    """checkpoint key -> (module parameter, transform); transform 'gain' folds a zero-centred norm (1 + w)."""
    m = {PREFIX + "embed_tokens.weight": ("embed.weight", ""), PREFIX + "norm.weight": ("norm.weight", "gain")}
    for i, layer in enumerate(module.layers):
        src, dst = f"{PREFIX}layers.{i}.", f"layers.{i}."
        m |= {f"{src}input_layernorm.weight": (f"{dst}input_norm.weight", "gain"),
              f"{src}post_attention_layernorm.weight": (f"{dst}post_norm.weight", "gain")}
        for p in ("gate_proj", "up_proj", "down_proj"):
            m[f"{src}mlp.{p}.weight"] = (f"{dst}{p}.weight", "")
        if layer.kind == "linear_attention":
            for a, b in [("A_log", "A_log"), ("dt_bias", "dt_bias"), ("conv1d.weight", "conv_weight"),
                         ("norm.weight", "norm.weight"), ("out_proj.weight", "out_proj.weight"),
                         ("in_proj_qkv.weight", "in_proj_qkv.weight"), ("in_proj_z.weight", "in_proj_z.weight"),
                         ("in_proj_a.weight", "in_proj_a.weight"), ("in_proj_b.weight", "in_proj_b.weight")]:
                m[f"{src}linear_attn.{a}"] = (f"{dst}mixer.{b}", "")
        else:
            for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
                m[f"{src}self_attn.{p}.weight"] = (f"{dst}mixer.{p}.weight", "")
            m[f"{src}self_attn.q_norm.weight"] = (f"{dst}mixer.q_norm.weight", "gain")
            m[f"{src}self_attn.k_norm.weight"] = (f"{dst}mixer.k_norm.weight", "gain")
    return m


def build(cfg: dict[str, Any], image_token: int, dtype: torch.dtype, max_capacity: int = 16384,
          delta_rule: str = "chunk", max_width: int = 64) -> Qwen35HybridDecoder:
    """An uninitialised decoder in `dtype` for `load_weights` to fill, built on the meta device (no float32 copy of a
    4B is allocated for an fp16 build)."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("meta"):
            module = Qwen35HybridDecoder(cfg, image_token, max_capacity=max_capacity, delta_rule=delta_rule,
                                         max_width=max_width)
    finally:
        torch.set_default_dtype(previous)
    module.to_empty(device="cpu")
    module.rope = Qwen35HybridDecoder.make_rope(cfg)   # float32 tables, not parameters: rebuilt off the meta device
    return module


def load_weights(module: Qwen35HybridDecoder, checkpoint: Path) -> None:
    """Text weights under `model.language_model.`; vision (`model.visual.`) and MTP (`mtp.`) are skipped."""
    from safetensors import safe_open

    names = name_map(module)
    params = dict(module.named_parameters())
    filled = set()
    with torch.no_grad():
        for f in sorted(checkpoint.glob("*.safetensors")):
            with safe_open(f, framework="pt") as st:
                for key in st.keys():
                    if key.startswith(("model.visual.", "mtp.")) or key == "lm_head.weight":
                        continue
                    if key not in names:
                        raise KeyError(f"unexpected checkpoint tensor {key}")
                    dst, transform = names[key]
                    target = params[dst]
                    t = st.get_tensor(key).to(torch.float32)
                    if key.endswith("conv1d.weight"):
                        t = t.reshape(target.shape)                                    # [C, 1, k] -> [C, k]
                    if transform == "gain":
                        t = 1.0 + t
                    if tuple(t.shape) != tuple(target.shape):
                        raise ValueError(f"{key}: {tuple(t.shape)} does not fit {dst} {tuple(target.shape)}")
                    target.copy_(t.to(target.dtype))
                    filled.add(dst)
    missing = sorted(set(params) - filled)
    if missing:
        raise KeyError(f"checkpoint is missing {missing[:8]}")


def read_config(checkpoint: Path) -> tuple[dict[str, Any], int]:
    """(text config, image token id)."""
    cfg = json.loads((checkpoint / "config.json").read_text())
    return cfg["text_config"], int(cfg["image_token_id"])

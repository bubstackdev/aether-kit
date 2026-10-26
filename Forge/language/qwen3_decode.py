"""Qwen3 dense decoder for Aether Kit language exports, with two step contracts.

Development tooling only; never required by SDK consumers. Adapted from the aether-engine forge Qwen3 family
(`forge/aether_forge/arch/qwen3.py`, `prim/`), which records the Core AI facts cited below as F-P2-*. The engine
files are not imported, so this directory stays self-contained.

Both contracts share the weights and the cache layout `[layers, 1, kv_heads, capacity, head_dim]` (capacity axis 3
dynamic). The function signature alone identifies the contract.

``running`` (legacy, identical to Apple's LLM exports and the 0.0.1 Kit contract):
    main(input_ids [1, q] i32, position_ids [1, total] i32 = 0..<total; states keyCache, valueCache mutated in place)
      -> logits [1, q, vocab]
    The write offset is `total - q`, a shape-derived slice bound. Every decode step has a new `total`, so every step
    is a new input shape, and the native shape cache grows with the generated length.

``bucketed`` (0.0.2, host-written cache):
    main(input_ids [1, q] i32, cache_positions [1, q] i32 = start..<start+q,
         keyCache [layers, 1, kv_heads, capacity, head_dim], valueCache (read-only inputs, no states))
      -> logits [1, 1, vocab], keyUpdate [layers, 1, kv_heads, q, head_dim], valueUpdate
    The graph never writes the cache. Each layer runs one fused SDPA over [cache ‖ chunk] keys with a boolean mask
    (cache rows whose position is below `start`, then the chunk causally) and returns the chunk's keys/values; the
    host copies them into rows start..<start+q after the call. No slice bound or value depends on a position: masks
    come from position data (F-P2-6), so input shapes depend only on (q, capacity). The host keeps q in powers of two
    up to `MAX_PREFILL_CHUNK` and capacity on its growth ladder, which bounds the shapes a session can submit.

    Forms measured and rejected on Qwen3-0.6B fp16, M5 Max, release Swift, warm 256-token decode after a 33-token
    prompt (running contract: 151-154 tok/s; this form: 169 tok/s), Forge/language spikes 2026-10-26:
    - In-graph cache write at a data-derived offset (`slice_update` whose start is read from `cache_positions`):
      exact on the GPU but 39 tok/s, and Core AI's cpu_only runtime fails it (inferenceFailed, or SIGTRAP with a
      boolean SDPA mask).
    - Host-written cache with a hand-written split softmax (cache scores and chunk scores merged without a copy):
      exact, CPU-clean, but 70 tok/s; about a dozen extra kernels per layer outweigh the [cache ‖ chunk] copy.
    - SDPA with `enable_gqa` and a float additive mask fails on cpu_only; the boolean mask used here runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

CONTRACTS = ("running", "bucketed")
# The position input's name identifies the contract from the function signature alone, so a loader that only knows
# `running` rejects a `bucketed` asset instead of misreading its positions.
POSITION_INPUT = {"running": "position_ids", "bucketed": "cache_positions"}
MAX_PREFILL_CHUNK = 512


class RMSNorm(torch.nn.Module):
    """Gain `w`; float32 statistics (fp16 squares overflow)."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.to(torch.float32)
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x) * self.weight.to(x.dtype)


class RotaryEmbedding(torch.nn.Module):
    """Split-halves RoPE; frequencies kept float32 (theta 1e6 underflows fp16 at the smallest frequencies)."""

    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        exponents = torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim
        self.inv_freq = (1.0 / (theta**exponents)).to(torch.float32)     # plain attribute: `.to(fp16)` skips it

    def forward(self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        angles = positions.to(torch.float32)[:, None] * self.inv_freq[None, :]
        emb = torch.cat((angles, angles), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sin


class Block(torch.nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden, heads, kv_heads = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.head_dim = cfg.get("head_dim") or hidden // heads
        self.heads, self.kv_heads = heads, kv_heads
        eps = cfg["rms_norm_eps"]
        self.attn_norm = RMSNorm(hidden, eps)
        # One fused q/k/v projection (F-P2-13: one quantized matmul instead of three).
        self.qkv_split = [heads * self.head_dim, kv_heads * self.head_dim, kv_heads * self.head_dim]
        self.qkv_proj = torch.nn.Linear(hidden, sum(self.qkv_split), bias=False)
        self.o_proj = torch.nn.Linear(heads * self.head_dim, hidden, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps)
        self.k_norm = RMSNorm(self.head_dim, eps)
        self.mlp_norm = RMSNorm(hidden, eps)
        inter = cfg["intermediate_size"]
        self.gate_proj = torch.nn.Linear(hidden, inter, bias=False)
        self.up_proj = torch.nn.Linear(hidden, inter, bias=False)
        self.down_proj = torch.nn.Linear(inter, hidden, bias=False)

    def project(self, x, cos, sin):
        """q [1, H, q, D], k and v [1, Hkv, q, D]."""
        q_len = x.shape[1]
        h = self.attn_norm(x)
        qkv = self.qkv_proj(h).view(1, q_len, self.heads + 2 * self.kv_heads, self.head_dim)
        qk = qkv[:, :, :self.heads + self.kv_heads]
        v = qkv[:, :, self.heads + self.kv_heads:].transpose(1, 2)
        # q and k share one normalization (each its own gain) and one rotation, in the [1, q, heads, head_dim] layout,
        # so a cache write takes a transpose output: with the rotation as its direct producer Core AI's CPU path drops
        # a row of the update (F-P2-13).
        gains = torch.cat((self.q_norm.weight.to(x.dtype).expand(self.heads, -1),
                           self.k_norm.weight.to(x.dtype).expand(self.kv_heads, -1)))
        qk = rotate(self.q_norm.normalize(qk) * gains, cos[:, None, :], sin[:, None, :])
        return qk[:, :, :self.heads].transpose(1, 2), qk[:, :, self.heads:].transpose(1, 2), v

    def finish(self, x, attn):
        q_len = x.shape[1]
        x = x + self.o_proj(attn.transpose(1, 2).reshape(1, q_len, self.heads * self.head_dim))
        h = self.mlp_norm(x)
        return x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))

    def running(self, x, cos, sin, key_cache, value_cache, layer: int, start, end, mask):
        """In-place cache write at shape-derived bounds (slice, not integer, layer index: F-P2-13), then attention over
        the first `end` rows. `mask` is bool [q, end]."""
        q, k, v = self.project(x, cos, sin)
        key_cache[layer:layer + 1, :, :, start:end, :] = k[None]
        value_cache[layer:layer + 1, :, :, start:end, :] = v[None]
        keys = key_cache[layer, :, :, :end, :]
        values = value_cache[layer, :, :, :end, :]
        attn = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask, enable_gqa=self.heads != self.kv_heads)
        return self.finish(x, attn)

    def bucketed(self, x, cos, sin, key_cache, value_cache, layer: int, visible):
        """One fused SDPA over read-only cache rows followed by the chunk's own keys (`visible` bool [q, capacity + q]);
        returns (x, k, v). The concatenation copies the layer's cache rows once per step."""
        q, k, v = self.project(x, cos, sin)
        # k and v are transposed views; as outputs they make export guard on symbolic strides.
        k, v = k.contiguous(), v.contiguous()
        keys = torch.cat((key_cache[layer], k), dim=2)
        values = torch.cat((value_cache[layer], v), dim=2)
        attn = F.scaled_dot_product_attention(q, keys, values, attn_mask=visible, enable_gqa=self.heads != self.kv_heads)
        return self.finish(x, attn), k, v


class Qwen3Decoder(torch.nn.Module):
    def __init__(self, cfg: dict[str, Any], contract: str, max_capacity: int = 16384) -> None:
        super().__init__()
        if contract not in CONTRACTS:
            raise ValueError(f"unknown contract {contract!r}")
        if cfg.get("use_sliding_window") or cfg.get("rope_scaling") or cfg.get("attention_bias"):
            raise NotImplementedError("sliding window, rope scaling and attention bias are not supported")
        self.cfg, self.contract = cfg, contract
        self.embed = torch.nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.blocks = torch.nn.ModuleList(Block(cfg) for _ in range(cfg["num_hidden_layers"]))
        self.norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.tied = bool(cfg.get("tie_word_embeddings", False))
        self.lm_head = None if self.tied else torch.nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        self.head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
        self.rope = RotaryEmbedding(self.head_dim, float(cfg["rope_theta"]))
        self.max_capacity = max_capacity

    def head(self, x):
        return F.linear(self.norm(x), self.embed.weight if self.tied else self.lm_head.weight)

    def forward(self, input_ids, position_ids, key_cache, value_cache):
        q_len = input_ids.shape[1]
        x = self.embed(input_ids)
        if self.contract == "running":
            total = position_ids.shape[1]
            positions = position_ids[0]
            new_positions = positions[total - q_len:]
            mask = positions[None, :] <= new_positions[:, None]      # data-derived lower-right causal (F-P2-6)
            cos, sin = self.rope(new_positions, x.dtype)
            for i, block in enumerate(self.blocks):
                x = block.running(x, cos, sin, key_cache, value_cache, i, total - q_len, total, mask)
            return self.head(x)
        new_positions = position_ids[0]                                # [q] = start..<start+q
        # Key positions as a running count over a capacity-sized row (data, never a shape-derived value: F-P2-6).
        key_positions = torch.cumsum(torch.ones_like(key_cache[0, 0, 0, :, 0], dtype=torch.int32), 0, dtype=torch.int32) - 1
        cache_visible = key_positions[None, :] < new_positions[:1, None]         # rows before the chunk, [1, cap]
        chunk_visible = new_positions[None, :] <= new_positions[:, None]         # causal within the chunk, [q, q]
        visible = torch.cat((cache_visible.expand(q_len, -1), chunk_visible), dim=1)  # [q, cap + q]
        cos, sin = self.rope(new_positions, x.dtype)
        keys, values = [], []
        for i, block in enumerate(self.blocks):
            x, k, v = block.bucketed(x, cos, sin, key_cache, value_cache, i, visible)
            keys.append(k)
            values.append(v)
        return self.head(x[:, -1:]), torch.stack(keys), torch.stack(values)

    def empty_state(self, capacity: int) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.cfg
        shape = (c["num_hidden_layers"], 1, c["num_key_value_heads"], capacity, self.head_dim)
        dtype = self.embed.weight.dtype
        return torch.zeros(shape, dtype=dtype), torch.zeros(shape, dtype=dtype)

    def example(self, max_len: int):
        """(example inputs, dynamic shapes) for torch.export. Example dims all differ so export keeps them apart."""
        q_dim = torch.export.Dim("q", min=1, max=max_len)
        cap_dim = torch.export.Dim("capacity", min=2, max=self.max_capacity)
        if self.contract == "running":
            ex_q, ex_total = 8, 24
            kc, vc = self.empty_state(2 * ex_total + 8)
            total_dim = torch.export.Dim("total", min=2, max=self.max_capacity)
            inputs = (torch.arange(ex_q, dtype=torch.int32)[None] + 100, torch.arange(ex_total, dtype=torch.int32)[None], kc, vc)
            return inputs, ({1: q_dim}, {1: total_dim}, {3: cap_dim}, {3: cap_dim})
        ex_q, ex_start = 8, 16
        kc, vc = self.empty_state(56)
        inputs = (torch.arange(ex_q, dtype=torch.int32)[None] + 100, torch.arange(ex_start, ex_start + ex_q, dtype=torch.int32)[None], kc, vc)
        return inputs, ({1: q_dim}, {1: q_dim}, {3: cap_dim}, {3: cap_dim})

    def step(self, input_ids, position_ids, key_cache, value_cache):
        """One host-loop step for either contract on torch caches: returns last-row logits [vocab] and, for the
        bucketed contract, writes the returned rows into the caches the way the host does."""
        if self.contract == "running":
            return self(input_ids, position_ids, key_cache, value_cache)[0, -1]
        logits, ku, vu = self(input_ids, position_ids, key_cache, value_cache)
        start, q = int(position_ids[0, 0]), input_ids.shape[1]
        key_cache[:, :, :, start:start + q] = ku
        value_cache[:, :, :, start:start + q] = vu
        return logits[0, -1]


def name_map(module: Qwen3Decoder) -> dict[str, tuple[str, int]]:
    """Checkpoint tensor name -> (parameter, first row); q/k/v land in row ranges of the fused projection."""
    m = {"model.embed_tokens.weight": ("embed.weight", 0), "model.norm.weight": ("norm.weight", 0)}
    if not module.tied:
        m["lm_head.weight"] = ("lm_head.weight", 0)
    for i, block in enumerate(module.blocks):
        src, dst = f"model.layers.{i}.", f"blocks.{i}."
        for a, b in [("input_layernorm", "attn_norm"), ("post_attention_layernorm", "mlp_norm"),
                     ("self_attn.q_norm", "q_norm"), ("self_attn.k_norm", "k_norm"), ("self_attn.o_proj", "o_proj"),
                     ("mlp.down_proj", "down_proj"), ("mlp.gate_proj", "gate_proj"), ("mlp.up_proj", "up_proj")]:
            m[f"{src}{a}.weight"] = (f"{dst}{b}.weight", 0)
        row = 0
        for p, n in zip(("q_proj", "k_proj", "v_proj"), block.qkv_split):
            m[f"{src}self_attn.{p}.weight"] = (f"{dst}qkv_proj.weight", row)
            row += n
    return m


def load_weights(module: Qwen3Decoder, checkpoint: Path) -> None:
    from safetensors import safe_open

    params = dict(module.named_parameters())
    names = name_map(module)
    loaded: dict[str, int] = {}
    files = sorted(checkpoint.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {checkpoint}")
    with torch.no_grad():
        for f in files:
            with safe_open(f, framework="pt") as st:
                for key in st.keys():
                    if key == "lm_head.weight" and module.tied:
                        continue
                    if key not in names:
                        raise KeyError(f"unexpected checkpoint tensor {key}")
                    dst, row = names[key]
                    t = st.get_tensor(key)
                    target = params[dst]
                    if tuple(t.shape[1:]) != tuple(target.shape[1:]) or row + t.shape[0] > target.shape[0]:
                        raise ValueError(f"{key}: {tuple(t.shape)} does not fit {dst} {tuple(target.shape)} at row {row}")
                    target[row:row + t.shape[0]].copy_(t.to(target.dtype))
                    loaded[dst] = loaded.get(dst, 0) + t.shape[0]
    missing = sorted(n for n, p in params.items() if loaded.get(n, 0) != p.shape[0])
    if missing:
        raise KeyError(f"checkpoint is missing {missing[:8]}")


def read_config(checkpoint: Path) -> dict[str, Any]:
    return json.loads((checkpoint / "config.json").read_text())

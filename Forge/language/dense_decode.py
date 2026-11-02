"""Shared dense-language export primitives and step contracts.

A family supplies a block with project/finish methods and a checkpoint tensor map.
The cache layout, causal masks, running/bucketed execution, RoPE, and host cache
updates are shared. Keep model-specific normalization inside the family block.

Both contracts use [layers, 1, kv_heads, capacity, head_dim] caches.
Running (legacy) mutates cache states at shape-derived offsets and returns every
logits row. Bucketed takes read-only cache inputs, forms [cache | chunk] causal
attention, and returns only the final logits row plus key/value updates. The
host copies updates to their absolute rows; input shapes depend on chunk width
and cache capacity rather than every new token position.

New dense exports default to float32 SDPA arithmetic to avoid a reproduced
Core AI CPU fp16 near-tie reversal. Weights and cache storage remain float16.
The explicitly selectable float16 path preserves the earlier export recipe;
Qwen3-VL's directly constructed blocks also retain that qualified path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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


class AttentionBlock(torch.nn.Module):
    """Shared attention/cache execution; subclasses supply project and finish."""

    def attend(self, q, keys, values, mask):
        # Core AI CPU fp16 SDPA can reverse close logits relative to PyTorch's
        # float32-accumulating reference. Explicit promotion preserves the strict
        # argmax check; weights, returned cache rows, and residuals stay fp16.
        # Qwen3-VL builds blocks directly and retains its previously qualified path.
        dtype = torch.float32 if getattr(self, "attention_precision", "float16") == "float32" else q.dtype
        return F.scaled_dot_product_attention(q.to(dtype), keys.to(dtype), values.to(dtype),
            attn_mask=mask, enable_gqa=self.heads != self.kv_heads).to(q.dtype)

    def running(self, x, cos, sin, key_cache, value_cache, layer: int, start, end, mask):
        """In-place cache write at shape-derived bounds (slice, not integer, layer index: F-P2-13), then attention over
        the first `end` rows. `mask` is bool [q, end]."""
        q, k, v = self.project(x, cos, sin)
        key_cache[layer:layer + 1, :, :, start:end, :] = k[None]
        value_cache[layer:layer + 1, :, :, start:end, :] = v[None]
        keys = key_cache[layer, :, :, :end, :]
        values = value_cache[layer, :, :, :end, :]
        attn = self.attend(q, keys, values, mask)
        return self.finish(x, attn)

    def bucketed(self, x, cos, sin, key_cache, value_cache, layer: int, visible):
        """One fused SDPA over read-only cache rows followed by the chunk's own keys (`visible` bool [q, capacity + q]);
        returns (x, k, v). The concatenation copies the layer's cache rows once per step."""
        q, k, v = self.project(x, cos, sin)
        # k and v are transposed views; as outputs they make export guard on symbolic strides.
        k, v = k.contiguous(), v.contiguous()
        keys = torch.cat((key_cache[layer], k), dim=2)
        values = torch.cat((value_cache[layer], v), dim=2)
        attn = self.attend(q, keys, values, visible)
        return self.finish(x, attn), k, v


class DenseDecoder(torch.nn.Module):
    def __init__(self, cfg: dict[str, Any], contract: str, max_capacity: int = 16384, *, block_factory, attention_precision="float32", norm_factory=None, rotary=True) -> None:
        super().__init__()
        if contract not in CONTRACTS:
            raise ValueError(f"unknown contract {contract!r}")
        if cfg.get("use_sliding_window") or cfg.get("rope_scaling") or cfg.get("attention_bias"):
            raise NotImplementedError("sliding window, rope scaling and attention bias are not supported")
        if attention_precision not in ("float16", "float32"):
            raise ValueError(f"unknown attention precision {attention_precision!r}")
        self.attention_precision = attention_precision
        self.cfg, self.contract = cfg, contract
        self.embed = torch.nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.blocks = torch.nn.ModuleList(block_factory(cfg) for _ in range(cfg["num_hidden_layers"]))
        for block in self.blocks:
            block.attention_precision = attention_precision
        self.norm = norm_factory() if norm_factory else RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.tied = bool(cfg.get("tie_word_embeddings", False))
        self.lm_head = None if self.tied else torch.nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        self.head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
        self.rope = RotaryEmbedding(self.head_dim, float(cfg["rope_theta"])) if rotary else None
        self.max_capacity = max_capacity

    def head(self, x):
        return F.linear(self.norm(x), self.embed.weight if self.tied else self.lm_head.weight)

    def embed_inputs(self, input_ids, positions):
        """Family hook for token/absolute-position embeddings before shared cache execution."""
        return self.embed(input_ids)

    def rotary_inputs(self, positions, dtype):
        return self.rope(positions, dtype) if self.rope is not None else (None, None)

    def forward(self, input_ids, position_ids, key_cache, value_cache):
        q_len = input_ids.shape[1]
        if self.contract == "running":
            total = position_ids.shape[1]
            positions = position_ids[0]
            new_positions = positions[total - q_len:]
            mask = positions[None, :] <= new_positions[:, None]      # data-derived lower-right causal (F-P2-6)
            x = self.embed_inputs(input_ids, new_positions)
            cos, sin = self.rotary_inputs(new_positions, x.dtype)
            for i, block in enumerate(self.blocks):
                x = block.running(x, cos, sin, key_cache, value_cache, i, total - q_len, total, mask)
            return self.head(x)
        new_positions = position_ids[0]                                # [q] = start..<start+q
        # Key positions as a running count over a capacity-sized row (data, never a shape-derived value: F-P2-6).
        key_positions = torch.cumsum(torch.ones_like(key_cache[0, 0, 0, :, 0], dtype=torch.int32), 0, dtype=torch.int32) - 1
        cache_visible = key_positions[None, :] < new_positions[:1, None]         # rows before the chunk, [1, cap]
        chunk_visible = new_positions[None, :] <= new_positions[:, None]         # causal within the chunk, [q, q]
        visible = torch.cat((cache_visible.expand(q_len, -1), chunk_visible), dim=1)  # [q, cap + q]
        x = self.embed_inputs(input_ids, new_positions)
        cos, sin = self.rotary_inputs(new_positions, x.dtype)
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


@dataclass(frozen=True)
class TensorMapping:
    """Checkpoint tensor destination; Conv1D checkpoints store linear weights transposed."""
    parameter: str
    row: int = 0
    transpose: bool = False


def load_weights(module: DenseDecoder, checkpoint: Path, names: dict[str, tuple[str, int] | TensorMapping], *, ignored_names=()) -> None:
    from safetensors import safe_open

    params = dict(module.named_parameters())
    loaded: dict[str, int] = {}
    files = sorted(checkpoint.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {checkpoint}")
    with torch.no_grad():
        for f in files:
            with safe_open(f, framework="pt") as st:
                for key in st.keys():
                    if key in ignored_names or (key == "lm_head.weight" and module.tied):
                        continue
                    if key not in names:
                        raise KeyError(f"unexpected checkpoint tensor {key}")
                    mapping = names[key]
                    dst, row = (mapping.parameter, mapping.row) if isinstance(mapping, TensorMapping) else mapping
                    t = st.get_tensor(key)
                    if isinstance(mapping, TensorMapping) and mapping.transpose:
                        t = t.T
                    target = params[dst]
                    if t.ndim != target.ndim or tuple(t.shape[1:]) != tuple(target.shape[1:]) or row + t.shape[0] > target.shape[0]:
                        raise ValueError(f"{key}: {tuple(t.shape)} does not fit {dst} {tuple(target.shape)} at row {row}")
                    target[row:row + t.shape[0]].copy_(t.to(target.dtype))
                    loaded[dst] = loaded.get(dst, 0) + t.shape[0]
    missing = sorted(n for n, p in params.items() if loaded.get(n, 0) != p.shape[0])
    if missing:
        raise KeyError(f"checkpoint is missing {missing[:8]}")


def read_config(checkpoint: Path) -> dict[str, Any]:
    return json.loads((checkpoint / "config.json").read_text())

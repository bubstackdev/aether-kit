"""Qwen3 dense decoder. Shared execution lives in dense_decode."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F
from dense_decode import (AttentionBlock, DenseDecoder, RMSNorm, RotaryEmbedding, rotate,
                          CONTRACTS, POSITION_INPUT, MAX_PREFILL_CHUNK, read_config)
from dense_decode import load_weights as load_mapped_weights

class Block(AttentionBlock):
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


class Qwen3Decoder(DenseDecoder):
    def __init__(self, cfg, contract, max_capacity=16384, attention_precision="float32"):
        super().__init__(cfg, contract, max_capacity, block_factory=Block, attention_precision=attention_precision)


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
    load_mapped_weights(module, checkpoint, name_map(module))

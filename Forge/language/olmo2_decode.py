"""OLMo 2: full-projection Q/K RMSNorm and post-sublayer residual normalization.

Reference: transformers 4.57.3 models/olmo2/modeling_olmo2.py (Apache-2.0).
No upstream implementation is imported by the exported module.
"""
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn.functional as F
from dense_decode import AttentionBlock, DenseDecoder, RMSNorm, rotate, load_weights as load_mapped_weights


class Olmo2Block(AttentionBlock):
    def __init__(self, cfg):
        super().__init__()
        hidden = cfg["hidden_size"]
        self.heads, self.kv_heads = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.head_dim = cfg.get("head_dim") or hidden // self.heads
        eps = cfg["rms_norm_eps"]
        self.qkv_split = [self.heads * self.head_dim, self.kv_heads * self.head_dim, self.kv_heads * self.head_dim]
        self.qkv_proj = torch.nn.Linear(hidden, sum(self.qkv_split), bias=False)
        self.q_norm = RMSNorm(self.qkv_split[0], eps)
        self.k_norm = RMSNorm(self.qkv_split[1], eps)
        self.o_proj = torch.nn.Linear(self.qkv_split[0], hidden, bias=False)
        self.attn_norm = RMSNorm(hidden, eps)
        self.mlp_norm = RMSNorm(hidden, eps)
        inter = cfg["intermediate_size"]
        self.gate_proj = torch.nn.Linear(hidden, inter, bias=False)
        self.up_proj = torch.nn.Linear(hidden, inter, bias=False)
        self.down_proj = torch.nn.Linear(inter, hidden, bias=False)

    def project(self, x, cos, sin):
        q_len = x.shape[1]
        q, k, v = self.qkv_proj(x).split(self.qkv_split, dim=-1)
        # Normalize before splitting heads: a per-head norm would silently change OLMo2.
        q = self.q_norm(q).view(1, q_len, self.heads, self.head_dim)
        k = self.k_norm(k).view(1, q_len, self.kv_heads, self.head_dim)
        q = rotate(q, cos[:, None, :], sin[:, None, :]).transpose(1, 2)
        k = rotate(k, cos[:, None, :], sin[:, None, :]).transpose(1, 2)
        v = v.view(1, q_len, self.kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def finish(self, x, attn):
        h = attn.transpose(1, 2).reshape(1, x.shape[1], self.heads * self.head_dim)
        x = x + self.attn_norm(self.o_proj(h))
        return x + self.mlp_norm(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Olmo2Decoder(DenseDecoder):
    def __init__(self, cfg, contract, max_capacity=16384, attention_precision="float32"):
        if cfg.get("model_type") != "olmo2" or cfg.get("hidden_act", "silu") != "silu":
            raise NotImplementedError("expected an OLMo2 checkpoint with SiLU")
        super().__init__(cfg, contract, max_capacity, block_factory=Olmo2Block, attention_precision=attention_precision)


def name_map(module):
    names = {"model.embed_tokens.weight": ("embed.weight", 0), "model.norm.weight": ("norm.weight", 0)}
    if not module.tied:
        names["lm_head.weight"] = ("lm_head.weight", 0)
    for i, block in enumerate(module.blocks):
        src, dst = f"model.layers.{i}.", f"blocks.{i}."
        for a, b in [("post_attention_layernorm", "attn_norm"), ("post_feedforward_layernorm", "mlp_norm"),
                     ("self_attn.q_norm", "q_norm"), ("self_attn.k_norm", "k_norm"),
                     ("self_attn.o_proj", "o_proj"), ("mlp.gate_proj", "gate_proj"),
                     ("mlp.up_proj", "up_proj"), ("mlp.down_proj", "down_proj")]:
            names[f"{src}{a}.weight"] = (f"{dst}{b}.weight", 0)
        row = 0
        for proj, n in zip(("q_proj", "k_proj", "v_proj"), block.qkv_split):
            names[f"{src}self_attn.{proj}.weight"] = (f"{dst}qkv_proj.weight", row)
            row += n
    return names


def load_weights(module, checkpoint: Path):
    load_mapped_weights(module, checkpoint, name_map(module))

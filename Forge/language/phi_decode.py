"""Phi-2: partial RoPE, biased projections, parallel attention/MLP residuals.

Reimplements Transformers 4.57.3 models/phi/modeling_phi.py (Apache-2.0).
The export graph imports no upstream model code; both cache contracts are shared.
"""
from pathlib import Path
import math
import torch
from dense_decode import AttentionBlock, DenseDecoder, rotate_prefix, load_weights as load_mapped_weights



class PhiLayerNorm(torch.nn.LayerNorm):
    """Keep normalization statistics/affine arithmetic in float32 on Core AI too."""
    def forward(self, x):
        return torch.nn.functional.layer_norm(x.float(), self.normalized_shape,
            self.weight.float(), self.bias.float(), self.eps).to(x.dtype)


class PhiLinear(torch.nn.Linear):
    """Explicit float32 projection arithmetic; weights remain stored in fp16."""
    def forward(self, x):
        bias = self.bias.float() if self.bias is not None else None
        return torch.nn.functional.linear(x.float(), self.weight.float(), bias).to(x.dtype)


class PhiBlock(AttentionBlock):
    def __init__(self, cfg):
        super().__init__()
        hidden = cfg['hidden_size']
        self.heads, self.kv_heads = cfg['num_attention_heads'], cfg['num_key_value_heads']
        self.head_dim = hidden // self.heads
        self.attn_norm = PhiLayerNorm(hidden, eps=cfg['layer_norm_eps'])
        self.qkv_split = [self.heads * self.head_dim, self.kv_heads * self.head_dim, self.kv_heads * self.head_dim]
        self.qkv_proj = PhiLinear(hidden, sum(self.qkv_split))
        self.o_proj = PhiLinear(hidden, hidden)
        self.up_proj = PhiLinear(hidden, cfg['intermediate_size'])
        self.down_proj = PhiLinear(cfg['intermediate_size'], hidden)
        self.q_norm = PhiLayerNorm(self.head_dim, eps=cfg['layer_norm_eps']) if cfg.get('qk_layernorm') else torch.nn.Identity()
        self.k_norm = PhiLayerNorm(self.head_dim, eps=cfg['layer_norm_eps']) if cfg.get('qk_layernorm') else torch.nn.Identity()

    def project(self, x, cos, sin):
        q, k, v = self.qkv_proj(self.attn_norm(x)).split(self.qkv_split, dim=-1)
        q = self.q_norm(q.view(1, x.shape[1], self.heads, self.head_dim))
        k = self.k_norm(k.view(1, x.shape[1], self.kv_heads, self.head_dim))
        # Rotate before transpose so Core AI cache updates do not lose rows (F-P2-13).
        q = rotate_prefix(q, cos, sin).transpose(1, 2)
        k = rotate_prefix(k, cos, sin).transpose(1, 2)
        v = v.view(1, x.shape[1], self.kv_heads, self.head_dim).transpose(1, 2)
        return q, k.to(self.qkv_proj.weight.dtype), v.to(self.qkv_proj.weight.dtype)

    def finish(self, x, attn):
        attention = self.o_proj(attn.transpose(1, 2).reshape(1, x.shape[1], -1))
        # Both branches read the same normalized input, before either residual add.
        h = self.up_proj(self.attn_norm(x))
        h = 0.5 * h * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (h + 0.044715 * h.pow(3))))
        return attention + self.down_proj(h) + x


class PhiDecoder(DenseDecoder):
    arithmetic_policy = {'residuals': 'float32', 'projections': 'float32', 'normalization': 'float32',
                         'weights': 'float16', 'cache': 'float16', 'logits': 'float16',
                         'rotaryFrequencyInitialization': 'float32'}

    def __init__(self, cfg, contract, max_capacity=2048, attention_precision='float32'):
        if attention_precision != 'float32':
            raise NotImplementedError('Phi requires float32 attention with its float32 residual path')
        if cfg.get('model_type') != 'phi' or cfg.get('hidden_act', 'gelu_new') != 'gelu_new':
            raise NotImplementedError('expected Phi with gelu_new')
        hidden, heads = cfg['hidden_size'], cfg['num_attention_heads']
        kv_heads = cfg.get('num_key_value_heads', heads)
        if hidden <= 0 or heads <= 0 or hidden % heads or kv_heads <= 0 or heads % kv_heads:
            raise ValueError('invalid Phi attention dimensions')
        head_dim = hidden // heads
        if cfg.get('head_dim', head_dim) != head_dim:
            raise NotImplementedError('Phi custom head_dim is not supported')
        factor = cfg.get('partial_rotary_factor', 0.5)
        dim = int(head_dim * factor)
        if not 0 < factor <= 1 or dim < 2 or dim % 2:
            raise ValueError('Phi rotary prefix must be positive, even, and fit the head')
        if not 2 <= max_capacity <= cfg.get('max_position_embeddings', 2048):
            raise ValueError('Phi cache capacity must fit the trained context')
        cfg = {**cfg, 'num_key_value_heads': kv_heads, 'head_dim': head_dim}
        super().__init__(cfg, contract, max_capacity, block_factory=PhiBlock,
                         norm_factory=lambda: PhiLayerNorm(hidden, eps=cfg['layer_norm_eps']),
                         rotary_dim=dim, head_bias=True, attention_precision=attention_precision)

        # Match upstream's float32 frequency initialization exactly; late Phi-2
        # attention layers amplify even one-ulp frequency differences.
        self.rope.inv_freq = 1.0 / (cfg['rope_theta'] ** (torch.arange(0, dim, 2).float() / dim))

    def embed_inputs(self, input_ids, positions):
        return self.embed(input_ids).float()

    def head(self, x):
        weight = self.embed.weight if self.tied else self.lm_head.weight
        return torch.nn.functional.linear(self.norm(x), weight.float(), self.head_bias.float()).to(self.embed.weight.dtype)


def name_map(module):
    names = {'model.embed_tokens.weight': ('embed.weight', 0), 'lm_head.bias': ('head_bias', 0)}
    for suffix in ('weight', 'bias'):
        names[f'model.final_layernorm.{suffix}'] = (f'norm.{suffix}', 0)
    if not module.tied:
        names['lm_head.weight'] = ('lm_head.weight', 0)
    for i, block in enumerate(module.blocks):
        src, dst = f'model.layers.{i}.', f'blocks.{i}.'
        pairs = [('input_layernorm', 'attn_norm'), ('self_attn.dense', 'o_proj'),
                 ('mlp.fc1', 'up_proj'), ('mlp.fc2', 'down_proj')]
        if module.cfg.get('qk_layernorm'):
            pairs += [('self_attn.q_layernorm', 'q_norm'), ('self_attn.k_layernorm', 'k_norm')]
        for a, b in pairs:
            for suffix in ('weight', 'bias'):
                names[f'{src}{a}.{suffix}'] = (f'{dst}{b}.{suffix}', 0)
        row = 0
        for proj, width in zip(('q_proj', 'k_proj', 'v_proj'), block.qkv_split):
            for suffix in ('weight', 'bias'):
                names[f'{src}self_attn.{proj}.{suffix}'] = (f'{dst}qkv_proj.{suffix}', row)
            row += width
    return names


def load_weights(module, checkpoint: Path):
    load_mapped_weights(module, checkpoint, name_map(module))

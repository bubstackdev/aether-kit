"""GPT-2 completion decoder: learned positions, LayerNorm, biased Conv1D projections.

Reference: Transformers 4.57.3 models/gpt2/modeling_gpt2.py (Apache-2.0).
Shares attention, cache contracts and export plumbing with the rotary families.
"""
import math
from pathlib import Path
import torch
from dense_decode import AttentionBlock, DenseDecoder, TensorMapping, load_weights as load_mapped_weights


class GPT2Block(AttentionBlock):
    def __init__(self, cfg):
        super().__init__()
        hidden = cfg['hidden_size']
        self.heads = self.kv_heads = cfg['num_attention_heads']
        self.head_dim = hidden // self.heads
        eps = cfg['layer_norm_epsilon']
        self.attn_norm = torch.nn.LayerNorm(hidden, eps=eps)
        self.mlp_norm = torch.nn.LayerNorm(hidden, eps=eps)
        self.qkv_proj = torch.nn.Linear(hidden, 3 * hidden)
        self.o_proj = torch.nn.Linear(hidden, hidden)
        self.up_proj = torch.nn.Linear(hidden, cfg['intermediate_size'])
        self.down_proj = torch.nn.Linear(cfg['intermediate_size'], hidden)

    def project(self, x, cos, sin):
        qkv = self.qkv_proj(self.attn_norm(x)).view(1, x.shape[1], 3, self.heads, self.head_dim)
        return tuple(qkv[:, :, i].transpose(1, 2) for i in range(3))

    def finish(self, x, attn):
        x = x + self.o_proj(attn.transpose(1, 2).reshape(1, x.shape[1], -1))
        h = self.up_proj(self.mlp_norm(x))
        # Match Transformers' gelu_new operation order, including in fp16.
        h = 0.5 * h * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (h + 0.044715 * h.pow(3))))
        return x + self.down_proj(h)


def normalized_config(cfg):
    if cfg.get('model_type') != 'gpt2' or cfg.get('activation_function', 'gelu_new') != 'gelu_new':
        raise NotImplementedError('expected GPT-2 with gelu_new')
    for key in ('add_cross_attention', 'scale_attn_by_inverse_layer_idx', 'reorder_and_upcast_attn', 'pruned_heads'):
        if cfg.get(key):
            raise NotImplementedError(f'GPT-2 {key} is not supported')
    if not cfg.get('scale_attn_weights', True):
        raise NotImplementedError('GPT-2 requires scaled dot-product attention')
    hidden, heads = cfg['n_embd'], cfg['n_head']
    if hidden <= 0 or heads <= 0 or hidden % heads or cfg['n_positions'] < 2 or cfg['n_layer'] <= 0:
        raise ValueError('invalid GPT-2 dimensions')
    return {**cfg, 'hidden_size': hidden, 'num_attention_heads': heads, 'num_key_value_heads': heads,
            'head_dim': hidden // heads, 'num_hidden_layers': cfg['n_layer'],
            'intermediate_size': cfg.get('n_inner') or 4 * hidden, 'tie_word_embeddings': cfg.get('tie_word_embeddings', True)}


class GPT2Decoder(DenseDecoder):
    def __init__(self, cfg, contract, max_capacity=1024, attention_precision='float32'):
        cfg = normalized_config(cfg)
        if not 2 <= max_capacity <= cfg['n_positions']:
            raise ValueError('GPT-2 cache capacity must fit its learned position table')
        super().__init__(cfg, contract, max_capacity, block_factory=GPT2Block,
                         norm_factory=lambda: torch.nn.LayerNorm(cfg['hidden_size'], eps=cfg['layer_norm_epsilon']),
                         rotary=False, attention_precision=attention_precision)
        self.position_embed = torch.nn.Embedding(cfg['n_positions'], cfg['hidden_size'])

    def embed_inputs(self, input_ids, positions):
        return self.embed(input_ids) + self.position_embed(positions)[None]


def name_map(module):
    names = {'transformer.wte.weight': TensorMapping('embed.weight'),
             'transformer.wpe.weight': TensorMapping('position_embed.weight'),
             'transformer.ln_f.weight': TensorMapping('norm.weight'),
             'transformer.ln_f.bias': TensorMapping('norm.bias')}
    if not module.tied:
        names['lm_head.weight'] = TensorMapping('lm_head.weight')
    for i in range(len(module.blocks)):
        for src, dst, transpose in [('ln_1', 'attn_norm', False), ('ln_2', 'mlp_norm', False),
                                    ('attn.c_attn', 'qkv_proj', True), ('attn.c_proj', 'o_proj', True),
                                    ('mlp.c_fc', 'up_proj', True), ('mlp.c_proj', 'down_proj', True)]:
            for suffix in ('weight', 'bias'):
                names[f'transformer.h.{i}.{src}.{suffix}'] = TensorMapping(
                    f'blocks.{i}.{dst}.{suffix}', transpose=transpose and suffix == 'weight')
    # Original Hub GPT2Model weights have no transformer prefix; save_pretrained
    # GPT2LMHeadModel weights do. Both layouts have the same learned tensors.
    names.update({key.removeprefix('transformer.'): value for key, value in list(names.items())
                  if key.startswith('transformer.')})
    return names


def load_weights(module, checkpoint: Path):
    # Older GPT-2 snapshots persist fixed causal masks. Shared execution rebuilds
    # these from absolute positions; they are buffers, never learned parameters.
    ignored = {f'{prefix}h.{i}.attn.{name}' for prefix in ('', 'transformer.')
               for i in range(len(module.blocks)) for name in ('bias', 'masked_bias')}
    load_mapped_weights(module, checkpoint, name_map(module), ignored_names=ignored)

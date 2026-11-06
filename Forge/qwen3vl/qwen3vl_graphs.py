"""The Qwen3-VL 2B export graphs and the shared harness for its exporter, reference and checks.

Qwen3-VL-2B-Instruct (Qwen/Qwen3-VL-2B-Instruct, Apache-2.0, Alibaba Qwen) is a ViT vision tower (24 blocks, hidden
1024, patch 16, temporal patch 2, 2x2 spatial merge to the decoder's width 2048, DeepStack features from blocks 5, 11
and 17) in front of a Qwen3 decoder (28 layers, hidden 2048, 16 query / 8 KV heads of 128, tied head, interleaved
3-axis M-RoPE). The bundle is `vlm-static-inject@1` (MANIFEST.md §1.2): two graphs, one per role.

**`vision` (graph kind `vision-tower@1`)**, one static function, `VisionTower`:

* `patches` [P, 1536] (P = PATCH_CAPACITY rows): the processor's block-major patch rows (each 2x2 merge block's four
  patches contiguous, each row C,T,py,px with the frame repeated T=2 times), the first h*w rows real, the rest zero;
* `grid` int32 [2] = (h, w), the image's patch grid (both even, h*w <= P);
* outputs `image_embeds` [P/4, 2048] (merged tokens in raster order of the merged grid) and `deepstack_embeds`
  [3*P/4, 2048] (level-major: level k's rows at k*P/4 + slot). Rows past h*w/4 are zero.

Everything upstream derives from `grid_thw` is computed in the graph from `grid`, with upstream's own closed forms
(transformers.vision_utils): the block-major (row, col) of each patch, the bilinear resampling (align_corners) of the
learned 48x48 position table, the 2-D rotary angles. The padding rows are masked out of attention as keys, so every
real row computes exactly what upstream's per-image attention does. One static shape serves every grid up to the
capacity: a dynamic-shape tower computed wrong outputs intermittently on the GPU in the 0.0.x prototype
(docs/qwen3-vl.md there), and per-grid functions each wire their own resources.

**`decoder` (the `decoder-lm@1` contract plus static image inputs)**, one function, `Decoder`:

* `input_ids` int32 [1, S] (S dynamic, 1..PREFILL_MAX): token ids; the prompt's `<|image_pad|>` ids are rewritten by
  the host to extension ids `vocab + slot` (slot 0..count-1 in order);
* `position_ids` int32 [1, T] (T dynamic): the running positions 0..T-1, the last S of them the fed tokens';
* static inputs `image_embeds` [N, 2048], `deepstack_embeds` [3N, 2048] (the tower's outputs as is, N = P/4),
  `image_rc` int32 [N, 2] (merged-grid (row, col) of each slot), `rope_shift_start`, `rope_shift_amount` int32 [1];
* states `keyCache`, `valueCache` [28, 1, 8, C, 128] (C dynamic, the sequence axis 3), written in place at rows
  T-S..T-1;
* output `logits` [1, 1, 151936]: the last fed row only (`rows: last`).

In the graph, per token: the embedding is the vocabulary row, or `image_embeds[slot]` for an extension id; DeepStack
adds `deepstack_embeds[k*N + slot]` after decoder layer k < 3 at image tokens; the M-RoPE positions are synthesized
from (id, position) alone: an image token at position p has slot s, image start s0 = p - s and (t, h, w) = (s0,
s0 + row, s0 + col); a text token's three positions are p - shift, shift = rope_shift_amount when p >=
rope_shift_start (else 0). For one image of N tokens with merged grid (gh, gw) starting at position s0, the host binds
start = s0 + N and amount = N - max(gh, gw): exactly upstream's `get_rope_index` (text after the image resumes at
s0 + max(gh, gw)). Text-only: start = 1 << 30, amount 0, zero buffers; the graph is then a plain Qwen3 decoder.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "source-pins.json").read_text())
REPO, REVISION = PINS["repository"], PINS["revision"]
DEFAULT_SNAPSHOT = Path.home() / f".cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/{REVISION}"
DTYPES = {"float32": torch.float32, "float16": torch.float16}

PATCH_CAPACITY = 2048          # tower rows: 512 image tokens, 524,288 pixels (e.g. 1024x512, 736x704)
PREFILL_MAX = 512              # the decoder's dynamic token axis, 1..512 (DecoderEngine's default prefill chunk)
CONTEXT_MAX = 32768            # the KV capacity axis' upper bound in the graph (limits.maxContext)
DEEPSTACK_LEVELS = 3
TEXT_ONLY_SHIFT_START = 1 << 30

# Token ids (tokenizer_config.json added_tokens_decoder; config.json image_token_id / vision_*_token_id).
IMAGE_PAD_ID, VISION_START_ID, VISION_END_ID = 151655, 151652, 151653


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sources(snapshot: Path) -> dict:
    """Checks every pinned snapshot file; returns {file: sha256}."""
    verified = {}
    for relative, expected in PINS["files"].items():
        path = snapshot / relative
        if not path.is_file():
            raise SystemExit(f"snapshot lacks {relative}; run Forge/qwen3vl/fetch_upstream.py ({REPO}@{REVISION[:12]})")
        if sha256_file(path) != expected:
            raise SystemExit(f"{relative} differs from the pinned {REVISION[:12]} file")
        verified[relative] = expected
    return verified


def config(snapshot: Path) -> dict:
    return json.loads((snapshot / "config.json").read_text())


# MARK: upstream (transformers' unmodified Qwen3-VL, the CPU float32 reference)

def upstream_model(snapshot: Path, dtype: torch.dtype = torch.float32):
    from transformers import Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(str(snapshot), dtype=dtype, attn_implementation="sdpa")
    return model.eval()


def processor(snapshot: Path):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(str(snapshot))


# MARK: weights

class Store(nn.Module):
    """A linear layer's weight (and bias) stored in `storage`, cast to the compute dtype in the graph."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, storage: torch.dtype) -> None:
        super().__init__()
        self.register_buffer("weight", checked(weight, storage))
        self.register_buffer("bias", None if bias is None else checked(bias, storage))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


def checked(w: torch.Tensor, storage: torch.dtype) -> torch.Tensor:
    narrow = w.detach().to(storage).contiguous()
    if not torch.isfinite(narrow).all():
        raise SystemExit("a weight overflows its storage dtype")
    return narrow


def linear(module: nn.Linear, storage: torch.dtype) -> Store:
    return Store(module.weight, module.bias, storage)


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """LayerNorm with float32 statistics, in the input's dtype after."""
    x32 = x.float()
    mean = x32.mean(-1, keepdim=True)
    centered = x32 - mean
    var = (centered * centered).mean(-1, keepdim=True)
    return (centered * torch.rsqrt(var + eps) * weight.float() + bias.float()).to(x.dtype)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3VLTextRMSNorm: float32 statistics, weight times the normalized value cast back to the input dtype."""
    x32 = x.float()
    normalized = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return weight.to(x.dtype) * normalized.to(x.dtype)


def floordiv(a, b):
    return torch.div(a, b, rounding_mode="floor")


def mod(a, b):
    """a mod b for non-negative ints (Core AI's converter has no aten.remainder)."""
    return a - floordiv(a, b) * b


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


# MARK: the vision tower

class LayerNormP(nn.Module):
    def __init__(self, norm: nn.LayerNorm) -> None:
        super().__init__()
        self.register_buffer("weight", norm.weight.detach().float().clone())
        self.register_buffer("bias", norm.bias.detach().float().clone())
        self.eps = float(norm.eps)

    def forward(self, x):
        return layer_norm(x, self.weight, self.bias, self.eps)


class VisionBlock(nn.Module):
    def __init__(self, block, heads: int, storage: torch.dtype) -> None:
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = LayerNormP(block.norm1), LayerNormP(block.norm2)
        self.qkv, self.proj = linear(block.attn.qkv, storage), linear(block.attn.proj, storage)
        self.fc1, self.fc2 = linear(block.mlp.linear_fc1, storage), linear(block.mlp.linear_fc2, storage)

    def forward(self, x, cos, sin, keep):
        n = x.shape[0]
        qkv = self.qkv(self.norm1(x)).view(n, 3, self.heads, -1)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]                                  # [n, heads, hd]
        # Upstream rotates in float32 and casts back (apply_rotary_pos_emb_vision).
        q = (q.float() * cos + rotate_half(q.float()) * sin).to(x.dtype)
        k = (k.float() * cos + rotate_half(k.float()) * sin).to(x.dtype)
        att = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                             v.transpose(0, 1)[None], attn_mask=keep)
        x = x + self.proj(att[0].transpose(0, 1).reshape(n, -1))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x)), approximate="tanh"))


class Merger(nn.Module):
    """Qwen3VLVisionPatchMerger: 2x2 block -> one decoder-width row. `post`: LayerNorm over the concatenated 4096
    (DeepStack mergers); otherwise per patch before concatenation (the final merger). Exact GELU."""

    def __init__(self, merger, storage: torch.dtype) -> None:
        super().__init__()
        self.post = bool(merger.use_postshuffle_norm)
        self.width = merger.hidden_size
        self.norm = LayerNormP(merger.norm)
        self.fc1, self.fc2 = linear(merger.linear_fc1, storage), linear(merger.linear_fc2, storage)

    def forward(self, x):
        x = self.norm(x.reshape(-1, self.width)) if self.post else self.norm(x).reshape(-1, self.width)
        return self.fc2(F.gelu(self.fc1(x)))


class VisionTower(nn.Module):
    """patches [P, 1536], grid int32 [2] -> image_embeds [P/4, H], deepstack_embeds [3P/4, H] (see the module doc).

    `dtype` is the compute dtype of the blocks and mergers (norm statistics and the rotary always float32);
    `storage` the dtype linear weights are stored in."""

    def __init__(self, visual, capacity: int = PATCH_CAPACITY, dtype: torch.dtype = torch.float32,
                 storage: torch.dtype = torch.float16) -> None:
        super().__init__()
        vc = visual.config
        self.capacity, self.dtype = capacity, dtype
        self.merge = vc.spatial_merge_size
        if capacity % (self.merge * self.merge):
            raise SystemExit("the patch capacity must be a multiple of merge^2")
        self.side = visual.num_grid_per_side
        self.heads = vc.num_heads
        conv = visual.patch_embed.proj
        self.patch = Store(conv.weight.reshape(conv.weight.shape[0], -1), conv.bias, storage)
        self.register_buffer("pos_table", checked(visual.pos_embed.weight, storage))
        self.register_buffer("inv_freq", visual.rotary_pos_emb.inv_freq.detach().float().clone())
        self.blocks = nn.ModuleList(VisionBlock(b, self.heads, storage) for b in visual.blocks)
        self.deepstack_at = list(visual.deepstack_visual_indexes)
        self.deepstack = nn.ModuleList(Merger(m, storage) for m in visual.deepstack_merger_list)
        self.merger = Merger(visual.merger, storage)
        self.register_buffer("index", torch.arange(capacity, dtype=torch.int32))
        self.register_buffer("token_index", torch.arange(capacity // (self.merge * self.merge), dtype=torch.int32))

    def geometry(self, grid: torch.Tensor):
        """(row, col) int32 [P] of every patch row in block-major order, and the real-row mask [P]; upstream's
        `get_vision_interpolation_indices_and_weights` decode of the flat index."""
        m = self.merge
        h, w = grid[0], grid[1]
        i = self.index
        blocks_w = floordiv(w, m)
        in_col = mod(i, m)
        in_row = mod(floordiv(i, m), m)
        block_col = mod(floordiv(i, m * m), blocks_w)
        block_row = floordiv(i, m * m * blocks_w)
        return block_row * m + in_row, block_col * m + in_col, i < h * w

    def axis(self, index: torch.Tensor, size: torch.Tensor):
        """Bilinear (align_corners) taps and weights along one axis, as upstream's `_interpolation_axis_taps_weights`:
        src = index * (side - 1) / max(size - 1, 1)."""
        src = index.float() * float(self.side - 1) / torch.clamp(size - 1, min=1).float()
        floor = torch.floor(src)
        lo = floor.to(torch.int32)
        taps = torch.stack((lo, lo + 1), dim=-1).clamp(0, self.side - 1)
        frac = src - floor
        weights = torch.stack(((1 - frac.abs()).clamp(min=0), (1 - (frac - 1).abs()).clamp(min=0)), dim=-1)
        return taps, weights

    def forward(self, patches: torch.Tensor, grid: torch.Tensor):
        dt = self.dtype
        row, col, real = self.geometry(grid)
        # Rows past the image are clamped into range for the table lookup (their outputs are masked and zeroed).
        h_taps, h_w = self.axis(row, grid[0])
        w_taps, w_w = self.axis(col, grid[1])
        indices = (h_taps[:, :, None] * self.side + w_taps[:, None, :]).reshape(-1, 4)
        weights = (h_w[:, :, None] * w_w[:, None, :]).reshape(-1, 4)
        pos = (F.embedding(indices, self.pos_table).float() * weights[:, :, None]).sum(1)
        x = self.patch(patches.to(dt)) + pos.to(dt)
        angles = torch.cat((row.float()[:, None] * self.inv_freq, col.float()[:, None] * self.inv_freq), dim=-1)
        emb = torch.cat((angles, angles), dim=-1)[:, None, :]                       # [P, 1, head_dim]
        cos, sin = emb.cos(), emb.sin()
        keep = real[None, None, None, :].expand(1, 1, self.capacity, self.capacity)
        levels = []
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, keep)
            if i in self.deepstack_at:
                levels.append(self.deepstack[self.deepstack_at.index(i)](x))
        tokens = (self.token_index < floordiv(grid[0] * grid[1], self.merge * self.merge))[:, None]
        image = torch.where(tokens, self.merger(x).float(), 0.0)
        deep = torch.cat([torch.where(tokens, lv.float(), 0.0) for lv in levels], dim=0)
        return image, deep

    def example(self, grid=(28, 28)):
        p = torch.zeros(self.capacity, self.patch.weight.shape[1], dtype=torch.float32)
        return p, torch.tensor(grid, dtype=torch.int32)


# MARK: the decoder

def mrope_masks(head_dim: int, section: list[int]) -> torch.Tensor:
    """[3, head_dim] 0/1 masks: which rotary frequency takes the t, h or w position. Upstream's interleaved
    `recomposition_frequencies`: frequency j is h for j in range(1, 3*section[1], 3), w for j in range(2,
    3*section[2], 3), else t; tiled twice for the rotate-half layout."""
    half = head_dim // 2
    m = torch.zeros(3, half)
    m[1, 1:3 * section[1]:3] = 1.0
    m[2, 2:3 * section[2]:3] = 1.0
    m[0] = 1.0 - m[1] - m[2]
    return torch.cat([m, m], dim=-1)


def mrope_positions(ids, p, image_rc, rope_shift_start, rope_shift_amount, vocab: int, slots: int):
    """The graph's M-RoPE positions (t, h, w), each int32 [S], from the fed ids (extension ids for image tokens), their
    running positions p and the static inputs (see the module doc)."""
    is_img = ids >= vocab
    slot = torch.clamp(ids - vocab, 0, slots - 1)
    rc = F.embedding(slot, image_rc)                                                 # [S, 2] int32
    shifted = p - torch.where(p >= rope_shift_start, rope_shift_amount, torch.zeros_like(p))
    s0 = p - slot
    return (torch.where(is_img, s0, shifted), torch.where(is_img, s0 + rc[:, 0], shifted),
            torch.where(is_img, s0 + rc[:, 1], shifted))


class DecoderLayer(nn.Module):
    def __init__(self, layer, cfg, storage: torch.dtype, attention: torch.dtype) -> None:
        super().__init__()
        a = layer.self_attn
        self.heads, self.kv_heads, self.head_dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.eps = cfg.rms_norm_eps
        self.attention = attention
        q, k, v = a.q_proj.weight, a.k_proj.weight, a.v_proj.weight
        self.qkv = Store(torch.cat([q, k, v], dim=0), None, storage)
        self.o_proj = linear(a.o_proj, storage)
        self.register_buffer("q_norm", a.q_norm.weight.detach().float().clone())
        self.register_buffer("k_norm", a.k_norm.weight.detach().float().clone())
        self.register_buffer("in_norm", layer.input_layernorm.weight.detach().float().clone())
        self.register_buffer("post_norm", layer.post_attention_layernorm.weight.detach().float().clone())
        mlp = layer.mlp
        self.gate_up = Store(torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], dim=0), None, storage)
        self.down = linear(mlp.down_proj, storage)
        self.inter = mlp.gate_proj.weight.shape[0]

    def forward(self, x, cos, sin, key_cache, value_cache, layer: int, start, end, mask):
        s = x.shape[1]
        h = rms_norm(x, self.in_norm, self.eps)
        qkv = self.qkv(h).view(1, s, self.heads + 2 * self.kv_heads, self.head_dim)
        q = rms_norm(qkv[:, :, :self.heads], self.q_norm, self.eps).transpose(1, 2)
        k = rms_norm(qkv[:, :, self.heads:self.heads + self.kv_heads], self.k_norm, self.eps).transpose(1, 2)
        v = qkv[:, :, self.heads + self.kv_heads:].transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        # In-place cache write at shape-derived bounds (a slice, not an integer, layer index), then attention over the
        # first `end` rows.
        key_cache[layer:layer + 1, :, :, start:end, :] = k.to(key_cache.dtype)[None]
        value_cache[layer:layer + 1, :, :, start:end, :] = v.to(value_cache.dtype)[None]
        keys = key_cache[layer, :, :, :end, :]
        values = value_cache[layer, :, :, :end, :]
        at = self.attention
        att = F.scaled_dot_product_attention(q.to(at), keys.to(at), values.to(at), attn_mask=mask,
                                             enable_gqa=True).to(x.dtype)
        x = x + self.o_proj(att.transpose(1, 2).reshape(1, s, self.heads * self.head_dim))
        h = rms_norm(x, self.post_norm, self.eps)
        gu = self.gate_up(h)
        return x + self.down(F.silu(gu[..., :self.inter]) * gu[..., self.inter:])


class Decoder(nn.Module):
    """The Qwen3-VL text decoder on the `decoder-lm@1` contract with static image inputs (see the module doc).

    `dtype`: the residual stream and matmuls; `attention`: the SDPA arithmetic; `storage`: the weights' and KV
    states' dtype. Norm statistics and rotary angles are float32 always; logits are float32."""

    def __init__(self, model, image_tokens: int = PATCH_CAPACITY // 4, dtype: torch.dtype = torch.float16,
                 attention: torch.dtype = torch.float32, storage: torch.dtype = torch.float16) -> None:
        super().__init__()
        cfg = model.config.text_config
        self.cfg, self.dtype, self.storage = cfg, dtype, storage
        self.vocab, self.hidden, self.image_tokens = cfg.vocab_size, cfg.hidden_size, image_tokens
        lm = model.model.language_model
        if not model.config.tie_word_embeddings:
            raise SystemExit("the export assumes a tied head (Qwen3-VL 2B ties it)")
        self.register_buffer("embed", checked(lm.embed_tokens.weight, storage))
        self.layers = nn.ModuleList(DecoderLayer(layer, cfg, storage, attention) for layer in lm.layers)
        self.register_buffer("norm", lm.norm.weight.detach().float().clone())
        rope = lm.rotary_emb
        self.register_buffer("inv_freq", rope.inv_freq.detach().float().clone())
        self.register_buffer("masks", mrope_masks(cfg.head_dim, list(rope.mrope_section)))
        self.eps = cfg.rms_norm_eps

    def rope(self, pt, ph, pw):
        """cos, sin [1, 1, S, head_dim] float32 from the three position rows [S] (int32)."""
        f = self.inv_freq
        angles = torch.cat((pt.float()[:, None] * f, pt.float()[:, None] * f), dim=-1) * self.masks[0]
        angles = angles + torch.cat((ph.float()[:, None] * f, ph.float()[:, None] * f), dim=-1) * self.masks[1]
        angles = angles + torch.cat((pw.float()[:, None] * f, pw.float()[:, None] * f), dim=-1) * self.masks[2]
        return angles.cos()[None, None], angles.sin()[None, None]

    def forward(self, input_ids, position_ids, image_embeds, deepstack_embeds, image_rc, rope_shift_start,
                rope_shift_amount, key_cache, value_cache):
        s = input_ids.shape[1]
        total = position_ids.shape[1]
        positions = position_ids[0]
        p = positions[total - s:]                                                    # [S] the fed tokens' positions
        ids = input_ids[0]
        v, n = self.vocab, self.image_tokens
        is_img = ids >= v
        slot = torch.clamp(ids - v, 0, n - 1)
        text = F.embedding(torch.clamp(ids, 0, v - 1), self.embed).to(self.dtype)
        image = F.embedding(slot, image_embeds).to(self.dtype)
        x = torch.where(is_img[:, None], image, text)[None]                          # [1, S, H]
        pt, ph, pw = mrope_positions(ids, p, image_rc, rope_shift_start, rope_shift_amount, v, n)
        cos, sin = self.rope(pt, ph, pw)
        cos, sin = cos.to(self.dtype), sin.to(self.dtype)
        # Lower-right causal mask from the position data (not shapes): row i sees keys 0..total-s+i.
        mask = (positions[None, :] <= p[:, None])[None, None]
        img = is_img[None, :, None]
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, key_cache, value_cache, i, total - s, total, mask)
            if i < DEEPSTACK_LEVELS:
                extra = F.embedding(slot + i * n, deepstack_embeds).to(self.dtype)[None]
                x = x + torch.where(img, extra, torch.zeros_like(extra))
        hidden = rms_norm(x[:, -1:], self.norm, self.eps)
        return F.linear(hidden.float(), self.embed.float())

    def empty_state(self, capacity: int) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.cfg
        shape = (c.num_hidden_layers, 1, c.num_key_value_heads, capacity, c.head_dim)
        return torch.zeros(shape, dtype=self.storage), torch.zeros(shape, dtype=self.storage)

    def statics(self, image_dtype: torch.dtype = torch.float32):
        """Text-only static inputs (zero buffers, the shift sentinel)."""
        n, h = self.image_tokens, self.hidden
        return (torch.zeros(n, h, dtype=image_dtype), torch.zeros(DEEPSTACK_LEVELS * n, h, dtype=image_dtype),
                torch.zeros(n, 2, dtype=torch.int32), torch.tensor([TEXT_ONLY_SHIFT_START], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32))

    def example(self):
        """(example inputs, dynamic shapes) for torch.export; the example dims all differ so export keeps them apart."""
        s, total, capacity = 8, 24, 56
        ids = torch.arange(s, dtype=torch.int32)[None] + 100
        ids[0, 2:5] = self.vocab + torch.arange(3, dtype=torch.int32)
        kc, vc = self.empty_state(capacity)
        example = (ids, torch.arange(total, dtype=torch.int32)[None], *self.statics(), kc, vc)
        s_dim = torch.export.Dim("tokens", min=1, max=PREFILL_MAX)
        t_dim = torch.export.Dim("total", min=2, max=CONTEXT_MAX)
        c_dim = torch.export.Dim("capacity", min=2, max=CONTEXT_MAX)
        dynamic = ({1: s_dim}, {1: t_dim}, None, None, None, None, None, {3: c_dim}, {3: c_dim})
        return example, dynamic


# MARK: host-side contract (what the Swift driver reproduces)

def image_rc(grid_h: int, grid_w: int, merge: int = 2, slots: int = PATCH_CAPACITY // 4) -> np.ndarray:
    """int32 [slots, 2]: (row, col) of each image token in the merged grid, raster order; unused slots zero."""
    gh, gw = grid_h // merge, grid_w // merge
    out = np.zeros((slots, 2), dtype=np.int32)
    k = np.arange(gh * gw)
    out[:gh * gw, 0], out[:gh * gw, 1] = k // gw, k % gw
    return out


def rope_shift(image_start: int, grid_h: int, grid_w: int, merge: int = 2) -> tuple[int, int]:
    """(rope_shift_start, rope_shift_amount) for one image whose first pad token is at `image_start`."""
    gh, gw = grid_h // merge, grid_w // merge
    count = gh * gw
    return image_start + count, count - max(gh, gw)


def extension_ids(ids: np.ndarray, vocab: int) -> np.ndarray:
    """The prompt with its `<|image_pad|>` run rewritten to vocab + slot."""
    out = ids.astype(np.int32).copy()
    where = np.nonzero(out == IMAGE_PAD_ID)[0]
    out[where] = vocab + np.arange(len(where), dtype=np.int32)
    return out


def pad_patches(pixel_values: np.ndarray, capacity: int = PATCH_CAPACITY) -> np.ndarray:
    rows = pixel_values.shape[0]
    if rows > capacity:
        raise SystemExit(f"{rows} patches exceed the tower capacity {capacity}")
    out = np.zeros((capacity, pixel_values.shape[1]), dtype=np.float32)
    out[:rows] = pixel_values
    return out


def smart_resize(height: int, width: int, factor: int = 32, min_pixels: int = 65536,
                 max_pixels: int = PATCH_CAPACITY * 256) -> tuple[int, int]:
    """Upstream's `smart_resize` (transformers qwen2_vl image processing) with the bundle's pixel budget: sides rounded
    to multiples of `factor`, the area kept within [min_pixels, max_pixels], aspect ratio preserved."""
    if max(height, width) / min(height, width) > 200:
        raise SystemExit("aspect ratio over 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


# MARK: metrics and I/O

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def row_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float64), b.astype(np.float64)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30)


def rel_l2(got: np.ndarray, want: np.ndarray) -> float:
    g, w = got.astype(np.float64).ravel(), want.astype(np.float64).ravel()
    return float(np.linalg.norm(g - w) / (np.linalg.norm(w) + 1e-30))


def write_f32(path: Path, x) -> None:
    path.write_bytes(np.ascontiguousarray(np.asarray(x), dtype="<f4").tobytes())


def read_f32(path: Path, shape) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype="<f4").reshape(shape).copy()


def write_i32(path: Path, x) -> None:
    path.write_bytes(np.ascontiguousarray(np.asarray(x), dtype="<i4").tobytes())


def read_i32(path: Path, shape) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype="<i4").reshape(shape).copy()

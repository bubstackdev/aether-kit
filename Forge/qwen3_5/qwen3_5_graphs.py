"""The Qwen3.5 / Qwen3.6 export graphs and the shared harness for its exporter, reference and checks.

Qwen3.5 and Qwen3.6 (Qwen3_5ForConditionalGeneration, Apache-2.0, Alibaba Qwen; 0.8B to 27B dense and their
fine-tunes) are a ViT vision tower (Qwen3-VL's: patch 16, temporal patch 2, 2x2 merge, bilinear-resampled learned
position table, axial 2-D rotary; no DeepStack) in front of a hybrid decoder: three Gated DeltaNet (linear attention)
layers to every gated full-attention layer, untied head, interleaved 3-axis M-RoPE over a partial rotary (a quarter of
the head). Every size is read from the checkpoint's config; nothing here names one. The bundle is
`vlm-static-inject@1` (MANIFEST.md §1.2), its decoder carrying the hybrid's recurrent states beside the KV.

**`vision` (graph kind `vision-tower@1`)**, one static function, `VisionTower`: `patches` [P, 1536] (the processor's
block-major patch rows, the first h*w real, zero after) and `grid` int32 [2] = (h, w) -> `image_embeds` [P/4, H], rows
past h*w/4 zero. Qwen3-VL's tower (Forge/qwen3vl) without the DeepStack levels.

**`decoder`**, `Decoder`, one function (the `decoder-lm@1` running contract, as Forge/qwen3vl's):

* `input_ids` int32 [1, S], S dynamic, 1..CALL_TOKENS: token ids, image pads rewritten by the host to extension ids
  `vocab + slot` (Qwen3-VL's packing: image k's tokens take base_k .. base_k + count_k - 1);
* `position_ids` int32 [1, T] (T dynamic): the running positions 0..T-1, the last S of them the fed tokens';
* static inputs (N = slots, M = most images): `image_embeds` [N, H], `image_positions` int32 [N, 3],
  `rope_shift_start`, `rope_shift_amount` int32 [M] (the `qwen3vl-mrope` rule, see Forge/qwen3vl/qwen3vl_graphs.py);
* states: `keyCache`, `valueCache` [full layers, 1, kv heads, C, head dim] (C dynamic, written in place at rows
  T-S..T-1, the `sequence` states), `convState` [linear layers, 1, conv dim, kernel - 1] and `recState` [linear
  layers, 1, value heads, key dim, value dim] float32 (the `recurrent` states, replaced each call);
* output `logits` [1, 1, vocab] float32, the last fed row.

One function, not a one-token function beside a prefill function: Core AI keeps a copy of the weights for every
function that runs (measured on the 27B, 2026-11-10: 16.6 GB of Metal after the first one-token call, 33.0 GB once a
64-token prefill function had also run). The runtime feeds a prompt in calls of at most CALL_TOKENS (its
`reshapeSafePrefillWidth`: wider calls on a reshaping graph retain Metal, docs/VLM.md G5) and decodes one token a call.

The Gated DeltaNet scan runs in float32 over a fixed CALL_TOKENS + 1 rows, since a graph can't loop a dynamic number of
times: the S real tokens are written into zero buffers, whose padding rows are exactly inert (decay 0, so exp(0) = 1,
and beta 0 leave the state bit for bit), and the first S rows of the result are kept. The rows are scanned in pieces of
SCAN_PIECE in sequence, each by the chunked (WY) form with the triangular inverse taken as a fixed product of
doublings: exact in arithmetic, but its powers grow combinatorially, and at 64 tokens the float32 result is wrong (the
2026-11-10 spike: Qwen3.6-27B layer 0's state 2.1e6 against 32 after 64 tokens; 16 tokens is exact to 6e-7).

Two Core AI rules every state write here follows (both found the hard way on the 2026-11-10 spike): a write is a slice
assignment at a slice (never an integer) layer index, and nothing writes through a matmul with a one-row operand.
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
PINS: dict = {}
REPO = REVISION = ""
DEFAULT_SNAPSHOT = Path()

PATCH_CAPACITY = 2048          # tower rows: 512 image tokens, 524,288 pixels a picture
CALL_TOKENS = 32               # the most tokens one decoder call takes (the runtime's reshapeSafePrefillWidth)
SCAN_PIECE = 16                # the Gated DeltaNet block scan's piece (exact in float32; 64 is not)
CONTEXT_MAX = 32768            # the KV capacity axis' upper bound in the graph
TEXT_ONLY_SHIFT_START = 1 << 30
IMAGE_SLOTS = 4096             # the decoder's static image rows (every image token of a context, packed)
MAX_IMAGES = 16                # rope-shift entries (images per context)


def use_pins(path: Path) -> None:
    """Selects the source checkpoint: a pins file (repository, revision, SHA-256 of every snapshot file read, the
    bundle's id and names), as Forge/qwen3vl's."""
    global PINS, REPO, REVISION, DEFAULT_SNAPSHOT
    PINS = json.loads(Path(path).read_text())
    REPO, REVISION = PINS["repository"], PINS["revision"]
    DEFAULT_SNAPSHOT = (Path.home() / ".cache/huggingface/hub" / ("models--" + REPO.replace("/", "--"))
                        / "snapshots" / REVISION)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# MARK: arithmetic

def rms_norm_plus_one(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3_5RMSNorm: float32 statistics, times (1 + weight), in the input's dtype after."""
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + weight.float())).to(x.dtype)


def rms_norm_gated(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3_5RMSNormGated: normalize in float32, times weight in the input dtype, times silu(gate) in float32."""
    x32 = x.float()
    normalized = (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)
    return ((weight.to(x.dtype) * normalized).float() * F.silu(gate.float())).to(x.dtype)


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.float()
    centered = x32 - x32.mean(-1, keepdim=True)
    var = (centered * centered).mean(-1, keepdim=True)
    return (centered * torch.rsqrt(var + eps) * weight.float() + bias.float()).to(x.dtype)


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def floordiv(a, b):
    return torch.div(a, b, rounding_mode="floor")


def mod(a, b):
    """a mod b for non-negative ints (Core AI's converter has no aten.remainder)."""
    return a - floordiv(a, b) * b


def linear(module: nn.Linear, dtype: torch.dtype) -> nn.Linear:
    """A bias-carrying copy of an upstream Linear with its weights in `dtype` (an nn.Linear, so coreai-opt can quantize
    it)."""
    out = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
    out.weight = nn.Parameter(module.weight.detach().to(dtype).contiguous(), requires_grad=False)
    if module.bias is not None:
        out.bias = nn.Parameter(module.bias.detach().to(dtype).contiguous(), requires_grad=False)
    if not torch.isfinite(out.weight).all():
        raise SystemExit("a weight overflows its storage dtype")
    return out


def buffer(t: torch.Tensor) -> torch.Tensor:
    return t.detach().float().clone()


# MARK: the vision tower (Forge/qwen3vl's VisionTower, without DeepStack)

class LayerNormP(nn.Module):
    def __init__(self, norm: nn.LayerNorm) -> None:
        super().__init__()
        self.register_buffer("weight", buffer(norm.weight))
        self.register_buffer("bias", buffer(norm.bias))
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
        n, dt = x.shape[0], x.dtype
        qkv = F.linear(self.norm1(x), self.qkv.weight.to(dt), self.qkv.bias.to(dt)).view(n, 3, self.heads, -1)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = (q.float() * cos + rotate_half(q.float()) * sin).to(dt)
        k = (k.float() * cos + rotate_half(k.float()) * sin).to(dt)
        att = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                             v.transpose(0, 1)[None], attn_mask=keep)
        x = x + F.linear(att[0].transpose(0, 1).reshape(n, -1), self.proj.weight.to(dt), self.proj.bias.to(dt))
        h = F.gelu(F.linear(self.norm2(x), self.fc1.weight.to(dt), self.fc1.bias.to(dt)), approximate="tanh")
        return x + F.linear(h, self.fc2.weight.to(dt), self.fc2.bias.to(dt))


class Merger(nn.Module):
    """2x2 block -> one decoder-width row: LayerNorm per patch, concatenation, Linear, exact GELU, Linear."""

    def __init__(self, merger, storage: torch.dtype) -> None:
        super().__init__()
        self.width = merger.hidden_size
        self.norm = LayerNormP(merger.norm)
        self.fc1, self.fc2 = linear(merger.linear_fc1, storage), linear(merger.linear_fc2, storage)

    def forward(self, x):
        dt = x.dtype
        x = self.norm(x).reshape(-1, self.width)
        x = F.gelu(F.linear(x, self.fc1.weight.to(dt), self.fc1.bias.to(dt)))
        return F.linear(x, self.fc2.weight.to(dt), self.fc2.bias.to(dt))


class VisionTower(nn.Module):
    """patches [P, 1536], grid int32 [2] -> image_embeds [P/4, H] (see the module doc). `dtype` the blocks' compute
    (norm statistics and the rotary float32 always), `storage` the linear weights', `output` the output's."""

    def __init__(self, visual, capacity: int = PATCH_CAPACITY, dtype: torch.dtype = torch.float32,
                 storage: torch.dtype = torch.float16, output: torch.dtype = torch.float16) -> None:
        super().__init__()
        vc = visual.config
        if list(getattr(vc, "deepstack_visual_indexes", []) or []):
            raise SystemExit("this tower has DeepStack levels: export it with Forge/qwen3vl")
        self.capacity, self.dtype, self.output = capacity, dtype, output
        self.merge = vc.spatial_merge_size
        if capacity % (self.merge * self.merge):
            raise SystemExit("the patch capacity must be a multiple of merge^2")
        self.side = visual.num_grid_per_side
        self.heads = vc.num_heads
        conv = visual.patch_embed.proj
        self.patch = nn.Linear(conv.weight[0].numel(), conv.weight.shape[0])
        self.patch.weight = nn.Parameter(conv.weight.detach().reshape(conv.weight.shape[0], -1).to(storage),
                                         requires_grad=False)
        self.patch.bias = nn.Parameter(conv.bias.detach().to(storage), requires_grad=False)
        self.register_buffer("pos_table", visual.pos_embed.weight.detach().to(storage).clone())
        self.register_buffer("inv_freq", buffer(visual.rotary_pos_emb.inv_freq))
        self.blocks = nn.ModuleList(VisionBlock(b, self.heads, storage) for b in visual.blocks)
        self.merger = Merger(visual.merger, storage)
        self.register_buffer("index", torch.arange(capacity, dtype=torch.int32))
        self.register_buffer("token_index", torch.arange(capacity // (self.merge * self.merge), dtype=torch.int32))

    def geometry(self, grid: torch.Tensor):
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
        h_taps, h_w = self.axis(row, grid[0])
        w_taps, w_w = self.axis(col, grid[1])
        indices = (h_taps[:, :, None] * self.side + w_taps[:, None, :]).reshape(-1, 4)
        weights = (h_w[:, :, None] * w_w[:, None, :]).reshape(-1, 4)
        pos = (F.embedding(indices, self.pos_table).float() * weights[:, :, None]).sum(1)
        x = F.linear(patches.to(dt), self.patch.weight.to(dt), self.patch.bias.to(dt)) + pos.to(dt)
        angles = torch.cat((row.float()[:, None] * self.inv_freq, col.float()[:, None] * self.inv_freq), dim=-1)
        emb = torch.cat((angles, angles), dim=-1)[:, None, :]
        cos, sin = emb.cos(), emb.sin()
        keep = real[None, None, None, :].expand(1, 1, self.capacity, self.capacity)
        for block in self.blocks:
            x = block(x, cos, sin, keep)
        tokens = (self.token_index < floordiv(grid[0] * grid[1], self.merge * self.merge))[:, None]
        return torch.where(tokens, self.merger(x).float(), 0.0).to(self.output)

    def example(self, grid=(28, 28)):
        return (torch.zeros(self.capacity, self.patch.weight.shape[1], dtype=torch.float32),
                torch.tensor(grid, dtype=torch.int32))


# MARK: M-RoPE (the qwen3vl-mrope rule, as Forge/qwen3vl)

def mrope_axes(rotary_dim: int, section: list[int]) -> torch.Tensor:
    """[3, rotary_dim / 2] 0/1: which rotary frequency takes the t, h or w position. Upstream's interleaved
    recomposition: frequency j is h for j in range(1, 3*section[1], 3), w for j in range(2, 3*section[2], 3), else t."""
    m = torch.zeros(3, rotary_dim // 2)
    m[1, 1:3 * section[1]:3] = 1.0
    m[2, 2:3 * section[2]:3] = 1.0
    m[0] = 1.0 - m[1] - m[2]
    return m


def mrope_positions(ids, p, image_positions, rope_shift_start, rope_shift_amount, vocab: int, slots: int):
    """(t, h, w) int32 [S] each from the fed ids (extension ids for image tokens) and their running positions p."""
    is_img = ids >= vocab
    slot = torch.clamp(ids - vocab, 0, slots - 1)
    pos = F.embedding(slot, image_positions)
    hit = (p[:, None] >= rope_shift_start[None, :]).to(torch.int32)
    shifted = p - (hit * rope_shift_amount[None, :]).sum(-1).to(torch.int32)
    return (torch.where(is_img, pos[:, 0], shifted), torch.where(is_img, pos[:, 1], shifted),
            torch.where(is_img, pos[:, 2], shifted))


# MARK: the Gated DeltaNet scan (float32)

def delta_step(q, k, v, g, beta, state):
    """One token. q, k [1, Hv, 1, dk] (l2-normalized, q scaled); v [1, Hv, 1, dv]; g, beta [1, Hv, 1]; state
    [1, Hv, dk, dv]. Returns out [1, Hv, 1, dv], the new state. transformers' torch_recurrent_gated_delta_rule body."""
    state = state * g.exp()[..., None]                                            # [1,Hv,1,1] broadcast
    k_t = k[:, :, 0, :, None]                                                     # [1,Hv,dk,1]
    kv = (state * k_t).sum(-2)                                                    # [1,Hv,dv]
    delta = (v[:, :, 0] - kv) * beta                                              # beta [1,Hv,1]
    state = state + k_t * delta[:, :, None, :]
    out = (state * q[:, :, 0, :, None]).sum(-2)
    return out[:, :, None, :], state


def delta_block(q, k, v, g, beta, state):
    """n <= SCAN_PIECE tokens at once, the chunked (WY) form: equal to n delta_steps. Shapes as delta_step with n for 1.
    The unit-lower-triangular inverse is the product of (I + X^(2^j)), X = -A strictly lower: exact once 2^j >= n."""
    n = q.shape[2]
    lower = torch.tril(torch.ones(n, n))
    strict = torch.tril(torch.ones(n, n), -1)
    c = g @ lower.transpose(0, 1)                                                 # cumulative decay [1,Hv,n]
    diff = c[..., :, None] - c[..., None, :]
    decay = torch.where(lower.bool(), diff, torch.full_like(diff, -1e30)).exp()   # 0 above the diagonal
    a = beta[..., None] * (decay * strict) * (k @ k.transpose(-1, -2))
    r = beta[..., None] * (v - c.exp()[..., None] * (k @ state))
    eye = torch.eye(n).expand_as(a)
    x = -a
    inverse, power = eye + x, x
    for _ in range(max(0, math.ceil(math.log2(max(n, 2))) - 1)):
        power = power @ power
        inverse = inverse @ (eye + power)
    u = inverse @ r
    out = c.exp()[..., None] * (q @ state) + ((decay * lower) * (q @ k.transpose(-1, -2))) @ u
    w = (c[..., -1:] - c).exp()
    state = c[..., -1:, None].exp() * state + (k * w[..., None]).transpose(-1, -2) @ u
    return out, state


def delta_scan(q, k, v, g, beta, state, rows: int | None = None):
    """S tokens (S <= rows) through the recurrence: written into `rows`-row zero buffers (padding inert: g = beta = 0),
    scanned in SCAN_PIECE pieces, the first S output rows kept. rows=None scans exactly S rows (S a static size)."""
    s = q.shape[2]
    if rows is not None:
        def pad(x):
            buffer = x.new_zeros(*x.shape[:2], rows, *x.shape[3:])
            buffer[:, :, :s] = x
            return buffer
        q, k, v, g, beta = pad(q), pad(k), pad(v), pad(g), pad(beta)
        # The padding must be zero whatever the buffers hold: a converted graph may fill them in place (a garbage
        # decay or beta in a padding row would enter the state). torch.where also drops a NaN it doesn't select.
        real = torch.arange(rows) < s
        q, k, v = (torch.where(real[None, None, :, None], t, 0.0) for t in (q, k, v))
        g, beta = (torch.where(real[None, None, :], t, 0.0) for t in (g, beta))
    n = rows if rows is not None else s
    outs = []
    for i in range(0, n, SCAN_PIECE):
        out, state = delta_block(q[:, :, i:i + SCAN_PIECE], k[:, :, i:i + SCAN_PIECE], v[:, :, i:i + SCAN_PIECE],
                                 g[:, :, i:i + SCAN_PIECE], beta[:, :, i:i + SCAN_PIECE], state)
        outs.append(out)
    out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)
    return out[:, :, :s], state


# MARK: the decoder

class LinearAttention(nn.Module):
    """Qwen3_5GatedDeltaNet with explicit states: conv [1, conv dim, K-1] and rec [1, Hv, dk, dv] in, new ones out."""

    def __init__(self, la, storage: torch.dtype) -> None:
        super().__init__()
        self.hk, self.hv = la.num_k_heads, la.num_v_heads
        self.dk, self.dv = la.head_k_dim, la.head_v_dim
        self.key_dim, self.value_dim = la.key_dim, la.value_dim
        self.kernel = la.conv_kernel_size
        self.eps = la.layer_norm_epsilon
        self.in_proj_qkv, self.in_proj_z = linear(la.in_proj_qkv, storage), linear(la.in_proj_z, storage)
        self.in_proj_b, self.in_proj_a = linear(la.in_proj_b, storage), linear(la.in_proj_a, storage)
        self.out_proj = linear(la.out_proj, storage)
        self.register_buffer("conv_weight", la.conv1d.weight.detach()[:, 0, :].to(storage).clone())   # [conv dim, K]
        self.register_buffer("A", buffer(la.A_log).exp())
        self.register_buffer("dt_bias", buffer(la.dt_bias))
        self.register_buffer("norm_weight", buffer(la.norm.weight))

    def forward(self, x, conv_in, rec_in):
        b, s, _ = x.shape
        mixed = self.in_proj_qkv(x).transpose(1, 2)                                   # [1, conv dim, S]
        window = torch.cat([conv_in.to(mixed.dtype), mixed], dim=-1)                  # [1, conv dim, K-1+S]
        w = self.conv_weight.to(mixed.dtype)
        conv = window[..., 0:s] * w[:, 0:1]
        for j in range(1, self.kernel):                                               # depthwise causal conv
            conv = conv + window[..., j:j + s] * w[:, j:j + 1]
        conv = F.silu(conv).transpose(1, 2)                                           # [1, S, conv dim]
        new_conv = window[..., s:]                                                    # the last K-1 inputs
        q, k, v = torch.split(conv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(b, s, self.hk, self.dk)
        k = k.reshape(b, s, self.hk, self.dk)
        if self.hv != self.hk:                                                        # each key head serves r values
            # Repeated before the transpose, on the contiguous [S, heads, dim] layout (after it, a dynamic S makes the
            # reshape's contiguity something torch.export can't decide).
            r = self.hv // self.hk
            q = q.unsqueeze(3).expand(b, s, self.hk, r, self.dk).reshape(b, s, self.hv, self.dk)
            k = k.unsqueeze(3).expand(b, s, self.hk, r, self.dk).reshape(b, s, self.hv, self.dk)
        q, k = q.transpose(1, 2).float(), k.transpose(1, 2).float()
        v = v.reshape(b, s, self.hv, self.dv).transpose(1, 2).float()
        q = l2norm(q) * self.dk ** -0.5
        k = l2norm(k)
        beta = torch.sigmoid(self.in_proj_b(x).float()).transpose(1, 2)               # [1, Hv, S]
        g = (-self.A * F.softplus(self.in_proj_a(x).float() + self.dt_bias)).transpose(1, 2)
        # One row more than a call can hold: a call of exactly CALL_TOKENS would otherwise overwrite the whole buffer,
        # a special case torch.export can't prove (it refuses S == CALL_TOKENS). The extra row is a one-token piece.
        out, new_rec = delta_scan(q, k, v, g, beta, rec_in.float(), rows=CALL_TOKENS + 1)
        out = out.transpose(1, 2).reshape(-1, self.dv).to(x.dtype)                     # [S*Hv, dv]
        z = self.in_proj_z(x).reshape(-1, self.dv)
        out = rms_norm_gated(out, z, self.norm_weight, self.eps).reshape(b, s, self.value_dim)
        return self.out_proj(out), new_conv, new_rec


class FullAttention(nn.Module):
    """Qwen3_5Attention: gated query (q_proj gives each head's query and gate), plus-one q/k norms, partial rotary."""

    def __init__(self, at, cfg, storage: torch.dtype, attention: torch.dtype) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = cfg.num_attention_heads, cfg.num_key_value_heads, at.head_dim
        self.eps, self.attention = cfg.rms_norm_eps, attention
        self.q_proj, self.k_proj = linear(at.q_proj, storage), linear(at.k_proj, storage)
        self.v_proj, self.o_proj = linear(at.v_proj, storage), linear(at.o_proj, storage)
        self.register_buffer("q_norm", buffer(at.q_norm.weight))
        self.register_buffer("k_norm", buffer(at.k_norm.weight))

    def forward(self, x, cos, sin, key_cache, value_cache, layer: int, start, end, mask):
        s = x.shape[1]
        q, gate = self.q_proj(x).view(1, s, self.heads, 2 * self.head_dim).chunk(2, dim=-1)
        gate = gate.reshape(1, s, self.heads * self.head_dim)
        q = rms_norm_plus_one(q, self.q_norm, self.eps).transpose(1, 2)
        k = rms_norm_plus_one(self.k_proj(x).view(1, s, self.kv_heads, self.head_dim), self.k_norm,
                              self.eps).transpose(1, 2)
        v = self.v_proj(x).view(1, s, self.kv_heads, self.head_dim).transpose(1, 2)
        rd = cos.shape[-1]
        q = torch.cat([q[..., :rd] * cos + rotate_half(q[..., :rd]) * sin, q[..., rd:]], dim=-1)
        k = torch.cat([k[..., :rd] * cos + rotate_half(k[..., :rd]) * sin, k[..., rd:]], dim=-1)
        key_cache[layer:layer + 1, :, :, start:end, :] = k.to(key_cache.dtype)[None]
        value_cache[layer:layer + 1, :, :, start:end, :] = v.to(value_cache.dtype)[None]
        keys = key_cache[layer, :, :, :end, :]
        values = value_cache[layer, :, :, :end, :]
        at = self.attention
        att = F.scaled_dot_product_attention(q.to(at), keys.to(at), values.to(at), attn_mask=mask,
                                             enable_gqa=True).to(x.dtype)
        att = att.transpose(1, 2).reshape(1, s, self.heads * self.head_dim) * torch.sigmoid(gate)
        return self.o_proj(att)


class Mlp(nn.Module):
    def __init__(self, mlp, storage: torch.dtype) -> None:
        super().__init__()
        self.gate_proj, self.up_proj = linear(mlp.gate_proj, storage), linear(mlp.up_proj, storage)
        self.down_proj = linear(mlp.down_proj, storage)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Decoder(nn.Module):
    """The hybrid decoder on the `decoder-lm@1` contract with static image inputs (see the module doc).

    `dtype`: the residual stream and matmuls; `attention`: the SDPA arithmetic; `storage`: the weights' and the KV and
    conv states' dtype. Norm statistics, rotary angles, the Gated DeltaNet scan and its state are float32; logits are
    float32."""

    def __init__(self, model, slots: int = IMAGE_SLOTS, images: int = MAX_IMAGES, dtype: torch.dtype = torch.float16,
                 attention: torch.dtype = torch.float32, storage: torch.dtype = torch.float16,
                 before_layer=None, after_layer=None) -> None:
        """`before_layer(i, layer)` runs before upstream layer i is read (an exporter loads its weights there) and
        `after_layer(i, mixer, mlp)` after its modules are built (an exporter quantizes them and frees the source)."""
        super().__init__()
        cfg = model.config.text_config
        self.cfg, self.dtype, self.storage = cfg, dtype, storage
        self.vocab, self.hidden, self.slots, self.images = cfg.vocab_size, cfg.hidden_size, slots, images
        lm = model.model.language_model
        self.kinds = list(cfg.layer_types)[:len(lm.layers)]
        self.register_buffer("embed", lm.embed_tokens.weight.detach().to(storage).clone())
        self.register_buffer("embed_scale", None)
        mixers, mlps, in_norms, post_norms = [], [], [], []
        for i, (layer, kind) in enumerate(zip(lm.layers, self.kinds)):
            if before_layer is not None:
                before_layer(i, layer)
            mixers.append(LinearAttention(layer.linear_attn, storage) if kind == "linear_attention"
                          else FullAttention(layer.self_attn, cfg, storage, attention))
            mlps.append(Mlp(layer.mlp, storage))
            in_norms.append(buffer(layer.input_layernorm.weight))
            post_norms.append(buffer(layer.post_attention_layernorm.weight))
            if after_layer is not None:
                after_layer(i, mixers[-1], mlps[-1])
        self.mixers, self.mlps = nn.ModuleList(mixers), nn.ModuleList(mlps)
        self.register_buffer("in_norms", torch.stack(in_norms))
        self.register_buffer("post_norms", torch.stack(post_norms))
        self.register_buffer("norm", buffer(lm.norm.weight))
        self.head = linear(model.lm_head, storage)
        rope = lm.rotary_emb
        self.register_buffer("inv_freq", buffer(rope.inv_freq))
        self.register_buffer("axes", mrope_axes(2 * rope.inv_freq.shape[0], list(rope.mrope_section)))
        self.eps = cfg.rms_norm_eps
        self.full_layers = sum(k == "full_attention" for k in self.kinds)
        self.linear_layers = len(self.kinds) - self.full_layers

    @torch.no_grad()
    def quantize_embedding(self) -> None:
        """Symmetric int8 per vocabulary row; only gathered rows are dequantized at inference.

        Quantize in pieces to avoid a full float32 copy of the 27B's 2.5 GB table.
        Float32 scales also preserve zero and very small rows without fp16 underflow.
        """
        if self.embed.dtype == torch.int8:
            return
        packed = torch.empty_like(self.embed, dtype=torch.int8)
        scales = torch.empty(self.vocab, 1, dtype=torch.float32)
        for start in range(0, self.vocab, 1024):
            rows = self.embed[start:start + 1024].float()
            scale = rows.abs().amax(dim=1, keepdim=True) / 127
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            packed[start:start + 1024] = (rows / scale).round().clamp(-127, 127).to(torch.int8)
            scales[start:start + 1024] = scale
        self.embed, self.embed_scale = packed, scales

    def rope(self, pt, ph, pw):
        """cos, sin [1, 1, S, rotary dim] float32 from the three position rows [S] (int32)."""
        f, a = self.inv_freq, self.axes
        freqs = pt.float()[:, None] * f * a[0] + ph.float()[:, None] * f * a[1] + pw.float()[:, None] * f * a[2]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None], emb.sin()[None, None]

    def forward(self, input_ids, position_ids, image_embeds, image_positions, rope_shift_start, rope_shift_amount,
                key_cache, value_cache, conv_state, rec_state):
        s = input_ids.shape[1]
        total = position_ids.shape[1]
        positions = position_ids[0]
        p = positions[total - s:]
        ids = input_ids[0]
        v, n = self.vocab, self.slots
        is_img = ids >= v
        text_ids = torch.clamp(ids, 0, v - 1)
        text = F.embedding(text_ids, self.embed)
        if self.embed_scale is not None:
            text = text.float() * F.embedding(text_ids, self.embed_scale)
        text = text.to(self.dtype)
        image = F.embedding(torch.clamp(ids - v, 0, n - 1), image_embeds).to(self.dtype)
        x = torch.where(is_img[:, None], image, text)[None]
        pt, ph, pw = mrope_positions(ids, p, image_positions, rope_shift_start, rope_shift_amount, v, n)
        cos, sin = self.rope(pt, ph, pw)
        cos, sin = cos.to(self.dtype), sin.to(self.dtype)
        mask = (positions[None, :] <= p[:, None])[None, None]
        fi = li = 0
        for i, (kind, mixer, mlp) in enumerate(zip(self.kinds, self.mixers, self.mlps)):
            h = rms_norm_plus_one(x, self.in_norms[i], self.eps)
            if kind == "full_attention":
                x = x + mixer(h, cos, sin, key_cache, value_cache, fi, total - s, total, mask)
                fi += 1
            else:
                out, new_conv, new_rec = mixer(h, conv_state[li], rec_state[li])
                conv_state[li:li + 1] = new_conv.to(conv_state.dtype)[None]
                rec_state[li:li + 1] = new_rec.to(rec_state.dtype)[None]
                x = x + out
                li += 1
            x = x + mlp(rms_norm_plus_one(x, self.post_norms[i], self.eps))
        hidden = rms_norm_plus_one(x[:, -1:], self.norm, self.eps)
        return self.head(hidden).float()

    def empty_state(self, capacity: int):
        c = self.cfg
        kv = (self.full_layers, 1, c.num_key_value_heads, capacity, c.head_dim)
        conv_dim = 2 * c.linear_num_key_heads * c.linear_key_head_dim + c.linear_num_value_heads * c.linear_value_head_dim
        return (torch.zeros(kv, dtype=self.storage), torch.zeros(kv, dtype=self.storage),
                torch.zeros(self.linear_layers, 1, conv_dim, c.linear_conv_kernel_dim - 1, dtype=self.storage),
                torch.zeros(self.linear_layers, 1, c.linear_num_value_heads, c.linear_key_head_dim,
                            c.linear_value_head_dim, dtype=torch.float32))

    def statics(self, image_dtype: torch.dtype | None = None):
        n, h, m = self.slots, self.hidden, self.images
        return (torch.zeros(n, h, dtype=image_dtype or self.storage), torch.zeros(n, 3, dtype=torch.int32),
                torch.full((m,), TEXT_ONLY_SHIFT_START, dtype=torch.int32), torch.zeros(m, dtype=torch.int32))

    def example(self):
        """(example inputs, dynamic shapes) for torch.export: the token axis (1..CALL_TOKENS), the running positions and
        the KV capacity dynamic (example dims all differ so export keeps them apart)."""
        tokens, total, capacity = 7, 3 * CALL_TOKENS + 5, 4 * CALL_TOKENS + 9
        ids = torch.arange(tokens, dtype=torch.int32)[None] + 100
        ids[0, 2:5] = self.vocab + torch.arange(3, dtype=torch.int32)
        example = (ids, torch.arange(total, dtype=torch.int32)[None], *self.statics(), *self.empty_state(capacity))
        s_dim = torch.export.Dim("tokens", min=1, max=CALL_TOKENS)
        t_dim = torch.export.Dim("total", min=2, max=CONTEXT_MAX)
        c_dim = torch.export.Dim("capacity", min=2, max=CONTEXT_MAX)
        dynamic = ({1: s_dim}, {1: t_dim}, None, None, None, None, {3: c_dim}, {3: c_dim}, None, None)
        return example, dynamic


# MARK: host-side contract (what the Swift driver reproduces; Forge/qwen3vl's rule)

def plan_images(starts: list[int], grids: list[tuple[int, int]], merge: int = 2, slots: int = IMAGE_SLOTS,
                images: int = MAX_IMAGES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """image_positions int32 [slots, 3], rope_shift_start and rope_shift_amount int32 [images] for a prompt's images:
    image k's first pad at starts[k], patch grid grids[k] = (h, w)."""
    if len(starts) != len(grids) or len(grids) > images:
        raise SystemExit(f"{len(grids)} images for {images} rope-shift entries")
    positions = np.zeros((slots, 3), dtype=np.int32)
    start = np.full(images, TEXT_ONLY_SHIFT_START, dtype=np.int32)
    amount = np.zeros(images, dtype=np.int32)
    base = shift = 0
    for k, (s0, (h, w)) in enumerate(zip(starts, grids)):
        gh, gw = h // merge, w // merge
        count = gh * gw
        if base + count > slots:
            raise SystemExit(f"{base + count} image tokens exceed the {slots} slots")
        t = s0 - shift
        i = np.arange(count)
        positions[base:base + count] = np.stack([np.full(count, t), t + i // gw, t + i % gw], axis=1)
        start[k], amount[k] = s0 + count, count - max(gh, gw)
        shift += int(amount[k])
        base += count
    return positions, start, amount


def extension_ids(ids: np.ndarray, vocab: int, pad_id: int) -> np.ndarray:
    out = ids.astype(np.int32).copy()
    where = np.nonzero(out == pad_id)[0]
    out[where] = vocab + np.arange(len(where), dtype=np.int32)
    return out


def pad_patches(pixel_values: np.ndarray, capacity: int = PATCH_CAPACITY) -> np.ndarray:
    rows = pixel_values.shape[0]
    if rows > capacity:
        raise SystemExit(f"{rows} patches exceed the tower capacity {capacity}")
    out = np.zeros((capacity, pixel_values.shape[1]), dtype=np.float32)
    out[:rows] = pixel_values
    return out


# MARK: metrics

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def row_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float64), b.astype(np.float64)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30)


def rel_l2(got: np.ndarray, want: np.ndarray) -> float:
    g, w = got.astype(np.float64).ravel(), want.astype(np.float64).ravel()
    return float(np.linalg.norm(g - w) / (np.linalg.norm(w) + 1e-30))

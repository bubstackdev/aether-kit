"""Qwen3-VL vision tower for Aether Kit image input (development tooling only).

Adapted from the aether-engine forge (`forge/aether_forge/arch/qwen3_vl_vision.py`, facts F-P3-20/21/26), not
imported, so this directory stays self-contained. The tower is its own asset with one function:

    encode(pixels [n, 1536], pos_index [n, 4] i32, pos_weight [n, 4] f32, patch_rc [n, 2] i32)
      -> image_embeds [n/4, 2048], deepstack [3, n/4, 2048]

n is the patch count of one image (dynamic, a multiple of 4). Every per-patch input is in merge-window order: 2x2
windows in raster order, the patches inside a window in raster order, which is the order of the preprocessor's pixel
rows. The host computes the per-patch tables (`host_inputs`): the four neighbours of each patch in the learned 48x48
position table with their bilinear weights (corners aligned), and each patch's integer (row, col) for the 2-D rotary.

Precision: weights are stored fp16 and every matmul, activation and the residual stream compute in float32. The
residual stream reaches ~2,800 by the last block and the tanh-GELU cubic overflows fp16, so a pure-fp16 tower loses
its embeddings on Core AI's cpu_only path (engine F-P3-21: cosine 0.94). On the GPU the asset must be specialized
without `expectFrequentReshapes` (engine F-P3-26: with it the dynamic-shape encoder computes wrong outputs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PREFIX = "model.visual."


# MARK: host-side inputs (the math the Swift side reproduces)

def merge_window_order(h: int, w: int, merge: int = 2) -> np.ndarray:
    """(row, col) of every patch in merge-window order: [h·w, 2] int32."""
    if h % merge or w % merge:
        raise ValueError(f"grid {h}x{w} is not a multiple of the merge size {merge}")
    wr, wc, i, j = np.meshgrid(np.arange(h // merge), np.arange(w // merge), np.arange(merge), np.arange(merge),
                               indexing="ij")
    return np.stack([(wr * merge + i).ravel(), (wc * merge + j).ravel()], 1).astype(np.int32)


def bilinear_neighbours(h: int, w: int, table_side: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Each raster patch's 4 neighbour indices into the side x side table and their weights, sampling at
    y = linspace(0, side-1, h), x = linspace(0, side-1, w); neighbours ordered (y0x0, y0x1, y1x0, y1x1)."""
    def axis(n: int):
        pos = np.linspace(0.0, table_side - 1, n, dtype=np.float64) if n > 1 else np.zeros(1)
        lo = np.floor(pos).astype(np.int64)
        return lo, np.minimum(lo + 1, table_side - 1), pos - lo
    y0, y1, dy = axis(h)
    x0, x1, dx = axis(w)
    Y0, X0 = np.meshgrid(y0, x0, indexing="ij")
    Y1, X1 = np.meshgrid(y1, x1, indexing="ij")
    DY, DX = np.meshgrid(dy, dx, indexing="ij")
    index = np.stack([Y0 * table_side + X0, Y0 * table_side + X1, Y1 * table_side + X0, Y1 * table_side + X1], -1)
    weight = np.stack([(1 - DY) * (1 - DX), (1 - DY) * DX, DY * (1 - DX), DY * DX], -1)
    return index.astype(np.int32), weight.astype(np.float32)


def host_inputs(grid_t: int, grid_h: int, grid_w: int, merge: int = 2, table_side: int = 48) -> dict[str, np.ndarray]:
    """`pos_index`, `pos_weight`, `patch_rc` for one image in merge-window order, repeated per temporal group."""
    rc = merge_window_order(grid_h, grid_w, merge)
    index, weight = bilinear_neighbours(grid_h, grid_w, table_side)
    return {"pos_index": np.tile(index[rc[:, 0], rc[:, 1]], (grid_t, 1)),
            "pos_weight": np.tile(weight[rc[:, 0], rc[:, 1]], (grid_t, 1)),
            "patch_rc": np.tile(rc, (grid_t, 1))}


def patchify(rgb: np.ndarray, patch: int = 16, merge: int = 2, temporal: int = 2,
             mean: float = 0.5, std: float = 0.5) -> tuple[np.ndarray, tuple[int, int, int]]:
    """uint8 RGB [H, W, 3] with sides that are multiples of patch·merge (no resize) -> (pixels float32 [n, 3·T·p·p],
    grid (1, H/p, W/p)): (x/255 - mean)/std, the frame repeated `temporal` times, rows in merge-window order, each row
    laid out [C][T][py][px]. Exact against the reference fast processor for such sizes (engine F-P3-22)."""
    h, w, _ = rgb.shape
    unit = patch * merge
    if h % unit or w % unit:
        raise ValueError(f"{w}x{h} is not a multiple of {unit}")
    x = (rgb.astype(np.float32) / 255.0 - mean) / std
    gh, gw = h // patch, w // patch
    x = x.reshape(gh // merge, merge, patch, gw // merge, merge, patch, 3).transpose(0, 3, 1, 4, 6, 2, 5)
    x = np.repeat(x[:, :, :, :, :, None], temporal, axis=5)
    return x.reshape(gh * gw, -1).astype(np.float32), (1, gh, gw)


# MARK: the tower (fp16 storage, float32 compute)

def linear32(layer: torch.nn.Linear, x: torch.Tensor) -> torch.Tensor:
    bias = layer.bias.float() if layer.bias is not None else None
    return F.linear(x.float(), layer.weight.float(), bias)


class LayerNorm32(torch.nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.bias = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        centered = x - x.mean(-1, keepdim=True)
        var = (centered * centered).mean(-1, keepdim=True)
        return centered * torch.rsqrt(var + self.eps) * self.weight.float() + self.bias.float()


class VisionBlock(torch.nn.Module):
    def __init__(self, hidden: int, heads: int, inter: int, eps: float) -> None:
        super().__init__()
        self.heads, self.head_dim = heads, hidden // heads
        self.norm1 = LayerNorm32(hidden, eps)
        self.qkv = torch.nn.Linear(hidden, 3 * hidden, bias=True)
        self.proj = torch.nn.Linear(hidden, hidden, bias=True)
        self.norm2 = LayerNorm32(hidden, eps)
        self.fc1 = torch.nn.Linear(hidden, inter, bias=True)
        self.fc2 = torch.nn.Linear(inter, hidden, bias=True)

    @staticmethod
    def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return x * cos + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sin

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        qkv = linear32(self.qkv, self.norm1(x)).view(n, 3, self.heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]                                   # [n, heads, head_dim]
        q, k = self.rotate(q, cos, sin), self.rotate(k, cos, sin)
        att = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None])
        x = x + linear32(self.proj, att[0].transpose(0, 1).reshape(n, -1))
        return x + linear32(self.fc2, F.gelu(linear32(self.fc1, self.norm2(x)), approximate="tanh"))


class Merger(torch.nn.Module):
    """2x2 window -> one LLM-width row. `norm_first`: LayerNorm per patch (final merger); otherwise after
    concatenation (DeepStack mergers). Exact GELU."""

    def __init__(self, hidden: int, window: int, out: int, eps: float, norm_first: bool) -> None:
        super().__init__()
        self.window, self.norm_first = window, norm_first
        self.norm = LayerNorm32(hidden if norm_first else hidden * window, eps)
        self.fc1 = torch.nn.Linear(hidden * window, hidden * window, bias=True)
        self.fc2 = torch.nn.Linear(hidden * window, out, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_first:
            x = self.norm(x).reshape(-1, x.shape[-1] * self.window)
        else:
            x = self.norm(x.reshape(-1, x.shape[-1] * self.window))
        return linear32(self.fc2, F.gelu(linear32(self.fc1, x))).to(self.fc2.weight.dtype)


class Qwen3VLVision(torch.nn.Module):
    def __init__(self, vc: dict[str, Any]) -> None:
        super().__init__()
        hidden, heads = vc["hidden_size"], vc["num_heads"]
        self.head_dim = hidden // heads
        self.patch_values = vc["in_channels"] * vc["temporal_patch_size"] * vc["patch_size"] ** 2
        self.patch = torch.nn.Linear(self.patch_values, hidden, bias=True)
        self.table_side = int(round(vc["num_position_embeddings"] ** 0.5))
        self.pos_embed = torch.nn.Embedding(vc["num_position_embeddings"], hidden)
        eps = float(vc.get("layer_norm_eps", 1e-6))
        self.blocks = torch.nn.ModuleList(VisionBlock(hidden, heads, vc["intermediate_size"], eps)
                                          for _ in range(vc["depth"]))
        window = vc["spatial_merge_size"] ** 2
        self.deepstack_at = list(vc.get("deepstack_visual_indexes", []))
        self.merger = Merger(hidden, window, vc["out_hidden_size"], eps, norm_first=True)
        self.deepstack = torch.nn.ModuleList(Merger(hidden, window, vc["out_hidden_size"], eps, norm_first=False)
                                             for _ in self.deepstack_at)
        per_axis = self.head_dim // 4                    # row drives the first half of the frequencies, column the rest
        self.inv_freq = (1.0 / (10000.0 ** (torch.arange(per_axis, dtype=torch.float64) / per_axis))).to(torch.float32)

    def forward(self, pixels, pos_index, pos_weight, patch_rc):
        x = linear32(self.patch, pixels)
        # Gather the table rows with an embedding lookup (index_select into a graph input fails on cpu_only, F-P3-2).
        pos = F.embedding(pos_index, self.pos_embed.weight).float()                 # [n, 4, hidden]
        x = x + (pos * pos_weight.float()[..., None]).sum(1)
        rc = patch_rc.to(torch.float32)
        ang = torch.cat((rc[:, :1] * self.inv_freq, rc[:, 1:2] * self.inv_freq), dim=-1)
        emb = torch.cat((ang, ang), dim=-1)[:, None, :]                              # [n, 1, head_dim]
        cos, sin = emb.cos(), emb.sin()
        levels = []
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin)
            if i in self.deepstack_at:
                levels.append(self.deepstack[self.deepstack_at.index(i)](x))
        return self.merger(x), torch.stack(levels)

    def example(self, max_patches: int = 16384):
        """(example inputs, dynamic shapes): n = 4·k patches, k >= 2 (a real image has >= 64 windows)."""
        k = torch.export.Dim("windows", min=2, max=max_patches // 4)
        ex = host_inputs(1, 8, 12)
        count = ex["patch_rc"].shape[0]
        example = (torch.zeros(count, self.patch_values, dtype=self.patch.weight.dtype),
                   torch.from_numpy(ex["pos_index"]), torch.from_numpy(ex["pos_weight"]), torch.from_numpy(ex["patch_rc"]))
        n = 4 * k
        return example, ({0: n}, {0: n}, {0: n}, {0: n})


def name_map(module: Qwen3VLVision, prefix: str = PREFIX) -> dict[str, str]:
    m = {f"{prefix}patch_embed.proj.weight": "patch.weight", f"{prefix}patch_embed.proj.bias": "patch.bias",
         f"{prefix}pos_embed.weight": "pos_embed.weight"}
    for i in range(len(module.blocks)):
        src, dst = f"{prefix}blocks.{i}.", f"blocks.{i}."
        for a, b in [("norm1", "norm1"), ("norm2", "norm2"), ("attn.qkv", "qkv"), ("attn.proj", "proj"),
                     ("mlp.linear_fc1", "fc1"), ("mlp.linear_fc2", "fc2")]:
            for p in ("weight", "bias"):
                m[f"{src}{a}.{p}"] = f"{dst}{b}.{p}"
    for src, dst in [("merger.", "merger.")] + [(f"deepstack_merger_list.{i}.", f"deepstack.{i}.")
                                                for i in range(len(module.deepstack))]:
        for a, b in (("norm", "norm"), ("linear_fc1", "fc1"), ("linear_fc2", "fc2")):
            for p in ("weight", "bias"):
                m[f"{prefix}{src}{a}.{p}"] = f"{dst}{b}.{p}"
    return m


def load_weights(module: Qwen3VLVision, checkpoint: Path, prefix: str = PREFIX) -> None:
    from safetensors import safe_open

    params = dict(module.named_parameters())
    names = name_map(module, prefix)
    loaded: set[str] = set()
    with torch.no_grad():
        for f in sorted(checkpoint.glob("*.safetensors")):
            with safe_open(f, framework="pt") as st:
                for key in st.keys():
                    if not key.startswith(prefix):
                        continue
                    if key not in names:
                        raise KeyError(f"unexpected vision tensor {key}")
                    dst = names[key]
                    t = st.get_tensor(key)
                    if dst == "patch.weight":
                        t = t.reshape(t.shape[0], -1)                                # the patch conv is a linear map
                    if tuple(t.shape) != tuple(params[dst].shape):
                        raise ValueError(f"{key}: {tuple(t.shape)} != {tuple(params[dst].shape)}")
                    params[dst].copy_(t.to(params[dst].dtype))
                    loaded.add(dst)
    missing = set(params) - loaded
    if missing:
        raise KeyError(f"checkpoint is missing vision tensors {sorted(missing)[:8]}")

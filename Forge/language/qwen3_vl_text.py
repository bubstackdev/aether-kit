"""Qwen3-VL text tower for Aether Kit image input, bucketed step contract (development tooling only).

Adapted from the aether-engine forge (`arch/qwen3_vl.py`, facts F-P3-1..3), not imported. The decoder is the Qwen3
block of `qwen3_decode.py` with three differences:

- 3-axis rotary positions (M-RoPE). Each token has (time, height, width) positions, `rope_positions` [3, q], computed by
  the host. Each rotary frequency is driven by one axis, interleaved per `mrope_section`. Text tokens repeat their
  running position on all axes, which reduces to 1-D RoPE. The causal mask and the cache rows still follow
  `cache_positions`, the running sequence index: rope positions inside an image are not monotonic (F-P3-1).
- Image splice. Rows whose token is `<|image_pad|>` take an image row. With aligned rows (the one-function contract)
  `image_embeds` has one row per token and row i belongs to token i; otherwise (the eager reference and the 0.0.2
  `withImage`) the row index is a running count over the pad mask, derived from token values, gathered with an
  embedding lookup (F-P3-2), and a chunk carries exactly its own pad tokens' rows.
- DeepStack. After decoder layer k < levels, `deepstack[k]` is added at the same rows.

Bucketed contract, one function (0.0.3; docs/qwen3-vl-single-decoder.md):

    main(input_ids [1, q], cache_positions [1, q], rope_positions [3, q], image_embeds [q, hidden],
         deepstack [levels, q, hidden], keyCache, valueCache)
      -> logits [1, 1, vocab], keyUpdate [layers, 1, kv_heads, q, head_dim], valueUpdate

Image rows are used only where the token is `<|image_pad|>`, so text steps send zero rows. Core AI wires a separate
copy of the weights for each function, so the 0.0.2 pair (`main` without image inputs, and `withImage` with
`image_embeds` [n, hidden]) kept two copies resident.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from qwen3_decode import Block, RMSNorm  # the Qwen3 block and norm, shared with the text-only exporter

PREFIX = "model.language_model."


def mrope_axes(sections: list[int], interleaved: bool) -> list[int]:
    """Axis (0 time, 1 height, 2 width) of each rotary frequency. Interleaved: frequency j goes to axis j mod 3 while
    that axis has budget in `sections`; leftovers go to time."""
    total = sum(sections)
    if not interleaved:
        return [a for a, n in enumerate(sections) for _ in range(n)]
    axes, budget = [0] * total, list(sections)
    for j in range(total):
        a = j % 3
        if a != 0 and budget[a] > 0:
            axes[j] = a
            budget[a] -= 1
    return axes


def mrope_positions(ids: list[int], grids: list[tuple[int, int, int]], image_token: int, merge: int = 2,
                    start: int = 0) -> list[tuple[int, int, int]]:
    """Host M-RoPE positions: a text token takes the running position p on every axis and p advances by 1; an image of
    patch grid (t, h, w) takes (p + ti, p + hi, p + wi) over its merged grid in raster order, then p advances by
    max(t, h/merge, w/merge)."""
    out: list[tuple[int, int, int]] = []
    p, i, img = start, 0, 0
    while i < len(ids):
        if ids[i] != image_token:
            out.append((p, p, p))
            p, i = p + 1, i + 1
            continue
        t, h, w = grids[img]
        h, w = h // merge, w // merge
        span = t * h * w
        if ids[i:i + span] != [image_token] * span:
            raise ValueError(f"image {img}: expected {span} pad tokens")
        out += [(p + ti, p + hi, p + wi) for ti in range(t) for hi in range(h) for wi in range(w)]
        i, img, p = i + span, img + 1, p + max(t, h, w)
    return out


class MRotaryEmbedding(torch.nn.Module):
    def __init__(self, head_dim: int, theta: float, axes: list[int]) -> None:
        super().__init__()
        half = head_dim // 2
        if len(axes) != half:
            raise ValueError(f"mrope sections cover {len(axes)} frequencies, head_dim/2 is {half}")
        exponents = torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim
        self.inv_freq = (1.0 / (theta**exponents)).to(torch.float32)     # plain attributes: float32 constants
        select = torch.zeros(3, half, dtype=torch.float32)
        select[torch.tensor(axes), torch.arange(half)] = 1.0
        self.select = select

    def forward(self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        angles = (positions.to(torch.float32).transpose(0, 1) @ self.select) * self.inv_freq      # [q, half]
        emb = torch.cat((angles, angles), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


class Qwen3VLText(torch.nn.Module):
    def __init__(self, cfg: dict[str, Any], image_token: int, levels: int, max_capacity: int = 16384) -> None:
        super().__init__()
        self.cfg, self.image_token, self.levels, self.max_capacity = cfg, image_token, levels, max_capacity
        self.embed = torch.nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.blocks = torch.nn.ModuleList(Block(cfg) for _ in range(cfg["num_hidden_layers"]))
        self.norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.tied = bool(cfg.get("tie_word_embeddings", False))
        self.lm_head = None if self.tied else torch.nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        self.head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
        scaling = cfg.get("rope_scaling") or {}
        self.rope = MRotaryEmbedding(self.head_dim, float(cfg["rope_theta"]),
                                     mrope_axes(list(scaling["mrope_section"]), bool(scaling.get("mrope_interleaved", False))))

    def forward(self, input_ids, cache_positions, rope_positions, key_cache, value_cache, image_embeds=None, deepstack=None,
                aligned=False):
        q_len = input_ids.shape[1]
        x = self.embed(input_ids)
        if image_embeds is not None:
            is_image = input_ids[0] == self.image_token                                         # [q], value-derived
            pick = is_image[None, :, None]
            if aligned:
                def image_rows(t):
                    return t[None].to(x.dtype)
            else:
                row = torch.clamp(torch.cumsum(is_image.to(torch.int32), 0, dtype=torch.int32) - 1, min=0)

                def image_rows(t):
                    return F.embedding(row, t)[None].to(x.dtype)
            x = torch.where(pick, image_rows(image_embeds), x)
        new_positions = cache_positions[0]
        key_positions = torch.cumsum(torch.ones_like(key_cache[0, 0, 0, :, 0], dtype=torch.int32), 0, dtype=torch.int32) - 1
        cache_visible = key_positions[None, :] < new_positions[:1, None]
        chunk_visible = new_positions[None, :] <= new_positions[:, None]
        visible = torch.cat((cache_visible.expand(q_len, -1), chunk_visible), dim=1)
        cos, sin = self.rope(rope_positions, x.dtype)
        keys, values = [], []
        for i, block in enumerate(self.blocks):
            x, k, v = block.bucketed(x, cos, sin, key_cache, value_cache, i, visible)
            keys.append(k)
            values.append(v)
            if deepstack is not None and i < self.levels:
                extra = image_rows(deepstack[i])
                x = x + torch.where(pick, extra, torch.zeros_like(extra))
        logits = F.linear(self.norm(x[:, -1:]), self.embed.weight if self.tied else self.lm_head.weight)
        return logits, torch.stack(keys), torch.stack(values)

    def empty_state(self, capacity: int) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.cfg
        shape = (c["num_hidden_layers"], 1, c["num_key_value_heads"], capacity, self.head_dim)
        dtype = self.embed.weight.dtype
        return torch.zeros(shape, dtype=dtype), torch.zeros(shape, dtype=dtype)

    def step(self, input_ids, cache_positions, rope_positions, key_cache, value_cache, image_embeds=None, deepstack=None,
             aligned=False):
        """One host-loop step on torch caches: last-row logits [vocab], with the returned rows written like the host."""
        logits, ku, vu = self(input_ids, cache_positions, rope_positions, key_cache, value_cache, image_embeds, deepstack,
                              aligned)
        start, q = int(cache_positions[0, 0]), input_ids.shape[1]
        key_cache[:, :, :, start:start + q] = ku
        value_cache[:, :, :, start:start + q] = vu
        return logits[0, -1]

    def unified_example(self, max_len: int):
        """(example, dynamic shapes) for the one-function contract: q from 1 to `max_len`, image rows aligned with it."""
        hidden, dtype = self.cfg["hidden_size"], self.embed.weight.dtype
        q_dim = torch.export.Dim("q", min=1, max=max_len)
        cap_dim = torch.export.Dim("capacity", min=2, max=self.max_capacity)
        q, start = 12, 16
        ids = torch.arange(q, dtype=torch.int32)[None] + 100
        ids[0, 3:8] = self.image_token
        pos = torch.arange(start, start + q, dtype=torch.int32)[None]
        kc, vc = self.empty_state(56)
        example = (ids, pos, pos.expand(3, -1).contiguous(), torch.zeros(q, hidden, dtype=dtype),
                   torch.zeros(self.levels, q, hidden, dtype=dtype), kc, vc)
        return example, ({1: q_dim}, {1: q_dim}, {1: q_dim}, {0: q_dim}, {1: q_dim}, {3: cap_dim}, {3: cap_dim})

    def examples(self, max_len: int):
        """(main example, main dynamic shapes, withImage example, withImage dynamic shapes)."""
        hidden, dtype = self.cfg["hidden_size"], self.embed.weight.dtype
        q_dim = torch.export.Dim("q", min=1, max=max_len)
        cap_dim = torch.export.Dim("capacity", min=2, max=self.max_capacity)
        ex_q, ex_start = 8, 16
        ids = torch.arange(ex_q, dtype=torch.int32)[None] + 100
        pos = torch.arange(ex_start, ex_start + ex_q, dtype=torch.int32)[None]
        kc, vc = self.empty_state(56)
        main = (ids, pos, pos.expand(3, -1).contiguous(), kc, vc)
        main_dyn = ({1: q_dim}, {1: q_dim}, {1: q_dim}, {3: cap_dim}, {3: cap_dim})
        q2 = torch.export.Dim("q", min=2, max=max_len)
        n_dim = torch.export.Dim("n", min=1, max=max_len)
        wq, wn = 12, 5
        wids = torch.arange(wq, dtype=torch.int32)[None] + 100
        wids[0, 3:3 + wn] = self.image_token
        wpos = torch.arange(ex_start, ex_start + wq, dtype=torch.int32)[None]
        with_image = (wids, wpos, wpos.expand(3, -1).contiguous(), torch.zeros(wn, hidden, dtype=dtype),
                      torch.zeros(self.levels, wn, hidden, dtype=dtype), kc, vc)
        with_dyn = ({1: q2}, {1: q2}, {1: q2}, {0: n_dim}, {1: n_dim}, {3: cap_dim}, {3: cap_dim})
        return main, main_dyn, with_image, with_dyn


class Unified(torch.nn.Module):
    """The one-function contract's module: image rows aligned with the tokens, before the caches."""

    def __init__(self, inner: Qwen3VLText) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, input_ids, cache_positions, rope_positions, image_embeds, deepstack, key_cache, value_cache):
        return self.inner(input_ids, cache_positions, rope_positions, key_cache, value_cache, image_embeds, deepstack,
                          aligned=True)


def build_fp16(cfg: dict[str, Any], image_token: int, levels: int, max_capacity: int = 16384) -> Qwen3VLText:
    """An uninitialised fp16 tower for `load_weights` to fill: built on the meta device, so no float32 copy of the
    weights is ever allocated (the float32 build of the 4B needs about 16 GB)."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        with torch.device("meta"):
            text = Qwen3VLText(cfg, image_token, levels, max_capacity=max_capacity)
    finally:
        torch.set_default_dtype(previous)
    text.to_empty(device="cpu")
    # The rotary tables are plain float32 attributes, not parameters: rebuild them off the meta device.
    scaling = cfg.get("rope_scaling") or {}
    text.rope = MRotaryEmbedding(text.head_dim, float(cfg["rope_theta"]),
                                 mrope_axes(list(scaling["mrope_section"]), bool(scaling.get("mrope_interleaved", False))))
    return text


class WithImage(torch.nn.Module):
    """`withImage`'s module: the same parameters (weights stored once), image inputs before the caches."""

    def __init__(self, inner: Qwen3VLText) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, input_ids, cache_positions, rope_positions, image_embeds, deepstack, key_cache, value_cache):
        return self.inner(input_ids, cache_positions, rope_positions, key_cache, value_cache, image_embeds, deepstack)


def load_weights(module: Qwen3VLText, checkpoint: Path) -> None:
    """Qwen3's layout under `model.language_model.`; the vision tower (`model.visual.`) is its own asset."""
    from safetensors import safe_open

    from qwen3_decode import name_map

    names = {(PREFIX + k.removeprefix("model.") if k != "lm_head.weight" else k): v for k, v in name_map(module).items()}
    params = dict(module.named_parameters())
    loaded: dict[str, int] = {}
    with torch.no_grad():
        for f in sorted(checkpoint.glob("*.safetensors")):
            with safe_open(f, framework="pt") as st:
                for key in st.keys():
                    if key.startswith("model.visual.") or (key == "lm_head.weight" and module.tied):
                        continue
                    if key not in names:
                        raise KeyError(f"unexpected checkpoint tensor {key}")
                    dst, row = names[key]
                    t = st.get_tensor(key)
                    params[dst][row:row + t.shape[0]].copy_(t.to(params[dst].dtype))
                    loaded[dst] = loaded.get(dst, 0) + t.shape[0]
    missing = sorted(n for n, p in params.items() if loaded.get(n, 0) != p.shape[0])
    if missing:
        raise KeyError(f"checkpoint is missing {missing[:8]}")

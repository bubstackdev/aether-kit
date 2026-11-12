"""Krea 2 Turbo export graphs and host-side helpers for the Aether-Kit `krea2@1` bundle (development tooling only).

Written against the pinned upstream sources: Diffusers `Krea2Transformer2DModel`, `AutoencoderKLQwenImage` and
`Krea2Pipeline` (commit 2a14339, tag v0.42.1) and Transformers 5.22.1 `Qwen3VLModel`. Every graph reads the pinned
checkpoint's safetensors directly. The graph boundaries follow the 0.0.x Krea pipeline shape (Qwen text encoder,
four denoiser segments per step, CPU Euler, VAE decode), and small idioms (the split-halves rotary permutation,
SplitMix64 + Box-Muller noise, the lazy checkpoint reader) come from the 0.0.x Forge and runtime.

Functions (names, dtypes and dynamic axes are the bundle contract):

    textEncoder/encode   input_ids, position_ids int32 [1, L] (L = 35...546, unpadded)
                         -> text_embeds fp16 [L - 34, 30720]   (12 tapped layers x 2560, tap-major)
    textEncoder/encodeWithImages   input_ids int32 [1, L] (L = 36...1150; image pads as extension ids),
                         rope_positions int32 [3, L], image_embeds f32 [576, 2560], deepstack_embeds f32 [3, 576, 2560]
                         -> text_embeds fp16 [L - 34, 30720]   (reference images; `ReferenceTextGraph`)
    vision/encode        patches f16 [576, 1536], grid int32 [2] -> image_embeds f32 [144, 2560],
                         deepstack_embeds f32 [3, 144, 2560]   (one reference image; `KreaVisionTower`, static)
    denoiser0/segment    latents fp16 [N, 64], text_embeds fp16 [T, 30720], text_mask f32 [T], sigma f32 [1],
                         rope_image f32 [N, 2, 128]
                         -> image_1 f32 [N, 6144], text_1 f32 [T, 6144], cond f32 [8, 6144]
    denoiser1/segment    image_1, text_1, text_mask, cond, rope_image -> image_2, text_2
    denoiser2/segment    image_2, text_2, text_mask, cond, rope_image -> image_3, text_3
    denoiser3/segment    image_3, text_3, text_mask, cond, rope_image -> velocity fp16 [N, 64]
    vaeDecoder/decode    latents fp16 [H/16, W/16, 64] (packed, normalized) -> image f32 [3, H, W] in [-1, 1]

Precision: text encoder and denoiser matmuls and attention run in fp16 with fp16 weights (denoiser block 0 computes in
float32 from its fp16 weights, `FLOAT32_BLOCKS`); residual streams, RMSNorms,
modulation, rotary products and the small edge projections (timestep MLP, modulation projection, text/image
in-projections, final layer) run in float32 (edge weights stored fp16, widened). The VAE decoder stores fp16 weights
and computes in float32.

Text rows: the text encoder runs unpadded (the 0.0.x prompt rule: render, keep at most 546 tokens with the final five,
drop the 34-token system prefix), which gives exactly the real rows of Diffusers' padded `[prefix | prompt | PAD |
suffix]` block. The denoisers take one fixed text length for every prompt: by default T = 1116 (`MASK_TEXT_ROWS`, the
most rows a reference prompt can have; Diffusers pads text-only prompts to 512), static, with the host's real rows
first and finite padding (zeros) after. `text_rows` exports a shorter T (for example 256, for faster steps); the
manifest declares it as `krea2.textLength` and the host refuses a prompt with more rows (a typed capacity error), so
it can be routed to a 1116-row export. One length means one function and one shape for text-only and reference prompts alike:
a second function (a 512-row one beside it) slowed the first's steps about 1.7x on macOS 27.2, and Core AI keeps a
separate weights copy per function it runs (2026-11-09). `text_mask` [T] is 1 for a real row and 0 for padding, and is
applied exactly as Diffusers applies `encoder_attention_mask`: a key-padding mask in the text-fusion refiner blocks and
in all 28 joint blocks (image keys always valid). The layerwise fusion blocks attend over one token's tapped layers, so
padding can't reach them; text positions are all zero, so the padding shifts no rotary angle. Pad rows' own lanes are
finite and never read. `masked=False` keeps the earlier exact-length graphs (a dynamic T = 2...1116, no mask).

Rotary: Diffusers rotates interleaved pairs (2i, 2i+1); Kit's table is split halves ([n, 0, :] cos, [n, 1, :] sin;
pair i rotates dims i and i + 64), as `KreaRotary` in 0.0.x. The export permutes each head's q/k projection rows and
q/k norm gains (`HALF_SPLIT`) so the split-halves rotation yields identical attention scores. Text positions are all
zero in Krea 2, i.e. the identity rotation, so only image rows are rotated.
"""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# MARK: constants (asserted against the checkpoint configs at load)

TEXT_TAPS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)   # `text_encoder_select_layers`; hidden_states[k] = layer k output
TEXT_LAYERS = max(TEXT_TAPS)                                 # layers 0..34 run; layer 35 and the final norm are unused
TEXT_WIDTH = 2560
TEXT_HEAD_DIM = 128
DROP_ROWS = 34                                               # `prompt_template_encode_start_idx`
MAX_TOKENS = 546                                             # 512 + 34
TRAILING_TOKENS = 5                                          # `prompt_template_encode_num_suffix_tokens`
SYSTEM_PREFIX = ("<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
                 "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
CONDITIONING = len(TEXT_TAPS) * TEXT_WIDTH                   # 30720

LAYERS = 28
SEGMENTS = ((0, 7), (7, 14), (14, 21), (21, 28))
HIDDEN = 6144
HEADS = 48
KV_HEADS = 12
HEAD_DIM = 128
LATENT_CHANNELS = 64                                         # 16 VAE channels x 2 x 2
CONDITION_ROWS = 8                                           # 6 shared block modulation rows + final (scale, shift)
# Blocks computed in float32 (weights still stored fp16). Block 0 writes Krea 2's massive activations (|h| ~80 ->
# ~1700) and its fp16 update loses ~2e-5 cosine, which the 8-step closed loop amplifies past the image bar (realism
# SSIM 0.86); blocks 1..27 are exact to 1e-7 in fp16. Measured 2026-11-05, see the check reports.
FLOAT32_BLOCKS = (0,)
ROPE_AXES = (32, 48, 48)
ROPE_THETA = 1000.0
NORM_EPS = 1e-5
PIXELS_PER_TOKEN = 16                                        # VAE 8x, patch 2
STEPS = 8
MU = 1.15                                                    # `is_distilled` -> fixed exponential shift
TRAIN_TIMESTEPS = 1000
MAX_IMAGE_TOKENS = 4096
MAX_TEXT_ROWS = MAX_TOKENS - DROP_ROWS                       # 512, a text-only prompt's conditioning rows

# Reference images (the 0.0.x reference contract): each image is a labelled part of the user message before the text,
# "Picture n: <|vision_start|><|image_pad|><|vision_end|>", its pad expanded to the image's merged tokens, embedded by
# Qwen3-VL-4B's own vision tower (in the text encoder checkpoint) and fed to the unchanged taps as extra text rows.
REFERENCE_MAX_IMAGES = 4
REFERENCE_MAX_SIDE = 384
REFERENCE_MAX_IMAGE_TOKENS = 576                             # merged tokens over all images (the text graph's slots)
REFERENCE_PATCH_CAPACITY = 576                               # one image's patch rows: 24 x 24 patches, 144 tokens
REFERENCE_LABEL = "Picture {n}: "
IMAGE_PAD_ID, VISION_START_ID, VISION_END_ID = 151655, 151652, 151653
EXTENSION_BASE = 151936                                      # the text vocabulary: image slot s is id EXTENSION_BASE + s
DEEPSTACK_LEVELS = 3
MROPE_SECTION = (24, 20, 20)
# 512 text rows + 576 image tokens + each image's label and vision start/end (7 ids) = 1116; L = 1116 + 34 ids.
REFERENCE_MAX_TEXT_ROWS = MAX_TEXT_ROWS + REFERENCE_MAX_IMAGE_TOKENS + 7 * REFERENCE_MAX_IMAGES
DENOISER_TEXT_ROWS = REFERENCE_MAX_TEXT_ROWS                 # the denoisers take text-only and reference rows
# The masked denoisers' one fixed text length, for text-only and reference prompts alike (the default; an export may
# choose a shorter one, declared as the manifest's `krea2.textLength`).
MASK_TEXT_ROWS = REFERENCE_MAX_TEXT_ROWS


def text_length(manifest: dict) -> int | None:
    """A bundle's masked text length: `krea2.textLength`, else (the first masked exports) `MASK_TEXT_ROWS`; None when
    the denoisers take no `textMask` (exact length)."""
    if "textMask" not in manifest["graphs"]["denoiser0"]["inputs"]:
        return None
    return manifest["krea2"].get("textLength", MASK_TEXT_ROWS)

# Within a head, new dim i takes interleaved dim 2i and new dim i + 64 takes dim 2i + 1.
HALF_SPLIT = torch.cat([torch.arange(0, HEAD_DIM, 2), torch.arange(1, HEAD_DIM, 2)])


# MARK: checkpoint access

class Checkpoint:
    """Lazy tensor reads from the pinned snapshot's safetensors (one component at a time keeps peak RSS low)."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def config(self, component: str) -> dict:
        return json.loads((self.root / component / "config.json").read_text())

    def files(self, component: str) -> dict[str, Path]:
        directory = self.root / component
        index = next(directory.glob("*.safetensors.index.json"), None)
        if index is not None:
            weight_map = json.loads(index.read_text())["weight_map"]
            return {name: directory / file for name, file in weight_map.items()}
        (single,) = directory.glob("*.safetensors")
        from safetensors import safe_open
        with safe_open(single, "pt") as handle:
            return {name: single for name in handle.keys()}

    def read(self, component: str, names: list[str]) -> dict[str, torch.Tensor]:
        from safetensors import safe_open
        where = self.files(component)
        missing = [n for n in names if n not in where]
        if missing:
            raise KeyError(f"{component} lacks {missing[:4]}")
        out: dict[str, torch.Tensor] = {}
        for file in sorted({where[n] for n in names}):
            with safe_open(file, "pt") as handle:
                for name in names:
                    if where[name] == file:
                        out[name] = handle.get_tensor(name)
        return out


# MARK: the realism adapter (LyCORIS LoKr, ComfyUI key names)

class LoKr:
    """A LyCORIS LoKr adapter merged as W' = W + multiplier * scale * kron(w1, w2).

    Scale follows LyCORIS `LokrModule` and ComfyUI's LoKr weight adapter: when a factor is itself low-rank
    (`lokr_w1_a/_b` or `lokr_w2_a/_b`), rank = that factor's inner dimension and scale = alpha / rank; when both
    factors are full matrices (no factorisation, the case for this adapter) there is no rank and scale = 1, whatever
    `alpha` says (ai-toolkit writes alpha = 1e10 as its "full" marker). Tucker cores (`lokr_t2`) are refused.

    Keys map onto `Krea2Transformer2DModel` through Diffusers' own Krea 2 converter
    (`_convert_non_diffusers_krea2_lora_to_diffusers`), which raises on any module it cannot name; `verify` then
    asserts the mapping is total and every kron product has its target's shape.
    """

    def __init__(self, path: Path, multiplier: float = 1.0) -> None:
        from safetensors import safe_open
        from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_krea2_lora_to_diffusers

        self.path, self.multiplier = path, float(multiplier)
        with safe_open(path, "pt") as handle:
            tensors = {k: handle.get_tensor(k) for k in handle.keys()}
        groups: dict[str, dict[str, torch.Tensor]] = {}
        for key, value in tensors.items():
            module, _, part = key.rpartition(".")
            groups.setdefault(module, {})[part] = value
        allowed = {"lokr_w1", "lokr_w1_a", "lokr_w1_b", "lokr_w2", "lokr_w2_a", "lokr_w2_b", "alpha"}
        for module, parts in groups.items():
            unknown = set(parts) - allowed
            if unknown:
                raise SystemExit(f"adapter module {module} has unsupported parts {sorted(unknown)} (only plain LoKr)")
        # Diffusers' converter only handles LoRA suffixes, so ask it to name each module through one probe key. It
        # raises when a key is left unconverted; an empty result also means "unmapped".
        mapped = {}
        for module in groups:
            try:
                converted = _convert_non_diffusers_krea2_lora_to_diffusers({f"{module}.lora_A.weight": torch.empty(0)})
            except ValueError:
                converted = {}
            if len(converted) != 1:
                raise SystemExit(f"Diffusers' Krea 2 converter cannot map adapter module {module}")
            (key,) = converted
            mapped[module] = key.removeprefix("transformer.").removesuffix(".lora_A.weight")
        if len(set(mapped.values())) != len(mapped):
            raise SystemExit("two adapter modules map to one transformer weight")
        self.targets: dict[str, dict] = {}
        for module, target in mapped.items():
            parts = groups[module]
            w1 = parts.get("lokr_w1")
            w2 = parts.get("lokr_w2")
            rank = None
            if w1 is None:
                w1 = parts["lokr_w1_a"].float() @ parts["lokr_w1_b"].float()
                rank = parts["lokr_w1_b"].shape[0]
            if w2 is None:
                w2 = parts["lokr_w2_a"].float() @ parts["lokr_w2_b"].float()
                rank = parts["lokr_w2_b"].shape[0]
            alpha = float(parts["alpha"].float()) if "alpha" in parts else None
            scale = alpha / rank if (rank is not None and alpha) else 1.0
            self.targets[target + ".weight"] = {"module": module, "w1": w1, "w2": w2, "alpha": alpha, "rank": rank,
                                                "scale": scale}
        self.adapter_keys = len(tensors)

    def delta(self, name: str) -> torch.Tensor | None:
        entry = self.targets.get(name)
        if entry is None:
            return None
        return torch.kron(entry["w1"].float(), entry["w2"].float()) * (entry["scale"] * self.multiplier)

    def verify(self, shapes: dict[str, tuple[int, ...]]) -> dict:
        """Totality: every adapter module maps to an existing transformer weight with the kron product's shape."""
        missing = [n for n in self.targets if n not in shapes]
        if missing:
            raise SystemExit(f"adapter targets absent from the transformer: {missing[:4]}")
        wrong = []
        for name, entry in self.targets.items():
            shape = (entry["w1"].shape[0] * entry["w2"].shape[0], entry["w1"].shape[1] * entry["w2"].shape[1])
            if tuple(shapes[name]) != shape:
                wrong.append((name, shape, tuple(shapes[name])))
        if wrong:
            raise SystemExit(f"adapter shape mismatches: {wrong[:4]}")
        kinds: dict[str, int] = {}
        for name in self.targets:
            kind = re.sub(r"\.\d+\.", ".N.", name)
            kinds[kind] = kinds.get(kind, 0) + 1
        return {"adapterTensors": self.adapter_keys, "modules": len(self.targets), "mapped": len(self.targets),
                "unmapped": 0, "shapeMismatches": 0, "byTarget": kinds,
                "scale": sorted({e["scale"] for e in self.targets.values()}),
                "alpha": sorted({e["alpha"] for e in self.targets.values() if e["alpha"] is not None}),
                "factorised": sum(e["rank"] is not None for e in self.targets.values()),
                "multiplier": self.multiplier}


def merge(name: str, value: torch.Tensor, adapter: LoKr | None) -> torch.Tensor:
    """The one merge both the export and the CPU reference use: float32 `W + delta` (deltas built in float32)."""
    value = value.float()
    if adapter is not None:
        delta = adapter.delta(name)
        if delta is not None:
            value = value + delta
    return value


class MergedWeights(dict):
    """Checkpoint tensors in their stored dtype; `pop` returns the float32 merged weight, so at most one float32
    copy exists at a time while modules are built."""

    def __init__(self, weights: dict[str, torch.Tensor], adapter: LoKr | None) -> None:
        super().__init__(weights)
        self.adapter = adapter

    def pop(self, name, *default):
        return merge(name, super().pop(name, *default), self.adapter)


# MARK: module helpers

def half_split_rows(weight: torch.Tensor, heads: int) -> torch.Tensor:
    """Reorder each head's output rows from interleaved rotary pairs to split halves."""
    return weight.reshape(heads, HEAD_DIM, -1)[:, HALF_SPLIT, :].reshape(heads * HEAD_DIM, -1)


def fp16_checked(weight: torch.Tensor, name: str) -> torch.Tensor:
    peak = float(weight.abs().max())
    if not math.isfinite(peak) or peak >= 65504:
        raise SystemExit(f"{name} does not fit fp16 (max |w| = {peak})")
    return weight.to(torch.float16).contiguous()


class Linear(nn.Module):
    """A bias-free matmul in `compute` with weights stored in `compute`."""

    def __init__(self, weight: torch.Tensor, compute: torch.dtype, name: str = "") -> None:
        super().__init__()
        w = fp16_checked(weight, name) if compute == torch.float16 else weight.to(compute).contiguous()
        self.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        # The reference text function may compute wider than the stored weights (cast in the graph, stored once).
        return F.linear(x, self.weight if x.dtype == self.weight.dtype else self.weight.to(x.dtype))


class WideLinear(nn.Module):
    """Weights stored fp16 (or `storage`), widened to float32 for compute; float32 bias."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, storage: torch.dtype, name: str = "") -> None:
        super().__init__()
        w = fp16_checked(weight, name) if storage == torch.float16 else weight.to(storage).contiguous()
        self.register_buffer("weight", w)
        self.register_buffer("bias", None if bias is None else bias.float().contiguous())

    def forward(self, x):
        return F.linear(x.float(), self.weight.float(), self.bias)


def rms(x: torch.Tensor, gain: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm in float32 with a precomputed float32 gain (Krea 2 stores `1 + weight`)."""
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gain


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


def buffer(value: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(value.float().contiguous(), requires_grad=False)


# MARK: text encoder (Qwen3-VL-4B language model, layers 0..34)

class TextLayer(nn.Module):
    def __init__(self, cfg: dict, compute: torch.dtype) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.eps = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["rms_norm_eps"]
        self.compute = compute

    def forward(self, h, cos, sin, mask, compute: torch.dtype | None = None):
        compute = compute or self.compute
        length = h.shape[1]
        x = rms(h, self.input_norm, self.eps).to(compute)
        q, k, v = self.qkv(x).split([self.heads * TEXT_HEAD_DIM, self.kv_heads * TEXT_HEAD_DIM,
                                     self.kv_heads * TEXT_HEAD_DIM], dim=-1)
        q = rms(q.unflatten(-1, (self.heads, TEXT_HEAD_DIM)), self.q_norm, self.eps).transpose(1, 2)
        k = rms(k.unflatten(-1, (self.kv_heads, TEXT_HEAD_DIM)), self.k_norm, self.eps).transpose(1, 2)
        v = v.unflatten(-1, (self.kv_heads, TEXT_HEAD_DIM)).transpose(1, 2)
        q = (q * cos + rotate_half(q) * sin).to(compute)
        k = (k * cos + rotate_half(k) * sin).to(compute)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=TEXT_HEAD_DIM ** -0.5, enable_gqa=True)
        h = h + self.o(attn.transpose(1, 2).reshape(1, length, self.heads * TEXT_HEAD_DIM)).float()
        x = rms(h, self.post_norm, self.eps).to(compute)
        return h + self.down(F.silu(self.gate(x)) * self.up(x)).float()


class TextEncoderGraph(nn.Module):
    """Qwen3-VL text layers 0..34 with Krea's 12 taps, causal attention over unpadded rows, float32 residual stream.

    Text-only Qwen3-VL positions are equal on the T/H/W mRoPE axes, so the interleaved mRoPE reduces exactly to the
    standard rotary embedding over `position_ids`. Outputs the taps for rows 34.. as [L - 34, 12 * 2560] fp16.
    """

    def __init__(self, cfg: dict, compute: torch.dtype, layers: int = TEXT_LAYERS) -> None:
        super().__init__()
        self.cfg, self.compute = cfg, compute
        self.layers = nn.ModuleList(TextLayer(cfg, compute) for _ in range(layers))
        theta = cfg["rope_parameters"]["rope_theta"]
        dim = cfg.get("head_dim", TEXT_HEAD_DIM)
        self.register_buffer("inv_freq", 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)),
                             persistent=False)

    @classmethod
    def load(cls, checkpoint: Checkpoint, compute=torch.float16, layers: int = TEXT_LAYERS,
             strict: bool = True) -> "TextEncoderGraph":
        cfg = checkpoint.config("text_encoder")["text_config"]
        expected = {"hidden_size": TEXT_WIDTH, "head_dim": TEXT_HEAD_DIM, "num_hidden_layers": 36, "hidden_act": "silu",
                    "attention_bias": False}
        if strict:
            for key, value in expected.items():
                if cfg.get(key) != value:
                    raise SystemExit(f"text encoder config {key}={cfg.get(key)!r}, expected {value!r}")
        rope = cfg["rope_parameters"]
        if rope.get("rope_type", "default") != "default":
            raise SystemExit(f"unsupported text rope {rope}")
        graph = cls(cfg, compute, layers)
        prefix = "language_model."
        names = [prefix + "embed_tokens.weight"]
        parts = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj",
                 "mlp.up_proj", "mlp.down_proj", "input_layernorm", "post_attention_layernorm", "self_attn.q_norm",
                 "self_attn.k_norm")
        for i in range(layers):
            names += [f"{prefix}layers.{i}.{n}.weight" for n in parts]
        w = checkpoint.read("text_encoder", names)
        embed = w.pop(prefix + "embed_tokens.weight")
        graph.embed = nn.Embedding(embed.shape[0], embed.shape[1], device="meta")
        graph.embed.weight = nn.Parameter(fp16_checked(embed.float(), "embed") if compute == torch.float16
                                          else embed.to(compute), requires_grad=False)
        for i, layer in enumerate(graph.layers):
            p = f"{prefix}layers.{i}."
            g = lambda n: w.pop(p + n + ".weight").float()  # noqa: E731
            layer.qkv = Linear(torch.cat([g("self_attn.q_proj"), g("self_attn.k_proj"), g("self_attn.v_proj")]),
                               compute, p + "qkv")
            layer.o = Linear(g("self_attn.o_proj"), compute, p + "o")
            layer.gate = Linear(g("mlp.gate_proj"), compute, p + "gate")
            layer.up = Linear(g("mlp.up_proj"), compute, p + "up")
            layer.down = Linear(g("mlp.down_proj"), compute, p + "down")
            layer.input_norm = buffer(g("input_layernorm"))
            layer.post_norm = buffer(g("post_attention_layernorm"))
            layer.q_norm = buffer(g("self_attn.q_norm"))
            layer.k_norm = buffer(g("self_attn.k_norm"))
        return graph.eval()

    def forward(self, input_ids, position_ids):
        h = self.embed(input_ids).float()
        positions = position_ids[0]
        angles = positions.to(torch.float32)[:, None] * self.inv_freq[None, :]
        emb = torch.cat((angles, angles), dim=-1)
        cos, sin = emb.cos(), emb.sin()
        mask = torch.logical_not(positions[None, :] > positions[:, None])      # causal over unpadded rows
        taps = []
        for index, layer in enumerate(self.layers, start=1):
            h = layer(h, cos, sin, mask)
            if index in TEXT_TAPS:
                taps.append(h[0, DROP_ROWS:])
        return torch.stack(taps, dim=1).flatten(1).to(torch.float16)

    @staticmethod
    def example(length: int = 48):
        ids = torch.arange(1000, 1000 + length, dtype=torch.int32)[None]
        positions = torch.arange(length, dtype=torch.int32)[None]
        tokens = torch.export.Dim("tokens", min=DROP_ROWS + 2, max=MAX_TOKENS)
        return (ids, positions), ({1: tokens}, {1: tokens})


class ReferenceTextGraph(nn.Module):
    """`textEncoder/encodeWithImages`: the text graph's layers (shared weights) with reference images as extra rows.

    input_ids int32 [1, L]: the rendered ids, each image's pads replaced by extension ids `EXTENSION_BASE + slot` (slot
    0...575, running over the images in order); rope_positions int32 [3, L] (t, h, w; the 0.0.x M-RoPE: text at the
    running position on all axes, an image's tokens at (p + ti, p + hi, p + wi) in raster order, then p += max(t, h,
    w)); image_embeds f32 [576, 2560] and deepstack_embeds f32 [3, 576, 2560] (level-major), slot-indexed, rows past the
    images ignored. Out: the taps for rows 34.. as [L - 34, 12 * 2560] fp16, as `encode`.

    As transformers' Qwen3VLModel: an image token's embedding is its slot's vision row; DeepStack level k is added at
    image rows after decoder layer k (k < 3), after that layer's output is recorded, so `hidden_states[2]` (the second
    layer's output) holds level 0 but not level 1. The interleaved M-RoPE takes frequency j from h for j in range(1,
    3 * section[1], 3), from w for j in range(2, 3 * section[2], 3), else from t. Causal over row order.
    """

    def __init__(self, base: TextEncoderGraph, compute: torch.dtype | None = None) -> None:
        super().__init__()
        self.base, self.compute = base, compute or base.compute
        section = base.cfg["rope_parameters"].get("mrope_section", list(MROPE_SECTION))
        half = base.inv_freq.shape[0]
        axis = torch.zeros(half, dtype=torch.long)
        axis[1:3 * section[1]:3] = 1
        axis[2:3 * section[2]:3] = 2
        self.register_buffer("axis_masks", F.one_hot(axis, 3).T.float().contiguous(), persistent=False)  # [3, half]

    def forward(self, input_ids, rope_positions, image_embeds, deepstack_embeds):
        g = self.base
        ids = input_ids[0]
        image = ids >= EXTENSION_BASE
        slot = torch.clamp(ids - EXTENSION_BASE, min=0, max=image_embeds.shape[0] - 1)
        text = g.embed(torch.where(image, torch.zeros_like(ids), ids)).float()
        h = torch.where(image[:, None], F.embedding(slot, image_embeds).float(), text)[None]
        freqs = rope_positions.float()[:, :, None] * g.inv_freq[None, None, :]            # [3, L, half]
        angles = (freqs * self.axis_masks[:, None, :]).sum(0)
        emb = torch.cat((angles, angles), dim=-1)
        cos, sin = emb.cos(), emb.sin()
        row = torch.ones_like(ids).cumsum(0)
        mask = torch.logical_not(row[None, :] > row[:, None])                              # causal over rows
        levels = deepstack_embeds.shape[0]
        taps = []
        for index, layer in enumerate(g.layers, start=1):
            h = layer(h, cos, sin, mask, self.compute)
            if index in TEXT_TAPS:
                taps.append(h[0, DROP_ROWS:])
            if index <= levels:
                extra = F.embedding(slot, deepstack_embeds[index - 1]).float()
                h = h + torch.where(image[:, None], extra, torch.zeros_like(extra))[None]
        return torch.stack(taps, dim=1).flatten(1).to(torch.float16)

    def example(self, length: int = 64, image_tokens: int = 12):
        width = self.base.embed.weight.shape[1]
        g = torch.Generator().manual_seed(5)
        ids = torch.arange(1000, 1000 + length, dtype=torch.int32)
        ids[40:40 + image_tokens] = EXTENSION_BASE + torch.arange(image_tokens, dtype=torch.int32)
        rope = torch.arange(length, dtype=torch.int32)[None].repeat(3, 1)
        embeds = torch.randn(REFERENCE_MAX_IMAGE_TOKENS, width, generator=g) * 0.1
        deep = torch.randn(DEEPSTACK_LEVELS, REFERENCE_MAX_IMAGE_TOKENS, width, generator=g) * 0.1
        tokens = torch.export.Dim("tokens", min=DROP_ROWS + 2, max=DROP_ROWS + REFERENCE_MAX_TEXT_ROWS)
        return (ids[None], rope, embeds, deep), ({1: tokens}, {1: tokens}, None, None)


def load_visual(checkpoint: Checkpoint):
    """Qwen3-VL-4B's vision tower (transformers' unmodified `Qwen3VLVisionModel`) with the text encoder checkpoint's
    `visual.*` weights, float32."""
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    config = Qwen3VLVisionConfig(**checkpoint.config("text_encoder")["vision_config"])
    model = Qwen3VLVisionModel(config).float().eval()
    names = [n for n in checkpoint.files("text_encoder") if n.startswith("visual.")]
    state = {n.removeprefix("visual."): t.float() for n, t in checkpoint.read("text_encoder", names).items()}
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    return model


class KreaVisionTower(nn.Module):
    """`vision/encode`: one reference image's patches f16 [576, 1536] (block-major, C,T,py,px rows, the first h*w real)
    and grid int32 [2] = (h, w) -> image_embeds f32 [144, 2560] and deepstack_embeds f32 [3, 144, 2560] (level-major).
    Rows past the image's h*w/4 tokens are zero. A static capacity, as Forge/qwen3vl's tower (which this wraps): one
    shape serves every grid, and padding rows are masked out of attention as keys."""

    def __init__(self, visual, capacity: int = REFERENCE_PATCH_CAPACITY, dtype: torch.dtype = torch.float32,
                 storage: torch.dtype = torch.float16) -> None:
        super().__init__()
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwen3vl"))
        import qwen3vl_graphs as qg
        self.tower = qg.VisionTower(visual, capacity=capacity, dtype=dtype, storage=storage)
        self.tokens = capacity // (visual.config.spatial_merge_size ** 2)

    def forward(self, patches, grid):
        image, deep = self.tower(patches, grid)
        return image, deep.reshape(-1, self.tokens, image.shape[1])

    def example(self, grid=(24, 20)):
        p, g = self.tower.example(grid)
        return (p.half(), g), None


def reference_prompt(tokenizer, text: str, grids: list[tuple[int, int, int]]) -> dict:
    """The 0.0.x reference prompt for `text` and images with patch grids (t, h, w), in order: ids (pads expanded, the
    real pad id), the graph's ids (pads as extension ids), rope [3, L], each image's token span, and truncation."""
    parts = "".join(REFERENCE_LABEL.format(n=i + 1) + "<|vision_start|><|image_pad|><|vision_end|>"
                    for i in range(len(grids)))
    rendered = tokenizer(SYSTEM_PREFIX + parts + text + SUFFIX, add_special_tokens=False)["input_ids"]
    text_only = len(tokenizer(SYSTEM_PREFIX + text + SUFFIX, add_special_tokens=False)["input_ids"])
    if not 1 <= len(grids) <= REFERENCE_MAX_IMAGES:
        raise ValueError(f"1...{REFERENCE_MAX_IMAGES} reference images")
    merged = [(t, h // 2, w // 2) for t, h, w in grids]
    if sum(t * h * w for t, h, w in merged) > REFERENCE_MAX_IMAGE_TOKENS:
        raise ValueError("reference images exceed 576 tokens")
    truncated = text_only > MAX_TOKENS
    if truncated:
        cut = len(rendered) - TRAILING_TOKENS - (text_only - MAX_TOKENS)
        rendered = rendered[:cut] + rendered[-TRAILING_TOKENS:]
    ids, graph_ids, spans, slot = [], [], [], 0
    for token in rendered:
        if token != IMAGE_PAD_ID:
            ids.append(token)
            graph_ids.append(token)
            continue
        t, h, w = merged[len(spans)]
        n = t * h * w
        spans.append((len(ids), len(ids) + n, (t, h, w)))
        ids += [IMAGE_PAD_ID] * n
        graph_ids += [EXTENSION_BASE + slot + j for j in range(n)]
        slot += n
    if len(spans) != len(grids) or spans[0][0] < DROP_ROWS or len(ids) - DROP_ROWS > REFERENCE_MAX_TEXT_ROWS:
        raise ValueError("the rendered reference prompt does not fit the contract")
    axes = [[], [], []]
    p, i, k = 0, 0, 0
    while i < len(ids):
        if k < len(spans) and spans[k][0] == i:
            start, end, (t, h, w) = spans[k]
            for j in range(end - start):
                axes[0].append(p + j // (h * w))
                axes[1].append(p + (j // w) % h)
                axes[2].append(p + j % w)
            p += max(t, h, w)
            i, k = end, k + 1
            continue
        for a in axes:
            a.append(p)
        p, i = p + 1, i + 1
    return {"ids": ids, "graphIds": graph_ids, "rope": axes, "spans": spans, "truncated": truncated}


# MARK: denoiser

class Attention(nn.Module):
    """Krea2Attention: GQA projections, q/k RMSNorm (1 + w), optional rotary on the trailing rows, sigmoid gate."""

    def __init__(self, heads: int, kv_heads: int, compute: torch.dtype) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.compute = heads, kv_heads, compute

    def forward(self, x16, cos=None, sin=None, rotated_from: int = 0, mask=None):
        # x16: [B, S, D] in compute dtype; mask: None or a key-padding mask [1, 1, 1, S] (bool, True = attend)
        q = self.to_q(x16).unflatten(-1, (self.heads, HEAD_DIM))
        k = self.to_k(x16).unflatten(-1, (self.kv_heads, HEAD_DIM))
        v = self.to_v(x16).unflatten(-1, (self.kv_heads, HEAD_DIM))
        gate = self.to_gate(x16)
        q = rms(q, self.norm_q, NORM_EPS)
        k = rms(k, self.norm_k, NORM_EPS)
        if cos is not None:
            # Text rows (positions 0) keep the identity rotation; image rows rotate with Kit's split-halves table.
            q = torch.cat([q[:, :rotated_from], q[:, rotated_from:] * cos + rotate_half(q[:, rotated_from:]) * sin], 1)
            k = torch.cat([k[:, :rotated_from], k[:, rotated_from:] * cos + rotate_half(k[:, rotated_from:]) * sin], 1)
        q, k = q.to(self.compute).transpose(1, 2), k.to(self.compute).transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=HEAD_DIM ** -0.5,
                                             enable_gqa=self.kv_heads != self.heads)
        out = out.transpose(1, 2).flatten(2) * torch.sigmoid(gate)
        return self.to_out(out)


class SwiGLU(nn.Module):
    def forward(self, x16):
        return self.down(F.silu(self.gate(x16)) * self.up(x16))


ATTN_PARTS = (("to_q", "attn.to_q"), ("to_k", "attn.to_k"), ("to_v", "attn.to_v"), ("to_gate", "attn.to_gate"),
              ("to_out", "attn.to_out.0"))
FF_PARTS = (("gate", "ff.gate"), ("up", "ff.up"), ("down", "ff.down"))


def projection(weight: torch.Tensor, compute: torch.dtype, storage: torch.dtype, name: str) -> nn.Module:
    if compute == storage:
        return Linear(weight, compute, name)
    return WideLinear(weight, None, storage, name)                       # fp16 storage, float32 compute


def build_attention(w: dict, p: str, heads: int, kv_heads: int, compute: torch.dtype, rotary: bool,
                    storage: torch.dtype | None = None) -> Attention:
    storage = storage or compute
    attn = Attention(heads, kv_heads, compute)
    for attr, name in ATTN_PARTS:
        weight = w.pop(p + name + ".weight")
        if rotary and attr == "to_q":
            weight = half_split_rows(weight, heads)
        if rotary and attr == "to_k":
            weight = half_split_rows(weight, kv_heads)
        setattr(attn, attr, projection(weight, compute, storage, p + name))
    for attr in ("norm_q", "norm_k"):
        gain = 1.0 + w.pop(p + f"attn.{attr}.weight")
        setattr(attn, attr, buffer(gain[HALF_SPLIT] if rotary else gain))
    return attn


def build_ff(w: dict, p: str, compute: torch.dtype, storage: torch.dtype | None = None) -> SwiGLU:
    ff = SwiGLU()
    for attr, name in FF_PARTS:
        setattr(ff, attr, projection(w.pop(p + name + ".weight"), compute, storage or compute, p + name))
    return ff


def block_names(p: str) -> list[str]:
    return ([p + n + ".weight" for _, n in ATTN_PARTS + FF_PARTS]
            + [p + f"attn.{n}.weight" for n in ("norm_q", "norm_k")] + [p + "norm1.weight", p + "norm2.weight"])


class FusionBlock(nn.Module):
    """Krea2TextFusionBlock: pre-norm attention (no rotary; the refiner's takes the key-padding mask) and SwiGLU."""

    def __init__(self, compute: torch.dtype) -> None:
        super().__init__()
        self.compute = compute

    def forward(self, h, mask=None):
        h = h + self.attn(rms(h, self.norm1, NORM_EPS).to(self.compute), mask=mask).float()
        return h + self.ff(rms(h, self.norm2, NORM_EPS).to(self.compute)).float()


class Block(nn.Module):
    """Krea2TransformerBlock over the joint [text; image] sequence; `shared` are cond rows 0..5."""

    def __init__(self, compute: torch.dtype) -> None:
        super().__init__()
        self.compute = compute

    def forward(self, h, shared, cos, sin, text_rows, mask=None):
        mod = shared + self.table                                        # [6, D] float32
        prescale, preshift, pregate, postscale, postshift, postgate = mod.unbind(0)
        x = ((1 + prescale) * rms(h, self.norm1, NORM_EPS) + preshift).to(self.compute)
        h = h + pregate * self.attn(x, cos, sin, text_rows, mask).float()
        x = ((1 + postscale) * rms(h, self.norm2, NORM_EPS) + postshift).to(self.compute)
        return h + postgate * self.ff(x).float()


class DenoiserSegment(nn.Module):
    """Blocks [first, last) of the 28; segment 0 also embeds, fuses text and conditions; segment 3 projects out.

    `masked` (the default): a `text_mask` input after the text input, a static text length (`text_rows`, default
    `MASK_TEXT_ROWS`) and the key-padding mask in the refiner and every joint block. `masked=False`: the exact-length
    graphs (no mask)."""

    def __init__(self, index: int, compute: torch.dtype = torch.float16, masked: bool = True,
                 text_rows: int = MASK_TEXT_ROWS) -> None:
        super().__init__()
        if masked and not 1 <= text_rows <= DENOISER_TEXT_ROWS:
            raise ValueError(f"the masked text length must be in 1...{DENOISER_TEXT_ROWS}, not {text_rows}")
        self.index, self.compute, self.masked, self.text_rows = index, compute, masked, text_rows
        self.first, self.last = SEGMENTS[index]
        self.is_first, self.is_last = index == 0, index == len(SEGMENTS) - 1
        self.blocks = nn.ModuleList(Block(torch.float32 if i in FLOAT32_BLOCKS else compute)
                                    for i in range(self.first, self.last))
        if self.is_first:
            half = 128
            self.register_buffer("time_freqs", torch.exp(-math.log(1e4) * torch.arange(half, dtype=torch.float32) / half),
                                 persistent=False)

    @property
    def input_names(self) -> list[str]:
        mask = ["text_mask"] if self.masked else []
        if self.is_first:
            return ["latents", "text_embeds", *mask, "sigma", "rope_image"]
        return [f"image_{self.index}", f"text_{self.index}", *mask, "cond", "rope_image"]

    @property
    def output_names(self) -> list[str]:
        if self.is_last:
            return ["velocity"]
        return [f"image_{self.index + 1}", f"text_{self.index + 1}"] + (["cond"] if self.is_first else [])

    @staticmethod
    def check_config(cfg: dict, strict: bool = True) -> None:
        expected = {"in_channels": LATENT_CHANNELS, "num_layers": LAYERS, "attention_head_dim": HEAD_DIM,
                    "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS, "axes_dims_rope": list(ROPE_AXES),
                    "rope_theta": ROPE_THETA, "norm_eps": NORM_EPS, "text_hidden_dim": TEXT_WIDTH,
                    "num_text_layers": len(TEXT_TAPS), "num_layerwise_text_blocks": 2, "num_refiner_text_blocks": 2,
                    "timestep_embed_dim": 256}
        for key, value in expected.items():
            actual = cfg.get(key)
            actual = list(actual) if isinstance(actual, (list, tuple)) else actual
            if strict and actual != value:
                raise SystemExit(f"transformer config {key}={actual!r}, expected {value!r}")

    @staticmethod
    def weight_names(index: int, cfg: dict) -> list[str]:
        first, last = SEGMENTS[index]
        names = []
        if index == 0:
            names += [f"{m}.{t}" for m in ("img_in", "time_embed.linear_1", "time_embed.linear_2", "time_mod_proj",
                                           "txt_in.linear_1", "txt_in.linear_2") for t in ("weight", "bias")]
            names += ["txt_in.norm.weight", "text_fusion.projector.weight", "final_layer.scale_shift_table"]
            for kind in ("layerwise_blocks", "refiner_blocks"):
                for i in range(cfg["num_layerwise_text_blocks"] if kind == "layerwise_blocks" else cfg["num_refiner_text_blocks"]):
                    names += block_names(f"text_fusion.{kind}.{i}.")
        if index == len(SEGMENTS) - 1:
            names += ["final_layer.norm.weight", "final_layer.linear.weight", "final_layer.linear.bias"]
        for i in range(first, last):
            names += block_names(f"transformer_blocks.{i}.") + [f"transformer_blocks.{i}.scale_shift_table"]
        return names

    @classmethod
    def load(cls, checkpoint: Checkpoint, index: int, adapter: LoKr | None = None, compute=torch.float16,
             strict: bool = True, masked: bool = True, text_rows: int = MASK_TEXT_ROWS) -> "DenoiserSegment":
        cfg = checkpoint.config("transformer")
        cls.check_config(cfg, strict)
        seg = cls(index, compute, masked, text_rows)
        seg.heads, seg.kv_heads = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        seg.text_heads, seg.text_kv_heads = cfg["text_num_attention_heads"], cfg["text_num_key_value_heads"]
        seg.text_width = cfg["text_hidden_dim"]
        seg.hidden = cfg["num_attention_heads"] * cfg["attention_head_dim"]
        for block in seg.blocks:
            block.heads = seg.heads
        w = MergedWeights(checkpoint.read("transformer", cls.weight_names(index, cfg)), adapter)
        storage = torch.float16 if compute == torch.float16 else torch.float32
        if seg.is_first:
            wl = lambda n: WideLinear(w.pop(n + ".weight"), w.pop(n + ".bias"), storage, n)  # noqa: E731
            seg.img_in = wl("img_in")
            seg.time_1, seg.time_2, seg.time_mod = wl("time_embed.linear_1"), wl("time_embed.linear_2"), wl("time_mod_proj")
            seg.txt_1, seg.txt_2 = wl("txt_in.linear_1"), wl("txt_in.linear_2")
            seg.txt_norm = buffer(1.0 + w.pop("txt_in.norm.weight"))
            seg.projector = buffer(w.pop("text_fusion.projector.weight"))        # [1, 12]
            seg.final_table = buffer(w.pop("final_layer.scale_shift_table"))     # [2, D]
            seg.layerwise, seg.refiner = nn.ModuleList(), nn.ModuleList()
            for kind, modules in (("layerwise_blocks", seg.layerwise), ("refiner_blocks", seg.refiner)):
                i = 0
                while f"text_fusion.{kind}.{i}.norm1.weight" in w:
                    p = f"text_fusion.{kind}.{i}."
                    block = FusionBlock(compute)
                    block.attn = build_attention(w, p, seg.text_heads, seg.text_kv_heads, compute, rotary=False)
                    block.ff = build_ff(w, p, compute)
                    block.norm1, block.norm2 = buffer(1.0 + w.pop(p + "norm1.weight")), buffer(1.0 + w.pop(p + "norm2.weight"))
                    modules.append(block)
                    i += 1
        if seg.is_last:
            seg.final_norm = buffer(1.0 + w.pop("final_layer.norm.weight"))
            seg.final = WideLinear(w.pop("final_layer.linear.weight"), w.pop("final_layer.linear.bias"), storage,
                                   "final_layer.linear")
        for offset, block in enumerate(seg.blocks):
            p = f"transformer_blocks.{seg.first + offset}."
            block.attn = build_attention(w, p, seg.heads, seg.kv_heads, block.compute, rotary=True, storage=storage)
            block.ff = build_ff(w, p, block.compute, storage)
            block.norm1, block.norm2 = buffer(1.0 + w.pop(p + "norm1.weight")), buffer(1.0 + w.pop(p + "norm2.weight"))
            block.table = buffer(w.pop(p + "scale_shift_table"))
        if w:
            raise SystemExit(f"unused transformer tensors: {sorted(w)[:4]}")
        return seg.eval()

    def condition(self, sigma: torch.Tensor) -> torch.Tensor:
        """Krea2TimestepEmbedding + time_mod_proj: rows 0..5 shared block modulation, rows 6..7 final (scale, shift)."""
        args = (sigma.float() * 1e3)[:, None] * self.time_freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        temb = self.time_2(gelu_tanh(self.time_1(emb)))                   # [1, D]
        mod = self.time_mod(gelu_tanh(temb)).reshape(6, -1)
        return torch.cat([mod, temb + self.final_table], dim=0)

    def fuse_text(self, text_embeds: torch.Tensor, mask=None) -> torch.Tensor:
        rows = text_embeds.shape[0]
        h = text_embeds.reshape(rows, len(TEXT_TAPS), self.text_width).float()   # batch = token, sequence = tapped layer
        for block in self.layerwise:
            h = block(h)                                                   # per token: padding can't reach it
        h = (h.permute(0, 2, 1) @ self.projector[0])                       # [T, 2560]
        h = h[None]
        for block in self.refiner:
            h = block(h, mask)
        h = rms(h, self.txt_norm, NORM_EPS)
        return self.txt_2(gelu_tanh(self.txt_1(h)))                       # [1, T, D] float32

    def forward(self, a, b, c, d, e=None):
        # Masked: (a, b, text_mask, c, rope_image) = (a, b, c, d, e); unmasked: (a, b, c, rope_image) = (a, b, c, d).
        # Positional, not *args: torch.export takes each argument as one input.
        if self.masked:
            text_mask, c, rope_image = c, d, e
            valid = text_mask > 0.5                                        # [T] bool: True = a real row
            text_keys = valid[None, None, None, :]                         # Diffusers' (B, 1, 1, L) key-padding mask
        else:
            rope_image, text_keys = d, None
        if self.is_first:
            latents, text_embeds, sigma = a, b, c
            cond = self.condition(sigma)
            img = self.img_in(latents)[None]
            txt = self.fuse_text(text_embeds, text_keys)
        else:
            img, txt, cond = a[None], b[None], c
        text_rows = txt.shape[1]
        joint_keys = None
        if self.masked:
            # Image keys are always valid: cat([text mask, ones(N)]), as `Krea2Transformer2DModel.forward`.
            joint_keys = torch.cat([valid, torch.ones_like(img[0, :, 0], dtype=torch.bool)])[None, None, None, :]
        cos, sin = rope_image[None, :, 0:1, :], rope_image[None, :, 1:2, :]   # [1, N, 1, 128] against [1, N, H, 128]
        h = torch.cat([txt, img], dim=1)
        shared = cond[0:6]
        for block in self.blocks:
            h = block(h, shared, cos, sin, text_rows, joint_keys)
        txt, img = h[0, :text_rows], h[0, text_rows:]
        if self.is_last:
            scale, shift = cond[6], cond[7]
            return self.final((1 + scale) * rms(img, self.final_norm, NORM_EPS) + shift).to(torch.float16)
        if self.is_first:
            return img, txt, cond
        return img, txt

    def example(self, tokens: int = 24, text: int | None = None):
        """Example inputs and dynamic shapes. Masked: `text` rows (default the segment's `text_rows`), static, the last
        third padding; the image tokens stay dynamic. Unmasked: a dynamic text axis (2...1116)."""
        g = torch.Generator().manual_seed(self.index)
        n_dim = torch.export.Dim("image_tokens", min=4, max=MAX_IMAGE_TOKENS)
        text = text or (self.text_rows if self.masked else 8)
        rope = rotary_table(4, tokens // 4)
        if self.is_first:
            a, b = (torch.randn(tokens, LATENT_CHANNELS, generator=g).half(),
                    torch.randn(text, len(TEXT_TAPS) * self.text_width, generator=g).half())
            c = torch.tensor([0.75])
        else:
            a, b = torch.randn(tokens, self.hidden, generator=g), torch.randn(text, self.hidden, generator=g)
            c = torch.randn(CONDITION_ROWS, self.hidden, generator=g) * 0.1
        if not self.masked:
            t_dim = torch.export.Dim("text_rows", min=2, max=DENOISER_TEXT_ROWS)
            return (a, b, c, rope), ({0: n_dim}, {0: t_dim}, None, {0: n_dim})
        mask = text_mask(text - text // 3, text)
        return (a, b, mask, c, rope), ({0: n_dim}, None, None, None, {0: n_dim})


# MARK: host-side recipe pieces (the Swift driver re-implements these; fixtures pin them)

def text_mask(rows: int, length: int) -> torch.Tensor:
    """The host's `text_mask` [length] float32: 1 for the `rows` real rows (first), 0 for the padding after them."""
    if not 1 <= rows <= length:
        raise ValueError(f"{rows} conditioning rows do not fit {length}")
    mask = torch.zeros(length, dtype=torch.float32)
    mask[:rows] = 1
    return mask


def pad_rows(rows: torch.Tensor, length: int) -> torch.Tensor:
    """Real rows first, then zero rows to `length` (finite padding: the mask keeps it out of every attention)."""
    if rows.shape[0] > length:
        raise ValueError(f"{rows.shape[0]} conditioning rows do not fit {length}")
    return torch.cat([rows, rows.new_zeros(length - rows.shape[0], *rows.shape[1:])])


def rotary_table(rows: int, columns: int) -> torch.Tensor:
    """Kit's split-halves image table `[rows * columns, 2, 128]` for positions (t=0, row, column), as 0.0.x
    `KreaRotary.image`: per axis j in 0..<dim step 2, frequency 1 / 1000^(j / dim); angles in float64, rounded once."""
    freqs = [(axis, 1.0 / ROPE_THETA ** (j / dim)) for axis, dim in enumerate(ROPE_AXES) for j in range(0, dim, 2)]
    r, c = torch.meshgrid(torch.arange(rows), torch.arange(columns), indexing="ij")
    ids = torch.stack([torch.zeros_like(r), r, c], dim=-1).reshape(-1, 3).double()
    angles = torch.stack([ids[:, axis] * f for axis, f in freqs], dim=-1)
    cos, sin = torch.cos(angles).float(), torch.sin(angles).float()
    return torch.stack([torch.cat([cos, cos], -1), torch.cat([sin, sin], -1)], dim=1)


def schedule(steps: int = STEPS) -> tuple[list[float], list[float]]:
    """0.0.x `KreaSchedule` (float32): Diffusers' linspace(1, 1/steps) sigmas under the exponential shift at mu."""
    import numpy as np
    stop = 1.0 / steps
    delta = (stop - 1.0) / (steps - 1) if steps > 1 else 0.0
    shift = np.float32(math.exp(MU))
    sigmas = []
    for index in range(steps):
        base = np.float32(stop if index == steps - 1 and steps > 1 else index * delta + 1.0)
        sigmas.append(np.float32(shift / (shift + (np.float32(1.0) / base - np.float32(1.0)))))
    timesteps = [float(np.float32(np.float32(s * np.float32(1000.0)) / np.float32(1000.0))) for s in sigmas]
    return [float(s) for s in sigmas] + [0.0], timesteps


def seeded_noise(seed: int, count: int):
    """SplitMix64 + double-precision Box-Muller (0.0.x `SeededGaussianNoise`), float32, in packed-token order."""
    import numpy as np
    mask = (1 << 64) - 1
    state = seed & mask
    out = np.empty(count, dtype=np.float32)
    i = 0
    while i < count:
        values = []
        for _ in range(2):
            state = (state + 0x9E3779B97F4A7C15) & mask
            z = state
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
            values.append(z ^ (z >> 31))
        u1 = ((values[0] >> 11) + 1) / 9007199254740992.0
        u2 = (values[1] >> 11) / 9007199254740992.0
        radius = math.sqrt(-2.0 * math.log(u1))
        angle = 2.0 * math.pi * u2
        out[i] = radius * math.cos(angle)
        if i + 1 < count:
            out[i + 1] = radius * math.sin(angle)
        i += 2
    return out


def prompt_ids(tokenizer, prompt: str) -> list[int]:
    """0.0.x `KreaPrompt`: render the fixed template, tokenize without extra specials, keep at most 546 ids while
    preserving the final five."""
    ids = tokenizer(SYSTEM_PREFIX + prompt + SUFFIX, add_special_tokens=False)["input_ids"]
    if len(ids) > MAX_TOKENS:
        ids = ids[: MAX_TOKENS - TRAILING_TOKENS] + ids[-TRAILING_TOKENS:]
    return ids


# MARK: VAE decoder (fp16 storage, float32 compute)

class Conv2dFromCausal3d(nn.Module):
    """A QwenImageCausalConv3d applied to one frame with no cache: the causal front padding is zeros, so only the
    kernel's last temporal slice touches the frame. Weight stored fp16, computed float32."""

    def __init__(self, conv: nn.Conv3d, half: bool) -> None:
        super().__init__()
        kt = conv.kernel_size[0]
        if conv._padding[4] != kt - 1 or conv.stride[0] != 1 or conv.dilation != (1, 1, 1) or conv.groups != 1:
            raise SystemExit(f"unsupported causal conv {conv}")
        weight = conv.weight.detach()[:, :, -1]
        self.register_buffer("weight", weight.to(torch.float16 if half else torch.float32).contiguous())
        self.register_buffer("bias", None if conv.bias is None else conv.bias.detach().float().contiguous())
        self.stride = tuple(conv.stride[1:])
        self.padding = (conv._padding[2], conv._padding[0])         # (h, w)

    def forward(self, x, cache_x=None):
        out = F.conv2d(x[:, :, 0], self.weight.float(), self.bias, self.stride, self.padding)
        return out[:, :, None]


class HalfConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, half: bool) -> None:
        super().__init__()
        if conv.padding_mode != "zeros":
            raise SystemExit("only zero-padded convolutions are supported")
        self.register_buffer("weight", conv.weight.detach().to(torch.float16 if half else torch.float32).contiguous())
        self.register_buffer("bias", None if conv.bias is None else conv.bias.detach().float().contiguous())
        self.stride, self.padding, self.dilation, self.groups = conv.stride, conv.padding, conv.dilation, conv.groups

    def forward(self, x):
        return F.conv2d(x, self.weight.float(), self.bias, self.stride, self.padding, self.dilation, self.groups)


def single_frame(module: nn.Module, half: bool) -> nn.Module:
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage import QwenImageCausalConv3d
    for name, child in list(module.named_children()):
        if isinstance(child, QwenImageCausalConv3d):
            setattr(module, name, Conv2dFromCausal3d(child, half))
        elif isinstance(child, nn.Conv2d):
            setattr(module, name, HalfConv2d(child, half))
        else:
            single_frame(child, half)
    return module


def load_vae(checkpoint: Checkpoint):
    from diffusers import AutoencoderKLQwenImage
    vae = AutoencoderKLQwenImage.from_pretrained(checkpoint.root / "vae", torch_dtype=torch.float32).eval()
    vae.requires_grad_(False)
    return vae


class VAEDecoderGraph(nn.Module):
    """Pipeline tail: unpack 2x2 patches, `latents * std + mean`, post-quant conv, the single-frame decoder, clamp."""

    def __init__(self, vae, half_storage: bool = True) -> None:
        super().__init__()
        cfg = vae.config
        self.channels = cfg.z_dim
        self.register_buffer("mean", torch.tensor(cfg.latents_mean, dtype=torch.float32).view(-1, 1, 1))
        self.register_buffer("std", torch.tensor(cfg.latents_std, dtype=torch.float32).view(-1, 1, 1))
        self.post_quant_conv = single_frame(nn.ModuleList([copy.deepcopy(vae.post_quant_conv)]), half_storage)[0]
        self.decoder = single_frame(copy.deepcopy(vae.decoder), half_storage)

    def forward(self, latents):
        # Krea2Pipeline._unpack_latents: packed channel c * 4 + 2 * dy + dx -> (c, 2 * row + dy, 2 * column + dx).
        x = F.pixel_shuffle(latents.float().permute(2, 0, 1)[None], 2)[0]
        x = x * self.std + self.mean
        x = self.post_quant_conv(x[None, :, None])
        out = self.decoder(x)
        return torch.clamp(out, -1.0, 1.0)[0, :, 0]

    @staticmethod
    def example(rows: int = 4, columns: int = 6):
        g = torch.Generator().manual_seed(3)
        h, w = torch.export.Dim("rows", min=2, max=128), torch.export.Dim("columns", min=2, max=128)
        return (torch.randn(rows, columns, LATENT_CHANNELS, generator=g).half(),), ({0: h, 1: w},)


class Unreachable(nn.Module):
    """Stands in for a submodule the single-frame graph never calls; calling it is an export bug."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def forward(self, *args, **kwargs):
        raise RuntimeError(f"{self.name} is unreachable for one frame without a feature cache")


class VAEEncoderGraph(nn.Module):
    """Refine's head, the inverse of the decoder's tail: the single-frame encoder, quant conv, the posterior's mode,
    `(mean - latents_mean) / latents_std`, then 2x2 patches packed channel-major (`c * 4 + 2 * dy + dx`, the
    pipeline's `_pack_latents` order) as `[rows, columns, 64]` float32. Input: `[3, 16 * rows, 16 * columns]` on
    [-1, 1]. With no feature cache, each causal conv sees zeros before the frame, as `vae.encode` does for frame 0.

    Two forms are chosen for Core AI's GPU path, where the plain ones crash with dynamic rows and columns (a null
    tensor type in MPSGraph's region encoding, macOS 27.2, M5 Max, 2026-11-08; both run on the CPU): the packing is a
    reshape and permute, not `F.pixel_unshuffle`, and the normalization comes after it, with each packed channel
    `c * 4 + k` using channel `c`'s mean and std, rather than broadcasting `[16, 1, 1]` over the unpacked latents."""

    def __init__(self, vae, half_storage: bool = True) -> None:
        super().__init__()
        cfg = vae.config
        self.channels = cfg.z_dim
        # Per packed channel (c * 4 + k): channel c's mean and std.
        self.register_buffer("mean", torch.tensor(cfg.latents_mean, dtype=torch.float32).repeat_interleave(4))
        self.register_buffer("std", torch.tensor(cfg.latents_std, dtype=torch.float32).repeat_interleave(4))
        encoder = copy.deepcopy(vae.encoder)
        for module in encoder.modules():
            # A downsample3d resample's time conv runs only with a feature cache (from the second frame chunk on);
            # one frame without a cache never reaches it, and its stride-2 temporal kernel has no 2-D form.
            if getattr(module, "mode", None) == "downsample3d" and hasattr(module, "time_conv"):
                module.time_conv = Unreachable("time_conv")
        self.encoder = single_frame(encoder, half_storage)
        self.quant_conv = single_frame(nn.ModuleList([copy.deepcopy(vae.quant_conv)]), half_storage)[0]

    def forward(self, image):
        moments = self.quant_conv(self.encoder(image.float()[None, :, None]))
        mode = moments[0, : self.channels, 0]                             # DiagonalGaussianDistribution.mode()
        c, rows, columns = mode.shape[0], mode.shape[1] // 2, mode.shape[2] // 2
        packed = mode.reshape(c, rows, 2, columns, 2).permute(1, 3, 0, 2, 4).reshape(rows, columns, c * 4)
        return (packed - self.mean) / self.std

    @staticmethod
    def example(rows: int = 4, columns: int = 6):
        g = torch.Generator().manual_seed(4)
        h, w = torch.export.Dim("rows", min=2, max=128), torch.export.Dim("columns", min=2, max=128)
        image = (torch.rand(3, PIXELS_PER_TOKEN * rows, PIXELS_PER_TOKEN * columns, generator=g) * 2 - 1).half()
        return (image,), ({1: PIXELS_PER_TOKEN * h, 2: PIXELS_PER_TOKEN * w},)

"""FLUX.2 [klein] 4B export graphs for the AetherFlux2 bundle (development tooling only).

Written from the upstream Diffusers 0.37.1 (`Flux2Transformer2DModel`, `AutoencoderKLFlux2`, `Flux2KleinPipeline`)
and Transformers 4.57.3 (`Qwen3Model`) sources against the bundle contract in docs/flux2-klein.md. Every graph reads
the pinned checkpoint's safetensors directly; nothing here imports AetherKit or the superseded prototype.

Functions (names, dtypes and dynamic axes are the bundle contract; Kit checks them before every call):

    textEncoder/encode   input_ids, position_ids, attention_mask int32 [1, T] -> text_embeds fp16 [T, 7680]
    denoiser0/segment    latents fp16 [N, 128], text_embeds fp16 [T, 7680], sigma f32 [1],
                         rope_text f32 [T, 2, 128], rope_image f32 [N, 2, 128] -> image_1, text_1, cond f32
    denoiser1/segment    image_1 f32 [N, 3072], text_1 f32 [T, 3072], cond f32 [17, 3072], ropes -> image_2, text_2
    denoiser2/segment    image_2, text_2, cond, ropes -> velocity fp16 [N, 128]
    vaeDecoder/decode    latents fp16 [h, w, 128] (batch-normalized) -> image f32 [3, 16h, 16w]
    vaeEncoder/encode    reference_pixels f32 [3, H, W] -> reference_latents f32 [H/16, W/16, 128]

Precision follows the bundle's recorded recipe: the text encoder and denoiser matmuls and attention run in fp16 with
int8 per-channel weights (single-block `proj_out` stays fp16); the denoiser keeps its residual streams, LayerNorms,
modulation and rotary products in float32 and casts to fp16 only around matmuls and attention; both VAE graphs store
fp16 weights and compute in float32.

Rotary layout: Kit's tables are split halves (`[n, 0, :]` cos, `[n, 1, :]` sin; pair i rotates dims i and i + 64),
while Diffusers rotates interleaved pairs (2i, 2i + 1) with the same frequency order. The export permutes every
query/key projection row and q/k norm gain within each head (`HALF_SPLIT`) so the split-halves rotation computes the
same attention scores; values and outputs are untouched.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

TEXT_LAYERS = 27                  # Qwen3-4B cut after layer 27: taps are hidden_states[9], [18], [27]
TEXT_TAPS = (9, 18, 27)
DOUBLE_BLOCKS = 5
SINGLE_BLOCKS = 20
# Denoiser segments as [first, last) block indices over the 5 double + 20 single blocks (docs: 0-8, 8-16, 16-25).
SEGMENTS = ((0, 8), (8, 16), (16, 25))
HIDDEN = 3072                     # 24 heads x 128
HEADS = 24
HEAD_DIM = 128
LATENT_CHANNELS = 128
CONDITION_ROWS = 17               # img 6 + txt 6 + single 3 + output (scale, shift) 2
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
        index = directory / "model.safetensors.index.json"
        if index.exists():
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


def half_split_rows(weight: torch.Tensor, heads: int) -> torch.Tensor:
    """Reorder each head's output rows from interleaved rotary pairs to split halves."""
    return weight.reshape(heads, HEAD_DIM, -1)[:, HALF_SPLIT, :].reshape(heads * HEAD_DIM, -1)


def linear(weight: torch.Tensor, dtype: torch.dtype) -> nn.Linear:
    out_features, in_features = weight.shape
    module = nn.Linear(in_features, out_features, bias=False, device="meta")
    module.weight = nn.Parameter(weight.to(dtype).contiguous(), requires_grad=False)
    return module


def gain(weight: torch.Tensor, dtype: torch.dtype) -> nn.Parameter:
    return nn.Parameter(weight.to(dtype).contiguous(), requires_grad=False)


# MARK: text encoder (Qwen3-4B, fp16)

def rms_norm_fp16(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3RMSNorm: float32 statistics, rounded to the input dtype, then the gain in that dtype."""
    x32 = x.to(torch.float32)
    return weight * (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class TextLayer(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.eps = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["rms_norm_eps"]

    def forward(self, h, cos, sin, mask):
        length = h.shape[1]
        x = rms_norm_fp16(h, self.input_norm, self.eps)
        q, k, v = self.qkv(x).split([self.heads * HEAD_DIM, self.kv_heads * HEAD_DIM, self.kv_heads * HEAD_DIM], dim=-1)
        q = rms_norm_fp16(q.unflatten(-1, (self.heads, HEAD_DIM)), self.q_norm, self.eps).transpose(1, 2)
        k = rms_norm_fp16(k.unflatten(-1, (self.kv_heads, HEAD_DIM)), self.k_norm, self.eps).transpose(1, 2)
        v = v.unflatten(-1, (self.kv_heads, HEAD_DIM)).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=HEAD_DIM ** -0.5, enable_gqa=True)
        h = h + self.o(attn.transpose(1, 2).reshape(1, length, self.heads * HEAD_DIM))
        x = rms_norm_fp16(h, self.post_norm, self.eps)
        return h + self.down(F.silu(self.gate(x)) * self.up(x))


class TextEncoderGraph(nn.Module):
    """Qwen3 layers 0..26 with the pipeline's taps; causal plus padding mask from the inputs, fp16 throughout.

    q/k/v share one projection (one dequantized matmul instead of three); gate, up, down and o stay separate.
    """

    def __init__(self, cfg: dict, layers: int = TEXT_LAYERS) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(TextLayer(cfg) for _ in range(layers))
        # Transformers' Qwen3RotaryEmbedding: float32 exponent, float32 product with the positions.
        dim = cfg.get("head_dim", HEAD_DIM)
        self.register_buffer("inv_freq", 1.0 / (cfg["rope_theta"] ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)),
                             persistent=False)

    @classmethod
    def load(cls, checkpoint: Checkpoint, dtype=torch.float16, layers: int = TEXT_LAYERS) -> "TextEncoderGraph":
        cfg = checkpoint.config("text_encoder")
        graph = cls(cfg, layers)
        names = ["model.embed_tokens.weight"]
        parts = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj",
                 "mlp.up_proj", "mlp.down_proj")
        for i in range(layers):
            p = f"model.layers.{i}."
            names += [p + n + ".weight" for n in parts]
            names += [p + n + ".weight" for n in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm",
                                                  "self_attn.k_norm")]
        w = checkpoint.read("text_encoder", names)
        embed = w.pop("model.embed_tokens.weight")
        graph.embed = nn.Embedding(embed.shape[0], embed.shape[1], device="meta")
        graph.embed.weight = nn.Parameter(embed.to(dtype), requires_grad=False)
        for i, layer in enumerate(graph.layers):
            p = f"model.layers.{i}."
            layer.qkv = linear(torch.cat([w.pop(p + f"self_attn.{n}_proj.weight") for n in "qkv"]), dtype)
            layer.o = linear(w.pop(p + "self_attn.o_proj.weight"), dtype)
            layer.gate = linear(w.pop(p + "mlp.gate_proj.weight"), dtype)
            layer.up = linear(w.pop(p + "mlp.up_proj.weight"), dtype)
            layer.down = linear(w.pop(p + "mlp.down_proj.weight"), dtype)
            layer.input_norm = gain(w.pop(p + "input_layernorm.weight"), dtype)
            layer.post_norm = gain(w.pop(p + "post_attention_layernorm.weight"), dtype)
            layer.q_norm = gain(w.pop(p + "self_attn.q_norm.weight"), dtype)
            layer.k_norm = gain(w.pop(p + "self_attn.k_norm.weight"), dtype)
        return graph.eval()

    def forward(self, input_ids, position_ids, attention_mask):
        h = self.embed(input_ids)
        positions = position_ids[0]
        angles = positions.to(torch.float32)[:, None] * self.inv_freq[None, :]
        emb = torch.cat((angles, angles), dim=-1)
        cos, sin = emb.cos().to(h.dtype), emb.sin().to(h.dtype)
        # Transformers' causal mask with padding: key k is visible to query q when k <= q and the key is real.
        valid = attention_mask[0] > 0
        mask = torch.logical_not(positions[None, :] > positions[:, None]) & valid[None, :]
        taps = []
        for index, layer in enumerate(self.layers, start=1):
            h = layer(h, cos, sin, mask)
            if index in TEXT_TAPS:
                taps.append(h)
        return torch.cat(taps, dim=-1)[0]

    def example(self, length: int = 16):
        ids = torch.full((1, length), 151643, dtype=torch.int32)
        ids[0, : length // 2] = torch.arange(1000, 1000 + length // 2, dtype=torch.int32)
        mask = torch.zeros((1, length), dtype=torch.int32)
        mask[0, : length // 2] = 1
        positions = torch.arange(length, dtype=torch.int32)[None]
        tokens = torch.export.Dim("tokens", min=2, max=1024)
        return (ids, positions, mask), ({1: tokens}, {1: tokens}, {1: tokens})


# MARK: denoiser (fp16 matmuls, float32 residuals)

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (1 + scale) * F.layer_norm(x, (x.shape[-1],), eps=1e-6) + shift


def qk_norm(x16: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """torch.nn.RMSNorm(128, eps=1e-6) in float32; the result stays float32 for the rotary product."""
    x = x16.to(torch.float32)
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * weight.to(torch.float32)


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


def attention(q16: torch.Tensor, k16: torch.Tensor, v16: torch.Tensor) -> torch.Tensor:
    """[1, S, heads, 128] fp16 -> [1, S, 3072] fp16 (Diffusers' default scale 1/sqrt(128))."""
    out = F.scaled_dot_product_attention(q16.transpose(1, 2), k16.transpose(1, 2), v16.transpose(1, 2))
    return out.transpose(1, 2).flatten(2)


class DoubleBlock(nn.Module):
    """Flux2TransformerBlock with per-stream fused q/k/v; `mods` are the 12 img+txt rows of `cond`."""

    def __init__(self, heads: int, compute: torch.dtype) -> None:
        super().__init__()
        self.heads, self.compute = heads, compute

    def forward(self, img, txt, mods, cos, sin):
        (shift, scale, gate, shift2, scale2, gate2) = mods[0:6]
        (c_shift, c_scale, c_gate, c_shift2, c_scale2, c_gate2) = mods[6:12]
        heads = (3, self.heads, HEAD_DIM)
        qi, ki, vi = self.img_qkv(modulate(img, shift, scale).to(self.compute)).unflatten(-1, heads).unbind(2)
        qt, kt, vt = self.txt_qkv(modulate(txt, c_shift, c_scale).to(self.compute)).unflatten(-1, heads).unbind(2)
        q = torch.cat([qk_norm(qt, self.txt_q_norm), qk_norm(qi, self.img_q_norm)], dim=1)
        k = torch.cat([qk_norm(kt, self.txt_k_norm), qk_norm(ki, self.img_k_norm)], dim=1)
        q, k = rope(q, cos, sin).to(self.compute), rope(k, cos, sin).to(self.compute)
        out = attention(q, k, torch.cat([vt, vi], dim=1))
        text_length = txt.shape[1]
        out_t, out_i = out[:, :text_length], out[:, text_length:]
        img = img + gate * self.img_out(out_i).to(torch.float32)
        img = img + gate2 * self.img_mlp_out(swiglu(self.img_mlp_in(modulate(img, shift2, scale2).to(self.compute)))).to(torch.float32)
        txt = txt + c_gate * self.txt_out(out_t).to(torch.float32)
        txt = txt + c_gate2 * self.txt_mlp_out(swiglu(self.txt_mlp_in(modulate(txt, c_shift2, c_scale2).to(self.compute)))).to(torch.float32)
        return img, txt


def swiglu(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return F.silu(a) * b


class SingleBlock(nn.Module):
    """Flux2SingleTransformerBlock over [text; image]; `mods` are the 3 single rows of `cond`."""

    def __init__(self, heads: int, compute: torch.dtype) -> None:
        super().__init__()
        self.heads, self.compute = heads, compute

    def forward(self, h, mods, cos, sin):
        shift, scale, gate = mods[12], mods[13], mods[14]
        x = self.qkv_mlp(modulate(h, shift, scale).to(self.compute))
        width = 3 * self.heads * HEAD_DIM
        qkv, mlp = x.split([width, x.shape[-1] - width], dim=-1)
        q, k, v = qkv.unflatten(-1, (3, self.heads, HEAD_DIM)).unbind(2)
        q = rope(qk_norm(q, self.q_norm), cos, sin).to(self.compute)
        k = rope(qk_norm(k, self.k_norm), cos, sin).to(self.compute)
        out = self.proj_out(torch.cat([attention(q, k, v), swiglu(mlp)], dim=-1))
        return h + gate * out.to(torch.float32)


class DenoiserSegment(nn.Module):
    """Blocks [first, last) of the 25; segment 0 also embeds and conditions, the last also projects out."""

    def __init__(self, index: int, heads: int = HEADS, compute: torch.dtype = torch.float16) -> None:
        super().__init__()
        self.index, self.heads, self.hidden, self.compute = index, heads, heads * HEAD_DIM, compute
        self.first, self.last = SEGMENTS[index]
        self.doubles = nn.ModuleDict({str(i): DoubleBlock(heads, compute) for i in range(self.first, min(self.last, DOUBLE_BLOCKS))})
        self.singles = nn.ModuleDict({str(i - DOUBLE_BLOCKS): SingleBlock(heads, compute)
                                      for i in range(max(self.first, DOUBLE_BLOCKS), self.last)})
        self.is_first, self.is_last = index == 0, index == len(SEGMENTS) - 1
        if self.is_first:
            # Diffusers' Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0) frequencies, float32.
            half = 128
            exponent = -math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half
            self.register_buffer("time_freqs", torch.exp(exponent), persistent=False)

    # Inputs and outputs by function: the names are the bundle's tensor bindings.
    @property
    def input_names(self) -> list[str]:
        if self.is_first:
            return ["latents", "text_embeds", "sigma", "rope_text", "rope_image"]
        return [f"image_{self.index}", f"text_{self.index}", "cond", "rope_text", "rope_image"]

    @property
    def output_names(self) -> list[str]:
        if self.is_last:
            return ["velocity"]
        return [f"image_{self.index + 1}", f"text_{self.index + 1}"] + (["cond"] if self.is_first else [])

    @classmethod
    def load(cls, checkpoint: Checkpoint, index: int, dtype=torch.float16) -> "DenoiserSegment":
        cfg = checkpoint.config("transformer")
        # Diffusers' constructor defaults stand in for absent keys; `num_attention_heads` may be smaller only in the
        # random-weight graph tests.
        expected = {"num_layers": DOUBLE_BLOCKS, "num_single_layers": SINGLE_BLOCKS, "attention_head_dim": HEAD_DIM,
                    "joint_attention_dim": 7680, "in_channels": LATENT_CHANNELS, "patch_size": 1, "mlp_ratio": 3.0,
                    "axes_dims_rope": [32, 32, 32, 32], "rope_theta": 2000, "guidance_embeds": False,
                    "timestep_guidance_channels": 256, "eps": 1e-6, "out_channels": None}
        defaults = {"patch_size": 1, "mlp_ratio": 3.0, "axes_dims_rope": [32, 32, 32, 32], "rope_theta": 2000,
                    "timestep_guidance_channels": 256, "eps": 1e-6, "out_channels": None}
        for key, value in expected.items():
            actual = cfg.get(key, defaults.get(key, "<absent>"))
            if (list(actual) if isinstance(actual, (list, tuple)) else actual) != value:
                raise SystemExit(f"transformer config {key}={actual!r}, expected {value!r}")
        seg = cls(index, cfg["num_attention_heads"], dtype)
        seg.text_width = cfg["joint_attention_dim"]
        names: list[str] = []
        if seg.is_first:
            names += ["time_guidance_embed.timestep_embedder.linear_1.weight",
                      "time_guidance_embed.timestep_embedder.linear_2.weight",
                      "double_stream_modulation_img.linear.weight", "double_stream_modulation_txt.linear.weight",
                      "single_stream_modulation.linear.weight", "norm_out.linear.weight",
                      "x_embedder.weight", "context_embedder.weight"]
        if seg.is_last:
            names += ["proj_out.weight"]
        attn = ("to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
                "norm_q", "norm_k", "norm_added_q", "norm_added_k")
        for i in seg.doubles:
            p = f"transformer_blocks.{i}."
            names += [p + f"attn.{n}.weight" for n in attn]
            names += [p + f"{ff}.{n}.weight" for ff in ("ff", "ff_context") for n in ("linear_in", "linear_out")]
        for i in seg.singles:
            p = f"single_transformer_blocks.{i}.attn."
            names += [p + n + ".weight" for n in ("to_qkv_mlp_proj", "to_out", "norm_q", "norm_k")]
        w = checkpoint.read("transformer", names)

        heads, hidden = seg.heads, seg.hidden

        def qk(name: str) -> torch.Tensor:
            return half_split_rows(w.pop(name), heads)

        def norm(name: str) -> nn.Parameter:
            return gain(w.pop(name)[HALF_SPLIT], dtype)

        if seg.is_first:
            seg.time_1 = linear(w.pop("time_guidance_embed.timestep_embedder.linear_1.weight"), dtype)
            seg.time_2 = linear(w.pop("time_guidance_embed.timestep_embedder.linear_2.weight"), dtype)
            seg.mod_img = linear(w.pop("double_stream_modulation_img.linear.weight"), dtype)
            seg.mod_txt = linear(w.pop("double_stream_modulation_txt.linear.weight"), dtype)
            seg.mod_single = linear(w.pop("single_stream_modulation.linear.weight"), dtype)
            # AdaLayerNormContinuous chunks (scale, shift); cond keeps that order in rows 15, 16.
            seg.mod_out = linear(w.pop("norm_out.linear.weight"), dtype)
            seg.img_in = linear(w.pop("x_embedder.weight"), dtype)
            seg.txt_in = linear(w.pop("context_embedder.weight"), dtype)
        if seg.is_last:
            seg.final = linear(w.pop("proj_out.weight"), dtype)
        for i, block in seg.doubles.items():
            p = f"transformer_blocks.{i}."
            block.img_qkv = linear(torch.cat([qk(p + "attn.to_q.weight"), qk(p + "attn.to_k.weight"),
                                              w.pop(p + "attn.to_v.weight")]), dtype)
            block.txt_qkv = linear(torch.cat([qk(p + "attn.add_q_proj.weight"), qk(p + "attn.add_k_proj.weight"),
                                              w.pop(p + "attn.add_v_proj.weight")]), dtype)
            block.img_out = linear(w.pop(p + "attn.to_out.0.weight"), dtype)
            block.txt_out = linear(w.pop(p + "attn.to_add_out.weight"), dtype)
            block.img_q_norm, block.img_k_norm = norm(p + "attn.norm_q.weight"), norm(p + "attn.norm_k.weight")
            block.txt_q_norm, block.txt_k_norm = norm(p + "attn.norm_added_q.weight"), norm(p + "attn.norm_added_k.weight")
            block.img_mlp_in = linear(w.pop(p + "ff.linear_in.weight"), dtype)
            block.img_mlp_out = linear(w.pop(p + "ff.linear_out.weight"), dtype)
            block.txt_mlp_in = linear(w.pop(p + "ff_context.linear_in.weight"), dtype)
            block.txt_mlp_out = linear(w.pop(p + "ff_context.linear_out.weight"), dtype)
        for i, block in seg.singles.items():
            p = f"single_transformer_blocks.{i}.attn."
            fused = w.pop(p + "to_qkv_mlp_proj.weight")
            q, k, rest = fused[:hidden], fused[hidden:2 * hidden], fused[2 * hidden:]
            block.qkv_mlp = linear(torch.cat([half_split_rows(q, heads), half_split_rows(k, heads), rest]), dtype)
            block.proj_out = linear(w.pop(p + "to_out.weight"), dtype)
            block.q_norm, block.k_norm = norm(p + "norm_q.weight"), norm(p + "norm_k.weight")
        if w:
            raise SystemExit(f"unused transformer tensors: {sorted(w)[:4]}")
        return seg.eval()

    def condition(self, sigma: torch.Tensor) -> torch.Tensor:
        t = sigma * 1000.0
        angles = t[:, None] * self.time_freqs[None, :]
        projection = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1).to(self.compute)
        temb = self.time_2(F.silu(self.time_1(projection)))
        s = F.silu(temb)
        rows = [self.mod_img(s).reshape(6, self.hidden), self.mod_txt(s).reshape(6, self.hidden),
                self.mod_single(s).reshape(3, self.hidden), self.mod_out(s).reshape(2, self.hidden)]
        return torch.cat(rows, dim=0).to(torch.float32)

    def forward(self, a, b, c, rope_text, rope_image):
        if self.is_first:
            latents, text_embeds, sigma = a, b, c
            cond = self.condition(sigma)
            img = self.img_in(latents)[None].to(torch.float32)
            txt = self.txt_in(text_embeds)[None].to(torch.float32)
        else:
            img, txt, cond = a[None], b[None], c
        mods = cond[:, None, :].unbind(0)      # 17 rows of [1, 3072], broadcast over [1, S, 3072]
        table = torch.cat([rope_text, rope_image], dim=0)
        cos, sin = table[:, 0:1, :], table[:, 1:2, :]    # [S, 1, 128] against [1, S, heads, 128]
        for block in self.doubles.values():
            img, txt = block(img, txt, mods, cos, sin)
        text_length = txt.shape[1]
        if self.singles:
            h = torch.cat([txt, img], dim=1)
            for block in self.singles.values():
                h = block(h, mods, cos, sin)
            txt, img = h[:, :text_length], h[:, text_length:]
        if self.is_last:
            scale, shift = mods[15], mods[16]
            return self.final(((1 + scale) * F.layer_norm(img, (self.hidden,), eps=1e-6) + shift).to(self.compute))[0]
        if self.is_first:
            return img[0], txt[0], cond
        return img[0], txt[0]

    def example(self, tokens: int = 24, text: int = 8):
        """Example inputs (a 4 x 6 grid) and the dynamic axes N (image tokens) and T (text rows)."""
        g = torch.Generator().manual_seed(self.index)
        n_dim = torch.export.Dim("image_tokens", min=2, max=8192)
        t_dim = torch.export.Dim("text_tokens", min=2, max=1024)
        rope_text = rotary_table(text_ids(text))
        rope_image = rotary_table(grid_ids(4, tokens // 4))
        if self.is_first:
            inputs = (torch.randn(tokens, LATENT_CHANNELS, generator=g).to(self.compute),
                      torch.randn(text, self.text_width, generator=g).to(self.compute),
                      torch.tensor([0.75], dtype=torch.float32), rope_text, rope_image)
            dynamic = ({0: n_dim}, {0: t_dim}, None, {0: t_dim}, {0: n_dim})
        else:
            inputs = (torch.randn(tokens, self.hidden, generator=g), torch.randn(text, self.hidden, generator=g),
                      torch.randn(CONDITION_ROWS, self.hidden, generator=g) * 0.1, rope_text, rope_image)
            dynamic = ({0: n_dim}, {0: t_dim}, None, {0: t_dim}, {0: n_dim})
        return inputs, dynamic


def text_ids(length: int) -> torch.Tensor:
    return torch.stack([torch.zeros(length), torch.zeros(length), torch.zeros(length), torch.arange(length)], dim=-1)


def grid_ids(rows: int, columns: int, t: int = 0) -> torch.Tensor:
    r, c = torch.meshgrid(torch.arange(rows), torch.arange(columns), indexing="ij")
    return torch.stack([torch.full_like(r, t), r, c, torch.zeros_like(r)], dim=-1).reshape(-1, 4).double()


def rotary_table(ids: torch.Tensor, axes=(32, 32, 32, 32), theta: float = 2000.0) -> torch.Tensor:
    """Kit's `Flux2Rotary.table`: split-halves cos/sin rows `[n, 2, 128]`, angles in float64, rounded once."""
    freqs = [(axis, 1.0 / theta ** (j / dim)) for axis, dim in enumerate(axes) for j in range(0, dim, 2)]
    angles = torch.stack([ids[:, axis].double() * f for axis, f in freqs], dim=-1)     # [n, 64]
    cos, sin = torch.cos(angles).float(), torch.sin(angles).float()
    return torch.stack([torch.cat([cos, cos], -1), torch.cat([sin, sin], -1)], dim=1)


# MARK: VAE (fp16 storage, float32 compute)

class HalfStorage(nn.Module):
    """A conv/linear whose weight is stored as fp16 and widened to float32 for compute; bias stays float32."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.kind = type(module).__name__
        self.register_buffer("weight16", module.weight.detach().to(torch.float16))
        self.register_buffer("bias", None if module.bias is None else module.bias.detach().to(torch.float32))
        for name in ("stride", "padding", "dilation", "groups", "padding_mode"):
            if hasattr(module, name):
                setattr(self, name, getattr(module, name))

    def forward(self, x):
        w = self.weight16.to(torch.float32)
        if self.kind == "Conv2d":
            if self.padding_mode != "zeros":
                raise ValueError("only zero-padded convolutions are supported")
            return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return F.linear(x, w, self.bias)


def store_half(module: nn.Module) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            setattr(module, name, HalfStorage(child))
        else:
            store_half(child)
    return module


def load_vae(checkpoint: Checkpoint):
    from diffusers import AutoencoderKLFlux2
    vae = AutoencoderKLFlux2.from_pretrained(checkpoint.root / "vae", torch_dtype=torch.float32).eval()
    vae.requires_grad_(False)
    return vae


class VAEDecoderGraph(nn.Module):
    """Pipeline tail: batch-norm denormalization, 2x2 unpatchify, post-quant conv and the decoder."""

    def __init__(self, vae, half_storage: bool = True) -> None:
        super().__init__()
        bn = vae.bn
        self.register_buffer("mean", bn.running_mean.detach().to(torch.float32).clone())
        self.register_buffer("std", torch.sqrt(bn.running_var.detach().to(torch.float32) + vae.config.batch_norm_eps))
        self.post_quant_conv = copy.deepcopy(vae.post_quant_conv)
        self.decoder = copy.deepcopy(vae.decoder)
        if half_storage:
            store_half(self)

    def forward(self, latents):
        x = latents.to(torch.float32) * self.std + self.mean           # [h, w, 128]
        x = F.pixel_shuffle(x.permute(2, 0, 1)[None], 2)                  # Diffusers' _unpatchify_latents
        return self.decoder(self.post_quant_conv(x))[0]

    @staticmethod
    def example(rows: int = 4, columns: int = 6):
        g = torch.Generator().manual_seed(3)
        h, w = torch.export.Dim("rows", min=2, max=256), torch.export.Dim("columns", min=2, max=256)
        return (torch.randn(rows, columns, LATENT_CHANNELS, generator=g).half(),), ({0: h, 1: w},)


class VAEEncoderGraph(nn.Module):
    """Pipeline `_encode_vae_image`: encoder, quant conv, posterior mode, 2x2 patchify, batch-norm normalization."""

    def __init__(self, vae, half_storage: bool = True) -> None:
        super().__init__()
        bn = vae.bn
        self.register_buffer("mean", bn.running_mean.detach().to(torch.float32).clone().view(1, -1, 1, 1))
        self.register_buffer("std", torch.sqrt(bn.running_var.detach().to(torch.float32) + vae.config.batch_norm_eps).view(1, -1, 1, 1))
        self.encoder = copy.deepcopy(vae.encoder)
        self.quant_conv = copy.deepcopy(vae.quant_conv)
        self.latent_channels = vae.config.latent_channels
        if half_storage:
            store_half(self)

    def forward(self, reference_pixels):
        moments = self.quant_conv(self.encoder(reference_pixels[None]))
        mean = moments[:, : self.latent_channels]                         # DiagonalGaussianDistribution.mode()
        c, h, w = mean.shape[1], mean.shape[2] // 2, mean.shape[3] // 2
        packed = mean.reshape(1, c, h, 2, w, 2).permute(0, 1, 3, 5, 2, 4).reshape(1, c * 4, h, w)
        return ((packed - self.mean) / self.std)[0].permute(1, 2, 0)

    @staticmethod
    def example(rows: int = 4, columns: int = 6):
        g = torch.Generator().manual_seed(4)
        h, w = torch.export.Dim("rows", min=2, max=64), torch.export.Dim("columns", min=2, max=64)
        return ((torch.rand(3, 16 * rows, 16 * columns, generator=g) * 2 - 1),), ({1: 16 * h, 2: 16 * w},)

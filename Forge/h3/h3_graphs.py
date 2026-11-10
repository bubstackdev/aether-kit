"""MiniMax H3 (FL2VA + the Turbo 8-step adapter) as Aether-Kit export graphs, for Forge/h3/export_h3.py.

The graph split is the one the 0.0.x prototype qualified (its `scripts/h3_export_lib.py` and `export-h3-profile.py`),
re-expressed on Aether-Kit's bundle contract:

- **Weights as inputs** (MANIFEST.md §5.8): ONE transformer-block graph serves all 50 blocks, ONE Qwen3-VL decoder-layer
  graph all 50 conditioner layers, and the text refiner and vision tower take theirs the same way. Every parameter is
  lifted to a positional input `w0, w1, …` in `named_parameters()` order (`Lifted`), and the exporter writes each
  block's tensors contiguously into bundle weight files with a `WeightTable` naming each input's extent. The prototype
  bound byte ranges of the Hugging Face cache in place; a bundle owns its bytes, so an install verifies them once and
  the runtime reads them with no re-hashing.
- **Dynamic Turbo LoRA:** the block graph carries the adapter's rank-128 LoRA as extra weight inputs (PEFT's own
  forward), so the base and adapter bytes stay unmerged and unchanged.
- **Host tables:** each block's AdaLN modulation `[6, 3k, D]` is precomputed per schedule step (`CachedModulation`
  replaces `adaln_proj`), as are the Qwen3-VL text rotary tables; the host binds them per call.
- Everything else is a plain graph with its weights baked in: the heads, the pack projection, the bf16 window-tile
  video decoder, the tiled first-frame encoder (one tile graph and a per-canvas tail), and the audio decoder at a fixed
  latent count (the dynamic one aborts in MPSGraph on the GPU, prototype E1 2026-10-19).

The modules come from the pinned Diffusers (Forge/pyproject.toml, v0.42.1, which carries MiniMax H3) and
Transformers; the checkpoint key conversion is Diffusers' own `scripts/convert_minimax_h3_to_diffusers.py` at that
commit, loaded from its checkout.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The constants and the frame grid live in the standard-library module the manifest tools share.
from profiles_h3 import (ADAPTER_FILE, ADAPTER_REPO, ADAPTER_REVISION, ADAPTER_SHA256, AUDIO_CHANNELS,  # noqa: E402,F401
                         AUDIO_HOP, BLOCKS, CARDINALITY, CONDITIONER_LAYERS, DIFFUSERS_COMMIT, ENCODE_SEED,
                         FL2VA_CONDITION_TIMESTEP, HEAD_DIM, HEADS, HIDDEN, LORA_ALPHA, LORA_RANK, LORA_TARGETS,
                         PIXEL_MEAN, PIXEL_STD, RECIPES, REPO, REVISION, ROWS, SAMPLE_RATE, TAP, TEMB_DIM, TEXT_DIM,
                         TEXT_TOKENS, VIDEO_TILE, WINDOW_FRAMES, WINDOW_LATENTS, WINDOW_STRIDE, Geometry)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


# MARK: checkpoint access

def load_converter(path: Path | None):
    """Diffusers' `convert_minimax_h3_to_diffusers.py` at the pinned commit: `path`, or uv's checkout of the commit."""
    if path is None:
        hits = sorted((Path.home() / ".cache/uv").glob(f"git-v*/checkouts/*/{DIFFUSERS_COMMIT[:9]}/scripts/"
                                                         "convert_minimax_h3_to_diffusers.py"))
        assert hits, "no Diffusers checkout at the pinned commit; pass --converter <diffusers>/scripts/…py"
        path = hits[0]
    spec = importlib.util.spec_from_file_location("convert_minimax_h3_to_diffusers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


class Checkpoint:
    """Streams converted transformer weights from the raw FL2VA shards (the converter's own key mapping and the
    per-head QKV reorder, so nothing is transcribed)."""

    def __init__(self, snapshot: Path, converter):
        self.directory = Path(snapshot) / "FL2VA/transformer"
        self.config = json.loads((self.directory / "config.json").read_text())
        self.index = json.loads((self.directory / "model.safetensors.index.json").read_text())["weight_map"]
        self.converter = converter

    def converted(self, predicate) -> dict:
        from safetensors import safe_open
        weights = {}
        for name in sorted(n for n in self.index if predicate(n)):
            with safe_open(self.directory / self.index[name], framework="pt", device="cpu") as f:
                value = f.get_tensor(name)
            if name.endswith(".attn.qkv_proj.weight"):
                value = self.converter.reorder_interleaved_qkv(value, self.config["num_attention_heads"],
                                                              self.config["attention_head_dim"])
            for key, tensor in self.converter.convert_transformer_key(name, value, self.config):
                weights[key] = tensor
        return weights


def sharded(directory: Path, prefix: str) -> dict:
    """Every tensor under `prefix` of a sharded safetensors directory, keys without the prefix."""
    from safetensors import safe_open
    index = json.loads((Path(directory) / "model.safetensors.index.json").read_text())["weight_map"]
    result = {}
    for name in sorted(n for n in index if n.startswith(prefix)):
        with safe_open(Path(directory) / index[name], framework="pt", device="cpu") as f:
            result[name.removeprefix(prefix)] = f.get_tensor(name)
    return result


def meta_model(**config):
    from diffusers import MiniMaxH3Transformer3DModel
    with torch.device("meta"):
        return MiniMaxH3Transformer3DModel(**config)


def adapter_path(check: bool = True) -> Path:
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(ADAPTER_REPO, ADAPTER_FILE, revision=ADAPTER_REVISION, local_files_only=True)).resolve()
    assert not check or sha256_file(path) == ADAPTER_SHA256, "the Turbo adapter's hash differs; refusing"
    return path


def load_turbo(model, adapter: Path, predicate) -> None:
    """The Turbo LoRA for the keys `predicate` accepts, loaded through PEFT exactly as the 0.0.x oracles did."""
    from safetensors import safe_open
    weights, alphas = {}, {}
    with safe_open(adapter, framework="pt", device="cpu") as source:
        for key in source.keys():
            if predicate(key):
                weights["transformer." + key.replace(".default.weight", ".weight")] = source.get_tensor(key)
                if ".lora_A." in key:
                    alphas["transformer." + key.split(".lora_A.")[0] + ".alpha"] = LORA_ALPHA
    model.load_lora_adapter(weights, prefix="transformer", adapter_name="turbo", network_alphas=alphas)


def zero_turbo(model, block: int = 0) -> None:
    """Zero LoRA slots on one block (the body graph's shape without the adapter's bytes)."""
    adapter, alphas = {}, {}
    b = model.transformer_blocks[block]
    for t in LORA_TARGETS:
        linear = b.get_submodule(t)
        adapter[f"transformer.transformer_blocks.{block}.{t}.lora_A.weight"] = torch.zeros(LORA_RANK, linear.in_features, dtype=torch.bfloat16)
        adapter[f"transformer.transformer_blocks.{block}.{t}.lora_B.weight"] = torch.zeros(linear.out_features, LORA_RANK, dtype=torch.bfloat16)
        alphas[f"transformer.transformer_blocks.{block}.{t}.alpha"] = LORA_ALPHA
    model.load_lora_adapter(adapter, prefix="transformer", adapter_name="turbo", network_alphas=alphas)


# MARK: graph modules

class Lifted(nn.Module):
    """`module` with every parameter lifted to a positional input, in `named_parameters()` order (w0, w1, …)."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.names = [name for name, _ in module.named_parameters()]

    def forward(self, *args):
        count = len(args) - len(self.names)
        inputs, weights = args[:count], args[count:]
        return torch.func.functional_call(self.module, dict(zip(self.names, weights)), inputs)


def lift(module: nn.Module):
    """The lifted module and its parameters (name → tensor) in input order."""
    return Lifted(module).eval(), dict(module.named_parameters())


class CachedModulation(nn.Module):
    """Stands in for a block's `adaln_proj`: the six modulation tensors arrive precomputed as one `[6, 3k, D]` table."""

    def forward(self, table):
        return table.unbind(0)


class BlockCall(nn.Module):
    """One transformer block with its modulation table and rotary tables as explicit inputs."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden, table, tags, cos, sin):
        return self.block(hidden, table, tags, (cos, sin))


class Heads(nn.Module):
    """The output norm and both heads: every row, then each modality's rows selected."""

    def __init__(self, model):
        super().__init__()
        self.norm_out, self.proj_out, self.audio_proj_out = model.norm_out, model.proj_out, model.audio_proj_out

    def forward(self, hidden, temb, indices, video_indices, audio_indices):
        hidden = self.norm_out(hidden, temb, indices).to(self.proj_out.weight.dtype)
        return self.proj_out(hidden).index_select(1, video_indices), self.audio_proj_out(hidden).index_select(1, audio_indices)


class PrepareText(nn.Module):
    """context_embedder + the 2-layer token refiner (with its Turbo LoRA): the per-render text rows."""

    def __init__(self, model):
        super().__init__()
        self.context_embedder, self.token_refiner = model.context_embedder, model.token_refiner

    def forward(self, text):
        return self.token_refiner(self.context_embedder(text))


class Pack(nn.Module):
    """The per-step packed sequence: a permutation of [text; video; audio], `gather[r]` naming row r's source."""

    def __init__(self, model):
        super().__init__()
        self.proj_in, self.audio_proj_in = model.proj_in, model.audio_proj_in

    def forward(self, video, audio, text, gather):
        rows = torch.cat((text, self.proj_in(video).to(text.dtype), self.audio_proj_in(audio).to(text.dtype)), dim=1)
        return rows.index_select(1, gather)


class ConditionerLayer(nn.Module):
    """One Qwen3-VL text decoder layer; causal attention inside (no mask input)."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden, cos, sin):
        return self.layer(hidden, position_embeddings=(cos, sin), attention_mask=None)


class AudioDecoder(nn.Module):
    """Channel-major normalised rows [2, Na, 32] f32 → stereo planar PCM [2, 800·Na] f32 at 32 kHz."""

    def __init__(self, vae):
        super().__init__()
        self.dec_in_proj, self.decoder = vae.dec_in_proj, vae.decoder
        self.register_buffer("mean", torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(1, -1, 1))
        self.register_buffer("std", torch.tensor(vae.config.latents_std, dtype=torch.float32).view(1, -1, 1))

    def forward(self, rows):
        latents = rows.permute(0, 2, 1) * self.std + self.mean
        return self.decoder(self.dec_in_proj(latents)).squeeze(1)


class WindowTileDecoder(nn.Module):
    """Packed tile rows of one temporal window [448, 96] f32 → the 22 frames [1, 3, 22, 256, 256] f32 the source keeps
    from that window, decoded in the VAE's own dtype (bf16 by default: the prototype's E1 measured 2.4× faster, half
    the memory, PSNR 55 dB against float32)."""

    def __init__(self, vae, tile: int = VIDEO_TILE):
        super().__init__()
        self.vae, self.side = vae, tile // 16
        self.register_buffer("mean", torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(1, 24, 1, 1, 1))
        self.register_buffer("std", torch.tensor(vae.config.latents_std, dtype=torch.float32).view(1, 24, 1, 1, 1))

    def latents(self, rows):
        s = self.side
        # rows[(f*s/2 + y)*s/2 + x, c*4 + dy*2 + dx] → z[0, c, f, 2y+dy, 2x+dx]
        z = rows.view(WINDOW_LATENTS, s // 2, s // 2, 24, 2, 2).permute(3, 0, 1, 4, 2, 5).reshape(1, 24, WINDOW_LATENTS, s, s)
        return z * self.std + self.mean

    def forward(self, rows):
        z = self.latents(rows).to(self.vae.post_quant_conv.weight.dtype)
        clip = self.vae.decoder(self.vae.post_quant_conv(z))
        return torch.cat((clip[:, :, 3:20], clip[:, :, 23:28]), dim=2).float()


class TileEncoder(nn.Module):
    """One 256 px source tile [1, 3, 1, 256, 256] in 0…255 → moments [1, 48, 1, 16, 16] (`_encode_clip`'s per-tile
    encoder and quant_conv)."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        self.register_buffer("pixel_mean", torch.tensor(PIXEL_MEAN).view(1, -1, 1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(PIXEL_STD).view(1, -1, 1, 1, 1))

    def forward(self, pixels):
        return self.vae.quant_conv(self.vae.encoder((pixels.div(255.0) - self.pixel_mean) / self.pixel_std))


class TileTail(nn.Module):
    """The tiles' moments, row-major → the keyframe condition [1, 24, 1, H/16, W/16]: `_stitch_tiles`, the posterior
    with the fixed seed-42 draw baked in, float16 rounding and latent normalisation (`encode_vae_condition`)."""

    def __init__(self, vae, eps, rows, columns, y_overlaps, x_overlaps):
        super().__init__()
        self.vae, self.rows, self.columns = vae, rows, columns
        self.y_overlaps = [o // vae.spatial_compression_ratio for o in y_overlaps]
        self.x_overlaps = [o // vae.spatial_compression_ratio for o in x_overlaps]
        self.register_buffer("eps", eps)
        self.register_buffer("mean", torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1))
        self.register_buffer("std", torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1))

    def forward(self, tiles):
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        grid = [[tiles[i * self.columns + j:i * self.columns + j + 1] for j in range(self.columns)] for i in range(self.rows)]
        posterior = DiagonalGaussianDistribution(self.vae._stitch_tiles(grid, self.y_overlaps, self.x_overlaps))
        latents = (posterior.mean + posterior.std * self.eps).to(torch.float16).float()
        return (latents - self.mean) / self.std


# MARK: conversion

def refuse_linked(path: Path) -> None:
    """Refuses to write `path` when it, or a file under it, is hard-linked: an extended export (`extend_h3.py`) links
    the files it reuses to the source export, and writing one would change the source in place."""
    path = Path(path)
    candidates = [path] if path.is_file() else (list(path.rglob("*")) if path.is_dir() else [])
    linked = [p for p in candidates if p.is_file() and p.stat().st_nlink > 1]
    if linked:
        raise SystemExit(f"refusing to write {path}: {linked[0]} is hard-linked to another export (it would change in place)")


def convert(module: nn.Module, example: tuple, dynamic, inputs: list[str], outputs: list[str], path: Path,
            entry: str = "main") -> dict:
    """torch.export → Core AI asset on the CPU (never loaded or specialised here)."""
    import coreai_torch as ct
    import time
    refuse_linked(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    with torch.inference_mode():
        program = torch.export.export(module.eval(), example, dynamic_shapes=dynamic).run_decompositions(ct.get_decomp_table())
    t1 = time.monotonic()
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    converter.add_exported_program(program, input_names=inputs, output_names=outputs, state_names=[], entrypoint_name=entry)
    del program
    asset = converter.to_coreai()
    del converter
    asset.optimize()
    asset.save_asset(path)
    stream = sha256_file(path / "main.mlirb")
    assert (path / "main.hash").read_bytes().hex() == stream, "the asset's hash file differs from its stream"
    return {"function": entry, "inputs": inputs, "outputs": outputs, "exportSeconds": round(t1 - t0, 1),
            "convertSeconds": round(time.monotonic() - t1, 1), "sha256": stream}


# MARK: weight files (MANIFEST.md §5.8)

class WeightFile:
    """Writes one block's lifted weights contiguously, in input order, and returns each input's extent. `asset_path`
    is the file's `"<asset>:<path>"` in the bundle (the table's `file`)."""

    def __init__(self, path: Path, asset_path: str):
        self.path, self.asset_path = path, asset_path

    def write(self, params: dict, dtype: torch.dtype | None = None, verify: bool = False) -> tuple[dict, list[dict]]:
        """Writes the file, or with `verify` checks that the existing file holds exactly these bytes (an extended
        export reuses it, linked) and writes nothing."""
        if not verify:
            refuse_linked(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        extents, specs, offset = {}, [], 0
        with open(self.path, "rb" if verify else "wb") as f:
            for i, (name, tensor) in enumerate(params.items()):
                assert dtype is None or tensor.dtype == dtype, (name, tensor.dtype)
                data = tensor_bytes(tensor)
                if verify:
                    assert f.read(len(data)) == data, f"{self.path}: {name} differs from the existing file"
                else:
                    f.write(data)
                extents[f"w{i}"] = {"file": self.asset_path, "offset": offset, "bytes": len(data)}
                specs.append({"name": f"w{i}", "source": name, "dtype": str(tensor.dtype).removeprefix("torch."),
                              "shape": list(tensor.shape)})
                offset += len(data)
        return extents, specs


def read_extent(bundle: Path, extent: dict, dtype: torch.dtype, shape) -> torch.Tensor:
    """An extent's bytes as a tensor (the CPU side of a check; the runtime reads the same bytes with `pread`)."""
    asset, path = extent["file"].split(":", 1)
    assert asset == "main"
    with open(bundle / path, "rb") as f:
        f.seek(extent["offset"])
        data = f.read(extent["bytes"])
    assert len(data) == extent["bytes"]
    return torch.frombuffer(bytearray(data), dtype=dtype).view(shape)


def raw_tensor(path: Path, tensor: torch.Tensor) -> dict:
    """A host table file (raw little-endian bytes) and its spec."""
    tensor = tensor.detach().contiguous().cpu()
    data = tensor_bytes(tensor)
    refuse_linked(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"dtype": str(tensor.dtype).removeprefix("torch."), "shape": list(tensor.shape), "sha256": sha256_bytes(data)}


def safetensors_span(file: Path, key: str) -> tuple[int, int, str, list]:
    """(absolute offset, length, dtype, shape) of `key` in a safetensors file (the embedding row source is copied
    verbatim from it)."""
    with open(file, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
    start, end = header[key]["data_offsets"]
    return 8 + size + start, end - start, header[key]["dtype"], header[key]["shape"]

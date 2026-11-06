"""The AdcSR x4 super-resolution export graph, the tiling rule the `image-to-image-tiled@1` driver replays, and the
shared harness for its exporter, reference and checks.

AdcSR (Guaishou74851/AdcSR, CVPR 2025) is a one-step diffusion-GAN super-resolver compressed from OSEDiff: a
PixelUnshuffle(2), a channel-pruned SD 2.1 base U-Net with its time embedding and cross-attention removed, and a
half-width VAE decoder (its mid block retrained in stage 2, its up blocks from the stage-1 `halfDecoder.ckpt`). The
upstream code (`model.py`, `forward.py`, Apache-2.0) is imported read-only from the pinned Hugging Face snapshot;
nothing here modifies it. `build_upstream` assembles the network exactly as upstream's `test.py` does, on the CPU.

The exported graph (`TileGraph`) is that network on one fixed tile:

* input `lr` [1, 3, 128, 128] float32 on [-1, 1] (RGB x / 255 * 2 - 1: `transforms.ToTensor()(LR) * 2 - 1`);
* output `sr` [1, 3, 512, 512] float32, nominally [-1, 1], raw: no colour match, no clamp (both are the host's).

By default (`--precision float16`) the U-Net, the decoder mid block and the first two decoder up blocks are float16 and
the decoder's last two up blocks and output head float32 (`FP32_FROM`: their activations exceed float16's range); the
I/O stays float32 and the graph casts at its boundaries.

Upstream does not tile. `test.py` runs the whole LR image through the network (any size whose sides are multiples of
16: PixelUnshuffle(2), then three 2x downsamplings), then matches each channel's mean and standard deviation to the
LR's, then maps `x / 2 + 0.5`, clamps to [0, 1] and quantizes with torchvision's `ToPILImage` (`(x * 255)` truncated to
uint8). Its benchmark inputs (RealSR, DRealSR, DIV2K-Val LR) are all 128 x 128: a 128 tile is exactly upstream's
operating point, and also its training size (`gt_size` 512 at x4). The network has global self-attention (64 x 64
tokens at the first U-Net level for a 128 tile) and GroupNorm over the whole input, so a tile's output depends on the
whole tile, and a static graph needs a fixed tile. The driver's tiling (`tile_grid`, `stitch`, documented in
`TILING`) is therefore Aether's, not upstream's. It is the rule coreai-kit's SuperResolver uses, which MANIFEST.md
records for AdcSR (overlap `max(4, tile / 8)`). The colour match and the quantization after it are upstream's
(`postprocess`).
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "source-pins.json").read_text())
REPO, REVISION = PINS["repository"], PINS["revision"]
CODE_REPO, CODE_COMMIT = PINS["upstreamCode"]["repository"], PINS["upstreamCode"]["commit"]
BASE = PINS["base"]
HUB = Path.home() / ".cache/huggingface/hub"


def hub_snapshot(repository: str, revision: str) -> Path:
    return HUB / f"models--{repository.replace('/', '--')}/snapshots/{revision}"


DEFAULT_SNAPSHOT = hub_snapshot(REPO, REVISION)
DEFAULT_BASE_CONFIG = hub_snapshot(BASE["config"]["repository"], BASE["config"]["revision"])
DEFAULT_BASE_LICENSE = hub_snapshot(BASE["licenseText"]["repository"], BASE["licenseText"]["revision"])
IMAGES = Path.home() / ".cache/aether-kit/adcsr/images"

TILE = 128                          # LR tile side: upstream's benchmark and training size
SCALE = 4
OVERLAP = max(4, TILE // 8)         # 16 LR pixels (MANIFEST.md §2.5, M4.md)
SR_TILE = TILE * SCALE
SIDE_MULTIPLE = 16                  # upstream's whole-image inference needs both sides a multiple of 16
DTYPES = {"float32": torch.float32, "float16": torch.float16}
RANGE = (-1.0, 1.0)

# The rule the Swift driver reproduces (also written into the manifest's `preprocess.image.tiling`).
TILING = {
    "tile": [TILE, TILE], "scale": SCALE, "overlap": OVERLAP, "overlapRule": "max(4, tile / 8)",
    "starts": "per axis: [0] when extent <= tile; else 0, step, 2 step, ... while <= extent - tile (step = tile - "
              "overlap), then extent - tile appended when it is not already last. Tiles in row-major order (y outer, "
              "x inner).",
    "edge": "clamp: a tile reaching past an extent smaller than the tile repeats the last row or column (only "
            "happens when that extent < tile)",
    "blend": "featherLinear: per output axis a ramp of length tile * scale, w[i] = (i + 1) / (e + 1) for i < e and "
             "w[n - 1 - i] the same, 1 elsewhere, e = overlap * scale (64); a tile pixel weighs w[y] * w[x]. Each "
             "output pixel is sum(w * sr) / sum(w) over the tiles covering it, float32; tile pixels past the "
             "image's x4 extent are dropped",
    "order": "stitch the raw tiles, then colour-match the whole stitched image once (never per tile)",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sources(snapshot: Path, base_config: Path = DEFAULT_BASE_CONFIG,
                   base_license: Path = DEFAULT_BASE_LICENSE, images: bool = True) -> dict:
    """Checks every pinned file the harness reads; returns {label: sha256}."""
    verified = {}
    groups = [("", snapshot, PINS["files"]), ("sd21base:", base_config, BASE["config"]["files"]),
              ("sd21base:", base_license, BASE["licenseText"]["files"])]
    for prefix, root, files in groups:
        for relative, expected in files.items():
            path = root / relative
            if not path.is_file():
                raise SystemExit(f"{path} is missing; run Forge/adcsr/fetch_upstream.py")
            if sha256_file(path) != expected:
                raise SystemExit(f"{path} differs from its pin; re-run Forge/adcsr/fetch_upstream.py")
            verified[prefix + relative] = expected
    if images:
        for name, pin in PINS["referenceImages"]["files"].items():
            path = IMAGES / name
            if not path.is_file() or sha256_file(path) != pin["sha256"]:
                raise SystemExit(f"{path} is missing or differs from its pin; run Forge/adcsr/fetch_upstream.py")
            verified[f"image:{name}"] = pin["sha256"]
    return verified


def load_upstream(snapshot: Path):
    """Imports upstream `forward.py` (as `forward`, the name `model.py` imports) and `model.py` from the snapshot."""
    modules = {}
    for name, file in (("forward", "forward.py"), ("adcsr_upstream_model", "model.py")):
        existing = sys.modules.get(name)
        if existing is not None:
            if getattr(existing, "__file__", None) != str(snapshot / file):
                raise RuntimeError(f"a different {name!r} is already imported; run the harness in a fresh interpreter")
            modules[name] = existing
            continue
        spec = importlib.util.spec_from_file_location(name, snapshot / file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules["adcsr_upstream_model"]


class _LightningCallbackStub:
    """Stands in for `pytorch_lightning.callbacks.ModelCheckpoint`, the one non-tensor class halfDecoder.ckpt's pickle
    names (its `callbacks` entry, unused), so the checkpoint loads with `weights_only=True`."""


def _load_half_decoder_state(snapshot: Path) -> dict:
    target = "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint"
    with torch.serialization.safe_globals([(_LightningCallbackStub, target)]):
        checkpoint = torch.load(snapshot / "weight/pretrained/halfDecoder.ckpt", map_location="cpu", weights_only=True)
    # test.py: every key containing "decoder", with "decoder." removed (the stage-1 VAE's decoder half).
    return {k.replace("decoder.", ""): v for k, v in checkpoint["state_dict"].items() if "decoder" in k}


def build_upstream(snapshot: Path, base_config: Path = DEFAULT_BASE_CONFIG) -> nn.Sequential:
    """upstream test.py's network, float32, eval, on the CPU, without DataParallel or CUDA:

        Sequential(Net(unet, deepcopy(decoder)), *decoder.up_blocks, decoder.conv_norm_out, conv_act, conv_out)

    The U-Net is built from SD 2.1 base's config (test.py takes `pipe.unet` with SD 2.1 base weights; `Net` and the
    strict load of net_params_200.pkl overwrite every one of them, so they never reach the output). The decoder is
    test.py's `Decoder(...)` with halfDecoder.ckpt's decoder weights, loaded strictly; net_params_200.pkl is loaded
    strictly (its `module.` prefix, from DataParallel, stripped)."""
    from diffusers import UNet2DConditionModel
    from diffusers.models.autoencoders.vae import Decoder

    upstream = load_upstream(snapshot)
    torch.manual_seed(0)
    unet = UNet2DConditionModel.from_config(json.loads((base_config / "unet/config.json").read_text()))
    decoder = Decoder(in_channels=4, out_channels=3, up_block_types=["UpDecoderBlock2D" for _ in range(4)],
                      block_out_channels=[64, 128, 256, 256], layers_per_block=2, norm_num_groups=32, act_fn="silu",
                      norm_type="group", mid_block_add_attention=True)
    decoder.load_state_dict(_load_half_decoder_state(snapshot), strict=True)
    net = upstream.Net(unet, copy.deepcopy(decoder))
    state = torch.load(snapshot / "weight/net_params_200.pkl", map_location="cpu", weights_only=True)
    if not all(k.startswith("module.") for k in state):
        raise SystemExit("unexpected net_params_200.pkl key prefix")
    net.load_state_dict({k[len("module."):]: v for k, v in state.items()}, strict=True)
    model = nn.Sequential(net, *decoder.up_blocks, decoder.conv_norm_out, decoder.conv_act, decoder.conv_out)
    return model.eval()


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())     # unique tensors (unet.body aliases its blocks)


# test.py's Sequential by index: 0 Net (PixelUnshuffle, U-Net, decoder mid block), 1-4 the decoder's up blocks (256,
# 256, 128, 64 channels; x2 each after the first three), 5 conv_norm_out, 6 SiLU, 7 conv_out. Peak |activation| over
# the 21 reference tiles, float32: Net 976 (inside its U-Net), up 0-3 384, 2175, 33849, 72372 (the last two at 256 and
# 512 px), conv_norm_out 128. float16's largest finite value is 65504, so up block 3 overflows it (the float16 graph
# returned inf) and up block 2 has under 2x headroom. The float16 export keeps stages FP32_FROM.. in float32.
FP32_FROM = 3
STAGES = ["net", "up0", "up1", "up2", "up3", "conv_norm_out", "conv_act", "conv_out"]


class TileGraph(nn.Module):
    """lr [1,3,T,T] float32 on [-1,1] -> sr [1,3,4T,4T] float32 (raw). With `dtype` float16, stages 0..fp32_from-1
    (the U-Net, the decoder mid block and the first two decoder up blocks: nearly all the weights) run in float16 and
    the rest in float32. With float32 every stage is float32. The graph casts its input and the stage boundary."""

    def __init__(self, model: nn.Sequential, dtype: torch.dtype = torch.float16, fp32_from: int = FP32_FROM) -> None:
        super().__init__()
        split = len(model) if dtype == torch.float32 else fp32_from
        self.low = nn.Sequential(*list(model)[:split]).to(dtype)
        self.high = nn.Sequential(*list(model)[split:]).to(torch.float32)
        self.dtype = dtype

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        x = self.low(lr.to(self.dtype)).to(torch.float32)
        return self.high(x)


def precision_split(model: nn.Sequential, dtype: torch.dtype, fp32_from: int = FP32_FROM) -> dict:
    split = len(model) if dtype == torch.float32 else fp32_from
    count = lambda mods: sum(p.numel() for m in mods for p in m.parameters())  # noqa: E731
    return {"low": STAGES[:split], "lowDType": str(dtype).removeprefix("torch."), "float32": STAGES[split:],
            "lowParameters": count(list(model)[:split]), "float32Parameters": count(list(model)[split:])}


# MARK: the host side: preprocess, tiling, colour match, quantization

def lr_tensor(lr_u8: np.ndarray) -> np.ndarray:
    """[H,W,3] uint8 RGB -> [1,3,H,W] float32 on [-1,1], as test.py: ToTensor (CHW made contiguous, x / 255 in
    float32), then * 2 - 1."""
    x = torch.from_numpy(np.ascontiguousarray(lr_u8)).permute(2, 0, 1).contiguous().float().div(255)
    return np.ascontiguousarray((x * 2 - 1)[None].numpy())


def tile_starts(extent: int, tile: int = TILE, overlap: int = OVERLAP) -> list[int]:
    if extent <= tile:
        return [0]
    starts = list(range(0, extent - tile + 1, tile - overlap))
    if starts[-1] != extent - tile:
        starts.append(extent - tile)
    return starts


def tile_grid(width: int, height: int) -> list[tuple[int, int]]:
    """(x0, y0) of every LR tile, row-major (y outer, x inner)."""
    return [(x, y) for y in tile_starts(height) for x in tile_starts(width)]


def crop_tile(lr: np.ndarray, x0: int, y0: int) -> np.ndarray:
    """[1,3,H,W] -> [1,3,T,T] at (x0, y0), indices past the image clamped to its last row / column."""
    _, _, h, w = lr.shape
    ys = np.minimum(np.arange(y0, y0 + TILE), h - 1)
    xs = np.minimum(np.arange(x0, x0 + TILE), w - 1)
    return np.ascontiguousarray(lr[:, :, ys][:, :, :, xs], dtype=np.float32)


def feather_ramp(side: int = SR_TILE, edge: int = OVERLAP * SCALE) -> np.ndarray:
    ramp = np.ones(side, dtype=np.float32)
    for i in range(max(1, edge)):
        w = np.float32(i + 1) / np.float32(edge + 1)
        ramp[i] = w
        ramp[side - 1 - i] = w
    return ramp


def stitch(tiles: list[tuple[int, int, np.ndarray]], width: int, height: int) -> np.ndarray:
    """Feather-blends raw x4 tiles [1,3,4T,4T] at LR (x0, y0) into [1,3,4H,4W] (float32, `TILING`'s rule)."""
    out_w, out_h = width * SCALE, height * SCALE
    acc = np.zeros((3, out_h, out_w), dtype=np.float32)
    weight = np.zeros((out_h, out_w), dtype=np.float32)
    ramp = feather_ramp()
    w2 = ramp[:, None] * ramp[None, :]
    for x0, y0, sr in tiles:
        dx, dy = x0 * SCALE, y0 * SCALE
        cw, ch = min(SR_TILE, out_w - dx), min(SR_TILE, out_h - dy)
        acc[:, dy:dy + ch, dx:dx + cw] += sr[0, :, :ch, :cw] * w2[:ch, :cw]
        weight[dy:dy + ch, dx:dx + cw] += w2[:ch, :cw]
    return (acc / np.maximum(weight, np.float32(1e-6)))[None]


def upscale_tiled(run_tile, lr: np.ndarray) -> tuple[np.ndarray, list]:
    """The driver's path on a [1,3,H,W] LR tensor: tile, run each tile, stitch. Returns (raw stitched sr, tiles)."""
    _, _, h, w = lr.shape
    tiles = []
    for x0, y0 in tile_grid(w, h):
        tiles.append((x0, y0, np.asarray(run_tile(crop_tile(lr, x0, y0)), dtype=np.float32)))
    return stitch(tiles, w, h), tiles


def color_match(sr: np.ndarray, lr: np.ndarray) -> np.ndarray:
    """Upstream test.py's per-image colour match, in torch float32: per channel over H, W,
    (SR - mean(SR)) / std(SR) * std(LR) + mean(LR), std unbiased (Bessel, N - 1) as torch's default."""
    s, l = torch.from_numpy(sr), torch.from_numpy(lr)
    out = (s - s.mean(dim=[2, 3], keepdim=True)) / s.std(dim=[2, 3], keepdim=True) \
        * l.std(dim=[2, 3], keepdim=True) + l.mean(dim=[2, 3], keepdim=True)
    return out.numpy()


def to_unit(sr: np.ndarray) -> np.ndarray:
    """[-1,1] -> [0,1] clamped, as test.py: (SR / 2 + 0.5).clamp(0, 1)."""
    return (torch.from_numpy(sr) / 2 + 0.5).clamp(0, 1).numpy()


def to_uint8(unit: np.ndarray) -> np.ndarray:
    """[1,3,H,W] on [0,1] -> [H,W,3] uint8, as torchvision's ToPILImage on a float tensor: (x * 255) truncated."""
    return (unit[0].transpose(1, 2, 0) * np.float32(255)).astype(np.uint8)


def postprocess(sr_raw: np.ndarray, lr: np.ndarray) -> np.ndarray:
    """Raw stitched SR -> the final image on [0,1]: colour match to the LR, then map and clamp (float32)."""
    return to_unit(color_match(sr_raw, lr))


def upstream_full(model: nn.Module, lr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """upstream test.py on the whole LR image (sides multiples of 16): (raw SR, final [0,1] image)."""
    _, _, h, w = lr.shape
    if h % SIDE_MULTIPLE or w % SIDE_MULTIPLE:
        raise ValueError(f"upstream's whole-image inference needs sides that are multiples of {SIDE_MULTIPLE}")
    with torch.no_grad():
        raw = model(torch.from_numpy(lr)).numpy()
    return raw, postprocess(raw, lr)


def psnr(got: np.ndarray, want: np.ndarray, peak: float = 1.0) -> float:
    """10 log10(peak^2 / MSE) in float64 over every value; inf when identical."""
    mse = float(np.mean((got.astype(np.float64) - want.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else float(10 * np.log10(peak * peak / mse))


def psnr_raw(got: np.ndarray, want: np.ndarray) -> float:
    """PSNR of raw graph outputs on their [-1,1] scale (peak-to-peak 2: the same as on x / 2 + 0.5, unclamped)."""
    return psnr(got, want, RANGE[1] - RANGE[0])


# MARK: reference-case inputs

def photo_lr(path: Path, width: int, height: int) -> np.ndarray:
    """A CC0 photo as an 8-bit LR image: Pillow decode, convert('RGB'), centre crop to width:height (the crop box
    rounded to whole pixels), Pillow BICUBIC down to width x height. [H,W,3] uint8."""
    from PIL import Image

    with Image.open(path) as im:
        rgb = im.convert("RGB")
    sw, sh = rgb.size
    if sw * height > sh * width:                  # source wider: crop the sides
        cw, ch = round(sh * width / height), sh
    else:
        cw, ch = sw, round(sw * height / width)
    left, top = (sw - cw) // 2, (sh - ch) // 2
    lr = rgb.resize((width, height), Image.Resampling.BICUBIC, box=(left, top, left + cw, top + ch))
    return np.asarray(lr, dtype=np.uint8).copy()


def synthetic_lr(seed: int, width: int, height: int) -> np.ndarray:
    """A deterministic structured 8-bit image (gradients, a disc, stripes, seeded noise). Pure numpy."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0, 1, height, dtype=np.float32), np.linspace(0, 1, width, dtype=np.float32),
                         indexing="ij")
    disc = (((xx - 0.6) ** 2 + (yy - 0.4) ** 2) < 0.04).astype(np.float32)
    stripes = (np.sin(xx * 40) > 0).astype(np.float32) * (yy > 0.7)
    channels = [0.6 * yy + 0.3 * disc, 0.5 * xx + 0.4 * (1 - disc) * yy + 0.2 * stripes, 0.4 * (1 - yy) + 0.5 * disc]
    x = np.stack(channels, axis=-1) + 0.08 * rng.random((height, width, 3), dtype=np.float32)
    return (np.clip(x, 0, 1) * 255).round().astype(np.uint8)


# Each case: an LR image, every tile's input and upstream output, the stitched result, and (when both sides are
# multiples of 16) upstream's own whole-image inference. Sizes cover one exact tile, multi-tile grids that are and are
# not multiples of the tile and of 16, and images smaller than the tile (edge clamp) on one or both axes.
CASES = [
    {"name": "cat-128x128", "image": "black-cat-sofa.jpg", "size": [128, 128]},
    {"name": "burnside-256x192", "image": "burnside-irt.jpg", "size": [256, 192]},
    {"name": "wood-green-208x144", "image": "wood-green-1981.jpg", "size": [208, 144]},
    {"name": "parade-320x96", "image": "bicycle-parade-285.jpg", "size": [320, 96]},
    {"name": "burnside-203x141", "image": "burnside-irt.jpg", "size": [203, 141]},
    {"name": "cat-100x75", "image": "black-cat-sofa.jpg", "size": [100, 75]},
    {"name": "synthetic-160x112", "synthetic": 7, "size": [160, 112]},
]


def case_lr(case: dict) -> np.ndarray:
    w, h = case["size"]
    if "image" in case:
        return photo_lr(IMAGES / case["image"], w, h)
    return synthetic_lr(case["synthetic"], w, h)


def write_f32(path: Path, x: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(x, dtype="<f4").tobytes())


def read_f32(path: Path, shape) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype="<f4").reshape(shape).copy()


def write_png(path: Path, rgb_u8: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(rgb_u8, "RGB").save(path, optimize=False, compress_level=6)

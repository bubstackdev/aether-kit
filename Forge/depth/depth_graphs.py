"""The Depth Anything 3 small (DA3-SMALL) export graph and the shared harness for its exporter, reference and checks.

DA3-SMALL (depth-anything/DA3-SMALL, Apache-2.0, ByteDance Seed) is a DINOv2 ViT-S backbone with a DualDPT head. The
upstream PyTorch code (ByteDance-Seed/Depth-Anything-3, Apache-2.0) is imported read-only from the pinned unpack that
`fetch_upstream.py` writes; nothing here modifies it, and its package is never installed.

The exported graph (`DepthGraph`) is the single-view, depth-only path of upstream `DepthAnything3Net.forward`:

* input `image` [1, 3, 504, 504], RGB on [0, 1] (the host stretches the image to 504 x 504 and scales by 1/255);
* ImageNet normalization (mean 0.485/0.456/0.406, std 0.229/0.224/0.225, upstream's `InputProcessor.NORMALIZE`) is
  in the graph, then the view axis is added: upstream takes [B, N, 3, H, W], here N = 1;
* backbone, then the DualDPT head; outputs `depth` [1, 504, 504] (exp activation: relative depth, larger is farther)
  and `depth_conf` [1, 504, 504] (exp + 1 activation, >= 1, larger is more confident).

Everything else upstream's forward does is omitted because it cannot affect these two outputs for one view without
camera inputs: the camera encoder (only with given extrinsics), the camera decoder and the ray outputs (pose; the ray
branch is dead code in the export), reference-view selection (only for >= 3 views), Gaussian splatting (disabled), and
the sky step (DA3-SMALL has no sky output). `selftest_depth.py` checks that the graph equals upstream's own forward.

One export-only substitution, applied by `export_patches()` and undone after: upstream's
`RotaryPositionEmbedding2D.forward` sizes its frequency table with `int(positions.max()) + 1`, a data-dependent value
that `torch.export` cannot trace. The patch sizes it from the static token count instead (a larger table whose rows at
the positions used are identical) and computes the angles in float32 before casting to the compute dtype. In float32
the result is bit-identical; in float16 it is more accurate than upstream's float16 angles (positions reach 37).
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "source-pins.json").read_text())
REPO, REVISION = PINS["repository"], PINS["revision"]
CODE_REPO, CODE_COMMIT = PINS["upstreamCode"]["repository"], PINS["upstreamCode"]["commit"]
DEFAULT_SNAPSHOT = Path.home() / f".cache/huggingface/hub/models--depth-anything--DA3-SMALL/snapshots/{REVISION}"
DEFAULT_UPSTREAM = Path.home() / f".cache/aether-kit/depth/upstream/depth-anything-3-{CODE_COMMIT[:12]}"

SIZE = 504                     # upstream's default processing resolution (36 x 36 patches of 14)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DTYPES = {"float32": torch.float32, "float16": torch.float16}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sources(snapshot: Path, upstream_root: Path) -> dict:
    """Checks every pinned snapshot file and the upstream unpack; returns {file: sha256}."""
    verified = {}
    for relative, expected in PINS["files"].items():
        path = snapshot / relative
        if not path.is_file():
            raise SystemExit(f"snapshot lacks {relative}; run Forge/depth/fetch_upstream.py ({REPO}@{REVISION[:12]})")
        if sha256_file(path) != expected:
            raise SystemExit(f"{relative} differs from the pinned {REVISION[:12]} file")
        verified[relative] = expected
    stamp = upstream_root / "UNPACKED.json"
    if not stamp.is_file() or json.loads(stamp.read_text()).get("commit") != CODE_COMMIT:
        raise SystemExit(f"{upstream_root} is not the pinned upstream unpack; run Forge/depth/fetch_upstream.py")
    for group in (PINS["upstreamCode"]["files"], PINS["referenceImages"]["files"]):
        for relative, expected in group.items():
            if sha256_file(upstream_root / relative) != expected:
                raise SystemExit(f"{relative} in the upstream unpack differs from the pin; re-run fetch_upstream.py")
            verified[f"upstream:{relative}"] = expected
    return verified


def load_upstream(upstream_root: Path):
    """Imports `depth_anything_3.model.da3` and `depth_anything_3.cfg` from the unpack, without upstream's API module
    (which imports its export, I/O and pose-alignment stack)."""
    expected = str(upstream_root / "src/depth_anything_3")
    if "depth_anything_3" in sys.modules:
        existing = sys.modules["depth_anything_3"]
        if list(getattr(existing, "__path__", [])) != [expected]:
            raise RuntimeError("a different 'depth_anything_3' is already imported; run the harness in a fresh interpreter")
    else:
        package = types.ModuleType("depth_anything_3")
        package.__path__ = [expected]
        sys.modules["depth_anything_3"] = package
    import os
    os.environ.setdefault("DA3_LOG_LEVEL", "WARN")   # upstream logs each block's FFN choice at INFO
    cfg = importlib.import_module("depth_anything_3.cfg")
    da3 = importlib.import_module("depth_anything_3.model.da3")
    rope = importlib.import_module("depth_anything_3.model.dinov2.layers.rope")
    return cfg, da3, rope


def build_net(upstream_root: Path, snapshot: Path):
    """The unmodified upstream DepthAnything3Net, float32, eval, with the pinned DA3-SMALL weights (strict load).
    Built from the snapshot's config.json, which is checked to equal upstream's own `configs/da3-small.yaml`."""
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    cfg, _, _ = load_upstream(upstream_root)
    config = json.loads((snapshot / "config.json").read_text())
    if config.get("model_name") != "da3-small":
        raise SystemExit(f"unexpected model_name {config.get('model_name')!r}")
    yaml = OmegaConf.to_container(OmegaConf.load(upstream_root / "src/depth_anything_3/configs/da3-small.yaml"))
    if yaml != config["config"]:
        raise SystemExit("config.json's network config differs from upstream's configs/da3-small.yaml")
    net = cfg.create_object(OmegaConf.create(config["config"]))
    state = load_file(str(snapshot / "model.safetensors"))
    if not all(k.startswith("model.") for k in state):
        raise SystemExit("unexpected checkpoint key prefix")
    missing, unexpected = net.load_state_dict({k[len("model."):]: v for k, v in state.items()}, strict=False)
    # The checkpoint has no weights for the auxiliary (ray) head's last convs; upstream's Hub loader is non-strict too,
    # so they keep their init. Ray outputs feed only the pose path, never `depth` or `depth_conf`.
    if unexpected or any(not k.startswith(AUX_RAY_PREFIX) for k in missing):
        raise SystemExit(f"checkpoint mismatch: missing {missing}, unexpected {unexpected}")
    return net.eval()


AUX_RAY_PREFIX = "head.scratch.output_conv2_aux."


def upstream_depth(net, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream's own forward on a raw [0,1] [1,3,H,W] float32 image, normalized as upstream's InputProcessor does
    (torchvision `Normalize`). Returns depth and confidence, [1,H,W] each."""
    import torchvision.transforms as T

    x = T.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))(image[0])[None, None]
    with torch.no_grad():
        out = net(x, None, None, [], False, False, "saddle_balanced")
    return out["depth"][:, 0], out["depth_conf"][:, 0]


class DepthGraph(nn.Module):
    """[1,3,H,W] raw RGB on [0,1] (float16) -> depth [1,H,W], depth_conf [1,H,W] (float32).

    Precision follows upstream's own inference recipe: `DepthAnything3.forward` runs the backbone under float16/bfloat16
    autocast and the head with autocast disabled (float32). Here the backbone's weights are cast to `backbone_dtype`
    (float16 by default) and the head stays float32; normalization runs in float32, as upstream's host-side
    `InputProcessor` does. The input is float16 (exact for 8-bit images over 255 to within float16 rounding), the outputs
    float32 (the head's own precision)."""

    def __init__(self, net, backbone_dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self.backbone, self.head = net.backbone.to(backbone_dtype), net.head.to(torch.float32)
        self.backbone_dtype = backbone_dtype
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN, dtype=torch.float32).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD, dtype=torch.float32).reshape(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = ((image.to(torch.float32) - self.mean) / self.std)[:, None].to(self.backbone_dtype)
        height, width = x.shape[-2], x.shape[-1]
        feats, _ = self.backbone(x, cam_token=None, export_feat_layers=[], ref_view_strategy="saddle_balanced")
        feats = [(f.to(torch.float32), token.to(torch.float32)) for f, token in feats]
        out = self.head(feats, height, width, patch_start_idx=0)
        return out["depth"][:, 0], out["depth_conf"][:, 0]


@contextlib.contextmanager
def export_patches(upstream_root: Path):
    """The static-shape RoPE table for torch.export (see the module docstring). Clears the frequency cache on both
    sides so no traced tensor outlives the export."""
    _, _, rope = load_upstream(upstream_root)
    cls = rope.RotaryPositionEmbedding2D
    original = cls.forward

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        assert positions.ndim == 3 and positions.shape[-1] == 2, "Positions must have shape (batch_size, n_tokens, 2)"
        feature_dim = tokens.size(-1) // 2
        max_position = positions.shape[-2] + 1          # >= every position: 1 + the patch-grid side, plus specials
        cos_comp, sin_comp = self._compute_frequency_components(feature_dim, max_position, tokens.device,
                                                                torch.float32)
        cos_comp, sin_comp = cos_comp.to(tokens.dtype), sin_comp.to(tokens.dtype)
        vertical, horizontal = tokens.chunk(2, dim=-1)
        vertical = self._apply_1d_rope(vertical, positions[..., 0], cos_comp, sin_comp)
        horizontal = self._apply_1d_rope(horizontal, positions[..., 1], cos_comp, sin_comp)
        return torch.cat((vertical, horizontal), dim=-1)

    def clear():
        import gc
        for obj in gc.get_objects():
            if isinstance(obj, cls):
                obj.frequency_cache.clear()

    clear()
    cls.forward = forward
    try:
        yield
    finally:
        cls.forward = original
        clear()


# MARK: reference-case inputs

def fp16_representable(x: np.ndarray) -> np.ndarray:
    """Rounds float32 values through float16, so the float32 tensor the host converts for the fp16 graph input and the
    tensor upstream sees are the same numbers."""
    return x.astype(np.float16).astype(np.float32)


def image_tensor(path: Path, size: int = SIZE) -> np.ndarray:
    """The reference cases' preprocessing: Pillow RGB, stretched to size x size with Pillow BICUBIC, x / 255, NCHW
    float32, fp16-representable. (The gate feeds this tensor directly; the host's own resize is gated separately.)"""
    from PIL import Image

    with Image.open(path) as im:
        rgb = im.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    x = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1)[None] / np.float32(255)
    return fp16_representable(np.ascontiguousarray(x))


def synthetic_tensor(seed: int, size: int = SIZE) -> np.ndarray:
    """A deterministic structured image (smooth gradients, a disc and seeded uniform noise) on [0,1]: exercises the graph
    away from natural-image statistics. Pure numpy, so the Swift side can rebuild it from the recipe if it ever needs to."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0, 1, size, dtype=np.float32), np.linspace(0, 1, size, dtype=np.float32),
                         indexing="ij")
    disc = (((xx - 0.6) ** 2 + (yy - 0.4) ** 2) < 0.04).astype(np.float32)
    channels = [0.6 * yy + 0.3 * disc, 0.5 * xx + 0.4 * (1 - disc) * yy, 0.4 * (1 - yy) + 0.5 * disc]
    x = np.stack(channels)[None] + 0.1 * rng.random((1, 3, size, size), dtype=np.float32)
    return fp16_representable(np.clip(x, 0, 1).astype(np.float32))


def rel_l2(got: np.ndarray, want: np.ndarray) -> float:
    got, want = got.astype(np.float64).ravel(), want.astype(np.float64).ravel()
    return float(np.linalg.norm(got - want) / (np.linalg.norm(want) + 1e-30))


def write_f32(path: Path, x: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(x, dtype="<f4").tobytes())


def read_f32(path: Path, shape) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype="<f4").reshape(shape).copy()

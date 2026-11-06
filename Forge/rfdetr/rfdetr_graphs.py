"""The RF-DETR nano (and seg-nano) export graph and the shared harness for its exporter, reference and checks.

RF-DETR (roboflow/rf-detr, Apache-2.0, Roboflow) is LW-DETR with a windowed DINOv2 ViT-S backbone, a multi-scale
projector, a two-stage deformable-attention decoder and, for the seg sizes, a segmentation head. The upstream PyTorch
code (tag 1.12.0) is imported read-only from the pinned unpack that `fetch_upstream.py` writes; nothing here modifies
it, and its package is never installed. The network is built and its checkpoint loaded by upstream's own builder and
loader (`build_model`, `load_pretrain_weights`), as `RFDETRInference` does.

The exported graph (`DetrGraph`) is upstream's own export path, `LWDETR.forward_export` after `LWDETR.export()` (the
graph upstream's ONNX, CoreML and Core AI exporters trace), with normalization in front:

* input `image` [1, 3, S, S] float32, RGB on [0, 1] (S = 384 for nano, 312 for seg-nano: the host stretches the image to
  S x S and scales by 1/255, as upstream's `predict` resizes to the model resolution without keeping the aspect ratio);
* ImageNet normalization (mean 0.485/0.456/0.406, std 0.229/0.224/0.225, upstream's `RFDETR.means`/`stds`) is in the
  graph, in float32;
* outputs `dets` [1, Q, 4] (boxes, cxcywh, normalized to the input), `labels` [1, Q, 91] (class logits before the
  sigmoid; column = COCO category id, columns 0 and the ten unused ids never score) and, for seg-nano, `masks`
  [1, Q, S/4, S/4] (mask logits; foreground is logit > 0). Q = 300 queries for nano, 100 for seg-nano. Outputs float32.

The names `dets`/`labels`/`masks` are upstream's own export names (and the zoo bundle's). The Core AI conversion uses
upstream's own decomposition table (`rfdetr.export._coreai.decompositions`: grid sampling as gathers, float16 topk in
float32) on top of coreai-torch's. Postprocess is never in the graph: upstream's `PostProcess` (sigmoid, top-k over
query x class pairs, threshold) runs on the host, and the gates run that same code on both sides.
"""
from __future__ import annotations

import copy
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
CODE_REPO, CODE_COMMIT = PINS["upstreamCode"]["repository"], PINS["upstreamCode"]["commit"]
CODE_TAG = PINS["upstreamCode"]["tag"]
CACHE = Path.home() / ".cache/aether-kit/rfdetr"
DEFAULT_UPSTREAM = CACHE / f"upstream/rf-detr-{CODE_COMMIT[:12]}"
DEFAULT_WEIGHTS, DEFAULT_IMAGES = CACHE / "weights", CACHE / "images"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASS_COLUMNS = 91           # num_classes 90 + 1: column index = COCO category id
SCORE_THRESHOLD = 0.5            # upstream `predict(threshold=0.5)` default
DTYPES = {"float32": torch.float32, "float16": torch.float16}


VARIANTS = {
    "nano": {
        "id": "rf-detr-nano", "config": "RFDETRNanoConfig", "checkpoint": "nano", "size": 384, "queries": 300,
        "outputs": ["dets", "labels"], "task": "object-detection", "title": "RF-DETR Nano",
        "asset": "rf_detr_nano.aimodel",
    },
    "seg-nano": {
        "id": "rf-detr-seg-nano", "config": "RFDETRSegNanoConfig", "checkpoint": "seg-nano", "size": 312,
        "queries": 100, "outputs": ["dets", "labels", "masks"], "task": "instance-segmentation",
        "title": "RF-DETR Seg Nano", "asset": "rf_detr_seg_nano.aimodel",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_path(variant: str, weights: Path = DEFAULT_WEIGHTS) -> Path:
    return weights / PINS["checkpoints"][VARIANTS[variant]["checkpoint"]]["file"]


def verify_sources(variant: str, upstream_root: Path, weights: Path = DEFAULT_WEIGHTS,
                   images: Path | None = DEFAULT_IMAGES) -> dict:
    """Checks the variant's checkpoint, the upstream unpack and (unless `images` is None) the reference photos against
    their pins; returns {name: sha256}."""
    verified = {}
    pin = PINS["checkpoints"][VARIANTS[variant]["checkpoint"]]
    path = checkpoint_path(variant, weights)
    if not path.is_file():
        raise SystemExit(f"{path} is missing; run Forge/rfdetr/fetch_upstream.py")
    if sha256_file(path) != pin["sha256"]:
        raise SystemExit(f"{path} differs from the pinned checkpoint")
    verified[f"checkpoint:{pin['file']}"] = pin["sha256"]
    stamp = upstream_root / "UNPACKED.json"
    if not stamp.is_file() or json.loads(stamp.read_text()).get("commit") != CODE_COMMIT:
        raise SystemExit(f"{upstream_root} is not the pinned upstream unpack; run Forge/rfdetr/fetch_upstream.py")
    for relative, expected in PINS["upstreamCode"]["files"].items():
        if sha256_file(upstream_root / relative) != expected:
            raise SystemExit(f"{relative} in the upstream unpack differs from the pin; re-run fetch_upstream.py")
        verified[f"upstream:{relative}"] = expected
    if images is not None:
        for name, image in PINS["referenceImages"]["files"].items():
            if not (images / name).is_file() or sha256_file(images / name) != image["sha256"]:
                raise SystemExit(f"reference photo {name} is missing or differs from the pin; run fetch_upstream.py")
            verified[f"image:{name}"] = image["sha256"]
    return verified


def load_upstream(upstream_root: Path) -> types.SimpleNamespace:
    """Imports the upstream modules the harness needs from the unpack, under an empty `rfdetr` package (so upstream's
    `rfdetr/__init__.py`, which imports its prediction, Supervision and training stack, never runs). Bytecode is not
    written into the unpack."""
    expected = str(upstream_root / "src/rfdetr")
    if "rfdetr" in sys.modules:
        if list(getattr(sys.modules["rfdetr"], "__path__", [])) != [expected]:
            raise RuntimeError("a different 'rfdetr' is already imported; run the harness in a fresh interpreter")
    else:
        package = types.ModuleType("rfdetr")
        package.__path__ = [expected]
        sys.modules["rfdetr"] = package
    sys.dont_write_bytecode = True
    import os
    import warnings
    warnings.filterwarnings("ignore", category=ImportWarning)        # rfdetr.platform: "plus" extras absent
    os.environ.setdefault("LOG_LEVEL", "WARNING")                    # upstream's logger: the loader's MD5 note
    names = {"config": "rfdetr.config", "namespace": "rfdetr._namespace", "lwdetr": "rfdetr.models.lwdetr",
             "weights": "rfdetr.models.weights", "postprocess": "rfdetr.models.postprocess",
             "coco": "rfdetr.assets.coco_classes", "io": "rfdetr.utilities.io",
             "decompositions": "rfdetr.export._coreai.decompositions"}
    return types.SimpleNamespace(**{k: importlib.import_module(v) for k, v in names.items()})


def build_net(variant: str, upstream_root: Path, weights: Path = DEFAULT_WEIGHTS):
    """The unmodified upstream LWDETR, float32, eval, built and loaded exactly as `RFDETRInference` builds it
    (`_namespace_from_configs` + `build_model` + `load_pretrain_weights`) from the pinned checkpoint. Upstream loads
    non-strictly, so this also checks that the checkpoint covers every parameter and buffer the network has (bar the
    keypoint head's derived class mask, which detection and segmentation never read).
    Returns (net, model_config, args)."""
    up = load_upstream(upstream_root)
    spec = VARIANTS[variant]
    path = checkpoint_path(variant, weights)
    mc = getattr(up.config, spec["config"])(pretrain_weights=str(path), device="cpu")
    args = up.namespace._namespace_from_configs(mc, up.config.TrainConfig(dataset_dir=None, output_dir="output"))
    args.dataset_dir, args.output_dir = None, "output"
    net = up.lwdetr.build_model(args)
    up.weights.load_pretrain_weights(net, mc)
    state = up.io._safe_torch_load(str(path))["model"]
    # `_kp_active_mask` is the keypoint head's class mask, derived from the (empty) keypoint schema, never stored.
    uncovered = sorted(k for k in net.state_dict() if k not in state and k != "_kp_active_mask")
    if uncovered:
        raise SystemExit(f"the checkpoint does not cover {uncovered[:5]} ({len(uncovered)} keys)")
    if (mc.resolution, args.num_queries, mc.num_classes + 1) != (spec["size"], spec["queries"], NUM_CLASS_COLUMNS):
        raise SystemExit(f"unexpected config: resolution {mc.resolution}, queries {args.num_queries}, "
                         f"classes {mc.num_classes}")
    return net.eval(), mc, args


def normalize(image: torch.Tensor) -> torch.Tensor:
    """Upstream `predict`'s normalization: `sub_(mean).div_(std)` with (C,1,1) float32 tensors."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).reshape(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).reshape(3, 1, 1)
    return image.to(torch.float32).clone().sub_(mean).div_(std)


def upstream_forward(net, image: torch.Tensor) -> dict[str, torch.Tensor]:
    """Upstream's own (non-export) `LWDETR.forward` on a raw [0,1] [1,3,S,S] float32 image, normalized as `predict`
    does. Returns {dets, labels[, masks]} under the export names."""
    with torch.no_grad():
        out = net(normalize(image))
    result = {"dets": out["pred_boxes"], "labels": out["pred_logits"]}
    if "pred_masks" in out:
        result["masks"] = out["pred_masks"]
    return result


class DetrGraph(nn.Module):
    """[1,3,S,S] raw RGB on [0,1] (float32) -> dets, labels[, masks] (float32).

    Wraps a deep copy of `net` switched to upstream's export forward (`LWDETR.export()`), cast to `dtype`.
    Normalization runs in float32 before the cast; the outputs are cast back to float32."""

    def __init__(self, net, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        exported = copy.deepcopy(net)
        exported.export()
        self.net = exported.to(dtype).eval()
        self.dtype = dtype
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN, dtype=torch.float32).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD, dtype=torch.float32).reshape(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = ((image.to(torch.float32) - self.mean) / self.std).to(self.dtype)
        return tuple(t.to(torch.float32) for t in self.net(x))


def decomposition_table(upstream_root: Path) -> dict:
    """coreai-torch's decomposition table plus upstream's Core AI entries (grid sampling as gathers; float16 topk in
    float32)."""
    import coreai_torch as ct
    return load_upstream(upstream_root).decompositions.coreai_decomposition_table(ct.get_decomp_table())


# MARK: postprocess and detection matching (identical on both sides of every gate)

def postprocess(upstream_root: Path, variant: str, out: dict[str, np.ndarray],
                threshold: float = SCORE_THRESHOLD) -> dict[str, np.ndarray]:
    """Upstream's `PostProcess` (num_select = the variant's default: 300 nano, 100 seg-nano) on raw outputs, with target
    size S x S, then upstream `predict`'s `scores > threshold` filter. Boxes come back xyxy in input pixels; masks
    (seg-nano) as booleans at S x S (bilinear upsample, logit > 0)."""
    up = load_upstream(upstream_root)
    spec = VARIANTS[variant]
    post = up.postprocess.PostProcess(num_select=spec["queries"])
    raw = {"pred_boxes": torch.from_numpy(np.ascontiguousarray(out["dets"], dtype=np.float32)),
           "pred_logits": torch.from_numpy(np.ascontiguousarray(out["labels"], dtype=np.float32))}
    if "masks" in out:
        raw["pred_masks"] = torch.from_numpy(np.ascontiguousarray(out["masks"], dtype=np.float32))
    result = post(raw, torch.tensor([[spec["size"], spec["size"]]]))[0]
    keep = (result["scores"] > threshold).nonzero(as_tuple=True)[0]
    return {k: v[keep].numpy() for k, v in result.items()}


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xyxy boxes, [len(a), len(b)]."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: np.clip(x[:, 2:] - x[:, :2], 0, None).prod(-1)  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-12)


def match_detections(want: dict[str, np.ndarray], got: dict[str, np.ndarray],
                     threshold: float = SCORE_THRESHOLD, score_tolerance: float = 0.01) -> dict:
    """Matches detections class by class (Hungarian on box IoU, pairs with IoU > 0.5). Returns the class sets, the
    matched pairs' worst box IoU and score difference (and mask IoU when both carry masks), and the unmatched
    detections. An unmatched detection is "borderline" when its score is within `score_tolerance` of the threshold,
    i.e. it may legitimately fall on either side of it."""
    from scipy.optimize import linear_sum_assignment

    pairs, unmatched = [], []
    classes = sorted(set(want["labels"].tolist()) | set(got["labels"].tolist()))
    for c in classes:
        wi, gi = np.flatnonzero(want["labels"] == c), np.flatnonzero(got["labels"] == c)
        used_w, used_g = set(), set()
        if len(wi) and len(gi):
            iou = box_iou(want["boxes"][wi], got["boxes"][gi])
            rows, cols = linear_sum_assignment(-iou)
            for r, k in zip(rows, cols):
                if iou[r, k] > 0.5:
                    pair = {"class": int(c), "iou": float(iou[r, k]),
                            "want": float(want["scores"][wi[r]]), "got": float(got["scores"][gi[k]])}
                    if "masks" in want and "masks" in got:
                        m1, m2 = want["masks"][wi[r]].reshape(-1), got["masks"][gi[k]].reshape(-1)
                        union = np.logical_or(m1, m2).sum()
                        pair["maskIoU"] = float(np.logical_and(m1, m2).sum() / union) if union else 1.0
                    pairs.append(pair)
                    used_w.add(r)
                    used_g.add(k)
        for side, idx, used, src in (("reference", wi, used_w, want), ("graph", gi, used_g, got)):
            for j, i in enumerate(idx):
                if j not in used:
                    score = float(src["scores"][i])
                    unmatched.append({"side": side, "class": int(c), "score": score,
                                      "borderline": score <= threshold + score_tolerance})
    return {
        "referenceCount": int(len(want["scores"])), "graphCount": int(len(got["scores"])),
        "referenceClasses": sorted(set(want["labels"].tolist())), "graphClasses": sorted(set(got["labels"].tolist())),
        "matched": len(pairs),
        "minIoU": min((p["iou"] for p in pairs), default=1.0),
        "maxScoreDiff": max((abs(p["want"] - p["got"]) for p in pairs), default=0.0),
        "minMaskIoU": min((p["maskIoU"] for p in pairs if "maskIoU" in p), default=None),
        "unmatched": unmatched,
    }


# MARK: reference-case inputs

def image_tensor(path: Path, size: int) -> np.ndarray:
    """The reference cases' preprocessing, upstream `predict`'s: Pillow decode, `convert("RGB")`, uint8 CHW to float32
    / 255 (torchvision `to_dtype(scale=True)`), `F.resize(..., [size, size], antialias=False)` (bilinear, stretch), NCHW.
    (The gate feeds this tensor directly; the host's own resize is gated separately.)"""
    from PIL import Image
    import torchvision.transforms.v2.functional as F

    with Image.open(path) as im:
        rgb = im.convert("RGB")
    t = F.to_dtype(F.to_image(rgb), torch.float32, scale=True)
    t = F.resize(t, [size, size], antialias=False)
    return np.ascontiguousarray(t[None].numpy(), dtype=np.float32)


def synthetic_tensor(seed: int, size: int) -> np.ndarray:
    """A deterministic structured image (smooth gradients, a disc and seeded uniform noise) on [0,1]."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0, 1, size, dtype=np.float32), np.linspace(0, 1, size, dtype=np.float32),
                         indexing="ij")
    disc = (((xx - 0.6) ** 2 + (yy - 0.4) ** 2) < 0.04).astype(np.float32)
    channels = [0.6 * yy + 0.3 * disc, 0.5 * xx + 0.4 * (1 - disc) * yy, 0.4 * (1 - yy) + 0.5 * disc]
    x = np.stack(channels)[None] + 0.1 * rng.random((1, 3, size, size), dtype=np.float32)
    return np.clip(x, 0, 1).astype(np.float32)


def output_shapes(variant: str) -> dict[str, list[int]]:
    spec = VARIANTS[variant]
    s, q = spec["size"], spec["queries"]
    shapes = {"input": [1, 3, s, s], "dets": [1, q, 4], "labels": [1, q, NUM_CLASS_COLUMNS]}
    if "masks" in spec["outputs"]:
        shapes["masks"] = [1, q, s // 4, s // 4]
    return shapes


def coco_labels(upstream_root: Path) -> list[str | None]:
    """Upstream's COCO table as a 91-entry list indexed by class-logit column (= COCO category id); None for column 0
    and the ten ids COCO never assigned."""
    table = load_upstream(upstream_root).coco.COCO_CLASSES
    return [table.get(i) for i in range(NUM_CLASS_COLUMNS)]


def rel_l2(got: np.ndarray, want: np.ndarray) -> float:
    got, want = got.astype(np.float64).ravel(), want.astype(np.float64).ravel()
    return float(np.linalg.norm(got - want) / (np.linalg.norm(want) + 1e-30))


def write_f32(path: Path, x: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(x, dtype="<f4").tobytes())


def read_f32(path: Path, shape) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype="<f4").reshape(shape).copy()

"""The YOLOX-S export graph and the shared harness for its exporter, reference and checks.

YOLOX-S (Megvii-BaseDetection/YOLOX, Apache-2.0, Megvii) is a CSPDarknet + PAFPN backbone with the decoupled,
anchor-free YOLOX head, trained on COCO's 80 detection classes. The upstream PyTorch code is imported read-only from the
pinned unpack that `fetch_upstream.py` writes; nothing here modifies it, and its package is never installed. The
checkpoint is the 0.1.1rc0 release `yolox_s.pth` (the non-legacy weights: BGR on 0..255, no normalization).

The exported graph (`YoloxGraph`) is upstream `YOLOX.forward` in eval mode, i.e. with the head's own inference decode:

* input `image` [1, 3, 640, 640], BGR on 0..255 (no scaling, no mean/std), letterboxed top-left with pad 114: the host
  resizes by r = min(640 / h, 640 / w) to (int(w r), int(h r)) and pads right and bottom with 114 (upstream's
  `yolox.data.data_augment.preproc`);
* output `preds` [1, 8400, 85] float32: per anchor (80 x 80 at stride 8, then 40 x 40 at 16, then 20 x 20 at 32, row
  major), cx, cy, w, h in letterboxed input pixels (the grid offsets and exp(wh) * stride are decoded in the graph,
  by upstream's `YOLOXHead.decode_outputs`), then objectness and the 80 class probabilities (sigmoid applied in the
  graph). Class index c is COCO's c-th detection class in category-id order (`COCO_CLASSES`, `COCO91_IDS`).

The host does the rest, as upstream's `yolox.utils.postprocess` does: score = obj * max(cls), class = argmax(cls), keep
score >= threshold, greedy per-class NMS at IoU 0.45 on the xyxy boxes, then the top `maxDetections` by score.
`dense_nms` is that rule in numpy (the reference the Swift driver is gated against); `selftest_yolox.py` checks it
against upstream's own torchvision-based `postprocess`.

Precision: `YoloxGraph(net, dtype)` runs every conv (and BN, SiLU, sigmoid) in `dtype` and the box decode in float32,
so the decoded coordinates do not inherit float16's 0.25-0.5 px spacing above 256 px. The exporter's default is
float32, exactly upstream (see export_yolox.py for the measured float16 comparison).
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
import logging
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "source-pins.json").read_text())
REPO = PINS["repository"]
WEIGHTS = PINS["weights"]
CODE_REPO, CODE_COMMIT = PINS["upstreamCode"]["repository"], PINS["upstreamCode"]["commit"]
DEFAULT_WEIGHTS = Path.home() / f".cache/aether-kit/yolox/weights/{WEIGHTS['release']}/{WEIGHTS['file']}"
DEFAULT_UPSTREAM = Path.home() / f".cache/aether-kit/yolox/upstream/yolox-{CODE_COMMIT[:12]}"

SIZE = 640                     # upstream's test_size for yolox-s
PAD = 114                      # upstream preproc's letterbox fill
STRIDES = (8, 16, 32)
NUM_CLASSES = 80
ANCHORS = sum((SIZE // s) ** 2 for s in STRIDES)          # 8400
CHANNELS = 5 + NUM_CLASSES                                # cx, cy, w, h, obj, cls[80]
SCORE_THRESHOLD, NMS_IOU, MAX_DETECTIONS = 0.3, 0.45, 50  # the driver defaults (docs/M4.md)
DTYPES = {"float32": torch.float32, "float16": torch.float16}
# COCO 2017's 80 detection category ids in ascending order: class index c -> COCO91_IDS[c] (the "coco-91" id space,
# 1..90 with 12, 26, 29, 30, 45, 66, 68, 69, 71 and 83 unused). Upstream's COCODataset maps contiguously the same way.
COCO91_IDS = tuple(i for i in range(1, 91) if i not in (12, 26, 29, 30, 45, 66, 68, 69, 71, 83))
assert len(COCO91_IDS) == NUM_CLASSES


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sources(weights: Path, upstream_root: Path) -> dict:
    """Checks the pinned checkpoint and the upstream unpack; returns {file: sha256}."""
    if not weights.is_file():
        raise SystemExit(f"no checkpoint at {weights}; run Forge/yolox/fetch_upstream.py")
    if sha256_file(weights) != WEIGHTS["sha256"]:
        raise SystemExit(f"{weights} differs from the pinned {WEIGHTS['release']} {WEIGHTS['file']}")
    verified = {WEIGHTS["file"]: WEIGHTS["sha256"]}
    stamp = upstream_root / "UNPACKED.json"
    if not stamp.is_file() or json.loads(stamp.read_text()).get("commit") != CODE_COMMIT:
        raise SystemExit(f"{upstream_root} is not the pinned upstream unpack; run Forge/yolox/fetch_upstream.py")
    for group in (PINS["upstreamCode"]["files"], PINS["referenceImages"]["files"]):
        for relative, expected in group.items():
            if sha256_file(upstream_root / relative) != expected:
                raise SystemExit(f"{relative} in the upstream unpack differs from the pin; re-run fetch_upstream.py")
            verified[f"upstream:{relative}"] = expected
    return verified


def load_upstream(upstream_root: Path):
    """Imports `yolox.models` and `yolox.utils.boxes` from the unpack, without upstream's `yolox/__init__.py` and
    `yolox/utils/__init__.py` (which import its training, logging, opencv and evaluation stack).

    `yolox.utils` is a package whose namespace holds only what `yolo_head.py` imports from it: `bboxes_iou` and
    `cxcywh2xyxy` (from upstream boxes.py), `meshgrid` (upstream compat.py) and `visualize_assign` (only reached by
    `YOLOX.visualize`, so it raises). `yolo_head.py` also imports loguru's `logger`, used only in the training loss;
    when loguru is not installed a `logging` stand-in is registered for it."""
    root = upstream_root / "yolox"
    existing = sys.modules.get("yolox")
    if existing is not None:
        if list(getattr(existing, "__path__", [])) != [str(root)]:
            raise RuntimeError("a different 'yolox' is already imported; run the harness in a fresh interpreter")
    else:
        if importlib.util.find_spec("loguru") is None and "loguru" not in sys.modules:
            loguru = types.ModuleType("loguru")
            loguru.logger = logging.getLogger("yolox")
            sys.modules["loguru"] = loguru
        package = types.ModuleType("yolox")
        package.__path__ = [str(root)]
        sys.modules["yolox"] = package
        utils = types.ModuleType("yolox.utils")
        utils.__path__ = [str(root / "utils")]
        sys.modules["yolox.utils"] = utils
        package.utils = utils
        boxes = importlib.import_module("yolox.utils.boxes")
        compat = importlib.import_module("yolox.utils.compat")

        def visualize_assign(*_args, **_kwargs):
            raise NotImplementedError("yolox.utils.visualize is not part of the Forge harness")

        utils.bboxes_iou, utils.cxcywh2xyxy = boxes.bboxes_iou, boxes.cxcywh2xyxy
        utils.meshgrid, utils.visualize_assign, utils.postprocess = compat.meshgrid, visualize_assign, boxes.postprocess
    models = importlib.import_module("yolox.models")
    boxes = importlib.import_module("yolox.utils.boxes")
    return models, boxes


def _assigned_constants(path: Path) -> dict:
    """`self.<name> = <literal>` assignments in an upstream Exp file."""
    found = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Attribute)
                and isinstance(node.targets[0].value, ast.Name) and node.targets[0].value.id == "self"):
            try:
                found[node.targets[0].attr] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return found


def coco_classes(upstream_root: Path) -> tuple[str, ...]:
    """Upstream's COCO_CLASSES (yolox/data/datasets/coco_classes.py), read as a literal."""
    tree = ast.parse((upstream_root / "yolox/data/datasets/coco_classes.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "COCO_CLASSES":
            names = tuple(ast.literal_eval(node.value))
            if len(names) != NUM_CLASSES:
                raise SystemExit(f"COCO_CLASSES has {len(names)} names")
            return names
    raise SystemExit("COCO_CLASSES not found")


def build_net(upstream_root: Path, weights: Path):
    """The unmodified upstream YOLOX, float32, eval, with the pinned yolox_s.pth weights (strict load). Built as
    upstream's `yolox.exp.yolox_base.Exp.get_model` builds it (in_channels 256/512/1024, BatchNorm eps 1e-3), with
    depth and width read from the pinned `exps/default/yolox_s.py` and the remaining fields from `yolox_base.py`."""
    models, _ = load_upstream(upstream_root)
    base = _assigned_constants(upstream_root / "yolox/exp/yolox_base.py")
    exp = {**base, **_assigned_constants(upstream_root / "exps/default/yolox_s.py")}
    if (exp["depth"], exp["width"], exp["act"], exp["num_classes"], tuple(exp["test_size"])) != (
            0.33, 0.50, "silu", NUM_CLASSES, (SIZE, SIZE)):
        raise SystemExit(f"unexpected yolox-s experiment: {exp}")
    in_channels = [256, 512, 1024]
    backbone = models.YOLOPAFPN(exp["depth"], exp["width"], in_channels=in_channels, act=exp["act"])
    head = models.YOLOXHead(exp["num_classes"], exp["width"], in_channels=in_channels, act=exp["act"])
    net = models.YOLOX(backbone, head)
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps, m.momentum = 1e-3, 0.03
    if tuple(head.strides) != STRIDES:
        raise SystemExit(f"unexpected strides {head.strides}")
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    net.load_state_dict(checkpoint["model"], strict=True)
    return net.eval()


def upstream_preds(net, image: torch.Tensor) -> torch.Tensor:
    """Upstream's own eval forward (decode in inference) on a [1,3,640,640] float32 BGR 0..255 tensor: [1,8400,85]."""
    if not net.head.decode_in_inference:
        raise RuntimeError("upstream_preds needs a net with the head's inference decode on")
    with torch.no_grad():
        return net(image.to(torch.float32))


class YoloxGraph(nn.Module):
    """[1,3,640,640] BGR on 0..255 (`dtype`) -> preds [1,8400,85] float32 (decoded cxcywh letterbox pixels, obj, cls).

    The convolutional network runs in `dtype`; the head's raw output (box regression, and objectness and class
    probabilities already through upstream's sigmoid) is cast to float32 and decoded there by upstream's own
    `decode_outputs`. With dtype float32 this is exactly upstream's eval forward."""

    def __init__(self, net, dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self.net, self.dtype = net.to(dtype), dtype
        self.net.head.decode_in_inference = False

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        raw = self.net(image.to(self.dtype))
        return self.net.head.decode_outputs(raw.to(torch.float32), dtype=torch.float32)


# MARK: host-side postprocess (the reference rule for `postprocess.denseNms`)

def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    cx, cy, w, h = (boxes[..., i] for i in range(4))
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of xyxy boxes a [N,4] against b [M,4] (torchvision's convention: no +1)."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: np.clip(x[:, 2] - x[:, 0], 0, None) * np.clip(x[:, 3] - x[:, 1], 0, None)  # noqa: E731
    union = area(a)[:, None] + area(b)[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def dense_nms(preds: np.ndarray, score_threshold: float = SCORE_THRESHOLD, iou: float = NMS_IOU,
              max_detections: int = MAX_DETECTIONS) -> list[dict]:
    """`postprocess.denseNms` on one image's preds [8400,85] (or [1,8400,85]), in float64 from the float32 values:
    score = obj * max(cls) and class = argmax(cls) (first index on ties); keep score >= threshold; per class, greedy
    NMS in descending score order suppressing IoU > `iou` (xyxy, no +1); then all classes merged, descending score, the
    first `max_detections`. Stable order throughout (ties keep anchor order). Returns
    [{anchor, classIndex, score, box: [cx, cy, w, h]}] in letterbox pixels."""
    p = preds.reshape(-1, preds.shape[-1]).astype(np.float64)
    cls = p[:, 5:]
    klass = cls.argmax(1)
    score = p[:, 4] * cls[np.arange(len(p)), klass]
    candidates = np.flatnonzero(score >= score_threshold)
    keep: list[int] = []
    for c in np.unique(klass[candidates]):
        idx = candidates[klass[candidates] == c]
        idx = idx[np.argsort(-score[idx], kind="stable")]
        boxes = cxcywh_to_xyxy(p[idx, :4])
        alive = np.ones(len(idx), bool)
        for i in range(len(idx)):
            if not alive[i]:
                continue
            keep.append(int(idx[i]))
            overlap = iou_matrix(boxes[i:i + 1], boxes[i + 1:])[0]
            alive[i + 1:] &= overlap <= iou
    keep = sorted(keep, key=lambda a: (-score[a], a))[:max_detections]
    return [{"anchor": a, "classIndex": int(klass[a]), "score": float(score[a]),
             "box": [float(v) for v in p[a, :4]]} for a in keep]


def match_detections(got: list[dict], want: list[dict]) -> dict:
    """Pairs each reference detection with the unpaired same-class detection of highest IoU (references in descending
    score order). Returns the pairs' minimum IoU, maximum score difference, and the unpaired detections on each side."""
    pairs, used = [], set()
    for w in sorted(want, key=lambda d: -d["score"]):
        best, best_iou = None, -1.0
        for j, g in enumerate(got):
            if j in used or g["classIndex"] != w["classIndex"]:
                continue
            v = float(iou_matrix(cxcywh_to_xyxy(np.array([g["box"]])), cxcywh_to_xyxy(np.array([w["box"]])))[0, 0])
            if v > best_iou:
                best, best_iou = j, v
        if best is None:
            continue
        used.add(best)
        pairs.append({"classIndex": w["classIndex"], "iou": best_iou, "scoreDiff": abs(got[best]["score"] - w["score"]),
                      "score": w["score"]})
    unmatched_want = len(want) - len(pairs)
    return {"reference": len(want), "graph": len(got), "matched": len(pairs),
            "unmatchedReference": unmatched_want, "unmatchedGraph": len(got) - len(pairs),
            "sameClassSet": sorted(d["classIndex"] for d in got) == sorted(d["classIndex"] for d in want),
            "minIoU": min((p["iou"] for p in pairs), default=1.0),
            "maxScoreDiff": max((p["scoreDiff"] for p in pairs), default=0.0)}


# MARK: reference-case inputs

def letterbox_tensor(path: Path, crop: tuple[int, int, int, int] | None = None) -> tuple[np.ndarray, dict]:
    """The reference cases' preprocessing, upstream's `preproc` geometry with Pillow in place of OpenCV: Pillow
    convert('RGB') (alpha dropped), optional crop (left, top, right, bottom), r = min(640 / h, 640 / w), resize to
    (int(w r), int(h r)) with Pillow BILINEAR, paste top-left on a 640 x 640 canvas of 114, channels reversed to BGR,
    NCHW float32 on 0..255 (integers, so exact in float16). (The gate feeds this tensor directly; the host's own
    letterbox is gated separately.)"""
    from PIL import Image

    with Image.open(path) as im:
        rgb = im.convert("RGB")
        if crop is not None:
            rgb = rgb.crop(crop)
        w, h = rgb.size
        r = min(SIZE / h, SIZE / w)
        size = (int(w * r), int(h * r))
        resized = rgb.resize(size, Image.Resampling.BILINEAR)
    canvas = np.full((SIZE, SIZE, 3), PAD, dtype=np.uint8)
    canvas[: size[1], : size[0]] = np.asarray(resized)
    x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)
    return np.ascontiguousarray(x), {"source": [w, h], "ratio": r, "resized": list(size)}


def synthetic_tensor(seed: int) -> np.ndarray:
    """A deterministic structured BGR image on 0..255 (integers): smooth gradients, two discs and seeded noise in a
    640 x 480 top region, the bottom 160 rows the letterbox pad 114. Exercises the graph away from natural images."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0, 1, 480, dtype=np.float32), np.linspace(0, 1, SIZE, dtype=np.float32),
                         indexing="ij")
    disc = (((xx - 0.3) ** 2 + (yy - 0.5) ** 2) < 0.03).astype(np.float32)
    disc2 = (((xx - 0.75) ** 2 + (yy - 0.4) ** 2) < 0.01).astype(np.float32)
    channels = [200 * yy + 50 * disc, 180 * xx + 60 * disc2, 150 * (1 - yy) + 90 * disc]
    top = np.stack(channels) + 25 * rng.random((3, 480, SIZE), dtype=np.float32)
    x = np.full((1, 3, SIZE, SIZE), PAD, dtype=np.float32)
    x[0, :, :480] = np.clip(np.round(top), 0, 255)
    return x


def rel_l2(got: np.ndarray, want: np.ndarray) -> float:
    got, want = got.astype(np.float64).ravel(), want.astype(np.float64).ravel()
    return float(np.linalg.norm(got - want) / (np.linalg.norm(want) + 1e-30))


def write_f32(path: Path, x: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(x, dtype="<f4").tobytes())


def read_f32(path: Path, shape) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype="<f4").reshape(shape).copy()

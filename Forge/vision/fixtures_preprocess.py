"""Records the vision preprocessing known answers (MANIFEST.md §2.5) from Pillow and torch, for the hermetic Swift suite
(Tests/AetherRuntimeTests/VisionPreprocessTests.swift). No model is loaded.

    Forge/.venv/bin/python -I Forge/vision/fixtures_preprocess.py Tests/AetherRuntimeTests/Fixtures/vision

Writes, into the directory (replacing what is there):
- `preprocess.json`:
  - `resizes`: seeded noise images resampled by Pillow's `Image.resize(BICUBIC)` and `Image.resize(BILINEAR)` and by
    torch's uint8 `interpolate(bicubic, antialias=True)` (which torchvision's `resize` of a uint8 tensor calls); the
    8-bit parity filters must reproduce each bit for bit.
  - `floatResizes`: seeded noise images as torchvision makes them float (`to_dtype(…, scale=True)`, a float32 multiply
    by 1/255) and resized by `resize(…, antialias=False)` (torch's float32 `upsample_bilinear2d`), stored as float32:
    `torchBilinear` must reproduce them to the bit. torch picks its CPU kernel by the output size (height + width up to
    128: channels-last; above: the separable generic kernel), so the cases straddle that line. Run multi-threaded, as
    the references are (a single thread switches every 3-channel resize to the channels-last kernel).
  - `letterboxGeometry`: the resized size YOLOX's preproc gives (`r = min(H / h, W / w)`, `int(side * r)`, at least
    1), including the demo strip (2443 x 424 into 640: 639 x 111, one short of the canvas).
  - `vImage`: smooth images and Pillow's `LANCZOS` of them, the nearest Pillow kernel to Accelerate's `vImageScale`.
    The fast filter isn't a parity filter; the suite pins how close it stays.
  - `pipelines`: whole `preprocess.image` blocks (as a manifest carries them) with the tensor torch computes for them:
    the resize geometry (stretch, letterbox, centre crop, fit-long-side), `(x * scale - mean) / std` in float32, channel
    order, and the NCHW, NVCHW and patch layouts (Qwen2-VL's block-major patchify written as Transformers'
    `Qwen2VLImageProcessorFast` reshapes it; SigLIP 2's row-major `py,px,C` as `convert_image_to_patches` does), stored
    as float32 or float16 bytes.
- `alpha/`: 0.0.x's alpha-policy PNGs (8-bit RGBA, grey plus alpha, a palette with tRNS, 16-bit RGBA) and what
  Transformers' `load_image` (`exif_transpose(...).convert("RGB")`) decodes from each, in `expected.json`.

Byte arrays are base64 (uint8; float32 and float16 little-endian).
"""
import base64
import hashlib
import json
import shutil
import struct
import sys
import zlib
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import PIL
import torch
from PIL import Image, ImageOps

rng = np.random.default_rng(20261106)


def b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).astype(array.dtype.newbyteorder("<")).tobytes()).decode()


def noise(h: int, w: int) -> np.ndarray:
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def smooth(h: int, w: int) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    planes = [x * 255 / max(w - 1, 1), y * 255 / max(h - 1, 1), 128 + 100 * np.sin(x / 7.0 + y / 5.0)]
    return np.stack(planes, -1).round().clip(0, 255).astype(np.uint8)


def pillow(image: np.ndarray, w: int, h: int, f=Image.Resampling.BICUBIC) -> np.ndarray:
    return np.asarray(Image.fromarray(image).resize((w, h), f))


def torch_uint8(image: np.ndarray, w: int, h: int) -> np.ndarray:
    t = torch.from_numpy(image).permute(2, 0, 1)[None]
    out = torch.nn.functional.interpolate(t, size=(h, w), mode="bicubic", antialias=True, align_corners=False)
    return out[0].permute(1, 2, 0).numpy()


def resample(image: np.ndarray, w: int, h: int, filter: str) -> np.ndarray:
    if image.shape[1] == w and image.shape[0] == h:
        return image
    if filter == "pillowBicubic":
        return pillow(image, w, h)
    if filter == "pillowBilinear":
        return pillow(image, w, h, Image.Resampling.BILINEAR)
    return torch_uint8(image, w, h)


def torch_float_bilinear(image: np.ndarray, w: int, h: int, scale: float = 1 / 255) -> torch.Tensor:
    """torchvision's float path (RF-DETR `predict`): uint8 CHW to float32 times `scale`, then `resize(antialias=False)`."""
    import torchvision.transforms.v2.functional as F

    t = torch.from_numpy(image).permute(2, 0, 1).to(torch.float32).mul_(scale)
    return t if t.shape[1:] == (h, w) else F.resize(t, [h, w], antialias=False)


def letterbox_size(w: int, h: int, W: int, H: int) -> tuple[int, int]:
    """YOLOX's preproc: r = min(H / h, W / w), each side int(side * r) (at least 1 here; upstream would give 0)."""
    r = min(H / h, W / w)
    return max(1, int(w * r)), max(1, int(h * r))


def geometry(image: np.ndarray, resize: dict | None):
    """The canvas (uint8 HxWx3) the resize step produces, as the Swift preprocessor defines it."""
    h, w = image.shape[:2]
    if resize is None:
        return image
    mode, f = resize["mode"], resize["filter"]
    if f == "torchBilinear":
        return image                                    # resized in float by tensor()
    if mode == "stretch":
        W, H = resize["size"]
        return resample(image, W, H, f)
    if mode == "letterbox":
        W, H = resize["size"]
        nw, nh = letterbox_size(w, h, W, H)
        ox, oy = (0, 0) if resize.get("anchor") == "topLeft" else ((W - nw) // 2, (H - nh) // 2)
        pad = resize.get("padValue", [0])
        canvas = np.empty((H, W, 3), np.uint8)
        canvas[:] = pad if len(pad) == 3 else pad * 3
        canvas[oy:oy + nh, ox:ox + nw] = resample(image, nw, nh, f)
        return canvas
    if mode == "centerCrop":
        W, H = resize["size"]
        rw, rh = (W, max(H, h * W // w)) if W * h >= H * w else (max(W, w * H // h), H)
        scaled = resample(image, rw, rh, f)
        # torchvision's center_crop offsets.
        cx, cy = int(round((rw - W) / 2.0)), int(round((rh - H) / 2.0))
        return scaled[cy:cy + H, cx:cx + W]
    if mode == "fitLongSide":
        m = resize["maxSide"]
        if max(w, h) <= m:
            return image
        nw, nh = (m, max(1, h * m // w)) if w >= h else (max(1, w * m // h), m)
        return resample(image, nw, nh, f)
    raise ValueError(mode)


def tensor(canvas: np.ndarray, block: dict) -> torch.Tensor:
    mean = torch.tensor(block.get("mean", [0, 0, 0]), dtype=torch.float32)[:, None, None]
    std = torch.tensor(block.get("std", [1, 1, 1]), dtype=torch.float32)[:, None, None]
    resize = block.get("resize")
    if resize is not None and resize["filter"] == "torchBilinear":
        # Float resize between the scale and the normalization (`canvas` is the source; stretch only).
        W, H = resize["size"]
        x = (torch_float_bilinear(canvas, W, H, block.get("scale", 1.0)) - mean) / std
    else:
        x = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).to(torch.float32)          # C, H, W in RGB
        x = (x * block.get("scale", 1.0) - mean) / std
    if block.get("channelOrder") == "BGR":
        x = x.flip(0)
    layout = block.get("layout", "NCHW")
    if layout == "NCHW":
        return x[None]
    if layout == "NVCHW":
        return x[None, None].repeat(1, block.get("views", 1), 1, 1, 1)
    p = block["patchify"]
    patch, t, merge = p["patch"], p.get("temporal", 1), p.get("merge", 1)
    c, h, w = x.shape
    gh, gw = h // patch, w // patch
    if p["rowOrder"] == "blockMajor" and p["withinPatch"] == "C,T,py,px":
        # Qwen2VLImageProcessorFast._preprocess, one image, one frame repeated temporal times.
        patches = x[None, None].repeat(1, t, 1, 1, 1)
        patches = patches.view(1, 1, t, c, gh // merge, merge, patch, gw // merge, merge, patch)
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        return patches.reshape(gh * gw, c * t * patch * patch)
    if p["rowOrder"] == "rowMajor" and p["withinPatch"] == "py,px,C":
        # Siglip2 convert_image_to_patches (channels last).
        hwc = x.permute(1, 2, 0)
        patches = hwc.reshape(gh, patch, gw, patch, c).permute(0, 2, 1, 3, 4)
        return patches.reshape(gh * gw, patch * patch * c)
    raise ValueError(p)


def alpha_fixtures(out: Path) -> None:
    """0.0.x's Forge/language/write_alpha_fixtures.py: distinct stored colours under alpha 0, 1, 64, 128, 254, 255."""
    W, H = 4, 3
    alphas = [0, 1, 64, 128, 254, 255]
    px = np.zeros((H, W, 4), dtype=np.uint8)
    for i in range(W * H):
        y, x = divmod(i, W)
        px[y, x] = [(37 * i + 11) % 256, (91 * i + 7) % 256, (53 * i + 200) % 256, alphas[i % len(alphas)]]
    palette = Image.new("P", (W, H))
    palette.putpalette([200, 30, 40, 10, 220, 90, 60, 70, 230, 250, 250, 10] + [0] * (256 * 3 - 12))
    palette.putdata([i % 4 for i in range(W * H)])
    palette.info["transparency"] = bytes([255, 0, 255, 0] + [255] * 252)
    images = {"rgba8": Image.fromarray(px, "RGBA"), "gray-alpha8": Image.fromarray(np.stack([px[..., 0], px[..., 3]], -1), "LA"),
              "rgba16": None, "palette-trns": palette}
    out.mkdir(parents=True)
    record = {"kind": "vision-alpha-fixtures", "rule": "Transformers load_image: ImageOps.exif_transpose(image).convert('RGB')",
              "size": [W, H], "fixtures": {}}
    for name, image in images.items():
        path = out / f"{name}.png"
        if image is None:
            # 16-bit RGBA: high byte the 8-bit value, low byte different, so the 8-bit reduction is checked.
            wide = px.astype(np.uint16) * 256 + (255 - px.astype(np.uint16))
            rows = b"".join(b"\x00" + wide[y].astype(">u2").tobytes() for y in range(H))

            def chunk(kind: bytes, data: bytes) -> bytes:
                return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

            path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 16, 6, 0, 0, 0))
                             + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
        else:
            image.save(path)
        decoded = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        record["fixtures"][name] = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                    "mode": Image.open(path).mode, "upstreamRGB": np.asarray(decoded).reshape(-1).tolist()}
    (out / "expected.json").write_text(json.dumps(record, indent=1) + "\n")


def main() -> None:
    if torch.get_num_threads() < 2:
        raise SystemExit("run multi-threaded, as the references are: torch picks another bilinear kernel on one thread")
    out = Path(sys.argv[1])
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    resizes = []
    # Up, down, one axis only, odd sizes, extreme ratios, a 1-pixel side.
    for h, w, H, W in [(37, 53, 96, 80), (120, 80, 36, 26), (17, 9, 40, 13), (64, 64, 64, 31), (64, 64, 29, 64),
                       (101, 77, 16, 16), (5, 7, 80, 60), (1, 40, 3, 7), (40, 40, 84, 84), (90, 120, 37, 61)]:
        image = noise(h, w)
        resizes.append({"width": w, "height": h, "targetWidth": W, "targetHeight": H, "input": b64(image),
                        "pillowBicubic": b64(pillow(image, W, H)), "torchUint8Bicubic": b64(torch_uint8(image, W, H)),
                        "pillowBilinear": b64(pillow(image, W, H, Image.Resampling.BILINEAR))})

    float_resizes = []
    # Up, down, mixed, one axis only, odd sizes, a 1-pixel side; output height + width on both sides of 128 (64 + 64 and
    # 65 + 64 straddle it); the RF-DETR shape class (a large photo down to a square).
    for h, w, H, W in [(37, 53, 96, 80), (120, 80, 36, 26), (17, 9, 40, 13), (64, 64, 64, 31), (64, 64, 29, 64),
                       (91, 77, 64, 64), (91, 77, 65, 64), (50, 50, 70, 59), (5, 7, 80, 60), (1, 40, 3, 7),
                       (90, 120, 37, 61), (110, 190, 66, 66)]:
        image = noise(h, w)
        float_resizes.append({"width": w, "height": h, "targetWidth": W, "targetHeight": H, "input": b64(image),
                              "torchBilinear": b64(torch_float_bilinear(image, W, H).permute(1, 2, 0).numpy())})

    letterboxes = [{"width": w, "height": h, "canvasWidth": W, "canvasHeight": H, "resized": list(letterbox_size(w, h, W, H))}
                   for w, h, W, H in [(2443, 424, 640, 640), (610, 424, 640, 640), (640, 480, 640, 640),
                                      (1920, 1080, 640, 640), (100, 50, 64, 64), (50, 99, 64, 64), (49, 30, 8, 8),
                                      (30, 49, 8, 8), (49, 2, 8, 8), (77, 5, 10, 10), (424, 2443, 640, 640),
                                      (3000, 7, 640, 640), (53, 37, 32, 24)]]

    vimage = []
    for h, w, H, W in [(37, 53, 90, 90), (150, 100, 64, 48), (64, 64, 32, 32), (48, 64, 96, 128), (101, 77, 40, 40)]:
        image = smooth(h, w)
        vimage.append({"width": w, "height": h, "targetWidth": W, "targetHeight": H, "input": b64(image),
                       "pillowLanczos": b64(pillow(image, W, H, Image.Resampling.LANCZOS))})

    clip = {"mean": [0.48145466, 0.4578275, 0.40821073], "std": [0.26862954, 0.26130258, 0.27577711]}
    blocks = [
        ("stretch-nchw-torch", (37, 53), {"resize": {"mode": "stretch", "size": [32, 24], "filter": "torchUint8Bicubic"},
                                          "scale": 1 / 255, **clip, "layout": "NCHW", "hostDType": "float32"}),
        ("letterbox-topleft-bgr", (30, 50), {"resize": {"mode": "letterbox", "size": [32, 32], "anchor": "topLeft",
                                                        "padValue": [114], "filter": "pillowBicubic"},
                                             "channelOrder": "BGR", "scale": 1, "layout": "NCHW"}),
        ("letterbox-center-fp16", (41, 23), {"resize": {"mode": "letterbox", "size": [24, 24], "padValue": [255, 128, 0],
                                                        "filter": "pillowBicubic"},
                                             "scale": 1 / 255, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
                                             "hostDType": "float16"}),
        ("centercrop-clip", (37, 53), {"resize": {"mode": "centerCrop", "size": [24, 24], "filter": "pillowBicubic"},
                                       "scale": 1 / 255, **clip}),
        ("centercrop-odd", (50, 31), {"resize": {"mode": "centerCrop", "size": [16, 20], "filter": "torchUint8Bicubic"},
                                      "scale": 1 / 255}),
        ("fitlongside-nvchw-fp16", (37, 53), {"resize": {"mode": "fitLongSide", "maxSide": 40, "filter": "pillowBicubic"},
                                              "scale": 1 / 255, "layout": "NVCHW", "views": 2, "hostDType": "float16"}),
        ("qwen-patches", (37, 45), {"resize": {"mode": "stretch", "size": [64, 32], "filter": "torchUint8Bicubic"},
                                    "scale": 1 / 255, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "layout": "patches",
                                    "patchify": {"patch": 16, "temporal": 2, "merge": 2, "rowOrder": "blockMajor",
                                                 "withinPatch": "C,T,py,px"}}),
        ("siglip-patches", (20, 28), {"resize": {"mode": "stretch", "size": [32, 24], "filter": "pillowBicubic"},
                                      "scale": 1 / 255, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "layout": "patches",
                                      "patchify": {"patch": 8, "rowOrder": "rowMajor", "withinPatch": "py,px,C"}}),
        ("letterbox-yolox-bilinear", (30, 49), {"resize": {"mode": "letterbox", "size": [8, 8], "anchor": "topLeft",
                                                           "padValue": [114], "filter": "pillowBilinear"},
                                                "channelOrder": "BGR", "scale": 1, "layout": "NCHW"}),
        ("letterbox-center-bilinear", (61, 40), {"resize": {"mode": "letterbox", "size": [24, 20], "padValue": [0, 50, 255],
                                                            "filter": "pillowBilinear"}, "scale": 1 / 255}),
        ("stretch-rfdetr-float", (45, 67), {"resize": {"mode": "stretch", "size": [32, 32], "filter": "torchBilinear"},
                                            "scale": 1 / 255, "layout": "NCHW", "hostDType": "float32"}),
        ("stretch-float-normalized-bgr", (50, 31), {"resize": {"mode": "stretch", "size": [70, 61], "filter": "torchBilinear"},
                                                    "scale": 1 / 255, **clip, "channelOrder": "BGR"}),
        ("stretch-float-nvchw-fp16", (23, 29), {"resize": {"mode": "stretch", "size": [17, 40], "filter": "torchBilinear"},
                                                "scale": 1 / 255, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
                                                "layout": "NVCHW", "views": 2, "hostDType": "float16"}),
        ("unresized-nchw", (9, 13), {"scale": 1 / 255, "mean": [0.1, 0.2, 0.3], "std": [0.9, 0.8, 0.7]}),
    ]
    pipelines = []
    for name, (h, w), block in blocks:
        image = noise(h, w)
        canvas = geometry(image, block.get("resize"))
        t = tensor(canvas, block)
        dtype = block.get("hostDType", "float32")
        float_resize = block.get("resize", {}).get("filter") == "torchBilinear"
        values = t.to(torch.float16 if dtype == "float16" else torch.float32).numpy()
        pipelines.append({"name": name, "width": w, "height": h, "input": b64(image), "preprocess": block,
                          "canvasWidth": t.shape[-1] if float_resize else canvas.shape[1],
                          "canvasHeight": t.shape[-2] if float_resize else canvas.shape[0],
                          "shape": list(t.shape), "hostDType": dtype, "expected": b64(values)})

    (out / "preprocess.json").write_text(json.dumps(
        {"generator": "Forge/vision/fixtures_preprocess.py", "pillow": PIL.__version__,
         "torch": torch.__version__, "torchThreads": torch.get_num_threads(), "resizes": resizes,
         "floatResizes": float_resizes, "letterboxGeometry": letterboxes, "vImage": vimage, "pipelines": pipelines}) + "\n")
    alpha_fixtures(out / "alpha")


if __name__ == "__main__":
    main()

"""The CLIP ViT-B/32 export graphs and the shared harness for its exporter, reference and checks.

CLIP ViT-B/32 (openai/clip-vit-base-patch32; OpenAI, MIT) is a ViT-B/32 image tower and a 12-layer causal text
transformer, each followed by a linear projection into one 512-wide joint space. The upstream PyTorch implementation is
transformers' `CLIPModel` (the version pinned by Forge's uv.lock); nothing here modifies it. The weights are the pinned
snapshot's `pytorch_model.bin` (the repo ships no safetensors), loaded with `weights_only=True` after its SHA-256 check.

The export is two independent static graphs instead of one joint function, so encoding one modality never runs (or
feeds dummy data to) the other:

* `ImageGraph`: `image` [1, 3, 224, 224] float16, RGB on [0, 1] (the host resizes the shortest side to 224 with
  bicubic, center-crops 224 x 224 and scales by 1/255). OpenAI's CLIP normalization (mean 0.48145466/0.4578275/
  0.40821073, std 0.26862954/0.26130258/0.27577711) is in the graph, in float32, as for DA3. Then upstream's
  `get_image_features` (vision transformer, CLS token, post-layernorm, `visual_projection`) and an L2 normalization in
  float32: `image_embedding` [1, 512] float32, unit length.
* `TextGraph`: `tokens` [B, 77] int32 (CLIP BPE ids: BOS 49406, the text, EOS 49407, padded to 77; any pad id). Then
  upstream's text transformer (causal attention), the hidden state at the first EOS, `text_projection` and an L2
  normalization in float32: `text_embedding` [B, 512] float32, unit length.

The text graph takes no attention mask. Upstream's mask is redundant for the pooled output: attention is causal, so the
first-EOS position (the only row pooled) attends only to positions at or before it, all of which the mask keeps.
Upstream pools at `argmax(input_ids)` (this checkpoint's config has the legacy `eos_token_id: 2`), which is the first
EOS when padding is EOS, but relies on argmax's tie-breaking; the graph instead pools with an exact one-hot of the first
EOS (`cumsum(ids == EOS) == 1 and ids == EOS`), which is the same row for every input with an EOS and does not depend
on the pad id. `selftest_clip.py` checks both graphs equal upstream bit for bit in float32.

The cosine similarity of the two unit vectors times `logit_scale.exp()` (100.0 in this checkpoint) is upstream's
`logits_per_image`; the manifest records the scale for zero-shot classification (softmax over the labels).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "source-pins.json").read_text())
REPO, REVISION = PINS["repository"], PINS["revision"]
CODE_REPO, CODE_COMMIT = PINS["upstreamCode"]["repository"], PINS["upstreamCode"]["commit"]
IMAGES_REPO, IMAGES_COMMIT = PINS["referenceImages"]["repository"], PINS["referenceImages"]["commit"]
DEFAULT_SNAPSHOT = Path.home() / f".cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/{REVISION}"
DEFAULT_UPSTREAM = Path.home() / f".cache/aether-kit/clip/upstream/clip-{CODE_COMMIT[:12]}"

SIZE = 224                     # the vision tower's input (7 x 7 patches of 32)
TEXT_LENGTH = 77               # the text tower's context
BOS, EOS = 49406, 49407
EMBED_DIM = 512
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
TOKENIZER_FILES = ("merges.txt", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json")
DTYPES = {"float32": torch.float32, "float16": torch.float16}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sources(snapshot: Path, upstream_root: Path) -> dict:
    """Checks every pinned snapshot file and the upstream unpack (LICENSE, model card, reference images); returns
    {file: sha256}."""
    verified = {}
    for relative, expected in PINS["files"].items():
        path = snapshot / relative
        if not path.is_file():
            raise SystemExit(f"snapshot lacks {relative}; run Forge/clip/fetch_upstream.py ({REPO}@{REVISION[:12]})")
        if sha256_file(path) != expected:
            raise SystemExit(f"{relative} differs from the pinned {REVISION[:12]} file")
        verified[relative] = expected
    stamp = upstream_root / "UNPACKED.json"
    if not stamp.is_file() or json.loads(stamp.read_text()).get("commit") != CODE_COMMIT:
        raise SystemExit(f"{upstream_root} is not the pinned upstream unpack; run Forge/clip/fetch_upstream.py")
    for prefix, group in (("upstream", PINS["upstreamCode"]["files"]), ("images", PINS["referenceImages"]["files"])):
        for relative, expected in group.items():
            path = upstream_root / prefix / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise SystemExit(f"{prefix}/{relative} in the upstream unpack differs from the pin; re-run "
                                 "fetch_upstream.py")
            verified[f"{prefix}:{relative}"] = expected
    return verified


def build_model(snapshot: Path, attn: str = "sdpa"):
    """The unmodified transformers `CLIPModel`, float32, eval, with the pinned weights (strict load). `attn` is
    transformers' attention implementation: `sdpa` (its default) for the reference and the export."""
    from transformers import CLIPConfig, CLIPModel

    config = CLIPConfig.from_pretrained(snapshot, local_files_only=True)
    config._attn_implementation = attn
    config.text_config._attn_implementation = attn
    config.vision_config._attn_implementation = attn
    with torch.device("meta"):
        model = CLIPModel(config)
    state = torch.load(snapshot / "pytorch_model.bin", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    # The checkpoint stores the two `position_ids` buffers (arange) that newer transformers keep non-persistent.
    unexpected = [k for k in unexpected if not k.endswith("embeddings.position_ids")]
    if missing or unexpected:
        raise SystemExit(f"checkpoint mismatch: missing {missing}, unexpected {unexpected}")
    for embeddings in (model.text_model.embeddings, model.vision_model.embeddings):
        n = embeddings.position_embedding.num_embeddings
        embeddings.register_buffer("position_ids", torch.arange(n).expand((1, -1)), persistent=False)
    if any(p.device.type == "meta" for p in list(model.parameters()) + list(model.buffers())):
        raise SystemExit("a parameter or buffer was not loaded")
    return model.float().eval()


def logit_scale(model) -> float:
    return float(model.logit_scale.detach().exp())


def normalize_pixels(image: torch.Tensor) -> torch.Tensor:
    """OpenAI CLIP normalization of a [B,3,H,W] float32 [0,1] image, as torchvision `Normalize` computes it (upstream
    CLIP's own transform)."""
    import torchvision.transforms as T
    return torch.stack([T.Normalize(mean=list(CLIP_MEAN), std=list(CLIP_STD))(x) for x in image])


def upstream_embeddings(model, image: torch.Tensor | None, tokens: torch.Tensor | None,
                        mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Upstream's own `CLIPModel.forward` (float32): unit image and text embeddings and `logits_per_image`. Either side
    may be omitted (then upstream's `get_*_features` alone, normalized as forward does)."""
    with torch.no_grad():
        if image is not None and tokens is not None:
            out = model(input_ids=tokens.long(), pixel_values=normalize_pixels(image), attention_mask=mask)
            return {"image": out.image_embeds, "text": out.text_embeds, "logitsPerImage": out.logits_per_image}
        result = {}
        if image is not None:
            result["image"] = upstream_unit(model.get_image_features(pixel_values=normalize_pixels(image)).pooler_output)
        if tokens is not None:
            result["text"] = upstream_unit(model.get_text_features(input_ids=tokens.long(),
                                                                   attention_mask=mask).pooler_output)
        return result


def upstream_unit(x: torch.Tensor) -> torch.Tensor:
    """`CLIPModel.forward`'s own normalization (`_get_vector_norm`)."""
    from transformers.models.clip.modeling_clip import _get_vector_norm
    return x / _get_vector_norm(x)


def unit(x: torch.Tensor) -> torch.Tensor:
    """The graphs' L2 normalization, in float32, written as upstream's `_get_vector_norm`."""
    x = x.to(torch.float32)
    return x / torch.pow(torch.sum(torch.pow(x, 2), dim=-1, keepdim=True), 0.5)


class ImageGraph(nn.Module):
    """image [1,3,224,224] raw RGB on [0,1] (float16) -> image_embedding [1,512] float32, unit length.

    Normalization and the final L2 normalization run in float32; the tower and its projection in `dtype`."""

    def __init__(self, model, dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self.vision_model, self.projection = model.vision_model.to(dtype), model.visual_projection.to(dtype)
        self.dtype = dtype
        self.register_buffer("mean", torch.tensor(CLIP_MEAN, dtype=torch.float32).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CLIP_STD, dtype=torch.float32).reshape(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = ((image.to(torch.float32) - self.mean) / self.std).to(self.dtype)
        pooled = self.vision_model(pixel_values=x, return_dict=True).pooler_output
        return unit(self.projection(pooled))


class TextGraph(nn.Module):
    """tokens [B,77] int32 -> text_embedding [B,512] float32, unit length. Pools the first EOS (exact one-hot, any pad
    id); every row must contain an EOS. The tower and projection run in `dtype`, the L2 normalization in float32."""

    def __init__(self, model, dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self.text_model, self.projection = model.text_model.to(dtype), model.text_projection.to(dtype)
        self.dtype = dtype

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.text_model(input_ids=tokens, return_dict=True).last_hidden_state      # [B,77,512]
        eos = (tokens == EOS).to(torch.int32)
        first = ((torch.cumsum(eos, dim=-1) == 1) & (eos == 1)).to(hidden.dtype)          # [B,77] one-hot
        pooled = torch.bmm(first[:, None, :], hidden)[:, 0]                                 # [B,512]
        return unit(self.projection(pooled))


# MARK: reference-case inputs

def fp16_representable(x: np.ndarray) -> np.ndarray:
    """Rounds float32 values through float16, so the float32 tensor the host converts for the fp16 graph input and the
    tensor upstream sees are the same numbers."""
    return x.astype(np.float16).astype(np.float32)


def resize_crop(path: Path, size: int = SIZE):
    """Upstream CLIP's image transform before normalization (openai/CLIP `clip._transform`; transformers 5's
    CLIPImageProcessor floors the crop offset instead and can be a pixel off on odd margins): Pillow RGB, the shortest side resized to `size` with Pillow BICUBIC (the long side
    `int(size * long / short)`, as torchvision's Resize computes it), then a center crop of size x size with
    torchvision's offsets (`round((side - size) / 2)`)."""
    from PIL import Image

    with Image.open(path) as im:
        rgb = im.convert("RGB")
    w, h = rgb.size
    if w <= h:
        new = (size, int(size * h / w))
    else:
        new = (int(size * w / h), size)
    rgb = rgb.resize(new, Image.Resampling.BICUBIC)
    left, top = int(round((new[0] - size) / 2.0)), int(round((new[1] - size) / 2.0))
    return rgb.crop((left, top, left + size, top + size))


def image_tensor(path: Path, size: int = SIZE) -> np.ndarray:
    """The reference cases' preprocessing: `resize_crop`, x / 255, NCHW float32, fp16-representable. (The gate feeds
    this tensor directly; the host's own resize is gated separately.)"""
    rgb = resize_crop(path, size)
    x = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1)[None] / np.float32(255)
    return fp16_representable(np.ascontiguousarray(x))


def synthetic_tensor(seed: int, size: int = SIZE) -> np.ndarray:
    """A deterministic structured image (smooth gradients, a disc and seeded uniform noise) on [0,1], as DA3's harness
    builds it: exercises the graph away from natural-image statistics."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0, 1, size, dtype=np.float32), np.linspace(0, 1, size, dtype=np.float32),
                         indexing="ij")
    disc = (((xx - 0.6) ** 2 + (yy - 0.4) ** 2) < 0.04).astype(np.float32)
    channels = [0.6 * yy + 0.3 * disc, 0.5 * xx + 0.4 * (1 - disc) * yy, 0.4 * (1 - yy) + 0.5 * disc]
    x = np.stack(channels)[None] + 0.1 * rng.random((1, 3, size, size), dtype=np.float32)
    return fp16_representable(np.clip(x, 0, 1).astype(np.float32))


def tokenizer(snapshot: Path):
    """transformers' CLIPTokenizer from the pinned snapshot files (the reference tokenization)."""
    from transformers import CLIPTokenizer
    return CLIPTokenizer.from_pretrained(snapshot, local_files_only=True)


def tokenize(tok, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Upstream tokenization: BOS + at most 75 pieces + EOS, right-padded with the pad token (EOS, 49407) to 77.
    Returns ids and upstream's attention mask, int32 [N,77] each."""
    enc = tok(texts, padding="max_length", max_length=TEXT_LENGTH, truncation=True)
    return np.asarray(enc.input_ids, dtype=np.int32), np.asarray(enc.attention_mask, dtype=np.int32)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity, float64."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30)


def write_f32(path: Path, x: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(x, dtype="<f4").tobytes())


def read_f32(path: Path, shape) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype="<f4").reshape(shape).copy()


def write_i32(path: Path, x: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(x, dtype="<i4").tobytes())


def read_i32(path: Path, shape) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype="<i4").reshape(shape).copy()

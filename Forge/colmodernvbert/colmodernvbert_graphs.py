"""The ColModernVBERT export graphs and the shared harness for its exporter, reference and checks.

ColModernVBERT (ModernVBERT/colmodernvbert-merged; Illuin Technology and the ModernVBERT authors, MIT) is a late-
interaction (ColBERT-style) visual document retriever: a SigLIP 2 B/16 vision tower at 512 x 512, a 4x pixel shuffle and
a linear connector into ModernBERT-150M (bidirectional, 22 layers, alternating global and sliding-window attention),
then a linear projection to 128 and a per-token L2 normalization. A query and a page each become a set of unit
vectors; a page's score for a query is MaxSim, the sum over query vectors of the best dot product with any page vector.
The upstream PyTorch implementation is transformers' `ModernVBertModel` and `ColModernVBertForRetrieval` (the version
pinned by Forge's uv.lock), loaded as sentence-transformers loads this checkpoint (`key_mapping {"^model\\.": ""}`,
the projection from `1_Dense`); nothing here modifies it.

The export is two independent static graphs, as the zoo bundle ships them:

* `QueryGraph`: `tokens` [1, 64] int32 and `attention_mask` [1, 64] int32 -> `token_embeddings` [1, 64, 128]
  float32. The query is upstream's `process_queries` text (the query, then ten `<end_of_utterance>` augmentation
  tokens), encoded with `[CLS]` 50281 ... `[SEP]` 50282 and right-padded with `[PAD]` 50283; the mask is 1 for every
  real id. Upstream's own forward: ModernBERT over the tokens with the padding mask, the projection, the L2
  normalization, and the rows multiplied by the mask (padding rows are exactly zero).
* `DocGraph`: `image` [1, 1, 3, 512, 512] float32, upstream's `pixel_values` for one page as a single global image
  (RGB, x/255 normalized with mean 0.5 and std 0.5, so on [-1, 1]) -> `token_embeddings` [1, 89, 128] float32. The
  document prompt is fixed, so its 89 ids are graph constants: `[CLS]<|begin_of_text|>User:<fake_token_around_image>
  <global-img>`, 64 `<image>` slots (rows 13..76), `<fake_token_around_image>Describe the image.<end_of_utterance>\n
  Assistant:[SEP]`. The vision tower's 32 x 32 patches are pixel-shuffled into the 64 slots row-major, so slot
  13 + 8 r + c covers the page's 64 x 64-pixel cell (r, c) of the 512 x 512 input.

Upstream splits a page larger than 512 into up to 4 x 4 tiles of 512 plus the global image (`do_image_splitting`, its
default: 1149 doc tokens for a portrait page); this export is the single-tile path (`do_image_splitting=False`), as the
zoo's. Upstream's `pixel_attention_mask` has no effect in this transformers version (`SiglipVisionModel.forward`
ignores `patch_attention_mask`), so the doc graph takes no mask; `selftest_colmodernvbert.py` checks the mask fact.

Precision (`PRECISIONS`): the shipped `mixed` export runs the vision tower in float16 and the connector and ModernBERT
in float32, the projection and normalization always in float32. ModernBERT builds residual outliers near 2.3e4 and
some query rows (commas between clauses) are sensitive enough that rounding ModernBERT's weights to float16, even
with float32 arithmetic, drops them to cosine 0.992-0.998 against upstream, and float16 arithmetic to 0.95 (measured,
CPU and GPU); the vision tower in float16 keeps every doc row above 0.9997.

`selftest_colmodernvbert.py` checks the float32 graphs against upstream's `ColModernVBertForRetrieval` output.
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
ZOO_REPO, ZOO_REVISION = PINS["zooCrossCheck"]["repository"], PINS["zooCrossCheck"]["revision"]
DEFAULT_SNAPSHOT = (Path.home() /
                    f".cache/huggingface/hub/models--ModernVBERT--colmodernvbert-merged/snapshots/{REVISION}")
DEFAULT_UPSTREAM = Path.home() / ".cache/aether-kit/colmodernvbert"

SIZE = 512                      # the vision tower's input (32 x 32 patches of 16)
PATCH = 16
SHUFFLE = 4                     # pixel shuffle: 32 x 32 patches -> 8 x 8 tokens
PATCH_GRID = (SIZE // PATCH // SHUFFLE, SIZE // PATCH // SHUFFLE)
QUERY_LENGTH = 64               # the query graph's static sequence
DOC_LENGTH = 89                 # the document prompt with one global image
IMAGE_SPAN = (13, 77)           # the 64 `<image>` rows of the doc output, [start, end)
EMBED_DIM = 128
CLS, SEP, PAD = 50281, 50282, 50283
END_OF_UTTERANCE = 50405
IMAGE_TOKEN = 50407
AUGMENTATION = "<end_of_utterance>" * 10      # upstream process_queries' default suffix
NORM_MEAN, NORM_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
TOKENIZER_FILES = ("special_tokens_map.json", "tokenizer.json", "tokenizer_config.json")
DTYPES = {"float32": torch.float32, "float16": torch.float16}
# Export precisions: (vision tower, text side) dtypes. The projection and normalization are always float32. `mixed`
# (the default) keeps ModernBERT and the connector in float32: some query rows (commas between clauses, measured) are so
# sensitive that rounding ModernBERT's weights to float16 alone, in float32 arithmetic, moves them below cosine 0.999.
PRECISIONS = {"mixed": (torch.float16, torch.float32), "float16": (torch.float16, torch.float16),
              "float32": (torch.float32, torch.float32)}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sources(snapshot: Path, upstream_root: Path) -> dict:
    """Checks every pinned snapshot file and fetch_upstream.py's output (license, README, pages, zoo references);
    returns {file: sha256}."""
    verified = {}
    for relative, expected in PINS["files"].items():
        path = snapshot / relative
        if not path.is_file():
            raise SystemExit(f"snapshot lacks {relative}; run Forge/colmodernvbert/fetch_upstream.py "
                             f"({REPO}@{REVISION[:12]})")
        if sha256_file(path) != expected:
            raise SystemExit(f"{relative} differs from the pinned {REVISION[:12]} file")
        verified[relative] = expected
    groups = [("upstream", PINS["upstreamCode"]["files"]),
              ("images", {n: m["sha256"] for n, m in PINS["referenceImages"]["files"].items()}),
              ("zoo", PINS["zooCrossCheck"]["files"])]
    for prefix, group in groups:
        for relative, expected in group.items():
            path = upstream_root / prefix / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise SystemExit(f"{prefix}/{relative} differs from the pin; run "
                                 "Forge/colmodernvbert/fetch_upstream.py")
            verified[f"{prefix}:{relative}"] = expected
    return verified


def quiet_transformers() -> None:
    from transformers.utils import logging
    logging.set_verbosity_error()
    logging.disable_progress_bar()


def build_model(snapshot: Path, attn: str = "sdpa"):
    """Upstream's retriever, float32, eval: transformers' `ColModernVBertForRetrieval` whose `vlm` is the checkpoint's
    `ModernVBertModel` (loaded as sentence-transformers loads it, every key used) and whose `embedding_proj_layer` is
    `1_Dense` (equal to the checkpoint's `custom_text_proj`, checked). `attn`: transformers' attention implementation,
    `sdpa` (its default) for the reference and the export."""
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import ColModernVBertConfig, ColModernVBertForRetrieval, ModernVBertModel

    quiet_transformers()
    vlm, info = ModernVBertModel.from_pretrained(snapshot, key_mapping={r"^model\.": ""}, attn_implementation=attn,
                                                 dtype=torch.float32, local_files_only=True, output_loading_info=True)
    projection = {"custom_text_proj.weight", "custom_text_proj.bias"}
    if info["missing_keys"] or info["mismatched_keys"] or set(info["unexpected_keys"]) != projection:
        raise SystemExit(f"checkpoint mismatch: {info}")
    dense = load_file(snapshot / "1_Dense/model.safetensors")
    with safe_open(snapshot / "model.safetensors", "pt") as f:
        for key in ("weight", "bias"):
            if not torch.equal(f.get_tensor(f"custom_text_proj.{key}"), dense[f"linear.{key}"]):
                raise SystemExit(f"1_Dense linear.{key} differs from the checkpoint's custom_text_proj.{key}")
    config = ColModernVBertConfig(vlm_config=vlm.config.to_dict(), embedding_dim=EMBED_DIM)
    with torch.device("meta"):
        model = ColModernVBertForRetrieval(config)
    model.vlm = vlm
    model.embedding_proj_layer = nn.Linear(vlm.config.text_config.hidden_size, EMBED_DIM)
    model.embedding_proj_layer.load_state_dict({"weight": dense["linear.weight"], "bias": dense["linear.bias"]})
    if any(p.device.type == "meta" for p in list(model.parameters()) + list(model.buffers())):
        raise SystemExit("a parameter or buffer was not loaded")
    return model.float().eval()


def processor(snapshot: Path):
    """transformers' `ColModernVBertProcessor` from the pinned snapshot (the reference preprocessing and tokenizer)."""
    from transformers import ColModernVBertProcessor

    quiet_transformers()
    return ColModernVBertProcessor.from_pretrained(snapshot, local_files_only=True)


def upstream_query(model, proc, text: str) -> np.ndarray:
    """Upstream on one query, unpadded (`process_queries`, then `ColModernVBertForRetrieval.forward`): [n, 128]."""
    batch = proc.process_queries([text])
    with torch.no_grad():
        return model(**batch).embeddings[0].numpy()


def upstream_doc(model, pixel_values: np.ndarray, pixel_mask: np.ndarray | None = None) -> np.ndarray:
    """Upstream on one page (`pixel_values` [1,1,3,512,512]) with the fixed document prompt: [89, 128]."""
    ids = torch.tensor([DOC_IDS], dtype=torch.long)
    feeds = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "pixel_values": torch.from_numpy(pixel_values)}
    if pixel_mask is not None:
        feeds["pixel_attention_mask"] = torch.from_numpy(pixel_mask)
    with torch.no_grad():
        return model(**feeds).embeddings[0].numpy()


def unit(x: torch.Tensor) -> torch.Tensor:
    """`ColModernVBertForRetrieval`'s normalization, in float32."""
    x = x.to(torch.float32)
    return x / x.norm(dim=-1, keepdim=True)


class QueryGraph(nn.Module):
    """tokens, attention_mask [1,64] int32 -> token_embeddings [1,64,128] float32: unit rows, padding rows zero.

    ModernBERT runs in `dtype`; the projection and the normalization in float32."""

    def __init__(self, model, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.vlm = model.vlm.to(dtype)
        self.projection = model.embedding_proj_layer.float()
        self.dtype = dtype

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.vlm(input_ids=tokens, attention_mask=attention_mask, return_dict=True).last_hidden_state
        embeddings = unit(self.projection(hidden.to(torch.float32)))
        return embeddings * attention_mask.to(torch.float32).unsqueeze(-1)


class DocGraph(nn.Module):
    """image [1,1,3,512,512] float32 (upstream pixel_values, on [-1,1]) -> token_embeddings [1,89,128] float32, unit
    rows. The vision tower runs in `vision_dtype`; the connector and ModernBERT in `text_dtype`; the projection and the
    normalization in float32.

    The prompt's 25 text ids are constants; the 64 image features replace the `<image>` rows, as upstream's
    `masked_scatter` does."""

    def __init__(self, model, vision_dtype: torch.dtype = torch.float16,
                 text_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        vlm = model.vlm
        self.vision_model = vlm.vision_model.to(vision_dtype)
        self.connector, self.text_model = vlm.connector.to(text_dtype), vlm.text_model.to(text_dtype)
        self.projection = model.embedding_proj_layer.float()
        self.vision_dtype, self.dtype = vision_dtype, text_dtype
        start, end = IMAGE_SPAN
        self.register_buffer("prefix", torch.tensor([DOC_IDS[:start]], dtype=torch.long), persistent=False)
        self.register_buffer("suffix", torch.tensor([DOC_IDS[end:]], dtype=torch.long), persistent=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pixels = image[:, 0].to(self.vision_dtype)                                          # [1,3,512,512]
        vision = self.vision_model(pixel_values=pixels, return_dict=True).last_hidden_state  # [1,1024,768]
        features = self.connector(vision.to(self.dtype))                                    # [1,64,768]
        embed = self.text_model.get_input_embeddings()
        inputs = torch.cat([embed(self.prefix), features.to(self.dtype), embed(self.suffix)], dim=1)
        hidden = self.text_model(inputs_embeds=inputs, return_dict=True).last_hidden_state   # [1,89,768]
        return unit(self.projection(hidden.to(torch.float32)))


# The document prompt (`process_images` with do_image_splitting=False): checked against the processor in the selftest
# and in every reference case.
DOC_IDS = [CLS, 29, 93, 2043, 64, 1171, 64, 1156, 49651, 6989, 27, 50406, 50368] + [IMAGE_TOKEN] * 64 + [
    50406, 4476, 19268, 253, 2460, 15, END_OF_UTTERANCE, 187, 6717, 5567, 27, SEP]
assert len(DOC_IDS) == DOC_LENGTH and DOC_IDS[IMAGE_SPAN[0]:IMAGE_SPAN[1]] == [IMAGE_TOKEN] * 64


# MARK: tokenization (the reference and the bundle's tokenizer.text contract)

def tokenize_query(tok, text: str, length: int = QUERY_LENGTH) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """The query graph's row: [CLS] + the query's ids + ten <end_of_utterance> + [SEP] (upstream `process_queries`),
    the query's own ids cut so the row fits `length` (the suffix and both specials kept), right-padded with [PAD].
    Returns ids and mask [1, length] int32, the real count and whether the query was cut."""
    body = tok(text, add_special_tokens=False).input_ids
    suffix = tok(AUGMENTATION, add_special_tokens=False).input_ids
    room = length - 2 - len(suffix)
    cut = len(body) > room
    ids = [CLS] + body[:room] + suffix + [SEP]
    row = np.full((1, length), PAD, dtype=np.int32)
    row[0, :len(ids)] = ids
    mask = np.zeros((1, length), dtype=np.int32)
    mask[0, :len(ids)] = 1
    return row, mask, len(ids), cut


# MARK: reference-case inputs

def page_pixels(path: Path, proc) -> np.ndarray:
    """Upstream's single-tile document preprocessing, exactly: `process_images(do_image_splitting=False)`'s
    pixel_values [1,1,3,512,512] float32 (Pillow convert('RGB'); LANCZOS to longest edge 2048, aspect kept, the short
    side even; LANCZOS stretch to 512 x 512; x/255; (x - 0.5) / 0.5). Also checks the prompt ids."""
    from PIL import Image

    with Image.open(path) as im:
        rgb = im.convert("RGB")
    batch = proc.process_images([rgb], do_image_splitting=False)
    if batch["input_ids"][0].tolist() != DOC_IDS:
        raise SystemExit(f"{path.name}: the processor's document prompt differs from DOC_IDS")
    if not bool((batch["pixel_attention_mask"] == 1).all()):
        raise SystemExit(f"{path.name}: a single tile should have an all-ones pixel mask")
    return np.ascontiguousarray(batch["pixel_values"].numpy().astype(np.float32))


def stretch_pixels(path: Path, resample) -> np.ndarray:
    """The page through one stretch to 512 x 512 with a Pillow filter, then the same normalization in float32
    ((x * (1/255) - 0.5) / 0.5, as Aether-Kit's ImagePreprocessor computes it): what a single-stage host resize
    gives."""
    from PIL import Image

    with Image.open(path) as im:
        rgb = im.convert("RGB").resize((SIZE, SIZE), resample)
    x = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1)[None, None]
    x = (x * np.float32(1 / 255) - np.float32(0.5)) / np.float32(0.5)
    return np.ascontiguousarray(x.astype(np.float32))


def synthetic_pixels(seed: int) -> np.ndarray:
    """A deterministic structured page on [-1, 1] (white background, dark text-like bars, a coloured block, seeded
    noise): exercises the doc graph away from the reference pages."""
    rng = np.random.default_rng(seed)
    x = np.ones((3, SIZE, SIZE), dtype=np.float32)
    for row in range(40, 470, 24):
        width = int(rng.integers(200, 440))
        x[:, row:row + 10, 36:36 + width] = 0.15
    x[0, 300:420, 300:470], x[1, 300:420, 300:470], x[2, 300:420, 300:470] = 0.9, 0.4, 0.2
    x = np.clip(x + 0.05 * rng.standard_normal(x.shape).astype(np.float32), 0, 1)
    return np.ascontiguousarray(((x - 0.5) / 0.5)[None, None].astype(np.float32))


# MARK: scoring and files

def maxsim(query: np.ndarray, doc: np.ndarray) -> float:
    """ColBERT MaxSim (upstream `score_retrieval`): the sum over query rows of the best dot product with a doc row, in
    float64. Zero (padding) query rows add nothing."""
    return float((query.astype(np.float64) @ doc.astype(np.float64).T).max(axis=1).sum())


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

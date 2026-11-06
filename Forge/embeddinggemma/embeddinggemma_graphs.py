"""The EmbeddingGemma 300M export graph and the shared harness for its exporter, reference and checks.

EmbeddingGemma (google/embeddinggemma-300m, Gemma Terms of Use, Google DeepMind) is a sentence-transformers model:
a bidirectional Gemma 3 text encoder (transformers `Gemma3TextModel`, 24 layers, hidden 768), then mean pooling over
the attention mask, Dense 768 -> 3072, Dense 3072 -> 768 (no bias, identity activation) and L2 normalization
(`modules.json`). The encoder runs on transformers' own Gemma 3 code at the version Forge/uv.lock pins; the four
sentence-transformers modules are re-expressed here from their pinned configs (they are a few lines each, and
sentence-transformers is not in the Forge environment).

The exported graph (`EmbeddingGraph`) is the whole pipeline at a fixed sequence of 256:

* inputs `input_ids` int32 [1, 256] (the tokenizer's ids with `<bos>` ... `<eos>`, right-padded with `<pad>` = 0) and
  `attention_mask` int32 [1, 256] (1 for real tokens, 0 for padding);
* the encoder, with the key-padding mask built in the graph (see below);
* mean pooling over the mask (prompt tokens included, `include_prompt: true`), the two dense layers and the L2
  normalization; output `embedding` float32 [1, 768], unit length.

One export-only substitution: transformers' `create_causal_mask` / `create_sliding_window_causal_mask` build the
bidirectional masks through vmapped mask functions that `torch.export` does not trace to a static graph. The graph
passes the per-layer-type masks precomputed instead (transformers accepts a `{layer_type: mask}` dict). For a
sequence of 256 they are the same boolean key-padding mask for both layer types: the sliding window is 512, so the
bidirectional window overlay `|q - k| < 512` is always true, and full attention is bidirectional. `selftest_*.py` checks
that the graph equals upstream's own forward (2-D mask) in float32.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "source-pins.json").read_text())
REPO, REVISION = PINS["repository"], PINS["revision"]
DEFAULT_SNAPSHOT = Path.home() / f".cache/huggingface/hub/models--google--embeddinggemma-300m/snapshots/{REVISION}"
TERMS_DIR = HERE / "terms"

SEQUENCE = 256                 # the fixed graph sequence (the zoo export's and coreai-kit's choice)
DIMENSION = 768
PAD_ID = 0
DTYPES = {"float32": torch.float32, "float16": torch.float16}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sources(snapshot: Path) -> dict:
    """Checks every pinned snapshot file and the checked-in Gemma terms; returns {file: sha256}."""
    verified = {}
    for relative, expected in PINS["files"].items():
        path = snapshot / relative
        if not path.is_file():
            raise SystemExit(f"snapshot lacks {relative}; run Forge/embeddinggemma/fetch_upstream.py "
                             f"({REPO}@{REVISION[:12]})")
        if sha256_file(path) != expected:
            raise SystemExit(f"{relative} differs from the pinned {REVISION[:12]} file")
        verified[relative] = expected
    for name, meta in PINS["terms"]["files"].items():
        if sha256_file(TERMS_DIR / name) != meta["sha256"]:
            raise SystemExit(f"terms/{name} differs from its pin in source-pins.json")
        verified[f"terms:{name}"] = meta["sha256"]
    check_st_config(snapshot)
    return verified


def check_st_config(snapshot: Path) -> None:
    """The sentence-transformers pipeline this harness re-expresses: refuse anything else."""
    modules = [m["type"] for m in json.loads((snapshot / "modules.json").read_text())]
    if modules != ["sentence_transformers.models.Transformer", "sentence_transformers.models.Pooling",
                   "sentence_transformers.models.Dense", "sentence_transformers.models.Dense",
                   "sentence_transformers.models.Normalize"]:
        raise SystemExit(f"unexpected modules.json pipeline {modules}")
    pooling = json.loads((snapshot / "1_Pooling/config.json").read_text())
    modes = {k for k, v in pooling.items() if k.startswith("pooling_mode_") and v}
    if modes != {"pooling_mode_mean_tokens"} or pooling.get("include_prompt") is not True:
        raise SystemExit(f"unexpected pooling config {pooling}")
    for name, shape in (("2_Dense", (768, 3072)), ("3_Dense", (3072, 768))):
        dense = json.loads((snapshot / name / "config.json").read_text())
        if ((dense["in_features"], dense["out_features"]) != shape or dense["bias"]
                or dense["activation_function"] != "torch.nn.modules.linear.Identity"):
            raise SystemExit(f"unexpected {name} config {dense}")


def prompts(snapshot: Path) -> dict:
    """The query and document prompts from upstream's config_sentence_transformers.json."""
    config = json.loads((snapshot / "config_sentence_transformers.json").read_text())
    return {"query": config["prompts"]["query"], "document": config["prompts"]["document"]}


class Upstream(nn.Module):
    """The unmodified transformers Gemma3TextModel (float32, eval, SDPA attention, strict load of the pinned weights)
    and the two pinned Dense layers."""

    def __init__(self, snapshot: Path) -> None:
        super().__init__()
        from safetensors.torch import load_file
        from transformers import Gemma3TextConfig, Gemma3TextModel

        config = Gemma3TextConfig.from_pretrained(str(snapshot))
        if not config.use_bidirectional_attention or config.sliding_window < SEQUENCE:
            raise SystemExit("the static mask assumes bidirectional attention and sliding_window >= 256")
        config._attn_implementation = "sdpa"
        self.encoder = Gemma3TextModel(config)
        self.encoder.load_state_dict(load_file(str(snapshot / "model.safetensors")), strict=True)
        self.dense1 = nn.Linear(768, 3072, bias=False)
        self.dense2 = nn.Linear(3072, 768, bias=False)
        for layer, name in ((self.dense1, "2_Dense"), (self.dense2, "3_Dense")):
            state = load_file(str(snapshot / name / "model.safetensors"))    # sentence-transformers' `linear.` prefix
            layer.load_state_dict({k.removeprefix("linear."): v for k, v in state.items()}, strict=True)
        self.eval()

    def head(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """sentence-transformers Pooling (mean, clamped count), Dense, Dense, Normalize, as upstream writes them."""
        expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
        pooled = torch.sum(token_embeddings * expanded, 1) / torch.clamp(expanded.sum(1), min=1e-9)
        return F.normalize(self.dense2(self.dense1(pooled)), p=2, dim=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Upstream's own path: int64 ids and the 2-D mask straight into the encoder (as sentence-transformers feeds
        it), any length. Returns [B, 768] float32."""
        with torch.no_grad():
            hidden = self.encoder(input_ids=input_ids.long(), attention_mask=attention_mask.long()).last_hidden_state
            return self.head(hidden, attention_mask)


class WideLinear(nn.Module):
    """A bias-free linear layer whose weight is stored in `storage` (float16) and widened to float32 for compute."""

    def __init__(self, linear: nn.Linear, storage: torch.dtype) -> None:
        super().__init__()
        if linear.bias is not None:
            raise SystemExit("WideLinear expects bias-free layers (Gemma 3 and the dense head have none)")
        self.register_buffer("weight", fp16_checked(linear.weight.detach(), storage))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype))


class WideEmbedding(nn.Module):
    """Gemma3TextScaledWordEmbedding with the table stored in `storage`: gather, widen to float32, scale by
    sqrt(hidden) in float32 (upstream multiplies by the scale cast to the table's dtype, float32 in this checkpoint)."""

    def __init__(self, embedding: nn.Module, storage: torch.dtype) -> None:
        super().__init__()
        self.register_buffer("weight", fp16_checked(embedding.weight.detach(), storage))
        self.register_buffer("scale", embedding.embed_scale.detach().to(torch.float32))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.weight).to(torch.float32) * self.scale


def fp16_checked(w: torch.Tensor, storage: torch.dtype) -> torch.Tensor:
    narrow = w.to(storage).contiguous()
    if not torch.isfinite(narrow).all():
        raise SystemExit("a weight overflows float16 storage")
    return narrow


def widen_weights(module: nn.Module, storage: torch.dtype) -> None:
    """Replaces every bias-free nn.Linear (and the scaled embedding) under `module` by its `storage`-stored widened
    twin. RMSNorm gains (768 values each) stay float32."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, WideLinear(child, storage))
        elif type(child).__name__ == "Gemma3TextScaledWordEmbedding":
            setattr(module, name, WideEmbedding(child, storage))
        else:
            widen_weights(child, storage)


class EmbeddingGraph(nn.Module):
    """input_ids, attention_mask int32 [1,256] -> embedding [1,768] float32 (unit length).

    `dtype` is the encoder's activation dtype; `storage` the dtype every linear and embedding weight is stored in.
    The shipped recipe is float32 compute with float16 storage (widened in the graph): upstream's activations do not
    fit float16 (the residual stream reaches ~1.3e5 on the reference cases), while rounding the weights to float16 moves
    the embeddings by ~3e-7 in cosine (selftest_embeddinggemma.py measures both). Pooling, the dense layers and the
    normalization always compute in float32."""

    def __init__(self, upstream: Upstream, dtype: torch.dtype = torch.float32,
                 storage: torch.dtype | None = None) -> None:
        super().__init__()
        self.encoder = upstream.encoder.to(dtype)
        self.dense1, self.dense2 = upstream.dense1.to(torch.float32), upstream.dense2.to(torch.float32)
        if storage is not None and storage != dtype:
            if dtype != torch.float32:
                raise SystemExit("narrow weight storage is only supported with float32 compute")
            widen_weights(self.encoder, storage)
            self.dense1, self.dense2 = WideLinear(self.dense1, storage), WideLinear(self.dense2, storage)
        self.dtype = dtype

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        keep = attention_mask.to(torch.bool)
        keys = keep[:, None, None, :].expand(keep.shape[0], 1, keep.shape[1], keep.shape[1])
        masks = {"full_attention": keys, "sliding_attention": keys}
        hidden = self.encoder(input_ids=input_ids.to(torch.int64), attention_mask=masks).last_hidden_state
        hidden = hidden.to(torch.float32)
        weights = attention_mask.to(torch.float32).unsqueeze(-1)
        pooled = (hidden * weights).sum(1) / torch.clamp(weights.sum(1), min=1e-9)
        return F.normalize(self.dense2(self.dense1(pooled)), p=2, dim=1)


# MARK: tokenization (the reference cases; the host's own tokenizer is gated separately)

def tokenizer(snapshot: Path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(snapshot))


def encode(tok, text: str, sequence: int = SEQUENCE, pad: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """sentence-transformers' tokenization of one already-prompted text at max_seq_length `sequence`: strip, then the
    tokenizer with its special tokens (`<bos>` ... `<eos>`) and right truncation, which keeps both special tokens.
    With `pad`, right-padded with id 0 to exactly `sequence` (the graph input). int32 [1, n] each."""
    batch = tok([text.strip()], truncation="longest_first", max_length=sequence,
                padding="max_length" if pad else False, return_tensors="np")
    ids, mask = batch["input_ids"].astype(np.int32), batch["attention_mask"].astype(np.int32)
    if pad and (ids.shape != (1, sequence) or (ids[mask == 0] != PAD_ID).any()):
        raise SystemExit("tokenizer padding is not right-padding with id 0 to the fixed sequence")
    return ids, mask


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def write_f32(path: Path, x: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(x, dtype="<f4").tobytes())


def read_f32(path: Path, shape) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype="<f4").reshape(shape).copy()


def write_i32(path: Path, x: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(x, dtype="<i4").tobytes())


def read_i32(path: Path, shape) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype="<i4").reshape(shape).copy()

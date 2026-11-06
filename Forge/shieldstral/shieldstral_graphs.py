"""The Shieldstral 1.0 3B export graph and the shared harness for its exporter, reference and checks.

Shieldstral 1.0 3B (mistralai/Shieldstral-1.0-3B, Apache-2.0, Mistral AI) is a policy-adaptive safety classifier built
on Ministral-3-3B-Base-2512: a `Mistral3ForConditionalGeneration` checkpoint (Pixtral vision tower, patch-merge
projector, Ministral 3 text decoder: 26 layers, hidden 3072, GQA 32/8, YaRN rope, vocabulary 131072, tied embeddings).
It answers one yes/no question per forward: the chat is a fixed system prompt plus a user message
`<Instruct>: ...\n\n<Query>: ...\n\n<Document>: ...`, and the score is upstream's own (the model card's helper, which
"mirrors the reference evaluation code"):

    logits = the next-token logits at the last prompt position (after `[/INST]`)
    z_yes  = max log-softmax over the top-20 tokens whose decoded text, stripped and lower-cased, is one of
             yes, yes., "yes", 'yes'; z_no likewise for no (absent: -1e9)
    score  = exp(z_yes) / (exp(z_yes) + exp(z_no))        flagged = score > threshold (0.5)

The exported graph (`ClassifierGraph`) is text only (the vision tower is not exported) and has that whole tail in it:

* inputs `input_ids` int32 [1, S] (the flattened template's ids, right-padded with `<pad>` = 11) and
  `attention_mask` int32 [1, S] (1 for real tokens, 0 for padding), S = 512 (`SEQUENCE`);
* the Ministral 3 decoder (transformers' own `Ministral3Model` code at the version Forge/uv.lock pins, weights loaded
  strictly from the pinned checkpoint's `language_model.model.*`), with the causal + key-padding mask built in the
  graph as a [1,1,S,S] boolean (transformers returns a prepared 4-D mask as is; its vmapped mask builders do not trace
  to a static graph);
* the hidden state of the last real token (position `sum(attention_mask) - 1`), dotted with the tied-embedding rows
  of every yes and no variant (`YES_IDS`, `NO_IDS`, enumerated from the vocabulary with upstream's rule), the maximum
  per class, and a softmax over [z_no, z_yes] (log-softmax over the vocabulary shifts both by the same constant, so
  raw logits give the same ratio); output `probs` float32 [1, 2] = [P(no), P(yes)], index 1 = P(violation).

The one place the graph and upstream's helper can differ is the helper's top-20 cut: a variant outside the top 20 is
ignored there (and a class with no variant there gets -1e9). The graph takes the maximum over every variant. The
reference records both and the gap (zero on every reference case, where the canonical `yes`/`no` tokens dominate).
The tokenization contract (the flattened chat template, `add_special_tokens=False`, body-only truncation, pad 11 on
the right) is in `tokenize()`; reference_shieldstral.py checks it against `apply_chat_template` on every case.
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
DEFAULT_SNAPSHOT = Path.home() / f".cache/huggingface/hub/models--mistralai--Shieldstral-1.0-3B/snapshots/{REVISION}"
TERMS_DIR = HERE / "terms"

SEQUENCE = 512                 # the macOS grid (the zoo's macOS choice; ~450 tokens left for the body)
HIDDEN = 3072
PAD_ID = 11                    # <pad>; text_config.pad_token_id and generation_config.pad_token_id
BOS_ID = 1
DTYPES = {"float32": torch.float32, "float16": torch.float16}

# The flattened chat template. upstream chat_template.jinja renders [system, user] as
#   bos_token + '[SYSTEM_PROMPT]' + system + '[/SYSTEM_PROMPT]' + '[INST]' + user + '[/INST]'
# and the model card fixes the system prompt and the user message's three fields.
SYSTEM_PROMPT = ("Judge whether the Document meets the requirements based on the Query and the Instruction provided. "
                 'Note that the answer can only be "yes" or "no".')
PREFIX = "<s>[SYSTEM_PROMPT]" + SYSTEM_PROMPT + "[/SYSTEM_PROMPT][INST]"
SUFFIX = "[/INST]"
BODY = "<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {document}"

# Upstream's label rule (the model card's helper): a token is a yes (no) if its decoded text, stripped and
# lower-cased, is one of these.
YES_FORMS = ("yes", "yes.", '"yes"', "'yes'")
NO_FORMS = ("no", "no.", '"no"', "'no'")
POSITIVE_INDEX = 1             # probs = [P(no), P(yes)]
THRESHOLD = 0.5                # flagged = P(yes) > 0.5 (strict, as the helper; the card's benchmarks use 0.5)
# The variant ids in this vocabulary: label_ids() over all 131072 ids (selftest_shieldstral.py re-derives them).
# yes, ĠYes, Ġyes, Yes, YES, ĠYES and Ġno, no, ĠNo, No, ĠNO, NO; the multi-character forms (yes., "yes", ...) are
# not single tokens here. The zoo export and coreai-kit read only 13059 and 2649.
YES_IDS = (13059, 13830, 14842, 16860, 51935, 57445)
NO_IDS = (1836, 2649, 3501, 4753, 14089, 16071)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sources(snapshot: Path, weights: bool = True) -> dict:
    """Checks every pinned snapshot file (all but the 7.7 GB weights when `weights` is false) and the checked-in
    Apache-2.0 text; returns {file: sha256}."""
    verified = {}
    for relative, expected in PINS["files"].items():
        path = snapshot / relative
        if not path.is_file():
            raise SystemExit(f"snapshot lacks {relative}; run Forge/shieldstral/fetch_upstream.py "
                             f"({REPO}@{REVISION[:12]})")
        if relative == "model.safetensors" and not weights:
            continue
        if sha256_file(path) != expected:
            raise SystemExit(f"{relative} differs from the pinned {REVISION[:12]} file")
        verified[relative] = expected
    for name, meta in PINS["terms"]["files"].items():
        if sha256_file(TERMS_DIR / name) != meta["sha256"]:
            raise SystemExit(f"terms/{name} differs from its pin in source-pins.json")
        verified[f"terms:{name}"] = meta["sha256"]
    check_config(snapshot)
    return verified


def check_config(snapshot: Path) -> None:
    """The architecture this harness assumes: refuse anything else."""
    config = json.loads((snapshot / "config.json").read_text())
    text = config["text_config"]
    expected = {"model_type": "ministral3", "hidden_size": HIDDEN, "num_hidden_layers": 26, "vocab_size": 131072,
                "pad_token_id": PAD_ID, "bos_token_id": BOS_ID, "sliding_window": None, "tie_word_embeddings": True}
    got = {k: text.get(k) for k in expected}
    if config["model_type"] != "mistral3" or got != expected:
        raise SystemExit(f"unexpected text_config {got}")
    if text["rope_parameters"]["original_max_position_embeddings"] < SEQUENCE:
        raise SystemExit("the llama-4 attention scale is 1 only below original_max_position_embeddings")


# MARK: tokenization

def tokenizer(snapshot: Path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(snapshot))


def encode_text(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def label_ids(tok) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Every vocabulary id that upstream's rule reads as yes or as no (decoded alone, stripped, lower-cased)."""
    yes, no = [], []
    for i in range(len(tok)):
        text = tok.decode([i]).strip().lower()
        if text in YES_FORMS:
            yes.append(i)
        elif text in NO_FORMS:
            no.append(i)
    return tuple(yes), tuple(no)


def body(instruction: str, query: str, document: str) -> str:
    return BODY.format(instruction=instruction, query=query, document=document)


def tokenize(tok, instruction: str, query: str, document: str, sequence: int = SEQUENCE,
             pad: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """The host contract: PREFIX, BODY and SUFFIX encoded separately with add_special_tokens=False (`<s>` is template
    text and maps to id 1; [SYSTEM_PROMPT], [/SYSTEM_PROMPT], [INST], [/INST] are control tokens, so the three pieces
    tokenize exactly as the whole string does), the body cut to `sequence - len(prefix) - len(suffix)` ids from its
    end (the document's tail goes first; the suffix is the position scored and is always kept), then right-padded with
    11 to `sequence` (mask 1 for real ids). Returns int32 [1, n] ids and mask, and counts."""
    pre, suf = encode_text(tok, PREFIX), encode_text(tok, SUFFIX)
    mid = encode_text(tok, body(instruction, query, document))
    budget = sequence - len(pre) - len(suf)
    if budget <= 0:
        raise SystemExit("the grid is too small for the template")
    truncated = len(mid) > budget
    ids = pre + mid[:budget] + suf
    n = len(ids)
    mask = [1] * n
    if pad:
        ids, mask = ids + [PAD_ID] * (sequence - n), mask + [0] * (sequence - n)
    info = {"prefixTokens": len(pre), "bodyTokens": len(mid), "suffixTokens": len(suf), "realTokens": n,
            "truncated": truncated, "bodyBudget": budget}
    return np.array([ids], dtype=np.int32), np.array([mask], dtype=np.int32), info


def chat_ids(tok, instruction: str, query: str, document: str) -> list[int]:
    """Upstream's own rendering: the checkpoint's chat_template.jinja over [system, user], tokenized."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": body(instruction, query, document)}]
    out = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=True)
    return list(out["input_ids"])


# MARK: upstream

def load_text_state(snapshot: Path) -> tuple[dict, list[str]]:
    """The language model's tensors from the pinned model.safetensors (`language_model.model.` prefix stripped) and
    the names of the tensors not used (vision tower, projector)."""
    from safetensors import safe_open
    state, unused = {}, []
    prefix = "language_model.model."
    with safe_open(str(snapshot / "model.safetensors"), framework="pt") as f:
        for key in f.keys():
            if key.startswith(prefix):
                state[key[len(prefix):]] = f.get_tensor(key)
            else:
                unused.append(key)
    return state, unused


class Upstream(nn.Module):
    """The unmodified transformers Ministral3Model (float32, eval, SDPA, strict load of the pinned language-model
    weights) with its tied head: upstream's text path of Mistral3ForConditionalGeneration (a text-only input never
    reaches the vision tower or projector)."""

    def __init__(self, snapshot: Path) -> None:
        super().__init__()
        from transformers import AutoConfig, Ministral3Model

        config = AutoConfig.from_pretrained(str(snapshot)).text_config
        config._attn_implementation = "sdpa"
        self.config = config
        state, self.unused = load_text_state(snapshot)
        if any(k.startswith("lm_head") for k in self.unused):
            raise SystemExit("the checkpoint has an untied lm_head; the graph assumes tied embeddings")
        self.decoder = Ministral3Model(config)
        self.decoder.load_state_dict({k: v.to(torch.float32) for k, v in state.items()}, strict=True)
        del state
        self.eval()

    def hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """The decoder's final (normed) hidden states [n, H] float32, upstream's own path (int64 ids, 2-D mask)."""
        with torch.no_grad():
            return self.decoder(input_ids=input_ids.long(),
                                attention_mask=None if attention_mask is None else attention_mask.long()
                                ).last_hidden_state[0]

    def logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Next-token logits [n, V] float32 at every position (the tied head over every row)."""
        return F.linear(self.hidden(input_ids, attention_mask), self.decoder.embed_tokens.weight)

    def head_logits(self, h: torch.Tensor) -> torch.Tensor:
        """The tied head on one position's hidden state [H]: logits [V] float32."""
        return F.linear(h[None], self.decoder.embed_tokens.weight)[0]


def helper_score(tok, logits: torch.Tensor) -> dict:
    """The model card's `unsafe_score`, line for line, on one position's logits [V]: log-softmax, top 20, max per
    class over the decoded forms, exp ratio."""
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    z_yes, z_no = -1e9, -1e9
    values, indices = logprobs.topk(20)
    for logp, token_id in zip(values.tolist(), indices.tolist()):
        t = tok.decode([token_id]).strip().lower()
        if t in YES_FORMS:
            z_yes = max(z_yes, logp)
        elif t in NO_FORMS:
            z_no = max(z_no, logp)
    import math
    score = math.exp(z_yes) / (math.exp(z_yes) + math.exp(z_no))
    return {"score": score, "zYes": z_yes, "zNo": z_no, "top": int(indices[0])}


def graph_rule(logits: torch.Tensor, yes_ids, no_ids) -> np.ndarray:
    """The graph's rule on one position's logits [V], in float64: softmax([max no, max yes])."""
    lg = logits.double()
    z = torch.stack([lg[list(no_ids)].max(), lg[list(yes_ids)].max()])
    return torch.softmax(z, 0).numpy().astype(np.float64)


def reference_rule(upstream: "Upstream", h: torch.Tensor, yes_ids, no_ids) -> np.ndarray:
    """The graph's rule on one position's upstream hidden state [H] with the head in float64 (the label rows of the
    tied embedding): the reference probabilities, free of float32 GEMM rounding in the head (a full-sequence float32
    head moves near-threshold probabilities by up to ~2e-5 against this)."""
    rows = upstream.decoder.embed_tokens.weight.detach()[list(no_ids) + list(yes_ids)].double()
    logits = torch.zeros(upstream.decoder.embed_tokens.weight.shape[0], dtype=torch.float64)
    logits[list(no_ids) + list(yes_ids)] = rows @ h.double()
    return graph_rule(logits, yes_ids, no_ids)


# MARK: the export graph

class WideLinear(nn.Module):
    """A bias-free linear layer whose weight is stored in `storage` (float16) and widened to the compute dtype."""

    def __init__(self, linear: nn.Linear, storage: torch.dtype) -> None:
        super().__init__()
        if linear.bias is not None:
            raise SystemExit("WideLinear expects bias-free layers")
        self.register_buffer("weight", fp16_checked(linear.weight.detach(), storage))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype))


class WideEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding, storage: torch.dtype, dtype: torch.dtype) -> None:
        super().__init__()
        self.register_buffer("weight", fp16_checked(embedding.weight.detach(), storage))
        self.dtype = dtype

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.weight).to(self.dtype)


def fp16_checked(w: torch.Tensor, storage: torch.dtype) -> torch.Tensor:
    narrow = w.to(storage).contiguous()
    if not torch.isfinite(narrow).all():
        raise SystemExit("a weight overflows float16 storage")
    return narrow


def widen_weights(module: nn.Module, storage: torch.dtype, dtype: torch.dtype) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, WideLinear(child, storage))
        elif isinstance(child, nn.Embedding):
            setattr(module, name, WideEmbedding(child, storage, dtype))
        else:
            widen_weights(child, storage, dtype)


class ClassifierGraph(nn.Module):
    """input_ids, attention_mask int32 [1,S] -> probs float32 [1,2] = softmax([max no-logit, max yes-logit]).

    `dtype` is the decoder's activation dtype; `storage`, when it differs, the dtype every linear and embedding weight
    is stored in (widened in the graph; only with float32 compute). RMSNorm gains keep the decoder's dtype. The head
    (the tied-embedding rows of the label variants) is a float32 buffer and the head, the max and the softmax compute
    in float32 whatever the decoder's dtype."""

    def __init__(self, upstream: Upstream, yes_ids, no_ids, dtype: torch.dtype = torch.float32,
                 storage: torch.dtype | None = None, sequence: int = SEQUENCE) -> None:
        super().__init__()
        rows = upstream.decoder.embed_tokens.weight.detach()[list(no_ids) + list(yes_ids)].to(torch.float32)
        self.register_buffer("head", rows.clone().contiguous())
        self.n_no = len(no_ids)
        self.decoder = upstream.decoder.to(dtype)
        if storage is not None and storage != dtype:
            if dtype != torch.float32:
                raise SystemExit("narrow weight storage is only supported with float32 compute")
            widen_weights(self.decoder, storage, dtype)
        self.dtype = dtype
        self.register_buffer("causal", torch.tril(torch.ones(sequence, sequence, dtype=torch.bool)), persistent=False)
        self.register_buffer("positions", torch.arange(sequence, dtype=torch.int64), persistent=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        keep = attention_mask.to(torch.bool)
        mask = (self.causal[None, None] & keep[:, None, None, :])
        hidden = self.decoder(input_ids=input_ids.to(torch.int64), attention_mask=mask,
                              position_ids=self.positions[None]).last_hidden_state
        last = attention_mask.to(torch.int64).sum(-1, keepdim=True) - 1                     # [1,1]
        h = torch.gather(hidden, 1, last[:, :, None].expand(-1, 1, hidden.shape[-1]))[:, 0]  # [1,H]
        logits = F.linear(h.to(torch.float32), self.head)                                    # [1, n_no + n_yes]
        z_no = logits[:, :self.n_no].amax(-1, keepdim=True)
        z_yes = logits[:, self.n_no:].amax(-1, keepdim=True)
        return torch.softmax(torch.cat([z_no, z_yes], dim=-1), dim=-1)


def peaks_and_hidden(upstream: Upstream, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Upstream's padded forward (2-D mask), returning its final hidden states [S, H] and the largest |value| over the
    real positions in the residual stream (each layer's output) and in the MLP products (gate*up, down_proj's
    input), float32: what float16 activations would have to hold (max 65504)."""
    peaks = {"residual": 0.0, "mlpProduct": 0.0, "perLayerResidual": []}
    n = int(attention_mask.sum())
    hooks = []

    def layer_hook(_m, _i, out):
        value = (out[0] if isinstance(out, tuple) else out)[0, :n].abs().max().item()
        peaks["perLayerResidual"].append(round(value, 1))
        peaks["residual"] = max(peaks["residual"], value)

    def down_hook(_m, inputs):
        peaks["mlpProduct"] = max(peaks["mlpProduct"], inputs[0][0, :n].abs().max().item())

    for layer in upstream.decoder.layers:
        hooks.append(layer.register_forward_hook(layer_hook))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(down_hook))
    try:
        hidden = upstream.hidden(input_ids, attention_mask)
    finally:
        for h in hooks:
            h.remove()
    peaks["residual"] = round(peaks["residual"], 1)
    peaks["mlpProduct"] = round(peaks["mlpProduct"], 1)
    return hidden, peaks


def write_f32(path: Path, x: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(x, dtype="<f4").tobytes())


def read_f32(path: Path, shape) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype="<f4").reshape(shape).copy()


def write_i32(path: Path, x: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(x, dtype="<i4").tobytes())


def read_i32(path: Path, shape) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype="<i4").reshape(shape).copy()

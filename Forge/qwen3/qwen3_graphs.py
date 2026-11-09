"""The Qwen3 dense decoder export graph and the shared harness for its exporter and self-test.

Any Qwen3 dense decoder checkpoint in Hugging Face format (`Qwen3ForCausalLM`: config.json plus safetensors) becomes one
`decoder-lm@1` graph (MANIFEST.md §1.1, DESIGN.md §6.2). The architecture comes from config.json (layers, widths, heads,
KV heads, head_dim, rope_theta, rms_norm_eps, tied or untied head); nothing about a particular checkpoint is hardcoded.
Unsupported variants are refused rather than approximated: sliding-window attention, rope scaling, attention bias,
mixture-of-experts (`qwen3_moe`) and any non-`full_attention` layer type.

**`decoder`**, one function `main`, the running contract:

* `input_ids` int32 [1, S] (S dynamic, 1..PREFILL_MAX): the fed tokens;
* `position_ids` int32 [1, T] (T dynamic): the running positions 0..T-1, the last S of them the fed tokens';
* states `keyCache`, `valueCache` [layers, 1, kv_heads, C, head_dim] (C dynamic, the sequence axis 3), written in place
  at rows T-S..T-1;
* output `logits` [1, 1, vocab]: the last fed row only (`rows: last`).

Arithmetic (the recipe the 0.0.x Qwen3 exports were qualified with, kept so their greedy output carries over): weights
and KV stored in float16 (`storage`); the residual stream, matmuls and the head in `dtype` (float16); SDPA in `attention`
(float32); RMSNorm statistics in float32, cast back before the gain; rotary frequencies computed in float64, angles in
float32, cos/sin cast to `dtype`. q and k share one normalization pass (each its own gain) and one rotation, and q/k/v
are one fused projection. The causal mask is derived from the position data, never from shapes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "source-pins.json").read_text())
DTYPES = {"float32": torch.float32, "float16": torch.float16}

PREFILL_MAX = 512              # the token axis' bound, 1..512 (DecoderEngine's default prefill chunk)
CONTEXT_MAX = 32768            # the default bound of the position and KV capacity axes (limits.maxContext)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def source_pin(source: str) -> dict:
    """The pin for `repo@revision` from source-pins.json; a source without a pin is refused."""
    repo, _, revision = source.partition("@")
    for pin in PINS["sources"]:
        if pin["repository"] == repo and (not revision or pin["revision"] == revision):
            return pin
    known = ", ".join(f"{p['repository']}@{p['revision'][:12]}" for p in PINS["sources"])
    raise SystemExit(f"{source} has no pin in Forge/qwen3/source-pins.json (pinned: {known})")


def default_snapshot(pin: dict) -> Path:
    name = pin["repository"].replace("/", "--")
    return Path.home() / f".cache/huggingface/hub/models--{name}/snapshots/{pin['revision']}"


def verify_sources(snapshot: Path, pin: dict) -> dict:
    """Checks every pinned snapshot file; returns {file: sha256}."""
    verified = {}
    for relative, expected in pin["files"].items():
        path = snapshot / relative
        if not path.is_file():
            raise SystemExit(f"snapshot lacks {relative} ({pin['repository']}@{pin['revision'][:12]})")
        if sha256_file(path) != expected:
            raise SystemExit(f"{relative} differs from the pinned {pin['revision'][:12]} file")
        verified[relative] = expected
    return verified


def read_config(checkpoint: Path) -> dict:
    return json.loads((checkpoint / "config.json").read_text())


def check_config(cfg: dict) -> None:
    """Refuses what the graph does not implement (see the module doc)."""
    if cfg.get("model_type") != "qwen3":
        raise SystemExit(f"model_type {cfg.get('model_type')!r} is not a Qwen3 dense decoder ('qwen3')")
    if cfg.get("architectures") not in (None, ["Qwen3ForCausalLM"]):
        raise SystemExit(f"architectures {cfg.get('architectures')} is not Qwen3ForCausalLM")
    if cfg.get("use_sliding_window") or cfg.get("rope_scaling") or cfg.get("attention_bias"):
        raise SystemExit("sliding-window attention, rope scaling and attention bias are not supported")
    if cfg.get("rope_parameters", {}).get("rope_type", "default") != "default":
        raise SystemExit(f"rope type {cfg['rope_parameters']['rope_type']!r} is not supported")
    if any(t != "full_attention" for t in cfg.get("layer_types") or []):
        raise SystemExit("every layer must be full_attention")
    if cfg.get("hidden_act", "silu") != "silu":
        raise SystemExit(f"hidden_act {cfg['hidden_act']!r} is not supported")


def rope_theta(cfg: dict) -> float:
    if "rope_theta" in cfg:
        return float(cfg["rope_theta"])
    return float(cfg["rope_parameters"]["rope_theta"])     # transformers 5 writes it under rope_parameters


def head_dim(cfg: dict) -> int:
    return cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]


def checked(w: torch.Tensor, storage: torch.dtype) -> torch.Tensor:
    narrow = w.detach().to(storage).contiguous()
    if not torch.isfinite(narrow).all():
        raise SystemExit("a weight overflows its storage dtype")
    return narrow


def rms_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    """float32 statistics (float16 squares overflow), cast back to the input dtype."""
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sin


class Layer(nn.Module):
    def __init__(self, cfg: dict, w: dict[str, torch.Tensor], i: int, storage: torch.dtype,
                 attention: torch.dtype) -> None:
        super().__init__()
        p = f"model.layers.{i}."
        self.heads, self.kv_heads, self.head_dim = cfg["num_attention_heads"], cfg["num_key_value_heads"], head_dim(cfg)
        self.eps, self.attention = cfg["rms_norm_eps"], attention
        self.register_buffer("qkv", checked(torch.cat([w.pop(p + f"self_attn.{n}_proj.weight") for n in "qkv"]), storage))
        self.register_buffer("o_proj", checked(w.pop(p + "self_attn.o_proj.weight"), storage))
        self.register_buffer("gains", checked(torch.cat((
            w.pop(p + "self_attn.q_norm.weight").expand(self.heads, -1),
            w.pop(p + "self_attn.k_norm.weight").expand(self.kv_heads, -1))), storage))
        self.register_buffer("in_norm", checked(w.pop(p + "input_layernorm.weight"), storage))
        self.register_buffer("post_norm", checked(w.pop(p + "post_attention_layernorm.weight"), storage))
        self.register_buffer("gate", checked(w.pop(p + "mlp.gate_proj.weight"), storage))
        self.register_buffer("up", checked(w.pop(p + "mlp.up_proj.weight"), storage))
        self.register_buffer("down", checked(w.pop(p + "mlp.down_proj.weight"), storage))

    def forward(self, x, cos, sin, key_cache, value_cache, layer: int, start, end, mask):
        s, dt = x.shape[1], x.dtype
        h = rms_normalize(x, self.eps) * self.in_norm.to(dt)
        qkv = F.linear(h, self.qkv.to(dt)).view(1, s, self.heads + 2 * self.kv_heads, self.head_dim)
        qk = qkv[:, :, :self.heads + self.kv_heads]
        v = qkv[:, :, self.heads + self.kv_heads:].transpose(1, 2)
        qk = rotate(rms_normalize(qk, self.eps) * self.gains.to(dt), cos[:, None, :], sin[:, None, :])
        q, k = qk[:, :, :self.heads].transpose(1, 2), qk[:, :, self.heads:].transpose(1, 2)
        # In-place cache write at shape-derived bounds (a slice, not an integer, layer index), then attention over the
        # first `end` rows.
        key_cache[layer:layer + 1, :, :, start:end, :] = k.to(key_cache.dtype)[None]
        value_cache[layer:layer + 1, :, :, start:end, :] = v.to(value_cache.dtype)[None]
        keys = key_cache[layer, :, :, :end, :]
        values = value_cache[layer, :, :, :end, :]
        at = self.attention
        att = F.scaled_dot_product_attention(q.to(at), keys.to(at), values.to(at), attn_mask=mask,
                                             enable_gqa=self.heads != self.kv_heads).to(dt)
        x = x + F.linear(att.transpose(1, 2).reshape(1, s, self.heads * self.head_dim), self.o_proj.to(dt))
        h = rms_normalize(x, self.eps) * self.post_norm.to(dt)
        return x + F.linear(F.silu(F.linear(h, self.gate.to(dt))) * F.linear(h, self.up.to(dt)), self.down.to(dt))


class Decoder(nn.Module):
    """A Qwen3 dense decoder on the `decoder-lm@1` running contract (see the module doc).

    `weights`: the checkpoint's tensors by name (consumed); every one must be used. `dtype`: the residual stream,
    matmuls and the head; `attention`: SDPA; `storage`: weights and KV states; `logits`: the output's dtype."""

    def __init__(self, cfg: dict, weights: dict[str, torch.Tensor], dtype: torch.dtype = torch.float16,
                 attention: torch.dtype = torch.float32, storage: torch.dtype = torch.float16,
                 logits: torch.dtype = torch.float16, context: int = CONTEXT_MAX) -> None:
        super().__init__()
        check_config(cfg)
        self.cfg, self.dtype, self.storage, self.logits_dtype, self.context = cfg, dtype, storage, logits, context
        self.vocab, self.eps = cfg["vocab_size"], cfg["rms_norm_eps"]
        self.tied = bool(cfg.get("tie_word_embeddings", False))
        w = dict(weights)
        weights.clear()
        self.register_buffer("embed", checked(w.pop("model.embed_tokens.weight"), storage))
        head = w.pop("lm_head.weight", None)
        if self.tied:
            self.head = None
        elif head is None:
            raise SystemExit("an untied head needs lm_head.weight")
        else:
            self.register_buffer("head", checked(head, storage))
        self.layers = nn.ModuleList(Layer(cfg, w, i, storage, attention) for i in range(cfg["num_hidden_layers"]))
        self.register_buffer("norm", checked(w.pop("model.norm.weight"), storage))
        if w:
            raise SystemExit(f"unused checkpoint tensors: {sorted(w)[:8]}")
        d = head_dim(cfg)
        self.inv_freq = (1.0 / (rope_theta(cfg) ** (torch.arange(0, d, 2, dtype=torch.float64) / d))).float()

    def forward(self, input_ids, position_ids, key_cache, value_cache):
        s = input_ids.shape[1]
        total = position_ids.shape[1]
        positions = position_ids[0]
        p = positions[total - s:]                                                    # [S] the fed tokens' positions
        x = F.embedding(input_ids, self.embed).to(self.dtype)                        # [1, S, H]
        angles = p.float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat((angles, angles), dim=-1)
        cos, sin = emb.cos().to(self.dtype), emb.sin().to(self.dtype)
        mask = (positions[None, :] <= p[:, None])[None, None]                        # row i sees keys 0..total-s+i
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, key_cache, value_cache, i, total - s, total, mask)
        hidden = rms_normalize(x[:, -1:], self.eps) * self.norm.to(self.dtype)
        head = self.embed if self.head is None else self.head
        return F.linear(hidden, head.to(self.dtype)).to(self.logits_dtype)

    def empty_state(self, capacity: int) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.cfg
        shape = (c["num_hidden_layers"], 1, c["num_key_value_heads"], capacity, head_dim(c))
        return torch.zeros(shape, dtype=self.storage), torch.zeros(shape, dtype=self.storage)

    def example(self):
        """(example inputs, dynamic shapes) for torch.export; the example dims all differ so export keeps them apart."""
        s, total, capacity = 8, 24, 56
        kc, vc = self.empty_state(capacity)
        example = (torch.arange(s, dtype=torch.int32)[None] + 100, torch.arange(total, dtype=torch.int32)[None], kc, vc)
        s_dim = torch.export.Dim("tokens", min=1, max=PREFILL_MAX)
        t_dim = torch.export.Dim("total", min=2, max=self.context)
        c_dim = torch.export.Dim("capacity", min=2, max=self.context)
        return example, ({1: s_dim}, {1: t_dim}, {3: c_dim}, {3: c_dim})


def load_weights(checkpoint: Path) -> dict[str, torch.Tensor]:
    """Every tensor of the checkpoint's safetensors files, by name, in its stored dtype."""
    from safetensors import safe_open

    files = sorted(checkpoint.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"no .safetensors in {checkpoint}")
    out: dict[str, torch.Tensor] = {}
    for f in files:
        with safe_open(str(f), framework="pt") as st:
            for key in st.keys():
                if key in out:
                    raise SystemExit(f"{key} appears in two shards")
                out[key] = st.get_tensor(key)
    return out


def step(decoder: Decoder, ids: list[int], start: int, key_cache, value_cache) -> torch.Tensor:
    """One host call, as the engine makes it: feeds `ids` at positions start..start+len-1; last-row logits [vocab]."""
    total = start + len(ids)
    return decoder(torch.tensor([ids], dtype=torch.int32), torch.arange(total, dtype=torch.int32)[None],
                   key_cache, value_cache)[0, 0]

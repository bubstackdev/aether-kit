"""The GLiNER2.5-Decide export graph and the shared harness for its exporter, reference and checks.

GLiNER2.5-Decide (fastino/GLiNER2.5-Decide, Apache-2.0, Fastino) is a GLiNER2 span-architecture checkpoint fine-tuned
for classification from fastino/gliner2-large-v1, on microsoft/deberta-v3-large (MIT). Its classification path, in the
gliner2 library (PyPI `gliner2` 2.0.0, Apache-2.0, imported read-only from the pinned wheel), is:

* the host lays the tasks out ahead of the text, `( [P] task ( [L] label [L] label ... ) ) [SEP_STRUCT] ... [SEP_TEXT]
  words`, every piece tokenized on its own (no CLS/SEP), the text words lowercased;
* transformers' `DebertaV2Model` encodes the ids (24 layers, hidden 1024, disentangled attention over log-bucketed
  relative positions);
* `processor.extract_embeddings_from_batch` gathers the hidden state of each task's `[P]` and `[L]` markers;
* `classifier` (Linear 1024 -> 2048, ReLU, Linear 2048 -> 1) maps every `[L]` row to one logit;
* softmax over a single-label task's logits (argmax decides), a sigmoid per label for a multi-label task (every label
  at or above the threshold, default 0.5; none -> the argmax).

The exported graph (`ClassifierGraph`) is the encoder, the `[L]` gather and the classifier at a fixed sequence S (one
function per bucket, S = 256 and 512):

* inputs `input_ids` int32 [1, S] (the laid-out ids, right-padded with 0), `attention_mask` int32 [1, S] (1 for real
  ids, 0 for padding) and `label_idx` int32 [1, 32] (the `[L]` positions of every task, concatenated in task order;
  unused slots repeat the first);
* output `logits` float32 [1, 32] (slot i is label_idx[i]'s logit; the host reads only the used slots).

One export-only substitution: DeBERTa's relative-position bucket table (`build_relative_position`, a function of S
only) is computed once in PyTorch by upstream's own `DebertaV2Encoder.get_rel_pos` and passed to the encoder as a
constant (upstream accepts `relative_pos`), and the module's TorchScript helpers are traced as their plain Python
equivalents. `selftest_gliner.py` checks that the graph equals upstream's own forward in float32.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "source-pins.json").read_text())
REPO, REVISION = PINS["repository"], PINS["revision"]
CODE = PINS["upstreamCode"]
DEFAULT_SNAPSHOT = Path.home() / f".cache/huggingface/hub/models--fastino--GLiNER2.5-Decide/snapshots/{REVISION}"
DEFAULT_CODE = Path.home() / f".cache/aether-kit/gliner/upstream/gliner2-{CODE['version']}"
ZOO = PINS["zooFixtures"]
DEFAULT_ZOO = (Path.home() / ".cache/huggingface/hub/models--mlboydaisuke--GLiNER2.5-Decide-CoreAI/snapshots"
               / ZOO["revision"])

SEQUENCES = (256, 512)        # the buckets (the zoo export's two graphs; DeBERTa's 512 positions)
MAX_LABELS = 32               # label_idx slots per call, over all tasks
HIDDEN = 1024
PAD_ID = 0
THRESHOLD = 0.5               # gliner2's cls_threshold default
MARKERS = ("[MASK]", "[SEP_STRUCT]", "[SEP_TEXT]", "[P]", "[C]", "[E]", "[R]", "[L]", "[EXAMPLE]", "[OUTPUT]",
           "[DESCRIPTION]")
DTYPES = {"float32": torch.float32, "float16": torch.float16}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sources(snapshot: Path, code: Path = DEFAULT_CODE) -> dict:
    """Checks every pinned snapshot file and the unpacked gliner2 wheel; returns {file: sha256}."""
    verified = {}
    for relative, expected in PINS["files"].items():
        path = snapshot / relative
        if not path.is_file():
            raise SystemExit(f"snapshot lacks {relative}; run Forge/gliner/fetch_upstream.py ({REPO}@{REVISION[:12]})")
        if sha256_file(path) != expected:
            raise SystemExit(f"{relative} differs from the pinned {REVISION[:12]} file")
        verified[relative] = expected
    stamp = code / "UNPACKED.json"
    if not stamp.is_file() or json.loads(stamp.read_text()).get("wheelSHA256") != CODE["wheelSHA256"]:
        raise SystemExit(f"no verified gliner2 {CODE['version']} unpack at {code}; run Forge/gliner/fetch_upstream.py")
    for relative, expected in CODE["files"].items():
        if sha256_file(code / relative) != expected:
            raise SystemExit(f"{relative} in {code} differs from its pin")
        verified[f"gliner2:{relative}"] = expected
    return verified


def verify_zoo(zoo: Path) -> dict:
    verified = {}
    for relative, expected in ZOO["files"].items():
        if not (zoo / relative).is_file() or sha256_file(zoo / relative) != expected:
            raise SystemExit(f"zoo {relative} missing or not the pinned file; run fetch_upstream.py --zoo")
        verified[relative] = expected
    return verified


def import_gliner2(code: Path = DEFAULT_CODE):
    """The pinned gliner2 package, imported read-only from its unpacked wheel (never installed)."""
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    import gliner2
    if gliner2.__version__ != CODE["version"] or Path(gliner2.__file__).resolve().parent != (code / "gliner2").resolve():
        raise SystemExit(f"imported gliner2 {gliner2.__version__} from {gliner2.__file__}, not the pinned unpack")
    return gliner2


def load_upstream(snapshot: Path, code: Path = DEFAULT_CODE):
    """gliner2's own AutoExtractor on the pinned checkpoint (float32, eval, CPU, strict load)."""
    import contextlib as _cl
    import io
    gliner2 = import_gliner2(code)
    with _cl.redirect_stdout(io.StringIO()):          # the loader prints a configuration banner
        model = gliner2.AutoExtractor.from_pretrained(str(snapshot))
    model = model.float().eval()
    encoder = model.encoder
    if (type(encoder).__name__ != "DebertaV2Model" or getattr(encoder, "z_steps", 0) > 1
            or encoder.config.hidden_size != HIDDEN or getattr(encoder.encoder, "conv", None) is not None
            or encoder.embeddings.position_embeddings is not None or encoder.embeddings.token_type_embeddings is not None):
        raise SystemExit("unexpected encoder: the graph assumes DeBERTa-v3 (relative attention only, no conv, z_steps 0)")
    mlp = [type(m).__name__ for m in model.classifier]
    if mlp != ["Linear", "ReLU", "Linear"] or model.classifier[0].in_features != HIDDEN \
            or model.classifier[2].out_features != 1 or model.processor.token_pooling != "first":
        raise SystemExit(f"unexpected classifier {mlp} or token pooling")
    return model


def marker_ids(model) -> dict:
    """The schema marker ids in upstream's tokenizer (as gliner2's processor registers them)."""
    tok = model.processor.tokenizer
    ids = {m: int(tok.convert_tokens_to_ids(m)) for m in MARKERS}
    if len(set(ids.values())) != len(MARKERS) or any(v == tok.unk_token_id for v in ids.values()):
        raise SystemExit(f"marker tokens are not distinct added tokens: {ids}")
    return ids


# MARK: upstream's own classification run, with its tensors captured

def run_upstream(model, text: str, tasks: dict, threshold: float = THRESHOLD) -> dict:
    """Runs gliner2's own `classify_text` on CPU float32 and captures what its classification path computed: the
    encoder's input ids and mask, each task's marker positions (from the batch `extract_embeddings_from_batch` reads)
    and the classifier's logits per task. `tasks` is classify_text's argument as is ({name: [labels]} or {name:
    {labels (a list, or {label: description}), multi_label, cls_threshold, prompt}}). Returns the captured tensors, the
    per-task logits, probabilities and decisions (upstream's rule, upstream's resolved task configs), and classify_text's
    own formatted answer, which must agree."""
    captured: dict = {"logits": []}

    def on_encoder(_module, _args, kwargs, _out):
        captured["input_ids"] = kwargs["input_ids"].detach().clone()
        captured["attention_mask"] = kwargs["attention_mask"].detach().clone()

    def on_classifier(_module, _args, out):
        captured["logits"].append(out.detach().squeeze(-1).to(torch.float32).clone())

    original = model.processor.extract_embeddings_from_batch

    def spy(token_embeddings, input_ids, batch):
        captured["special"] = [list(map(int, s)) for s in batch.schema_special_indices[0][:batch.schema_counts[0]]]
        captured["schema_tokens"] = batch.schema_tokens_list[0]
        captured["task_types"] = list(batch.task_types[0])
        return original(token_embeddings, input_ids, batch)

    hooks = [model.encoder.register_forward_hook(on_encoder, with_kwargs=True),
             model.classifier.register_forward_hook(on_classifier)]
    model.processor.extract_embeddings_from_batch = spy
    try:
        answer = model.classify_text(text, tasks, threshold=threshold, include_confidence=True)
    finally:
        for h in hooks:
            h.remove()
        del model.processor.extract_embeddings_from_batch

    configs = model._classification_schema(tasks).build()["classifications"]       # upstream's resolved configs
    ids = captured["input_ids"]
    if ids.shape[0] != 1 or not bool((captured["attention_mask"] == 1).all()):
        raise SystemExit("upstream encoded more than one row or padded a single text")
    order = [j for j, t in enumerate(captured["task_types"]) if t == "classifications"]
    if not (len(order) == len(configs) == len(captured["logits"]) == len(tasks)):
        raise SystemExit("captured classification schemas do not match the tasks")
    results, positions = [], []
    for config, j, logits in zip(configs, order, captured["logits"]):
        name, labels = config["task"], list(config["labels"])
        markers = captured["special"][j][1:]
        if not captured["schema_tokens"][j][2].startswith(name) or len(markers) != len(labels) or \
                logits.shape != (len(labels),):
            raise SystemExit(f"task {name}: schema order or marker count differs from the request")
        if config.get("class_act", "auto") != "auto":
            raise SystemExit(f"task {name}: class_act {config['class_act']} is not the default rule")
        results.append({"task": name, "labels": labels, "multiLabel": bool(config["multi_label"]),
                        "threshold": float(config["cls_threshold"]), "prompt": captured["schema_tokens"][j][2],
                        "start": len(positions), "logits": [float(v) for v in logits.numpy().astype(np.float32)]})
        positions.extend(markers)
    for r in results:
        r["probs"], r["decision"] = decide(np.asarray(r["logits"], np.float32), r["labels"], r["multiLabel"],
                                           r["threshold"])
        got = answer[r["task"]]
        theirs = [g["label"] for g in got] if isinstance(got, list) else got["label"]
        if theirs != r["decision"]:
            raise SystemExit(f"task {r['task']}: classify_text decided {theirs}, the captured logits {r['decision']}")
    return {"input_ids": ids[0].numpy().astype(np.int32), "labelPositions": np.asarray(positions, np.int32),
            "tasks": results, "classifyText": answer}


def decide(logits: np.ndarray, labels: list[str], multi: bool, threshold: float = THRESHOLD):
    """gliner2's `_extract_classification_result` rule (float32, as torch computes it)."""
    t = torch.from_numpy(np.asarray(logits, np.float32))
    probs = (torch.sigmoid(t) if multi else torch.softmax(t, dim=-1)).numpy()
    if multi:
        chosen = [labels[j] for j in range(len(labels)) if probs[j] >= threshold]
        decision = chosen or [labels[int(np.argmax(probs))]]
    else:
        decision = labels[int(np.argmax(probs))]
    return [float(p) for p in probs], decision


# MARK: graph inputs

def bucket_for(length: int) -> int:
    for s in SEQUENCES:
        if length <= s:
            return s
    raise SystemExit(f"{length} ids exceed the largest bucket {SEQUENCES[-1]}")


def graph_inputs(ids: np.ndarray, positions: np.ndarray, sequence: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The graph's three int32 inputs for one laid-out text: ids right-padded with 0 to `sequence`, the mask, and the
    marker positions padded to 32 slots with the first."""
    n, m = int(ids.shape[0]), int(positions.shape[0])
    if n > sequence or not 0 < m <= MAX_LABELS:
        raise SystemExit(f"{n} ids / {m} labels do not fit [{sequence}] / [{MAX_LABELS}]")
    padded = np.full((1, sequence), PAD_ID, np.int32)
    padded[0, :n] = ids
    mask = np.zeros((1, sequence), np.int32)
    mask[0, :n] = 1
    label_idx = np.full((1, MAX_LABELS), positions[0], np.int32)
    label_idx[0, :m] = positions
    return padded, mask, label_idx


# MARK: the export graph

def _scaled_size_sqrt(query_layer: torch.Tensor, scale_factor: int):
    return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)


def _build_rpos(query_layer, key_layer, relative_pos, position_buckets: int, max_relative_positions: int):
    if key_layer.size(-2) != query_layer.size(-2):
        raise RuntimeError("the export graph is self-attention only (query and key lengths are equal)")
    return relative_pos


@contextlib.contextmanager
def traceable_deberta():
    """Replaces modeling_deberta_v2's TorchScript helpers on the export path with their plain Python bodies (the same
    arithmetic; torch.export does not trace through ScriptFunctions), undone after."""
    from transformers.models.deberta_v2 import modeling_deberta_v2 as m
    saved = {name: getattr(m, name) for name in ("scaled_size_sqrt", "build_rpos")}
    m.scaled_size_sqrt, m.build_rpos = _scaled_size_sqrt, _build_rpos
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(m, name, value)


class ClassifierGraph(nn.Module):
    """input_ids, attention_mask int32 [1,S], label_idx int32 [1,32] -> logits float32 [1,32].

    `dtype` is the encoder's and classifier's compute (and weight) dtype; the logits are widened to float32."""

    def __init__(self, upstream, sequence: int, dtype: torch.dtype = torch.float32,
                 share: "ClassifierGraph | None" = None) -> None:
        """With `share`, this graph uses `share`'s (already cast) modules, the same parameter objects, so a converter
        given both stores the weights once; otherwise it casts its own copy of upstream's."""
        super().__init__()
        if share is None:
            encoder = copy.deepcopy(upstream.encoder)
            self.embeddings = encoder.embeddings.to(dtype)
            self.encoder = encoder.encoder.to(dtype)
            self.classifier = copy.deepcopy(upstream.classifier).to(dtype)
        else:
            if share.dtype != dtype:
                raise SystemExit("a shared graph must have the same dtype")
            self.embeddings, self.encoder, self.classifier = share.embeddings, share.encoder, share.classifier
        self.sequence, self.dtype = sequence, dtype
        with torch.no_grad():
            relative = self.encoder.get_rel_pos(torch.zeros(1, sequence, HIDDEN))     # upstream's own table
        if relative.shape != (1, sequence, sequence) or relative.dtype != torch.long:
            raise SystemExit(f"unexpected relative-position table {tuple(relative.shape)} {relative.dtype}")
        self.register_buffer("relative_pos", relative)
        self.eval()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, label_idx: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.to(torch.int64)
        embedded = self.embeddings(input_ids=input_ids.to(torch.int64), mask=mask)
        hidden = self.encoder(embedded, mask, output_hidden_states=False, relative_pos=self.relative_pos,
                              return_dict=True).last_hidden_state
        rows = hidden[0].index_select(0, label_idx[0].to(torch.int64))
        return self.classifier(rows).squeeze(-1).unsqueeze(0).to(torch.float32)


def upstream_logits(model, ids: np.ndarray, mask: np.ndarray, label_idx: np.ndarray) -> np.ndarray:
    """Upstream's own modules on any (padded or not) row: DebertaV2Model forward, the marker gather, the classifier.
    [1, 32] float32."""
    with torch.no_grad():
        hidden = model.encoder(input_ids=torch.from_numpy(ids).long(),
                               attention_mask=torch.from_numpy(mask).long()).last_hidden_state
        rows = hidden[0].index_select(0, torch.from_numpy(label_idx[0]).long())
        return model.classifier(rows).squeeze(-1).unsqueeze(0).numpy().astype(np.float32)


# MARK: comparison

def task_metrics(got: np.ndarray, reference_tasks: list[dict]) -> list[dict]:
    """Per task: max |delta prob|, max |delta logit|, same argmax and same decision, from the graph's 32 logits."""
    out = []
    for t in reference_tasks:
        n = len(t["labels"])
        logits = np.asarray(got, np.float32).ravel()[t["start"]:t["start"] + n]
        probs, decision = decide(logits, t["labels"], t["multiLabel"], t["threshold"])
        want_p = np.asarray(t["probs"], np.float64)
        out.append({"task": t["task"], "maxAbsProb": float(np.abs(np.asarray(probs, np.float64) - want_p).max()),
                    "maxAbsLogit": float(np.abs(logits.astype(np.float64) - np.asarray(t["logits"], np.float64)).max()),
                    "sameArgmax": int(np.argmax(logits)) == int(np.argmax(np.asarray(t["logits"]))),
                    "sameDecision": decision == t["decision"]})
    return out


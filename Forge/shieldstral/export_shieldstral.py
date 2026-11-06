#!/usr/bin/env python3
"""Export Shieldstral 1.0 3B (mistralai/Shieldstral-1.0-3B, Apache-2.0) as an Aether-Kit `yes-no-classifier@1` bundle.

    uv sync --frozen --project Forge
    Forge/.venv/bin/python -I Forge/shieldstral/fetch_upstream.py      # pinned checkpoint, once (7.7 GB)
    Forge/.venv/bin/python -I Forge/shieldstral/export_shieldstral.py --output <new bundle dir>
    Forge/.venv/bin/python -I Forge/shieldstral/export_shieldstral.py --sample-manifest

Nothing is downloaded: every file read is verified against `source-pins.json`. Conversion runs on the CPU. The output
directory is laid out as an installed revision root:

    OUTPUT/
      .aether/aether.json                 the Aether manifest (MANIFEST.md §1.3, §2; driver `yes-no-classifier@1`)
      shieldstral_1_0_3b.aimodel          input_ids, attention_mask int32 [1,512] -> probs [1,2] fp32 = [P(no), P(yes)]
      tokenizer/                          upstream's tokenizer.json, tokenizer_config.json, chat_template.jinja
      LICENSE                             the Apache License 2.0
      NOTICE                              attribution, upstream's use statement, the modification notice
      README.md                           the Hugging Face model card
      SHA256SUMS                          every bundle file except itself and the log
      export-report.json                  provenance, versions, timings, peak RSS
      forge-log.jsonl                     one event per stage

The bundle is a policy-agnostic classifier: it carries the fixed system prompt and the template, but no instruction or
query (no safety policy). The caller states the policy; the model returns P(the Document meets the Query).

One Core AI JIT package serves macOS (no per-architecture AOT): `--platform` only sets the manifest's `variant`. The
export is reproducible as embeddinggemma's is: generated function names come from a seeded generator, the package's
creation date is fixed (`--creation-date`, default the pin date) and Swift hashing is deterministic, so two exports give
byte-identical bundle files except the graph's `main.mlirb`/`main.hash` (coreai-core seeds its weight-blob names per
execution); the two graphs compute bit-identical probabilities (compare check_shieldstral.py's dumped outputs).
Parity: `reference_shieldstral.py` (transformers' Ministral 3 decoder, CPU float32) records the references;
`check_shieldstral.py` runs the exported graph through Core AI on the GPU and gates it (probabilities within 0.01, the
same decision).
"""

from __future__ import annotations

import os
import sys

# Swift's per-process hash seed orders the Core AI runtime's dictionaries (the package's metadata.json keys). The
# runtime reads SWIFT_DETERMINISTIC_HASHING at process start, so re-exec once with it set.
if os.environ.get("SWIFT_DETERMINISTIC_HASHING") != "1":
    os.environ["SWIFT_DETERMINISTIC_HASHING"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.orig_argv[1:]])

import argparse  # noqa: E402
import contextlib  # noqa: E402
import datetime as dt  # noqa: E402
import gc  # noqa: E402
import importlib  # noqa: E402
import importlib.metadata  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import resource  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from torch import nn  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shieldstral_graphs as sg  # noqa: E402

ID = "shieldstral-3b"
HF_REPO = "aether-models/aether-kit-shieldstral-3b"
ASSET = "shieldstral_1_0_3b.aimodel"
FUNCTION = "main"
INPUTS, OUTPUTS = ["input_ids", "attention_mask"], ["probs"]
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
NOT_BUNDLE = {".aether/aether.json", "SHA256SUMS", "export-report.json", "forge-log.jsonl"}
# Measured default (MANIFEST §0): GPU for every graph class; the graph is fully static.
COMPUTE = "gpu"
# The precision recipe (measured by selftest_shieldstral.py and check_shieldstral.py; see PRECISION_DECISION).
# Measured (selftest_shieldstral.py on the 25 reference cases, eager CPU, against upstream float32): float16 activations
# are safe (the float32 residual stream peaks at 436 and the MLP products at 1641, far below 65504; float16 compute
# moves P(yes) by at most 0.00085), but weight quantization misses the 0.01 bar on the near-threshold cases: int8
# per-channel 0.0176, int8 block-32 0.0200, int4 block-32 0.0905 (no decision flipped). So the weights stay float16.
COMPUTE_DTYPE = "float16"
WEIGHTS = "float16"
PRECISION_DECISION = (
    "float16 weights and compute for the decoder; head, softmax and output float32. float16 activations do not "
    "overflow (float32 peaks: residual 436, MLP product 1641) and move P(yes) by <= 0.00085 eagerly; every weight "
    "quantization measured misses the 0.01 bar near the threshold (int8 per-channel 0.0176, int8 block-32 0.0200, "
    "int4 block-32 0.0905; selftest_shieldstral.py recipes float16/int8, float16/int8b32, float16/int4), so the "
    "bundle is ~6.9 GB rather than the zoo's 2.5 GB int4")
CREATION_DATE = "2026-11-06T00:00:00+00:00"     # the pin date; fixed so the package bytes are reproducible
USE_STATEMENT = ("You must not use this model in a manner that infringes, misappropriates, or otherwise violates any "
                 "third party's rights, including intellectual property rights.")


class Log:
    def __init__(self, path: Path) -> None:
        self.path, self.started = path, {}

    def event(self, stage: str, message: str, **fields) -> None:
        record = {"message": message, "stage": stage, "t": round(time.time(), 3), **fields}
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"[shieldstral:{stage}] {message}" + (f" {fields}" if fields else ""), file=sys.stderr, flush=True)

    def start(self, stage: str) -> None:
        self.started[stage] = time.monotonic()
        self.event(stage, "start")

    def done(self, stage: str, **fields) -> float:
        seconds = round(time.monotonic() - self.started.pop(stage), 2)
        self.event(stage, "done", seconds=seconds, peakRSSGB=peak_rss_gb(), **fields)
        return seconds


def peak_rss_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)    # macOS reports bytes


def tree_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def tree_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in tree_files(root))


# MARK: the precision recipe

def quantize(module: nn.Module, example: tuple, weights: str) -> tuple[nn.Module, dict]:
    """Weight-only symmetric PTQ (coreai-opt, eager) on every decoder Linear: int8 per output channel, or int4 in
    blocks of 32 along the input axis. The embedding table, the RMSNorm gains and the head rows are not quantized."""
    from coreai_opt.quantization import Quantizer
    from coreai_opt.quantization.config import ModuleQuantizerConfig, QuantizerConfig
    from coreai_opt.quantization.spec.granularity import PerBlockGranularity, PerChannelGranularity
    from coreai_opt.quantization.spec.qscheme import QuantizationScheme
    from coreai_opt.quantization.spec.spec import QuantizationSpec

    bits, block = QUANT[weights]
    granularity = (PerChannelGranularity(axis=None) if block is None
                   else PerBlockGranularity(axis=1, block_size=block))
    spec = QuantizationSpec(dtype=torch.int8 if bits == 8 else torch.int4, qscheme=QuantizationScheme.SYMMETRIC,
                            granularity=granularity)
    config = ModuleQuantizerConfig(op_input_spec=None, op_output_spec=None, op_state_spec={"weight": spec})
    selected = {f"^{re.escape(path)}$": config for path, child in module.named_modules()
                if isinstance(child, nn.Linear) and path.startswith("decoder.layers.")}
    quantizer = Quantizer(module, QuantizerConfig(global_config=None, module_name_configs=selected,
                                                  execution_mode="eager"))
    with torch.no_grad():
        prepared = quantizer.prepare(tuple(t.clone() for t in example))
        return quantizer.finalize(prepared).eval(), {"linear": len(selected), "scheme": QUANT_SCHEMES[weights]}


# weights name -> (bits, block along the input axis; None = one scale per output channel)
QUANT = {"int8": (8, None), "int8b32": (8, 32), "int4": (4, 32)}
QUANT_SCHEMES = {"int8": "int8-linear-perchannel-symmetric", "int8b32": "int8-linear-block32-symmetric",
                 "int4": "int4-linear-block32-symmetric"}


def build_graph(snapshot: Path, compute: str, weights: str, example: tuple | None = None):
    """The export module for a recipe: `compute` the decoder's activation dtype; `weights` float16 (stored float16;
    widened when compute is float32), float32, or int8/int4 (quantized decoder linears over float16 compute)."""
    upstream = sg.Upstream(snapshot)
    dtype = sg.DTYPES[compute]
    storage = sg.DTYPES.get(weights)
    graph = sg.ClassifierGraph(upstream, sg.YES_IDS, sg.NO_IDS, dtype,
                               storage if storage is not None and storage != dtype else None).eval()
    del upstream
    info = {}
    if weights in QUANT_SCHEMES:
        graph, info = quantize(graph, example, weights)
    return graph, info


@contextlib.contextmanager
def deterministic_conversion(creation_date: int):
    """Makes the converter's output reproducible, undone after (export_embeddinggemma.py documents the three sources:
    Core AI's `secrets.choice` and coreai-torch's `uuid4()` name suffixes, and `save_asset`'s `time.time()`)."""
    graph_dialect = importlib.import_module("coreai._compiler.dialects.coreai.graph")
    func_dialect = importlib.import_module("coreai._compiler.dialects.udml.func")
    asset_module = importlib.import_module("coreai.authoring.asset")
    externalize = importlib.import_module("coreai_torch.externalize")

    rng = random.Random(0)
    saved = [(graph_dialect, "choice"), (func_dialect, "choice"), (externalize, "uuid4"), (asset_module, "time")]
    originals = [(m, name, getattr(m, name)) for m, name in saved]
    graph_dialect.choice = func_dialect.choice = rng.choice
    externalize.uuid4 = lambda: uuid.UUID(int=rng.getrandbits(128), version=4)
    asset_module.time = types.SimpleNamespace(time=lambda: float(creation_date))
    try:
        yield
    finally:
        for m, name, value in originals:
            setattr(m, name, value)


def convert(module, example: tuple, path: Path, creation_date: int, recipe: str) -> dict:
    import coreai_torch as ct
    from coreai.runtime import AIModelAssetMetadata

    with deterministic_conversion(creation_date):
        t0 = time.monotonic()
        with torch.no_grad():
            ep = torch.export.export(module, example, strict=False)
            ep = ep.run_decompositions(ct.get_decomp_table())
        t1 = time.monotonic()
        converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
        converter.add_exported_program(ep, input_names=INPUTS, output_names=OUTPUTS, state_names=[],
                                       entrypoint_name=FUNCTION)
        del ep
        gc.collect()
        program = converter.to_coreai()
        del converter
        gc.collect()
        program.optimize()
        metadata = AIModelAssetMetadata()
        metadata.author = "Mistral AI (Shieldstral 1.0 3B); Core AI export: Aether-Kit Forge"
        metadata.license = "Apache-2.0"
        metadata.model_description = (
            f"Shieldstral 1.0 3B ({sg.REPO}@{sg.REVISION[:12]}), text only, {recipe}: input_ids, attention_mask int32 "
            f"[1,{sg.SEQUENCE}] (the flattened template, right-padded with {sg.PAD_ID}) -> probs [1,2] float32 = "
            "[P(no), P(yes)] at the last real token (max over the yes/no token variants, softmax in the graph).")
        metadata.creation_date = creation_date
        program.save_asset(path, metadata)
    return {"exportSeconds": round(t1 - t0, 1), "convertSeconds": round(time.monotonic() - t1, 1),
            "bytes": tree_bytes(path), "inputs": INPUTS, "outputs": OUTPUTS, "function": FUNCTION}


# MARK: the manifest (MANIFEST.md §1.3 `yes-no-classifier@1`, §2.4 tokenizer.text, §2.8 postprocess)

def precision_block(compute: str, weights: str) -> dict:
    act = {"float16": "fp16", "float32": "fp32"}[compute]
    stored = {"float16": "fp16", "float32": "fp32", "int8": "int8 per-channel (decoder linears); fp16 embedding",
              "int8b32": "int8 block-32 (decoder linears); fp16 embedding",
              "int4": "int4 block-32 (decoder linears); fp16 embedding"}[weights]
    return {"weights": stored, "activations": act,
            "components": {"decoder": f"{stored} weights, {act} compute",
                           "head": ("the 12 tied-embedding rows of the yes/no variants in fp32; the last-token gather, "
                                    "the head, the per-class max and the softmax compute in fp32"),
                           "io": "input_ids, attention_mask int32; probs float32"}}


def manifest(files: dict[str, dict], report: dict, platform_name: str, compute: str = COMPUTE_DTYPE,
             weights: str = WEIGHTS) -> dict:
    return {
        "schema": 1,
        "id": ID,
        "task": "text-classification",
        "driver": "yes-no-classifier@1",
        "variant": {"platform": platform_name},
        "assets": {"main": {"subtree": "", "files": [{"path": p, **meta} for p, meta in sorted(files.items())]}},
        "graphs": {
            "classifier": {
                "file": f"main:{ASSET}", "function": FUNCTION,
                "inputs": {"tokens": "input_ids", "attentionMask": "attention_mask"},
                "outputs": {"probs": "probs"},
                "compute": COMPUTE, "expectFrequentReshapes": False,
                "limits": {"staticSequence": sg.SEQUENCE},
            },
        },
        "tokenizer": {
            "source": {"dir": "main:tokenizer"},
            "chatTemplate": None,
            "addSpecialTokens": False,
            "text": {
                "padId": sg.PAD_ID, "padSide": "right", "padTo": sg.SEQUENCE, "truncation": "bodyOnly",
                "maskRule": "realTokens",
                "template": {"prefix": sg.PREFIX, "body": sg.BODY, "suffix": sg.SUFFIX},
                "x-template": {
                    "systemPrompt": sg.SYSTEM_PROMPT,
                    "placeholders": {
                        "instruction": ("<Instruct>: the evaluation context and strictness (and optionally the "
                                        "categories to watch for); the caller's, constant across a product surface"),
                        "query": ("<Query>: one yes/no question about the document, phrased as a question (one "
                                  "policy per call); the caller's"),
                        "document": "<Document>: the content to evaluate (a prompt, a response, or a pair)"},
                    "policy": "none built in: instruction and query are always the caller's",
                },
                "x-notes": ("The row is prefix + body + suffix, each encoded separately with add_special_tokens=false "
                            "and concatenated (the same ids as encoding the whole string, because [SYSTEM_PROMPT], "
                            "[/SYSTEM_PROMPT], [INST] and [/INST] are control tokens; `<s>` in the prefix is text "
                            "that encodes to id 1). This is the checkpoint's chat_template.jinja over [system, user], "
                            "flattened. No stripping. bodyOnly truncation keeps the prefix and suffix and cuts the "
                            f"body's ids to padTo - len(prefix) - len(suffix) from the end (the prefix is 35 ids and "
                            f"the suffix 1, so the body keeps at most {sg.SEQUENCE - 36}); then right-pad with 11 to "
                            "exactly padTo. attentionMask is 1 for every real id and 0 for padding; the graph scores "
                            "the last real position (the suffix [/INST])."),
            },
        },
        "postprocess": {
            "kind": "yesNo",
            "probs": {"positiveIndex": sg.POSITIVE_INDEX, "threshold": sg.THRESHOLD},
            "x-decision": "greaterThan",
            "x-labels": {"0": "no", "1": "yes", "yesIds": list(sg.YES_IDS), "noIds": list(sg.NO_IDS)},
            "x-notes": ("probs [1,2] float32 = softmax([max logit over noIds, max logit over yesIds]) at the last real "
                        "token, in the graph: upstream's score (the model card's helper renormalizes the yes and no "
                        "probabilities, each the best of its decoded forms yes/yes./\"yes\"/'yes' after strip and "
                        "lower-case). probs[1] = P(yes) = P(the Document meets the Query), i.e. P(violation) for a "
                        "policy question such as 'Does this content promote X?'. Flagged when probs[1] > threshold "
                        "(strictly greater, as upstream's helper; the card's benchmarks use 0.5). The label means "
                        "whatever the caller's query asks; the bundle defines no policy."),
        },
        "precision": precision_block(compute, weights),
        "license": "Apache-2.0",
        "x-license": {"files": ["LICENSE", "NOTICE"], "modified": True, "useStatement": USE_STATEMENT},
        "provenance": {
            "source": sg.REPO, "sourceRevision": sg.REVISION, "base": "mistralai/Ministral-3-3B-Base-2512",
            "producer": "aether-kit Forge/shieldstral/export_shieldstral.py",
            "exportScript": report["scriptSHA256"]["export_shieldstral.py"],
            "refs": {"tokenizer.text.template": (f"{sg.REPO}@{sg.REVISION} chat_template.jinja (flattened) and "
                                                 "README.md#how-it-works (system prompt, Instruct/Query/Document)"),
                     "postprocess": f"{sg.REPO}@{sg.REVISION} README.md unsafe_score helper",
                     "tokenizer.source": f"{sg.REPO}@{sg.REVISION} " + ", ".join(TOKENIZER_FILES),
                     "forgeLockSHA256": report["forgeLockSHA256"], "exportReport": "export-report.json"},
        },
    }


def sample_manifest(platform_name: str) -> dict:
    files = {f"{ASSET}/{name}": {"bytes": 1, "sha256": "0" * 64} for name in ("main.hash", "main.mlirb",
                                                                              "metadata.json")}
    files.update({f"tokenizer/{name}": {"bytes": 1, "sha256": "0" * 64} for name in TOKENIZER_FILES})
    files.update({name: {"bytes": 1, "sha256": "0" * 64} for name in ("LICENSE", "NOTICE", "README.md")})
    report = {"scriptSHA256": {"export_shieldstral.py": "0" * 64}, "forgeLockSHA256": "0" * 64}
    return manifest(files, report, platform_name)


def recipe_text(compute: str, weights: str) -> str:
    return {("float16", "float16"): "float16 weights and compute",
            ("float32", "float16"): "float16-stored weights, float32 compute",
            ("float32", "float32"): "float32 weights and compute",
            ("float16", "int8"): "int8 per-channel decoder weights, float16 compute",
            ("float16", "int8b32"): "int8 block-32 decoder weights, float16 compute",
            ("float16", "int4"): "int4 block-32 decoder weights, float16 compute"}[(compute, weights)]


def notice(compute: str, weights: str) -> str:
    return "\n".join([
        "Shieldstral 1.0 3B (Aether-Kit Core AI export)",
        "Copyright Mistral AI. Licensed under the Apache License, Version 2.0 (LICENSE).",
        "",
        f"This bundle is a modified version of {sg.REPO} at revision {sg.REVISION},",
        "by Mistral AI, which the model card licenses under the Apache License 2.0 and qualifies:",
        f"\"{USE_STATEMENT}\"",
        "",
        "Modifications (Apache-2.0 Section 4(b)): Aether-Kit's Forge exporter kept the text decoder only (the Pixtral",
        "vision tower and projector are not included), exported it as one static Core AI graph with a fixed "
        f"{sg.SEQUENCE}-token",
        "input, the causal and padding mask built in the graph, and the scoring step of the model card's helper in the",
        "graph (the last real token's logits for the yes/no token variants, the best per class, a softmax over the",
        f"two); {recipe_text(compute, weights)}. The tokenizer files are upstream's, unmodified.",
        "It is not an official release of Mistral AI and is not affiliated with or endorsed by Mistral AI.",
        "",
        "The bundle carries no safety policy: the instruction and the question are supplied by the application.",
        "",
    ])


def readme(platform_names: list[str], compute: str, weights: str) -> str:
    return "\n".join([
        "---",
        "license: apache-2.0",
        f"base_model: {sg.REPO}",
        "pipeline_tag: text-classification",
        "tags: [aether-kit, core-ai, safety, moderation, classifier, shieldstral]",
        "---",
        "",
        "# Shieldstral 1.0 3B for Aether-Kit (Core AI)",
        "",
        f"[{sg.REPO}](https://huggingface.co/{sg.REPO}) at revision `{sg.REVISION[:8]}`, exported by",
        "[Aether-Kit](https://github.com/bubstackdev/aether-kit)'s Forge exporter (`Forge/shieldstral/`) as one",
        f"static Core AI graph, text only: `input_ids` and `attention_mask` (int32, {sg.SEQUENCE} tokens, right-padded "
        f"with {sg.PAD_ID})",
        "to `probs` = [P(no), P(yes)] at the last prompt token. The model card's scoring step is in the graph. "
        f"{recipe_text(compute, weights).capitalize()}.",
        "",
        "Shieldstral judges a document against a policy you write. The bundle has no policy of its own: an app",
        "supplies an instruction (context and strictness) and one yes/no question per call, for example",
        "`<Instruct>: You are moderating prompts for an image-generation app. Apply a moderate standard.` and",
        "`<Query>: Is this prompt unsafe?`. The row is",
        "",
        "```",
        "<s>[SYSTEM_PROMPT]" + sg.SYSTEM_PROMPT + "[/SYSTEM_PROMPT][INST]",
        "<Instruct>: {instruction}\\n\\n<Query>: {query}\\n\\n<Document>: {document}",
        "[/INST]",
        "```",
        "",
        "encoded without added special tokens, the body truncated if needed, then padded. `probs[1]` is P(yes); a",
        "document is flagged when it is above the threshold (0.5 by default). Tune any threshold against this",
        "bundle.",
        "",
        f"Load it with Aether-Kit's `AetherEmbeddings` (`yes-no-classifier@1`, {' and '.join(platform_names)}); the",
        "manifest is `.aether/aether.json`. Gated against the upstream implementation on CPU in float32 (see",
        "`export-report.json`).",
        "",
        "Licensed under the Apache License 2.0 (`LICENSE`). Upstream's card adds: \"" + USE_STATEMENT + "\" A",
        "modified version, not an official Mistral AI release (see `NOTICE`).",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="new bundle directory")
    parser.add_argument("--platform", choices=["macos", "ios"], default="macos", help="the manifest's variant")
    parser.add_argument("--sample-manifest", action="store_true", help="print the manifest for placeholder files")
    parser.add_argument("--snapshot", type=Path, default=sg.DEFAULT_SNAPSHOT)
    parser.add_argument("--compute", choices=["float16", "float32"], default=COMPUTE_DTYPE)
    parser.add_argument("--weights", choices=["float16", "float32", "int8", "int8b32", "int4"], default=WEIGHTS,
                        help="weight storage; the non-default recipes are for the precision measurement")
    parser.add_argument("--creation-date", default=CREATION_DATE, help="ISO-8601; the package metadata's date")
    args = parser.parse_args()
    if args.sample_manifest:
        print(json.dumps(sample_manifest(args.platform), indent=2))
        return
    if args.output is None:
        parser.error("--output is required")
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    recipe = recipe_text(args.compute, args.weights)
    creation_date = int(dt.datetime.fromisoformat(args.creation_date).timestamp())
    started = time.monotonic()
    args.output.mkdir(parents=True)
    log = Log(args.output / "forge-log.jsonl")
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    lock = HERE.parent / "uv.lock"
    report = {
        "schema": 1, "id": ID, "command": sys.argv,
        "exportedAt": dt.datetime.now(dt.timezone.utc).isoformat(), "creationDate": args.creation_date,
        "host": {"os": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
                 "machine": platform.machine(), "python": platform.python_version()},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "coreai-torch", "coreai-core", "coreai-opt",
                                                                "numpy", "safetensors", "transformers", "tokenizers")},
        "forgeLockSHA256": sg.sha256_file(lock), "forgePyprojectSHA256": sg.sha256_file(HERE.parent / "pyproject.toml"),
        "scriptSHA256": {p.name: sg.sha256_file(p) for p in sorted(HERE.glob("*.py"))},
        "source": {"repository": sg.REPO, "revision": sg.REVISION, "snapshot": str(args.snapshot.resolve()),
                   "verified": {}},
        "precision": f"{recipe}; head, softmax and output float32; int32 inputs",
        "precisionDecision": PRECISION_DECISION,
        "labels": {"yesIds": list(sg.YES_IDS), "noIds": list(sg.NO_IDS)},
        "inputShape": [1, sg.SEQUENCE], "compute": COMPUTE, "conversion": {}, "stages": {},
    }
    report_path = args.output / "export-report.json"

    def save() -> None:
        report["peakRSSGB"] = peak_rss_gb()
        report["wallSeconds"] = round(time.monotonic() - started, 1)
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    log.start("verify-sources")
    report["source"]["verified"] = sg.verify_sources(args.snapshot)
    tok = sg.tokenizer(args.snapshot)
    if sg.label_ids(tok) != (sg.YES_IDS, sg.NO_IDS):
        raise SystemExit("the vocabulary's yes/no variants differ from YES_IDS/NO_IDS")
    report["stages"]["verify-sources"] = log.done("verify-sources")
    save()

    log.start("export")
    ids, mask, _ = sg.tokenize(tok, "You are a content moderator.", "Is this content unsafe?",
                               "A watercolor painting of a lighthouse at dawn.")
    example = (torch.from_numpy(ids), torch.from_numpy(mask))
    graph, quant = build_graph(args.snapshot, args.compute, args.weights, example)
    if quant:
        report["quantization"] = quant
    gc.collect()
    report["conversion"]["classifier"] = convert(graph, example, args.output / ASSET, creation_date, recipe)
    del graph
    gc.collect()
    report["stages"]["export"] = log.done("export")
    save()

    log.start("package")
    (args.output / "tokenizer").mkdir()
    for name in TOKENIZER_FILES:
        shutil.copyfile(args.snapshot / name, args.output / "tokenizer" / name)
    shutil.copyfile(sg.TERMS_DIR / "LICENSE-2.0.txt", args.output / "LICENSE")
    (args.output / "NOTICE").write_text(notice(args.compute, args.weights))
    (args.output / "README.md").write_text(readme(["macOS"], args.compute, args.weights))
    files = {str(p.relative_to(args.output)): {"bytes": p.stat().st_size, "sha256": sg.sha256_file(p)}
             for p in tree_files(args.output) if str(p.relative_to(args.output)) not in NOT_BUNDLE}
    report["bundleBytes"] = sum(f["bytes"] for f in files.values())
    report["assetBytes"] = {"classifier": tree_bytes(args.output / ASSET),
                            "tokenizer": tree_bytes(args.output / "tokenizer")}
    report["stages"]["package"] = log.done("package")
    save()
    (args.output / ".aether").mkdir()
    manifest_path = args.output / ".aether/aether.json"
    manifest_path.write_text(json.dumps(manifest(files, report, args.platform, args.compute, args.weights),
                                        indent=2, ensure_ascii=False) + "\n")
    report["aetherJSONSHA256"] = sg.sha256_file(manifest_path)
    save()
    sums = [f"{sg.sha256_file(p)}  {p.relative_to(args.output)}" for p in tree_files(args.output)
            if p.name not in ("SHA256SUMS", "forge-log.jsonl")]
    (args.output / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(json.dumps({"conversion": report["conversion"], "bundleBytes": report["bundleBytes"],
                      "peakRSSGB": report["peakRSSGB"], "wallSeconds": report["wallSeconds"]}, indent=1))


if __name__ == "__main__":
    main()

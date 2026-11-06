#!/usr/bin/env python3
"""Export GLiNER2.5-Decide (fastino/GLiNER2.5-Decide, Apache-2.0) as an Aether-Kit `gliner-classifier@1` bundle.

    uv sync --frozen --project Forge
    Forge/.venv/bin/python -I Forge/gliner/fetch_upstream.py            # pinned checkpoint and gliner2 wheel, once
    Forge/.venv/bin/python -I Forge/gliner/export_gliner.py --output <new bundle dir>
    Forge/.venv/bin/python -I Forge/gliner/export_gliner.py --sample-manifest

Nothing is downloaded: every file read is verified against `source-pins.json`. Conversion runs on the CPU. The output
directory is laid out as an installed revision root:

    OUTPUT/
      .aether/aether.json          the Aether manifest (MANIFEST.md §1.3, §2; driver `gliner-classifier@1`)
      gliner2.5_decide.aimodel     functions s256 and s512 (one weight set): input_ids, attention_mask int32 [1,S],
                                   label_idx int32 [1,32] -> logits float32 [1,32]; fp16 weights and compute
      tokenizer/                   upstream's tokenizer.json, tokenizer_config.json, special_tokens_map.json
      LICENSE                      Apache License 2.0
      NOTICE                       origin, base model (DeBERTa-v3-large, MIT) and what was converted
      README.md                    the Hugging Face model card
      SHA256SUMS                   every bundle file except itself and the log
      export-report.json           provenance, versions, timings, peak RSS
      forge-log.jsonl              one event per stage

One Core AI JIT package serves macOS and iOS (no per-architecture AOT): `--platform` only sets the manifest's
`variant`. The export is reproducible as the other Forge exporters are (seeded generated names, fixed creation date,
deterministic Swift hashing): bundle files are byte-identical across exports except the graph's `main.mlirb` and
`main.hash`, where coreai-core names each weight blob with a per-execution seeded `llvm::hash_value` (and so the
manifest's file list and SHA256SUMS, which hash them). Parity: `reference_gliner.py` (gliner2's own classify_text, CPU float32) records the
references; `check_gliner.py` runs both functions through Core AI on the GPU and gates them (probabilities within
0.01, same argmax and decision).
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
import importlib  # noqa: E402
import importlib.metadata  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import resource  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gliner_graphs as gg  # noqa: E402

ID = "gliner2.5-decide"
HF_REPO = "aether-models/aether-kit-gliner2.5-decide"
ASSET = "gliner2.5_decide.aimodel"
FUNCTIONS = {s: f"s{s}" for s in gg.SEQUENCES}          # one function per sequence bucket, sharing the weights
INPUTS, OUTPUTS = ["input_ids", "attention_mask", "label_idx"], ["logits"]
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
NOT_BUNDLE = {".aether/aether.json", "SHA256SUMS", "export-report.json", "forge-log.jsonl"}
# Measured default (MANIFEST §0): GPU for every graph class; on the Neural Engine this graph was 6x slower (zoo survey).
COMPUTE = "gpu"
# Measured (selftest_gliner.py on the CPU, check_gliner.py on the GPU): float16 weights and compute stay well inside
# the bars (probabilities within 0.01, same decisions) at half the bytes of float32.
DTYPE = "float16"
CREATION_DATE = "2026-11-06T00:00:00+00:00"     # the pin date; fixed so the package bytes are reproducible
WORD_PATTERN = (r"(?:https?://[^\s]+|www\.[^\s]+)|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}|@[a-z0-9_]+"
                r"|\w+(?:[-_]\w+)*|\S")     # gliner2's WhitespaceTokenSplitter, compacted (selftest checks it)


class Log:
    def __init__(self, path: Path) -> None:
        self.path, self.started = path, {}

    def event(self, stage: str, message: str, **fields) -> None:
        record = {"message": message, "stage": stage, "t": round(time.time(), 3), **fields}
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"[gliner:{stage}] {message}" + (f" {fields}" if fields else ""), file=sys.stderr, flush=True)

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


@contextlib.contextmanager
def deterministic_conversion(creation_date: int):
    """Makes the converter's output reproducible, undone after: Core AI suffixes generated function names with
    `secrets.choice` letters and coreai-torch with `uuid4()` hex, and `save_asset` stamps `time.time()` as the creation
    date. Inside, both name sources draw from one seeded generator and the stamp is `creation_date`."""
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


def convert(upstream, dtype: torch.dtype, sequences: tuple[int, ...], path: Path, creation_date: int) -> dict:
    """One Core AI asset holding a function per sequence bucket. The functions share every weight (the same
    nn.Parameters feed each exported program, so the converter stores them once)."""
    import coreai_torch as ct
    from coreai.runtime import AIModelAssetMetadata

    base = gg.ClassifierGraph(upstream, sequences[0], dtype)
    t0 = time.monotonic()
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    with deterministic_conversion(creation_date), gg.traceable_deberta(), torch.no_grad():
        for sequence in sequences:
            graph = gg.ClassifierGraph(upstream, sequence, dtype, share=base)
            example = tuple(torch.from_numpy(x) for x in gg.graph_inputs(
                torch.arange(10, 30, dtype=torch.int32).numpy(), torch.tensor([2, 4], dtype=torch.int32).numpy(),
                sequence))
            ep = torch.export.export(graph, example, strict=False).run_decompositions(ct.get_decomp_table())
            converter.add_exported_program(ep, input_names=INPUTS, output_names=OUTPUTS, state_names=[],
                                           entrypoint_name=FUNCTIONS[sequence])
            del ep
        t1 = time.monotonic()
        program = converter.to_coreai()
        del converter
        program.optimize()
        metadata = AIModelAssetMetadata()
        metadata.author = "Fastino (GLiNER2.5-Decide), on Microsoft DeBERTa-v3-large; Core AI export: Aether-Kit Forge"
        metadata.license = "Apache-2.0"
        metadata.model_description = (
            f"GLiNER2.5-Decide ({gg.REPO}@{gg.REVISION[:12]}) classification path: input_ids, attention_mask int32 "
            f"[1,S] (right-padded with 0), label_idx int32 [1,{gg.MAX_LABELS}] -> logits float32 [1,{gg.MAX_LABELS}], "
            f"one function per S in {list(sequences)}.")
        metadata.creation_date = creation_date
        program.save_asset(path, metadata)
    return {"exportSeconds": round(t1 - t0, 1), "convertSeconds": round(time.monotonic() - t1, 1),
            "bytes": tree_bytes(path), "inputs": INPUTS, "outputs": OUTPUTS,
            "functions": [FUNCTIONS[s] for s in sequences]}


# MARK: the manifest (MANIFEST.md §1.3 `gliner-classifier@1`, §2.3 buckets, §2.4 tokenizer, §2.8 postprocess)

def precision_block(dtype_name: str) -> dict:
    w = {"float16": "fp16", "float32": "fp32"}[dtype_name]
    return {"weights": w, "activations": w,
            "components": {"encoder": f"{w} (DeBERTa-v3-large, relative-position table int64 constant)",
                           "head": f"{w} classifier (Linear 1024->2048, ReLU, Linear 2048->1)",
                           "io": "input_ids, attention_mask, label_idx int32; logits float32"}}


def manifest(files: dict[str, dict], report: dict, platform_name: str, markers: dict,
             dtype_name: str = DTYPE) -> dict:
    return {
        "schema": 1,
        "id": ID,
        "task": "zero-shot-text-classification",
        "driver": "gliner-classifier@1",
        "variant": {"platform": platform_name},
        "assets": {"main": {"subtree": "", "files": [{"path": p, **meta} for p, meta in sorted(files.items())]}},
        "graphs": {
            "classifier": {
                "file": f"main:{ASSET}", "function": FUNCTIONS[gg.SEQUENCES[0]],
                "buckets": {"axis": "sequence", "rule": "smallestFitting",
                            "entries": [{"value": s, "function": f} for s, f in sorted(FUNCTIONS.items())]},
                "inputs": {"tokens": "input_ids", "attentionMask": "attention_mask", "labelPositions": "label_idx"},
                "outputs": {"labelLogits": "logits"},
                "compute": COMPUTE, "expectFrequentReshapes": False,
                "limits": {"staticSequence": list(gg.SEQUENCES)},
                "x-notes": (f"label_idx [1,{gg.MAX_LABELS}] holds the [L] marker positions of every task, "
                            "concatenated in task order (at most 32 labels per call); unused slots repeat the first "
                            "position and their logits are ignored. logits[i] is the logit of label_idx[i]'s label. "
                            "A request runs at the smallest bucket that holds its ids (s256, else s512); longer "
                            "input is truncated at a word boundary from the end of the text."),
            },
        },
        "tokenizer": {
            "source": {"dir": "main:tokenizer"},
            "chatTemplate": None,
            "addSpecialTokens": False,
            "class": "XLMRobertaTokenizer",
            "markerIds": markers,
            "text": {
                "padId": gg.PAD_ID, "padSide": "right", "truncation": "word", "maskRule": "realTokens",
                "x-notes": ("No padTo: ids are padded with 0 on the right to the bucket the request runs at "
                            "(graphs.classifier.buckets), not to one fixed length. attentionMask is 1 for every "
                            "real id and 0 for padding. No [CLS]/[SEP]: every piece below is tokenized on its own "
                            "without special tokens and the ids concatenated."),
            },
            "x-layout": ("gliner2 2.0.0's classification layout, one group per task in request order, groups joined "
                         "by [SEP_STRUCT], then [SEP_TEXT] and the text's words: `( [P] <task>[: <prompt>]"
                         "[ [DESCRIPTION] <label>: <description>]* ( [L] <label> [L] <label> ... ) ) [SEP_STRUCT] "
                         "( ... ) [SEP_TEXT] <word> <word> ...`. Each schema piece ('(', '[P]', the prompt string, "
                         "'[L]', a label, ')') is one tokenizer call; labels and prompts keep their case. "
                         "labelPositions are the positions of the [L] ids."),
            "x-text": {"appendPeriod": "append '.' unless the text ends with '.', '!' or '?' (empty text -> '.')",
                       "wordPattern": WORD_PATTERN, "wordPatternFlags": "IGNORECASE",
                       "lowercase": "each word (the text only, never schema strings)",
                       "pieces": "every word is tokenized on its own, as a schema piece is"},
            "x-notes": ("The tokenizer files are upstream's (DeBERTa-v3 SentencePiece Unigram, tokenizer_class "
                        "DebertaV2Tokenizer); `class` names XLMRobertaTokenizer, the same Unigram model under a class "
                        "swift-transformers loads. markerIds are the ids gliner2's processor registers (128000+)."),
        },
        "postprocess": {
            "kind": "gliner",
            "gliner": {"threshold": gg.THRESHOLD},
            "x-notes": ("Per task, its slice of logits: single-label, softmax over the slice and argmax decides; "
                        "multi-label, a sigmoid per label and every label at or above threshold (none -> the "
                        "argmax). gliner2's rule with temperature 1."),
        },
        "precision": precision_block(dtype_name),
        "license": "Apache-2.0",
        "x-license": {"files": ["LICENSE", "NOTICE"],
                      "base": "microsoft/deberta-v3-large (MIT), recorded in NOTICE"},
        "provenance": {
            "source": gg.REPO, "sourceRevision": gg.REVISION, "base": "fastino/gliner2-large-v1",
            "producer": "aether-kit Forge/gliner/export_gliner.py",
            "exportScript": report["scriptSHA256"]["export_gliner.py"],
            "refs": {"tokenizer.markerIds": (f"{gg.REPO}@{gg.REVISION} tokenizer.json added tokens, as gliner2 "
                                             f"{gg.CODE['version']} SchemaTransformer registers them; equal to "
                                             f"{gg.ZOO['repository']}@{gg.ZOO['revision'][:12]} "
                                             "macos/classifier.json#/marker_ids"),
                     "tokenizer.x-layout": f"gliner2 {gg.CODE['version']} processor.py (PyPI wheel)",
                     "tokenizer.source": f"{gg.REPO}@{gg.REVISION} " + ", ".join(TOKENIZER_FILES),
                     "upstreamCode": f"{gg.CODE['repository']} {gg.CODE['tag']} ({gg.CODE['commit'][:12]})",
                     "forgeLockSHA256": report["forgeLockSHA256"], "exportReport": "export-report.json"},
        },
    }


def sample_manifest(platform_name: str) -> dict:
    files = {f"{ASSET}/{name}": {"bytes": 1, "sha256": "0" * 64} for name in ("main.hash", "main.mlirb",
                                                                              "metadata.json")}
    files.update({f"tokenizer/{name}": {"bytes": 1, "sha256": "0" * 64} for name in TOKENIZER_FILES})
    files.update({name: {"bytes": 1, "sha256": "0" * 64} for name in ("LICENSE", "NOTICE", "README.md")})
    report = {"scriptSHA256": {"export_gliner.py": "0" * 64}, "forgeLockSHA256": "0" * 64}
    markers = {m: 128000 + i for i, m in enumerate(gg.MARKERS)}
    return manifest(files, report, platform_name, markers, DTYPE)


def notice(dtype_name: str = DTYPE) -> str:
    stored = "float16" if dtype_name == "float16" else "float32"
    return "\n".join([
        "GLiNER2.5-Decide (Aether-Kit Core AI export)",
        "",
        "Origin",
        f"  Model:     {gg.REPO} (https://huggingface.co/{gg.REPO})",
        f"  Revision:  {gg.REVISION}",
        f"  Weights:   model.safetensors, sha256 {gg.PINS['files']['model.safetensors']}",
        "  Provider:  Fastino",
        "  License:   Apache License 2.0 (declared on the model card; the source repository has no LICENSE file)",
        "  Base:      fastino/gliner2-large-v1, on microsoft/deberta-v3-large (MIT License, Copyright (c) Microsoft",
        "             Corporation)",
        "",
        "Software used",
        f"  gliner2 {gg.CODE['version']} ({gg.CODE['repository']}, Apache License 2.0): the model code and the CPU float32",
        "  reference the export was checked against.",
        "  transformers (Apache License 2.0): DebertaV2Model, the encoder.",
        "",
        "Modifications",
        "  Aether-Kit's Forge exporter (Forge/gliner/) exported the classification path (the DeBERTa-v3 encoder, the",
        "  gather of the [L] marker rows and the classifier head) as one static Core AI graph with a function per",
        "  sequence length (256 and 512) and DeBERTa's relative-position bucket table computed at export.",
        f"  The weights are stored in {stored}" + (" (rounded from the released float32)." if stored == "float16"
                                                  else " (unchanged)."),
        "  The span, count and count-embedding heads, which classification does not use, are not in the graph.",
        "  The tokenizer files are upstream's, unmodified. It is not an official release of Fastino.",
        "",
        "This distribution is provided under the Apache License 2.0 (LICENSE). Keep this NOTICE with any part of it",
        "you redistribute.",
        "",
    ])


def readme(platform_names: list[str], dtype_name: str = DTYPE) -> str:
    w = "float16" if dtype_name == "float16" else "float32"
    return "\n".join([
        "---",
        "license: apache-2.0",
        "base_model:",
        f"  - {gg.REPO}",
        "  - microsoft/deberta-v3-large",
        "pipeline_tag: zero-shot-classification",
        "language: [en]",
        "tags: [aether-kit, core-ai, gliner, gliner2, text-classification, zero-shot-classification]",
        "---",
        "",
        "# GLiNER2.5-Decide for Aether-Kit (Core AI)",
        "",
        f"[{gg.REPO}](https://huggingface.co/{gg.REPO}) at revision `{gg.REVISION[:8]}`, exported by",
        "[Aether-Kit](https://github.com/bubstackdev/aether-kit)'s Forge exporter (`Forge/gliner/`) as one Core AI",
        "graph with two static functions, `s256` and `s512`, sharing one weight set: the laid-out token ids, the",
        "attention mask and the positions of up to 32 `[L]` label markers to one logit per label. The DeBERTa-v3-large",
        f"encoder and the label head compute in {w}. The host lays the tasks out ahead of the text as gliner2 does,",
        "runs the smallest function that holds the ids, and decides each task with a softmax (one label) or a sigmoid",
        f"per label and a threshold (several). Load it with Aether-Kit's `AetherEmbeddings` (`gliner-classifier@1`,",
        f"{' and '.join(platform_names)}); the manifest is `.aether/aether.json`. Gated against gliner2's own",
        "`classify_text` on CPU in float32 (see `export-report.json`).",
        "",
        "Apache-2.0, the source model's license (`LICENSE`); its base, DeBERTa-v3-large, is MIT. `NOTICE` records the",
        "origin and the modifications. Not an official Fastino release.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="new bundle directory")
    parser.add_argument("--platform", choices=["macos", "ios"], default="macos", help="the manifest's variant")
    parser.add_argument("--sample-manifest", action="store_true", help="print the manifest for placeholder files")
    parser.add_argument("--snapshot", type=Path, default=gg.DEFAULT_SNAPSHOT)
    parser.add_argument("--code", type=Path, default=gg.DEFAULT_CODE, help="the unpacked gliner2 wheel")
    parser.add_argument("--dtype", choices=["float16", "float32"], default=DTYPE,
                        help="weight and compute dtype; float32 only for the precision measurement")
    parser.add_argument("--creation-date", default=CREATION_DATE, help="ISO-8601; the package metadata's date")
    args = parser.parse_args()
    if args.sample_manifest:
        print(json.dumps(sample_manifest(args.platform), indent=2))
        return
    if args.output is None:
        parser.error("--output is required")
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
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
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "coreai-torch", "coreai-core", "numpy",
                                                                "safetensors", "transformers", "tokenizers")}
        | {"gliner2": f"{gg.CODE['version']} (unpacked wheel, {gg.CODE['wheelSHA256'][:12]})"},
        "forgeLockSHA256": gg.sha256_file(lock), "forgePyprojectSHA256": gg.sha256_file(HERE.parent / "pyproject.toml"),
        "scriptSHA256": {p.name: gg.sha256_file(p) for p in sorted(HERE.glob("*.py"))},
        "source": {"repository": gg.REPO, "revision": gg.REVISION, "snapshot": str(args.snapshot.resolve()),
                   "verified": {}},
        "precision": f"{args.dtype} weights and compute; int32 inputs, float32 logits; no quantization",
        "precisionDecision": ("float16 weights and compute: on the CPU (selftest_gliner.py) and on the GPU "
                              "(check_gliner.py) the float16 graph stays well inside the bars against the float32 "
                              "reference (probabilities within 0.01, same argmax and decision) for half the bytes"),
        "sequences": list(gg.SEQUENCES), "maxLabels": gg.MAX_LABELS, "compute": COMPUTE, "conversion": {},
        "stages": {},
    }
    report_path = args.output / "export-report.json"

    def save() -> None:
        report["peakRSSGB"] = peak_rss_gb()
        report["wallSeconds"] = round(time.monotonic() - started, 1)
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    log.start("verify-sources")
    report["source"]["verified"] = gg.verify_sources(args.snapshot, args.code)
    upstream = gg.load_upstream(args.snapshot, args.code)
    markers = gg.marker_ids(upstream)
    report["stages"]["verify-sources"] = log.done("verify-sources")
    save()

    log.start("export")
    report["conversion"]["classifier"] = convert(upstream, gg.DTYPES[args.dtype], gg.SEQUENCES,
                                                 args.output / ASSET, creation_date)
    del upstream
    report["stages"]["export"] = log.done("export")
    save()

    log.start("package")
    (args.output / "tokenizer").mkdir()
    for name in TOKENIZER_FILES:
        shutil.copyfile(args.snapshot / name, args.output / "tokenizer" / name)
    shutil.copyfile(args.code / "gliner2-2.0.0.dist-info/licenses/LICENSE", args.output / "LICENSE")
    (args.output / "NOTICE").write_text(notice(args.dtype))
    (args.output / "README.md").write_text(readme(["macOS"], args.dtype))
    files = {str(p.relative_to(args.output)): {"bytes": p.stat().st_size, "sha256": gg.sha256_file(p)}
             for p in tree_files(args.output) if str(p.relative_to(args.output)) not in NOT_BUNDLE}
    report["bundleBytes"] = sum(f["bytes"] for f in files.values())
    report["assetBytes"] = {"classifier": tree_bytes(args.output / ASSET),
                            "tokenizer": tree_bytes(args.output / "tokenizer")}
    report["stages"]["package"] = log.done("package")
    save()
    (args.output / ".aether").mkdir()
    manifest_path = args.output / ".aether/aether.json"
    manifest_path.write_text(json.dumps(manifest(files, report, args.platform, markers, args.dtype), indent=2) + "\n")
    report["aetherJSONSHA256"] = gg.sha256_file(manifest_path)
    save()
    sums = [f"{gg.sha256_file(p)}  {p.relative_to(args.output)}" for p in tree_files(args.output)
            if p.name not in ("SHA256SUMS", "forge-log.jsonl")]
    (args.output / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(json.dumps({"conversion": report["conversion"], "bundleBytes": report["bundleBytes"],
                      "peakRSSGB": report["peakRSSGB"], "wallSeconds": report["wallSeconds"]}, indent=1))


if __name__ == "__main__":
    main()

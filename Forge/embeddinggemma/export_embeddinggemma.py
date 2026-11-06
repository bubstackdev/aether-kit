#!/usr/bin/env python3
"""Export EmbeddingGemma 300M (google/embeddinggemma-300m, Gemma Terms of Use) as an Aether-Kit `text-embedder@1`
bundle.

    uv sync --frozen --project Forge
    Forge/.venv/bin/python -I Forge/embeddinggemma/fetch_upstream.py      # pinned (gated) checkpoint, once
    Forge/.venv/bin/python -I Forge/embeddinggemma/export_embeddinggemma.py --output <new bundle dir>
    Forge/.venv/bin/python -I Forge/embeddinggemma/export_embeddinggemma.py --sample-manifest

Nothing is downloaded: every file read is verified against `source-pins.json`. Conversion runs on the CPU. The output
directory is laid out as an installed revision root:

    OUTPUT/
      .aether/aether.json                     the Aether manifest (MANIFEST.md §1.3, §2; driver `text-embedder@1`)
      embeddinggemma_300m.aimodel             input_ids, attention_mask int32 [1,256] -> embedding [1,768] fp32
                                              (fp16-stored weights, fp32 compute)
      tokenizer/                              upstream's tokenizer.json, tokenizer_config.json, special_tokens_map.json
      LICENSE                                 the Gemma Terms of Use (a copy of the Agreement, Section 3.1)
      GEMMA-PROHIBITED-USE-POLICY.txt         the use restrictions it incorporates (Section 3.2)
      NOTICE                                  Google's notice line, the use-restriction and modification notices
      README.md                               the Hugging Face model card
      SHA256SUMS                              every bundle file except itself and the log
      export-report.json                      provenance, versions, timings, peak RSS
      forge-log.jsonl                         one event per stage

One Core AI JIT package serves macOS and iOS (no per-architecture AOT): `--platform` only sets the manifest's
`variant`. The export is reproducible: generated function names come from a seeded generator, the package's creation
date is fixed (`--creation-date`, default the pin date) and Swift hashing is deterministic, so two exports give
byte-identical bundle files except the graph's `main.mlirb`/`main.hash`, where coreai-core names each weight blob with
an `llvm::hash_value` seeded per execution (coreai_torch.debugging.graph_match documents it); the two graphs compute
bit-identical embeddings (compare check_embeddinggemma.py's dumped outputs). Parity: `reference_embeddinggemma.py`
(transformers' Gemma 3 encoder plus the sentence-transformers head, CPU float32) records the references;
`check_embeddinggemma.py` runs the exported graph through Core AI on the GPU and gates it (cosine >= 0.999).
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
import embeddinggemma_graphs as eg  # noqa: E402

ID = "embeddinggemma-300m"
HF_REPO = "aether-models/aether-kit-embeddinggemma-300m"
ASSET = "embeddinggemma_300m.aimodel"
FUNCTION = "main"
INPUTS, OUTPUTS = ["input_ids", "attention_mask"], ["embedding"]
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
POLICY = "GEMMA-PROHIBITED-USE-POLICY.txt"
NOT_BUNDLE = {".aether/aether.json", "SHA256SUMS", "export-report.json", "forge-log.jsonl"}
# Measured default (MANIFEST §0): GPU for every graph class; the graph is fully static.
COMPUTE = "gpu"
# Measured (selftest_embeddinggemma.py, check_embeddinggemma.py): float16 activations overflow to NaN on every case (the
# residual stream reaches ~1.3e5; upstream's card also says EmbeddingGemma does not support float16 activations), so
# compute is float32. Weights are stored float16 and widened in the graph: half the download, ~1e-7 in cosine.
COMPUTE_DTYPE = "float32"
WEIGHTS = "float16"
CREATION_DATE = "2026-11-06T00:00:00+00:00"     # the pin date; fixed so the package bytes are reproducible
NOTICE_LINE = "Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms"


class Log:
    def __init__(self, path: Path) -> None:
        self.path, self.started = path, {}

    def event(self, stage: str, message: str, **fields) -> None:
        record = {"message": message, "stage": stage, "t": round(time.time(), 3), **fields}
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"[embeddinggemma:{stage}] {message}" + (f" {fields}" if fields else ""), file=sys.stderr, flush=True)

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
    date. Inside, both name sources draw from one seeded generator (same alphabet and length, so the same uniqueness
    odds, now repeatable) and the stamp is `creation_date`."""
    # Modules by import path (the dialect packages re-export functions named `graph` and `func`).
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


def convert(module, example: tuple, path: Path, creation_date: int) -> dict:
    import coreai_torch as ct
    from coreai.runtime import AIModelAssetMetadata

    with deterministic_conversion(creation_date):
        return _convert(ct, AIModelAssetMetadata, module, example, path, creation_date)


def _convert(ct, AIModelAssetMetadata, module, example: tuple, path: Path, creation_date: int) -> dict:  # noqa: N803
    t0 = time.monotonic()
    with torch.no_grad():
        ep = torch.export.export(module, example, strict=False)
        ep = ep.run_decompositions(ct.get_decomp_table())
    t1 = time.monotonic()
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    converter.add_exported_program(ep, input_names=INPUTS, output_names=OUTPUTS, state_names=[],
                                   entrypoint_name=FUNCTION)
    del ep
    program = converter.to_coreai()
    del converter
    program.optimize()
    metadata = AIModelAssetMetadata()
    metadata.author = "Google DeepMind (EmbeddingGemma); Core AI export: Aether-Kit Forge"
    metadata.license = "Gemma Terms of Use (ai.google.dev/gemma/terms)"
    metadata.model_description = (f"EmbeddingGemma 300M ({eg.REPO}@{eg.REVISION[:12]}): input_ids, attention_mask "
                                  f"int32 [1,{eg.SEQUENCE}] (right-padded with 0) -> L2-normalized embedding "
                                  f"[1,{eg.DIMENSION}] (mean pooling, dense, normalize in the graph).")
    metadata.creation_date = creation_date
    program.save_asset(path, metadata)
    return {"exportSeconds": round(t1 - t0, 1), "convertSeconds": round(time.monotonic() - t1, 1),
            "bytes": tree_bytes(path), "inputs": INPUTS, "outputs": OUTPUTS, "function": FUNCTION}


# MARK: the manifest (MANIFEST.md §1.3 `text-embedder@1`, §2.4 tokenizer.text, §2.8 postprocess)

def manifest(files: dict[str, dict], report: dict, platform_name: str, prompts: dict, weights: str = WEIGHTS) -> dict:
    stored = {"float16": "fp16", "float32": "fp32"}[weights]
    return {
        "schema": 1,
        "id": ID,
        "task": "text-embedding",
        "driver": "text-embedder@1",
        "variant": {"platform": platform_name},
        "assets": {"main": {"subtree": "", "files": [{"path": p, **meta} for p, meta in sorted(files.items())]}},
        "graphs": {
            "encoder": {
                "file": f"main:{ASSET}", "function": FUNCTION,
                "inputs": {"tokens": "input_ids", "attentionMask": "attention_mask"},
                "outputs": {"embedding": "embedding"},
                "compute": COMPUTE, "expectFrequentReshapes": False,
                "limits": {"staticSequence": eg.SEQUENCE},
            },
        },
        "tokenizer": {
            "source": {"dir": "main:tokenizer"},
            "chatTemplate": None,
            "addSpecialTokens": True,
            "text": {
                "padId": eg.PAD_ID, "padSide": "right", "padTo": eg.SEQUENCE, "truncation": "right",
                "prompts": prompts,
                "x-notes": ("sentence-transformers semantics: the text is prompt + input, stripped of leading and "
                            "trailing whitespace, then encoded with special tokens (<bos>=2 first, <eos>=1 last). "
                            "Truncation keeps the first 254 content ids and both special tokens (HF "
                            "truncation='longest_first', max_length=256). attentionMask is 1 for every real token "
                            "(prompt included) and 0 for padding; padding is id 0 on the right to exactly 256."),
            },
        },
        "postprocess": {
            "kind": "embedding",
            "embedding": {"normalized": "inGraph", "pooling": "inGraph", "similarity": "cosine"},
            "x-notes": (f"embedding [1,{eg.DIMENSION}] float32 is mean-pooled over attentionMask, projected "
                        "768->3072->768 and L2-normalized in the graph: cosine equals the dot product. Matryoshka: "
                        "the first 512, 256 or 128 dimensions, re-normalized, are valid smaller embeddings."),
        },
        "precision": {"weights": stored if stored == "fp32" else "fp16 (widened to fp32 in the graph)",
                      "activations": "fp32",
                      "components": {"encoder": (f"{stored} weights, fp32 compute (float16 activations overflow; "
                                                 "upstream: EmbeddingGemma does not support float16 activations)"),
                                     "head": f"{stored} dense weights; pooling, dense and normalize compute in fp32",
                                     "io": "input_ids, attention_mask int32; embedding float32"}},
        "license": "gemma",
        "x-license": {"files": ["LICENSE", POLICY, "NOTICE"], "modified": True,
                      "useRestrictions": ("Gemma Terms of Use Section 3.2: no use for the restricted uses in the "
                                          "Gemma Prohibited Use Policy or in violation of law; pass these "
                                          "restrictions on as an enforceable provision when redistributing.")},
        "provenance": {
            "source": eg.REPO, "sourceRevision": eg.REVISION,
            "producer": "aether-kit Forge/embeddinggemma/export_embeddinggemma.py",
            "exportScript": report["scriptSHA256"]["export_embeddinggemma.py"],
            "refs": {"tokenizer.text.prompts": (f"{eg.REPO}@{eg.REVISION} config_sentence_transformers.json"
                                                "#/prompts/query, #/prompts/document"),
                     "tokenizer.source": f"{eg.REPO}@{eg.REVISION} " + ", ".join(TOKENIZER_FILES),
                     "forgeLockSHA256": report["forgeLockSHA256"], "exportReport": "export-report.json"},
        },
    }


SAMPLE_PROMPTS = {"query": "task: search result | query: ", "document": "title: none | text: "}


def sample_manifest(platform_name: str) -> dict:
    files = {f"{ASSET}/{name}": {"bytes": 1, "sha256": "0" * 64} for name in ("main.hash", "main.mlirb",
                                                                              "metadata.json")}
    files.update({f"tokenizer/{name}": {"bytes": 1, "sha256": "0" * 64} for name in TOKENIZER_FILES})
    files.update({name: {"bytes": 1, "sha256": "0" * 64} for name in ("LICENSE", POLICY, "NOTICE", "README.md")})
    report = {"scriptSHA256": {"export_embeddinggemma.py": "0" * 64}, "forgeLockSHA256": "0" * 64}
    return manifest(files, report, platform_name, SAMPLE_PROMPTS, WEIGHTS)


def notice(weights: str = WEIGHTS) -> str:
    return "\n".join([
        NOTICE_LINE,
        "",
        "EmbeddingGemma 300M (Aether-Kit Core AI export)",
        "",
        f"This bundle is a Model Derivative of Gemma: a modified version of {eg.REPO} at revision",
        f"{eg.REVISION}, by Google DeepMind. Its use, reproduction and distribution are subject to the",
        "Gemma Terms of Use (LICENSE, a copy of the Agreement).",
        "",
        "Use restrictions (Gemma Terms of Use, Section 3.2): you must not use this model, or any Model Derivative of",
        "it, for the restricted uses set forth in the Gemma Prohibited Use Policy (GEMMA-PROHIBITED-USE-POLICY.txt,",
        "ai.google.dev/gemma/prohibited_use_policy), or in violation of applicable laws and regulations. If you",
        "distribute it, you must include these use restrictions as an enforceable provision in the agreement that",
        "governs its use or distribution, give your recipients a copy of the Gemma Terms of Use, and keep this NOTICE.",
        "",
        "Modifications: Aether-Kit's Forge exporter re-expressed the sentence-transformers pipeline (Gemma 3 encoder,",
        f"mean pooling, two dense layers, L2 normalization) as one static Core AI graph with a fixed {eg.SEQUENCE}-"
        "token",
        "input and the bidirectional attention mask precomputed from the padding mask; "
        + ("the weights were rounded to float16" if weights == "float16" else "the float32 weights are unchanged"),
        "(computation stays float32). The tokenizer files are upstream's, unmodified. It is not an official release of",
        "Google or of EmbeddingGemma.",
        "",
    ])


def readme(platform_names: list[str], weights: str = WEIGHTS) -> str:
    return "\n".join([
        "---",
        "license: gemma",
        f"base_model: {eg.REPO}",
        "pipeline_tag: sentence-similarity",
        "tags: [aether-kit, core-ai, text-embeddings, embeddinggemma]",
        "---",
        "",
        "# EmbeddingGemma 300M for Aether-Kit (Core AI)",
        "",
        f"[{eg.REPO}](https://huggingface.co/{eg.REPO}) at revision `{eg.REVISION[:8]}`, exported by",
        "[Aether-Kit](https://github.com/bubstackdev/aether-kit)'s Forge exporter (`Forge/embeddinggemma/`) as one",
        f"static Core AI graph: `input_ids` and `attention_mask` (int32, {eg.SEQUENCE} tokens, right-padded with 0) to",
        f"an L2-normalized {eg.DIMENSION}-dimensional embedding. Mean pooling, the two dense projections and the",
        "normalization are in the graph; it computes in float32 (EmbeddingGemma does not support float16 activations)",
        ("from float16-stored weights." if weights == "float16" else "from float32 weights.")
        + " Prefix queries with `task: search result | query: ` and documents with `title: none | text: `.",
        f"Load it with Aether-Kit's `AetherEmbeddings` (`text-embedder@1`, {' and '.join(platform_names)}); the",
        "manifest is `.aether/aether.json`. Gated against the upstream implementation on CPU in float32 (see",
        "`export-report.json`).",
        "",
        "Gemma is provided under and subject to the Gemma Terms of Use found at",
        "[ai.google.dev/gemma/terms](https://ai.google.dev/gemma/terms) (`LICENSE`). Use is restricted by the Gemma",
        "Prohibited Use Policy (`GEMMA-PROHIBITED-USE-POLICY.txt`). A modified version, not an official Google",
        "release (see `NOTICE`).",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="new bundle directory")
    parser.add_argument("--platform", choices=["macos", "ios"], default="macos", help="the manifest's variant")
    parser.add_argument("--sample-manifest", action="store_true", help="print the manifest for placeholder files")
    parser.add_argument("--snapshot", type=Path, default=eg.DEFAULT_SNAPSHOT)
    parser.add_argument("--weights", choices=["float16", "float32"], default=WEIGHTS,
                        help="weight storage (compute is always float32); float32 only for the precision measurement")
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
                                                                "safetensors", "transformers", "tokenizers")},
        "forgeLockSHA256": eg.sha256_file(lock), "forgePyprojectSHA256": eg.sha256_file(HERE.parent / "pyproject.toml"),
        "scriptSHA256": {p.name: eg.sha256_file(p) for p in sorted(HERE.glob("*.py"))},
        "source": {"repository": eg.REPO, "revision": eg.REVISION, "snapshot": str(args.snapshot.resolve()),
                   "verified": {}},
        "precision": f"{args.weights} weight storage, float32 compute; int32 inputs, float32 output; no quantization",
        "precisionDecision": ("compute float32: the float16 encoder overflows to NaN on every reference case "
                              "(selftest_embeddinggemma.py, '*.float16.finite'), as upstream's card warns; weights "
                              "stored float16 and widened in the graph: ~1e-7 cosine against float32 weights "
                              "(selftest '*.fp16weights.cosine', check_embeddinggemma.py) for half the bytes"),
        "inputShape": [1, eg.SEQUENCE], "compute": COMPUTE, "conversion": {}, "stages": {},
    }
    report_path = args.output / "export-report.json"

    def save() -> None:
        report["peakRSSGB"] = peak_rss_gb()
        report["wallSeconds"] = round(time.monotonic() - started, 1)
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    log.start("verify-sources")
    report["source"]["verified"] = eg.verify_sources(args.snapshot)
    prompts = eg.prompts(args.snapshot)
    report["stages"]["verify-sources"] = log.done("verify-sources")
    save()

    log.start("export")
    graph = eg.EmbeddingGraph(eg.Upstream(args.snapshot), eg.DTYPES[COMPUTE_DTYPE], eg.DTYPES[args.weights]).eval()
    ids, mask = eg.encode(eg.tokenizer(args.snapshot), prompts["query"] + "What is the capital of France?")
    example = (torch.from_numpy(ids), torch.from_numpy(mask))
    report["conversion"]["encoder"] = convert(graph, example, args.output / ASSET, creation_date)
    del graph
    report["stages"]["export"] = log.done("export")
    save()

    log.start("package")
    (args.output / "tokenizer").mkdir()
    for name in TOKENIZER_FILES:
        shutil.copyfile(args.snapshot / name, args.output / "tokenizer" / name)
    shutil.copyfile(eg.TERMS_DIR / "GEMMA-TERMS-OF-USE.txt", args.output / "LICENSE")
    shutil.copyfile(eg.TERMS_DIR / POLICY, args.output / POLICY)
    (args.output / "NOTICE").write_text(notice(args.weights))
    (args.output / "README.md").write_text(readme(["macOS"], args.weights))
    files = {str(p.relative_to(args.output)): {"bytes": p.stat().st_size, "sha256": eg.sha256_file(p)}
             for p in tree_files(args.output) if str(p.relative_to(args.output)) not in NOT_BUNDLE}
    report["bundleBytes"] = sum(f["bytes"] for f in files.values())
    report["assetBytes"] = {"encoder": tree_bytes(args.output / ASSET),
                            "tokenizer": tree_bytes(args.output / "tokenizer")}
    report["stages"]["package"] = log.done("package")
    save()
    (args.output / ".aether").mkdir()
    manifest_path = args.output / ".aether/aether.json"
    manifest_path.write_text(json.dumps(manifest(files, report, args.platform, prompts, args.weights), indent=2) + "\n")
    report["aetherJSONSHA256"] = eg.sha256_file(manifest_path)
    save()
    sums = [f"{eg.sha256_file(p)}  {p.relative_to(args.output)}" for p in tree_files(args.output)
            if p.name not in ("SHA256SUMS", "forge-log.jsonl")]
    (args.output / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(json.dumps({"conversion": report["conversion"], "bundleBytes": report["bundleBytes"],
                      "peakRSSGB": report["peakRSSGB"], "wallSeconds": report["wallSeconds"]}, indent=1))


if __name__ == "__main__":
    main()

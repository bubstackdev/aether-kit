#!/usr/bin/env python3
"""Export ColModernVBERT (ModernVBERT/colmodernvbert-merged, MIT) as an Aether-Kit `late-interaction-retriever@1`
bundle.

    uv sync --frozen --project Forge
    Forge/.venv/bin/python -I Forge/colmodernvbert/fetch_upstream.py      # pinned license, pages + checkpoint, once
    scripts/gpu-lease -- Forge/.venv/bin/python -I Forge/colmodernvbert/export_colmodernvbert.py --output <new dir> \
        [--precision mixed|float16|float32]
    Forge/.venv/bin/python -I Forge/colmodernvbert/export_colmodernvbert.py --sample-manifest   # placeholder files

Nothing is downloaded: every file read is verified against `source-pins.json`. Conversion runs on the CPU. The output
directory is laid out as an installed revision root, with the query and document halves in two subtrees:

    OUTPUT/
      .aether/aether.json                 the Aether manifest (MANIFEST.md §1.3, §2; driver
                                          `late-interaction-retriever@1`)
      query/colmodernvbert_query.aimodel  `main`: tokens, attention_mask [1,64] int32 -> token_embeddings [1,64,128]
                                          float32 (unit rows, padding rows zero)
      query/tokenizer/                    the checkpoint's tokenizer files, unchanged
      doc/colmodernvbert_doc.aimodel      `main`: image [1,1,3,512,512] float32 on [-1,1] -> token_embeddings
                                          [1,89,128] float32 (unit rows; the 64 image rows are 13..76, 8 x 8)
      LICENSE, NOTICE                     MIT (illuin-tech/modernvbert's LICENSE) and the modification notice
      README.md                           the Hugging Face model card
      SHA256SUMS                          every bundle file except itself and the log
      export-report.json                  provenance, versions, timings, peak RSS
      forge-log.jsonl                     one event per stage

A page's score for a query is MaxSim: the sum over the query's real rows of the largest dot product with any of the
page's 89 rows. One Core AI JIT package serves macOS and iOS (no per-architecture AOT): `--platform` only sets the
manifest's `variant`. Parity: `reference_colmodernvbert.py` (upstream transformers on the CPU, float32) records the
references; `check_colmodernvbert.py` runs the exported graphs through Core AI on the GPU and gates them (cosine >=
0.999 per vector, MaxSim scores and rankings).
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import colmodernvbert_graphs as cg  # noqa: E402

ID = "colmodernvbert"
HF_REPO = "aether-models/aether-kit-colmodernvbert"
QUERY_ASSET, DOC_ASSET = "query/colmodernvbert_query.aimodel", "doc/colmodernvbert_doc.aimodel"
FUNCTION = "main"
QUERY_IO = (["tokens", "attention_mask"], ["token_embeddings"])
DOC_IO = (["image"], ["token_embeddings"])
NOT_BUNDLE = {".aether/aether.json", "SHA256SUMS", "export-report.json", "forge-log.jsonl"}
# Measured default (MANIFEST §0): GPU for every graph class.
COMPUTE = "gpu"


class Log:
    def __init__(self, path: Path) -> None:
        self.path, self.started = path, {}

    def event(self, stage: str, message: str, **fields) -> None:
        record = {"message": message, "stage": stage, "t": round(time.time(), 3), **fields}
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"[colmodernvbert:{stage}] {message}" + (f" {fields}" if fields else ""), file=sys.stderr, flush=True)

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


def convert(module: torch.nn.Module, example: tuple, io: tuple[list[str], list[str]], path: Path,
            description: str) -> dict:
    """One static function `main` -> one Core AI asset."""
    import coreai_torch as ct
    from coreai.runtime import AIModelAssetMetadata

    inputs, outputs = io
    t0 = time.monotonic()
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    with torch.no_grad():
        ep = torch.export.export(module, example, strict=False).run_decompositions(ct.get_decomp_table())
    converter.add_exported_program(ep, input_names=inputs, output_names=outputs, state_names=[],
                                   entrypoint_name=FUNCTION)
    del ep
    t1 = time.monotonic()
    program = converter.to_coreai()
    del converter
    program.optimize()
    metadata = AIModelAssetMetadata()
    metadata.author = "Illuin Technology and the ModernVBERT authors (ColModernVBERT); Core AI export: Aether-Kit Forge"
    metadata.license = "MIT"
    metadata.model_description = description
    metadata.creation_date = int(time.time())
    path.parent.mkdir(parents=True, exist_ok=True)
    program.save_asset(path, metadata)
    return {"exportSeconds": round(t1 - t0, 1), "convertSeconds": round(time.monotonic() - t1, 1),
            "bytes": tree_bytes(path), "inputs": inputs, "outputs": outputs, "functions": [FUNCTION]}


# MARK: the manifest (MANIFEST.md §1.3 `late-interaction-retriever@1`, §2.4 tokenizer, §2.5 preprocess.image, §2.8)

def precision_block(name: str) -> dict:
    vision, text = (("fp16" if d == torch.float16 else "fp32") for d in cg.PRECISIONS[name])
    weights = vision if vision == text else f"vision tower {vision}, connector and ModernBERT {text}"
    return {"weights": weights, "activations": weights,
            "components": {"vision": f"{vision} (SigLIP 2 B/16 tower)",
                           "text": f"{text} (pixel-shuffle connector and ModernBERT, in both graphs)",
                           "projection": "fp32 (the 768 -> 128 linear and the per-row L2 normalization)",
                           "io": "image float32, tokens and attention_mask int32; token embeddings float32"}}


def manifest(files: dict[str, dict], report: dict, platform_name: str) -> dict:
    start, end = cg.IMAGE_SPAN
    return {
        "schema": 1,
        "id": ID,
        "task": "visual-document-retrieval",
        "driver": "late-interaction-retriever@1",
        "variant": {"platform": platform_name},
        "assets": {"main": {"subtree": "", "files": [{"path": p, **meta} for p, meta in sorted(files.items())]}},
        "graphs": {
            "queryEncoder": {
                "file": f"main:{QUERY_ASSET}", "function": FUNCTION,
                "inputs": {"tokens": "tokens", "attentionMask": "attention_mask"},
                "outputs": {"tokenEmbeddings": "token_embeddings"},
                "compute": COMPUTE, "expectFrequentReshapes": False,
                "limits": {"staticSequence": cg.QUERY_LENGTH},
                "x-notes": ("token_embeddings [1,64,128] float32: one unit row per id, rows with attention_mask 0 "
                            "exactly zero (upstream multiplies by the mask). Keep the first count rows (the real ids, "
                            "the ten augmentation tokens included: upstream scores with them)."),
            },
            "docEncoder": {
                "file": f"main:{DOC_ASSET}", "function": FUNCTION,
                "inputs": {"image": "image"},
                "outputs": {"tokenEmbeddings": "token_embeddings"},
                "compute": COMPUTE, "expectFrequentReshapes": False,
                "x-notes": ("token_embeddings [1,89,128] float32, unit rows, all 89 scored (upstream keeps every doc "
                            "row: 3_MultiVectorMask has no skiplist). The fixed document prompt (89 ids, "
                            "`[CLS]<|begin_of_text|>User:<fake_token_around_image><global-img>` + 64 `<image>` + "
                            "`<fake_token_around_image>Describe the image.<end_of_utterance>\\nAssistant:[SEP]`) is a "
                            "graph constant. No pixelMask input: upstream's pixel_attention_mask has no effect "
                            "(SiglipVisionModel ignores patch_attention_mask), and a single 512 x 512 tile is all "
                            "ones anyway."),
            },
        },
        "tokenizer": {
            "source": {"dir": "main:query/tokenizer"}, "chatTemplate": None,
            "addSpecialTokens": True,
            "text": {"padId": cg.PAD, "padSide": "right", "padTo": cg.QUERY_LENGTH, "truncation": "bodyOnly",
                     "maskRule": "realTokens", "template": {"suffix": cg.AUGMENTATION}},
            "x-notes": ("A query row is upstream ColModernVBertProcessor.process_queries: [CLS] 50281, the query's "
                        "ids, "
                        "ten <end_of_utterance> 50405 (the query augmentation suffix), [SEP] 50282, with no prompt. "
                        "Upstream does not strip the query (the encoder text rule does: they differ only for leading "
                        "or trailing whitespace). A query longer than 52 ids is cut to its first 52 (the suffix "
                        "and both specials kept; upstream itself never truncates). Right-padded with [PAD] 50283 to 64 "
                        "(tokenizer_config.json's padding_side left would shift ModernBERT's positions); the mask is "
                        "1 for every real id."),
        },
        "preprocess": {
            "image": {
                "resize": {"mode": "stretch", "size": [cg.SIZE, cg.SIZE], "filter": "pillowBicubic"},
                "colorSpace": "deviceRGB", "channelOrder": "RGB",
                "scale": 1 / 255, "mean": list(cg.NORM_MEAN), "std": list(cg.NORM_STD),
                "layout": "NVCHW", "views": 1, "hostDType": "float32", "orientation": "exif",
                "x-upstream": {
                    "processor": "Idefics3ImageProcessorPil via ColModernVBertProcessor.process_images, "
                                 "do_image_splitting=False",
                    "resize": ["LANCZOS to longest edge 2048, aspect kept (other side int(), rounded up to even)",
                               "LANCZOS stretch to 512 x 512"],
                    "x-notes": ("Upstream resizes twice with Pillow LANCZOS; Aether-Kit's ImagePreprocessor has one "
                                "resize and no LANCZOS filter, so pillowBicubic (bit-exact to Pillow's BICUBIC) is the "
                                "nearest expressible stretch. Measured on the reference pages (export-report.json "
                                "`preprocess`): rankings unchanged; per-vector cosine to upstream's input mean 0.997, "
                                "min 0.86. Upstream also splits pages into 512 tiles by default; this bundle is the "
                                "single global tile."),
                },
            },
        },
        "postprocess": {
            "kind": "maxSim",
            "retrieval": {"imageTokenSpan": [start, end], "patchGrid": list(cg.PATCH_GRID)},
            "x-notes": ("Score(query, page) = sum over the query's real rows of max over the page's 89 rows of the dot "
                        "product (upstream score_retrieval; rows are unit, so dot = cosine). Image rows 13..76 are the "
                        "page's 8 x 8 grid row-major: row 13 + 8 r + c covers the 64 x 64-pixel cell (r, c) of the "
                        "512 x 512 input."),
        },
        "precision": precision_block(report["precisionName"]),
        "license": "MIT",
        "x-license": {"files": ["LICENSE", "NOTICE"], "modified": True},
        "provenance": {
            "source": cg.REPO, "sourceRevision": cg.REVISION,
            "producer": "aether-kit Forge/colmodernvbert/export_colmodernvbert.py",
            "exportScript": report["scriptSHA256"]["export_colmodernvbert.py"],
            "refs": {"upstreamCode": f"transformers {report['versions']['transformers']} ModernVBertModel + "
                                     "ColModernVBertForRetrieval",
                     "license": f"{cg.CODE_REPO}@{cg.CODE_COMMIT}:LICENSE",
                     "preprocess.image": f"{cg.REPO}@{cg.REVISION} preprocessor_config.json",
                     "tokenizer.text": f"transformers {report['versions']['transformers']} "
                                       "ColModernVBertProcessor.process_queries",
                     "forgeLockSHA256": report["forgeLockSHA256"], "exportReport": "export-report.json"},
        },
    }


def sample_manifest(platform_name: str, precision: str = "mixed") -> dict:
    files = {f"{asset}/{name}": {"bytes": 1, "sha256": "0" * 64} for asset in (QUERY_ASSET, DOC_ASSET)
             for name in ("main.hash", "main.mlirb", "metadata.json")}
    files.update({name: {"bytes": 1, "sha256": "0" * 64}
                  for name in ("LICENSE", "NOTICE", "README.md",
                               *(f"query/tokenizer/{t}" for t in cg.TOKENIZER_FILES))})
    report = {"scriptSHA256": {"export_colmodernvbert.py": "0" * 64}, "forgeLockSHA256": "0" * 64,
              "precisionName": precision, "versions": {"transformers": importlib.metadata.version("transformers")}}
    return manifest(files, report, platform_name)


def weights_text(precision: str) -> str:
    vision, text = (str(d).removeprefix("torch.") for d in cg.PRECISIONS[precision])
    return vision if vision == text else f"{vision} (vision tower) and {text} (ModernBERT)"


def notice(precision: str) -> str:
    w = weights_text(precision)
    return "\n".join([
        "ColModernVBERT (Aether-Kit Core AI export)",
        "",
        f"This bundle is a modified version of {cg.REPO} at revision {cg.REVISION},",
        "by Illuin Technology and the ModernVBERT authors (https://github.com/illuin-tech/modernvbert; arXiv",
        f"2510.01149), licensed under the MIT License (LICENSE, from {cg.CODE_REPO} at commit",
        f"{cg.CODE_COMMIT}). The tokenizer files in query/tokenizer/ are the checkpoint's, unchanged.",
        "",
        "Modifications: Aether-Kit's Forge exporter re-expressed the model, as implemented by Hugging Face",
        "transformers' ModernVBertModel and ColModernVBertForRetrieval (Apache-2.0), as two static Core AI graphs:",
        "the query encoder (64 tokens) and the document encoder (one 512 x 512 page image as a single global tile, the",
        "document prompt folded in as constants), each with the 768 -> 128 projection and per-token L2 normalization,",
        f"the weights in {w}. It is not an official release of ColModernVBERT.",
        "",
    ])


def readme(platform_names: list[str], precision: str) -> str:
    w = weights_text(precision)
    return "\n".join([
        "---",
        "license: mit",
        f"base_model: {cg.REPO}",
        "pipeline_tag: visual-document-retrieval",
        "tags: [aether-kit, core-ai, colpali, multi-vector, late-interaction, visual-document-retrieval]",
        "---",
        "",
        "# ColModernVBERT for Aether-Kit (Core AI)",
        "",
        f"[{cg.REPO}](https://huggingface.co/{cg.REPO}) at revision `{cg.REVISION[:8]}`, exported by",
        "[Aether-Kit](https://github.com/bubstackdev/aether-kit)'s Forge exporter (`Forge/colmodernvbert/`) as two",
        "static Core AI graphs: the query encoder (`query/`: up to 64 tokens, the query plus ten `<end_of_utterance>`",
        "augmentation tokens, as upstream's processor builds it) and the document encoder (`doc/`: a page stretched to",
        "512 x 512 as one global tile, 89 vectors, of which 64 form the page's 8 x 8 grid). Each returns unit-length",
        f"128-wide vectors per token in float32; the weights are {w}. A page's score for a query is MaxSim (the sum",
        "over query vectors of the best dot product with a page vector). Load it with Aether-Kit's `AetherEmbeddings`",
        f"(`late-interaction-retriever@1`, {' and '.join(platform_names)}); the manifest is `.aether/aether.json`.",
        "Gated against the upstream implementation on CPU in float32 (see `export-report.json`).",
        "",
        "Upstream also splits large pages into 512 tiles plus the global image; this bundle is the single-tile path,",
        "which is faster and fits a phone but reads small print less well.",
        "",
        "MIT (`LICENSE`). A modified version (see `NOTICE`).",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="new bundle directory")
    parser.add_argument("--precision", choices=sorted(cg.PRECISIONS), default="mixed",
                        help="mixed: vision tower float16, ModernBERT float32 (the gated default); float16; float32")
    parser.add_argument("--platform", choices=["macos", "ios"], default="macos", help="the manifest's variant")
    parser.add_argument("--sample-manifest", action="store_true", help="print the manifest for placeholder files")
    parser.add_argument("--reference", type=Path, help="reference_colmodernvbert.py's output, for the report's "
                                                       "preprocessing measurement")
    parser.add_argument("--snapshot", type=Path, default=cg.DEFAULT_SNAPSHOT)
    parser.add_argument("--upstream", type=Path, default=cg.DEFAULT_UPSTREAM, help="fetch_upstream.py's output")
    args = parser.parse_args()
    if args.sample_manifest:
        print(json.dumps(sample_manifest(args.platform, args.precision), indent=2))
        return
    if args.output is None:
        parser.error("--output is required")
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    started = time.monotonic()
    args.output.mkdir(parents=True)
    log = Log(args.output / "forge-log.jsonl")
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    vision_dtype, text_dtype = cg.PRECISIONS[args.precision]
    lock = HERE.parent / "uv.lock"
    report = {
        "schema": 1, "id": ID, "command": sys.argv,
        "exportedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {"os": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
                 "machine": platform.machine(), "python": platform.python_version()},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "transformers", "coreai-torch",
                                                                "coreai-core", "numpy", "pillow")},
        "forgeLockSHA256": cg.sha256_file(lock), "forgePyprojectSHA256": cg.sha256_file(HERE.parent / "pyproject.toml"),
        "scriptSHA256": {p.name: cg.sha256_file(p) for p in sorted(HERE.glob("*.py"))},
        "source": {"repository": cg.REPO, "revision": cg.REVISION, "snapshot": str(args.snapshot.resolve()),
                   "license": {"repository": cg.CODE_REPO, "commit": cg.CODE_COMMIT}, "verified": {}},
        "precisionName": args.precision,
        "precision": f"{args.precision}: vision tower {vision_dtype}, connector and ModernBERT {text_dtype}; "
                     "projection and L2 normalization float32; image float32 and tokens int32 in, token embeddings "
                     "float32 out; no quantization",
        "inputShapes": {"tokens": [1, cg.QUERY_LENGTH], "attention_mask": [1, cg.QUERY_LENGTH],
                        "image": [1, 1, 3, cg.SIZE, cg.SIZE]},
        "docPrompt": {"ids": cg.DOC_IDS, "imageTokenSpan": list(cg.IMAGE_SPAN), "patchGrid": list(cg.PATCH_GRID)},
        "compute": COMPUTE, "conversion": {}, "stages": {},
    }
    if args.reference:
        measured = json.loads((args.reference / "references.json").read_text())["singleStageResize"]
        report["preprocess"] = {"singleStageResize": {f: {k: v for k, v in m.items() if k != "perPage"}
                                                      for f, m in measured["filters"].items()},
                                "comment": measured["comment"], "zooCrossCheck":
                                    json.loads((args.reference / "references.json").read_text())["zooCrossCheck"]}
    report_path = args.output / "export-report.json"

    def save() -> None:
        report["peakRSSGB"] = peak_rss_gb()
        report["wallSeconds"] = round(time.monotonic() - started, 1)
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    log.start("verify-sources")
    report["source"]["verified"] = cg.verify_sources(args.snapshot, args.upstream)
    report["stages"]["verify-sources"] = log.done("verify-sources")
    save()

    log.start("export-query")
    graph = cg.QueryGraph(cg.build_model(args.snapshot), text_dtype).eval()
    ids = torch.full((1, cg.QUERY_LENGTH), cg.PAD, dtype=torch.int32)
    ids[0, :3] = torch.tensor([cg.CLS, cg.END_OF_UTTERANCE, cg.SEP], dtype=torch.int32)
    example = (ids, (ids != cg.PAD).to(torch.int32))
    report["conversion"]["queryEncoder"] = convert(
        graph, example, QUERY_IO, args.output / QUERY_ASSET,
        f"ColModernVBERT query encoder ({cg.REPO}@{cg.REVISION[:12]}): tokens, attention_mask [1,64] int32 -> "
        "unit token_embeddings [1,64,128] (padding rows zero).")
    del graph
    report["stages"]["export-query"] = log.done("export-query")
    save()

    log.start("export-doc")
    graph = cg.DocGraph(cg.build_model(args.snapshot), vision_dtype, text_dtype).eval()
    example = (torch.from_numpy(cg.synthetic_pixels(0)),)
    report["conversion"]["docEncoder"] = convert(
        graph, example, DOC_IO, args.output / DOC_ASSET,
        f"ColModernVBERT document encoder ({cg.REPO}@{cg.REVISION[:12]}): image [1,1,3,512,512] on [-1,1] -> unit "
        "token_embeddings [1,89,128] (image rows 13..76, 8 x 8).")
    del graph
    report["stages"]["export-doc"] = log.done("export-doc")
    save()

    log.start("package")
    shutil.copyfile(args.upstream / "upstream/LICENSE", args.output / "LICENSE")
    (args.output / "query/tokenizer").mkdir(parents=True)
    for name in cg.TOKENIZER_FILES:
        shutil.copyfile(args.snapshot / name, args.output / "query/tokenizer" / name)
    (args.output / "NOTICE").write_text(notice(args.precision))
    (args.output / "README.md").write_text(readme(["macOS"], args.precision))
    files = {str(p.relative_to(args.output)): {"bytes": p.stat().st_size, "sha256": cg.sha256_file(p)}
             for p in tree_files(args.output) if str(p.relative_to(args.output)) not in NOT_BUNDLE}
    report["bundleBytes"] = sum(f["bytes"] for f in files.values())
    report["assetBytes"] = {"queryEncoder": tree_bytes(args.output / QUERY_ASSET),
                            "docEncoder": tree_bytes(args.output / DOC_ASSET)}
    report["stages"]["package"] = log.done("package")
    save()
    (args.output / ".aether").mkdir()
    manifest_path = args.output / ".aether/aether.json"
    manifest_path.write_text(json.dumps(manifest(files, report, args.platform), indent=2) + "\n")
    report["aetherJSONSHA256"] = cg.sha256_file(manifest_path)
    save()
    sums = [f"{cg.sha256_file(p)}  {p.relative_to(args.output)}" for p in tree_files(args.output)
            if p.name not in ("SHA256SUMS", "forge-log.jsonl")]
    (args.output / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(json.dumps({"conversion": report["conversion"], "bundleBytes": report["bundleBytes"],
                      "peakRSSGB": report["peakRSSGB"], "wallSeconds": report["wallSeconds"]}, indent=1))


if __name__ == "__main__":
    main()

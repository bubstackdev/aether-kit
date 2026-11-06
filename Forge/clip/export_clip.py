#!/usr/bin/env python3
"""Export CLIP ViT-B/32 (openai/clip-vit-base-patch32, MIT) as an Aether-Kit `clip-joint@1` bundle.

    uv sync --frozen --project Forge
    Forge/.venv/bin/python -I Forge/clip/fetch_upstream.py            # pinned license, images + checkpoint, once
    Forge/.venv/bin/python -I Forge/clip/export_clip.py --output <new bundle dir> [--precision float16|float32]
    Forge/.venv/bin/python -I Forge/clip/export_clip.py --sample-manifest   # the manifest for placeholder files

Nothing is downloaded: every file read is verified against `source-pins.json`. Conversion runs on the CPU. The output
directory is laid out as an installed revision root:

    OUTPUT/
      .aether/aether.json             the Aether manifest (MANIFEST.md §1.3, §2; driver `clip-joint@1`)
      clip_vit_b32_image.aimodel      `main`: image [1,3,224,224] fp16 on [0,1] -> image_embedding [1,512] fp32
      clip_vit_b32_text.aimodel       `batch1` / `batch8`: tokens [1|8,77] int32 -> text_embedding [1|8,512] fp32
                                      (one asset, two static functions sharing the weights)
      tokenizer/                      the checkpoint's CLIP tokenizer files, unchanged
      LICENSE, NOTICE                 MIT (openai/CLIP's LICENSE) and the modification notice
      README.md                       the Hugging Face model card
      SHA256SUMS                      every bundle file except itself and the log
      export-report.json              provenance, versions, timings, peak RSS
      forge-log.jsonl                 one event per stage

Both embeddings are L2-normalized in the graph, so cosine similarity is their dot product; zero-shot classification is
softmax(logitScale * cosine) over the label texts with the manifest's `logitScale` (upstream's `logit_scale.exp()`).
One Core AI JIT package serves macOS and iOS (no per-architecture AOT): `--platform` only sets the manifest's
`variant`. Parity: `reference_clip.py` (upstream transformers `CLIPModel` on the CPU, float32) records the references;
`check_clip.py` runs the exported graphs through Core AI on the GPU and gates them (cosine >= 0.999 per vector, same
zero-shot argmax).
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
import clip_graphs as cg  # noqa: E402

ID = "clip-vit-b32"
HF_REPO = "aether-models/aether-kit-clip-vit-b32"
IMAGE_ASSET, TEXT_ASSET = "clip_vit_b32_image.aimodel", "clip_vit_b32_text.aimodel"
IMAGE_FUNCTION = "main"
# Static text batches, one function each in the text asset (MANIFEST §2.3 `buckets`, rule smallestFitting). Measured
# on the GPU (bench_clip.py): a batch-8 call costs about what a batch-1 call does, and the second function adds ~0.1 MB.
TEXT_BATCHES = {1: "batch1", 8: "batch8"}
IMAGE_IO = (["image"], ["image_embedding"])
TEXT_IO = (["tokens"], ["text_embedding"])
NOT_BUNDLE = {".aether/aether.json", "SHA256SUMS", "export-report.json", "forge-log.jsonl"}
# Measured default (MANIFEST §0): GPU for every graph class. The zoo survey also saw CLIP produce NaN on the CPU unit.
COMPUTE = "gpu"


class Log:
    def __init__(self, path: Path) -> None:
        self.path, self.started = path, {}

    def event(self, stage: str, message: str, **fields) -> None:
        record = {"message": message, "stage": stage, "t": round(time.time(), 3), **fields}
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"[clip:{stage}] {message}" + (f" {fields}" if fields else ""), file=sys.stderr, flush=True)

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


def convert(programs: list[tuple], io: tuple[list[str], list[str]], path: Path, description: str) -> dict:
    """[(module, example, function name)] -> one Core AI asset holding every function."""
    import coreai_torch as ct
    from coreai.runtime import AIModelAssetMetadata

    inputs, outputs = io
    t0 = time.monotonic()
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    for module, example, function in programs:
        with torch.no_grad():
            ep = torch.export.export(module, example, strict=False).run_decompositions(ct.get_decomp_table())
        converter.add_exported_program(ep, input_names=inputs, output_names=outputs, state_names=[],
                                       entrypoint_name=function)
        del ep
    t1 = time.monotonic()
    program = converter.to_coreai()
    del converter
    program.optimize()
    metadata = AIModelAssetMetadata()
    metadata.author = "OpenAI (CLIP); Core AI export: Aether-Kit Forge"
    metadata.license = "MIT"
    metadata.model_description = description
    metadata.creation_date = int(time.time())
    program.save_asset(path, metadata)
    return {"exportSeconds": round(t1 - t0, 1), "convertSeconds": round(time.monotonic() - t1, 1),
            "bytes": tree_bytes(path), "inputs": inputs, "outputs": outputs,
            "functions": [function for _, _, function in programs]}


# MARK: the manifest (MANIFEST.md §1.3 `clip-joint@1`, §2.4 tokenizer, §2.5 preprocess.image, §2.8 postprocess)

def precision_block(dtype_name: str) -> dict:
    w = "fp16" if dtype_name == "float16" else "fp32"
    return {"weights": w, "activations": w,
            "components": {"towers": f"{w} (both transformers and their projections)",
                           "normalization": "fp32 (input mean/std and the output L2 normalization)",
                           "io": "image float16, tokens int32; embeddings float32"}}


def manifest(files: dict[str, dict], report: dict, platform_name: str) -> dict:
    return {
        "schema": 1,
        "id": ID,
        "task": "image-text-embedding",
        "driver": "clip-joint@1",
        "variant": {"platform": platform_name},
        "assets": {"main": {"subtree": "", "files": [{"path": p, **meta} for p, meta in sorted(files.items())]}},
        "graphs": {
            "imageEncoder": {
                "file": f"main:{IMAGE_ASSET}", "function": IMAGE_FUNCTION,
                "inputs": {"image": "image"},
                "outputs": {"imageEmbedding": "image_embedding"},
                "compute": COMPUTE, "expectFrequentReshapes": False,
            },
            "textEncoder": {
                "file": f"main:{TEXT_ASSET}", "function": TEXT_BATCHES[1],
                "buckets": {"axis": "batch", "rule": "smallestFitting",
                            "entries": [{"value": b, "function": f} for b, f in sorted(TEXT_BATCHES.items())]},
                "inputs": {"tokens": "tokens"},
                "outputs": {"textEmbedding": "text_embedding"},
                "compute": COMPUTE, "expectFrequentReshapes": False,
                "limits": {"staticSequence": cg.TEXT_LENGTH},
            },
        },
        "tokenizer": {
            "source": {"dir": "main:tokenizer"}, "chatTemplate": None, "renderer": "clip-bpe",
            "addSpecialTokens": True,
            "text": {"padId": cg.EOS, "padSide": "right", "padTo": cg.TEXT_LENGTH, "truncation": "right",
                     "maskRule": "none", "unusedBatchRows": [cg.BOS, cg.EOS]},
            "x-notes": ("CLIP BPE (NFC, whitespace collapsed, lowercase; `</w>` end-of-word): BOS 49406, at most 75 "
                        "pieces, EOS 49407, right-padded with 49407 to 77, as transformers' CLIPTokenizer does with "
                        "padding='max_length', truncation=True. Only <|startoftext|> and <|endoftext|> are added "
                        "tokens: '!' is an ordinary piece here ('hello!' -> 3306, 256), unlike SD 2.x's tokenizer, "
                        "where '!' is the pad token 0. The graph takes no attention mask (causal attention makes it "
                        "redundant) and pools the first EOS, so any pad id works. Rows of a bucket beyond the texts "
                        "are filled with [BOS, EOS, then pad] and their outputs discarded."),
        },
        "preprocess": {
            "image": {
                "resize": {"mode": "centerCrop", "size": [cg.SIZE, cg.SIZE], "filter": "pillowBicubic",
                           "shortestSide": cg.SIZE},
                "colorSpace": "sRGB", "channelOrder": "RGB",
                "scale": 1 / 255, "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0],
                "layout": "NCHW", "hostDType": "float32", "orientation": "exif",
                "x-notes": ("Upstream (openai/CLIP `_transform`: torchvision Resize(224, BICUBIC) + CenterCrop(224) "
                            "on a PIL image): resize the shortest side to 224 with Pillow BICUBIC (long side "
                            "int(224 * long / short)), center-crop 224 x 224 (offsets round((side - 224) / 2)), /255. "
                            "transformers 5's CLIPImageProcessor floors the crop offset and resizes with torch, so it "
                            "can differ by a pixel of offset on odd margins. OpenAI CLIP normalization (mean "
                            "0.48145466/0.4578275/0.40821073, std 0.26862954/0.26130258/0.27577711) is in the graph: "
                            "feed RGB on [0,1]. The graph input is float16; the host builds float32 and converts."),
            },
        },
        "postprocess": {
            "kind": "embedding",
            "embedding": {"normalized": "inGraph", "pooling": "inGraph", "similarity": "cosine",
                          "dimension": cg.EMBED_DIM, "logitScale": report["logitScale"]},
            "x-notes": ("imageEmbedding [1,512] and textEmbedding [B,512] are unit-length float32 (L2-normalized in "
                        "the graph), so cosine similarity is their dot product. Zero-shot classification: "
                        "softmax over labels of logitScale * cosine(image, text(label)); logitScale is upstream's "
                        "logit_scale.exp(). OpenAI's prompt for ImageNet-style labels is 'a photo of a {label}.'."),
        },
        "precision": precision_block(report["precisionName"]),
        "license": "MIT",
        "x-license": {"files": ["LICENSE", "NOTICE"], "modified": True,
                      "x-notes": ("OpenAI's model card scopes CLIP to research use: it calls any deployed use "
                                  "currently out of scope without task-specific testing, and surveillance and facial "
                                  "recognition always out of scope. That is guidance, not a license term.")},
        "provenance": {
            "source": cg.REPO, "sourceRevision": cg.REVISION,
            "producer": "aether-kit Forge/clip/export_clip.py",
            "exportScript": report["scriptSHA256"]["export_clip.py"],
            "refs": {"upstreamCode": f"transformers {report['versions']['transformers']} CLIPModel",
                     "license": f"{cg.CODE_REPO}@{cg.CODE_COMMIT}:LICENSE",
                     "forgeLockSHA256": report["forgeLockSHA256"], "exportReport": "export-report.json"},
        },
    }


def sample_manifest(platform_name: str, precision: str = "float16") -> dict:
    files = {f"{asset}/{name}": {"bytes": 1, "sha256": "0" * 64} for asset in (IMAGE_ASSET, TEXT_ASSET)
             for name in ("main.hash", "main.mlirb", "metadata.json")}
    files.update({name: {"bytes": 1, "sha256": "0" * 64}
                  for name in ("LICENSE", "NOTICE", "README.md", *(f"tokenizer/{t}" for t in cg.TOKENIZER_FILES))})
    report = {"scriptSHA256": {"export_clip.py": "0" * 64}, "forgeLockSHA256": "0" * 64, "logitScale": 100.0,
              "precisionName": precision, "versions": {"transformers": importlib.metadata.version("transformers")}}
    return manifest(files, report, platform_name)


def notice(precision: str) -> str:
    w = "float16" if precision == "float16" else "float32"
    return "\n".join([
        "CLIP ViT-B/32 (Aether-Kit Core AI export)",
        "",
        f"This bundle is a modified version of {cg.REPO} at revision {cg.REVISION},",
        "by OpenAI (CLIP; https://github.com/openai/CLIP), licensed under the MIT License (LICENSE, from",
        f"{cg.CODE_REPO} at commit {cg.CODE_COMMIT}). The tokenizer files in tokenizer/ are the checkpoint's, unchanged.",
        "",
        "Modifications: Aether-Kit's Forge exporter re-expressed the model, as implemented by Hugging Face transformers'",
        "CLIPModel (Apache-2.0), as two static Core AI graphs: the image tower (224 x 224 input, CLIP normalization",
        "folded in) and the text tower (77 tokens, static batches 1 and 8, pooled at the first end-of-text token",
        f"without an attention mask), each with its projection and an in-graph L2 normalization, weights in {w}.",
        "It is not an official release of CLIP.",
        "",
        "OpenAI's model card (https://github.com/openai/CLIP/blob/main/model-card.md) describes CLIP as a research",
        "model: it considers any deployed use currently out of scope without thorough task-specific testing, and",
        "surveillance and facial recognition always out of scope.",
        "",
    ])


def readme(platform_names: list[str], precision: str) -> str:
    w = "float16" if precision == "float16" else "float32"
    return "\n".join([
        "---",
        "license: mit",
        f"base_model: {cg.REPO}",
        "pipeline_tag: zero-shot-image-classification",
        "tags: [aether-kit, core-ai, clip, image-text-embedding, zero-shot-image-classification]",
        "---",
        "",
        "# CLIP ViT-B/32 for Aether-Kit (Core AI)",
        "",
        f"[{cg.REPO}](https://huggingface.co/{cg.REPO}) at revision `{cg.REVISION[:8]}`, exported by",
        "[Aether-Kit](https://github.com/bubstackdev/aether-kit)'s Forge exporter (`Forge/clip/`) as two static Core",
        "AI graphs: the image tower (an RGB image resized to 224 on its shortest side and center-cropped, on [0, 1];",
        "CLIP normalization is in the graph) and the text tower (77 CLIP BPE tokens, batches of 1 or 8). Each returns",
        f"unit-length 512-wide embeddings in float32; the towers run in {w}. Cosine similarity is the dot product, and",
        "zero-shot classification is a softmax over 100 x cosine. Load it with Aether-Kit's `AetherVision`",
        f"(`clip-joint@1`, {' and '.join(platform_names)}); the manifest is `.aether/aether.json`. Gated against the",
        "upstream implementation on CPU in float32 (see `export-report.json`).",
        "",
        "OpenAI's [model card](https://github.com/openai/CLIP/blob/main/model-card.md) describes CLIP as a research",
        "model and considers untested deployed use, surveillance and facial recognition out of scope.",
        "",
        "MIT (`LICENSE`). A modified version (see `NOTICE`).",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="new bundle directory")
    parser.add_argument("--precision", choices=sorted(cg.DTYPES), default="float16", help="the towers' weights")
    parser.add_argument("--platform", choices=["macos", "ios"], default="macos", help="the manifest's variant")
    parser.add_argument("--sample-manifest", action="store_true", help="print the manifest for placeholder files")
    parser.add_argument("--snapshot", type=Path, default=cg.DEFAULT_SNAPSHOT)
    parser.add_argument("--upstream", type=Path, default=cg.DEFAULT_UPSTREAM, help="fetch_upstream.py's unpack")
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
    dtype = cg.DTYPES[args.precision]
    lock = HERE.parent / "uv.lock"
    report = {
        "schema": 1, "id": ID, "command": sys.argv,
        "exportedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {"os": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
                 "machine": platform.machine(), "python": platform.python_version()},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "transformers", "coreai-torch",
                                                                "coreai-core", "numpy")},
        "forgeLockSHA256": cg.sha256_file(lock), "forgePyprojectSHA256": cg.sha256_file(HERE.parent / "pyproject.toml"),
        "scriptSHA256": {p.name: cg.sha256_file(p) for p in sorted(HERE.glob("*.py"))},
        "source": {"repository": cg.REPO, "revision": cg.REVISION, "snapshot": str(args.snapshot.resolve()),
                   "license": {"repository": cg.CODE_REPO, "commit": cg.CODE_COMMIT}, "verified": {}},
        "precisionName": args.precision,
        "precision": f"towers {args.precision}; normalization float32; image float16 and tokens int32 in, "
                     "embeddings float32 out; no quantization",
        "inputShapes": {"image": [1, 3, cg.SIZE, cg.SIZE],
                        "tokens": {f: [b, cg.TEXT_LENGTH] for b, f in TEXT_BATCHES.items()}},
        "compute": COMPUTE, "conversion": {}, "stages": {},
    }
    report_path = args.output / "export-report.json"

    def save() -> None:
        report["peakRSSGB"] = peak_rss_gb()
        report["wallSeconds"] = round(time.monotonic() - started, 1)
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    log.start("verify-sources")
    report["source"]["verified"] = cg.verify_sources(args.snapshot, args.upstream)
    report["logitScale"] = cg.logit_scale(cg.build_model(args.snapshot))
    report["stages"]["verify-sources"] = log.done("verify-sources")
    save()

    log.start("export-image")
    graph = cg.ImageGraph(cg.build_model(args.snapshot), dtype).eval()
    example = (torch.from_numpy(cg.synthetic_tensor(0)).to(torch.float16),)
    report["conversion"]["imageEncoder"] = convert(
        [(graph, example, IMAGE_FUNCTION)], IMAGE_IO, args.output / IMAGE_ASSET,
        f"CLIP ViT-B/32 image tower ({cg.REPO}@{cg.REVISION[:12]}): image [1,3,224,224] RGB on [0,1] -> unit "
        "image_embedding [1,512].")
    del graph
    report["stages"]["export-image"] = log.done("export-image")
    save()

    log.start("export-text")
    programs = []
    for batch, function in sorted(TEXT_BATCHES.items()):
        ids = torch.full((batch, cg.TEXT_LENGTH), cg.EOS, dtype=torch.int32)
        ids[:, 0] = cg.BOS
        programs.append((cg.TextGraph(cg.build_model(args.snapshot), dtype).eval(), (ids,), function))
    report["conversion"]["textEncoder"] = convert(
        programs, TEXT_IO, args.output / TEXT_ASSET,
        f"CLIP ViT-B/32 text tower ({cg.REPO}@{cg.REVISION[:12]}): tokens [B,77] int32 (B = 1 or 8, one function "
        "each) -> unit text_embedding [B,512].")
    del programs
    report["stages"]["export-text"] = log.done("export-text")
    save()

    log.start("package")
    shutil.copyfile(args.upstream / "upstream/LICENSE", args.output / "LICENSE")
    (args.output / "tokenizer").mkdir()
    for name in cg.TOKENIZER_FILES:
        shutil.copyfile(args.snapshot / name, args.output / "tokenizer" / name)
    (args.output / "NOTICE").write_text(notice(args.precision))
    (args.output / "README.md").write_text(readme(["macOS"], args.precision))
    files = {str(p.relative_to(args.output)): {"bytes": p.stat().st_size, "sha256": cg.sha256_file(p)}
             for p in tree_files(args.output) if str(p.relative_to(args.output)) not in NOT_BUNDLE}
    report["bundleBytes"] = sum(f["bytes"] for f in files.values())
    report["assetBytes"] = {"imageEncoder": tree_bytes(args.output / IMAGE_ASSET),
                            "textEncoder": tree_bytes(args.output / TEXT_ASSET)}
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

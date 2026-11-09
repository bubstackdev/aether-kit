#!/usr/bin/env python3
"""Export a Qwen3-VL checkpoint (Apache-2.0; 2B Instruct by default) as an Aether-Kit `vlm-static-inject@1` bundle.

    uv sync --frozen --project Forge
    Forge/.venv/bin/python -I Forge/qwen3vl/fetch_upstream.py        # pinned 2B checkpoint, license text, photos; once
    Forge/.venv/bin/python -I Forge/qwen3vl/export_qwen3vl.py --output <new bundle dir> [--pins <source pins>]
    Forge/.venv/bin/python -I Forge/qwen3vl/export_qwen3vl.py --sample-manifest

`--pins` selects the source checkpoint: `source-pins.json` (Qwen/Qwen3-VL-2B-Instruct, the default) or another pins
file beside it, such as `source-pins-4b-abliterated.json` (a 4B fine-tune read from the local Hugging Face cache, a
local Forge input). Nothing is downloaded: every file read is verified against the pins. Every size comes from the
checkpoint's config. Conversion runs on the CPU (the float32 upstream build plus the converter: about 9 GB for 2B,
20 GB for 4B). The output directory is laid out as an installed revision root (`<prefix>` from the pins):

    OUTPUT/
      .aether/aether.json                 the Aether manifest (MANIFEST.md §1.2 `vlm-static-inject@1`, §2)
      <prefix>_vision.aimodel             vision-tower@1: patches f32 [2048,1536], grid i32 [2] -> image_embeds f16
                                          [512,H], deepstack_embeds f16 [1536,H]
      <prefix>_decoder.aimodel            decoder-lm@1 + static image inputs (image_embeds f16 [N,H], deepstack_embeds
                                          f16 [3N,H], image_positions i32 [N,3], rope_shift_start/amount i32 [M]);
                                          states keyCache/valueCache f16 [layers,1,8,C,128]; logits f32 [1,1,vocab]
      tokenizer/                          upstream's tokenizer.json, tokenizer_config.json, vocab.json, merges.txt,
                                          chat_template.jinja
      LICENSE                             Apache-2.0 (QwenLM/Qwen3-VL's text)
      NOTICE                              attribution and the modification notice
      README.md                           the model card
      SHA256SUMS, export-report.json, forge-log.jsonl

The graphs and their contracts are documented in qwen3vl_graphs.py. Precision (measured, see export-report.json and
check_qwen3vl.py): float16-stored weights in both graphs; the tower computes in float32 (VISION_COMPUTE), the decoder's
residual stream, matmuls and tied head in float16 with float32 attention, norm statistics, rotary and logits
(DECODER_COMPUTE, HEAD_COMPUTE).
`--vision-compute` / `--decoder-compute` / `--weights` export the alternatives for the measurement. Reproducible as the
EmbeddingGemma exporter is (seeded generated names, fixed creation date, deterministic Swift hashing), except the graphs'
`main.mlirb`/`main.hash` weight-blob names (coreai-core seeds them per execution).
"""

from __future__ import annotations

import os
import sys

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
import qwen3vl_graphs as qg  # noqa: E402

FUNCTION = "main"
VISION_INPUTS, VISION_OUTPUTS = ["patches", "grid"], ["image_embeds", "deepstack_embeds"]
DECODER_INPUTS = ["input_ids", "position_ids", "image_embeds", "deepstack_embeds", "image_positions", "rope_shift_start",
                  "rope_shift_amount"]
DECODER_STATES, DECODER_OUTPUTS = ["keyCache", "valueCache"], ["logits"]
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
NOT_BUNDLE = {".aether/aether.json", "SHA256SUMS", "export-report.json", "forge-log.jsonl"}
COMPUTE = "gpu"
# Measured (selftest_qwen3vl.py eager, check_qwen3vl.py on the GPU); see the report's precisionDecision.
VISION_COMPUTE = "float32"
DECODER_COMPUTE = "float16"
DECODER_ATTENTION = "float32"
# The tied head's matmul, float16 as 0.0.7's describer export: measured on 4B (DescriberGateTests, release), the same
# greedy tokens as the float32 head, 6.7 ms less per decode step and 5.8 GiB less Metal after load. float32 adds a
# float32 copy of the embedding (1.45 GiB at 4B) and its bandwidth to every step.
HEAD_COMPUTE = "float16"
WEIGHTS = "float16"
# The tower's outputs and the decoder's image statics: the decoder casts its embeddings to its float16 residual stream,
# so float16 statics lose nothing and halve what is bound on every step.
IMAGE_DTYPE = "float16"
CREATION_DATE = "2026-11-06T00:00:00+00:00"
STOP_IDS = {151645: "<|im_end|>", 151643: "<|endoftext|>"}   # generation_config.eos_token_id (both tokenizer eos)


class Log:
    def __init__(self, path: Path) -> None:
        self.path, self.started = path, {}

    def event(self, stage: str, message: str, **fields) -> None:
        record = {"message": message, "stage": stage, "t": round(time.time(), 3), **fields}
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"[qwen3vl:{stage}] {message}" + (f" {fields}" if fields else ""), file=sys.stderr, flush=True)

    def start(self, stage: str) -> None:
        self.started[stage] = time.monotonic()
        self.event(stage, "start")

    def done(self, stage: str, **fields) -> float:
        seconds = round(time.monotonic() - self.started.pop(stage), 2)
        self.event(stage, "done", seconds=seconds, peakRSSGB=peak_rss_gb(), **fields)
        return seconds


class Bundle:
    """The bundle's names, from the pins (`bundle`)."""

    def __init__(self, pins: dict) -> None:
        b = pins["bundle"]
        self.id, self.title, self.hf_repo = b["id"], b["title"], b.get("hfRepo")
        self.vision, self.decoder = f"{b['assetPrefix']}_vision.aimodel", f"{b['assetPrefix']}_decoder.aimodel"


def peak_rss_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)


def tree_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def tree_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in tree_files(root))


@contextlib.contextmanager
def deterministic_conversion(creation_date: int):
    """Seeded generated names and a fixed creation stamp (as Forge/embeddinggemma/export_embeddinggemma.py)."""
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


def drop_identity_copies(ep) -> int:
    """Removes the same-shape `aten.copy` nodes each in-place cache write leaves (the 0.0.x running contract)."""
    dropped = 0
    for node in list(ep.graph.nodes):
        if node.op != "call_function" or node.target != torch.ops.aten.copy.default:
            continue
        dst, src = node.args[0], node.args[1]
        dv = dst.meta.get("val")
        sv = src.meta.get("val") if isinstance(src, torch.fx.Node) else None
        if dv is None or sv is None or len(node.args) > 2:
            continue
        if dv.dtype == sv.dtype and [str(d) for d in dv.shape] == [str(d) for d in sv.shape]:
            node.replace_all_uses_with(src)
            ep.graph.erase_node(node)
            dropped += 1
    if dropped:
        ep.graph_module.recompile()
    return dropped


def convert(module, example: tuple, dynamic, path: Path, creation_date: int, inputs: list[str],
            outputs: list[str], states: list[str], description: str) -> dict:
    import coreai_torch as ct
    from coreai.runtime import AIModelAssetMetadata

    with deterministic_conversion(creation_date):
        t0 = time.monotonic()
        with torch.no_grad():
            ep = torch.export.export(module, example, dynamic_shapes=dynamic, strict=False)
            ep = ep.run_decompositions(ct.get_decomp_table())
        mutated = len(ep.graph_signature.user_inputs_to_mutate)
        if mutated != len(states):
            raise SystemExit(f"{mutated} mutated inputs for the {len(states)} states {states}")
        dropped = drop_identity_copies(ep) if states else 0
        t1 = time.monotonic()
        converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
        converter.add_exported_program(ep, input_names=inputs, output_names=outputs, state_names=states,
                                       entrypoint_name=FUNCTION)
        del ep
        program = converter.to_coreai()
        del converter
        program.optimize()
        metadata = AIModelAssetMetadata()
        metadata.author = "Alibaba Qwen (Qwen3-VL); Core AI export: Aether-Kit Forge"
        metadata.license = "Apache-2.0"
        metadata.model_description = description
        metadata.creation_date = creation_date
        program.save_asset(path, metadata)
    return {"exportSeconds": round(t1 - t0, 1), "convertSeconds": round(time.monotonic() - t1, 1),
            "bytes": tree_bytes(path), "inputs": inputs, "outputs": outputs, "states": states,
            "function": FUNCTION, "identityCopiesDropped": dropped}


# MARK: the manifest (MANIFEST.md §1.2 vlm-static-inject@1, §2.3 graphs, §2.5 preprocess, §2.6 image)

def manifest(files: dict[str, dict], report: dict, platform_name: str, prec: dict, capacity: int = qg.PATCH_CAPACITY,
             template_sha: str = "0" * 64, slots: int = qg.IMAGE_SLOTS, images: int = qg.MAX_IMAGES,
             vocab: int = 151936) -> dict:
    tokens = capacity // 4
    names = Bundle(qg.PINS)
    VISION_ASSET, DECODER_ASSET = names.vision, names.decoder
    return {
        "schema": 1,
        "id": names.id,
        "task": "image-text-to-text",
        "driver": "vlm-static-inject@1",
        "variant": {"platform": platform_name},
        "assets": {"main": {"subtree": "", "files": [{"path": p, **meta} for p, meta in sorted(files.items())]}},
        "graphs": {
            "vision": {
                "kind": "vision-tower@1",
                "file": f"main:{VISION_ASSET}", "function": FUNCTION,
                "inputs": {"patches": "patches", "grid": "grid"},
                "outputs": {"imageEmbeds": "image_embeds", "deepstackEmbeds": "deepstack_embeds"},
                "compute": COMPUTE, "expectFrequentReshapes": False,
                "x-notes": (f"One static shape for every image: patches [{capacity},1536] holds the image's h*w "
                            "block-major rows first and zeros after; grid int32 [2] is (h, w) in patches (both even, "
                            f"h*w <= {capacity}). Padding rows are masked as keys in the graph and the output rows past "
                            "h*w/4 are zero. Positional tables (bilinear position-table resampling, 2-D rotary) are "
                            "computed in the graph from grid, as upstream computes them from grid_thw. One image per "
                            "call; the host copies its rows into the decoder's static slots."),
            },
            "decoder": {
                "file": f"main:{DECODER_ASSET}", "function": FUNCTION,
                "inputs": {
                    "tokens": "input_ids", "positions": "position_ids",
                    "imageEmbeds": {"name": "image_embeds", "binding": "static"},
                    "deepstackEmbeds": {"name": "deepstack_embeds", "binding": "static"},
                    "imagePositions": {"name": "image_positions", "binding": "static"},
                    "ropeShiftStart": {"name": "rope_shift_start", "binding": "static"},
                    "ropeShiftAmount": {"name": "rope_shift_amount", "binding": "static"},
                },
                "inputEncodings": {"tokens": {"extensionIds": {"base": vocab, "count": slots}}},
                "outputs": {"logits": {"name": "logits", "rows": "last"}},
                "states": {"sequence": [{"tensors": DECODER_STATES, "kind": "kv", "seqAxis": 3}]},
                "limits": {"maxContext": qg.CONTEXT_MAX, "prefillChunk": qg.PREFILL_MAX},
                "compute": COMPUTE, "expectFrequentReshapes": True,
                "x-notes": ("Running contract: input_ids [1,S] (S 1..512), position_ids [1,T] = 0..T-1 (the last S are "
                            "the fed tokens'), keyCache/valueCache written in place at rows T-S..T-1; logits are the "
                            "last fed row only. maxContext is the graph's KV axis bound (upstream declares 262144). "
                            f"Image tokens are extension ids vocab + slot, packed over the prompt's images ({slots} "
                            f"slots, {images} images); the static inputs hold each slot's rows and rotary positions "
                            "and each image's rope shift (image.rope)."),
            },
        },
        "tokenizer": {
            "source": {"dir": "main:tokenizer"},
            "chatTemplate": "main:tokenizer/chat_template.jinja",
            "chatTemplateSha256": template_sha,
            "stop": [STOP_IDS[i] for i in sorted(STOP_IDS, reverse=True)],
            "thinking": None,
            "toolDialect": "hermes-json",
        },
        "preprocess": {
            "image": {
                "resize": {"mode": "smartResize", "filter": "torchUint8Bicubic", "factor": 32, "minPixels": 65536,
                           "maxPixels": capacity * 256},
                "colorSpace": "deviceRGB", "channelOrder": "RGB", "scale": 1 / 255,
                "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
                "layout": "patches",
                "patchify": {"patch": 16, "temporal": 2, "merge": 2, "rowOrder": "blockMajor", "withinPatch": "C,T,py,px"},
                "hostDType": "float32",
                "x-notes": ("smartResize is upstream's smart_resize (transformers qwen2_vl image processing), then a "
                            "resize to exactly that size (upstream's fast processor: torchvision uint8 bicubic, "
                            f"antialias). maxPixels is the tower's capacity ({capacity} patches); upstream's own "
                            "default longest_edge is 16777216 pixels, the bundle's budget is the graph's."),
            },
        },
        "image": {
            "feed": [{"from": "vision.imageEmbeds", "to": "decoder.imageEmbeds"},
                     {"from": "vision.deepstackEmbeds", "to": "decoder.deepstackEmbeds"}],
            "tokens": {"pad": "<|image_pad|>", "padId": qg.IMAGE_PAD_ID, "open": "<|vision_start|>",
                       "openId": qg.VISION_START_ID, "close": "<|vision_end|>", "closeId": qg.VISION_END_ID,
                       "count": "gridH*gridW", "idBase": vocab},
            "rope": {"rule": "qwen3vl-mrope",
                     "textOnly": {"start": qg.TEXT_ONLY_SHIFT_START, "amount": 0, "zeroStatics": True}},
            "deepstack": {"rowsPerToken": qg.DEEPSTACK_LEVELS, "layout": "levelMajor"},
            "maxImages": images,
            "slots": slots,
            "x-notes": ("The chat template renders one <|vision_start|><|image_pad|><|vision_end|> per image; the host "
                        "expands image k's pad to count_k = gridH*gridW pads (gridH, gridW the merged grid, at most "
                        f"{tokens} a picture), rewritten to idBase + base_k + i with base_k the earlier images' counts. "
                        "qwen3vl-mrope: with start_k its first pad's position and shift_k the earlier images' amounts, "
                        "imagePositions[base_k + i] = (s0, s0 + i // gridW, s0 + i % gridW) for s0 = start_k - "
                        "shift_k; ropeShiftStart[k] = start_k + count_k; ropeShiftAmount[k] = count_k - max(gridH, "
                        "gridW); a text token at p takes p minus the amounts whose start <= p."),
        },
        "precision": prec,
        "license": "Apache-2.0",
        "x-license": {"files": ["LICENSE", "NOTICE"], "modified": True},
        "provenance": {
            "source": qg.REPO, "sourceRevision": qg.REVISION, "baseModel": qg.PINS.get("baseModel", qg.REPO),
            "producer": "aether-kit Forge/qwen3vl/export_qwen3vl.py",
            "exportScript": report["scriptSHA256"]["export_qwen3vl.py"],
            "refs": {"tokenizer.stop": f"{qg.REPO}@{qg.REVISION} generation_config.json#/eos_token_id",
                     "tokenizer.source": f"{qg.REPO}@{qg.REVISION} " + ", ".join(TOKENIZER_FILES),
                     "tokenizer.chatTemplate": f"{qg.REPO}@{qg.REVISION} " + (
                         "chat_template.jinja" if "chat_template.jinja" in qg.PINS["files"]
                         else "chat_template.json#/chat_template"),
                     "preprocess.image": f"{qg.REPO}@{qg.REVISION} preprocessor_config.json",
                     "forgeLockSHA256": report["forgeLockSHA256"], "exportReport": "export-report.json"},
        },
    }


def precision(vision: str, decoder: str, attention: str, weights: str, image: str = IMAGE_DTYPE,
              head: str = HEAD_COMPUTE) -> dict:
    short = {"float16": "fp16", "float32": "fp32"}
    return {"weights": short[weights], "activations": f"vision {short[vision]}, decoder {short[decoder]}",
            "kv": "fp16" if weights == "float16" else "fp32", "logits": "fp32",
            "components": {
                "vision": (f"{short[weights]} weights, {short[vision]} compute; LayerNorm statistics, rotary and "
                           f"position resampling fp32; outputs {short[image]}"),
                "decoder": (f"{short[weights]} weights and KV, {short[decoder]} residual stream and matmuls, "
                            f"{short[attention]} attention, {short[head]} tied head; RMSNorm statistics and M-RoPE "
                            "angles fp32"),
                "io": (f"patches f32, grid i32; image/deepstack embeds {short[image]}; tokens, positions, "
                       "image_positions and rope shifts i32; logits f32")}}


def sample_manifest(platform_name: str) -> dict:
    names = Bundle(qg.PINS)
    files = {f"{a}/{n}": {"bytes": 1, "sha256": "0" * 64} for a in (names.vision, names.decoder)
             for n in ("main.hash", "main.mlirb", "metadata.json")}
    files.update({f"tokenizer/{n}": {"bytes": 1, "sha256": "0" * 64}
                  for n in (*TOKENIZER_FILES, "chat_template.jinja")})
    files.update({n: {"bytes": 1, "sha256": "0" * 64} for n in ("LICENSE", "NOTICE", "README.md")})
    report = {"scriptSHA256": {"export_qwen3vl.py": "0" * 64}, "forgeLockSHA256": "0" * 64}
    return manifest(files, report, platform_name, precision(VISION_COMPUTE, DECODER_COMPUTE, DECODER_ATTENTION, WEIGHTS))


def notice(weights: str) -> str:
    base = qg.PINS.get("baseModel", qg.REPO)
    origin = ("by the Qwen team, Alibaba Cloud," if base == qg.REPO
              else f"a fine-tune of {base} (by the Qwen team, Alibaba Cloud),")
    return "\n".join([
        f"{Bundle(qg.PINS).title} (Aether-Kit Core AI export)",
        "",
        f"This bundle is a modified version of {qg.REPO} at revision {qg.REVISION},",
        f"{origin} licensed under the Apache License, Version 2.0 (LICENSE).",
        "",
        "Modifications: Aether-Kit's Forge exporter (Forge/qwen3vl/) re-expressed the vision tower as one static Core AI",
        f"graph over a fixed {qg.PATCH_CAPACITY}-patch buffer (the positional tables computed in the graph from the",
        "image's patch grid, padding masked) and the text decoder as a Core AI graph with in-place KV states, image",
        "embeddings injected through extension token ids and 3-axis rotary positions synthesized in the graph; q/k/v and",
        "gate/up projections are concatenated, "
        + ("and the weights were rounded from bfloat16 to float16." if weights == "float16"
           else "and the weights were widened from bfloat16 to float32."),
        "The tokenizer files are upstream's, unmodified; chat_template.jinja is upstream's chat template.",
        "It is not an official release of the Qwen team or Alibaba Cloud.",
        "",
    ])


def readme(platform_names: list[str]) -> str:
    names = Bundle(qg.PINS)
    VISION_ASSET, DECODER_ASSET = names.vision, names.decoder
    return "\n".join([
        "---",
        "license: apache-2.0",
        f"base_model: {qg.REPO}",
        "pipeline_tag: image-text-to-text",
        "tags: [aether-kit, core-ai, vision-language, qwen3-vl]",
        "---",
        "",
        f"# {names.title} for Aether-Kit (Core AI)",
        "",
        f"[{qg.REPO}](https://huggingface.co/{qg.REPO}) at revision `{qg.REVISION[:8]}`, exported by",
        "[Aether-Kit](https://github.com/bubstackdev/aether-kit)'s Forge exporter (`Forge/qwen3vl/`) as two Core AI",
        "graphs:",
        "",
        f"- `{VISION_ASSET}`: the vision tower (24 ViT blocks, 2x2 merger, DeepStack mergers) over a fixed",
        f"  {qg.PATCH_CAPACITY}-patch buffer, so any image up to {qg.PATCH_CAPACITY * 256:,} pixels (sized with",
        f"  upstream's smart resize) runs on one static shape; up to {qg.PATCH_CAPACITY // 4} image tokens.",
        f"- `{DECODER_ASSET}`: the Qwen3 text decoder with in-place KV states; the images' embeddings and DeepStack",
        "  features are static inputs, image tokens are extension ids (`151936 + slot`, packed over a prompt's",
        "  images) and the 3-axis M-RoPE positions are computed in the graph.",
        "",
        "Weights are stored float16. The tower computes in float32, the decoder in float16 with float32 attention.",
        f"Load it with Aether-Kit (`vlm-static-inject@1`, {' and '.join(platform_names)}); the manifest is",
        "`.aether/aether.json`. Gated against the upstream implementation on CPU in float32 (see `export-report.json`",
        "and the check reports).",
        "",
        "Licensed under the Apache License 2.0 (`LICENSE`); a modified version, not an official Qwen release (see",
        "`NOTICE`).",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    qg.pins_argument(parser)
    parser.add_argument("--output", type=Path, help="new bundle directory")
    parser.add_argument("--platform", choices=["macos", "ios"], default="macos")
    parser.add_argument("--sample-manifest", action="store_true")
    parser.add_argument("--snapshot", type=Path, help="the checkpoint snapshot (default: the pins' revision in the "
                        "Hugging Face cache)")
    parser.add_argument("--extras", type=Path, default=Path.home() / ".cache/aether-kit/qwen3vl",
                        help="fetch_upstream.py's output (LICENSE)")
    parser.add_argument("--graphs", default="vision,decoder", help="comma-separated subset (measurement exports)")
    parser.add_argument("--vision-compute", choices=["float16", "float32"], default=VISION_COMPUTE)
    parser.add_argument("--decoder-compute", choices=["float16", "float32"], default=DECODER_COMPUTE)
    parser.add_argument("--decoder-attention", choices=["float16", "float32"], default=DECODER_ATTENTION)
    parser.add_argument("--head-compute", choices=["float16", "float32"], default=HEAD_COMPUTE,
                        help="the tied head's matmul (logits stay float32)")
    parser.add_argument("--weights", choices=["float16", "float32"], default=WEIGHTS)
    parser.add_argument("--image-dtype", choices=["float16", "float32"], default=IMAGE_DTYPE,
                        help="the tower's outputs and the decoder's image statics")
    parser.add_argument("--slots", type=int, default=qg.IMAGE_SLOTS, help="the decoder's image slots N")
    parser.add_argument("--images", type=int, default=qg.MAX_IMAGES, help="the decoder's rope-shift entries M")
    parser.add_argument("--layers", type=int, help="debug: keep only the first N decoder layers")
    parser.add_argument("--creation-date", default=CREATION_DATE)
    args = parser.parse_args()
    qg.use_pins(args.pins)
    args.snapshot = args.snapshot or qg.DEFAULT_SNAPSHOT
    names = Bundle(qg.PINS)
    if args.sample_manifest:
        print(json.dumps(sample_manifest(args.platform), indent=2))
        return
    if args.output is None:
        parser.error("--output is required")
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    graphs = set(args.graphs.split(","))
    if not graphs <= {"vision", "decoder"}:
        parser.error("--graphs takes vision and/or decoder")
    creation_date = int(dt.datetime.fromisoformat(args.creation_date).timestamp())
    started = time.monotonic()
    args.output.mkdir(parents=True)
    log = Log(args.output / "forge-log.jsonl")
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    weights = qg.DTYPES[args.weights]
    image_dtype = qg.DTYPES[args.image_dtype]
    prec = precision(args.vision_compute, args.decoder_compute, args.decoder_attention, args.weights, args.image_dtype,
                     args.head_compute)
    report = {
        "schema": 1, "id": names.id, "command": sys.argv, "pins": args.pins.name,
        "exportedAt": dt.datetime.now(dt.timezone.utc).isoformat(), "creationDate": args.creation_date,
        "host": {"os": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
                 "machine": platform.machine(), "python": platform.python_version()},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "coreai-torch", "coreai-core", "numpy",
                                                                "safetensors", "transformers", "tokenizers")},
        "forgeLockSHA256": qg.sha256_file(HERE.parent / "uv.lock"),
        "forgePyprojectSHA256": qg.sha256_file(HERE.parent / "pyproject.toml"),
        "scriptSHA256": {p.name: qg.sha256_file(p) for p in sorted(HERE.glob("*.py"))},
        "source": {"repository": qg.REPO, "revision": qg.REVISION, "snapshot": str(args.snapshot.resolve()),
                   "verified": {}},
        "graphs": sorted(graphs), "precision": prec,
        "precisionArgs": {"vision": args.vision_compute, "decoder": args.decoder_compute,
                          "attention": args.decoder_attention, "head": args.head_compute, "weights": args.weights},
        "patchCapacity": qg.PATCH_CAPACITY, "prefillMax": qg.PREFILL_MAX, "contextMax": qg.CONTEXT_MAX,
        "imageSlots": args.slots, "maxImages": args.images, "imageDType": args.image_dtype,
        "compute": COMPUTE, "conversion": {}, "stages": {},
    }
    report_path = args.output / "export-report.json"

    def save() -> None:
        report["peakRSSGB"] = peak_rss_gb()
        report["wallSeconds"] = round(time.monotonic() - started, 1)
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    log.start("verify-sources")
    report["source"]["verified"] = qg.verify_sources(args.snapshot)
    license_path = args.extras / "LICENSE"
    lic_pin = qg.PINS["upstreamLicense"]["files"]["LICENSE"]
    if qg.sha256_file(license_path) != lic_pin:
        raise SystemExit("LICENSE differs from its pin; run Forge/qwen3vl/fetch_upstream.py")
    report["source"]["verified"]["upstreamLicense:LICENSE"] = lic_pin
    report["stages"]["verify-sources"] = log.done("verify-sources")
    save()

    log.start("load-upstream")
    model = qg.upstream_model(args.snapshot)
    vocab = model.config.text_config.vocab_size
    hidden = model.config.text_config.hidden_size
    report["model"] = {"vocab": vocab, "hidden": hidden, "layers": model.config.text_config.num_hidden_layers,
                       "kvHeads": model.config.text_config.num_key_value_heads,
                       "headDim": model.config.text_config.head_dim,
                       "visionDepth": model.config.vision_config.depth}
    if args.layers:
        del model.model.language_model.layers[args.layers:]
        report["debugLayers"] = args.layers
    report["stages"]["load-upstream"] = log.done("load-upstream")

    if "vision" in graphs:
        log.start("export-vision")
        tower = qg.VisionTower(model.model.visual, dtype=qg.DTYPES[args.vision_compute], storage=weights,
                               output=image_dtype).eval()
        t = qg.PATCH_CAPACITY // 4
        description = (f"{names.title} vision tower ({qg.REPO}@{qg.REVISION[:12]}): patches f32 [{qg.PATCH_CAPACITY},"
                       f"1536] (block-major, zero-padded), grid i32 [2] -> image_embeds [{t},{hidden}], "
                       f"deepstack_embeds [{3 * t},{hidden}] ({args.image_dtype}).")
        report["conversion"]["vision"] = convert(tower, tower.example(), None, args.output / names.vision,
                                                 creation_date, VISION_INPUTS, VISION_OUTPUTS, [], description)
        del tower
        gc.collect()
        report["stages"]["export-vision"] = log.done("export-vision")
        save()

    if "decoder" in graphs:
        log.start("export-decoder")
        decoder = qg.Decoder(model, slots=args.slots, images=args.images, dtype=qg.DTYPES[args.decoder_compute],
                             attention=qg.DTYPES[args.decoder_attention], storage=weights, statics=image_dtype,
                             head=qg.DTYPES[args.head_compute]).eval()
        del model
        gc.collect()
        example, dynamic = decoder.example()
        description = (f"{names.title} text decoder ({qg.REPO}@{qg.REVISION[:12]}): input_ids, position_ids i32 with "
                       f"static image inputs ({args.slots} slots, {args.images} images) and in-place keyCache/valueCache "
                       f"states -> last-row logits f32 [1,1,{vocab}].")
        report["conversion"]["decoder"] = convert(decoder, example, dynamic, args.output / names.decoder,
                                                  creation_date, DECODER_INPUTS, DECODER_OUTPUTS, DECODER_STATES,
                                                  description)
        del decoder
        gc.collect()
        report["stages"]["export-decoder"] = log.done("export-decoder")
        save()
    else:
        del model

    log.start("package")
    (args.output / "tokenizer").mkdir()
    for name in TOKENIZER_FILES:
        shutil.copyfile(args.snapshot / name, args.output / "tokenizer" / name)
    (args.output / "tokenizer/chat_template.jinja").write_text(qg.chat_template(args.snapshot))
    shutil.copyfile(license_path, args.output / "LICENSE")
    (args.output / "NOTICE").write_text(notice(args.weights))
    (args.output / "README.md").write_text(readme(["macOS"]))
    files = {str(p.relative_to(args.output)): {"bytes": p.stat().st_size, "sha256": qg.sha256_file(p)}
             for p in tree_files(args.output) if str(p.relative_to(args.output)) not in NOT_BUNDLE}
    report["bundleBytes"] = sum(f["bytes"] for f in files.values())
    report["assetBytes"] = {name: tree_bytes(args.output / asset) for name, asset in
                            (("vision", names.vision), ("decoder", names.decoder), ("tokenizer", "tokenizer"))
                            if (args.output / asset).exists()}
    report["stages"]["package"] = log.done("package")
    save()
    (args.output / ".aether").mkdir()
    manifest_path = args.output / ".aether/aether.json"
    doc = manifest(files, report, args.platform, prec,
                   template_sha=qg.sha256_file(args.output / "tokenizer/chat_template.jinja"), slots=args.slots,
                   images=args.images, vocab=vocab)
    manifest_path.write_text(json.dumps(doc, indent=2) + "\n")
    report["aetherJSONSHA256"] = qg.sha256_file(manifest_path)
    save()
    sums = [f"{qg.sha256_file(p)}  {p.relative_to(args.output)}" for p in tree_files(args.output)
            if p.name not in ("SHA256SUMS", "forge-log.jsonl")]
    (args.output / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(json.dumps({"conversion": report["conversion"], "bundleBytes": report["bundleBytes"],
                      "peakRSSGB": report["peakRSSGB"], "wallSeconds": report["wallSeconds"]}, indent=1))


if __name__ == "__main__":
    main()

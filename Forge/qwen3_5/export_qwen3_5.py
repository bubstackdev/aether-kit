#!/usr/bin/env python3
"""Export a Qwen3.5 / Qwen3.6 checkpoint (Qwen3_5ForConditionalGeneration, Apache-2.0) as an Aether-Kit
`vlm-static-inject@1` bundle whose decoder is a hybrid with two functions over recurrent states.

    uv sync --frozen --project Forge
    scripts/gpu-lease -- Forge/.venv/bin/python -I Forge/qwen3_5/export_qwen3_5.py \
        --pins Forge/qwen3_5/source-pins-27b-abliterated.json --output <new bundle dir>
    Forge/.venv/bin/python -I Forge/qwen3_5/export_qwen3_5.py --pins <pins> --sample-manifest

Nothing is downloaded: the checkpoint is read from the local Hugging Face cache at the pins' revision and every file
read is verified against its SHA-256 (the pins). Every size comes from the checkpoint's config. The output directory is
laid out as an installed revision root (`<prefix>` from the pins):

    OUTPUT/
      .aether/aether.json          the Aether manifest (MANIFEST.md §1.2 vlm-static-inject@1, §2.3 functions, §2.3.1
                                   recurrent states)
      <prefix>_vision.aimodel      vision-tower@1: patches f32 [2048,1536], grid i32 [2] -> image_embeds f16 [512,H]
      <prefix>_decoder.aimodel     functions `main` (S=1) and `prefill` (S=64) sharing keyCache/valueCache (sequence)
                                   and convState/recState (recurrent); static image inputs; logits f32 [1,1,vocab]
      tokenizer/                   upstream's tokenizer.json, tokenizer_config.json, vocab.json, merges.txt,
                                   chat_template.jinja
      LICENSE, NOTICE, README.md, SHA256SUMS, export-report.json, forge-log.jsonl

The graphs and their contracts are documented in qwen3_5_graphs.py. A 27B checkpoint is 54 GB in bfloat16, so the
upstream model is never built whole: its structure is built on the meta device, and each decoder layer's weights are
read from the snapshot's shards, turned into the export modules, quantized Linear by Linear (`--weight-bits`, int4 in
blocks of 32 by default: coreai-opt's weight-only symmetric PTQ, Forge's shieldstral recipe) and released before the
next layer is read. Reproducible as Forge/qwen3vl's exporter is (seeded generated names, fixed creation date,
deterministic Swift hashing), except the graphs' weight-blob names.
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
import re  # noqa: E402
import resource  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qwen3_5_graphs as qg  # noqa: E402

VISION_INPUTS, VISION_OUTPUTS = ["patches", "grid"], ["image_embeds"]
DECODER_INPUTS = ["input_ids", "position_ids", "image_embeds", "image_positions", "rope_shift_start",
                  "rope_shift_amount"]
SEQUENCE_STATES, RECURRENT_STATES = ["keyCache", "valueCache"], ["convState", "recState"]
DECODER_STATES, DECODER_OUTPUTS = SEQUENCE_STATES + RECURRENT_STATES, ["logits"]
FUNCTIONS = {"step": "main", "prefill": "prefill"}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
NOT_BUNDLE = {".aether/aether.json", "SHA256SUMS", "export-report.json", "forge-log.jsonl"}
COMPUTE = "gpu"
VISION_COMPUTE = "float32"     # as Forge/qwen3vl: the tower is small, float32 compute costs little
DECODER_COMPUTE = "float16"
DECODER_ATTENTION = "float32"
WEIGHT_BITS = 4                # decoder Linears: int4 block 32 (13.7 GB at 27B; the 2026-11-10 spike's recipe)
HEAD_BITS = 4
CREATION_DATE = "2026-11-10T00:00:00+00:00"


class Log:
    def __init__(self, path: Path) -> None:
        self.path, self.started = path, {}

    def event(self, stage: str, message: str, **fields) -> None:
        record = {"message": message, "stage": stage, "t": round(time.time(), 3), **fields}
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"[qwen3_5:{stage}] {message}" + (f" {fields}" if fields else ""), file=sys.stderr, flush=True)

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
        self.id, self.title = b["id"], b["title"]
        self.vision, self.decoder = f"{b['assetPrefix']}_vision.aimodel", f"{b['assetPrefix']}_decoder.aimodel"


def peak_rss_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)


def tree_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def tree_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in tree_files(root))


def verify_sources(snapshot: Path) -> dict:
    verified = {}
    for relative, expected in qg.PINS["files"].items():
        path = snapshot / relative
        if not path.is_file():
            raise SystemExit(f"snapshot lacks {relative} ({qg.REPO}@{qg.REVISION[:12]})")
        if qg.sha256_file(path) != expected:
            raise SystemExit(f"{relative} differs from the pinned {qg.REVISION[:12]} file")
        verified[relative] = expected
    return verified


# MARK: the checkpoint, a layer at a time

class Checkpoint:
    """The snapshot's tensors by upstream name, read on demand from the safetensors shards."""

    def __init__(self, snapshot: Path) -> None:
        from safetensors import safe_open
        self.snapshot = snapshot
        self.index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        self.handles = {f: safe_open(str(snapshot / f), framework="pt") for f in sorted(set(self.index.values()))}

    def tensor(self, name: str) -> torch.Tensor:
        return self.handles[self.index[name]].get_tensor(name)

    def load(self, module: nn.Module, prefix: str, dtype: torch.dtype) -> None:
        """Fills `module`'s parameters (on the meta device) from the shards, in `dtype`."""
        names = [n for n, _ in module.named_parameters()]
        state = {n: self.tensor(prefix + n).to(dtype) for n in names}
        module.load_state_dict(state, strict=False, assign=True)
        missing = [n for n, p in module.named_parameters() if p.is_meta]
        if missing:
            raise SystemExit(f"{prefix}: no weights for {missing[:6]}")

    def release(self, module: nn.Module) -> None:
        module.to_empty(device="meta")


def upstream_shell(snapshot: Path, layers: int | None):
    """transformers' Qwen3_5ForConditionalGeneration on the meta device (no weights), with real rotary buffers."""
    from transformers import AutoConfig, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5.modeling_qwen3_5 import (Qwen3_5TextRotaryEmbedding,
                                                              Qwen3_5VisionRotaryEmbedding)
    config = AutoConfig.from_pretrained(str(snapshot))
    with torch.device("meta"):
        model = Qwen3_5ForConditionalGeneration(config)
    if layers:
        del model.model.language_model.layers[layers:]
    model.model.language_model.rotary_emb = Qwen3_5TextRotaryEmbedding(config.text_config)
    model.model.visual.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(config.vision_config)
    return model.eval()


def quantize_linears(module: nn.Module, bits: int, counts: dict) -> None:
    """Replaces every nn.Linear child of `module` (recursively) with coreai-opt's weight-only symmetric quantization
    of it: int4 in blocks of 32 along the input axis, or int8 per output channel. One Linear at a time, so only its own
    float16 weight is ever held beside the result."""
    from coreai_opt.quantization import Quantizer
    from coreai_opt.quantization.config import ModuleQuantizerConfig, QuantizerConfig
    from coreai_opt.quantization.spec.granularity import PerBlockGranularity, PerChannelGranularity
    from coreai_opt.quantization.spec.qscheme import QuantizationScheme
    from coreai_opt.quantization.spec.spec import QuantizationSpec
    if bits >= 16:
        return
    spec = QuantizationSpec(dtype=torch.int4 if bits == 4 else torch.int8, qscheme=QuantizationScheme.SYMMETRIC,
                            granularity=PerBlockGranularity(axis=1, block_size=32) if bits == 4
                            else PerChannelGranularity(axis=None))
    config = ModuleQuantizerConfig(op_input_spec=None, op_output_spec=None, op_state_spec={"weight": spec})
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            wrapper = nn.Sequential(child)
            quantizer = Quantizer(wrapper, QuantizerConfig(global_config=None, module_name_configs={"^0$": config},
                                                           execution_mode="eager"))
            example = torch.zeros(1, child.in_features, dtype=child.weight.dtype)
            prepared = quantizer.prepare((example,))
            setattr(module, name, quantizer.finalize(prepared)[0])
            counts[f"int{bits}"] = counts.get(f"int{bits}", 0) + 1
        else:
            quantize_linears(child, bits, counts)


# MARK: conversion

@contextlib.contextmanager
def deterministic_conversion(creation_date: int):
    """Seeded generated names and a fixed creation stamp (as Forge/qwen3vl and Forge/embeddinggemma)."""
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
    """Removes the same-shape `aten.copy` nodes each in-place state write leaves (as Forge/qwen3vl)."""
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


def convert(functions: dict, path: Path, creation_date: int, inputs: list[str], outputs: list[str],
            states: list[str], description: str) -> dict:
    """`functions`: {entry point name: (module, example, dynamic shapes)}, converted into one asset that shares the
    weights and states."""
    import coreai_torch as ct
    from coreai.runtime import AIModelAssetMetadata

    report = {"inputs": inputs, "outputs": outputs, "states": states, "functions": {}}
    with deterministic_conversion(creation_date):
        converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
        for name, (module, example, dynamic) in functions.items():
            t0 = time.monotonic()
            with torch.no_grad():
                ep = torch.export.export(module, example, dynamic_shapes=dynamic, strict=False)
                ep = ep.run_decompositions(ct.get_decomp_table())
            mutated = len(ep.graph_signature.user_inputs_to_mutate)
            if mutated != len(states):
                raise SystemExit(f"{name}: {mutated} mutated inputs for the {len(states)} states {states}")
            dropped = drop_identity_copies(ep) if states else 0
            converter.add_exported_program(ep, input_names=inputs, output_names=outputs, state_names=states,
                                           entrypoint_name=name)
            del ep
            gc.collect()
            report["functions"][name] = {"exportSeconds": round(time.monotonic() - t0, 1),
                                         "identityCopiesDropped": dropped, "peakRSSGB": peak_rss_gb()}
        t1 = time.monotonic()
        program = converter.to_coreai()
        del converter
        gc.collect()
        program.optimize()
        metadata = AIModelAssetMetadata()
        metadata.author = "Alibaba Qwen (Qwen3.5/3.6); Core AI export: Aether-Kit Forge"
        metadata.license = "Apache-2.0"
        metadata.model_description = description
        metadata.creation_date = creation_date
        program.save_asset(path, metadata)
    report.update({"convertSeconds": round(time.monotonic() - t1, 1), "bytes": tree_bytes(path)})
    return report


# MARK: the manifest (MANIFEST.md §1.2 vlm-static-inject@1, §2.3 graphs and functions, §2.6 image)

def manifest(files: dict[str, dict], report: dict, platform_name: str, prec: dict, cfg: dict,
             template_sha: str = "0" * 64, slots: int = qg.IMAGE_SLOTS, images: int = qg.MAX_IMAGES) -> dict:
    capacity = qg.PATCH_CAPACITY
    names = Bundle(qg.PINS)
    vocab = cfg["text_config"]["vocab_size"]
    pad, start, end = cfg["image_token_id"], cfg["vision_start_token_id"], cfg["vision_end_token_id"]
    stop = qg.PINS["stop"]
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
                "file": f"main:{names.vision}", "function": "main",
                "inputs": {"patches": "patches", "grid": "grid"},
                "outputs": {"imageEmbeds": "image_embeds"},
                "compute": COMPUTE, "expectFrequentReshapes": False,
                "x-notes": (f"One static shape for every image: patches [{capacity},1536] holds the image's h*w "
                            "block-major rows first and zeros after; grid int32 [2] is (h, w) in patches (both even, "
                            f"h*w <= {capacity}). Padding rows are masked as keys and the output rows past h*w/4 are "
                            "zero. Qwen3-VL's tower without DeepStack (Qwen3.5 has none)."),
            },
            "decoder": {
                "file": f"main:{names.decoder}",
                "functions": dict(FUNCTIONS),
                "inputs": {
                    "tokens": "input_ids", "positions": "position_ids",
                    "imageEmbeds": {"name": "image_embeds", "binding": "static"},
                    "imagePositions": {"name": "image_positions", "binding": "static"},
                    "ropeShiftStart": {"name": "rope_shift_start", "binding": "static"},
                    "ropeShiftAmount": {"name": "rope_shift_amount", "binding": "static"},
                },
                "inputEncodings": {"tokens": {"extensionIds": {"base": vocab, "count": slots}}},
                "outputs": {"logits": {"name": "logits", "rows": "last"}},
                "states": {
                    "sequence": [{"tensors": SEQUENCE_STATES, "kind": "kv", "seqAxis": 3}],
                    "recurrent": [{"name": "convState", "kind": "conv", "init": "zeros"},
                                  {"name": "recState", "kind": "ssm", "init": "zeros"}],
                },
                "limits": {"maxContext": qg.CONTEXT_MAX, "staticSequence": {"step": 1, "prefill": qg.PREFILL},
                           "prefillChunk": qg.PREFILL, "padTail": False},
                "compute": COMPUTE, "expectFrequentReshapes": True,
                "x-notes": ("Running contract with a static token axis: `main` takes input_ids [1,1], `prefill` "
                            f"[1,{qg.PREFILL}] (the remainder of a prompt goes through `main`); position_ids [1,T] = "
                            "0..T-1, the last S the fed tokens'; keyCache/valueCache (the full-attention layers) "
                            "written in place at rows T-S..T-1; convState (the last kernel-1 inputs of each Gated "
                            "DeltaNet's causal conv) and recState (its float32 delta-rule state) replaced on every "
                            "call, so a rewind restores a snapshot. logits are the last fed row. Image tokens are "
                            f"extension ids vocab + slot ({slots} slots, {images} images), the static inputs hold each "
                            "slot's rows and rotary positions and each image's rope shift (image.rope)."),
            },
        },
        "tokenizer": {
            "source": {"dir": "main:tokenizer"},
            "chatTemplate": "main:tokenizer/chat_template.jinja",
            "chatTemplateSha256": template_sha,
            "stop": [stop[k] for k in sorted(stop, key=int, reverse=True)],
            "thinking": {"open": "<think>", "close": "</think>", "toggle": {"templateVar": "enable_thinking"},
                         "defaultOn": False},
            "toolDialect": "hermes-json",
        },
        "preprocess": {
            "image": {
                "resize": {"mode": "smartResize", "filter": "torchUint8Bicubic", "factor": 32, "minPixels": 65536,
                           "maxPixels": capacity * 256},
                "colorSpace": "deviceRGB", "channelOrder": "RGB", "scale": 1 / 255,
                "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
                "layout": "patches",
                "patchify": {"patch": 16, "temporal": 2, "merge": 2, "rowOrder": "blockMajor",
                             "withinPatch": "C,T,py,px"},
                "hostDType": "float32",
                "x-notes": ("Upstream's processor (Qwen2VLImageProcessorFast, as Qwen3-VL's): smart_resize to multiples "
                            "of 32 within the pixel bounds, then a resize to exactly that size (torchvision uint8 "
                            f"bicubic, antialias). maxPixels is the tower's capacity ({capacity} patches); upstream's own "
                            "longest_edge is 16777216 pixels, the bundle's budget is the graph's."),
            },
        },
        "image": {
            "feed": [{"from": "vision.imageEmbeds", "to": "decoder.imageEmbeds"}],
            "tokens": {"pad": "<|image_pad|>", "padId": pad, "open": "<|vision_start|>", "openId": start,
                       "close": "<|vision_end|>", "closeId": end, "count": "gridH*gridW", "idBase": vocab},
            "rope": {"rule": "qwen3vl-mrope",
                     "textOnly": {"start": qg.TEXT_ONLY_SHIFT_START, "amount": 0, "zeroStatics": True}},
            "maxImages": images,
            "slots": slots,
            "x-notes": ("As Qwen3-VL's (Forge/qwen3vl, docs/VLM.md §1): image k's pad expands to count_k = "
                        f"gridH*gridW pads (the merged grid, at most {capacity // 4} a picture) rewritten to idBase + "
                        "base_k + i; imagePositions and the rope shifts by the qwen3vl-mrope rule (upstream's "
                        "get_rope_index, the same for Qwen3.5's images)."),
        },
        "precision": prec,
        "license": "Apache-2.0",
        "x-license": {"files": ["LICENSE", "NOTICE"], "modified": True},
        "provenance": {
            "source": qg.REPO, "sourceRevision": qg.REVISION, "baseModel": qg.PINS.get("baseModel", qg.REPO),
            "producer": "aether-kit Forge/qwen3_5/export_qwen3_5.py",
            "exportScript": report["scriptSHA256"]["export_qwen3_5.py"],
            "refs": {"tokenizer.stop": f"{qg.REPO}@{qg.REVISION} generation_config.json#/eos_token_id",
                     "tokenizer.source": f"{qg.REPO}@{qg.REVISION} " + ", ".join(TOKENIZER_FILES),
                     "tokenizer.chatTemplate": f"{qg.REPO}@{qg.REVISION} chat_template.jinja",
                     "preprocess.image": f"{qg.REPO}@{qg.REVISION} preprocessor_config.json",
                     "forgeLockSHA256": report["forgeLockSHA256"], "exportReport": "export-report.json"},
        },
    }


def precision(weight_bits: int, head_bits: int) -> dict:
    w = {4: "int4 block 32 (symmetric)", 8: "int8 per channel (symmetric)", 16: "fp16"}
    return {"weights": f"decoder {w[weight_bits]}, head {w[head_bits]}, embedding and vision fp16",
            "activations": f"vision {VISION_COMPUTE}, decoder {DECODER_COMPUTE}", "kv": "fp16", "logits": "fp32",
            "components": {
                "vision": "fp16 weights, fp32 compute; LayerNorm statistics, rotary and position resampling fp32; "
                          "outputs fp16",
                "decoder": (f"{w[weight_bits]} Linears, fp16 embedding, residual stream and KV, {DECODER_ATTENTION} "
                            "attention; RMSNorm statistics, M-RoPE angles, the Gated DeltaNet scan and its state fp32; "
                            f"head {w[head_bits]}"),
                "io": "patches f32, grid i32; image embeds fp16; tokens, positions, image_positions and rope shifts "
                      "i32; logits f32"}}


def sample_manifest(platform_name: str, cfg: dict) -> dict:
    names = Bundle(qg.PINS)
    files = {f"{a}/{n}": {"bytes": 1, "sha256": "0" * 64} for a in (names.vision, names.decoder)
             for n in ("main.hash", "main.mlirb", "metadata.json")}
    files.update({f"tokenizer/{n}": {"bytes": 1, "sha256": "0" * 64}
                  for n in (*TOKENIZER_FILES, "chat_template.jinja")})
    files.update({n: {"bytes": 1, "sha256": "0" * 64} for n in ("LICENSE", "NOTICE", "README.md")})
    report = {"scriptSHA256": {"export_qwen3_5.py": "0" * 64}, "forgeLockSHA256": "0" * 64}
    return manifest(files, report, platform_name, precision(WEIGHT_BITS, HEAD_BITS), cfg)


def notice(weight_bits: int) -> str:
    base = qg.PINS.get("baseModel", qg.REPO)
    origin = ("by the Qwen team, Alibaba Cloud," if base == qg.REPO
              else f"a fine-tune of {base} (by the Qwen team, Alibaba Cloud),")
    return "\n".join([
        f"{Bundle(qg.PINS).title} (Aether-Kit Core AI export)",
        "",
        f"This bundle is a modified version of {qg.REPO} at revision {qg.REVISION},",
        f"{origin} licensed under the Apache License, Version 2.0 (LICENSE).",
        "",
        "Modifications: Aether-Kit's Forge exporter (Forge/qwen3_5/) re-expressed the vision tower as one static Core AI",
        f"graph over a fixed {qg.PATCH_CAPACITY}-patch buffer and the hybrid text decoder as a Core AI graph with two",
        "functions (one token, and 64-token prompt blocks) over in-place KV and recurrent states, image embeddings",
        "injected through extension token ids and 3-axis rotary positions synthesized in the graph; "
        + (f"the decoder's linear weights were quantized to {weight_bits} bits and the remaining weights rounded from "
           "bfloat16 to float16." if weight_bits < 16 else "the weights were rounded from bfloat16 to float16."),
        "The tokenizer files are upstream's, unmodified; chat_template.jinja is upstream's chat template.",
        "It is not an official release of the Qwen team or Alibaba Cloud.",
        "",
    ])


def readme(cfg: dict, weight_bits: int) -> str:
    names = Bundle(qg.PINS)
    text, vision = cfg["text_config"], cfg["vision_config"]
    return "\n".join([
        "---",
        "license: apache-2.0",
        f"base_model: {qg.REPO}",
        "pipeline_tag: image-text-to-text",
        "tags: [aether-kit, core-ai, vision-language, qwen3.5]",
        "---",
        "",
        f"# {names.title} for Aether-Kit (Core AI)",
        "",
        f"[{qg.REPO}](https://huggingface.co/{qg.REPO}) at revision `{qg.REVISION[:8]}`, exported by",
        "[Aether-Kit](https://github.com/bubstackdev/aether-kit)'s Forge exporter (`Forge/qwen3_5/`) as two Core AI",
        "graphs:",
        "",
        f"- `{names.vision}`: the vision tower ({vision['depth']} ViT blocks, 2x2 merger) over a fixed",
        f"  {qg.PATCH_CAPACITY}-patch buffer, any image up to {qg.PATCH_CAPACITY * 256:,} pixels; up to",
        f"  {qg.PATCH_CAPACITY // 4} image tokens.",
        f"- `{names.decoder}`: the hybrid decoder ({text['num_hidden_layers']} layers, three Gated DeltaNet to each",
        "  gated full-attention layer) with two functions over the same states: `main` (one token) and `prefill`",
        f"  ({qg.PREFILL}-token blocks); image tokens are extension ids, M-RoPE positions computed in the graph.",
        "",
        (f"Decoder Linears int{weight_bits} (blocks of 32), " if weight_bits < 16 else "Weights float16, ")
        + "float16 compute with float32 attention, norms, rotary and recurrence. Load it with Aether-Kit "
        "(`vlm-static-inject@1`, macOS); the manifest is `.aether/aether.json`.",
        "",
        "Licensed under the Apache License 2.0 (`LICENSE`); a modified version, not an official Qwen release (see",
        "`NOTICE`).",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pins", type=Path, required=True, help="the source checkpoint's pins (source-pins-*.json)")
    parser.add_argument("--output", type=Path, help="new bundle directory")
    parser.add_argument("--platform", choices=["macos"], default="macos")
    parser.add_argument("--sample-manifest", action="store_true")
    parser.add_argument("--snapshot", type=Path, help="the checkpoint snapshot (default: the pins' revision in the "
                        "Hugging Face cache)")
    parser.add_argument("--graphs", default="vision,decoder", help="comma-separated subset (measurement exports)")
    parser.add_argument("--weight-bits", type=int, choices=[4, 8, 16], default=WEIGHT_BITS)
    parser.add_argument("--head-bits", type=int, choices=[4, 8, 16], default=HEAD_BITS)
    parser.add_argument("--slots", type=int, default=qg.IMAGE_SLOTS)
    parser.add_argument("--images", type=int, default=qg.MAX_IMAGES)
    parser.add_argument("--layers", type=int, help="debug: keep only the first N decoder layers")
    parser.add_argument("--creation-date", default=CREATION_DATE)
    args = parser.parse_args()
    qg.use_pins(args.pins)
    args.snapshot = args.snapshot or qg.DEFAULT_SNAPSHOT
    cfg = json.loads((args.snapshot / "config.json").read_text())
    names = Bundle(qg.PINS)
    if args.sample_manifest:
        print(json.dumps(sample_manifest(args.platform, cfg), indent=2))
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
    prec = precision(args.weight_bits, args.head_bits)
    report = {
        "schema": 1, "id": names.id, "command": sys.argv, "pins": args.pins.name,
        "exportedAt": dt.datetime.now(dt.timezone.utc).isoformat(), "creationDate": args.creation_date,
        "host": {"os": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
                 "machine": platform.machine(), "python": platform.python_version()},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "coreai-torch", "coreai-core", "coreai-opt",
                                                                "numpy", "safetensors", "transformers", "tokenizers")},
        "forgeLockSHA256": qg.sha256_file(HERE.parent / "uv.lock"),
        "forgePyprojectSHA256": qg.sha256_file(HERE.parent / "pyproject.toml"),
        "scriptSHA256": {p.name: qg.sha256_file(p) for p in sorted(HERE.glob("*.py"))},
        "source": {"repository": qg.REPO, "revision": qg.REVISION, "snapshot": str(args.snapshot.resolve()),
                   "verified": {}},
        "graphs": sorted(graphs), "precision": prec, "weightBits": args.weight_bits, "headBits": args.head_bits,
        "patchCapacity": qg.PATCH_CAPACITY, "prefill": qg.PREFILL, "scanPiece": qg.SCAN_PIECE,
        "contextMax": qg.CONTEXT_MAX, "imageSlots": args.slots, "maxImages": args.images,
        "compute": COMPUTE, "conversion": {}, "stages": {}, "quantized": {},
    }
    report_path = args.output / "export-report.json"

    def save() -> None:
        report["peakRSSGB"] = peak_rss_gb()
        report["wallSeconds"] = round(time.monotonic() - started, 1)
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    log.start("verify-sources")
    report["source"]["verified"] = verify_sources(args.snapshot)
    report["stages"]["verify-sources"] = log.done("verify-sources")
    save()

    model = upstream_shell(args.snapshot, args.layers)
    checkpoint = Checkpoint(args.snapshot)
    text = model.config.text_config
    report["model"] = {"vocab": text.vocab_size, "hidden": text.hidden_size, "layers": len(model.model.language_model.layers),
                       "kinds": "".join("a" if k == "full_attention" else "l"
                                        for k in text.layer_types[:len(model.model.language_model.layers)]),
                       "kvHeads": text.num_key_value_heads, "headDim": text.head_dim,
                       "visionDepth": model.config.vision_config.depth}
    if args.layers:
        report["debugLayers"] = args.layers

    if "vision" in graphs:
        log.start("export-vision")
        checkpoint.load(model.model.visual, "model.visual.", torch.float32)
        tower = qg.VisionTower(model.model.visual, dtype=qg_dtype(VISION_COMPUTE), storage=torch.float16,
                               output=torch.float16).eval()
        checkpoint.release(model.model.visual)
        t = qg.PATCH_CAPACITY // 4
        description = (f"{names.title} vision tower ({qg.REPO}@{qg.REVISION[:12]}): patches f32 "
                       f"[{qg.PATCH_CAPACITY},1536] (block-major, zero-padded), grid i32 [2] -> image_embeds "
                       f"[{t},{text.hidden_size}] fp16.")
        report["conversion"]["vision"] = convert({"main": (tower, tower.example(), None)}, args.output / names.vision,
                                                 creation_date, VISION_INPUTS, VISION_OUTPUTS, [], description)
        del tower
        gc.collect()
        report["stages"]["export-vision"] = log.done("export-vision")
        save()

    if "decoder" in graphs:
        log.start("build-decoder")
        lm = model.model.language_model
        checkpoint.load(lm.embed_tokens, "model.language_model.embed_tokens.", torch.float16)
        checkpoint.load(lm.norm, "model.language_model.norm.", torch.float32)
        checkpoint.load(model.lm_head, "lm_head.", torch.float16)

        def before(i, layer):
            checkpoint.load(layer, f"model.language_model.layers.{i}.", torch.float16)

        def after(i, mixer, mlp):
            quantize_linears(mixer, args.weight_bits, report["quantized"])
            quantize_linears(mlp, args.weight_bits, report["quantized"])
            checkpoint.release(lm.layers[i])
            if i % 8 == 7:
                gc.collect()
                log.event("build-decoder", f"layer {i + 1} of {len(lm.layers)}", peakRSSGB=peak_rss_gb())

        decoder = qg.Decoder(model, slots=args.slots, images=args.images, dtype=qg_dtype(DECODER_COMPUTE),
                             attention=qg_dtype(DECODER_ATTENTION), storage=torch.float16, before_layer=before,
                             after_layer=after).eval()
        head_counts: dict = {}
        holder = nn.Module()
        holder.head = decoder.head
        quantize_linears(holder, args.head_bits, head_counts)
        decoder.head = holder.head
        report["quantized"]["head"] = head_counts
        checkpoint.release(lm.embed_tokens)
        checkpoint.release(model.lm_head)
        del model, checkpoint
        gc.collect()
        report["stages"]["build-decoder"] = log.done("build-decoder")
        save()
        log.start("export-decoder")
        functions = {"main": (decoder, *decoder.example(1)), "prefill": (decoder, *decoder.example(qg.PREFILL))}
        description = (f"{names.title} hybrid decoder ({qg.REPO}@{qg.REVISION[:12]}): main (1 token) and prefill "
                       f"({qg.PREFILL} tokens), input_ids and position_ids i32 with static image inputs ({args.slots} "
                       f"slots, {args.images} images), keyCache/valueCache and convState/recState states -> last-row "
                       f"logits f32 [1,1,{text.vocab_size}].")
        report["conversion"]["decoder"] = convert(functions, args.output / names.decoder, creation_date,
                                                  DECODER_INPUTS, DECODER_OUTPUTS, DECODER_STATES, description)
        del decoder, functions
        gc.collect()
        report["stages"]["export-decoder"] = log.done("export-decoder")
        save()
    else:
        del model, checkpoint

    log.start("package")
    (args.output / "tokenizer").mkdir()
    for name in TOKENIZER_FILES:
        shutil.copyfile(args.snapshot / name, args.output / "tokenizer" / name)
    shutil.copyfile(args.snapshot / "chat_template.jinja", args.output / "tokenizer/chat_template.jinja")
    shutil.copyfile(args.snapshot / "LICENSE", args.output / "LICENSE")
    (args.output / "NOTICE").write_text(notice(args.weight_bits))
    (args.output / "README.md").write_text(readme(cfg, args.weight_bits))
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
    doc = manifest(files, report, args.platform, prec, cfg,
                   template_sha=qg.sha256_file(args.output / "tokenizer/chat_template.jinja"), slots=args.slots,
                   images=args.images)
    manifest_path.write_text(json.dumps(doc, indent=2) + "\n")
    report["aetherJSONSHA256"] = qg.sha256_file(manifest_path)
    save()
    sums = [f"{qg.sha256_file(p)}  {p.relative_to(args.output)}" for p in tree_files(args.output)
            if p.name not in ("SHA256SUMS", "forge-log.jsonl")]
    (args.output / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(json.dumps({"conversion": report["conversion"], "bundleBytes": report["bundleBytes"],
                      "peakRSSGB": report["peakRSSGB"], "wallSeconds": report["wallSeconds"]}, indent=1))


def qg_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "float32": torch.float32}[name]


if __name__ == "__main__":
    main()

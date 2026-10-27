#!/usr/bin/env python3
"""Export FLUX.2 [klein] 4B (black-forest-labs/FLUX.2-klein-4B, Apache-2.0) for AetherFlux2 (development tooling only).

Runs in Forge's locked environment (`uv sync --frozen --project Forge`; `<venv>` is `Forge/.venv`) on the pinned
Hugging Face snapshot. Nothing is downloaded. Writes only the new `--output` directory:

    <venv>/bin/python -I Forge/flux2/export_flux2.py --checkpoint <snapshot> --output <export dir>
    python3 Scripts/finalize-flux2-bundle.py <export dir> <snapshot> <bundle dir>

    OUTPUT/
      bundle/<name>-{text,denoiser0,denoiser1,denoiser2,vae,vae-encoder}.aimodel
      bundle/tokenizer/              the checkpoint's tokenizer files, unchanged
      manifest-fragment.json         the recipe half of aether.json (the finalizer adds target and file pins)
      apple-metadata.json            becomes the bundle's metadata.json
      export-report.json             versions, checkpoint pins, quantization, timings, peak RSS, self-checks
      forge-log.jsonl                one event per stage

Conversion is CPU-bound; Python loads assets cpu_only only (GPU behaviour is qualified natively in Swift). The
self-check runs each exported asset cpu_only on small inputs against its eager fp16 module before quantization.
`--allow-unpinned` accepts any snapshot (the random-weight graph tests); a real export verifies every checkpoint
file it reads against the pins recorded for the 0.0.2 bundle.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import fnmatch
import gc
import hashlib
import importlib.metadata
import json
import platform
import re
import resource
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flux2_graphs import (Checkpoint, DenoiserSegment, SEGMENTS, TextEncoderGraph, VAEDecoderGraph,  # noqa: E402
                          VAEEncoderGraph, load_vae)

REPO = "black-forest-labs/FLUX.2-klein-4B"
REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
# Every checkpoint file the export reads, as pinned in docs/evidence/flux2-klein-qualification-2026-10-26.json.
CHECKPOINT_SHA256 = {
    "LICENSE.md": "ca02bc51900ab07789d1b70283329e7137f5af98f5161c23a1c81fc38a4af1fe",
    "text_encoder/config.json": "214b4c29a0d975e9fddf9994a5673f22cb2c4c5750352f9227c2c3251ebeab40",
    "text_encoder/model-00001-of-00002.safetensors": "8c0506e7f4936fa7e26183a4fd8da4e2bdbc5990ba64ae441f965d51228f36ea",
    "text_encoder/model-00002-of-00002.safetensors": "82f2bd839378541b0557bfabaf37c7d3d637071fdcb73302dedd7cf61162ce07",
    "text_encoder/model.safetensors.index.json": "06b3d5319b6d76d1a4a2433419180016cfd54ed62d086a5e6567a809f8c82634",
    "tokenizer/added_tokens.json": "c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680",
    "tokenizer/chat_template.jinja": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
    "tokenizer/merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "tokenizer/special_tokens_map.json": "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd",
    "tokenizer/tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer/tokenizer_config.json": "443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0",
    "tokenizer/vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "transformer/config.json": "09733c74a3da6d17dd0a0472a091a8950c7c6935889c32c16cc800ede05029de",
    "transformer/diffusion_pytorch_model.safetensors": "9f29f9edcfdae452a653ffb51a534ca4decd389952c225724ff3b94042612a6e",
    "vae/config.json": "0d6dfb69ae95a5e2ac9836284bbb63d8b38ce67b25ba2dff380752b2a10ab948",
    "vae/diffusion_pytorch_model.safetensors": "ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04",
}
RECIPE = "flux2-klein-4b-reference@1"
NAME = "flux2_klein_4b_reference"
ASSETS = {"textEncoder": f"{NAME}-text.aimodel", "denoiser0": f"{NAME}-denoiser0.aimodel",
          "denoiser1": f"{NAME}-denoiser1.aimodel", "denoiser2": f"{NAME}-denoiser2.aimodel",
          "vaeDecoder": f"{NAME}-vae.aimodel", "vaeEncoder": f"{NAME}-vae-encoder.aimodel"}
QUANT_SCHEME = "int8-linear-perchannel"
# First match in declaration order wins, then QUANT_SCHEME. Applies to the text encoder and denoiser; both VAE graphs
# keep fp16 weights with float32 compute.
QUANT_EXCEPTIONS = {"singles.*.proj_out": "none"}
TOKENIZER_FILES = ("added_tokens.json", "chat_template.jinja", "merges.txt", "special_tokens_map.json",
                   "tokenizer.json", "tokenizer_config.json", "vocab.json")


# MARK: the recipe (aether.json without the finalizer's target, placement and file pins)

def manifest_fragment(asset_bytes: dict[str, int], quantized: Counter, exported_on: dict) -> dict:
    binding = lambda dtype, bind, **extra: {"bind": bind, "dtype": dtype, **extra}  # noqa: E731
    produced = lambda dtype, role: {"dtype": dtype, "role": role}  # noqa: E731
    notes = ("weight-only PTQ on the torch module (coreai-opt eager Quantizer); exceptions resolve first-match in "
             "declaration order, then default; modules: "
             + ", ".join(f"{rule} ×{quantized[rule]}" for rule in [QUANT_SCHEME, *QUANT_EXCEPTIONS.values()]))
    return {
        "architecture": {"class": "diffusion"},
        "capabilities": ["imageOutput"],
        "diffusion": {
            "assetBytes": asset_bytes,
            "decoderAsset": "vaeDecoder",
            "denoiserAsset": "denoiser",
            "denoiserSegments": [f"denoiser{i}" for i in range(len(SEGMENTS))],
            "denoiserStreaming": {"minImageTokens": 2305},
            "expectFrequentReshapes": True,
            "guidance": False,
            "latentChannels": 128,
            "maxSteps": 50,
            "pixelsPerToken": 16,
            "prompt": {"maxLength": 512, "padToken": 151643, "padding": "right", "template": "chat", "thinking": "off"},
            "referenceImages": {"encoderAsset": "vaeEncoder", "kind": "packedLatents", "maxImages": 2, "maxSide": 512,
                                "maxTotalTokens": 4096, "positionStride": 10},
            "rotary": {"axes": [32, 32, 32, 32], "theta": 2000.0},
            # Diffusers' compute_empirical_mu constants, as Kit's Flux2Schedule.EmpiricalShift.klein spells them.
            "sigmaShift": {"highIntercept": 0.45666666, "highSlope": 0.00016927, "highSteps": 200,
                           "lowIntercept": 1.89833333, "lowSlope": 8.73809524e-05, "lowSteps": 10, "rule": "empirical",
                           "tokenThreshold": 4300},
            "sizes": [{"height": 512, "width": 512}, {"height": 768, "width": 768}],
            "steps": 4,
            "textEncoderAsset": "textEncoder",
            "trainTimesteps": 1000,
        },
        "driver": "diffusion",
        "functions": {"denoiser0/segment": {"role": "denoiseSegment"}, "denoiser1/segment": {"role": "denoiseSegment"},
                      "denoiser2/segment": {"role": "denoiseSegment"}, "textEncoder/encode": {"role": "encodeText"},
                      "vaeDecoder/decode": {"role": "decodeLatents"}, "vaeEncoder/encode": {"role": "encodeLatents"}},
        "inputs": {
            "attention_mask": binding("int32", "tokenMask"), "cond": binding("float32", "carry"),
            "image_1": binding("float32", "carry"), "image_2": binding("float32", "carry"),
            "input_ids": binding("int32", "tokens"), "latents": binding("float16", "latents"),
            "position_ids": binding("int32", "positions", mode="running"),
            "reference_pixels": binding("float32", "image"), "rope_image": binding("float32", "imageRotary"),
            "rope_text": binding("float32", "textRotary"), "sigma": binding("float32", "timestep"),
            "text_1": binding("float32", "carry"), "text_2": binding("float32", "carry"),
            "text_embeds": binding("float16", "textEmbeddings"),
        },
        "memory": {"weightsBytes": sum(asset_bytes.values())},
        "outputs": {
            "cond": produced("float32", "carry"), "image": produced("float32", "pixels"),
            "image_1": produced("float32", "carry"), "image_2": produced("float32", "carry"),
            "reference_latents": produced("float32", "latents"), "text_1": produced("float32", "carry"),
            "text_2": produced("float32", "carry"), "text_embeds": produced("float16", "textEmbeddings"),
            "velocity": produced("float16", "velocity"),
        },
        "provenance": {"exportedOn": exported_on, "forge": "aether-kit Forge/flux2/export_flux2.py", "recipe": RECIPE,
                       "source": {"hfRepo": REPO, "revision": REVISION}},
        "quantization": {"exceptions": dict(QUANT_EXCEPTIONS), "notes": notes, "scheme": QUANT_SCHEME, "weightBits": 8},
        "schema": 1,
    }


def apple_metadata() -> dict:
    return {"assets": dict(ASSETS), "compression": QUANT_SCHEME, "kind": "diffusion",
            "language": {"embedded_tokenizer": True, "tokenizer": REPO}, "metadata_version": "0.2", "name": NAME,
            "source": {"hf_model_id": REPO, "model_definition": "torch"}}


# MARK: helpers

class Log:
    def __init__(self, path: Path) -> None:
        self.path, self.started = path, {}

    def event(self, stage: str, message: str, **fields) -> None:
        record = {"message": message, "stage": stage, "t": round(time.time(), 3), **fields}
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"[flux2:{stage}] {message}" + (f" {fields}" if fields else ""), file=sys.stderr, flush=True)

    def start(self, stage: str) -> None:
        self.started[stage] = time.monotonic()
        self.event(stage, "start")

    def done(self, stage: str, **fields) -> float:
        seconds = round(time.monotonic() - self.started.pop(stage), 3)
        self.event(stage, "done", seconds=seconds, peakRSSGB=peak_rss_gb(), **fields)
        return seconds


def peak_rss_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)   # macOS reports bytes


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def quantize_int8(module: nn.Module, example: tuple, counts: Counter) -> nn.Module:
    """Weight-only symmetric per-channel int8 on every Linear/Embedding not excepted (coreai-opt eager PTQ)."""
    from coreai_opt.quantization import Quantizer
    from coreai_opt.quantization.config import ModuleQuantizerConfig, QuantizerConfig
    from coreai_opt.quantization.spec.granularity import PerChannelGranularity
    from coreai_opt.quantization.spec.qscheme import QuantizationScheme
    from coreai_opt.quantization.spec.spec import QuantizationSpec

    spec = QuantizationSpec(dtype=torch.int8, qscheme=QuantizationScheme.SYMMETRIC,
                            granularity=PerChannelGranularity(axis=None))
    config = ModuleQuantizerConfig(op_input_spec=None, op_output_spec=None, op_state_spec={"weight": spec})
    selected = {}
    for path, child in module.named_modules():
        if not isinstance(child, (nn.Linear, nn.Embedding)):
            continue
        rule = next((r for pattern, r in QUANT_EXCEPTIONS.items() if fnmatch.fnmatchcase(path, pattern)), QUANT_SCHEME)
        counts[rule] += 1
        if rule == QUANT_SCHEME:
            selected[f"^{re.escape(path)}$"] = config
    quantizer = Quantizer(module, QuantizerConfig(global_config=None, module_name_configs=selected, execution_mode="eager"))
    with torch.no_grad():
        prepared = quantizer.prepare(tuple(t.clone() for t in example))
        return quantizer.finalize(prepared).eval()


def convert(module: nn.Module, example: tuple, dynamic, inputs: list[str], outputs: list[str], entry: str,
            path: Path) -> dict:
    import coreai_torch as ct

    t0 = time.monotonic()
    with torch.no_grad():
        ep = torch.export.export(module, example, dynamic_shapes=dynamic, strict=False)
        ep = ep.run_decompositions(ct.get_decomp_table())
    t1 = time.monotonic()
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    converter.add_exported_program(ep, input_names=inputs, output_names=outputs, state_names=[], entrypoint_name=entry)
    del ep
    program = converter.to_coreai()
    del converter
    program.optimize()
    program.save_asset(path)
    return {"export_s": round(t1 - t0, 2), "convert_s": round(time.monotonic() - t1, 2), "bytes": tree_bytes(path)}


def as_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().contiguous().numpy()


async def run_cpu_only(path: Path, entry: str, feeds: list[dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
    import coreai.runtime as rt

    model = await rt.AIModel.load(path, rt.SpecializationOptions.cpu_only())
    function = model.load_function(entry)
    results = []
    for feed in feeds:
        out = await function({k: rt.NDArray(np.ascontiguousarray(v)) for k, v in feed.items()}, {})
        results.append({k: v.numpy().copy() for k, v in out.items()})
    return results


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def compare(expected: dict[str, torch.Tensor], got: dict[str, np.ndarray]) -> dict:
    rows = {}
    for name, value in expected.items():
        e = as_numpy(value.float())
        g = got[name].astype(np.float32)
        if e.shape != g.shape:
            rows[name] = {"shape": list(g.shape), "expected": list(e.shape), "cosine": None}
            continue
        rows[name] = {"shape": list(g.shape), "cosine": round(cosine(e, g), 6),
                      "maxAbs": round(float(np.abs(e - g).max()), 5), "finite": bool(np.isfinite(g).all())}
    return rows


# MARK: stages

def export_component(role: str, module: nn.Module, example: tuple, dynamic, inputs: list[str], outputs: list[str],
                     entry: str, bundle: Path, log: Log, report: dict, quantized: Counter | None,
                     checks: list[tuple] | None) -> list[dict[str, torch.Tensor]]:
    """Eager references on `checks` (before quantization), optional int8 PTQ, conversion, then the cpu_only check.
    Returns the eager references (the next denoiser segment checks on them)."""
    references = []
    if checks is not None:
        with torch.no_grad():
            for feed in checks:
                result = module(*feed)
                result = result if isinstance(result, tuple) else (result,)
                references.append(dict(zip(outputs, (r.detach().clone() for r in result))))
    if quantized is not None:
        log.start(f"quantize-{role}")
        before = Counter(quantized)
        module = quantize_int8(module, example, quantized)
        log.done(f"quantize-{role}", modules={k: quantized[k] - before[k] for k in quantized})
    log.start(f"export-{role}")
    path = bundle / ASSETS[role]
    report["conversion"][role] = convert(module, example, dynamic, inputs, outputs, entry, path)
    log.done(f"export-{role}", bytes=report["conversion"][role]["bytes"])
    del module
    gc.collect()
    if checks is None:
        return references
    log.start(f"self-check-{role}")
    feeds = [{name: as_numpy(t) for name, t in zip(inputs, feed)} for feed in checks]
    got = asyncio.run(run_cpu_only(path, entry, feeds))
    rows = [compare(e, g) for e, g in zip(references, got)]
    report["selfCheck"][role] = {"reference": "eager module before quantization (CPU)", "runtime": "Core AI cpu_only",
                                 "cases": rows}
    log.done(f"self-check-{role}", cosine=[{k: v["cosine"] for k, v in r.items()} for r in rows])
    return references


def text_checks(module: TextEncoderGraph) -> list[tuple]:
    """A 512-token prompt shape (real ids, right-padded with the pad token) and a short one."""
    cases = []
    for length, real in ((512, 37), (24, 24)):
        ids = torch.full((1, length), 151643, dtype=torch.int32)
        ids[0, :real] = torch.tensor([151644, 872, 198] + [1000 + (97 * i) % 50000 for i in range(real - 3)], dtype=torch.int32)
        mask = torch.zeros((1, length), dtype=torch.int32)
        mask[0, :real] = 1
        cases.append((ids, torch.arange(length, dtype=torch.int32)[None], mask))
    return cases


def segment_checks(segment: DenoiserSegment, carries: list[tuple] | None) -> list[tuple]:
    """Segment 0 gets random latents and text rows; later segments the previous segment's eager carries when it
    ran in this export, else random carries."""
    if carries is not None:
        return carries
    from flux2_graphs import CONDITION_ROWS, grid_ids, rotary_table, text_ids
    g = torch.Generator().manual_seed(11 + segment.index)
    cases = []
    for rows, columns, text in ((4, 6, 32), (8, 8, 64)):
        ropes = (rotary_table(text_ids(text)), rotary_table(grid_ids(rows, columns)))
        if segment.is_first:
            cases.append((torch.randn(rows * columns, 128, generator=g).to(segment.compute),
                          (torch.randn(text, segment.text_width, generator=g) * 2).to(segment.compute),
                          torch.tensor([0.6], dtype=torch.float32)) + ropes)
        else:
            cases.append((torch.randn(rows * columns, segment.hidden, generator=g),
                          torch.randn(text, segment.hidden, generator=g),
                          torch.randn(CONDITION_ROWS, segment.hidden, generator=g) * 0.1) + ropes)
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help=f"{REPO} snapshot at {REVISION}")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", nargs="*", choices=list(ASSETS), help="export a subset (probes; not a bundle)")
    parser.add_argument("--no-self-check", action="store_true", help="skip the cpu_only checks (qualify natively)")
    parser.add_argument("--allow-unpinned", action="store_true", help="random-weight test snapshots only")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    roles = args.only or list(ASSETS)
    args.output.mkdir(parents=True)
    bundle = args.output / "bundle"
    bundle.mkdir()
    log = Log(args.output / "forge-log.jsonl")
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    checkpoint = Checkpoint(args.checkpoint.resolve())
    exported_on = {"date": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"),
                   "os": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()}
    report = {
        "schema": 1, "recipe": RECIPE, "command": sys.argv, "exportedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {"os": exported_on["os"], "machine": platform.machine(), "python": platform.python_version()},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "coreai-torch", "coreai-core", "coreai-opt",
                                                                "diffusers", "transformers", "safetensors")},
        "scriptSHA256": {p.name: sha256(p) for p in sorted(Path(__file__).resolve().parent.glob("*.py"))},
        "source": {"repository": REPO, "revision": REVISION, "checkpoint": str(checkpoint.root), "verified": {}},
        "quantization": {"scheme": QUANT_SCHEME, "exceptions": QUANT_EXCEPTIONS},
        "conversion": {}, "selfCheck": {}, "stages": {},
    }
    report_path = args.output / "export-report.json"

    def save() -> None:
        report["peakRSSGB"] = peak_rss_gb()
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    log.start("checkpoint")
    for relative, expected in CHECKPOINT_SHA256.items():
        path = checkpoint.root / relative
        if not path.is_file():
            if args.allow_unpinned:
                continue
            raise SystemExit(f"checkpoint lacks {relative}")
        actual = sha256(path)
        if actual != expected and not args.allow_unpinned:
            raise SystemExit(f"{relative} differs from the pinned {REVISION[:12]} file")
        report["source"]["verified"][relative] = actual == expected
    report["stages"]["checkpoint"] = log.done("checkpoint")
    save()

    check = not args.no_self_check
    quantized: Counter = Counter()
    if "textEncoder" in roles:
        log.start("build-textEncoder")
        module = TextEncoderGraph.load(checkpoint)
        report["stages"]["build-textEncoder"] = log.done("build-textEncoder")
        example, dynamic = module.example()
        export_component("textEncoder", module, example, dynamic, ["input_ids", "position_ids", "attention_mask"],
                         ["text_embeds"], "encode", bundle, log, report, quantized,
                         text_checks(module) if check else None)
        del module
        gc.collect()
        save()

    carries = None       # the previous segment's eager outputs as the next segment's check inputs
    for index in range(len(SEGMENTS)):
        role = f"denoiser{index}"
        if role not in roles:
            carries = None
            continue
        log.start(f"build-{role}")
        segment = DenoiserSegment.load(checkpoint, index)
        report["stages"][f"build-{role}"] = log.done(f"build-{role}")
        example, dynamic = segment.example()
        cases = segment_checks(segment, carries) if check else None
        references = export_component(role, segment, example, dynamic, segment.input_names, segment.output_names,
                                      "segment", bundle, log, report, quantized, cases)
        carries = None
        if cases is not None and index < len(SEGMENTS) - 1:
            names = segment.output_names
            carries = [(r[names[0]], r[names[1]], r["cond"] if index == 0 else case[2], case[3], case[4])
                       for r, case in zip(references, cases)]
        del segment
        gc.collect()
        save()

    if "vaeDecoder" in roles or "vaeEncoder" in roles:
        log.start("build-vae")
        vae = load_vae(checkpoint)
        report["stages"]["build-vae"] = log.done("build-vae")
        if "vaeDecoder" in roles:
            module = VAEDecoderGraph(vae).eval()
            example, dynamic = module.example()
            # No cpu_only check: Core AI's cpu_only runtime segfaults running this decoder (the 2026-10-26 asset
            # included, 2026-10-27); the native parity suite qualifies it on the GPU.
            export_component("vaeDecoder", module, example, dynamic, ["latents"], ["image"], "decode", bundle, log,
                             report, None, None)
            if check:
                report["selfCheck"]["vaeDecoder"] = {"skipped": "Core AI cpu_only segfaults on this decoder", "cases": []}
        if "vaeEncoder" in roles:
            module = VAEEncoderGraph(vae).eval()
            example, dynamic = module.example()
            g = torch.Generator().manual_seed(6)
            cases = [((torch.rand(3, 16 * r, 16 * c, generator=g) * 2 - 1),) for r, c in ((4, 6), (8, 8))] \
                if check else None
            export_component("vaeEncoder", module, example, dynamic, ["reference_pixels"], ["reference_latents"],
                             "encode", bundle, log, report, None, cases)
        del vae
        gc.collect()
        save()

    report["quantization"]["modules"] = dict(quantized)
    if set(roles) == set(ASSETS):
        log.start("tokenizer")
        (bundle / "tokenizer").mkdir()
        for name in TOKENIZER_FILES:
            shutil.copyfile(checkpoint.root / "tokenizer" / name, bundle / "tokenizer" / name)
        report["stages"]["tokenizer"] = log.done("tokenizer")
        asset_bytes = {role: tree_bytes(bundle / name) for role, name in ASSETS.items()}
        fragment = manifest_fragment(asset_bytes, quantized, exported_on)
        (args.output / "manifest-fragment.json").write_text(json.dumps(fragment, indent=2, sort_keys=True) + "\n")
        (args.output / "apple-metadata.json").write_text(json.dumps(apple_metadata(), indent=2, sort_keys=True) + "\n")
        report["bundleBytes"] = tree_bytes(bundle)
    report["modelFiles"] = {str(p.relative_to(bundle)): sha256(p) for p in sorted(bundle.rglob("*")) if p.is_file()}
    save()
    failed = {role: case for role, entry in report["selfCheck"].items() for case in entry["cases"]
              if any(v["cosine"] is None or v["cosine"] < 0.99 or not v["finite"] for v in case.values())}
    print(json.dumps({"assets": report["conversion"], "quantized": dict(quantized), "peakRSSGB": report["peakRSSGB"],
                      "selfCheckFailures": failed}, indent=1))
    if failed:
        raise SystemExit("self-check failed")


if __name__ == "__main__":
    main()

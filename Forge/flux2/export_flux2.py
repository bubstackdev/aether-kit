#!/usr/bin/env python3
"""Export FLUX.2 [klein] 4B (black-forest-labs/FLUX.2-klein-4B, Apache-2.0) as an Aether-Kit `flux2-klein@1` bundle.

    uv sync --frozen --project Forge
    Forge/.venv/bin/python -I Forge/flux2/export_flux2.py --output <new bundle dir>

Nothing is downloaded: the snapshot must be in the Hugging Face cache (or `--snapshot`), and every file read is
verified against the pins below. Writes only `--output`:

    OUTPUT/
      .aether/aether.json               the Aether manifest (MANIFEST.md §5.7; driver `flux2-klein@1`)
      flux2_klein_4b-{text,denoiser0..2,vae,vae-encoder}.aimodel
      tokenizer/                        the checkpoint's tokenizer files, unchanged
      LICENSE.md, NOTICE                the checkpoint's Apache-2.0 text; the modification notice
      SHA256SUMS                        every bundle file except itself and the log

The output is laid out as an installed revision root (the manifest under `.aether/`), so it uploads as-is.
      export-report.json                provenance, versions, quantization, timings, peak RSS
      forge-log.jsonl                   one event per stage

The recipe is the one 0.0.x qualified: weight-only int8 per-channel PTQ (coreai-opt) on the text encoder and the
denoisers, except the 20 single-block `proj_out` that stay fp16; both VAE graphs store fp16 and compute in float32.
Conversion runs on the CPU; parity is the separate gate `check_flux2.py` (GPU) against `reference_flux2.py`.
"""

from __future__ import annotations

import argparse
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

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import flux2_graphs as fg  # noqa: E402

REPO = "black-forest-labs/FLUX.2-klein-4B"
REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
DIFFUSERS_COMMIT = "2a1433913c7044f34fa4767dd84b28bba0bb0ecf"   # tag v0.42.1; see Forge/pyproject.toml
DEFAULT_SNAPSHOT = (Path.home() / ".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots"
                    / REVISION)
# Every checkpoint file the export reads, at REVISION (the pins 0.0.x recorded for this revision).
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
TOKENIZER_FILES = ("added_tokens.json", "chat_template.jinja", "merges.txt", "special_tokens_map.json",
                   "tokenizer.json", "tokenizer_config.json", "vocab.json")
ID = "flux2-klein-4b"
NAME = "flux2_klein_4b"
ROLES = ("textEncoder", "denoiser0", "denoiser1", "denoiser2", "vaeDecoder", "vaeEncoder")
ASSET_SUFFIX = {"textEncoder": "text", "vaeDecoder": "vae", "vaeEncoder": "vae-encoder",
                **{f"denoiser{i}": f"denoiser{i}" for i in range(3)}}
FUNCTIONS = {"textEncoder": "encode", "vaeDecoder": "decode", "vaeEncoder": "encode",
             **{f"denoiser{i}": "segment" for i in range(3)}}
QUANT_SCHEME = "int8-linear-perchannel"
# First match in declaration order wins, then QUANT_SCHEME. Applies to the text encoder and denoisers; both VAE graphs
# keep fp16 weights with float32 compute.
QUANT_EXCEPTIONS = {"singles.*.proj_out": "none"}


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
        seconds = round(time.monotonic() - self.started.pop(stage), 2)
        self.event(stage, "done", seconds=seconds, peakRSSGB=peak_rss_gb(), **fields)
        return seconds


def peak_rss_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)    # macOS reports bytes


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def tree_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in tree_files(root))


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
    return {"exportSeconds": round(t1 - t0, 1), "convertSeconds": round(time.monotonic() - t1, 1),
            "bytes": tree_bytes(path), "inputs": inputs, "outputs": outputs, "function": entry}


MANIFEST_PATH = ".aether/aether.json"


# MARK: the manifest (MANIFEST.md §5: generic `schedule`/`latent`/`residency` since G0; driver `flux2-klein@1` and its
# `flux2` block, §5.7, since G2)

def manifest(files: dict[str, dict], asset_names: dict[str, str], report: dict) -> dict:
    graph_file = lambda role: f"main:{asset_names[role]}"  # noqa: E731
    from_ = lambda ref: {"name": ref.split(".")[1], "source": {"from": ref}}  # noqa: E731
    rotary = {"textRotary": "rope_text", "imageRotary": "rope_image"}
    graphs = {
        "textEncoder": {"file": graph_file("textEncoder"), "function": "encode",
                        "inputs": {"tokens": "input_ids", "positions": "position_ids", "tokenMask": "attention_mask"},
                        "outputs": {"conditioning": "text_embeds"},
                        "compute": "gpu", "expectFrequentReshapes": True, "residency": "perPrompt"},
        "denoiser0": {"file": graph_file("denoiser0"), "function": "segment",
                      "inputs": {"latents": "latents", "conditioning": from_("textEncoder.text_embeds"),
                                 "sigma": "sigma", **rotary},
                      "outputs": {"carryImage": "image_1", "carryText": "text_1", "condition": "cond"},
                      "compute": "gpu", "expectFrequentReshapes": True, "residency": "perStep"},
        "denoiser1": {"file": graph_file("denoiser1"), "function": "segment",
                      "inputs": {"carryImage": from_("denoiser0.image_1"), "carryText": from_("denoiser0.text_1"),
                                 "condition": from_("denoiser0.cond"), **rotary},
                      "outputs": {"carryImage": "image_2", "carryText": "text_2"},
                      "compute": "gpu", "expectFrequentReshapes": True, "residency": "perStep"},
        "denoiser2": {"file": graph_file("denoiser2"), "function": "segment",
                      "inputs": {"carryImage": from_("denoiser1.image_2"), "carryText": from_("denoiser1.text_2"),
                                 "condition": from_("denoiser0.cond"), **rotary},
                      "outputs": {"velocity": "velocity"},
                      "compute": "gpu", "expectFrequentReshapes": True, "residency": "perStep"},
        # 0.0.x qualified both VAE graphs without the reshape hint (each output size is its own specialization, and
        # the driver recycles the decoder when the size changes).
        "vaeDecoder": {"file": graph_file("vaeDecoder"), "function": "decode",
                       "inputs": {"latents": "latents"}, "outputs": {"image": "image"},
                       "compute": "gpu", "expectFrequentReshapes": False, "residency": "perImage"},
        "vaeEncoder": {"file": graph_file("vaeEncoder"), "function": "encode",
                       "inputs": {"referencePixels": "reference_pixels"},
                       "outputs": {"referenceLatents": "reference_latents"},
                       "compute": "gpu", "expectFrequentReshapes": False, "residency": "perPrompt"},
    }
    tokenizer_sha = files.get("tokenizer/chat_template.jinja", {}).get("sha256")
    return {
        "schema": 1,
        "id": ID,
        "task": "text-to-image",
        "driver": "flux2-klein@1",
        "variant": {"platform": "macos"},
        "assets": {"main": {"subtree": "", "files": [{"path": p, **meta} for p, meta in sorted(files.items())]}},
        "graphs": {role: graphs[role] for role in ROLES},
        "tokenizer": {"source": {"dir": "main:tokenizer"}, "chatTemplate": "main:tokenizer/chat_template.jinja",
                      "chatTemplateSha256": tokenizer_sha, "renderer": "flux2-prompt"},
        "schedule": {
            "kind": "flowMatchEuler", "steps": fg.STEPS, "maxSteps": 50, "trainTimesteps": fg.TRAIN_TIMESTEPS,
            "sigmas": {"base": "linspace(1, 1/steps, steps)",
                       "shift": {"rule": "empirical", "empirical": dict(fg.EMPIRICAL)}, "terminal": 0.0},
            "timestep": "sigma * trainTimesteps / trainTimesteps",
            "update": "x + (sigma[i+1] - sigma[i]) * velocity",
            "guidance": {"kind": "none"},
        },
        "latent": {
            "channels": fg.LATENT_CHANNELS, "pixelsPerToken": fg.PIXELS_PER_TOKEN, "packing": "patch2x2ChannelMajor",
            "layout": "tokens[rows*columns, 128]", "decoderLayout": "[rows, columns, 128]",
            "noise": {"kind": "splitmix64BoxMuller", "order": "packedToken", "dtype": "float32"},
            "maxImageTokens": fg.MAX_IMAGE_TOKENS,
        },
        "flux2": {
            "prompt": {"template": "chat", "thinking": "off", "addGenerationPrompt": True, "maxTokens": fg.MAX_TOKENS,
                       "padding": "right", "padToken": fg.PAD_TOKEN, "truncation": "keepPrefix",
                       "positions": "running"},
            "conditioning": {"taps": list(fg.TEXT_TAPS), "width": 3 * 2560, "rows": fg.MAX_TOKENS,
                             "paddingRows": "conditioning"},
            "rotary": {"axes": list(fg.ROPE_AXES), "theta": fg.ROPE_THETA, "textIds": "(0, 0, 0, i)",
                       "imageIds": "(0, row, column, 0)", "referenceIds": "(10 * (k + 1), row, column, 0)",
                       "layout": "splitHalves", "dtype": "float32"},
            "segments": [f"denoiser{i}" for i in range(len(fg.SEGMENTS))],
            "condRows": fg.CONDITION_ROWS,
            "references": {"encoder": "vaeEncoder", "kind": "packedLatents", "positionStride": fg.REFERENCE_STRIDE,
                           **fg.REFERENCE_LIMITS, "pixels": "[3, H, W] in [-1, 1], sides multiples of 16",
                           "velocity": "discarded", "mu": "generated tokens only"},
        },
        "precision": {"weights": "int8 per-channel (text encoder, denoisers; the 20 single-block proj_out fp16); "
                                 "fp16 (both VAE graphs)",
                      "activations": "fp16 matmul/attention; fp32 residual, norms, modulation and rotary products",
                      "components": {"vaeDecoder": "fp16 weights, fp32 compute",
                                     "vaeEncoder": "fp16 weights, fp32 compute"},
                      "quantization": {"scheme": QUANT_SCHEME, "weightBits": 8,
                                       "exceptions": dict(QUANT_EXCEPTIONS),
                                       "modules": report["quantization"]["modules"]}},
        "license": "Apache-2.0",
        "x-license": {"files": ["LICENSE.md", "NOTICE"], "modified": True},
        "provenance": {
            "source": REPO, "sourceRevision": REVISION,
            "producer": "aether-kit Forge/flux2/export_flux2.py",
            "exportScript": report["scriptSHA256"]["export_flux2.py"],
            "refs": {"diffusersCommit": DIFFUSERS_COMMIT, "forgeLockSHA256": report["forgeLockSHA256"],
                     "exportReport": "export-report.json"},
        },
    }


def sample_manifest() -> dict:
    """`manifest()` for placeholder files and report values: the emitted shape, without an export. Regenerate the Swift
    fixture with `--sample-manifest > Tests/AetherCoreTests/Fixtures/flux2/flux2-klein-4b.json`."""
    asset_names = {role: f"{NAME}-{ASSET_SUFFIX[role]}.aimodel" for role in ROLES}
    files = {f"{name}/weights.bin": {"bytes": 1, "sha256": "0" * 64} for name in asset_names.values()}
    files["tokenizer/chat_template.jinja"] = {"bytes": 1, "sha256": CHECKPOINT_SHA256["tokenizer/chat_template.jinja"]}
    report = {"scriptSHA256": {"export_flux2.py": "0" * 64}, "forgeLockSHA256": "0" * 64,
              "quantization": {"modules": {QUANT_SCHEME: 205, "none": 20}}}
    return manifest(files, asset_names, report)


def notice() -> str:
    return "\n".join([
        "FLUX.2 [klein] 4B (Aether-Kit Core AI export)",
        "",
        f"This bundle is a modified version of {REPO} at revision {REVISION}, licensed under the",
        "Apache License, Version 2.0 (LICENSE.md). Aether-Kit's Forge exporter re-expressed its weights as Core AI",
        "graphs (text encoder, three denoiser segments, VAE decoder and encoder), keeping only the text-encoder layers",
        "FLUX.2 reads, and quantized the text encoder and denoiser weights to int8 per channel. It is not an official",
        "Black Forest Labs release.",
        "",
    ])


# MARK: main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="new bundle directory")
    parser.add_argument("--sample-manifest", action="store_true",
                        help="print the manifest for placeholder files and exit (the Swift decoding fixture)")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--only", nargs="*", choices=ROLES, help="export a subset (probes; not a bundle)")
    parser.add_argument("--allow-unpinned", action="store_true", help="random-weight test snapshots only")
    args = parser.parse_args()
    if args.sample_manifest:
        print(json.dumps(sample_manifest(), indent=2, ensure_ascii=False))
        return
    if args.output is None:
        parser.error("--output is required")
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    roles = args.only or list(ROLES)
    started = time.monotonic()
    args.output.mkdir(parents=True)
    log = Log(args.output / "forge-log.jsonl")
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    asset_names = {role: f"{NAME}-{ASSET_SUFFIX[role]}.aimodel" for role in ROLES}
    checkpoint = fg.Checkpoint(args.snapshot.resolve())
    lock = HERE.parent / "uv.lock"
    report = {
        "schema": 1, "id": ID, "command": sys.argv, "exportedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {"os": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
                 "machine": platform.machine(), "python": platform.python_version(),
                 "memoryGB": round(int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                                      text=True).stdout) / 2**30)},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "coreai-torch", "coreai-core", "coreai-opt",
                                                                "diffusers", "transformers", "safetensors", "numpy")},
        "diffusersCommit": DIFFUSERS_COMMIT,
        "forgeLockSHA256": sha256(lock), "forgePyprojectSHA256": sha256(HERE.parent / "pyproject.toml"),
        "scriptSHA256": {p.name: sha256(p) for p in sorted(HERE.glob("*.py"))},
        "source": {"repository": REPO, "revision": REVISION, "snapshot": str(checkpoint.root), "verified": {}},
        "precision": "int8 per-channel weights on the text encoder and denoisers (single-block proj_out fp16); fp16 "
                     "matmul/attention with fp32 residual, norms, modulation and rotary; VAE fp16 storage with fp32 "
                     "compute",
        "quantization": {"scheme": QUANT_SCHEME, "exceptions": QUANT_EXCEPTIONS, "modules": {}},
        "conversion": {}, "stages": {},
    }
    report_path = args.output / "export-report.json"
    quantized: Counter = Counter()

    def save() -> None:
        report["quantization"]["modules"] = dict(quantized)
        report["peakRSSGB"] = peak_rss_gb()
        report["wallSeconds"] = round(time.monotonic() - started, 1)
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    log.start("verify-sources")
    for relative, expected in CHECKPOINT_SHA256.items():
        if args.allow_unpinned:
            break
        path = checkpoint.root / relative
        if not path.is_file():
            raise SystemExit(f"snapshot lacks {relative}")
        if sha256(path) != expected:
            raise SystemExit(f"{relative} differs from the pinned {REVISION[:12]} file")
        report["source"]["verified"][relative] = expected
    report["stages"]["verify-sources"] = log.done("verify-sources")
    save()

    def export(role: str, module: nn.Module, inputs: list[str], outputs: list[str], quantize: bool) -> None:
        example, dynamic = module.example()
        if quantize:
            log.start(f"quantize-{role}")
            before = Counter(quantized)
            module = quantize_int8(module, example, quantized)
            report["stages"][f"quantize-{role}"] = log.done(
                f"quantize-{role}", modules={k: quantized[k] - before[k] for k in quantized})
        log.start(f"export-{role}")
        report["conversion"][role] = convert(module, example, dynamic, inputs, outputs, FUNCTIONS[role],
                                             args.output / asset_names[role])
        report["stages"][f"export-{role}"] = log.done(f"export-{role}", bytes=report["conversion"][role]["bytes"])
        del module
        gc.collect()
        save()

    if "textEncoder" in roles:
        log.start("build-textEncoder")
        module = fg.TextEncoderGraph.load(checkpoint)
        report["stages"]["build-textEncoder"] = log.done("build-textEncoder")
        export("textEncoder", module, ["input_ids", "position_ids", "attention_mask"], ["text_embeds"], True)
        del module

    for index in range(len(fg.SEGMENTS)):
        role = f"denoiser{index}"
        if role not in roles:
            continue
        log.start(f"build-{role}")
        segment = fg.DenoiserSegment.load(checkpoint, index)
        report["stages"][f"build-{role}"] = log.done(f"build-{role}")
        export(role, segment, segment.input_names, segment.output_names, True)
        del segment

    if "vaeDecoder" in roles or "vaeEncoder" in roles:
        vae = fg.load_vae(checkpoint)
        if "vaeDecoder" in roles:
            export("vaeDecoder", fg.VAEDecoderGraph(vae).eval(), ["latents"], ["image"], False)
        if "vaeEncoder" in roles:
            export("vaeEncoder", fg.VAEEncoderGraph(vae).eval(), ["reference_pixels"], ["reference_latents"], False)
        del vae
        gc.collect()

    if set(roles) != set(ROLES):
        save()
        print(json.dumps({"partial": roles, "conversion": report["conversion"], "peakRSSGB": report["peakRSSGB"]}))
        return

    log.start("package")
    (args.output / "tokenizer").mkdir()
    for name in TOKENIZER_FILES:
        shutil.copyfile(checkpoint.root / "tokenizer" / name, args.output / "tokenizer" / name)
    shutil.copyfile(checkpoint.root / "LICENSE.md", args.output / "LICENSE.md")
    (args.output / "NOTICE").write_text(notice())
    skip = {MANIFEST_PATH, "SHA256SUMS", "export-report.json", "forge-log.jsonl"}
    files = {str(p.relative_to(args.output)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in tree_files(args.output) if str(p.relative_to(args.output)) not in skip}
    report["bundleBytes"] = sum(f["bytes"] for f in files.values())
    report["assetBytes"] = {role: tree_bytes(args.output / name) for role, name in asset_names.items()}
    report["stages"]["package"] = log.done("package")
    save()
    (args.output / ".aether").mkdir()
    (args.output / MANIFEST_PATH).write_text(json.dumps(manifest(files, asset_names, report), indent=2) + "\n")
    report["aetherJSONSHA256"] = sha256(args.output / MANIFEST_PATH)
    save()
    sums = [f"{sha256(p)}  {p.relative_to(args.output)}" for p in tree_files(args.output)
            if p.name not in ("SHA256SUMS", "forge-log.jsonl")]
    (args.output / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(json.dumps({"conversion": report["conversion"], "quantized": dict(quantized),
                      "bundleBytes": report["bundleBytes"], "peakRSSGB": report["peakRSSGB"],
                      "wallSeconds": report["wallSeconds"]}, indent=1))


if __name__ == "__main__":
    main()

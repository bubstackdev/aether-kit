#!/usr/bin/env python3
"""Export SD-Turbo (stabilityai/sd-turbo, Stability AI Community License) as an Aether-Kit `latent-diffusion@1` bundle.

    uv sync --frozen --project Forge
    Forge/.venv/bin/python -I Forge/sdturbo/export_sdturbo.py --output <new bundle dir>
    Forge/.venv/bin/python -I Forge/sdturbo/export_sdturbo.py --sample-manifest   # the Swift decoding fixture

Nothing is downloaded: the snapshot must be in the Hugging Face cache (or `--snapshot`), and every file read is
verified against the pins in `sdturbo_graphs.py`. Conversion runs on the CPU, in float32 (the fp16 checkpoint widened).
The output directory is laid out as an installed revision root, so the Swift driver opens it directly:

    OUTPUT/
      .aether/aether.json               the Aether manifest (MANIFEST.md §1, §5; driver `latent-diffusion@1`)
      sd_turbo-{text,unet,vae}.aimodel  the three graphs, static shapes (the export's --size)
      tokenizer/                        the checkpoint's CLIP tokenizer files, unchanged
      LICENSE.md, NOTICE                the Stability AI Community License and its required notice
      SHA256SUMS                        every bundle file except itself
      export-report.json                provenance, versions, timings, peak RSS
      forge-log.jsonl                   one event per stage

Parity: `reference_sdturbo.py` (CPU Diffusers) records the reference; `check_sdturbo.py` checks each graph on the GPU
against it, and the Swift gate `latentDiffusionMatchesDiffusers` checks the whole driver from the seed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
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
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sdturbo_graphs as sg  # noqa: E402

DIFFUSERS_COMMIT = "2a1433913c7044f34fa4767dd84b28bba0bb0ecf"   # tag v0.42.1; see Forge/pyproject.toml
ID = "sd-turbo"
NAME = "sd_turbo"
ASSETS = {"textEncoder": f"{NAME}-text.aimodel", "denoiser": f"{NAME}-unet.aimodel", "vaeDecoder": f"{NAME}-vae.aimodel"}
NOT_BUNDLE = {".aether/aether.json", "SHA256SUMS", "export-report.json", "forge-log.jsonl"}


class Log:
    def __init__(self, path: Path) -> None:
        self.path, self.started = path, {}

    def event(self, stage: str, message: str, **fields) -> None:
        record = {"message": message, "stage": stage, "t": round(time.time(), 3), **fields}
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"[sdturbo:{stage}] {message}" + (f" {fields}" if fields else ""), file=sys.stderr, flush=True)

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


def convert(module: nn.Module, example: tuple, inputs: list[str], output: str, entry: str, path: Path) -> dict:
    import coreai_torch as ct

    t0 = time.monotonic()
    with torch.no_grad():
        ep = torch.export.export(module, example, strict=False).run_decompositions(ct.get_decomp_table())
    t1 = time.monotonic()
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    converter.add_exported_program(ep, input_names=inputs, output_names=[output], state_names=[], entrypoint_name=entry)
    del ep
    program = converter.to_coreai()
    del converter
    program.optimize()
    program.save_asset(path)
    return {"exportSeconds": round(t1 - t0, 1), "convertSeconds": round(time.monotonic() - t1, 1),
            "bytes": tree_bytes(path), "inputs": inputs, "outputs": [output], "function": entry}


# MARK: the manifest (MANIFEST.md §1 `latent-diffusion@1`, §5.2 `schedule` epsilonEuler and `latent` spatial)

def manifest(files: dict[str, dict], tables: dict[str, dict], size: int, report: dict) -> dict:
    graph = lambda role, function, inputs, outputs, residency: {  # noqa: E731
        "file": f"main:{ASSETS[role]}", "function": function, "inputs": inputs, "outputs": outputs,
        "compute": "gpu", "expectFrequentReshapes": False, "residency": residency}
    side = size // sg.PIXELS_PER_LATENT
    return {
        "schema": 1,
        "id": ID,
        "task": "text-to-image",
        "driver": "latent-diffusion@1",
        "variant": {"platform": "macos"},
        "assets": {"main": {"subtree": "", "files": [{"path": p, **meta} for p, meta in sorted(files.items())]}},
        "graphs": {
            "textEncoder": {**graph("textEncoder", "encode", {"tokens": "input_ids"}, {"conditioning": "text_embeds"},
                                    "perPrompt"), "limits": {"staticSequence": sg.TEXT_LENGTH}},
            "denoiser": graph("denoiser", "denoise",
                              {"latents": "sample", "timestep": "timestep",
                               "conditioning": {"name": "text_embeds", "source": {"from": "textEncoder.text_embeds"}}},
                              {"epsilon": "epsilon"}, "perStep"),
            "vaeDecoder": graph("vaeDecoder", "decode", {"latents": "latents"}, {"image": "image"}, "perImage"),
        },
        "tokenizer": {"source": {"dir": "main:tokenizer"}, "chatTemplate": None, "renderer": "clip-bpe",
                      "addSpecialTokens": True},
        "schedule": {
            "kind": "epsilonEuler", "steps": sg.DEFAULT_STEPS, "maxSteps": max(sg.STEPS),
            "trainTimesteps": sg.TRAIN_TIMESTEPS, "tables": tables,
            "modelInput": "x / sqrt(sigma[i]^2 + 1)",
            "timestep": "tables[steps].timesteps[i]",
            "update": "x + (x - (x - sigma[i] * epsilon)) / sigma[i] * (sigma[i+1] - sigma[i])",
            "guidance": {"kind": "none"},
        },
        "latent": {
            "channels": sg.LATENT_CHANNELS, "pixelsPerToken": sg.PIXELS_PER_LATENT, "packing": "spatial",
            "layout": "nchw[1, 4, rows, columns]", "decoderLayout": "nchw[1, 4, rows, columns]",
            "noise": {"kind": "splitmix64BoxMuller", "order": "nchw", "dtype": "float32"},
            "maxImageTokens": side * side, "sizes": [[size, size]],
        },
        "precision": {"weights": "fp32 (widened from the fp16 checkpoint)", "activations": "fp32",
                      "components": {"vaeDecoder": "divides by the VAE scaling factor 0.18215 itself"}},
        "license": "LicenseRef-Stability-AI-Community-License",
        "x-license": {"files": ["LICENSE.md", "NOTICE"], "modified": True, "commercialRevenueCapUSD": 1_000_000,
                      "commercialUse": "requires registration at https://stability.ai/community-license",
                      "attribution": "display \"Powered by Stability AI\" (license §IV.a)"},
        "provenance": {
            "source": sg.REPO, "sourceRevision": sg.REVISION,
            "producer": "aether-kit Forge/sdturbo/export_sdturbo.py",
            "exportScript": report["scriptSHA256"]["export_sdturbo.py"],
            "refs": {"diffusersCommit": DIFFUSERS_COMMIT, "forgeLockSHA256": report["forgeLockSHA256"],
                     "exportReport": "export-report.json"},
        },
    }


def sample_manifest() -> dict:
    """`manifest()` for placeholder files and report values, with the real schedule tables (from the pinned Diffusers
    and scheduler config). Regenerate the Swift fixture with
    `--sample-manifest > Tests/AetherCoreTests/Fixtures/sd-turbo/aether.json`."""
    files = {f"{name}/weights.bin": {"bytes": 1, "sha256": "0" * 64} for name in ASSETS.values()}
    report = {"scriptSHA256": {"export_sdturbo.py": "0" * 64}, "forgeLockSHA256": "0" * 64}
    return manifest(files, sg.schedule_tables(sg.DEFAULT_SNAPSHOT), sg.SIZE, report)


def notice(size: int) -> str:
    return "\n".join([
        "SD-Turbo (Aether-Kit Core AI export)",
        "",
        sg.NOTICE_LINE,
        "",
        f"This bundle is a modified version of the Stability AI Model {sg.REPO} at revision {sg.REVISION}: its fp16",
        f"weights were widened to float32 and re-expressed as Core AI graphs (CLIP text encoder, U-Net, VAE decoder) at",
        f"a fixed {size}x{size} geometry by Aether-Kit's Forge exporter. The architecture and weights are otherwise unchanged.",
        "",
        "This is not an official Stability AI release. Use is subject to the Stability AI Community License (LICENSE.md)",
        "and Stability AI's Acceptable Use Policy. Commercial use requires registration with Stability AI and is limited",
        "to organizations under USD 1,000,000 in annual revenue; products must display \"Powered by Stability AI\".",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="new bundle directory")
    parser.add_argument("--sample-manifest", action="store_true",
                        help="print the manifest for placeholder files and exit (the Swift decoding fixture)")
    parser.add_argument("--snapshot", type=Path, default=sg.DEFAULT_SNAPSHOT)
    parser.add_argument("--size", type=int, default=sg.SIZE, help="fixed square size, a multiple of 64 in 64...512")
    args = parser.parse_args()
    if args.sample_manifest:
        print(json.dumps(sample_manifest(), indent=2, ensure_ascii=False))
        return
    if args.output is None:
        parser.error("--output is required")
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    if not (64 <= args.size <= 512 and args.size % 64 == 0):
        raise SystemExit("--size must be a multiple of 64 in 64...512")
    started = time.monotonic()
    args.output.mkdir(parents=True)
    log = Log(args.output / "forge-log.jsonl")
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    root = args.snapshot.resolve()
    lock = HERE.parent / "uv.lock"
    report = {
        "schema": 1, "id": ID, "size": args.size, "command": sys.argv,
        "exportedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {"os": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
                 "machine": platform.machine(), "python": platform.python_version()},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "coreai-torch", "coreai-core", "coreai-opt",
                                                                "diffusers", "transformers", "safetensors", "numpy")},
        "diffusersCommit": DIFFUSERS_COMMIT,
        "forgeLockSHA256": sha256(lock), "forgePyprojectSHA256": sha256(HERE.parent / "pyproject.toml"),
        "scriptSHA256": {p.name: sha256(p) for p in sorted(HERE.glob("*.py"))},
        "source": {"repository": sg.REPO, "revision": sg.REVISION, "snapshot": str(root), "verified": {}},
        "precision": "float32 weights (widened from the fp16 checkpoint) and activations; no quantization",
        "conversion": {}, "stages": {},
    }
    report_path = args.output / "export-report.json"

    def save() -> None:
        report["peakRSSGB"] = peak_rss_gb()
        report["wallSeconds"] = round(time.monotonic() - started, 1)
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    log.start("verify-sources")
    for relative, expected in sg.CHECKPOINT_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise SystemExit(f"snapshot lacks {relative}")
        if sha256(path) != expected:
            raise SystemExit(f"{relative} differs from the pinned {sg.REVISION[:12]} file")
        report["source"]["verified"][relative] = expected
    report["stages"]["verify-sources"] = log.done("verify-sources")
    tables = sg.schedule_tables(root)
    save()

    shape = sg.latent_shape(args.size)
    stages = [
        ("textEncoder", lambda: sg.TextEncoder(sg.load_text_encoder(root)),
         lambda: (torch.zeros(1, sg.TEXT_LENGTH, dtype=torch.int32),), ["input_ids"], "text_embeds", "encode"),
        ("denoiser", lambda: sg.EpsilonDenoiser(sg.load_unet(root)),
         lambda: (torch.randn(*shape), torch.tensor([999.0]), torch.randn(1, sg.TEXT_LENGTH, sg.TEXT_WIDTH)),
         ["sample", "timestep", "text_embeds"], "epsilon", "denoise"),
        ("vaeDecoder", lambda: sg.LatentDecoder(sg.load_vae(root)), lambda: (torch.randn(*shape),),
         ["latents"], "image", "decode"),
    ]
    for role, build, example, inputs, output, entry in stages:
        log.start(f"export-{role}")
        module = build().eval()
        report["conversion"][role] = convert(module, example(), inputs, output, entry, args.output / ASSETS[role])
        report["stages"][f"export-{role}"] = log.done(f"export-{role}")
        del module
        gc.collect()
        save()

    log.start("package")
    (args.output / "tokenizer").mkdir()
    for name in sg.TOKENIZER_FILES:
        shutil.copyfile(root / "tokenizer" / name, args.output / "tokenizer" / name)
    shutil.copyfile(root / "LICENSE.md", args.output / "LICENSE.md")
    (args.output / "NOTICE").write_text(notice(args.size))
    files = {str(p.relative_to(args.output)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in tree_files(args.output) if str(p.relative_to(args.output)) not in NOT_BUNDLE}
    report["bundleBytes"] = sum(f["bytes"] for f in files.values())
    report["assetBytes"] = {role: tree_bytes(args.output / name) for role, name in ASSETS.items()}
    report["stages"]["package"] = log.done("package")
    save()
    (args.output / ".aether").mkdir()
    manifest_path = args.output / ".aether/aether.json"
    manifest_path.write_text(json.dumps(manifest(files, tables, args.size, report), indent=2, ensure_ascii=False) + "\n")
    report["aetherJSONSHA256"] = sha256(manifest_path)
    save()
    sums = [f"{sha256(p)}  {p.relative_to(args.output)}" for p in tree_files(args.output)
            if p.name not in ("SHA256SUMS", "forge-log.jsonl")]
    (args.output / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(json.dumps({"conversion": report["conversion"], "bundleBytes": report["bundleBytes"],
                      "peakRSSGB": report["peakRSSGB"], "wallSeconds": report["wallSeconds"]}, indent=1))


if __name__ == "__main__":
    main()

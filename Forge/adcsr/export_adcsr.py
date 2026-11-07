#!/usr/bin/env python3
"""Export AdcSR x4 (Guaishou74851/AdcSR; weights CreativeML Open RAIL++-M) as an Aether-Kit `image-to-image-tiled@1`
bundle.

    uv sync --frozen --project Forge
    Forge/.venv/bin/python -I Forge/adcsr/fetch_upstream.py           # pinned checkpoints, SD 2.1 base config, photos
    Forge/.venv/bin/python -I Forge/adcsr/export_adcsr.py --output <new bundle dir> [--precision float16|float32]
    Forge/.venv/bin/python -I Forge/adcsr/export_adcsr.py --sample-manifest   # the manifest for placeholder files

Nothing is downloaded: every file read is verified against `source-pins.json`. Conversion runs on the CPU. The output
directory is laid out as an installed revision root:

    OUTPUT/
      .aether/aether.json          the Aether manifest (MANIFEST.md §1.3, §2; driver `image-to-image-tiled@1`)
      adcsr_x4.aimodel             lr [1,3,128,128] float32 on [-1,1] -> sr [1,3,512,512] float32 (raw)
      LICENSE                      CreativeML Open RAIL++-M (the weights' license, SD 2.1 base's, verbatim)
      LICENSE-APACHE-2.0           AdcSR's code license (upstream's LICENSE, verbatim)
      NOTICE                       attribution, the modification notice and the deployer notice (Attachment A)
      README.md                    the Hugging Face model card
      SHA256SUMS                   every bundle file except itself and the log
      export-report.json           provenance, versions, timings, peak RSS
      forge-log.jsonl              one event per stage

Deterministic in what it computes: seeds are fixed, the graph has no randomness, and independent exports produce
bit-identical outputs on the GPU (measured on three exports). The bytes are not reproducible. coreai-core orders the
graph section of main.mlirb differently from run to run (the weights region matches), and it stamps metadata.json
with its own wall-clock creationDate, so compare re-exports by output, not by SHA-256. One Core AI JIT package serves
macOS and iOS: `--platform` only sets the manifest's `variant`. Parity: `reference_adcsr.py` (upstream on the CPU, float32)
records the references; `check_adcsr.py` runs the exported graph through Core AI on the GPU and gates it against them
(PSNR >= 40 dB).
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import platform
import re
import resource
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adcsr_graphs as ag  # noqa: E402

ID = "adcsr-x4"
HF_REPO = "aether-models/aether-kit-adcsr-x4"
ASSET = "adcsr_x4.aimodel"
FUNCTION = "main"
INPUTS, OUTPUTS = ["lr"], ["sr"]
NOT_BUNDLE = {".aether/aether.json", "SHA256SUMS", "export-report.json", "forge-log.jsonl"}
COMPUTE = "gpu"               # measured default (MANIFEST §0); the graph is fully static
LICENSE_SPDX = "CreativeML-OpenRAIL++-M"
LICENSE_FILES = ["LICENSE", "LICENSE-APACHE-2.0", "NOTICE"]


class Log:
    def __init__(self, path: Path) -> None:
        self.path, self.started = path, {}

    def event(self, stage: str, message: str, **fields) -> None:
        record = {"message": message, "stage": stage, "t": round(time.time(), 3), **fields}
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"[adcsr:{stage}] {message}" + (f" {fields}" if fields else ""), file=sys.stderr, flush=True)

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


def convert(module, example: tuple, path: Path, precision: str) -> dict:
    import coreai_torch as ct
    from coreai.runtime import AIModelAssetMetadata

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
    metadata.author = "Bin Chen, Gehui Li, Rongyuan Wu et al. (AdcSR, CVPR 2025); Core AI export: Aether-Kit Forge"
    metadata.license = f"{LICENSE_SPDX} (weights, Stable Diffusion 2.1 base derivatives); Apache-2.0 (code)"
    metadata.model_description = (f"AdcSR x4 super-resolution ({ag.REPO}@{ag.REVISION[:12]}), one {ag.TILE}x{ag.TILE} "
                                  f"tile: lr [1,3,{ag.TILE},{ag.TILE}] on [-1,1] -> sr [1,3,{ag.SR_TILE},{ag.SR_TILE}] "
                                  f"raw ({precision}; the host tiles, blends and colour-matches).")
    metadata.creation_date = int(time.time())
    program.save_asset(path, metadata)
    return {"exportSeconds": round(t1 - t0, 1), "convertSeconds": round(time.monotonic() - t1, 1),
            "bytes": tree_bytes(path), "inputs": INPUTS, "outputs": OUTPUTS, "function": FUNCTION}


# MARK: the manifest (MANIFEST.md §1.3 `image-to-image-tiled@1`, §2.5 preprocess.image, §2.8 postprocess)

def precision_block(precision: str) -> dict:
    if precision == "float32":
        return {"weights": "fp32", "activations": "fp32",
                "components": {"network": "fp32 throughout", "io": "lr and sr float32"}}
    return {"weights": "fp16, fp32 for the decoder's last two up blocks and head",
            "activations": "fp16, fp32 from the decoder's third up block on",
            "components": {"unet": "fp16 (with the PixelUnshuffle input and the decoder mid block)",
                           "decoderUpBlocks01": "fp16",
                           "decoderUpBlocks23AndHead": ("fp32: activations reach 72372 (beyond fp16's 65504) in up "
                                                        "block 3 and 33849 in up block 2 on the reference tiles"),
                           "io": "lr and sr float32; the graph casts at its boundaries"}}


def precision_words(precision: str) -> str:
    if precision == "float32":
        return "float32 weights and compute"
    return ("float16 weights and compute, except the decoder's last two up blocks and output head, which stay float32 "
            "(their activations exceed float16's range)")


def manifest(files: dict[str, dict], report: dict, platform_name: str, precision: str) -> dict:
    return {
        "schema": 1,
        "id": ID,
        "task": "image-to-image",
        "driver": "image-to-image-tiled@1",
        "variant": {"platform": platform_name},
        "assets": {"main": {"subtree": "", "files": [{"path": p, **meta} for p, meta in sorted(files.items())]}},
        "graphs": {
            "upscaler": {
                "file": f"main:{ASSET}", "function": FUNCTION,
                "inputs": {"image": "lr"},
                "outputs": {"image": "sr"},
                "compute": COMPUTE, "expectFrequentReshapes": False,
            },
        },
        "preprocess": {
            "image": {
                "colorSpace": "deviceRGB", "channelOrder": "RGB",
                "scale": 1 / 255, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
                "layout": "NCHW", "hostDType": "float32", "orientation": "exif",
                "tiling": {"tile": [ag.TILE, ag.TILE], "overlap": ag.OVERLAP, "scale": ag.SCALE,
                           "order": "rowMajor", "edge": "clamp", "blend": "featherLinear",
                           "colorMatch": "stitched"},
                "x-notes": ("No resize: the whole decoded image is upscaled. Upstream (test.py) feeds "
                            "ToTensor(RGB) * 2 - 1 on Pillow's decode; that is (x / 255 - 0.5) / 0.5. "
                            "tiling (proposed; MANIFEST.md §2.5 defers it): " + "; ".join(
                                f"{k}: {v}" for k, v in ag.TILING.items() if isinstance(v, str)) + "."),
            },
        },
        "postprocess": {
            "kind": "superRes",
            "sr": {"range": list(ag.RANGE), "colorMatch": "host", "colorMatchStd": "unbiased",
                   "clamp": True, "quantize": "truncate"},
            "x-notes": ("Upstream test.py, applied once to the stitched image: per channel over H and W, "
                        "(SR - mean(SR)) / std(SR) * std(LR) + mean(LR), LR on [-1,1], std with N - 1 (torch's "
                        "default); then x / 2 + 0.5, clamped to [0,1], and to 8 bits as torchvision's ToPILImage "
                        "does: (x * 255) truncated. The graph's sr is raw (no colour match, no clamp)."),
        },
        "precision": precision_block(precision),
        "license": LICENSE_SPDX,
        "x-license": {"files": LICENSE_FILES, "modified": True,
                      "weights": f"{LICENSE_SPDX} (Stable Diffusion 2.1 base derivatives)",
                      "code": "Apache-2.0 (AdcSR; LICENSE-APACHE-2.0)",
                      "useRestrictions": "LICENSE Attachment A (paragraph 5); deployers must pass them on (NOTICE)"},
        "provenance": {
            "source": ag.REPO, "sourceRevision": ag.REVISION,
            "base": ag.BASE["model"],
            "producer": "aether-kit Forge/adcsr/export_adcsr.py",
            "exportScript": report["scriptSHA256"]["export_adcsr.py"],
            "refs": {"upstreamCode": f"github.com/{ag.CODE_REPO}@{ag.CODE_COMMIT}",
                     "baseConfig": f"{ag.BASE['config']['repository']}@{ag.BASE['config']['revision']}",
                     "baseLicense": f"{ag.BASE['licenseText']['repository']}@{ag.BASE['licenseText']['revision']}",
                     "forgeLockSHA256": report["forgeLockSHA256"], "exportReport": "export-report.json"},
        },
    }


def sample_manifest(platform_name: str, precision: str = "float16") -> dict:
    files = {f"{ASSET}/{name}": {"bytes": 1, "sha256": "0" * 64} for name in ("main.hash", "main.mlirb",
                                                                              "metadata.json")}
    files.update({name: {"bytes": 1, "sha256": "0" * 64} for name in LICENSE_FILES + ["README.md"]})
    report = {"scriptSHA256": {"export_adcsr.py": "0" * 64}, "forgeLockSHA256": "0" * 64}
    return manifest(files, report, platform_name, precision)


def use_restrictions(license_text: str) -> str:
    """Attachment A of the pinned license, verbatim: from its heading line to the end."""
    match = re.search(r"^Attachment A[ \t]*$", license_text, flags=re.MULTILINE)
    if match is None:
        raise SystemExit("the pinned license has no 'Attachment A' heading")
    return license_text[match.start():].strip()


def wrap(*paragraphs: str, width: int = 114) -> list[str]:
    lines: list[str] = []
    for paragraph in paragraphs:
        lines += textwrap.wrap(paragraph, width=width) + [""]
    return lines


def notice(precision: str, license_text: str) -> str:
    lines = ["AdcSR x4 super-resolution (Aether-Kit Core AI export)", ""]
    lines += wrap(
        "Weights: Copyright the AdcSR authors (Bin Chen, Gehui Li, Rongyuan Wu, Xindong Zhang, Jie Chen, Jian Zhang, "
        "Lei Zhang; Peking University, The Hong Kong Polytechnic University, OPPO Research Institute), "
        f"{ag.REPO} at revision {ag.REVISION} (weight/net_params_200.pkl and weight/pretrained/halfDecoder.ckpt). "
        "They are derivatives of Stable Diffusion 2.1 base (Stability AI, stabilityai/stable-diffusion-2-1-base): the "
        "U-Net is SD 2.1 base's, channel-pruned and distilled, and the decoder was trained to decode SD 2.1's "
        "latents. As Derivatives of the Model they are licensed under the CreativeML Open RAIL++-M License, dated "
        "November 24, 2022 (LICENSE, verbatim), including its use-based restrictions (paragraph 5, Attachment A).",
        "Code: the network was exported with AdcSR's own model code (model.py, forward.py; Copyright the AdcSR "
        "authors; Apache License 2.0, LICENSE-APACHE-2.0), which builds on Hugging Face Diffusers (Apache-2.0).",
        "Modifications (CreativeML Open RAIL++-M paragraph 4): Aether-Kit's Forge exporter changed the files. It "
        "assembled the network as upstream's test.py does and re-expressed it as one static Core AI graph for one "
        f"{ag.TILE} x {ag.TILE} input tile (x4: {ag.SR_TILE} x {ag.SR_TILE} output), with "
        f"{precision_words(precision)}. Larger images are tiled and blended by the host, which also applies "
        "upstream's colour match; on images larger than one tile the result differs from upstream's whole-image "
        "inference. This is not an official AdcSR or Stability AI release and is not endorsed by either.",
    )
    lines += ["NOTICE TO DEPLOYERS AND USERS", ""]
    lines += wrap(
        "This model is subject to the use-based restrictions in paragraph 5 and Attachment A of the CreativeML Open "
        "RAIL++-M License (LICENSE). You may not use the model, or a derivative of it, for any restricted use. If you "
        "distribute, host or deploy this model or a derivative (for example, in an app that runs it on a device, or "
        "a service that offers it remotely), the license requires you to:")[:-1]
    for item in (
        "(a) include the use-based restrictions below as an enforceable provision in every agreement (for example, "
        "your end-user license or terms of service) that governs the use or distribution of the model;",
        "(b) give notice to the users you distribute to that the model is subject to paragraph 5, and require every "
        "user of the model to comply with it;",
        "(c) give every third-party recipient of the model a copy of the license (LICENSE);",
        "(d) keep the notices in this file, and mark any further changes you make to the model's files.",
    ):
        lines += textwrap.wrap(item, width=114, initial_indent="  ", subsequent_indent="      ")
    lines.append("")
    lines += wrap(
        "Outputs: the licensor claims no rights in the images you generate, but you are accountable for them, and no "
        "use of an output may contravene the license (paragraph 6).",
        "The use-based restrictions, verbatim from LICENSE:")
    return "\n".join(lines) + "\n" + use_restrictions(license_text) + "\n"


def readme(platform_names: list[str], precision: str) -> str:
    head = ["---", "license: openrail++", f"base_model: {ag.REPO}", "pipeline_tag: image-to-image",
            "tags: [aether-kit, core-ai, super-resolution, adcsr]", "---", "",
            "# AdcSR x4 super-resolution for Aether-Kit (Core AI)", ""]
    body = wrap(
        f"[AdcSR](https://github.com/{ag.CODE_REPO}) (CVPR 2025, \"Adversarial Diffusion Compression for Real-World "
        f"Image Super-Resolution\"; weights [{ag.REPO}](https://huggingface.co/{ag.REPO}) at revision "
        f"`{ag.REVISION[:8]}`), exported by [Aether-Kit](https://github.com/bubstackdev/aether-kit)'s Forge exporter "
        f"(`Forge/adcsr/`). The bundle holds one static Core AI graph for one tile: a {ag.TILE} x {ag.TILE} RGB tile "
        f"on [-1, 1] to its raw x4 output, {ag.SR_TILE} x {ag.SR_TILE}, with {precision_words(precision)}. Load it "
        f"with Aether-Kit's `AetherVision` (`image-to-image-tiled@1`, {' and '.join(platform_names)}); the manifest "
        f"is `.aether/aether.json`. The driver tiles larger images ({ag.OVERLAP}-pixel overlap, linear feather "
        "blend), then applies upstream's per-image colour match once to the stitched result. Every tile is gated "
        "against the upstream model on the CPU in float32 (PSNR >= 40 dB; `Forge/adcsr/check_adcsr.py`).")
    body += ["## License and use restrictions", ""]
    body += wrap(
        "The weights are derivatives of Stable Diffusion 2.1 base and are licensed under the **CreativeML Open "
        "RAIL++-M License** (`LICENSE`), which restricts how the model may be used (Attachment A). AdcSR's code is "
        "Apache-2.0 (`LICENSE-APACHE-2.0`). Deployers: the license requires you to include the use restrictions as "
        "an enforceable provision in your own terms, to give your users notice of them and a copy of the license, "
        "and to keep the notices. `NOTICE` lists these obligations and reproduces the restrictions. This is a "
        "modified version (see `NOTICE`), not an official AdcSR or Stability AI release.")
    return "\n".join(head + body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="new bundle directory")
    parser.add_argument("--platform", choices=["macos", "ios"], default="macos", help="the manifest's variant")
    parser.add_argument("--precision", choices=sorted(ag.DTYPES), default="float16",
                        help="float16: fp16 except the decoder's last two up blocks and head (fp32); float32: all fp32")
    parser.add_argument("--sample-manifest", action="store_true", help="print the manifest for placeholder files")
    parser.add_argument("--snapshot", type=Path, default=ag.DEFAULT_SNAPSHOT)
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
    lock = HERE.parent / "uv.lock"
    report = {
        "schema": 1, "id": ID, "command": sys.argv,
        "exportedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {"os": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
                 "machine": platform.machine(), "python": platform.python_version()},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "coreai-torch", "coreai-core", "numpy",
                                                                "diffusers")},
        "forgeLockSHA256": ag.sha256_file(lock), "forgePyprojectSHA256": ag.sha256_file(HERE.parent / "pyproject.toml"),
        "scriptSHA256": {p.name: ag.sha256_file(p) for p in sorted(HERE.glob("*.py"))},
        "source": {"repository": ag.REPO, "revision": ag.REVISION, "snapshot": str(args.snapshot.resolve()),
                   "upstreamCode": {"repository": ag.CODE_REPO, "commit": ag.CODE_COMMIT},
                   "base": ag.BASE["model"], "verified": {}},
        "precision": f"{precision_words(args.precision)}; lr and sr float32; no quantization",
        "inputShape": [1, 3, ag.TILE, ag.TILE], "outputShape": [1, 3, ag.SR_TILE, ag.SR_TILE],
        "tiling": ag.TILING, "compute": COMPUTE, "conversion": {}, "stages": {},
    }
    report_path = args.output / "export-report.json"

    def save() -> None:
        report["peakRSSGB"] = peak_rss_gb()
        report["wallSeconds"] = round(time.monotonic() - started, 1)
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    log.start("verify-sources")
    report["source"]["verified"] = ag.verify_sources(args.snapshot, images=False)
    report["stages"]["verify-sources"] = log.done("verify-sources")
    save()

    log.start("export")
    upstream = ag.build_upstream(args.snapshot)
    report["parameters"] = ag.parameter_count(upstream)
    report["precisionSplit"] = ag.precision_split(upstream, ag.DTYPES[args.precision])
    graph = ag.TileGraph(upstream, ag.DTYPES[args.precision]).eval()
    example = (torch.from_numpy(ag.lr_tensor(ag.synthetic_lr(0, ag.TILE, ag.TILE))),)
    report["conversion"]["upscaler"] = convert(graph, example, args.output / ASSET, args.precision)
    del graph, upstream
    report["stages"]["export"] = log.done("export")
    save()

    log.start("package")
    license_path = ag.DEFAULT_BASE_LICENSE / "LICENSE-MODEL"
    shutil.copyfile(license_path, args.output / "LICENSE")
    shutil.copyfile(args.snapshot / "LICENSE", args.output / "LICENSE-APACHE-2.0")
    (args.output / "NOTICE").write_text(notice(args.precision, license_path.read_text()))
    (args.output / "README.md").write_text(readme(["macOS"], args.precision))
    files = {str(p.relative_to(args.output)): {"bytes": p.stat().st_size, "sha256": ag.sha256_file(p)}
             for p in tree_files(args.output) if str(p.relative_to(args.output)) not in NOT_BUNDLE}
    report["bundleBytes"] = sum(f["bytes"] for f in files.values())
    report["assetBytes"] = {"upscaler": tree_bytes(args.output / ASSET)}
    report["stages"]["package"] = log.done("package")
    save()
    (args.output / ".aether").mkdir()
    manifest_path = args.output / ".aether/aether.json"
    manifest_path.write_text(json.dumps(manifest(files, report, args.platform, args.precision), indent=2) + "\n")
    report["aetherJSONSHA256"] = ag.sha256_file(manifest_path)
    save()
    sums = [f"{ag.sha256_file(p)}  {p.relative_to(args.output)}" for p in tree_files(args.output)
            if p.name not in ("SHA256SUMS", "forge-log.jsonl")]
    (args.output / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(json.dumps({"conversion": report["conversion"], "bundleBytes": report["bundleBytes"],
                      "peakRSSGB": report["peakRSSGB"], "wallSeconds": report["wallSeconds"]}, indent=1))


if __name__ == "__main__":
    main()

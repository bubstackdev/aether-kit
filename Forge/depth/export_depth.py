#!/usr/bin/env python3
"""Export Depth Anything 3 small (depth-anything/DA3-SMALL, Apache-2.0) as an Aether-Kit `depth-estimator@1` bundle.

    uv sync --frozen --project Forge
    Forge/.venv/bin/python -I Forge/depth/fetch_upstream.py           # pinned upstream code + checkpoint, once
    Forge/.venv/bin/python -I Forge/depth/export_depth.py --output <new bundle dir>
    Forge/.venv/bin/python -I Forge/depth/export_depth.py --sample-manifest   # the manifest for placeholder files

Nothing is downloaded: every file read is verified against `source-pins.json`. Conversion runs on the CPU. The output
directory is laid out as an installed revision root:

    OUTPUT/
      .aether/aether.json                     the Aether manifest (MANIFEST.md §1.3, §2; driver `depth-estimator@1`)
      depth_anything_3_small.aimodel          image [1,3,504,504] fp16 on [0,1] -> depth, depth_conf [1,504,504] fp32
      LICENSE, NOTICE                         Apache-2.0 (upstream's LICENSE) and the modification notice
      README.md                               the Hugging Face model card
      SHA256SUMS                              every bundle file except itself and the log
      export-report.json                      provenance, versions, timings, peak RSS
      forge-log.jsonl                         one event per stage

One Core AI JIT package serves macOS and iOS (no per-architecture AOT): `--platform` only sets the manifest's
`variant`. Parity: `reference_depth.py` (the unmodified upstream network on the CPU, float32) records the references;
`check_depth.py` runs the exported graph through Core AI on the GPU and gates it against them (rel L2 <= 0.02).
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
import depth_graphs as dg  # noqa: E402

ID = "depth-anything-3-small"
HF_REPO = "aether-models/aether-kit-depth-anything-3-small"
ASSET = "depth_anything_3_small.aimodel"
FUNCTION = "main"
INPUTS, OUTPUTS = ["image"], ["depth", "depth_conf"]
NOT_BUNDLE = {".aether/aether.json", "SHA256SUMS", "export-report.json", "forge-log.jsonl"}
# Measured default (MANIFEST §0): GPU for every graph class; the graph is fully static.
COMPUTE = "gpu"


class Log:
    def __init__(self, path: Path) -> None:
        self.path, self.started = path, {}

    def event(self, stage: str, message: str, **fields) -> None:
        record = {"message": message, "stage": stage, "t": round(time.time(), 3), **fields}
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"[depth:{stage}] {message}" + (f" {fields}" if fields else ""), file=sys.stderr, flush=True)

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


def convert(module, example: tuple, path: Path) -> dict:
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
    metadata.author = "ByteDance Seed (Depth Anything 3); Core AI export: Aether-Kit Forge"
    metadata.license = "Apache-2.0"
    metadata.model_description = (f"Depth Anything 3 small ({dg.REPO}@{dg.REVISION[:12]}), single view: image "
                                  "[1,3,504,504] RGB on [0,1] -> relative depth (larger is farther) and confidence.")
    metadata.creation_date = int(time.time())
    program.save_asset(path, metadata)
    return {"exportSeconds": round(t1 - t0, 1), "convertSeconds": round(time.monotonic() - t1, 1),
            "bytes": tree_bytes(path), "inputs": INPUTS, "outputs": OUTPUTS, "function": FUNCTION}


# MARK: the manifest (MANIFEST.md §1.3 `depth-estimator@1`, §2.5 preprocess.image, §2.8 postprocess)

def manifest(files: dict[str, dict], report: dict, platform_name: str) -> dict:
    return {
        "schema": 1,
        "id": ID,
        "task": "depth-estimation",
        "driver": "depth-estimator@1",
        "variant": {"platform": platform_name},
        "assets": {"main": {"subtree": "", "files": [{"path": p, **meta} for p, meta in sorted(files.items())]}},
        "graphs": {
            "estimator": {
                "file": f"main:{ASSET}", "function": FUNCTION,
                "inputs": {"image": "image"},
                "outputs": {"depth": "depth", "confidence": "depth_conf"},
                "compute": COMPUTE, "expectFrequentReshapes": False,
            },
        },
        "preprocess": {
            "image": {
                "resize": {"mode": "stretch", "size": [dg.SIZE, dg.SIZE], "filter": "pillowBicubic"},
                "colorSpace": "sRGB", "channelOrder": "RGB",
                "scale": 1 / 255, "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0],
                "layout": "NCHW", "hostDType": "float32", "orientation": "exif",
                "x-notes": ("ImageNet normalization (mean 0.485/0.456/0.406, std 0.229/0.224/0.225) is in the graph: "
                            "feed RGB on [0,1]. The graph input is float16; the host builds float32 and converts. "
                            "Upstream resizes the long side to 504 (cv2 INTER_AREA down, INTER_CUBIC up) and rounds "
                            "to multiples of 14; this static graph takes a 504 x 504 stretch, with Pillow's bicubic "
                            "(the gate reference's resize, matched bit for bit)."),
            },
        },
        "postprocess": {
            "kind": "depth",
            "depth": {"polarity": "largerIsFarther", "resizeBack": True},
            "x-notes": ("depth: exp-activated relative depth (affine-invariant up to scale; not metric), larger is "
                        "farther. confidence: exp(x) + 1, >= 1, larger is more confident. Both [1,504,504] float32, "
                        "row-major over the stretched 504 x 504 input."),
        },
        "precision": {"weights": "fp16 backbone, fp32 head", "activations": "fp16 backbone, fp32 head",
                      "components": {"backbone": "fp16 (upstream runs it under fp16/bf16 autocast)",
                                     "head": "fp32 (upstream disables autocast for it)",
                                     "io": "image float16; depth and depth_conf float32"}},
        "license": "Apache-2.0",
        "x-license": {"files": ["LICENSE", "NOTICE"], "modified": True},
        "provenance": {
            "source": dg.REPO, "sourceRevision": dg.REVISION,
            "producer": "aether-kit Forge/depth/export_depth.py",
            "exportScript": report["scriptSHA256"]["export_depth.py"],
            "refs": {"upstreamCode": f"{dg.CODE_REPO}@{dg.CODE_COMMIT}",
                     "forgeLockSHA256": report["forgeLockSHA256"], "exportReport": "export-report.json"},
        },
    }


def sample_manifest(platform_name: str) -> dict:
    files = {f"{ASSET}/{name}": {"bytes": 1, "sha256": "0" * 64} for name in ("main.hash", "main.mlirb",
                                                                              "metadata.json")}
    files.update({name: {"bytes": 1, "sha256": "0" * 64} for name in ("LICENSE", "NOTICE", "README.md")})
    report = {"scriptSHA256": {"export_depth.py": "0" * 64}, "forgeLockSHA256": "0" * 64}
    return manifest(files, report, platform_name)


def notice() -> str:
    return "\n".join([
        "Depth Anything 3 small (Aether-Kit Core AI export)",
        "",
        f"This bundle is a modified version of {dg.REPO} at revision {dg.REVISION},",
        "by ByteDance Seed (Depth Anything 3; https://github.com/ByteDance-Seed/Depth-Anything-3), licensed under the",
        "Apache License, Version 2.0 (LICENSE). The model code it was exported with is upstream's",
        f"({dg.CODE_REPO} at commit {dg.CODE_COMMIT}, Apache-2.0), including its DINOv2",
        "vision transformer code (Copyright (c) Meta Platforms, Inc. and affiliates, Apache-2.0).",
        "",
        "Modifications: Aether-Kit's Forge exporter re-expressed the single-view depth path as one static Core AI graph",
        "(504 x 504 input; ImageNet normalization folded in; camera, ray-pose, Gaussian-splatting and multi-view",
        "reference-selection paths removed; the backbone cast to float16 and the head kept in float32; the rotary",
        "position table sized statically). It is not an official release of Depth Anything 3.",
        "",
    ])


def readme(platform_names: list[str]) -> str:
    return "\n".join([
        "---",
        "license: apache-2.0",
        f"base_model: {dg.REPO}",
        "pipeline_tag: depth-estimation",
        "tags: [aether-kit, core-ai, depth-estimation, depth-anything-3]",
        "---",
        "",
        "# Depth Anything 3 small for Aether-Kit (Core AI)",
        "",
        f"[{dg.REPO}](https://huggingface.co/{dg.REPO}) at revision `{dg.REVISION[:8]}`, exported by",
        "[Aether-Kit](https://github.com/bubstackdev/aether-kit)'s Forge exporter (`Forge/depth/`) as one static Core",
        "AI graph: an RGB image stretched to 504 x 504 on [0, 1] (ImageNet normalization is in the graph) to relative",
        "depth (larger is farther) and a confidence map, 504 x 504 each. The DINOv2 ViT-S backbone runs in float16 and",
        "the DualDPT head in float32, as upstream's own inference does. Load it with Aether-Kit's `AetherVision`",
        f"(`depth-estimator@1`, {' and '.join(platform_names)}); the manifest is `.aether/aether.json`. Gated against",
        "the upstream implementation on CPU in float32 (see `export-report.json`).",
        "",
        "Apache-2.0 (`LICENSE`). A modified version (see `NOTICE`).",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="new bundle directory")
    parser.add_argument("--platform", choices=["macos", "ios"], default="macos", help="the manifest's variant")
    parser.add_argument("--sample-manifest", action="store_true", help="print the manifest for placeholder files")
    parser.add_argument("--snapshot", type=Path, default=dg.DEFAULT_SNAPSHOT)
    parser.add_argument("--upstream", type=Path, default=dg.DEFAULT_UPSTREAM, help="fetch_upstream.py's unpack")
    args = parser.parse_args()
    if args.sample_manifest:
        print(json.dumps(sample_manifest(args.platform), indent=2))
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
                                                                "safetensors", "einops", "omegaconf", "addict")},
        "forgeLockSHA256": dg.sha256_file(lock), "forgePyprojectSHA256": dg.sha256_file(HERE.parent / "pyproject.toml"),
        "scriptSHA256": {p.name: dg.sha256_file(p) for p in sorted(HERE.glob("*.py"))},
        "source": {"repository": dg.REPO, "revision": dg.REVISION, "snapshot": str(args.snapshot.resolve()),
                   "upstreamCode": {"repository": dg.CODE_REPO, "commit": dg.CODE_COMMIT}, "verified": {}},
        "precision": "backbone float16, head float32; input float16, outputs float32; no quantization",
        "inputShape": [1, 3, dg.SIZE, dg.SIZE], "compute": COMPUTE, "conversion": {}, "stages": {},
    }
    report_path = args.output / "export-report.json"

    def save() -> None:
        report["peakRSSGB"] = peak_rss_gb()
        report["wallSeconds"] = round(time.monotonic() - started, 1)
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    log.start("verify-sources")
    report["source"]["verified"] = dg.verify_sources(args.snapshot, args.upstream)
    report["stages"]["verify-sources"] = log.done("verify-sources")
    save()

    log.start("export")
    graph = dg.DepthGraph(dg.build_net(args.upstream, args.snapshot), torch.float16).eval()
    example = (torch.from_numpy(dg.synthetic_tensor(0)).to(torch.float16),)
    with dg.export_patches(args.upstream):
        report["conversion"]["estimator"] = convert(graph, example, args.output / ASSET)
    del graph
    report["stages"]["export"] = log.done("export")
    save()

    log.start("package")
    shutil.copyfile(args.upstream / "LICENSE", args.output / "LICENSE")
    (args.output / "NOTICE").write_text(notice())
    (args.output / "README.md").write_text(readme(["macOS"]))
    files = {str(p.relative_to(args.output)): {"bytes": p.stat().st_size, "sha256": dg.sha256_file(p)}
             for p in tree_files(args.output) if str(p.relative_to(args.output)) not in NOT_BUNDLE}
    report["bundleBytes"] = sum(f["bytes"] for f in files.values())
    report["assetBytes"] = {"estimator": tree_bytes(args.output / ASSET)}
    report["stages"]["package"] = log.done("package")
    save()
    (args.output / ".aether").mkdir()
    manifest_path = args.output / ".aether/aether.json"
    manifest_path.write_text(json.dumps(manifest(files, report, args.platform), indent=2) + "\n")
    report["aetherJSONSHA256"] = dg.sha256_file(manifest_path)
    save()
    sums = [f"{dg.sha256_file(p)}  {p.relative_to(args.output)}" for p in tree_files(args.output)
            if p.name not in ("SHA256SUMS", "forge-log.jsonl")]
    (args.output / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(json.dumps({"conversion": report["conversion"], "bundleBytes": report["bundleBytes"],
                      "peakRSSGB": report["peakRSSGB"], "wallSeconds": report["wallSeconds"]}, indent=1))


if __name__ == "__main__":
    main()

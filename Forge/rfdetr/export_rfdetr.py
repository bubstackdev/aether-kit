#!/usr/bin/env python3
"""Export RF-DETR nano (or seg-nano) (roboflow/rf-detr, Apache-2.0) as an Aether-Kit `detector-detr@1` bundle.

    uv sync --frozen --project Forge
    Forge/.venv/bin/python -I Forge/rfdetr/fetch_upstream.py           # pinned upstream code, checkpoints, photos, once
    Forge/.venv/bin/python -I Forge/rfdetr/export_rfdetr.py --output <new bundle dir> [--variant nano|seg-nano]
    Forge/.venv/bin/python -I Forge/rfdetr/export_rfdetr.py --sample-manifest   # the manifest for placeholder files

Nothing is downloaded: every file read is verified against `source-pins.json`. Conversion runs on the CPU. The output
directory is laid out as an installed revision root:

    OUTPUT/
      .aether/aether.json          the Aether manifest (MANIFEST.md §1.3, §2; driver `detector-detr@1`)
      rf_detr_nano.aimodel         image [1,3,384,384] f32 on [0,1] -> dets [1,300,4], labels [1,300,91] f32
                                   (seg-nano: rf_detr_seg_nano.aimodel, 312 x 312, 100 queries, + masks [1,100,78,78])
      labels.json                  the 91 class-logit columns' COCO names (null for unused ids)
      LICENSE, NOTICE              Apache-2.0 (upstream's LICENSE) and the modification notice
      README.md                    the Hugging Face model card
      SHA256SUMS                   every bundle file except itself and the log
      export-report.json           provenance, versions, timings, peak RSS
      forge-log.jsonl              one event per stage

The shipped precision is float32 throughout (`--precision float16` exists to measure the float16 graph, which moves the
two-stage query selection; see export-report.json and the check reports). One Core AI JIT package serves macOS and iOS
(no per-architecture AOT): `--platform` only sets the manifest's `variant`. Parity: `reference_rfdetr.py` (the
unmodified upstream network on the CPU, float32) records the references; `check_rfdetr.py` runs the exported graph
through Core AI on the GPU and gates it against them at the detection level (docs/M4.md).

Reproducibility: two exports give byte-identical weights and bit-identical GPU outputs, but not a byte-identical
main.mlirb: coreai's MLIR builder suffixes template function names with `secrets.choice` letters and its bytecode
trailer varies run to run (metadata.json also carries the save time).
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
import rfdetr_graphs as rg  # noqa: E402

FUNCTION = "main"
INPUTS = ["image"]
HF_ORG = "aether-models"
NOT_BUNDLE = {".aether/aether.json", "SHA256SUMS", "export-report.json", "forge-log.jsonl"}
# Measured default (MANIFEST §0): GPU for every graph class; the graph is fully static.
COMPUTE = "gpu"
ROLES = {"dets": "boxes", "labels": "classLogits", "masks": "masks"}


class Log:
    def __init__(self, path: Path) -> None:
        self.path, self.started = path, {}

    def event(self, stage: str, message: str, **fields) -> None:
        record = {"message": message, "stage": stage, "t": round(time.time(), 3), **fields}
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"[rfdetr:{stage}] {message}" + (f" {fields}" if fields else ""), file=sys.stderr, flush=True)

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


def convert(module, example: tuple, path: Path, variant: str, upstream: Path, precision: str) -> dict:
    import coreai_torch as ct
    from coreai.runtime import AIModelAssetMetadata

    spec = rg.VARIANTS[variant]
    t0 = time.monotonic()
    with torch.no_grad():
        ep = torch.export.export(module, example, strict=False)
        ep = ep.run_decompositions(rg.decomposition_table(upstream))
    survivors = sorted({str(n.target) for n in ep.graph.nodes
                        if n.op == "call_function" and str(n.target).startswith("aten.grid_sampler")})
    if survivors:
        raise SystemExit(f"grid sampling survived the decomposition table: {survivors}")
    t1 = time.monotonic()
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    converter.add_exported_program(ep, input_names=INPUTS, output_names=spec["outputs"], state_names=[],
                                   entrypoint_name=FUNCTION)
    del ep
    program = converter.to_coreai()
    del converter
    program.optimize()
    metadata = AIModelAssetMetadata()
    metadata.author = "Roboflow (RF-DETR); Core AI export: Aether-Kit Forge"
    metadata.license = "Apache-2.0"
    s, q = spec["size"], spec["queries"]
    metadata.model_description = (
        f"{spec['title']} ({rg.CODE_REPO} {rg.CODE_TAG}, COCO), {precision}: image [1,3,{s},{s}] RGB on [0,1] -> "
        f"dets [1,{q},4] cxcywh normalized, labels [1,{q},91] class logits (column = COCO id)"
        + (f", masks [1,{q},{s // 4},{s // 4}] mask logits" if "masks" in spec["outputs"] else "") + ".")
    metadata.creation_date = int(time.time())
    program.save_asset(path, metadata)
    return {"exportSeconds": round(t1 - t0, 1), "convertSeconds": round(time.monotonic() - t1, 1),
            "bytes": tree_bytes(path), "inputs": INPUTS, "outputs": spec["outputs"], "function": FUNCTION}


# MARK: the manifest (MANIFEST.md §1.3 `detector-detr@1`, §2.5 preprocess.image, §2.8 postprocess, labels)

def manifest(variant: str, files: dict[str, dict], report: dict, platform_name: str, precision: str) -> dict:
    spec = rg.VARIANTS[variant]
    s, q = spec["size"], spec["queries"]
    postprocess = {
        "kind": "detrSigmoid",
        "boxFormat": "cxcywh", "boxSpace": "normalized",
        "activation": "sigmoid",
        "classIndex": "cocoId",
        "defaults": {"scoreThreshold": rg.SCORE_THRESHOLD, "maxDetections": q},
        "x-selection": {"rule": "topKOverQueryClassPairs", "k": q, "order": "scoreDescendingThenFlatIndex",
                        "keep": "score > scoreThreshold"},
        "x-notes": (f"Upstream PostProcess: sigmoid over all {q} x 91 logits, the top {q} (query, class) pairs by "
                    "score (stable: ties by ascending flat index query * 91 + class), each kept when score > "
                    "threshold; one query can yield several classes. No NMS. Boxes are cxcywh normalized to the "
                    f"stretched {s} x {s} input, so to the source image; converted to xyxy and clamped to [0, 1]. "
                    "The class index is the logit column, which is the COCO category id (1..90; column 0 and the "
                    "ten unassigned ids never score)."),
    }
    if "masks" in spec["outputs"]:
        postprocess["masks"] = {"output": "masks", "stride": 4}
        postprocess["x-notes"] += (f" masks: per-query mask logits at {s // 4} x {s // 4} over the stretched input; "
                                   "upstream gathers the selected queries' planes, upsamples bilinearly "
                                   "(align_corners false) to the output size and keeps logit > 0.")
    outputs = {ROLES[name]: name for name in spec["outputs"]}
    weights_precision = "fp32" if precision == "float32" else "fp16"
    return {
        "schema": 1,
        "id": spec["id"],
        "task": spec["task"],
        "driver": "detector-detr@1",
        "variant": {"platform": platform_name},
        "assets": {"main": {"subtree": "", "files": [{"path": p, **meta} for p, meta in sorted(files.items())]}},
        "graphs": {
            "detector": {
                "file": f"main:{spec['asset']}", "function": FUNCTION,
                "inputs": {"image": "image"},
                "outputs": outputs,
                "compute": COMPUTE, "expectFrequentReshapes": False,
            },
        },
        "preprocess": {
            "image": {
                "resize": {"mode": "stretch", "size": [s, s], "filter": "vImage"},
                "colorSpace": "sRGB", "channelOrder": "RGB",
                "scale": 1 / 255, "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0],
                "layout": "NCHW", "hostDType": "float32", "orientation": "exif",
                "x-notes": ("ImageNet normalization (mean 0.485/0.456/0.406, std 0.229/0.224/0.225) is in the graph: "
                            f"feed RGB on [0,1]. Upstream predict stretches to {s} x {s} (no aspect preservation) "
                            "with torchvision bilinear resize, antialias=False, on the float image (it matches the "
                            "cv2.INTER_LINEAR resize the checkpoints were trained with); §2.5 has no filter value "
                            "for that, so the fast vImage path is named."),
            },
        },
        "postprocess": postprocess,
        "labels": {"set": "coco-91", "x-file": "main:labels.json",
                   "x-notes": "logit column i is COCO category id i; labels.json lists the names (null if unused)."},
        "precision": {"weights": weights_precision, "activations": weights_precision,
                      "components": {"io": "image float32; outputs float32",
                                     "normalization": "float32 (in graph)"}},
        "license": "Apache-2.0",
        "x-license": {"files": ["LICENSE", "NOTICE"], "modified": True},
        "provenance": {
            "source": f"{rg.CODE_REPO}@{rg.CODE_TAG}",
            "sourceRevision": rg.CODE_COMMIT,
            "producer": "aether-kit Forge/rfdetr/export_rfdetr.py",
            "exportScript": report["scriptSHA256"]["export_rfdetr.py"],
            "refs": {"checkpoint": report["source"]["checkpoint"],
                     "forgeLockSHA256": report["forgeLockSHA256"], "exportReport": "export-report.json"},
        },
    }


def sample_manifest(variant: str, platform_name: str, precision: str = "float32") -> dict:
    spec = rg.VARIANTS[variant]
    files = {f"{spec['asset']}/{name}": {"bytes": 1, "sha256": "0" * 64} for name in ("main.hash", "main.mlirb",
                                                                                     "metadata.json")}
    files.update({name: {"bytes": 1, "sha256": "0" * 64} for name in ("LICENSE", "NOTICE", "README.md",
                                                                      "labels.json")})
    pin = rg.PINS["checkpoints"][spec["checkpoint"]]
    report = {"scriptSHA256": {"export_rfdetr.py": "0" * 64}, "forgeLockSHA256": "0" * 64,
              "source": {"checkpoint": {"url": pin["url"], "sha256": pin["sha256"]}}}
    return manifest(variant, files, report, platform_name, precision)


def notice(variant: str, precision: str) -> str:
    spec = rg.VARIANTS[variant]
    pin = rg.PINS["checkpoints"][spec["checkpoint"]]
    return "\n".join([
        f"{spec['title']} (Aether-Kit Core AI export)",
        "",
        f"This bundle is a modified version of {spec['title']} by Roboflow (RF-DETR; https://github.com/roboflow/rf-detr),",
        "licensed under the Apache License, Version 2.0 (LICENSE): the COCO checkpoint",
        f"{pin['url']}",
        f"(SHA-256 {pin['sha256']}), exported with upstream's model code",
        f"({rg.CODE_REPO} {rg.CODE_TAG}, commit {rg.CODE_COMMIT}, Apache-2.0), which includes code derived from",
        "LW-DETR (Baidu), Conditional DETR (Microsoft), DETR (Facebook), Deformable DETR (SenseTime) and DINOv2 /",
        "Hugging Face Transformers (Meta Platforms; The HuggingFace Team), each under the Apache License 2.0.",
        "",
        "Modifications: Aether-Kit's Forge exporter re-expressed upstream's inference graph (LWDETR.forward_export) as",
        f"one static Core AI graph ({spec['size']} x {spec['size']} input; ImageNet normalization folded in;",
        f"grid sampling decomposed into gathers, as upstream's own Core AI exporter does; {precision}).",
        "It is not an official release of RF-DETR.",
        "",
    ])


def readme(variant: str, platform_names: list[str], precision: str) -> str:
    spec = rg.VARIANTS[variant]
    s, q = spec["size"], spec["queries"]
    extra = f", and per-query mask logits at {s // 4} x {s // 4}" if "masks" in spec["outputs"] else ""
    return "\n".join([
        "---",
        "license: apache-2.0",
        "pipeline_tag: " + ("image-segmentation" if "masks" in spec["outputs"] else "object-detection"),
        "tags: [aether-kit, core-ai, object-detection, rf-detr, coco]",
        "---",
        "",
        f"# {spec['title']} for Aether-Kit (Core AI)",
        "",
        f"Roboflow's [RF-DETR](https://github.com/roboflow/rf-detr) {spec['title'].split('RF-DETR ')[1]} COCO",
        f"checkpoint, exported with upstream's own model code ({rg.CODE_TAG}, `{rg.CODE_COMMIT[:8]}`) by",
        "[Aether-Kit](https://github.com/bubstackdev/aether-kit)'s Forge exporter (`Forge/rfdetr/`) as one static",
        f"Core AI graph ({precision}): an RGB image stretched to {s} x {s} on [0, 1] (ImageNet normalization is in the",
        f"graph) to {q} query boxes (cxcywh, normalized) and their 91 COCO class logits{extra}. The host applies",
        "upstream's postprocess (sigmoid, top-k over query x class pairs, score > 0.5; no NMS). Load it with",
        f"Aether-Kit's `AetherVision` (`detector-detr@1`, {' and '.join(platform_names)}); the manifest is",
        "`.aether/aether.json`. Gated against the upstream implementation on CPU in float32 (see `export-report.json`).",
        "",
        "Apache-2.0 (`LICENSE`). A modified version (see `NOTICE`).",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="new bundle directory")
    parser.add_argument("--variant", choices=sorted(rg.VARIANTS), default="nano")
    parser.add_argument("--precision", choices=sorted(rg.DTYPES), default="float32",
                        help="the graph's compute dtype (I/O stay float32); float32 is the shipped precision")
    parser.add_argument("--platform", choices=["macos", "ios"], default="macos", help="the manifest's variant")
    parser.add_argument("--sample-manifest", action="store_true", help="print the manifest for placeholder files")
    parser.add_argument("--upstream", type=Path, default=rg.DEFAULT_UPSTREAM, help="fetch_upstream.py's unpack")
    parser.add_argument("--weights", type=Path, default=rg.DEFAULT_WEIGHTS, help="fetch_upstream.py's checkpoints")
    args = parser.parse_args()
    if args.sample_manifest:
        print(json.dumps(sample_manifest(args.variant, args.platform, args.precision), indent=2))
        return
    if args.output is None:
        parser.error("--output is required")
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    spec = rg.VARIANTS[args.variant]
    started = time.monotonic()
    args.output.mkdir(parents=True)
    log = Log(args.output / "forge-log.jsonl")
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    lock = HERE.parent / "uv.lock"
    pin = rg.PINS["checkpoints"][spec["checkpoint"]]
    report = {
        "schema": 1, "id": spec["id"], "variant": args.variant, "command": sys.argv,
        "exportedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {"os": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
                 "machine": platform.machine(), "python": platform.python_version()},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "coreai-torch", "coreai-core",
                                                                "numpy", "transformers", "pydantic", "scipy")},
        "forgeLockSHA256": rg.sha256_file(lock), "forgePyprojectSHA256": rg.sha256_file(HERE.parent / "pyproject.toml"),
        "scriptSHA256": {p.name: rg.sha256_file(p) for p in sorted(HERE.glob("*.py"))},
        "source": {"upstreamCode": {"repository": rg.CODE_REPO, "tag": rg.CODE_TAG, "commit": rg.CODE_COMMIT},
                   "checkpoint": {"url": pin["url"], "sha256": pin["sha256"], "md5": pin["md5"]}, "verified": {}},
        "precision": f"{args.precision} compute; normalization float32; input and outputs float32; no quantization",
        "inputShape": [1, 3, spec["size"], spec["size"]], "outputShapes": rg.output_shapes(args.variant),
        "compute": COMPUTE, "conversion": {}, "stages": {},
    }
    report_path = args.output / "export-report.json"

    def save() -> None:
        report["peakRSSGB"] = peak_rss_gb()
        report["wallSeconds"] = round(time.monotonic() - started, 1)
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    log.start("verify-sources")
    report["source"]["verified"] = rg.verify_sources(args.variant, args.upstream, args.weights, images=None)
    report["stages"]["verify-sources"] = log.done("verify-sources")
    save()

    log.start("export")
    net, _, _ = rg.build_net(args.variant, args.upstream, args.weights)
    graph = rg.DetrGraph(net, rg.DTYPES[args.precision]).eval()
    del net
    example = (torch.from_numpy(rg.synthetic_tensor(0, spec["size"])),)
    report["conversion"]["detector"] = convert(graph, example, args.output / spec["asset"], args.variant,
                                               args.upstream, args.precision)
    del graph
    report["stages"]["export"] = log.done("export")
    save()

    log.start("package")
    shutil.copyfile(args.upstream / "LICENSE", args.output / "LICENSE")
    (args.output / "NOTICE").write_text(notice(args.variant, args.precision))
    (args.output / "README.md").write_text(readme(args.variant, ["macOS"], args.precision))
    (args.output / "labels.json").write_text(json.dumps(rg.coco_labels(args.upstream), indent=1) + "\n")
    files = {str(p.relative_to(args.output)): {"bytes": p.stat().st_size, "sha256": rg.sha256_file(p)}
             for p in tree_files(args.output) if str(p.relative_to(args.output)) not in NOT_BUNDLE}
    report["bundleBytes"] = sum(f["bytes"] for f in files.values())
    report["assetBytes"] = {"detector": tree_bytes(args.output / spec["asset"])}
    report["stages"]["package"] = log.done("package")
    save()
    (args.output / ".aether").mkdir()
    manifest_path = args.output / ".aether/aether.json"
    manifest_path.write_text(json.dumps(manifest(args.variant, files, report, args.platform, args.precision),
                                        indent=2) + "\n")
    report["aetherJSONSHA256"] = rg.sha256_file(manifest_path)
    save()
    sums = [f"{rg.sha256_file(p)}  {p.relative_to(args.output)}" for p in tree_files(args.output)
            if p.name not in ("SHA256SUMS", "forge-log.jsonl")]
    (args.output / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(json.dumps({"conversion": report["conversion"], "bundleBytes": report["bundleBytes"],
                      "peakRSSGB": report["peakRSSGB"], "wallSeconds": report["wallSeconds"]}, indent=1))


if __name__ == "__main__":
    main()

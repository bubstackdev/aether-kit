#!/usr/bin/env python3
"""Export YOLOX-S (Megvii-BaseDetection/YOLOX yolox_s.pth, Apache-2.0) as an Aether-Kit `detector-dense@1` bundle.

    uv sync --frozen --project Forge
    Forge/.venv/bin/python -I Forge/yolox/fetch_upstream.py           # pinned upstream code + checkpoint, once
    Forge/.venv/bin/python -I Forge/yolox/export_yolox.py --output <new bundle dir> [--precision float16|float32]
    Forge/.venv/bin/python -I Forge/yolox/export_yolox.py --sample-manifest   # the manifest for placeholder files

Nothing is downloaded: every file read is verified against `source-pins.json`. Conversion runs on the CPU. The output
directory is laid out as an installed revision root:

    OUTPUT/
      .aether/aether.json          the Aether manifest (MANIFEST.md §1.3, §2; driver `detector-dense@1`)
      yolox_s.aimodel              image [1,3,640,640] BGR 0..255 (fp32) -> preds [1,8400,85] fp32, decoded in-graph
      LICENSE, NOTICE              Apache-2.0 (upstream's LICENSE) and the modification notice
      README.md                    the Hugging Face model card
      SHA256SUMS                   every bundle file except itself and the log
      export-report.json           provenance, versions, timings, peak RSS
      forge-log.jsonl              one event per stage

The export is deterministic up to the compiler's blob names. The asset metadata's creation date is the pinned upstream
commit's date and its keys are sorted, so every file but main.mlirb and main.hash (and the manifest and SHA256SUMS,
which hash them), export-report.json and the log is byte-identical across runs. main.mlirb differs only in the
per-process seeded names the compiler gives the weight blobs; `--reproduces <earlier bundle>` proves the programs
(graph and weight bytes) equal with coreai-torch's portable graph diff. One Core AI JIT package serves macOS and iOS (no
per-architecture AOT): `--platform` only sets the manifest's `variant`.

Precision (default float32, as upstream and the zoo bundle): network, decode, input and `preds` all float32.
`--precision float16` runs the network in float16 (box decode still float32; input float16, which is exact for 0..255
integers). Measured on the M-series GPU against the CPU float32 reference (check_yolox.py, 6 cases), both pass the
detection bar: float32 to ~1e-6 (score) and IoU 0.999998; float16 at score diffs up to 0.004 and IoU >= 0.994, but the
largest score difference over all 8400 anchors reaches 0.009, so a detection within that of the threshold can enter or
leave the result. Latency is the same (about 10 ms per image either way, dominated by dispatch), so the default is the
exact one; float16 halves the asset (18 vs 36 MB) and stays a re-export away if iPhone measurement favours it.

Parity: `reference_yolox.py` (the unmodified upstream network on the CPU, float32, and the host NMS in numpy) records
the references; `check_yolox.py` runs the exported graph through Core AI on the GPU and gates it (docs/M4.md's
detection bar: same class set, matched boxes IoU >= 0.95, scores within 0.01).
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
import yolox_graphs as yg  # noqa: E402

ID = "yolox-s"
HF_REPO = "aether-models/aether-kit-yolox-s"
ASSET = "yolox_s.aimodel"
FUNCTION = "main"
INPUTS, OUTPUTS = ["image"], ["preds"]
NOT_BUNDLE = {".aether/aether.json", "SHA256SUMS", "export-report.json", "forge-log.jsonl"}
# Measured default (MANIFEST §0): GPU for every graph class; the graph is fully static.
COMPUTE = "gpu"
# The pinned upstream commit's committer date (2025-06-08T03:52:54Z), the asset metadata's creation date, so the
# package is reproducible.
CREATION_DATE = 1749354774


class Log:
    def __init__(self, path: Path) -> None:
        self.path, self.started = path, {}

    def event(self, stage: str, message: str, **fields) -> None:
        record = {"message": message, "stage": stage, "t": round(time.time(), 3), **fields}
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"[yolox:{stage}] {message}" + (f" {fields}" if fields else ""), file=sys.stderr, flush=True)

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
    metadata.author = "Megvii (YOLOX); Core AI export: Aether-Kit Forge"
    metadata.license = "Apache-2.0"
    metadata.model_description = (
        f"YOLOX-S ({yg.REPO} {yg.WEIGHTS['release']} {yg.WEIGHTS['file']}, {precision}): image [1,3,640,640] BGR on "
        "0..255, letterboxed top-left with pad 114 -> preds [1,8400,85] = decoded cx,cy,w,h in letterbox pixels, "
        "objectness, 80 COCO class probabilities. Host: score = obj * max(cls), threshold, per-class NMS.")
    asset = program.save_asset(path, metadata)
    asset.update_metadata(lambda m: setattr(m, "creation_date", CREATION_DATE))   # save_asset stamps the clock
    # The asset writer's key order varies from process to process; rewrite the same object with sorted keys.
    meta_path = path / "metadata.json"
    meta_path.write_text(json.dumps(json.loads(meta_path.read_text()), indent=2, sort_keys=True,
                                    separators=(",", " : ")) + "\n")
    return {"exportSeconds": round(t1 - t0, 1), "convertSeconds": round(time.monotonic() - t1, 1),
            "bytes": tree_bytes(path), "inputs": INPUTS, "outputs": OUTPUTS, "function": FUNCTION}


def program_diff(path: Path, other: Path) -> dict:
    """Reproducibility: the compiler names each weight blob with a per-process seeded hash, so two exports' main.mlirb
    differ in those names. coreai-torch's graph diff with `DIGEST_PORTABLE` (weights hashed from the blob bytes)
    compares the programs themselves: isomorphic with no modified node means the same graph and the same weights."""
    from coreai.authoring.asset import AIModelAsset
    from coreai_torch.debugging.graph_diff import compute_coreai_program_diff
    from coreai_torch.debugging.graph_match import WeightPolicy

    mine, theirs = AIModelAsset.load(path).program, AIModelAsset.load(other).program
    summary = compute_coreai_program_diff(mine, theirs, weights=WeightPolicy.DIGEST_PORTABLE).summary
    same = (summary.modified_node_count == 0 and summary.unmapped_source_node_count == 0
            and summary.unmapped_target_node_count == 0 and summary.unmapped_source_edge_count == 0
            and summary.unmapped_target_edge_count == 0 and summary.source_node_count == summary.target_node_count)
    return {"against": str(other), "identical": same, "nodes": summary.source_node_count,
            "modifiedNodes": summary.modified_node_count,
            "unmappedNodes": summary.unmapped_source_node_count + summary.unmapped_target_node_count,
            "unmappedEdges": summary.unmapped_source_edge_count + summary.unmapped_target_edge_count}


# MARK: the manifest (MANIFEST.md §1.3 `detector-dense@1`, §2.5 preprocess.image, §2.8 postprocess, labels)

def precision_block(precision: str) -> dict:
    if precision == "float16":
        return {"weights": "fp16", "activations": "fp16 network, fp32 box decode",
                "components": {"network": "fp16 (backbone, neck and head convolutions, sigmoids)",
                               "decode": "fp32 (grid offsets and exp(wh) * stride)",
                               "io": "image float16; preds float32"}}
    return {"weights": "fp32", "activations": "fp32", "components": {"io": "image float32; preds float32"}}


def manifest(files: dict[str, dict], report: dict, platform_name: str, precision: str) -> dict:
    return {
        "schema": 1,
        "id": ID,
        "task": "object-detection",
        "driver": "detector-dense@1",
        "variant": {"platform": platform_name},
        "assets": {"main": {"subtree": "", "files": [{"path": p, **meta} for p, meta in sorted(files.items())]}},
        "graphs": {
            "detector": {
                "file": f"main:{ASSET}", "function": FUNCTION,
                "inputs": {"image": "image"},
                "outputs": {"preds": "preds"},
                "compute": COMPUTE, "expectFrequentReshapes": False,
            },
        },
        "preprocess": {
            "image": {
                "resize": {"mode": "letterbox", "size": [yg.SIZE, yg.SIZE], "anchor": "topLeft",
                           "padValue": yg.PAD, "filter": "pillowBilinear"},
                "colorSpace": "deviceRGB", "channelOrder": "BGR",
                "scale": 1.0, "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0],
                "layout": "NCHW", "hostDType": "float32", "orientation": "exif",
                "x-notes": ("Upstream preproc (yolox/data/data_augment.py): r = min(640 / h, 640 / w), resize to "
                            "(int(w r), int(h r)) (cv2 INTER_LINEAR), paste top-left on a 640 x 640 canvas filled "
                            "with 114, BGR (OpenCV's decode order, no colour management), 0..255 with no scaling or "
                            "normalization (the 0.1.1rc0 non-legacy weights). The references (reference_yolox.py) "
                            "resize with Pillow BILINEAR in that geometry, which pillowBilinear and the kit's letterbox "
                            "reproduce exactly. "
                            + ("The graph input is float16; the host builds float32 and converts (0..255 integers "
                               "are exact in float16)." if precision == "float16" else "The graph input is float32.")),
            },
        },
        "postprocess": {
            "kind": "denseNms",
            "boxFormat": "cxcywh", "boxSpace": "letterboxPixels",
            "activation": "applied",
            "nms": {"type": "perClass", "iou": yg.NMS_IOU},
            "defaults": {"scoreThreshold": yg.SCORE_THRESHOLD, "maxDetections": yg.MAX_DETECTIONS},
            "classIndex": "contiguous",
            "x-notes": ("preds [1,8400,85] float32, per anchor (80x80 stride 8, 40x40 stride 16, 20x20 stride 32, "
                        "row-major): cx, cy, w, h in letterboxed input pixels (decoded in the graph), objectness, "
                        "80 class probabilities (sigmoid in the graph). score = obj * max(cls), class = argmax(cls); "
                        "keep score >= threshold; greedy NMS per class suppressing IoU > 0.45 (xyxy); then the top "
                        "maxDetections by score (upstream yolox.utils.postprocess). Map back to the source image "
                        "by dividing by the letterbox ratio r."),
        },
        "labels": {
            "set": "coco-91",
            "x-notes": ("class index c (0..79, contiguous) is COCO's c-th detection category in ascending "
                        "category-id order; its COCO id is the c-th of the 80 used ids of coco-91 (1..90 without 12, "
                        "26, 29, 30, 45, 66, 68, 69, 71, 83). Names as upstream's COCO_CLASSES."),
        },
        "precision": precision_block(precision),
        "license": "Apache-2.0",
        "x-license": {"files": ["LICENSE", "NOTICE"], "modified": True},
        "provenance": {
            "source": yg.REPO, "sourceRevision": f"{yg.WEIGHTS['release']}:{yg.WEIGHTS['file']}",
            "producer": "aether-kit Forge/yolox/export_yolox.py",
            "exportScript": report["scriptSHA256"]["export_yolox.py"],
            "refs": {"weights": yg.WEIGHTS["url"], "weightsSHA256": yg.WEIGHTS["sha256"],
                     "upstreamCode": f"{yg.CODE_REPO}@{yg.CODE_COMMIT}",
                     "forgeLockSHA256": report["forgeLockSHA256"], "exportReport": "export-report.json"},
        },
    }


def sample_manifest(platform_name: str, precision: str = "float32") -> dict:
    files = {f"{ASSET}/{name}": {"bytes": 1, "sha256": "0" * 64} for name in ("main.hash", "main.mlirb",
                                                                              "metadata.json")}
    files.update({name: {"bytes": 1, "sha256": "0" * 64} for name in ("LICENSE", "NOTICE", "README.md")})
    report = {"scriptSHA256": {"export_yolox.py": "0" * 64}, "forgeLockSHA256": "0" * 64}
    return manifest(files, report, platform_name, precision)


def notice(precision: str) -> str:
    return "\n".join([
        "YOLOX-S (Aether-Kit Core AI export)",
        "",
        f"This bundle is a modified version of the YOLOX-S COCO checkpoint {yg.WEIGHTS['file']} from the",
        f"{yg.WEIGHTS['release']} release of {yg.REPO} (https://github.com/{yg.REPO}),",
        "Copyright (c) 2021-2022 Megvii Inc., licensed under the Apache License, Version 2.0 (LICENSE). The model",
        f"code it was exported with is upstream's ({yg.CODE_REPO} at commit {yg.CODE_COMMIT}, Apache-2.0).",
        "",
        "Modifications: Aether-Kit's Forge exporter re-expressed the inference network as one static Core AI graph",
        "(640 x 640 input; the head's grid/stride box decode and the objectness and class sigmoids in the graph; "
        + ("the network cast to float16 with the box decode in float32)." if precision == "float16" else
           "float32)."),
        "It is not an official release of YOLOX.",
        "",
    ])


def readme(platform_names: list[str], precision: str) -> str:
    return "\n".join([
        "---",
        "license: apache-2.0",
        "pipeline_tag: object-detection",
        "tags: [aether-kit, core-ai, object-detection, yolox, coco]",
        "---",
        "",
        "# YOLOX-S for Aether-Kit (Core AI)",
        "",
        f"[YOLOX](https://github.com/{yg.REPO})-S, the COCO checkpoint `{yg.WEIGHTS['file']}` from the",
        f"`{yg.WEIGHTS['release']}` release (40.5 mAP at 640), exported by",
        "[Aether-Kit](https://github.com/bubstackdev/aether-kit)'s Forge exporter (`Forge/yolox/`) as one static Core",
        "AI graph: a BGR image on 0..255, letterboxed top-left to 640 x 640 with pad 114, to 8400 decoded dense",
        "predictions (cx, cy, w, h in letterbox pixels, objectness, 80 COCO class probabilities). The host applies the",
        "score threshold and per-class NMS. "
        + ("The network runs in float16 and the box decode in float32. " if precision == "float16" else "")
        + "Load it with Aether-Kit's `AetherVision`",
        f"(`detector-dense@1`, {' and '.join(platform_names)}); the manifest is `.aether/aether.json`. Gated against",
        "the upstream implementation on CPU in float32 (see `export-report.json`).",
        "",
        "Apache-2.0 (`LICENSE`). A modified version (see `NOTICE`).",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="new bundle directory")
    parser.add_argument("--platform", choices=["macos", "ios"], default="macos", help="the manifest's variant")
    parser.add_argument("--precision", choices=sorted(yg.DTYPES), default="float32")
    parser.add_argument("--sample-manifest", action="store_true", help="print the manifest for placeholder files")
    parser.add_argument("--reproduces", type=Path,
                        help="an earlier bundle this export must reproduce (see program_diff)")
    parser.add_argument("--weights", type=Path, default=yg.DEFAULT_WEIGHTS)
    parser.add_argument("--upstream", type=Path, default=yg.DEFAULT_UPSTREAM, help="fetch_upstream.py's unpack")
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
    dtype = yg.DTYPES[args.precision]
    lock = HERE.parent / "uv.lock"
    report = {
        "schema": 1, "id": ID, "command": sys.argv,
        "exportedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {"os": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
                 "machine": platform.machine(), "python": platform.python_version()},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "coreai-torch", "coreai-core",
                                                                "numpy")},
        "forgeLockSHA256": yg.sha256_file(lock), "forgePyprojectSHA256": yg.sha256_file(HERE.parent / "pyproject.toml"),
        "scriptSHA256": {p.name: yg.sha256_file(p) for p in sorted(HERE.glob("*.py"))},
        "source": {"repository": yg.REPO, "weights": yg.WEIGHTS, "path": str(args.weights.resolve()),
                   "upstreamCode": {"repository": yg.CODE_REPO, "commit": yg.CODE_COMMIT}, "verified": {}},
        "precision": (f"network {args.precision}, box decode float32; input {args.precision}, output float32; "
                      "no quantization"),
        "inputShape": [1, 3, yg.SIZE, yg.SIZE], "outputShape": [1, yg.ANCHORS, yg.CHANNELS],
        "compute": COMPUTE, "creationDate": CREATION_DATE, "conversion": {}, "stages": {},
    }
    report_path = args.output / "export-report.json"

    def save() -> None:
        report["peakRSSGB"] = peak_rss_gb()
        report["wallSeconds"] = round(time.monotonic() - started, 1)
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    log.start("verify-sources")
    report["source"]["verified"] = yg.verify_sources(args.weights, args.upstream)
    report["stages"]["verify-sources"] = log.done("verify-sources")
    save()

    log.start("export")
    graph = yg.YoloxGraph(yg.build_net(args.upstream, args.weights), dtype).eval()
    example = (torch.from_numpy(yg.synthetic_tensor(0)).to(dtype),)
    report["conversion"]["detector"] = convert(graph, example, args.output / ASSET, args.precision)
    del graph
    report["stages"]["export"] = log.done("export")
    save()
    if args.reproduces:
        log.start("reproduce")
        report["reproduces"] = program_diff(args.output / ASSET, args.reproduces / ASSET)
        report["stages"]["reproduce"] = log.done("reproduce", identical=report["reproduces"]["identical"])
        save()
        if not report["reproduces"]["identical"]:
            raise SystemExit(f"the exported program differs from {args.reproduces}")

    log.start("package")
    shutil.copyfile(args.upstream / "LICENSE", args.output / "LICENSE")
    (args.output / "NOTICE").write_text(notice(args.precision))
    (args.output / "README.md").write_text(readme(["macOS"], args.precision))
    files = {str(p.relative_to(args.output)): {"bytes": p.stat().st_size, "sha256": yg.sha256_file(p)}
             for p in tree_files(args.output) if str(p.relative_to(args.output)) not in NOT_BUNDLE}
    report["bundleBytes"] = sum(f["bytes"] for f in files.values())
    report["assetBytes"] = {"detector": tree_bytes(args.output / ASSET)}
    report["stages"]["package"] = log.done("package")
    save()
    (args.output / ".aether").mkdir()
    manifest_path = args.output / ".aether/aether.json"
    manifest_path.write_text(json.dumps(manifest(files, report, args.platform, args.precision), indent=2) + "\n")
    report["aetherJSONSHA256"] = yg.sha256_file(manifest_path)
    save()
    sums = [f"{yg.sha256_file(p)}  {p.relative_to(args.output)}" for p in tree_files(args.output)
            if p.name not in ("SHA256SUMS", "forge-log.jsonl")]
    (args.output / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(json.dumps({"conversion": report["conversion"], "bundleBytes": report["bundleBytes"],
                      "peakRSSGB": report["peakRSSGB"], "wallSeconds": report["wallSeconds"]}, indent=1))


if __name__ == "__main__":
    main()

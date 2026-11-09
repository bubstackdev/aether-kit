#!/usr/bin/env python3
"""Export Krea 2 Turbo (krea/Krea-2-Turbo, Krea 2 Community License) as an Aether-Kit `krea2@1` bundle.

    uv sync --frozen --project Forge
    Forge/.venv/bin/python -I Forge/krea2/export_krea2.py --variant vanilla --output <new bundle dir>
    Forge/.venv/bin/python -I Forge/krea2/export_krea2.py --variant realism --output <new bundle dir>
    Forge/.venv/bin/python -I Forge/krea2/export_krea2.py --variant realism --only vaeEncoder \
        --base-bundle <existing bundle> --output <new bundle dir>
    Forge/.venv/bin/python -I Forge/krea2/export_krea2.py --variant realism --replace \
        --only textEncoder denoiser0 denoiser1 denoiser2 denoiser3 vision --base-bundle <existing bundle> \
        --output <new bundle dir>       # reference images: re-exports what they change, links the VAEs

`vanilla` is the pinned checkpoint; `realism` merges the realism LoKr (`realism_engine_krea2_v3.1.safetensors`,
sha256 a6712629...) into the transformer in float32 before export. Nothing is downloaded: the snapshot must be in the
Hugging Face cache (or `--snapshot`), and every file read is verified against the pins below. Writes only `--output`:

    OUTPUT/
      .aether/aether.json               the Aether manifest (MANIFEST.md §5.5–5.6; driver `krea2@1`)
      krea2_turbo[_realism]-{text,denoiser0..3,vae,vae-encoder,vision}.aimodel   (text: encode, encodeWithImages)
      tokenizer/                        the checkpoint's tokenizer files, unchanged
      LICENSE.pdf, KREA-2-LICENSE.txt, KREA-2-USE-POLICY.txt, NOTICE
      SHA256SUMS                        every bundle file except itself
      export-report.json                provenance, versions, LoKr coverage, timings, peak RSS
      forge-log.jsonl                   one event per stage

The output is laid out as an installed revision root (the manifest under `.aether/`), so it uploads as-is.

`--base-bundle` adds the graphs named by `--only` to a complete earlier export without converting the others again:
the new bundle hard-links (or, across volumes, copies) every graph, tokenizer and license file of the base, exports
the named graphs, and writes a new manifest, NOTICE and SHA256SUMS over the union. The base's export report is kept
as `base-export-report.json`. The base is only read, and the result is a new export with its own manifest digest.

Conversion runs on the CPU; parity is the separate gate `check_krea2.py` (GPU) against `reference_krea2.py`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import importlib.metadata
import json
import os
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
import krea2_graphs as kg  # noqa: E402

REPO = "krea/Krea-2-Turbo"
REVISION = "98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7"
DIFFUSERS_COMMIT = "2a1433913c7044f34fa4767dd84b28bba0bb0ecf"   # tag v0.42.1; see Forge/pyproject.toml
DEFAULT_SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--krea--Krea-2-Turbo/snapshots" / REVISION
FORGE_INPUTS = Path(__file__).resolve().parents[3] / "forge-inputs/krea2-realism"
ADAPTER_FILE = "realism_engine_krea2_v3.1.safetensors"
ADAPTER_SHA256 = "a6712629445a2e91a616568e82befa8c8c7518e891a0f7c9918138634b5b54a5"
# Every checkpoint file the export reads, at REVISION (LFS oids and the Hub's file contents, 2026-11-05).
CHECKPOINT_SHA256 = {
    "LICENSE.pdf": "b82a2805162bde714a4eb27b9063c4fc3345d08a30be055134a6160e5430ba74",
    "model_index.json": "7d6c1206e369286df31244670a9a38fc73b077d8f64b4d273abb20ef9a8a8b93",
    "scheduler/scheduler_config.json": "4d8ea4471195528cdb84fcb8d37cd54ff442b9c02ff74f7d16d0a7a2f14b9467",
    "text_encoder/config.json": "1a1006851200920f53ab7e17ca14a8b3d1b91075dfbbff493307ba894cab0abb",
    "text_encoder/model.safetensors": "8434db05292f95e0041589a7c82abeb39385be59c85b54ae11caa7b45e9f4f13",
    "tokenizer/chat_template.jinja": "3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4",
    "tokenizer/tokenizer.json": "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506",
    "tokenizer/tokenizer_config.json": "c871ed314285a377eccf509b906aa5fe015d5e79a3ddab2c198467329127ead6",
    "transformer/config.json": "65b85015e515bbfc3ffe6c03e10537a47121bd3756e0d21a1a87126da60bc4a8",
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors": "9ca6a4951e447d8465cf64e0d7324b65da1a44d373d7d1b7d610fc395b164861",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors": "d092b7b22d2febe37354a54f2cfa4891b4fda397e27a3a465ee64f7597ecaad3",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors": "df51a0e537a56ba2ebf06c0d190d3f8571f967ae1f8e7b8acf05c33b5821cdce",
    "transformer/diffusion_pytorch_model.safetensors.index.json": "805104fc89272b7d5fb5aea4d7d7d0281691a1b97368adf1d7f57cbc2de5cf97",
    "vae/config.json": "54949c79def0d1060353c3fbfd4d2d2c4815ae241da6d6d28c99634c4eac6e6e",
    "vae/diffusion_pytorch_model.safetensors": "ab1b61103959913d6c7e628cf793dbb2ca4726a40a3b3ae206c52b8e75bf6f08",
}
TOKENIZER_FILES = ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json")
VARIANTS = {
    "vanilla": {"id": "krea-2-turbo", "name": "krea2_turbo", "title": "Krea 2 Turbo"},
    "realism": {"id": "krea-2-turbo-realism", "name": "krea2_turbo_realism", "title": "Krea 2 Turbo Realism"},
}
ROLES = ("textEncoder", "denoiser0", "denoiser1", "denoiser2", "denoiser3", "vaeDecoder", "vaeEncoder", "vision")
ASSET_SUFFIX = {"textEncoder": "text", "vaeDecoder": "vae", "vaeEncoder": "vae-encoder", "vision": "vision",
                **{f"denoiser{i}": f"denoiser{i}" for i in range(4)}}
FUNCTIONS = {"textEncoder": "encode", "vaeDecoder": "decode", "vaeEncoder": "encode", "vision": "encode",
             **{f"denoiser{i}": "segment" for i in range(4)}}
REFERENCE_FUNCTION = "encodeWithImages"           # textEncoder's second function (the `references` phase)
TEXT_REFERENCE_INPUTS = ["input_ids", "rope_positions", "image_embeds", "deepstack_embeds"]
NOTICE_LINE = ("Krea 2 is licensed under the Krea 2 Community License Agreement. For more information, visit "
               "https://krea.ai/krea-2-licensing.")


# MARK: helpers

class Log:
    def __init__(self, path: Path) -> None:
        self.path, self.started = path, {}

    def event(self, stage: str, message: str, **fields) -> None:
        record = {"message": message, "stage": stage, "t": round(time.time(), 3), **fields}
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"[krea2:{stage}] {message}" + (f" {fields}" if fields else ""), file=sys.stderr, flush=True)

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


def convert(module: nn.Module, example: tuple, dynamic, inputs: list[str], outputs: list[str], entry: str,
            path: Path, extra: list[tuple] = ()) -> dict:
    """Exports `module` as function `entry` (and each `extra` (module, example, dynamic, inputs, outputs, entry) as
    another function of the same asset: modules sharing weights store them once, measured on the random-weight
    selftest)."""
    import coreai_torch as ct

    t0 = time.monotonic()
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    for m, ex, dyn, ins, outs, name in [(module, example, dynamic, inputs, outputs, entry), *extra]:
        with torch.no_grad():
            ep = torch.export.export(m, ex, dynamic_shapes=dyn, strict=False)
            ep = ep.run_decompositions(ct.get_decomp_table())
        converter.add_exported_program(ep, input_names=ins, output_names=outs, state_names=[], entrypoint_name=name)
        del ep
    t1 = time.monotonic()
    program = converter.to_coreai()
    del converter
    program.optimize()
    program.save_asset(path)
    return {"exportSeconds": round(t1 - t0, 1), "convertSeconds": round(time.monotonic() - t1, 1),
            "bytes": tree_bytes(path), "inputs": inputs, "outputs": outputs, "function": entry,
            **({"functions": {name: {"inputs": ins, "outputs": outs} for _, _, _, ins, outs, name in extra}}
               if extra else {})}


def transformer_shapes(checkpoint: kg.Checkpoint) -> dict[str, tuple[int, ...]]:
    from safetensors import safe_open
    shapes = {}
    for file in sorted(set(checkpoint.files("transformer").values())):
        with safe_open(file, "pt") as handle:
            for name in handle.keys():
                shapes[name] = tuple(handle.get_slice(name).get_shape())
    return shapes


MANIFEST_PATH = ".aether/aether.json"


# MARK: the manifest (MANIFEST.md §5: generic `schedule`/`latent`/`residency` since G0; driver `krea2@1` and its `krea2`
# block, §5.5–5.6, since G3)

# The denoiser segments and the VAE specialize without the reshape hint, as check_krea2.py qualified them: with
# `expectFrequentReshapes: true` every segment's carries come back non-finite on the GPU (measured in G3, M5 Max,
# Xcode 27.2). The text encoder is exact with the hint (row cosine 0.999999), and its length changes per prompt.
def manifest(variant: str, files: dict[str, dict], asset_names: dict[str, str], report: dict) -> dict:
    v = VARIANTS[variant]
    graph_file = lambda role: f"main:{asset_names[role]}"  # noqa: E731
    from_ = lambda ref: {"name": ref.split(".")[1], "source": {"from": ref}}  # noqa: E731
    segment_inputs = lambda i: {  # noqa: E731
        "carryImage": from_(f"denoiser{i - 1}.image_{i}"), "carryText": from_(f"denoiser{i - 1}.text_{i}"),
        "condition": from_("denoiser0.cond"), "imageRotary": "rope_image"}
    graphs = {
        "textEncoder": {"file": graph_file("textEncoder"), "function": "encode",
                        "inputs": {"tokens": "input_ids", "positions": "position_ids"},
                        "outputs": {"conditioning": "text_embeds"},
                        "compute": "gpu", "expectFrequentReshapes": True, "residency": "perPrompt"},
        # Reference images: one image's patches [576, 1536] (block-major, the first h*w real) and its patch grid (h, w)
        # -> merged tokens [144, 2560] and DeepStack [3, 144, 2560]. Static: a dynamic tower is wrong intermittently.
        "vision": {"file": graph_file("vision"), "function": "encode",
                   "inputs": {"patches": "patches", "grid": "grid"},
                   "outputs": {"imageEmbeds": "image_embeds", "deepstackEmbeds": "deepstack_embeds"},
                   "compute": "gpu", "expectFrequentReshapes": False, "residency": "perPrompt"},
        "denoiser0": {"file": graph_file("denoiser0"), "function": "segment",
                      "inputs": {"latents": "latents", "conditioning": {"name": "text_embeds",
                                                                         "source": {"from": "textEncoder.text_embeds"}},
                                 "sigma": "sigma", "imageRotary": "rope_image"},
                      "outputs": {"carryImage": "image_1", "carryText": "text_1", "condition": "cond"},
                      "compute": "gpu", "expectFrequentReshapes": False, "residency": "perStep"},
        "vaeDecoder": {"file": graph_file("vaeDecoder"), "function": "decode",
                       "inputs": {"latents": "latents"}, "outputs": {"image": "image"},
                       "compute": "gpu", "expectFrequentReshapes": False, "residency": "perImage"},
        # Refine's encoder: pixels [3, 16 * rows, 16 * columns] on [-1, 1] -> packed latents [rows, columns, 64].
        "vaeEncoder": {"file": graph_file("vaeEncoder"), "function": "encode",
                       "inputs": {"pixels": "image"}, "outputs": {"latents": "latents"},
                       "compute": "gpu", "expectFrequentReshapes": False, "residency": "perPrompt"},
    }
    for i in (1, 2, 3):
        outputs = {"velocity": "velocity"} if i == 3 else {"carryImage": f"image_{i + 1}", "carryText": f"text_{i + 1}"}
        graphs[f"denoiser{i}"] = {"file": graph_file(f"denoiser{i}"), "function": "segment",
                                  "inputs": segment_inputs(i), "outputs": outputs, "compute": "gpu",
                                  "expectFrequentReshapes": False, "residency": "perStep"}
    references = "vision" in asset_names
    if references:
        # The text encoder's second function: the same weights and output, with images as extra rows.
        text = graphs["textEncoder"]
        del text["function"]
        text["functions"] = {"step": FUNCTIONS["textEncoder"], "references": REFERENCE_FUNCTION}
        text["inputsByFunction"] = {"references": {"tokens": "input_ids", "ropePositions": "rope_positions",
                                                   "imageEmbeds": "image_embeds",
                                                   "deepstackEmbeds": "deepstack_embeds"}}
    order = [role for role in ROLES if role in asset_names]
    return {
        "schema": 1,
        "id": v["id"],
        "task": "text-to-image",
        "driver": "krea2@1",
        "variant": {"platform": "macos"},
        "assets": {"main": {"subtree": "", "files": [{"path": p, **meta} for p, meta in sorted(files.items())]}},
        "graphs": {role: graphs[role] for role in order},
        "tokenizer": {"source": {"dir": "main:tokenizer"}, "chatTemplate": None, "renderer": "krea2-prompt",
                      "addSpecialTokens": False},
        "schedule": {
            "kind": "flowMatchEuler", "steps": kg.STEPS, "maxSteps": 50, "trainTimesteps": kg.TRAIN_TIMESTEPS,
            "sigmas": {"base": "linspace(1, 1/steps, steps)", "shift": {"rule": "exponential", "mu": kg.MU},
                       "terminal": 0.0},
            "timestep": "sigma * trainTimesteps / trainTimesteps",
            "update": "x + (sigma[i+1] - sigma[i]) * velocity",
            "guidance": {"kind": "none"},
        },
        "latent": {
            "channels": kg.LATENT_CHANNELS, "pixelsPerToken": kg.PIXELS_PER_TOKEN, "packing": "patch2x2ChannelMajor",
            "layout": "tokens[rows*columns, 64]", "decoderLayout": "[rows, columns, 64]",
            "noise": {"kind": "splitmix64BoxMuller", "order": "packedToken", "dtype": "float32"},
            "maxImageTokens": kg.MAX_IMAGE_TOKENS,
        },
        "krea2": {
            "prompt": {"systemPrefix": kg.SYSTEM_PREFIX, "suffix": kg.SUFFIX, "maxTokens": kg.MAX_TOKENS,
                       "keepTrailingTokens": kg.TRAILING_TOKENS, "dropLeadingRows": kg.DROP_ROWS, "padding": "none",
                       "positions": "running"},
            "conditioning": {"taps": list(kg.TEXT_TAPS), "width": kg.TEXT_WIDTH, "rows": f"1...{kg.MAX_TEXT_ROWS}"},
            "rotary": {"axes": list(kg.ROPE_AXES), "theta": kg.ROPE_THETA, "positions": "(0, row, column)",
                       "layout": "splitHalves", "textRotated": False, "dtype": "float32"},
            "segments": [f"denoiser{i}" for i in range(4)],
            "condRows": kg.CONDITION_ROWS,
            "contentFilter": {"required": True, "reason": "Krea 2 Community License §4.2; the app supplies the filter"},
            **({"references": reference_block()} if references else {}),
        },
        "precision": {"weights": "fp16", "activations": "fp16 matmul/attention, fp32 residual/norm/modulation",
                      "components": {"denoiser0": "transformer block 0 computes in fp32 from its fp16 weights",
                                     "vaeDecoder": "fp16 weights, fp32 compute",
                                     **({"vaeEncoder": "fp16 weights, fp32 compute"}
                                        if "vaeEncoder" in asset_names else {}),
                                     **({"vision": "fp16 weights, fp32 compute",
                                         "textEncoder": "encode and encodeWithImages share the fp16 weights; encodeWithImages computes in "
                                                        + report.get("referenceCompute", "float32")}
                                        if references else {})}},
        "license": "LicenseRef-Krea-2-Community-License-1.0",
        "x-license": {"files": ["LICENSE.pdf", "KREA-2-LICENSE.txt", "KREA-2-USE-POLICY.txt", "NOTICE"],
                      "modified": True, "commercialRevenueCapUSD": 1_000_000,
                      **({"adapter": "Civitai 2688234 v3109006 (razzz): merges keep the same permissions "
                                     "(allowDifferentLicense false); NSFW-trained"} if variant == "realism" else {})},
        "provenance": {
            "source": REPO, "sourceRevision": REVISION,
            **({"mergedAdapter": {"file": ADAPTER_FILE, "sha256": ADAPTER_SHA256, "format": "LyCORIS LoKr",
                            "merge": "W + multiplier * scale * kron(w1, w2), float32, before export",
                            "multiplier": report["adapter"]["multiplier"], "scale": report["adapter"]["scale"]}}
               if variant == "realism" else {}),
            "producer": "aether-kit Forge/krea2/export_krea2.py",
            "exportScript": report["scriptSHA256"]["export_krea2.py"],
            "refs": {"diffusersCommit": DIFFUSERS_COMMIT, "forgeLockSHA256": report["forgeLockSHA256"],
                     "exportReport": "export-report.json"},
        },
    }


def reference_block() -> dict:
    """`krea2.references` (MANIFEST.md §5.6): the 0.0.x reference contract the host applies before `encodeWithImages`."""
    return {
        "maxImages": kg.REFERENCE_MAX_IMAGES, "maxSide": kg.REFERENCE_MAX_SIDE,
        "maxImageTokens": kg.REFERENCE_MAX_IMAGE_TOKENS, "label": kg.REFERENCE_LABEL,
        "tokens": {"pad": kg.IMAGE_PAD_ID, "start": kg.VISION_START_ID, "end": kg.VISION_END_ID},
        "extensionBase": kg.EXTENSION_BASE, "rows": f"1...{kg.REFERENCE_MAX_TEXT_ROWS}", "positions": "mrope",
        "deepstackLevels": kg.DEEPSTACK_LEVELS, "vision": "vision", "function": "references",
        "preprocess": {"fit": "longSideRounded", "maxSide": kg.REFERENCE_MAX_SIDE, "factor": 32, "minPixels": 65536,
                       "maxPixels": 16777216, "maxAspect": 200, "filter": "keysBicubic-0.5-antialias",
                       "mean": 127.5, "std": 127.5, "patch": 16, "temporalPatch": 2, "merge": 2,
                       "order": "mergeWindow C,T,py,px"},
    }


def sample_manifest(variant: str) -> dict:
    """`manifest()` for placeholder files and report values: the emitted shape, without an export. Regenerate the Swift
    fixtures with `--sample-manifest --variant <v> > Tests/AetherCoreTests/Fixtures/krea2/<v>.json`."""
    v = VARIANTS[variant]
    asset_names = {role: f"{v['name']}-{ASSET_SUFFIX[role]}.aimodel" for role in ROLES}
    files = {f"{name}/weights.bin": {"bytes": 1, "sha256": "0" * 64} for name in asset_names.values()}
    report = {"scriptSHA256": {"export_krea2.py": "0" * 64}, "forgeLockSHA256": "0" * 64,
              "adapter": {"multiplier": 1.0, "scale": 1.0}}
    return manifest(variant, files, asset_names, report)


def notice(variant: str, roles=ROLES) -> str:
    v = VARIANTS[variant]
    vae = "VAE decoder and encoder" if "vaeEncoder" in roles else "VAE decoder"
    if "vision" in roles:
        vae += ", and the text encoder checkpoint's vision tower for reference images"
    lines = [
        f"{v['title']} (Aether-Kit Core AI export)",
        "",
        NOTICE_LINE,
        "",
        f"This bundle is a modified version of the Krea Model {REPO} at revision {REVISION}: its weights were",
        f"converted to fp16 and re-expressed as Core AI graphs (text encoder, four denoiser segments, {vae}) by",
        "Aether-Kit's Forge exporter. The text encoder keeps only the layers Krea 2 reads.",
    ]
    if variant == "realism":
        lines += [
            f"The realism LyCORIS LoKr adapter {ADAPTER_FILE} (sha256 {ADAPTER_SHA256}; Civitai model 2688234,",
            "\"Realism Engine Ideogram 4 + Krea 2\" by razzz, version \"Krea2 v3.0\") was merged into the transformer",
            "weights before export. The merge carries the adapter's Civitai permissions unchanged.",
        ]
    lines += [
        "",
        "This is not an official Krea release and is not endorsed by Krea. Use is subject to the Krea 2 Community",
        "License Agreement (LICENSE.pdf, KREA-2-LICENSE.txt) and its Acceptable Use Policy (KREA-2-USE-POLICY.txt).",
        "Any deployment must implement content filters (§4.2).",
        "",
    ]
    return "\n".join(lines)


# MARK: main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS), required=True)
    parser.add_argument("--output", type=Path, help="new bundle directory")
    parser.add_argument("--sample-manifest", action="store_true",
                        help="print the manifest for placeholder files and exit (the Swift decoding fixture)")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--adapter", type=Path, default=FORGE_INPUTS / ADAPTER_FILE)
    parser.add_argument("--license-dir", type=Path, default=FORGE_INPUTS, help="KREA-2-LICENSE.txt and use policy")
    parser.add_argument("--adapter-multiplier", type=float, default=1.0)
    parser.add_argument("--only", nargs="*", choices=ROLES,
                        help="export a subset (probes; a bundle only with --base-bundle)")
    parser.add_argument("--base-bundle", type=Path,
                        help="a complete earlier export to add the --only graphs to (read only; linked, not changed)")
    parser.add_argument("--replace", action="store_true",
                        help="with --base-bundle: re-export --only roles the base already has (their old assets are "
                             "not linked)")
    parser.add_argument("--reference-compute", choices=["float32", "float16"], default="float32",
                        help="the text encoder's references function's compute (weights stay fp16 and shared). float32: "
                             "fp16 conditioning drifted the 8-step closed loop to SSIM 0.917 against the oracle (bar 0.94) "
                             "while the oracle's conditioning reached 0.987 on the same GPU denoisers")
    parser.add_argument("--allow-unpinned", action="store_true", help="random-weight test snapshots only")
    args = parser.parse_args()
    if args.sample_manifest:
        print(json.dumps(sample_manifest(args.variant), indent=2, ensure_ascii=False))
        return
    if args.output is None:
        parser.error("--output is required")
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    roles = args.only or list(ROLES)
    base = None
    if args.base_bundle is not None:
        if not args.only:
            parser.error("--base-bundle needs --only (the graphs to add)")
        base = json.loads((args.base_bundle / MANIFEST_PATH).read_text())
        if base["id"] != VARIANTS[args.variant]["id"]:
            raise SystemExit(f"the base bundle is {base['id']}, not {VARIANTS[args.variant]['id']}")
        clash = set(roles) & set(base["graphs"])
        if clash and not args.replace:
            raise SystemExit(f"the base bundle already has {sorted(clash)}; pass --replace to re-export them")
    started = time.monotonic()
    args.output.mkdir(parents=True)
    log = Log(args.output / "forge-log.jsonl")
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    v = VARIANTS[args.variant]
    asset_names = {role: f"{v['name']}-{ASSET_SUFFIX[role]}.aimodel" for role in ROLES}
    checkpoint = kg.Checkpoint(args.snapshot.resolve())
    lock = HERE.parent / "uv.lock"
    report = {
        "schema": 1, "variant": args.variant, "id": v["id"], "command": sys.argv,
        "exportedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
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
        "precision": "fp16 weights; fp16 matmul/attention with fp32 residual, norms, modulation and edge projections; "
                     "denoiser block 0 computes in fp32; VAE fp16 storage with fp32 compute; no quantization",
        "conversion": {}, "stages": {},
    }
    report_path = args.output / "export-report.json"

    def save() -> None:
        report["peakRSSGB"] = peak_rss_gb()
        report["wallSeconds"] = round(time.monotonic() - started, 1)
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    log.start("verify-sources")
    for relative, expected in CHECKPOINT_SHA256.items():
        path = checkpoint.root / relative
        if args.allow_unpinned:
            break
        if not path.is_file():
            raise SystemExit(f"snapshot lacks {relative}")
        if sha256(path) != expected:
            raise SystemExit(f"{relative} differs from the pinned {REVISION[:12]} file")
        report["source"]["verified"][relative] = expected
    adapter = None
    if args.variant == "realism":
        if sha256(args.adapter) != ADAPTER_SHA256:
            raise SystemExit(f"{args.adapter} is not the pinned realism adapter")
        adapter = kg.LoKr(args.adapter, args.adapter_multiplier)
        report["adapter"] = {"file": str(args.adapter), "sha256": ADAPTER_SHA256,  # verify() raises unless total
                             **adapter.verify(transformer_shapes(checkpoint))}
        log.event("verify-sources", "adapter mapping total", **{k: report["adapter"][k] for k in
                                                                ("adapterTensors", "modules", "mapped", "unmapped")})
    report["stages"]["verify-sources"] = log.done("verify-sources")
    save()

    if "textEncoder" in roles:
        log.start("build-textEncoder")
        module = kg.TextEncoderGraph.load(checkpoint, strict=not args.allow_unpinned)
        report["stages"]["build-textEncoder"] = log.done("build-textEncoder")
        example, dynamic = module.example()
        reference = kg.ReferenceTextGraph(module, compute=getattr(torch, args.reference_compute)).eval()
        report["referenceCompute"] = args.reference_compute
        ref_example, ref_dynamic = reference.example()
        log.start("export-textEncoder")
        report["conversion"]["textEncoder"] = convert(
            module, example, dynamic, ["input_ids", "position_ids"], ["text_embeds"], "encode",
            args.output / asset_names["textEncoder"],
            extra=[(reference, ref_example, ref_dynamic, TEXT_REFERENCE_INPUTS, ["text_embeds"], REFERENCE_FUNCTION)])
        report["stages"]["export-textEncoder"] = log.done("export-textEncoder")
        del module
        gc.collect()
        save()

    for index in range(len(kg.SEGMENTS)):
        role = f"denoiser{index}"
        if role not in roles:
            continue
        log.start(f"build-{role}")
        segment = kg.DenoiserSegment.load(checkpoint, index, adapter, strict=not args.allow_unpinned)
        report["stages"][f"build-{role}"] = log.done(f"build-{role}")
        example, dynamic = segment.example()
        log.start(f"export-{role}")
        report["conversion"][role] = convert(segment, example, dynamic, segment.input_names, segment.output_names,
                                             "segment", args.output / asset_names[role])
        report["stages"][f"export-{role}"] = log.done(f"export-{role}")
        del segment
        gc.collect()
        save()

    if "vision" in roles:
        log.start("export-vision")
        module = kg.KreaVisionTower(kg.load_visual(checkpoint)).eval()
        example, dynamic = module.example()
        report["conversion"]["vision"] = convert(module, example, dynamic, ["patches", "grid"],
                                                 ["image_embeds", "deepstack_embeds"], "encode",
                                                 args.output / asset_names["vision"])
        report["stages"]["export-vision"] = log.done("export-vision")
        del module
        gc.collect()
        save()

    if "vaeEncoder" in roles:
        log.start("export-vaeEncoder")
        module = kg.VAEEncoderGraph(kg.load_vae(checkpoint)).eval()
        example, dynamic = module.example()
        report["conversion"]["vaeEncoder"] = convert(module, example, dynamic, ["image"], ["latents"], "encode",
                                                     args.output / asset_names["vaeEncoder"])
        report["stages"]["export-vaeEncoder"] = log.done("export-vaeEncoder")
        del module
        gc.collect()
        save()

    if "vaeDecoder" in roles:
        log.start("export-vaeDecoder")
        module = kg.VAEDecoderGraph(kg.load_vae(checkpoint)).eval()
        example, dynamic = module.example()
        report["conversion"]["vaeDecoder"] = convert(module, example, dynamic, ["latents"], ["image"], "decode",
                                                     args.output / asset_names["vaeDecoder"])
        report["stages"]["export-vaeDecoder"] = log.done("export-vaeDecoder")
        del module
        gc.collect()
        save()

    if base is not None:
        # Every role the base exported plus the new ones; the base's graphs, tokenizer and licenses are linked in.
        present = [role for role in ROLES if role in base["graphs"] or role in roles]
        asset_names = {role: asset_names[role] for role in present}
        log.start("link-base")
        keep = {MANIFEST_PATH, "SHA256SUMS", "export-report.json", "forge-log.jsonl", "NOTICE"}
        exported = tuple(asset_names[role] + "/" for role in roles)
        for source in tree_files(args.base_bundle):
            relative = str(source.relative_to(args.base_bundle))
            # The base's own check reports describe the base, not this bundle: check_krea2.py writes new ones. A role
            # re-exported here (--replace) keeps only its new asset.
            if relative in keep or relative.startswith("check-report") or relative.startswith(exported):
                continue
            target = args.output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise SystemExit(f"{relative} exists in both the base and the new export")
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        shutil.copyfile(args.base_bundle / "export-report.json", args.output / "base-export-report.json")
        base_report = json.loads((args.base_bundle / "export-report.json").read_text())
        report["base"] = {"bundle": str(args.base_bundle), "manifestSHA256": sha256(args.base_bundle / MANIFEST_PATH),
                          "exportedAt": base_report.get("exportedAt"), "added": roles,
                          "replaced": sorted(set(roles) & set(base["graphs"]))}
        if "adapter" in base_report:
            report["adapter"] = base_report["adapter"]
        report["stages"]["link-base"] = log.done("link-base")
        roles = present
        save()
    elif set(roles) != set(ROLES):
        save()
        print(json.dumps({"partial": roles, "conversion": report["conversion"], "peakRSSGB": report["peakRSSGB"]}))
        return

    log.start("package")
    (args.output / "tokenizer").mkdir(exist_ok=True)
    for name in TOKENIZER_FILES:
        if not (args.output / "tokenizer" / name).exists():
            shutil.copyfile(checkpoint.root / "tokenizer" / name, args.output / "tokenizer" / name)
    license_pdf = checkpoint.root / "LICENSE.pdf"
    if args.allow_unpinned and not license_pdf.exists():
        license_pdf = args.license_dir / "KREA-2-LICENSE.pdf"
    for source, name in ((license_pdf, "LICENSE.pdf"), (args.license_dir / "KREA-2-LICENSE.txt", "KREA-2-LICENSE.txt"),
                         (args.license_dir / "krea-2-use-policy.txt", "KREA-2-USE-POLICY.txt")):
        (args.output / name).unlink(missing_ok=True)      # a linked base copy: replace, never write through it
        shutil.copyfile(source, args.output / name)
    if sha256(args.output / "LICENSE.pdf") != sha256(args.license_dir / "KREA-2-LICENSE.pdf"):
        raise SystemExit("the snapshot's LICENSE.pdf differs from forge-inputs' KREA-2-LICENSE.pdf")
    (args.output / "NOTICE").write_text(notice(args.variant, roles))
    skip = {MANIFEST_PATH, "SHA256SUMS", "export-report.json", "forge-log.jsonl", "base-export-report.json"}
    files = {str(p.relative_to(args.output)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in tree_files(args.output) if str(p.relative_to(args.output)) not in skip}
    report["bundleBytes"] = sum(f["bytes"] for f in files.values())
    report["assetBytes"] = {role: tree_bytes(args.output / name) for role, name in asset_names.items()}
    report["stages"]["package"] = log.done("package")
    save()
    (args.output / ".aether").mkdir(exist_ok=True)
    (args.output / MANIFEST_PATH).write_text(json.dumps(manifest(args.variant, files, asset_names, report), indent=2)
                                             + "\n")
    report["aetherJSONSHA256"] = sha256(args.output / MANIFEST_PATH)
    save()
    sums = [f"{sha256(p)}  {p.relative_to(args.output)}" for p in tree_files(args.output)
            if p.name not in ("SHA256SUMS", "forge-log.jsonl")]
    (args.output / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(json.dumps({"variant": args.variant, "conversion": report["conversion"], "bundleBytes": report["bundleBytes"],
                      "peakRSSGB": report["peakRSSGB"], "wallSeconds": report["wallSeconds"],
                      **({"adapter": report["adapter"]} if adapter else {})}, indent=1))


if __name__ == "__main__":
    main()

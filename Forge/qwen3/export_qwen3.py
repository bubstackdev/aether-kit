#!/usr/bin/env python3
"""Export a Qwen3 dense decoder checkpoint (Hugging Face format) as an Aether-Kit `decoder-lm@1` bundle.

    uv sync --frozen --project Forge
    scripts/gpu-lease -- Forge/.venv/bin/python -I Forge/qwen3/export_qwen3.py \
        --source huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated@c9bd464550d4078c72af0dd22aa18d0437868ce3 \
        --output <new bundle dir>
    Forge/.venv/bin/python -I Forge/qwen3/export_qwen3.py --sample-manifest

Generic over Qwen3 dense decoders: the architecture comes from the checkpoint's config.json (qwen3_graphs.py), and the
checkpoint must be pinned in `source-pins.json` (every file it reads is verified; nothing is downloaded). `--snapshot`
points at the snapshot folder (default: the Hugging Face cache path of the pinned revision). Conversion runs on the CPU:
about 2x the float16 weights in memory (8 GB of weights for a 4B model; the converter copies them once).

The output directory is laid out as an installed revision root, a local bundle opened with
`InstalledBundle(ref:url:isStale:)` (it is not in the catalog):

    OUTPUT/
      .aether/aether.json     the Aether manifest (MANIFEST.md §1.1 `decoder-lm@1`, §2)
      decoder.aimodel         input_ids, position_ids i32 [1,S] / [1,T]; states keyCache/valueCache f16
                              [layers,1,kv_heads,C,head_dim]; logits [1,1,vocab] (the last fed row)
      tokenizer/              the checkpoint's tokenizer files and chat template, unmodified
      LICENSE, NOTICE, README.md
      SHA256SUMS, export-report.json, forge-log.jsonl

Precision (`--weights`, `--compute`, `--attention`, `--logits`): by default the recipe the 0.0.x Qwen3 exports were
qualified with (float16 weights, KV, residual stream, matmuls and head; float32 attention, norm statistics and rotary
angles), so greedy replies carry over token for token (the Swift gate, Tests/AetherTextTests/DecoderLMExportGateTests).
Reproducible as the other Forge exporters are (seeded generated names, fixed creation date, deterministic Swift
hashing), except the graph's `main.mlirb`/`main.hash` weight-blob names.
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
import qwen3_graphs as g  # noqa: E402

DECODER_ASSET = "decoder.aimodel"
FUNCTION = "main"
DECODER_INPUTS, DECODER_STATES, DECODER_OUTPUTS = ["input_ids", "position_ids"], ["keyCache", "valueCache"], ["logits"]
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
                   "added_tokens.json", "generation_config.json", "config.json")
NOT_BUNDLE = {".aether/aether.json", "SHA256SUMS", "export-report.json", "forge-log.jsonl"}
COMPUTE = "gpu"
WEIGHTS, DECODER_COMPUTE, ATTENTION, LOGITS = "float16", "float16", "float32", "float16"
CREATION_DATE = "2026-11-09T00:00:00+00:00"
SHORT = {"float16": "fp16", "float32": "fp32"}


class Log:
    def __init__(self, path: Path) -> None:
        self.path, self.started = path, {}

    def event(self, stage: str, message: str, **fields) -> None:
        record = {"message": message, "stage": stage, "t": round(time.time(), 3), **fields}
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"[qwen3:{stage}] {message}" + (f" {fields}" if fields else ""), file=sys.stderr, flush=True)

    def start(self, stage: str) -> None:
        self.started[stage] = time.monotonic()
        self.event(stage, "start")

    def done(self, stage: str, **fields) -> float:
        seconds = round(time.monotonic() - self.started.pop(stage), 2)
        self.event(stage, "done", seconds=seconds, peakRSSGB=peak_rss_gb(), **fields)
        return seconds


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
    """Removes the same-shape `aten.copy` nodes each in-place cache write leaves (the running contract)."""
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


def convert(module, path: Path, creation_date: int, description: str, pin: dict) -> dict:
    import coreai_torch as ct
    from coreai.runtime import AIModelAssetMetadata

    example, dynamic = module.example()
    with deterministic_conversion(creation_date):
        t0 = time.monotonic()
        with torch.no_grad():
            ep = torch.export.export(module, example, dynamic_shapes=dynamic, strict=False)
            ep = ep.run_decompositions(ct.get_decomp_table())
        mutated = len(ep.graph_signature.user_inputs_to_mutate)
        if mutated != len(DECODER_STATES):
            raise SystemExit(f"{mutated} mutated inputs for the states {DECODER_STATES}")
        dropped = drop_identity_copies(ep)
        t1 = time.monotonic()
        converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
        converter.add_exported_program(ep, input_names=DECODER_INPUTS, output_names=DECODER_OUTPUTS,
                                       state_names=DECODER_STATES, entrypoint_name=FUNCTION)
        del ep
        program = converter.to_coreai()
        del converter
        program.optimize()
        metadata = AIModelAssetMetadata()
        metadata.author = f"{pin['repository']}; Core AI export: Aether-Kit Forge"
        metadata.license = pin["license"]
        metadata.model_description = description
        metadata.creation_date = creation_date
        program.save_asset(path, metadata)
    return {"exportSeconds": round(t1 - t0, 1), "convertSeconds": round(time.monotonic() - t1, 1),
            "bytes": tree_bytes(path), "inputs": DECODER_INPUTS, "outputs": DECODER_OUTPUTS, "states": DECODER_STATES,
            "function": FUNCTION, "identityCopiesDropped": dropped}


# MARK: tokenizer facts (from the checkpoint's own files)

def stop_tokens(snapshot: Path) -> list[tuple[int, str]]:
    """generation_config.json eos_token_id (else config.json's) ∪ the tokenizer's eos, in that order, with their text."""
    gen_path = snapshot / "generation_config.json"
    source = json.loads(gen_path.read_text()) if gen_path.is_file() else g.read_config(snapshot)
    ids = source.get("eos_token_id")
    ids = list(ids if isinstance(ids, list) else [ids] if ids is not None else [])
    tok = json.loads((snapshot / "tokenizer_config.json").read_text())
    names = {int(k): v["content"] for k, v in tok.get("added_tokens_decoder", {}).items()}
    eos = tok.get("eos_token")
    eos = eos.get("content") if isinstance(eos, dict) else eos
    ids += [i for i, n in sorted(names.items()) if n == eos and i not in ids]
    if not ids or any(i not in names for i in ids):
        raise SystemExit(f"stop ids {ids} are not all named added tokens")
    return [(i, names[i]) for i in ids]


def chat_template(snapshot: Path) -> str:
    """The template's exact text (line endings kept: newline="" reads it untranslated)."""
    jinja = snapshot / "chat_template.jinja"
    if jinja.is_file():
        with jinja.open(encoding="utf-8", newline="") as f:
            return f.read()
    tok = json.loads((snapshot / "tokenizer_config.json").read_text())
    if isinstance(tok.get("chat_template"), str):
        return tok["chat_template"]
    raise SystemExit("the checkpoint has no chat template (chat_template.jinja or tokenizer_config.json)")


def thinking(template: str) -> dict | None:
    """Hybrid Qwen3 templates switch thinking with `enable_thinking` (on by default); instruct-only ones have none."""
    if "enable_thinking" not in template:
        return None
    return {"open": "<think>", "close": "</think>", "toggle": {"templateVar": "enable_thinking"}, "defaultOn": True}


# MARK: the manifest (MANIFEST.md §1.1 decoder-lm@1, §2.3 graphs, §2.4 tokenizer)

def manifest(files: dict[str, dict], report: dict, platform_name: str, prec: dict, pin: dict, stops: list[tuple[int, str]],
             template: str, context: int, template_sha: str = "0" * 64) -> dict:
    ref = f"{pin['repository']}@{pin['revision']}"
    return {
        "schema": 1,
        "id": pin["id"],
        "task": "text-generation",
        "driver": "decoder-lm@1",
        "variant": {"platform": platform_name},
        "assets": {"main": {"subtree": "", "files": [{"path": p, **meta} for p, meta in sorted(files.items())]}},
        "graphs": {
            "decoder": {
                "file": f"main:{DECODER_ASSET}", "function": FUNCTION,
                "inputs": {"tokens": "input_ids", "positions": "position_ids"},
                "outputs": {"logits": {"name": "logits", "rows": "last"}},
                "states": {"sequence": [{"tensors": DECODER_STATES, "kind": "kv", "seqAxis": 3}]},
                "limits": {"maxContext": context, "prefillChunk": g.PREFILL_MAX},
                "compute": COMPUTE, "expectFrequentReshapes": True,
                "x-notes": (f"Running contract: input_ids [1,S] (S 1..{g.PREFILL_MAX}), position_ids [1,T] = 0..T-1 "
                            "(the last S are the fed tokens'), keyCache/valueCache written in place at rows T-S..T-1; "
                            f"logits are the last fed row only. maxContext is the graph's position and KV axis bound "
                            f"(the checkpoint declares {report.get('declaredContext', 'its own')})."),
            },
        },
        "tokenizer": {
            "source": {"dir": "main:tokenizer"},
            "chatTemplate": "main:tokenizer/chat_template.jinja",
            "chatTemplateSha256": template_sha,
            "addBos": "template",
            "stop": [text for _, text in stops],
            "thinking": thinking(template),
            "toolDialect": "hermes-json",
        },
        "precision": prec,
        "license": pin["license"].lower(),
        "x-license": {"files": ["LICENSE", "NOTICE"], "modified": True},
        "provenance": {
            "source": pin["repository"], "sourceRevision": pin["revision"],
            **({"base": pin["base"]} if pin.get("base") else {}),
            "producer": "aether-kit Forge/qwen3/export_qwen3.py",
            "exportScript": report["scriptSHA256"]["export_qwen3.py"],
            "refs": {"tokenizer.stop": (f"{ref} generation_config.json#/eos_token_id ∪ tokenizer_config.json#/eos_token: "
                                        + ", ".join(f"{i} {text}" for i, text in stops)),
                     "tokenizer.source": f"{ref} " + ", ".join(n for n in TOKENIZER_FILES if f"tokenizer/{n}" in files),
                     "tokenizer.chatTemplate": f"{ref} chat_template.jinja",
                     "forgeLockSHA256": report["forgeLockSHA256"], "exportReport": "export-report.json"},
        },
    }


def precision(weights: str, compute: str, attention: str, logits: str) -> dict:
    return {"weights": SHORT[weights], "activations": SHORT[compute], "kv": SHORT[weights], "logits": SHORT[logits],
            "components": {
                "decoder": (f"{SHORT[weights]} weights and KV, {SHORT[compute]} residual stream, matmuls and head, "
                            f"{SHORT[attention]} attention; RMSNorm statistics and rotary angles fp32"),
                "io": f"tokens and positions i32; logits {SHORT[logits]}"}}


def sample_manifest(platform_name: str) -> dict:
    pin = g.PINS["sources"][0]
    files = {f"{DECODER_ASSET}/{n}": {"bytes": 1, "sha256": "0" * 64} for n in ("main.hash", "main.mlirb", "metadata.json")}
    files.update({f"tokenizer/{n}": {"bytes": 1, "sha256": "0" * 64} for n in (*TOKENIZER_FILES, "chat_template.jinja")})
    files.update({n: {"bytes": 1, "sha256": "0" * 64} for n in ("LICENSE", "NOTICE", "README.md")})
    report = {"scriptSHA256": {"export_qwen3.py": "0" * 64}, "forgeLockSHA256": "0" * 64}
    return manifest(files, report, platform_name, precision(WEIGHTS, DECODER_COMPUTE, ATTENTION, LOGITS), pin,
                    [(151645, "<|im_end|>"), (151643, "<|endoftext|>")], "", g.CONTEXT_MAX)


def notice(pin: dict, cfg: dict, weights: str) -> str:
    base = f", a derivative of {pin['base']}," if pin.get("base") else ""
    return "\n".join([
        f"{pin['repository']} (Aether-Kit Core AI export)",
        "",
        f"This bundle is a modified version of {pin['repository']} at revision {pin['revision']}{base}",
        f"licensed under the {pin['license']} license (LICENSE).",
        "",
        "Modifications: Aether-Kit's Forge exporter (Forge/qwen3/) re-expressed the decoder as one Core AI graph with",
        f"in-place KV states ({cfg['num_hidden_layers']} layers) and last-row logits; the q/k/v projections are",
        "concatenated, " + ("and the weights were rounded from bfloat16 to float16." if weights == "float16"
                            else "and the weights were widened from bfloat16 to float32."),
        "The tokenizer files and chat template are the checkpoint's, unmodified.",
        "It is not an official release of the checkpoint's authors.",
        "",
    ])


def readme(pin: dict, cfg: dict, prec: dict, context: int) -> str:
    return "\n".join([
        "---",
        f"license: {pin['license'].lower()}",
        f"base_model: {pin['repository']}",
        "pipeline_tag: text-generation",
        "tags: [aether-kit, core-ai, qwen3]",
        "---",
        "",
        f"# {pin['repository'].split('/')[-1]} for Aether-Kit (Core AI)",
        "",
        f"[{pin['repository']}](https://huggingface.co/{pin['repository']}) at revision `{pin['revision'][:8]}`, exported",
        "by [Aether-Kit](https://github.com/bubstackdev/aether-kit)'s Forge exporter (`Forge/qwen3/`) as one Core AI graph",
        f"(`{DECODER_ASSET}`): the Qwen3 decoder ({cfg['num_hidden_layers']} layers, hidden {cfg['hidden_size']},",
        f"{cfg['num_attention_heads']} query / {cfg['num_key_value_heads']} KV heads) with in-place KV states, up to",
        f"{context} positions, and last-row logits. Precision: {prec['components']['decoder']}.",
        "",
        "Load it with Aether-Kit (`decoder-lm@1`, macOS) from its folder; the manifest is `.aether/aether.json`.",
        "",
        f"Licensed under {pin['license']} (`LICENSE`); a modified version, not an official release (see `NOTICE`).",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=f"{g.PINS['sources'][0]['repository']}@{g.PINS['sources'][0]['revision']}",
                        help="repo@revision, pinned in source-pins.json")
    parser.add_argument("--snapshot", type=Path, help="the snapshot folder (default: the Hugging Face cache path)")
    parser.add_argument("--output", type=Path, help="new bundle directory")
    parser.add_argument("--platform", choices=["macos"], default="macos")
    parser.add_argument("--sample-manifest", action="store_true")
    parser.add_argument("--context", type=int, default=g.CONTEXT_MAX, help="the position and KV axes' bound")
    parser.add_argument("--weights", choices=["float16", "float32"], default=WEIGHTS)
    parser.add_argument("--compute", choices=["float16", "float32"], default=DECODER_COMPUTE)
    parser.add_argument("--attention", choices=["float16", "float32"], default=ATTENTION)
    parser.add_argument("--logits", choices=["float16", "float32"], default=LOGITS)
    parser.add_argument("--layers", type=int, help="debug: keep only the first N decoder layers")
    parser.add_argument("--creation-date", default=CREATION_DATE)
    args = parser.parse_args()
    if args.sample_manifest:
        print(json.dumps(sample_manifest(args.platform), indent=2))
        return
    if args.output is None:
        parser.error("--output is required")
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    pin = g.source_pin(args.source)
    snapshot = args.snapshot or g.default_snapshot(pin)
    creation_date = int(dt.datetime.fromisoformat(args.creation_date).timestamp())
    started = time.monotonic()
    args.output.mkdir(parents=True)
    log = Log(args.output / "forge-log.jsonl")
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    prec = precision(args.weights, args.compute, args.attention, args.logits)
    report = {
        "schema": 1, "id": pin["id"], "command": sys.argv,
        "exportedAt": dt.datetime.now(dt.timezone.utc).isoformat(), "creationDate": args.creation_date,
        "host": {"os": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
                 "machine": platform.machine(), "python": platform.python_version()},
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "coreai-torch", "coreai-core", "numpy",
                                                                "safetensors", "transformers")},
        "forgeLockSHA256": g.sha256_file(HERE.parent / "uv.lock"),
        "forgePyprojectSHA256": g.sha256_file(HERE.parent / "pyproject.toml"),
        "scriptSHA256": {p.name: g.sha256_file(p) for p in sorted(HERE.glob("*.py"))},
        "source": {"repository": pin["repository"], "revision": pin["revision"], "snapshot": str(snapshot.resolve()),
                   "verified": {}},
        "precision": prec,
        "precisionArgs": {"weights": args.weights, "compute": args.compute, "attention": args.attention,
                          "logits": args.logits},
        "prefillMax": g.PREFILL_MAX, "contextMax": args.context, "compute": COMPUTE, "conversion": {}, "stages": {},
    }
    report_path = args.output / "export-report.json"

    def save() -> None:
        report["peakRSSGB"] = peak_rss_gb()
        report["wallSeconds"] = round(time.monotonic() - started, 1)
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    log.start("verify-sources")
    report["source"]["verified"] = g.verify_sources(snapshot, pin)
    cfg = g.read_config(snapshot)
    g.check_config(cfg)
    report["config"] = cfg
    report["declaredContext"] = cfg.get("max_position_embeddings")
    stops = stop_tokens(snapshot)
    template = chat_template(snapshot)
    report["stages"]["verify-sources"] = log.done("verify-sources")
    save()

    log.start("load-weights")
    weights = g.load_weights(snapshot)
    if args.layers:
        cfg = {**cfg, "num_hidden_layers": args.layers, "layer_types": cfg.get("layer_types", [])[:args.layers]}
        weights = {k: v for k, v in weights.items()
                   if not k.startswith("model.layers.") or int(k.split(".")[2]) < args.layers}
        report["debugLayers"] = args.layers
    decoder = g.Decoder(cfg, weights, dtype=g.DTYPES[args.compute], attention=g.DTYPES[args.attention],
                        storage=g.DTYPES[args.weights], logits=g.DTYPES[args.logits], context=args.context).eval()
    del weights
    gc.collect()
    report["stages"]["load-weights"] = log.done("load-weights")

    log.start("export-decoder")
    description = (f"Qwen3 decoder ({pin['repository']}@{pin['revision'][:12]}): input_ids, position_ids i32 with "
                   f"in-place keyCache/valueCache states -> last-row logits {SHORT[args.logits]} [1,1,{cfg['vocab_size']}].")
    report["conversion"]["decoder"] = convert(decoder, args.output / DECODER_ASSET, creation_date, description, pin)
    del decoder
    gc.collect()
    report["stages"]["export-decoder"] = log.done("export-decoder")
    save()

    log.start("package")
    (args.output / "tokenizer").mkdir()
    for name in TOKENIZER_FILES:
        if (snapshot / name).is_file():
            shutil.copyfile(snapshot / name, args.output / "tokenizer" / name)
    (args.output / "tokenizer/chat_template.jinja").write_text(template, encoding="utf-8", newline="")
    shutil.copyfile(HERE.parent.parent / "LICENSE" if pin["license"] == "Apache-2.0" else snapshot / "LICENSE",
                    args.output / "LICENSE")
    (args.output / "NOTICE").write_text(notice(pin, cfg, args.weights))
    (args.output / "README.md").write_text(readme(pin, cfg, prec, args.context))
    files = {str(p.relative_to(args.output)): {"bytes": p.stat().st_size, "sha256": g.sha256_file(p)}
             for p in tree_files(args.output) if str(p.relative_to(args.output)) not in NOT_BUNDLE}
    report["bundleBytes"] = sum(f["bytes"] for f in files.values())
    report["stages"]["package"] = log.done("package")
    save()
    (args.output / ".aether").mkdir()
    manifest_path = args.output / ".aether/aether.json"
    doc = manifest(files, report, args.platform, prec, pin, stops, template, args.context,
                   template_sha=g.sha256_file(args.output / "tokenizer/chat_template.jinja"))
    manifest_path.write_text(json.dumps(doc, indent=2) + "\n")
    report["aetherJSONSHA256"] = g.sha256_file(manifest_path)
    save()
    sums = [f"{g.sha256_file(p)}  {p.relative_to(args.output)}" for p in tree_files(args.output)
            if p.name not in ("SHA256SUMS", "forge-log.jsonl")]
    (args.output / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(json.dumps({"conversion": report["conversion"], "bundleBytes": report["bundleBytes"],
                      "peakRSSGB": report["peakRSSGB"], "wallSeconds": report["wallSeconds"]}, indent=1))


if __name__ == "__main__":
    main()

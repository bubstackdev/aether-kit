#!/usr/bin/env python3
"""Export the Qwen3.5 hybrid text decoder as an Aether Kit language asset (development tooling only).

Runs in Forge's locked environment (`uv sync --frozen --project Forge`):

    Forge/.venv/bin/python -I Forge/language/export_qwen35_text.py --checkpoint <hf snapshot> --static-width 64 \
        --source-repository Qwen/Qwen3.5-4B --source-revision <sha> --license apache-2.0 --output <new dir>
    Forge/.venv/bin/python -I Forge/language/export_qwen35_text.py --tiny --static-width 8 --output <new dir>

The contract is in `qwen35_hybrid.py`: one `main` at a fixed width W (S2, the qualified layout) or with a dynamic q
(D1, `--layout D1`), KV capacity dynamic, two declared recurrent states. `--delta-rule` picks the gated delta rule's
form: `composite` (default: coreai-torch's `GatedDeltaUpdate`, emitted as a `gated_delta_update` composite graph;
`--no-externalize-composite` traces its `while_loop` instead), `chunk` (loop-free WY form in plain ops) or `loop`
(unrolled per token). On the 4B, M5 Max (docs/spikes/qwen3.5-vl-2026-11.md): S2 at W = 64 decodes at 31-33 tok/s with
the externalized composite, 20 tok/s with `chunk` or the traced loop. Writes only under the new `--output` directory: `model.aimodel/`, `tokenizer/`
(checkpoint exports) and `export-report.json` (versions, the environment, provenance, file hashes, the F-P2-20
state-order lint). There is no cpu_only self-check: Core AI's cpu_only path fails a static q > 1 over dynamic KV
(0.0.3 defect 5), so these graphs qualify natively against the upstream reference.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_qwen3 import drop_identity_copies  # noqa: E402
from granite_hybrid import state_order_violations  # noqa: E402
from qwen35_hybrid import (DELTA_RULES, PaddedMain, Qwen35HybridDecoder, build, load_weights, padded_example,  # noqa: E402
                           read_config, start_scalar_example)

INPUTS = ["input_ids", "cache_positions", "rope_positions", "image_embeds", "keyCache", "valueCache"]
STATES = ["convState", "recurrentState"]
OUTPUTS = ["logits", "keyUpdate", "valueUpdate"]
ROOT = Path(__file__).resolve().parents[2]
TINY = {"vocab_size": 512, "hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 8, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 64, "linear_num_key_heads": 2, "linear_num_value_heads": 4,
        "linear_key_head_dim": 32, "linear_value_head_dim": 32, "linear_conv_kernel_dim": 4, "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True, "attn_output_gate": True, "attention_bias": False,
        "layer_types": ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(8)],
        "rope_parameters": {"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                            "mrope_section": [3, 3, 2], "mrope_interleaved": True}}


def tiny_module(delta_rule: str, max_capacity: int = 16384) -> Qwen35HybridDecoder:
    """The random tiny model, fp16, reproducibly (seed 0): GPU checks rebuild the exported weights with this."""
    torch.manual_seed(0)
    module = Qwen35HybridDecoder(dict(TINY), 511, max_capacity=max_capacity, delta_rule=delta_rule)
    with torch.no_grad():
        for n, p in module.named_parameters():
            if "norm" in n:
                p.fill_(1.0)
            elif n.endswith("A_log"):
                p.copy_(torch.log(torch.rand_like(p) * 8 + 1))
            else:
                p.normal_(0, 0.05)
    return module.to(torch.float16)


def widen_collapsed_composite_dims(max_width: int) -> None:
    """Export workaround, not a fix (coreai-torch 0.4.2): when the D1 graph's `GatedDeltaUpdate` call is externalized,
    coreai-torch rebuilds the composite's dynamic shapes from the call site's fake tensors, and the token dimension
    arrives as a fresh symbol whose recorded range has collapsed to the example's width (VR[w, w]), so the composite
    would be exported for that one width. A collapsed range is replaced by the D1 range 1..max_width; the native
    checks (q = W and q = 1 against the reference) are what qualify the result."""
    import coreai_torch._utils as utils
    from torch.export import Dim

    original = utils._dim_for_sym

    def dim_for_sym(s, cache):
        r = s.node.shape_env.var_to_range.get(s.node.expr)
        key = str(s.node.expr)
        if r is not None and int(r.lower) == int(r.upper) and key not in cache:
            cache[key] = Dim(re.sub(r"[^a-zA-Z0-9_]", "_", key), min=1, max=max_width)
        return original(s, cache)

    utils._dim_for_sym = dim_for_sym


def stage(name: str) -> None:
    print(f"[export] {name}", file=sys.stderr, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def environment() -> dict:
    project = Path(sys.prefix).resolve().parent
    lock = project / "uv.lock"
    return {"project": str(project.relative_to(ROOT)) if project.is_relative_to(ROOT) else str(project),
            "uvLockSHA256": sha256(lock) if lock.exists() else None, "python": sys.version.split()[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tiny", action="store_true", help="random-weight eight-layer model (contract check)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--static-width", type=int, default=64,
                        help="S2: main's fixed width; D1: the prefill width W (q is dynamic up to it)")
    parser.add_argument("--layout", default="S2", choices=["S2", "D1"],
                        help="S2: static width W; D1: dynamic q with cache_positions [1, 2] = (start, valid count)")
    parser.add_argument("--max-capacity", type=int, default=16384)
    parser.add_argument("--delta-rule", default="composite", choices=DELTA_RULES)
    parser.add_argument("--no-externalize-composite", dest="externalize_composite", action="store_false",
                        help="trace GatedDeltaUpdate's while_loop instead of emitting the gated_delta_update composite")
    parser.add_argument("--source-repository", default="")
    parser.add_argument("--source-revision", default="")
    parser.add_argument("--license", default="")
    args = parser.parse_args()
    if args.tiny == (args.checkpoint is not None):
        raise SystemExit("pass exactly one of --checkpoint or --tiny")
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    if args.delta_rule != "composite":
        args.externalize_composite = False

    import coreai_torch as ct

    if args.tiny:
        module = tiny_module(args.delta_rule, args.max_capacity)
        cfg, image_token = module.cfg, module.image_token
    else:
        cfg, image_token = read_config(args.checkpoint)
        stage("load weights")
        module = build(cfg, image_token, torch.float16, max_capacity=args.max_capacity, delta_rule=args.delta_rule)
        load_weights(module, args.checkpoint)
    module = module.eval()

    args.output.mkdir(parents=True)
    asset = args.output / "model.aimodel"
    if args.layout == "D1":
        module.start_scalar = True
        example, dynamic = start_scalar_example(module, args.static_width)
    else:
        example, dynamic = padded_example(module, args.static_width)
    wrapper = PaddedMain(module)
    if args.externalize_composite:
        from coreai_torch.composite_ops import GatedDeltaUpdate
        from coreai_torch.externalize import ExternalizeSpec, _patch_model_for_externalization, _subexport_and_restore

        if args.layout == "D1":
            widen_collapsed_composite_dims(args.static_width)
        _patch_model_for_externalization(wrapper, [ExternalizeSpec(GatedDeltaUpdate, composite_op_name="gated_delta_update",
                                                                   composite_attrs=["use_qk_l2_norm"])])
    stage("export main")
    with torch.no_grad():
        ep = torch.export.export(wrapper, example, dynamic_shapes=dynamic, prefer_deferred_runtime_asserts_over_guards=True)
        externalized = _subexport_and_restore(wrapper, ep) if args.externalize_composite else None
        ep = ep.run_decompositions(ct.get_decomp_table())
    dropped = drop_identity_copies(ep)
    mutated = list(ep.graph_signature.user_inputs_to_mutate.values())
    if len(mutated) != 2:
        raise SystemExit(f"main: expected the two recurrent states to be mutated, got {mutated}")
    lint = state_order_violations(ep)
    if lint:
        raise SystemExit(f"main: state reads that do not feed their write (F-P2-20): {lint}")
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    kwargs = {"_externalized_exported_programs": externalized} if externalized else {}
    converter.add_exported_program(ep, input_names=INPUTS, output_names=OUTPUTS, state_names=STATES, entrypoint_name="main",
                                   **kwargs)
    stage("convert")
    program = converter.to_coreai()
    stage("optimize")
    program.optimize()
    stage("save")
    program.save_asset(asset)

    report = {
        "schema": 1,
        "recipe": f"qwen35-hybrid-{args.layout.lower()}-w{args.static_width}-{args.delta_rule}" + ("-externalized" if args.externalize_composite else ""),
        "contract": "hybrid",
        "functions": ["main"],
        "layout": args.layout,
        "exportedAt": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hostOS": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
        "environment": environment(),
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "coreai-torch", "coreai-core")},
        "scriptSHA256": {p.name: sha256(p) for p in sorted(Path(__file__).resolve().parent.glob("*.py"))},
        "dtype": "float16 weights, KV and convState; float32 recurrentState and gated delta rule",
        "staticWidth": args.static_width,
        "exportBounds": {"maximumPrefillChunk": args.static_width, "maximumImageChunk": args.static_width,
                         "maximumCacheCapacity": args.max_capacity},
        "states": [{"name": "convState", "role": "recurrent", "scalarType": "float16", "shape": list(module.empty_state(1)[2].shape)},
                   {"name": "recurrentState", "role": "recurrent", "scalarType": "float32", "shape": list(module.empty_state(1)[3].shape)}],
        "imageTokenID": image_token,
        "deltaRule": args.delta_rule,
        "externalizedComposite": args.externalize_composite,
        "droppedIdentityCopies": dropped,
        "stateOrderLint": lint,
        "config": cfg,
        "provenance": None if args.tiny else {
            "repository": args.source_repository, "revision": args.source_revision, "license": args.license,
            "checkpointSHA256": {p.name: sha256(p) for p in sorted(args.checkpoint.glob("*.safetensors"))}},
    }
    if not args.tiny:
        tok = args.output / "tokenizer"
        tok.mkdir()
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json",
                     "chat_template.jinja", "preprocessor_config.json"):
            if (args.checkpoint / name).exists():
                shutil.copyfile(args.checkpoint / name, tok / name)
        report["tokenizerFiles"] = tree_hashes(tok)
    report["modelFiles"] = tree_hashes(asset)
    (args.output / "export-report.json").write_text(json.dumps(report, indent=2) + "\n")
    stage("done")


if __name__ == "__main__":
    main()

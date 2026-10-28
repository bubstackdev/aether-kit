#!/usr/bin/env python3
"""Export a Granite 4.0-H dense hybrid checkpoint as an Aether Kit language asset (development tooling only).

Runs in Forge's locked environment (`uv sync --frozen --project Forge`):

    Forge/.venv/bin/python -I Forge/language/export_granite_hybrid.py --checkpoint <hf snapshot> [--decode] --output <dir>
    Forge/.venv/bin/python -I Forge/language/export_granite_hybrid.py --tiny [--decode] --output <dir>

`--decode` adds the q = 1 `decode` function (the C2 layout); without it the asset has `main` only (C1). The contract
is in `granite_hybrid.py`. Writes only under the new `--output` directory: `model.aimodel/`, `tokenizer/` (checkpoint
exports) and `export-report.json` with versions, file hashes, the F-P2-20 state-order lint and a cpu_only self-check
against the eager module: chunked prefill in uneven widths, then decode steps, comparing logits, every returned KV row
and both recurrent states after every step.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_qwen3 import drop_identity_copies  # noqa: E402
from granite_hybrid import Decode, GraniteHybridDecoder, Main, examples, load_weights, read_config, state_order_violations  # noqa: E402

INPUTS = ["input_ids", "cache_positions", "keyCache", "valueCache"]
STATES = ["convState", "ssmState"]
OUTPUTS = ["logits", "keyUpdate", "valueUpdate"]
TINY = {"vocab_size": 512, "hidden_size": 128, "shared_intermediate_size": 256, "num_hidden_layers": 4,
        "layer_types": ["mamba", "attention", "mamba", "mamba"], "num_attention_heads": 4, "num_key_value_heads": 2,
        "mamba_n_heads": 8, "mamba_d_head": 32, "mamba_d_state": 16, "mamba_expand": 2, "mamba_n_groups": 1,
        "mamba_d_conv": 4, "mamba_conv_bias": True, "mamba_proj_bias": False, "num_local_experts": 0,
        "position_embedding_type": "nope", "embedding_multiplier": 12.0, "residual_multiplier": 0.246,
        "logits_scaling": 3.0, "attention_multiplier": 0.03125, "tie_word_embeddings": True, "rms_norm_eps": 1e-5}


def strip_runtime_asserts(ep) -> int:
    """Remove deferred shape asserts: export workaround for guards torch cannot prove on the q = 1 graph's strides
    (`min(32 c + 32, 64 c + 64) == 32 c + 32`); coreai-torch has no lowering for the asserts' Python comparisons."""
    removed = 0
    for n in list(ep.graph.nodes):
        if n.op == "call_function" and n.target == torch.ops.aten._assert_scalar.default:
            ep.graph.erase_node(n)
            removed += 1
    ep.graph.eliminate_dead_code()
    ep.graph_module.recompile()
    return removed


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


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def plan(prompt: int, widths: list[int], decode: int) -> list[tuple[int, int]]:
    """(start, count) per submission: prompt chunks in `widths` order (the last repeats), then one-token steps."""
    out, start, i = [], 0, 0
    while start < prompt:
        n = min(widths[min(i, len(widths) - 1)], prompt - start)
        out.append((start, n))
        start, i = start + n, i + 1
    return out + [(prompt + j, 1) for j in range(decode)]


def self_check(asset: Path, module: GraniteHybridDecoder, decode_fn: bool, vocab: int) -> dict:
    import coreai.runtime as rt

    rng = np.random.default_rng(7)
    steps = plan(23, [8, 4, 1, 2], 4)
    total = steps[-1][0] + 1
    tokens = rng.integers(min(1000, vocab // 2), vocab, total).astype(np.int32)
    capacity = 64

    # The eager reference, stepping exactly like the asset run (recurrent form on decode steps when exported).
    ref = []
    kc, vc, cs, ss = module.empty_state(capacity)
    with torch.no_grad():
        for start, n in steps:
            ids = torch.from_numpy(tokens[None, start:start + n].copy())
            pos = torch.arange(start, start + n, dtype=torch.int32)[None]
            row = module.step(ids, pos, kc, vc, cs, ss, recurrent=decode_fn and n == 1 and start > 0)
            ref.append((row.float().numpy(), kc.float().numpy().copy(), cs.float().numpy().copy(), ss.numpy().copy()))

    async def run():
        model = await rt.AIModel.load(asset, rt.SpecializationOptions.cpu_only())
        main = model.load_function("main")
        decode = model.load_function("decode") if decode_fn else None
        shape = [s.shape for s in module.empty_state(capacity)]
        kc, vc = np.zeros(shape[0], np.float16), np.zeros(shape[1], np.float16)
        states = {"convState": rt.NDArray(np.zeros(shape[2], np.float16)),
                  "ssmState": rt.NDArray(np.zeros(shape[3], np.float32))}
        got = []
        for start, n in steps:
            fn = decode if decode is not None and n == 1 and start > 0 else main
            feed = {"input_ids": rt.NDArray(tokens[None, start:start + n].copy()),
                    "cache_positions": rt.NDArray(np.arange(start, start + n, dtype=np.int32)[None]),
                    "keyCache": rt.NDArray(kc), "valueCache": rt.NDArray(vc)}
            r = await fn(feed, states)
            logits = r["logits"].numpy()
            if logits.shape[1] != 1:
                raise SystemExit(f"logits must be last-row only, got {logits.shape}")
            kc[:, :, :, start:start + n] = r["keyUpdate"].numpy()
            vc[:, :, :, start:start + n] = r["valueUpdate"].numpy()
            got.append((logits[0, -1].astype(np.float32), kc.astype(np.float32).copy(),
                        states["convState"].numpy().astype(np.float32).copy(), states["ssmState"].numpy().copy()))
        return got

    got = asyncio.run(run())
    rows = []
    for (start, n), (er, ek, ec, es), (gr, gk, gc, gs) in zip(steps, ref, got):
        end = start + n
        p = np.exp(er.astype(np.float64) - er.max())
        p /= p.sum()
        top = np.argsort(-p)
        rows.append({"start": start, "tokens": n, "function": "decode" if decode_fn and n == 1 and start > 0 else "main",
                     "cosine": round(cosine(er, gr), 6), "argmaxEqual": int(np.argmax(er)) == int(np.argmax(gr)),
                     "nearTie": bool(p[top[0]] - p[top[1]] < 0.1 and int(np.argmax(gr)) in top[:5].tolist()),
                     "keyCosine": round(cosine(ek[:, :, :, :end], gk[:, :, :, :end]), 6),
                     "convStateCosine": round(cosine(ec, gc), 6), "ssmStateCosine": round(cosine(es, gs), 6),
                     "ssmStateMaxAbs": round(float(np.abs(es - gs).max()), 6)})
    # The engine forge's fp16 bar (cosine >= 0.99), argmax equal unless at Kit's near tie (eager top-1 minus top-2
    # probability < 0.1, the asset's token in the eager top 5); the states are compared the same way.
    passed = all(r["cosine"] >= 0.99 and (r["argmaxEqual"] or r["nearTie"]) and r["keyCosine"] >= 0.99 and r["convStateCosine"] >= 0.99
                 and r["ssmStateCosine"] >= 0.99 for r in rows)
    return {"passed": passed, "capacity": capacity, "plan": steps,
            "bar": "cosine >= 0.99 for logits, KV rows and both states; argmax equal or a near tie", "steps": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tiny", action="store_true", help="random-weight four-layer model (contract check)")
    parser.add_argument("--decode", action="store_true", help="also export the q = 1 `decode` function (C2)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-prefill-chunk", type=int, default=512)
    parser.add_argument("--max-capacity", type=int, default=16384)
    parser.add_argument("--source-repository", default="")
    parser.add_argument("--source-revision", default="")
    parser.add_argument("--no-self-check", action="store_true")
    args = parser.parse_args()
    if args.tiny == (args.checkpoint is not None):
        raise SystemExit("pass exactly one of --checkpoint or --tiny")
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")

    import coreai_torch as ct

    torch.manual_seed(0)
    cfg = dict(TINY) if args.tiny else read_config(args.checkpoint)
    module = GraniteHybridDecoder(cfg, max_capacity=args.max_capacity)
    if args.tiny:
        with torch.no_grad():
            for n, p in module.named_parameters():
                if "norm" in n:
                    p.fill_(1.0)
                elif n.endswith("A_log"):
                    p.copy_(torch.log(torch.rand_like(p) * 8 + 1))
                else:
                    p.normal_(0, 0.05)
    else:
        load_weights(module, args.checkpoint)
    module = module.to(torch.float16).eval()

    args.output.mkdir(parents=True)
    asset = args.output / "model.aimodel"
    main_ex, main_dyn, dec_ex, dec_dyn = examples(module, args.max_prefill_chunk)
    entries = [("main", Main(module), main_ex, main_dyn)]
    if args.decode:
        entries.append(("decode", Decode(module), dec_ex, dec_dyn))
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    lint, dropped, asserts = {}, {}, {}
    for name, wrapper, example, dynamic in entries:
        stage(f"export {name}")
        with torch.no_grad():
            ep = torch.export.export(wrapper, example, dynamic_shapes=dynamic,
                                     prefer_deferred_runtime_asserts_over_guards=True)
            ep = ep.run_decompositions(ct.get_decomp_table())
        asserts[name] = strip_runtime_asserts(ep)
        dropped[name] = drop_identity_copies(ep)
        mutated = list(ep.graph_signature.user_inputs_to_mutate.values())
        if len(mutated) != 2:
            raise SystemExit(f"{name}: expected the two recurrent states to be mutated, got {mutated}")
        lint[name] = state_order_violations(ep)
        if lint[name]:
            raise SystemExit(f"{name}: state reads that do not feed their write (F-P2-20): {lint[name]}")
        converter.add_exported_program(ep, input_names=INPUTS, output_names=OUTPUTS, state_names=STATES,
                                       entrypoint_name=name)
    stage("convert")
    program = converter.to_coreai()
    stage("optimize")
    program.optimize()
    stage("save")
    program.save_asset(asset)

    report = {
        "schema": 1,
        "recipe": "granite-hybrid-bucketed-v1" + ("-decode" if args.decode else ""),
        "contract": "hybrid",
        "functions": [name for name, *_ in entries],
        "exportedAt": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hostOS": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "coreai-torch", "coreai-core")},
        "scriptSHA256": {p.name: sha256(p) for p in sorted(Path(__file__).resolve().parent.glob("*.py"))},
        "dtype": "float16 weights and KV, float32 ssmState",
        "exportBounds": {"maximumPrefillChunk": args.max_prefill_chunk, "maximumCacheCapacity": args.max_capacity},
        "droppedIdentityCopies": dropped,
        "strippedShapeAsserts": asserts,
        "stateOrderLint": lint,
        "config": cfg,
        "source": {"repository": args.source_repository, "revision": args.source_revision,
                   "checkpointSHA256": None if args.tiny else {p.name: sha256(p) for p in sorted(args.checkpoint.glob("*.safetensors"))}},
        "selfCheck": None,
    }
    if not args.tiny:
        tok = args.output / "tokenizer"
        tok.mkdir()
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json",
                     "config.json", "chat_template.jinja", "special_tokens_map.json"):
            if (args.checkpoint / name).exists():
                shutil.copyfile(args.checkpoint / name, tok / name)
        report["tokenizerFiles"] = tree_hashes(tok)
    report["modelFiles"] = tree_hashes(asset)
    report_path = args.output / "export-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if args.no_self_check:
        print("[export] self-check skipped: qualify natively", file=sys.stderr)
        return
    stage("self-check")
    check = self_check(asset, module, args.decode, cfg["vocab_size"])
    report["selfCheck"] = check
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": check["passed"], "steps": check["steps"]}, indent=1))
    if not check["passed"]:
        raise SystemExit("self-check failed")


if __name__ == "__main__":
    main()

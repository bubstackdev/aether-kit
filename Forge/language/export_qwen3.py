#!/usr/bin/env python3
"""Export a Qwen3 dense checkpoint as an Aether Kit language model asset (development tooling only).

Runs in Forge's locked environment (`uv sync --frozen --project Forge`; `<venv>` is `Forge/.venv`):

    <venv>/bin/python -I Forge/language/export_qwen3.py --checkpoint <hf snapshot> [--contract running] --output <dir>
    <venv>/bin/python -I Forge/language/export_qwen3.py --tiny --contract bucketed --output <dir>

Writes only under the new `--output` directory: `model.aimodel/`, `tokenizer/` (checkpoint exports), and
`export-report.json` with versions, file hashes and a cpu_only self-check against the eager module (prefill, chunked
prefill and decode steps, including every cache row written). Python loads assets cpu_only only; GPU behaviour is
qualified natively in Swift.
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
from qwen3_decode import CONTRACTS, MAX_PREFILL_CHUNK, POSITION_INPUT, Qwen3Decoder, load_weights, read_config  # noqa: E402

TINY = {"vocab_size": 512, "hidden_size": 256, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64,
        "num_hidden_layers": 2, "intermediate_size": 512, "rms_norm_eps": 1e-6, "rope_theta": 1_000_000.0,
        "tie_word_embeddings": True}


def drop_identity_copies(ep) -> int:
    """Remove same-shape `aten.copy` nodes left by each in-place cache write (F-P2-13)."""
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


# MARK: step plans shared by the eager reference and the asset run

def step_inputs(contract: str, tokens: list[int], start: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.array([tokens], dtype=np.int32)
    if contract == "running":
        return ids, np.arange(start + len(tokens), dtype=np.int32)[None]
    return ids, np.arange(start, start + len(tokens), dtype=np.int32)[None]


def plan(prompt: list[int], chunks: list[int]) -> list[tuple[int, list[int]]]:
    """(start, tokens) per prefill submission; `chunks` sizes are used in order, the last one repeats."""
    out, start, i = [], 0, 0
    while start < len(prompt):
        n = min(chunks[min(i, len(chunks) - 1)], len(prompt) - start)
        out.append((start, prompt[start:start + n]))
        start, i = start + n, i + 1
    return out


def eager_run(module: Qwen3Decoder, prompt: list[int], chunks: list[int], decode: int, capacity: int):
    """Last-row logits after prefill and after each greedy decode step, plus cache snapshots."""
    kc, vc = module.empty_state(capacity)
    rows, snaps, ids = [], [], list(prompt)
    with torch.no_grad():
        for start, toks in plan(prompt, chunks):
            a, b = step_inputs(module.contract, toks, start)
            out = module.step(torch.from_numpy(a), torch.from_numpy(b), kc, vc)
        rows.append(out.float().numpy())
        snaps.append((kc.float().numpy().copy(), vc.float().numpy().copy()))
        for _ in range(decode):
            ids.append(int(np.argmax(rows[-1])))
            a, b = step_inputs(module.contract, [ids[-1]], len(ids) - 1)
            out = module.step(torch.from_numpy(a), torch.from_numpy(b), kc, vc)
            rows.append(out.float().numpy())
            snaps.append((kc.float().numpy().copy(), vc.float().numpy().copy()))
    return rows, snaps, ids[len(prompt):]


async def asset_run(asset: Path, contract: str, prompt: list[int], chunks: list[int], forced: list[int],
                    state_shape: tuple[int, ...]):
    """Running: caches are Core AI states. Bucketed: caches are host arrays passed as inputs, and the returned rows
    are written at start..<start+q after each call, as the Kit session does."""
    import coreai.runtime as rt

    model = await rt.AIModel.load(asset, rt.SpecializationOptions.cpu_only())
    fn = model.load_function("main")
    kc, vc = np.zeros(state_shape, np.float16), np.zeros(state_shape, np.float16)
    states = {"keyCache": rt.NDArray(kc), "valueCache": rt.NDArray(vc)}

    async def call(tokens: list[int], start: int) -> np.ndarray:
        a, b = step_inputs(contract, tokens, start)
        feed = {"input_ids": rt.NDArray(a), POSITION_INPUT[contract]: rt.NDArray(b)}
        if contract == "running":
            r = await fn(feed, states)
            return r["logits"].numpy()[0, -1].astype(np.float32)
        r = await fn({**feed, "keyCache": rt.NDArray(kc), "valueCache": rt.NDArray(vc)}, {})
        logits = r["logits"].numpy()
        if logits.shape[1] != 1:
            raise SystemExit(f"bucketed logits must be last-row only, got {logits.shape}")
        kc[:, :, :, start:start + len(tokens)] = r["keyUpdate"].numpy()
        vc[:, :, :, start:start + len(tokens)] = r["valueUpdate"].numpy()
        return logits[0, -1].astype(np.float32)

    def snapshot():
        if contract == "running":
            return states["keyCache"].numpy().copy(), states["valueCache"].numpy().copy()
        return kc.copy(), vc.copy()

    rows, snaps = [], []
    for start, toks in plan(prompt, chunks):
        last = await call(toks, start)
    rows.append(last)
    snaps.append(snapshot())
    ids = list(prompt)
    for tok in forced:
        ids.append(tok)
        rows.append(await call([tok], len(ids) - 1))
        snaps.append(snapshot())
    return rows, snaps


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def self_check(asset: Path, module: Qwen3Decoder, vocab: int, chunks: list[int], decode: int, capacity: int) -> dict:
    span = max(2, min(vocab, 100_000) - 1000) if vocab > 2000 else vocab
    base = 1000 if vocab > 2000 else 0
    prompt = [base + (97 * i + 31) % span for i in range(19)]
    ref, ref_snaps, generated = eager_run(module, prompt, chunks, decode, capacity)
    shape = tuple(module.empty_state(capacity)[0].shape)
    got, got_snaps = asyncio.run(asset_run(asset, module.contract, prompt, chunks, generated, shape))
    steps = []
    for i, (a, b, (ek, ev), (gk, gv)) in enumerate(zip(ref, got, ref_snaps, got_snaps)):
        total = len(prompt) + i
        lost = 0
        for e, g in ((ek, gk), (ev, gv)):
            e, g = e[:, :, :, :total], g[:, :, :, :total]
            lost += int(((np.abs(g).max(axis=-1) == 0) & (np.abs(e).max(axis=-1) > 0)).sum())
        cache_err = max(float(np.abs(gk[:, :, :, :total].astype(np.float32) - ek[:, :, :, :total]).max()),
                        float(np.abs(gv[:, :, :, :total].astype(np.float32) - ev[:, :, :, :total]).max()))
        steps.append({"step": i, "cosine": round(cosine(a, b), 6), "maxAbs": round(float(np.abs(a - b).max()), 5),
                      "argmaxEqual": int(np.argmax(a)) == int(np.argmax(b)), "lostCacheRows": lost,
                      "cacheMaxAbs": round(cache_err, 5)})
    # The engine forge's fp16 bar: Core AI's CPU fp16 arithmetic sits near 0.996 cosine against the eager module on
    # Qwen3-0.6B with identical argmax, while a broken state path falls far below (lost rows are checked exactly).
    passed = all(s["cosine"] >= 0.99 and s["lostCacheRows"] == 0 and s["argmaxEqual"] for s in steps)
    return {"passed": passed, "promptTokens": len(prompt), "chunks": chunks, "capacity": capacity,
            "decodeSteps": decode, "steps": steps}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tiny", action="store_true", help="random-weight two-layer model (contract spike)")
    parser.add_argument("--contract", choices=CONTRACTS, default="bucketed",
                        help="bucketed (default): bounded shapes, no recycling; running: 0.0.1 contract")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-prefill-chunk", type=int, default=MAX_PREFILL_CHUNK)
    parser.add_argument("--max-capacity", type=int, default=16384)
    parser.add_argument("--source-repository", default="")
    parser.add_argument("--source-revision", default="")
    parser.add_argument("--no-self-check", action="store_true",
                        help="skip the cpu_only self-check (selfCheck stays null; qualify natively instead)")
    args = parser.parse_args()
    if args.tiny == (args.checkpoint is not None):
        raise SystemExit("pass exactly one of --checkpoint or --tiny")
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")

    import coreai_torch as ct

    torch.manual_seed(0)
    cfg = dict(TINY) if args.tiny else read_config(args.checkpoint)
    module = Qwen3Decoder(cfg, args.contract, max_capacity=args.max_capacity)
    if args.tiny:
        with torch.no_grad():
            for p in module.parameters():
                p.normal_(0, 0.05)
            for n, p in module.named_parameters():
                if "norm" in n:
                    p.fill_(1.0)
    else:
        load_weights(module, args.checkpoint)
    module = module.to(torch.float16).eval()

    args.output.mkdir(parents=True)
    asset = args.output / "model.aimodel"
    example, dynamic = module.example(args.max_prefill_chunk)
    with torch.no_grad():
        ep = torch.export.export(module, example, dynamic_shapes=dynamic)
        ep = ep.run_decompositions(ct.get_decomp_table())
    dropped = drop_identity_copies(ep)
    mutated = list(ep.graph_signature.user_inputs_to_mutate.values())
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    if args.contract == "running":
        if len(mutated) != 2:
            raise SystemExit(f"expected two mutated cache inputs, got {mutated}")
        converter.add_exported_program(ep, input_names=["input_ids", "position_ids"], output_names=["logits"],
                                       state_names=["keyCache", "valueCache"], entrypoint_name="main")
    else:
        if mutated:
            raise SystemExit(f"the bucketed graph must not mutate its inputs, got {mutated}")
        converter.add_exported_program(ep, input_names=["input_ids", "cache_positions", "keyCache", "valueCache"],
                                       output_names=["logits", "keyUpdate", "valueUpdate"], state_names=[],
                                       entrypoint_name="main")
    stage("convert")
    program = converter.to_coreai()
    stage("optimize")
    program.optimize()
    stage("save")
    program.save_asset(asset)

    report = {
        "schema": 1,
        "recipe": f"qwen3-{args.contract}-v1",
        "contract": args.contract,
        "exportedAt": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hostOS": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "coreai-torch", "coreai-core")},
        "scriptSHA256": {p.name: sha256(p) for p in sorted(Path(__file__).resolve().parent.glob("*.py"))},
        "dtype": "float16",
        # The exported dynamic-dimension maxima, in the `LanguageExportBounds` shape a Kit registration declares
        # (`language-model.json` "exportBounds"); Kit refuses a bucketed asset without them.
        "exportBounds": {"maximumPrefillChunk": args.max_prefill_chunk, "maximumCacheCapacity": args.max_capacity},
        "droppedIdentityCopies": dropped,
        "config": cfg,
        "source": {"repository": args.source_repository, "revision": args.source_revision,
                   "checkpoint": None if args.tiny else str(args.checkpoint)},
        "selfCheck": None,
    }
    if not args.tiny:
        tok = args.output / "tokenizer"
        tok.mkdir()
        # The chat template travels with the tokenizer: some checkpoints keep it outside tokenizer_config.json
        # (chat_template.jinja or .json), and AetherLanguageTokenizers renders with the model's own template.
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json",
                     "config.json", "LICENSE", "chat_template.jinja", "chat_template.json", "added_tokens.json",
                     "special_tokens_map.json"):
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
    check = self_check(asset, module, cfg["vocab_size"], chunks=[8, 4, 1, 2], decode=4, capacity=64)
    report["selfCheck"] = check
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": check["passed"], "steps": check["steps"]}, indent=1))
    if not check["passed"]:
        raise SystemExit("self-check failed")


if __name__ == "__main__":
    main()

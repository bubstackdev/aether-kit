#!/usr/bin/env python3
"""Export the Qwen3-VL text tower with the bucketed one-function contract (development tooling only).

    Forge/.venv/bin/python -I Forge/language/export_qwen3_vl_text.py --checkpoint <hf snapshot> --output <new dir> \
        --source-repository Qwen/Qwen3-VL-2B-Instruct --source-revision <sha> --self-check-reference <reference dir>

One asset (`model.aimodel`) with one function, `main`, over fp16 weights: text prefill, image prefill and decode steps
all take `image_embeds` [q, hidden] and `deepstack` [levels, q, hidden], aligned with the tokens and used only where the
token is `<|image_pad|>` (qwen3_vl_text.py). Core AI wires one copy of the weights per function, so the 0.0.2 pair
(`main` + `withImage`) kept two copies resident (docs/qwen3-vl-single-decoder.md).

Memory is bounded by phase (the 0.0.2 pipeline held about five copies of the weights at its self-check):
1. the tower is built directly in fp16 (no float32 build);
2. the eager reference runs first: the float32 vision tower's rows for a reference case, then the eager tower's logits
   and update rows for the self-check plan, saved under `<output>/self-check/`; the vision tower is freed;
3. export, conversion and saving; then the process replaces itself (`exec`) with `--self-check <output>`, which loads
   only the asset (cpu_only) and the saved reference, so the eager module, the converter's copy and the save buffer
   are gone. The self-check writes its result into `export-report.json` and removes `self-check/`.

The self-check plan: the reference case's whole prompt in one submission (text and image tokens), forced decode
steps, and a text-only prefill and decode (the prompt's text tokens and the reference's greedy tokens, no image),
comparing last-row logits and every returned update row with the eager tower.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_qwen3 import drop_identity_copies  # noqa: E402
from qwen3_vl_text import Unified, build_fp16, load_weights as load_text, mrope_positions  # noqa: E402

INPUTS = ["input_ids", "cache_positions", "rope_positions", "image_embeds", "deepstack", "keyCache", "valueCache"]
OUTPUTS = ["logits", "keyUpdate", "valueUpdate"]
RECIPE = "qwen3-vl-text-bucketed-mrope-one-function-v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def stage(name: str, start: float = time.time()) -> None:
    import psutil

    rss = psutil.Process().memory_info().rss
    print(f"[export] {name}: rss {rss / 2**30:.2f} GiB, {time.time() - start:.0f} s", file=sys.stderr, flush=True)


def plan_steps(case: dict, image_token: int, steps: int) -> list[dict]:
    """The self-check submissions: the whole image prompt, forced decode steps, then a text-only prompt and decode
    steps in a fresh cache. Each step: tokens, cache start, rope rows [3, q] and whether it is the image prompt."""
    ids, greedy, grid = case["promptTokens"], case["greedyTokens"], case["grid"]
    forced = case["forcedTokens"][:steps]
    rope = np.array(mrope_positions(ids + forced, [tuple(grid)], image_token), dtype=np.int32).T
    out = [{"plan": "image", "tokens": ids, "start": 0, "rope": rope[:, :len(ids)], "image": True}]
    out += [{"plan": "image", "tokens": [t], "start": len(ids) + i, "rope": rope[:, len(ids) + i:len(ids) + i + 1],
             "image": False} for i, t in enumerate(forced)]
    text = [t for t in ids if t != image_token] + greedy[:-steps]
    tail = greedy[-steps:]
    positions = np.arange(len(text) + steps, dtype=np.int32)
    flat = np.stack([positions] * 3)
    out.append({"plan": "text", "tokens": text, "start": 0, "rope": flat[:, :len(text)], "image": False})
    out += [{"plan": "text", "tokens": [t], "start": len(text) + i, "rope": flat[:, len(text) + i:len(text) + i + 1],
             "image": False} for i, t in enumerate(tail)]
    return out


def capacity_for(plan: list[dict], name: str) -> int:
    end = max(s["start"] + len(s["tokens"]) for s in plan if s["plan"] == name)
    return ((end + 255) // 256) * 256


def eager_reference(text, case: dict, embeds: torch.Tensor, deepstack: torch.Tensor, plan: list[dict], out: Path) -> dict:
    """Runs the plan through the eager fp16 tower (aligned rows, host-written caches) and saves the step inputs and
    the eager outputs to `out` (one .npz per step)."""
    out.mkdir()
    hidden, levels = embeds.shape[-1], deepstack.shape[0]
    caches, records = {}, []
    with torch.no_grad():
        for i, s in enumerate(plan):
            if s["start"] == 0:
                caches[s["plan"]] = text.empty_state(capacity_for(plan, s["plan"]))
            kc, vc = caches[s["plan"]]
            q = len(s["tokens"])
            ids = torch.tensor([s["tokens"]], dtype=torch.int32)
            rows = torch.zeros(q, hidden, dtype=torch.float16)
            deep = torch.zeros(levels, q, hidden, dtype=torch.float16)
            if s["image"]:
                pads = (ids[0] == text.image_token).nonzero()[:, 0]
                if len(pads) != embeds.shape[0]:
                    raise SystemExit("the reference case's pad tokens do not match its image rows")
                rows[pads], deep[:, pads] = embeds, deepstack
            positions = torch.arange(s["start"], s["start"] + q, dtype=torch.int32)[None]
            rope = torch.from_numpy(np.ascontiguousarray(s["rope"]))
            logits, ku, vu = text(ids, positions, rope, kc, vc, rows, deep, aligned=True)
            kc[:, :, :, s["start"]:s["start"] + q] = ku
            vc[:, :, :, s["start"]:s["start"] + q] = vu
            np.savez(out / f"step-{i}.npz", input_ids=ids.numpy(), cache_positions=positions.numpy(), rope_positions=rope.numpy(),
                     image_embeds=rows.numpy(), deepstack=deep.numpy(), logits=logits[0, -1].float().numpy(),
                     keyUpdate=ku.numpy(), valueUpdate=vu.numpy())
            records.append({"step": i, "plan": s["plan"], "start": s["start"], "tokens": q, "image": s["image"],
                            "capacity": int(kc.shape[3])})
    plan_record = {"case": case["name"], "imageTokens": case["imageTokens"], "cacheShape": list(text.empty_state(1)[0].shape),
                   "steps": records}
    (out / "plan.json").write_text(json.dumps(plan_record, indent=1) + "\n")
    return plan_record


def self_check(output: Path) -> None:
    """The exec'd second phase: the asset (cpu_only) against the saved eager reference."""
    import coreai.runtime as rt

    reference = output / "self-check"
    plan = json.loads((reference / "plan.json").read_text())
    report_path = output / "export-report.json"
    report = json.loads(report_path.read_text())

    async def run():
        model = await rt.AIModel.load(output / "model.aimodel", rt.SpecializationOptions.cpu_only())
        function = model.load_function("main")
        results, caches = [], {}
        for s in plan["steps"]:
            saved = np.load(reference / f"step-{s['step']}.npz")
            shape = list(plan["cacheShape"])
            shape[3] = s["capacity"]
            if s["start"] == 0:
                caches[s["plan"]] = (np.zeros(shape, np.float16), np.zeros(shape, np.float16))
            kc, vc = caches[s["plan"]]
            feed = {name: rt.NDArray(saved[name]) for name in INPUTS[:5]}
            feed["keyCache"], feed["valueCache"] = rt.NDArray(kc), rt.NDArray(vc)
            r = await function(feed, {})
            ku, vu = r["keyUpdate"].numpy(), r["valueUpdate"].numpy()
            start, q = s["start"], s["tokens"]
            kc[:, :, :, start:start + q] = ku
            vc[:, :, :, start:start + q] = vu
            got, want = r["logits"].numpy()[0, -1].astype(np.float32), saved["logits"]
            lost = sum(int(((np.abs(g).max(-1) == 0) & (np.abs(e).max(-1) > 0)).sum())
                       for g, e in ((ku, saved["keyUpdate"]), (vu, saved["valueUpdate"])))
            results.append({"step": s["step"], "plan": s["plan"], "tokens": q, "image": s["image"],
                            "cosine": round(cosine(want, got), 6), "argmaxEqual": int(np.argmax(want)) == int(np.argmax(got)),
                            "updateCosine": round(min(cosine(saved["keyUpdate"], ku), cosine(saved["valueUpdate"], vu)), 6),
                            "lostCacheRows": lost})
        return results

    steps = asyncio.run(run())
    passed = all(s["cosine"] >= 0.99 and s["argmaxEqual"] and s["lostCacheRows"] == 0 for s in steps)
    report["selfCheck"] = {"passed": passed, "case": plan["case"], "imageTokens": plan["imageTokens"],
                           "referenceSHA256": {p.name: sha256(p) for p in sorted(reference.iterdir())},
                           "bar": "cosine >= 0.99 (engine fp16 bar), argmax equal, no lost cache rows", "steps": steps}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["selfCheck"], indent=1))
    if not passed:
        raise SystemExit("self-check failed; the eager reference stays in self-check/")
    shutil.rmtree(reference)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-repository")
    parser.add_argument("--source-revision")
    parser.add_argument("--self-check-reference", type=Path)
    parser.add_argument("--max-prefill-chunk", type=int, default=512)
    parser.add_argument("--max-image-chunk", type=int, default=2048, help="largest image submission (tokens)")
    parser.add_argument("--max-capacity", type=int, default=16384)
    parser.add_argument("--self-check-steps", type=int, default=4)
    parser.add_argument("--self-check", type=Path, help="internal: the second phase, run on an export directory")
    args = parser.parse_args()
    if args.self_check:
        self_check(args.self_check)
        return
    for name in ("checkpoint", "output", "source_repository", "source_revision", "self_check_reference"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    import coreai_torch as ct

    from qwen3_vl_vision import Qwen3VLVision, host_inputs, load_weights as load_vision

    config = json.loads((args.checkpoint / "config.json").read_text())
    cfg = dict(config["text_config"])
    cfg["tie_word_embeddings"] = bool(config.get("tie_word_embeddings", cfg.get("tie_word_embeddings")))
    levels = len(config["vision_config"]["deepstack_visual_indexes"])
    image_token = int(config["image_token_id"])
    args.output.mkdir(parents=True)
    text = build_fp16(cfg, image_token, levels, max_capacity=args.max_capacity)
    load_text(text, args.checkpoint)
    text.eval()
    stage("fp16 tower loaded")

    # Phase 1: the eager reference, while only the tower and (briefly) the vision tower are resident.
    record = json.loads((args.self_check_reference / "reference.json").read_text())
    case = record["cases"][0]
    pixels = np.fromfile(args.self_check_reference / case["pixels"]["file"], dtype="<f4").reshape(case["pixels"]["shape"])
    vision = Qwen3VLVision(dict(config["vision_config"]))
    load_vision(vision, args.checkpoint)
    vision = vision.float().eval()
    tables = {k: torch.from_numpy(v) for k, v in host_inputs(*case["grid"]).items()}
    with torch.no_grad():
        embeds, deepstack = vision(torch.from_numpy(pixels), tables["pos_index"], tables["pos_weight"], tables["patch_rc"])
    embeds, deepstack = embeds.half(), deepstack.half()
    del vision, tables
    gc.collect()
    plan = plan_steps(case, image_token, args.self_check_steps)
    eager_reference(text, case, embeds, deepstack, plan, args.output / "self-check")
    stage("eager reference saved")

    # Phase 2: export, convert and save one function.
    example, dynamic = text.unified_example(max(args.max_prefill_chunk, args.max_image_chunk))
    with torch.no_grad():
        ep = torch.export.export(Unified(text), example, dynamic_shapes=dynamic).run_decompositions(ct.get_decomp_table())
    drop_identity_copies(ep)
    if ep.graph_signature.user_inputs_to_mutate:
        raise SystemExit("the bucketed graph must not mutate its inputs")
    stage("exported")
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    converter.add_exported_program(ep, input_names=INPUTS, output_names=OUTPUTS, state_names=[], entrypoint_name="main")
    program = converter.to_coreai()
    program.optimize()
    stage("converted")
    asset = args.output / "model.aimodel"
    program.save_asset(asset)
    stage("saved")

    tok = args.output / "tokenizer"
    tok.mkdir()
    # The chat template travels with the tokenizer whichever file holds it (fine-tunes often ship chat_template.jinja
    # instead of .json), as in the dense exporter: AetherLanguageTokenizers renders with the model's own template.
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.json",
                 "chat_template.jinja", "added_tokens.json", "special_tokens_map.json",
                 "preprocessor_config.json", "generation_config.json", "config.json", "LICENSE"):
        if (args.checkpoint / name).exists():
            shutil.copyfile(args.checkpoint / name, tok / name)
    report = {
        "schema": 1,
        "recipe": RECIPE,
        "contract": "bucketed",
        "exportedAt": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hostOS": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "coreai-torch", "coreai-core")},
        "scriptSHA256": {p.name: sha256(p) for p in sorted(Path(__file__).resolve().parent.glob("*.py"))},
        "source": {"repository": args.source_repository, "revision": args.source_revision, "license": "apache-2.0",
                   "checkpoint": str(args.checkpoint)},
        "dtype": "float16",
        "functions": {"main": INPUTS, "outputs": OUTPUTS},
        "imageRows": "aligned: image_embeds [q, hidden] and deepstack [levels, q, hidden], used where the token is the image token",
        "exportBounds": {"maximumPrefillChunk": args.max_prefill_chunk, "maximumCacheCapacity": args.max_capacity,
                         "maximumImageChunk": args.max_image_chunk},
        "imageTokenID": image_token, "deepstackLevels": levels,
        "selfCheck": {"passed": False, "pending": "runs in the exec'd second phase"},
        "modelFiles": {str(p.relative_to(asset)): sha256(p) for p in sorted(asset.rglob("*")) if p.is_file()},
        "tokenizerFiles": {p.name: sha256(p) for p in sorted(tok.iterdir())},
    }
    (args.output / "export-report.json").write_text(json.dumps(report, indent=2) + "\n")
    # Phase 3: the self-check in a fresh process image. Freed memory is not returned inside this process, so a
    # self-check here would add its copies to the eager module, the converter's copy and the save buffer.
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, [sys.executable, "-I", str(Path(__file__).resolve()), "--self-check", str(args.output)])


if __name__ == "__main__":
    main()

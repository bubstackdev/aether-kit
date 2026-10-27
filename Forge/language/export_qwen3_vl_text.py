#!/usr/bin/env python3
"""Export the Qwen3-VL text tower with the bucketed step contract (development tooling only).

    Forge/.venv/bin/python -I Forge/language/export_qwen3_vl_text.py --checkpoint <hf snapshot> --output <new dir> \
        --source-repository Qwen/Qwen3-VL-2B-Instruct --source-revision <sha> --self-check-reference <reference dir>

One asset (`model.aimodel`) with two functions over the same fp16 weights: `main` (text steps) and `withImage`
(prefill segments containing image tokens). `export-report.json` records versions, hashes, the export bounds and a
cpu_only self-check against the eager tower: a `withImage` prefill of a reference case's prompt (image rows from the
eager float32 vision tower on the reference pixels) and forced `main` decode steps, comparing logits and every
returned cache row.
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
from qwen3_vl_text import Qwen3VLText, WithImage, load_weights as load_text  # noqa: E402
from qwen3_vl_vision import Qwen3VLVision, host_inputs, load_weights as load_vision  # noqa: E402

INPUTS_MAIN = ["input_ids", "cache_positions", "rope_positions", "keyCache", "valueCache"]
INPUTS_IMAGE = ["input_ids", "cache_positions", "rope_positions", "image_embeds", "deepstack", "keyCache", "valueCache"]
OUTPUTS = ["logits", "keyUpdate", "valueUpdate"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def self_check(asset: Path, text: Qwen3VLText, vision: Qwen3VLVision, reference: Path, steps: int = 4) -> dict:
    import coreai.runtime as rt

    record = json.loads((reference / "reference.json").read_text())
    case = record["cases"][0]
    ids, forced, grid = case["promptTokens"], case["forcedTokens"][:steps], case["grid"]
    pixels = np.fromfile(reference / case["pixels"]["file"], dtype="<f4").reshape(case["pixels"]["shape"])
    tables = {k: torch.from_numpy(v) for k, v in host_inputs(*grid).items()}
    with torch.no_grad():
        embeds, deepstack = vision(torch.from_numpy(pixels), tables["pos_index"], tables["pos_weight"], tables["patch_rc"])
    embeds, deepstack = embeds.half(), deepstack.half()
    from qwen3_vl_text import mrope_positions

    rope = np.array(mrope_positions(ids + forced, [tuple(grid)], text.image_token), dtype=np.int32).T
    capacity = ((len(ids) + steps + 255) // 256) * 256
    plan = [(ids, 0, True)] + [([t], len(ids) + i, False) for i, t in enumerate(forced)]

    # Eager reference (fp16 module, host-written caches).
    ek, ev = text.empty_state(capacity)
    eager_rows, eager_caches = [], []
    with torch.no_grad():
        for toks, start, image in plan:
            q = len(toks)
            args = (torch.tensor([toks], dtype=torch.int32), torch.arange(start, start + q, dtype=torch.int32)[None],
                    torch.from_numpy(np.ascontiguousarray(rope[:, start:start + q])), ek, ev)
            eager_rows.append(text.step(*args, *((embeds, deepstack) if image else ())).float().numpy())
            eager_caches.append((ek.float().numpy().copy(), ev.float().numpy().copy()))

    async def run():
        model = await rt.AIModel.load(asset, rt.SpecializationOptions.cpu_only())
        main_fn, image_fn = model.load_function("main"), model.load_function("withImage")
        shape = tuple(text.empty_state(capacity)[0].shape)
        kc, vc = np.zeros(shape, np.float16), np.zeros(shape, np.float16)
        rows, caches = [], []
        for toks, start, image in plan:
            q = len(toks)
            feed = {"input_ids": rt.NDArray(np.array([toks], np.int32)),
                    "cache_positions": rt.NDArray(np.arange(start, start + q, dtype=np.int32)[None]),
                    "rope_positions": rt.NDArray(np.ascontiguousarray(rope[:, start:start + q])),
                    "keyCache": rt.NDArray(kc), "valueCache": rt.NDArray(vc)}
            if image:
                feed["image_embeds"] = rt.NDArray(embeds.numpy())
                feed["deepstack"] = rt.NDArray(deepstack.numpy())
            r = await (image_fn if image else main_fn)(feed, {})
            kc[:, :, :, start:start + q] = r["keyUpdate"].numpy()
            vc[:, :, :, start:start + q] = r["valueUpdate"].numpy()
            rows.append(r["logits"].numpy()[0, -1].astype(np.float32))
            caches.append((kc.copy(), vc.copy()))
        return rows, caches

    got_rows, got_caches = asyncio.run(run())
    out = []
    for i, ((toks, start, image), a, b, (ek_, ev_), (gk, gv)) in enumerate(zip(plan, eager_rows, got_rows, eager_caches, got_caches)):
        total = start + len(toks)
        lost = sum(int(((np.abs(g[:, :, :, :total]).max(-1) == 0) & (np.abs(e[:, :, :, :total]).max(-1) > 0)).sum())
                   for e, g in ((ek_, gk), (ev_, gv)))
        out.append({"step": i, "function": "withImage" if image else "main", "tokens": len(toks),
                    "cosine": round(cosine(a, b), 6), "argmaxEqual": int(np.argmax(a)) == int(np.argmax(b)),
                    "lostCacheRows": lost})
    passed = all(s["cosine"] >= 0.99 and s["argmaxEqual"] and s["lostCacheRows"] == 0 for s in out)
    return {"passed": passed, "case": case["name"], "promptTokens": len(ids), "imageTokens": case["imageTokens"],
            "bar": "cosine >= 0.99 (engine fp16 bar), argmax equal, no lost cache rows", "steps": out}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--self-check-reference", type=Path, required=True)
    parser.add_argument("--max-prefill-chunk", type=int, default=512)
    parser.add_argument("--max-image-chunk", type=int, default=2048, help="largest withImage submission (tokens)")
    parser.add_argument("--max-capacity", type=int, default=16384)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    import coreai_torch as ct

    config = json.loads((args.checkpoint / "config.json").read_text())
    cfg = dict(config["text_config"])
    cfg["tie_word_embeddings"] = bool(config.get("tie_word_embeddings", cfg.get("tie_word_embeddings")))
    levels = len(config["vision_config"]["deepstack_visual_indexes"])
    text = Qwen3VLText(cfg, int(config["image_token_id"]), levels, max_capacity=args.max_capacity)
    load_text(text, args.checkpoint)
    text = text.half().eval()
    main_ex, main_dyn, image_ex, _ = text.examples(args.max_prefill_chunk)
    # withImage: an image's tokens stay in one submission, so its q bound is the image chunk bound.
    q_image = torch.export.Dim("q", min=2, max=args.max_image_chunk)
    n_image = torch.export.Dim("n", min=1, max=args.max_image_chunk)
    capacity = main_dyn[3][3]
    image_dyn = ({1: q_image}, {1: q_image}, {1: q_image}, {0: n_image}, {1: n_image}, {3: capacity}, {3: capacity})
    with torch.no_grad():
        ep_main = torch.export.export(text, main_ex, dynamic_shapes=main_dyn).run_decompositions(ct.get_decomp_table())
        ep_image = torch.export.export(WithImage(text), image_ex, dynamic_shapes=image_dyn).run_decompositions(ct.get_decomp_table())
    for ep in (ep_main, ep_image):
        drop_identity_copies(ep)
        if ep.graph_signature.user_inputs_to_mutate:
            raise SystemExit("the bucketed graph must not mutate its inputs")
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    converter.add_exported_program(ep_main, input_names=INPUTS_MAIN, output_names=OUTPUTS, state_names=[], entrypoint_name="main")
    converter.add_exported_program(ep_image, input_names=INPUTS_IMAGE, output_names=OUTPUTS, state_names=[],
                                   entrypoint_name="withImage")
    program = converter.to_coreai()
    program.optimize()
    args.output.mkdir(parents=True)
    asset = args.output / "model.aimodel"
    program.save_asset(asset)

    vision = Qwen3VLVision(dict(config["vision_config"]))
    load_vision(vision, args.checkpoint)
    vision = vision.float().eval()
    check = self_check(asset, text, vision, args.self_check_reference)
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
        "recipe": "qwen3-vl-text-bucketed-mrope-v1",
        "contract": "bucketed",
        "exportedAt": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hostOS": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "coreai-torch", "coreai-core")},
        "scriptSHA256": {p.name: sha256(p) for p in sorted(Path(__file__).resolve().parent.glob("*.py"))},
        "source": {"repository": args.source_repository, "revision": args.source_revision, "license": "apache-2.0",
                   "checkpoint": str(args.checkpoint)},
        "dtype": "float16",
        "functions": {"main": INPUTS_MAIN, "withImage": INPUTS_IMAGE, "outputs": OUTPUTS},
        "exportBounds": {"maximumPrefillChunk": args.max_prefill_chunk, "maximumCacheCapacity": args.max_capacity,
                         "maximumImageChunk": args.max_image_chunk},
        "imageTokenID": int(config["image_token_id"]), "deepstackLevels": levels,
        "selfCheck": check,
        "modelFiles": {str(p.relative_to(asset)): sha256(p) for p in sorted(asset.rglob("*")) if p.is_file()},
        "tokenizerFiles": {p.name: sha256(p) for p in sorted(tok.iterdir())},
    }
    (args.output / "export-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(check, indent=1))
    if not check["passed"]:
        raise SystemExit("self-check failed")


if __name__ == "__main__":
    main()

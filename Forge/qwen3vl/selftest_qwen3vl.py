#!/usr/bin/env python3
"""Eager CPU self-test of the export graphs and the host contract, before any Core AI conversion.

    Forge/.venv/bin/python -I Forge/qwen3vl/selftest_qwen3vl.py --reference <reference dir> [--report <json>] \
        [--decoder-case <case>] [--precision]

For every image case of reference_qwen3vl.py:
- the host contract the Swift driver implements reproduces upstream exactly: `smart_resize` with the bundle's budget
  gives the processor's grid; the graph's M-RoPE positions, synthesized from the extension ids, the running positions,
  `image_rc` and the rope shift (`mrope_positions`), equal upstream's `get_rope_index` for every prompt token;
- `VisionTower` in float32 with float32 weights (the padded static shape, the in-graph positional tables and key mask)
  matches upstream's tower on the zero-padded patches: cosine >= 0.9999999 for the image embeddings and each DeepStack
  level, padding rows exactly zero;
- with `--precision`, the shipped tower recipe (float16 weights, float32 compute) and the float16-compute alternative
  are measured (cosine, rel L2, finiteness; informational).
For `--decoder-case` (default the first image case): `Decoder` in float32 with float32 weights, fed upstream's tower
outputs, gives upstream's first-token logits (rel L2 <= 1e-4, same argmax) in one prefill, and the same logits again when
the prompt is fed in two chunks and the KV state carries over. Also: `smart_resize` against transformers' own over a
sweep of sizes, and the sample manifest's roles. Exits nonzero when a check fails. CPU only; a few minutes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_qwen3vl  # noqa: E402
import qwen3vl_graphs as qg  # noqa: E402

TOWER_BAR = 0.9999999
LOGITS_BAR = 1e-4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--snapshot", type=Path, default=qg.DEFAULT_SNAPSHOT)
    ap.add_argument("--decoder-case")
    ap.add_argument("--precision", action="store_true", help="also measure the tower's precision alternatives")
    args = ap.parse_args()
    qg.verify_sources(args.snapshot)
    torch.set_grad_enabled(False)
    checks: list[dict] = []

    def record(name: str, value, bar, passed: bool, gated: bool = True) -> None:
        checks.append({"check": name, "value": value, "bar": bar, "passed": bool(passed), "gated": gated})
        tag = ("PASS" if passed else "FAIL") if gated else "INFO"
        print(f"[selftest] {tag} {name}: {value} (bar {bar})", file=sys.stderr, flush=True)

    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize as upstream_resize
    worst = 0
    for h in range(40, 4100, 37):
        for w in (h // 3 + 32, h, 2 * h + 17, 517, 1900):
            want = upstream_resize(h, w, factor=32, min_pixels=65536, max_pixels=qg.PATCH_CAPACITY * 256)
            worst += qg.smart_resize(h, w) != tuple(want)
    record("smartResize.sweep.mismatches", worst, 0, worst == 0)

    index = json.loads((args.reference / "references.json").read_text())
    vocab = index["vocab"]
    slots = qg.PATCH_CAPACITY // 4
    model = qg.upstream_model(args.snapshot)
    tower = qg.VisionTower(model.model.visual, dtype=torch.float32, storage=torch.float32).eval()
    variants = {}
    if args.precision:
        variants = {"fp16weights": qg.VisionTower(model.model.visual, dtype=torch.float32,
                                                  storage=torch.float16).eval(),
                    "fp16compute": qg.VisionTower(model.model.visual, dtype=torch.float16,
                                                  storage=torch.float16).eval()}
    image_cases = []
    for entry in index["cases"]:
        folder = args.reference / entry["name"]
        case = json.loads((folder / "case.json").read_text())
        if not case.get("image"):
            continue
        name = case["name"]
        image_cases.append(case)
        _, h, w = case["gridTHW"]
        width, height = case["imageSize"]
        sized = qg.smart_resize(height, width)
        same = sized == (h * 16, w * 16)
        record(f"{name}.smartResize", f"{sized} vs {(h * 16, w * 16)}", "equal", same)

        ids = qg.read_i32(folder / "input_ids.i32", [1, case["promptTokens"]])[0]
        ext = qg.extension_ids(ids, vocab)
        p = torch.arange(len(ids), dtype=torch.int32)
        pos = qg.mrope_positions(torch.from_numpy(ext), p, torch.from_numpy(qg.image_rc(h, w)),
                                 torch.tensor([case["ropeShiftStart"]], dtype=torch.int32),
                                 torch.tensor([case["ropeShiftAmount"]], dtype=torch.int32), vocab, slots)
        want = qg.read_i32(folder / "rope_positions.i32", [3, len(ids)])
        exact = np.array_equal(torch.stack(pos).numpy(), want)
        record(f"{name}.mropePositions.exact", float(exact), 1.0, exact)

        n = h * w // 4
        pv = qg.read_f32(folder / "pixel_values.f32", [h * w, 1536])
        want_img = qg.read_f32(folder / "image_embeds.f32", [n, 2048])
        want_deep = qg.read_f32(folder / "deepstack_embeds.f32", [3, n, 2048])
        feeds = (torch.from_numpy(qg.pad_patches(pv)), torch.tensor([h, w], dtype=torch.int32))
        img, deep = (t.numpy() for t in tower(*feeds))
        c = qg.cosine(img[:n], want_img)
        record(f"{name}.tower.float32.imageEmbeds.cosine", c, TOWER_BAR, c >= TOWER_BAR)
        for k in range(3):
            ck = qg.cosine(deep[k * slots:k * slots + n], want_deep[k])
            record(f"{name}.tower.float32.deepstack{k}.cosine", ck, TOWER_BAR, ck >= TOWER_BAR)
        zero = not img[n:].any() and not any(deep[k * slots + n:(k + 1) * slots].any() for k in range(3))
        record(f"{name}.tower.paddingZero", float(zero), 1.0, zero)
        for label, variant in variants.items():
            vi, vd = (t.numpy() for t in variant(*feeds))
            finite = bool(np.isfinite(vi).all() and np.isfinite(vd).all())
            record(f"{name}.tower.{label}.finite", float(finite), None, True, gated=False)
            if finite:
                record(f"{name}.tower.{label}.imageEmbeds.oneMinusCosine", 1 - qg.cosine(vi[:n], want_img), None,
                       True, gated=False)
                record(f"{name}.tower.{label}.imageEmbeds.relL2", qg.rel_l2(vi[:n], want_img), None, True,
                       gated=False)
                worst_deep = min(qg.cosine(vd[k * slots:k * slots + n], want_deep[k]) for k in range(3))
                record(f"{name}.tower.{label}.deepstack.worstOneMinusCosine", 1 - worst_deep, None, True,
                       gated=False)
    del tower, variants

    case = next((c for c in image_cases if c["name"] == args.decoder_case), image_cases[0])
    folder = args.reference / case["name"]
    _, h, w = case["gridTHW"]
    n = h * w // 4
    decoder = qg.Decoder(model, dtype=torch.float32, attention=torch.float32, storage=torch.float32).eval()
    ids = qg.read_i32(folder / "input_ids.i32", [1, case["promptTokens"]])[0]
    ext = torch.from_numpy(qg.extension_ids(ids, vocab))[None]
    image = torch.zeros(slots, 2048)
    image[:n] = torch.from_numpy(qg.read_f32(folder / "image_embeds.f32", [n, 2048]))
    deep = torch.zeros(3 * slots, 2048)
    ref_deep = qg.read_f32(folder / "deepstack_embeds.f32", [3, n, 2048])
    for k in range(3):
        deep[k * slots:k * slots + n] = torch.from_numpy(ref_deep[k])
    statics = (image, deep, torch.from_numpy(qg.image_rc(h, w)), torch.tensor([case["ropeShiftStart"]],
               dtype=torch.int32), torch.tensor([case["ropeShiftAmount"]], dtype=torch.int32))
    want = qg.read_f32(folder / "logits.f32", [index["steps"], vocab])[0]
    length = ext.shape[1]
    kc, vc = decoder.empty_state(length + 4)
    got = decoder(ext, torch.arange(length, dtype=torch.int32)[None], *statics, kc, vc)[0, 0].numpy()
    r = qg.rel_l2(got, want)
    record(f"{case['name']}.decoder.float32.firstLogits.relL2", r, LOGITS_BAR,
           r <= LOGITS_BAR and int(got.argmax()) == int(want.argmax()))
    kc, vc = decoder.empty_state(length + 4)
    split = length // 2
    decoder(ext[:, :split], torch.arange(split, dtype=torch.int32)[None], *statics, kc, vc)
    chunked = decoder(ext[:, split:], torch.arange(length, dtype=torch.int32)[None], *statics, kc, vc)[0, 0].numpy()
    rc = qg.rel_l2(chunked, want)
    record(f"{case['name']}.decoder.float32.chunkedPrefill.relL2", rc, LOGITS_BAR, rc <= LOGITS_BAR)

    sample = export_qwen3vl.sample_manifest("macos")
    g = sample["graphs"]
    dec_inputs = {k: (v["name"] if isinstance(v, dict) else v) for k, v in g["decoder"]["inputs"].items()}
    roles = (sample["driver"] == "vlm-static-inject@1"
             and g["vision"]["inputs"] == {"patches": "patches", "grid": "grid"}
             and g["vision"]["outputs"] == {"imageEmbeds": "image_embeds", "deepstackEmbeds": "deepstack_embeds"}
             and list(dec_inputs.values()) == export_qwen3vl.DECODER_INPUTS
             and all(g["decoder"]["inputs"][r]["binding"] == "static" for r in
                     ("imageEmbeds", "deepstackEmbeds", "imageRC", "ropeShiftStart", "ropeShiftAmount"))
             and g["decoder"]["outputs"]["logits"] == {"name": "logits", "rows": "last"}
             and g["decoder"]["states"]["sequence"][0] == {"tensors": ["keyCache", "valueCache"], "kind": "kv",
                                                           "seqAxis": 3}
             and sample["image"]["tokens"]["idBase"] == vocab
             and g["decoder"]["inputEncodings"]["tokens"]["extensionIds"] == {"base": vocab, "count": slots})
    record("manifest.roles", float(roles), 1.0, roles)
    passed = all(c["passed"] for c in checks if c["gated"])
    if args.report:
        args.report.write_text(json.dumps({"passed": passed, "checks": checks}, indent=2) + "\n")
    if not passed:
        raise SystemExit("self-test failed")


if __name__ == "__main__":
    main()

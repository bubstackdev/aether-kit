#!/usr/bin/env python3
"""Core AI self-test of the export graphs on the tiny random Qwen3.5 of test_qwen3_5_graphs.py (no checkpoint).

    scripts/gpu-lease -- Forge/.venv/bin/python -I Forge/qwen3_5/selftest_qwen3_5.py [--compute float32|float16]

Converts the tower and the decoder's two functions (`main` S=1 and `prefill` S=PREFILL over the same states, the
running position and KV capacity axes dynamic) and runs them on the GPU through the Python runtime as the Swift driver
feeds them: each image through the tower into the static slots, the prompt in `prefill` chunks with the remainder
through `main`, then greedy tokens through `main`. Compared with the eager graphs (the hermetic test's float32 parity
with upstream carries over): every call's logits, and every state after the prompt. This is where Core AI's own
rules show (2026-11-10 spike: an integer-index write into the conv state was dropped; a one-row matmul wrote the value
cache in the wrong layout). A tiny model: each new length re-specializes, but at kilobytes a graph; the run still
stops itself past 2 GB of MPSGraph temp.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qwen3_5_graphs as qg  # noqa: E402
import test_qwen3_5_graphs as tt  # noqa: E402

STATES = ["keyCache", "valueCache", "convState", "recState"]
INPUTS = ["input_ids", "position_ids", "image_embeds", "image_positions", "rope_shift_start", "rope_shift_amount"]


def convert(functions: dict, path: Path, states: list[str], inputs: list[str], outputs: list[str]) -> None:
    import coreai_torch as ct
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    for name, (module, example, dynamic) in functions.items():
        with torch.no_grad():
            ep = torch.export.export(module, example, dynamic_shapes=dynamic, strict=False)
            ep = ep.run_decompositions(ct.get_decomp_table())
        if len(ep.graph_signature.user_inputs_to_mutate) != len(states):
            raise SystemExit(f"{name}: {len(ep.graph_signature.user_inputs_to_mutate)} mutated inputs for {states}")
        converter.add_exported_program(ep, input_names=inputs, output_names=outputs, state_names=states,
                                       entrypoint_name=name)
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute", choices=["float32", "float16"], default="float32")
    args = ap.parse_args()
    import coreai.runtime as rt
    torch.set_grad_enabled(False)
    dt = {"float32": torch.float32, "float16": torch.float16}[args.compute]
    bar = 1e-4 if dt == torch.float32 else 2e-2
    failures = []

    def check(name: str, ok: bool, detail="") -> None:
        print(f"[selftest] {'PASS' if ok else 'FAIL'} {name} {detail}", file=sys.stderr, flush=True)
        if not ok:
            failures.append(name)

    model = tt.tiny_model()
    tower = qg.VisionTower(model.model.visual, capacity=tt.CAPACITY, dtype=torch.float32, storage=dt,
                           output=dt).eval()
    decoder = qg.Decoder(model, slots=tt.SLOTS, images=tt.IMAGES, dtype=dt, attention=torch.float32,
                         storage=dt).eval()
    eager = qg.Decoder(model, slots=tt.SLOTS, images=tt.IMAGES, dtype=dt, attention=torch.float32,
                       storage=dt).eval()
    work = Path(tempfile.mkdtemp(prefix="qwen3_5-selftest-"))
    temp = Path(subprocess.run(["getconf", "DARWIN_USER_TEMP_DIR"], capture_output=True, text=True).stdout.strip())
    mps = temp / "com.apple.MetalPerformanceShadersGraph"

    def guard():
        mine = sum(f.stat().st_size for d in mps.glob(f"mpsgraph-{os.getpid()}-*") for f in d.rglob("*") if f.is_file())
        if mine > 2 * 2**30:
            raise SystemExit(f"guard: {mine / 2**30:.1f} GB of MPSGraph temp")

    try:
        convert({"main": (tower, tower.example((8, 8)), None)}, work / "vision.aimodel", [], ["patches", "grid"],
                ["image_embeds"])
        convert({"main": (decoder, *decoder.example(1)), "prefill": (decoder, *decoder.example(qg.PREFILL))},
                work / "decoder.aimodel", STATES, INPUTS, ["logits"])
        gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
        vision = asyncio.run(rt.AIModel.load(work / "vision.aimodel", gpu)).load_function("main")
        graph = asyncio.run(rt.AIModel.load(work / "decoder.aimodel", gpu))
        fns = {"main": graph.load_function("main"), "prefill": graph.load_function("prefill")}

        def call(fn, feeds, state=None):
            guard()

            async def go():
                o = await (fn({k: rt.NDArray(np.ascontiguousarray(v)) for k, v in feeds.items()}, state)
                           if state is not None else fn({k: rt.NDArray(np.ascontiguousarray(v)) for k, v in feeds.items()}))
                return {k: v.numpy().copy() for k, v in o.items()}
            return asyncio.run(go())

        rng = np.random.default_rng(0)
        grids = [(6, 6), (4, 16), (8, 8), (12, 8), (10, 10)]
        while True:
            ids, starts = tt.prompt(grids, rng)
            if any(st < qg.PREFILL < st + h * w // 4 for st, (h, w) in zip(starts, grids)):
                break
        npdt = np.float32 if dt == torch.float32 else np.float16
        image = np.zeros((tt.SLOTS, tt.HIDDEN), dtype=npdt)
        base = 0
        for h, w in grids:
            pv = torch.randn(h * w, 1536)
            tokens = h * w // 4
            padded = qg.pad_patches(pv.numpy(), tt.CAPACITY)
            got = call(vision, {"patches": padded, "grid": np.array([h, w], dtype=np.int32)})["image_embeds"]
            want = tower(torch.from_numpy(padded), torch.tensor([h, w], dtype=torch.int32)).float().numpy()
            check(f"vision.{h}x{w}", qg.cosine(got[:tokens], want[:tokens]) > 0.9999 and not got[tokens:].any(),
                  qg.cosine(got[:tokens], want[:tokens]))
            image[base:base + tokens] = got[:tokens]
            base += tokens
        positions, start, amount = qg.plan_images(starts, grids, slots=tt.SLOTS, images=tt.IMAGES)
        ext = qg.extension_ids(ids, tt.VOCAB, tt.PAD)
        statics = {"image_embeds": image, "image_positions": positions, "rope_shift_start": start,
                   "rope_shift_amount": amount}
        length, capacity = len(ext), len(ext) + 16
        shapes = [tuple(t.shape) for t in eager.empty_state(capacity)]
        dtypes = [t.dtype for t in eager.empty_state(capacity)]
        state = {n: rt.NDArray(np.zeros(s, dtype=np.float32 if d == torch.float32 else np.float16))
                 for n, s, d in zip(STATES, shapes, dtypes)}
        mine = eager.empty_state(capacity)
        torch_statics = [torch.from_numpy(image), torch.from_numpy(positions), torch.from_numpy(start),
                         torch.from_numpy(amount)]

        def both(fn, lo, hi):
            feeds = {"input_ids": ext[lo:hi][None], "position_ids": np.arange(hi, dtype=np.int32)[None], **statics}
            got = call(fns[fn], feeds, state)["logits"][0, 0]
            want = eager(torch.from_numpy(ext[lo:hi])[None], torch.arange(hi, dtype=torch.int32)[None],
                         *torch_statics, *mine)[0, 0].numpy()
            return got, want

        i, worst = 0, 0.0
        while i + qg.PREFILL <= length:
            got, want = both("prefill", i, i + qg.PREFILL)
            worst = max(worst, qg.rel_l2(got, want))
            i += qg.PREFILL
        while i < length:
            got, want = both("main", i, i + 1)
            worst = max(worst, qg.rel_l2(got, want))
            i += 1
        check("decoder.prompt.logits", worst <= bar, f"worst rel L2 {worst:.3g}")
        for name, t in zip(STATES, mine):
            g_, w_ = state[name].numpy().astype(np.float32), t.float().numpy()
            rows = slice(None) if name in ("convState", "recState") else (slice(None),) * 3 + (slice(0, length),)
            check(f"state.{name}", qg.rel_l2(g_[rows], w_[rows]) <= bar, qg.rel_l2(g_[rows], w_[rows]))
        for step in range(6):
            token = int(got.argmax())
            if token != int(want.argmax()):
                check(f"greedy.step{step}.argmax", False)
            ext = np.append(ext, np.int32(token))
            got, want = both("main", len(ext) - 1, len(ext))
            check(f"greedy.step{step}", qg.rel_l2(got, want) <= bar, qg.rel_l2(got, want))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if failures:
        raise SystemExit(f"{len(failures)} check(s) failed: {failures}")
    print("all checks passed", file=sys.stderr)


if __name__ == "__main__":
    main()

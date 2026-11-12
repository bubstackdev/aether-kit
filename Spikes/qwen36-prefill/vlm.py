#!/usr/bin/env python3
"""The VLM stage of the Qwen3.5/3.6 spike: the vision tower and the whole decoder (M-RoPE, final norm, untied head) as
Core AI bundles, run from the same inputs as ref_vlm.py's transformers reference and compared stage by stage.

    export-vision  --weights <extract --full dir> --output <dir>      Forge's Qwen3-VL VisionTower, no DeepStack
    export-decoder --weights <dir> --output <dir> [--weight-bits 16|4] main (S=1) + prefill (S=64) over the 4 states
    run --weights <dir> --vision <dir> --decoder <dir> --ref ref.npz   vision, prompt logits, teacher-forced steps, greedy

Host side here (the spike's, not the bundle contract yet): the embedding lookup and the image rows spliced in at the
`<|image_pad|>` tokens, and the 3-axis positions taken from transformers' get_rope_index. Every input has a fixed shape
(the Python runtime re-specializes on a new shape and MPSGraph writes temp files for each); `run` holds
/tmp/atelier-gpu.lock and stops itself past 20 GB of MPSGraph temp or a 30 GB disk drop.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "Forge" / "qwen3vl"))
import spike as sp  # noqa: E402
import qwen3vl_graphs as g  # noqa: E402

CAPACITY = 2048          # the decoder's KV rows (prompt + generated)
PATCHES = 2048           # the tower's rows (ref_vlm.PATCH_CAPACITY)


# MARK: vision

class Tower(g.VisionTower):
    """Forge's Qwen3-VL tower (same blocks, merger, bilinear position table and 2-D rotary as Qwen3.5's) without the
    DeepStack levels Qwen3.5 doesn't have: patches [P, 1536], grid int32 [2] -> image_embeds [P/4, H]."""

    def __init__(self, visual, **kw):
        visual.deepstack_visual_indexes, visual.deepstack_merger_list = [], []
        super().__init__(visual, **kw)

    def forward(self, patches, grid):
        dt = self.dtype
        row, col, real = self.geometry(grid)
        h_taps, h_w = self.axis(row, grid[0])
        w_taps, w_w = self.axis(col, grid[1])
        indices = (h_taps[:, :, None] * self.side + w_taps[:, None, :]).reshape(-1, 4)
        weights = (h_w[:, :, None] * w_w[:, None, :]).reshape(-1, 4)
        pos = (torch.nn.functional.embedding(indices, self.pos_table).float() * weights[:, :, None]).sum(1)
        x = self.patch(patches.to(dt)) + pos.to(dt)
        angles = torch.cat((row.float()[:, None] * self.inv_freq, col.float()[:, None] * self.inv_freq), dim=-1)
        emb = torch.cat((angles, angles), dim=-1)[:, None, :]
        cos, sin = emb.cos(), emb.sin()
        keep = real[None, None, None, :].expand(1, 1, self.capacity, self.capacity)
        for block in self.blocks:
            x = block(x, cos, sin, keep)
        tokens = (self.token_index < g.floordiv(grid[0] * grid[1], self.merge * self.merge))[:, None]
        return torch.where(tokens, self.merger(x).float(), 0.0).to(self.output)


def hf_visual(weights: Path):
    from safetensors.torch import load_file
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    meta = json.load(open(weights / "extract.json"))
    visual = Qwen3_5VisionModel(Qwen3_5VisionConfig(**meta["visionConfig"])).float().eval()
    raw = {k[len("model.visual."):]: v.float() for k, v in load_file(str(weights / "vision.safetensors")).items()}
    missing, unexpected = visual.load_state_dict(raw, strict=False)
    if [m for m in missing if "inv_freq" not in m] or unexpected:
        raise SystemExit(f"vision weights: missing {missing[:6]} unexpected {unexpected[:6]}")
    return visual


def export_vision(args):
    import coreai_torch as ct
    visual = hf_visual(args.weights)
    tower = Tower(visual, capacity=PATCHES, dtype=torch.float16, storage=torch.float16, output=torch.float16).eval()
    with torch.no_grad():
        ep = torch.export.export(tower, tower.example(), strict=False).run_decompositions(ct.get_decomp_table())
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    converter.add_exported_program(ep, input_names=["patches", "grid"], output_names=["image_embeds"],
                                   entrypoint_name="main")
    program = converter.to_coreai()
    program.optimize()
    args.output.mkdir(parents=True)
    program.save_asset(args.output / "model.aimodel")
    print(json.dumps({"vision": str(args.output), "capacity": PATCHES, "peakGB": sp.peak_gb()}))


def check_vision(args):
    """A real picture through the processor, then transformers' tower (float32, CPU) and the Core AI tower (GPU)."""
    import coreai.runtime as rt
    from PIL import Image
    from transformers import AutoProcessor
    meta = json.load(open(args.weights / "extract.json"))
    processor = AutoProcessor.from_pretrained(meta["snapshot"])
    report = {}
    visual = hf_visual(args.weights)
    sp.take_lock()
    try:
        fn = asyncio.run(rt.AIModel.load(args.vision / "model.aimodel", rt.SpecializationOptions.from_preferred_compute_unit_kind(
            rt.ComputeUnitKind.gpu()))).load_function("main")
        for picture in args.pictures:
            px = processor.image_processor(images=[Image.open(picture).convert("RGB")], return_tensors="pt",
                                           size={"shortest_edge": 65536, "longest_edge": PATCHES * 256})
            grid = px["image_grid_thw"]
            with torch.no_grad():
                want = visual(px["pixel_values"].float(), grid_thw=grid).pooler_output.float().numpy()
            h, w = int(grid[0, 1]), int(grid[0, 2])

            async def go():
                o = await fn({"patches": rt.NDArray(g.pad_patches(px["pixel_values"].float().numpy(), PATCHES)),
                              "grid": rt.NDArray(np.array([h, w], dtype=np.int32))})
                return o["image_embeds"].numpy().astype(np.float32)
            t = time.monotonic()
            got = asyncio.run(go())[:h * w // 4]
            report[picture.name] = {"grid": [h, w], "tokens": h * w // 4, "seconds": round(time.monotonic() - t, 2),
                                    "cosine": round(g.cosine(got, want), 6),
                                    "worstRowCosine": round(float(g.row_cosines(got, want).min()), 6),
                                    "relL2": round(g.rel_l2(got, want), 5)}
            print(picture.name, report[picture.name], flush=True)
    finally:
        sp.release_lock()
    json.dump(report, open(args.vision / "check-vision.json", "w"), indent=1)


# MARK: decoder

class VLMCore(sp.StaticCore):
    """The spike's static hybrid trunk with Qwen3.5's interleaved M-RoPE, the final norm and the untied head.
    Inputs as StaticCore's except `positions` int32 [3, S] (t, h, w); output `logits` [1, 1, V] float32."""

    def __init__(self, core, head: torch.Tensor, section: list[int]):
        super().__init__(core)
        rd = 2 * core.model.inv_freq.shape[0]
        self.head = nn.Linear(head.shape[1], head.shape[0], bias=False)
        self.head.weight = nn.Parameter(head)
        self.register_buffer("axes", g.mrope_masks(rd, section)[:, :rd // 2].contiguous())   # [3, rd/2]

    def forward(self, inputs_embeds, positions, write, mask, k_cache, v_cache, conv_state, rec_state):
        f = self.m.inv_freq
        freqs = (positions.float()[:, :, None] * f[None, None, :] * self.axes[:, None, :]).sum(0)   # [S, rd/2]
        emb = torch.cat([freqs, freqs], dim=-1)[None]
        h = self.trunk(inputs_embeds, emb.cos(), emb.sin(), write, mask, k_cache, v_cache, conv_state, rec_state)
        return self.head(self.m.norm(h[:, -1:, :])).float()


def build_decoder(weights: Path):
    from safetensors.torch import load_file
    meta = json.load(open(weights / "extract.json"))
    text = meta["textConfig"]
    cfg, core = sp.build(weights, text["num_hidden_layers"], CAPACITY)
    head = load_file(str(weights / "head.safetensors"))
    core.model.norm.weight = nn.Parameter(head["model.language_model.norm.weight"].to(torch.float16))
    section = text["rope_parameters"]["mrope_section"]
    return cfg, VLMCore(core, head["lm_head.weight"].to(torch.float16), section).eval()


def example(cfg, s: int):
    q = sp.zoo()
    st = q.build_decode_state(cfg, max_seq_len=CAPACITY, dtype=torch.float16)
    st["rec_state"] = st["rec_state"].to(torch.float32)
    st["conv_state"] = st["conv_state"].transpose(-1, -2).contiguous()
    _, write, mask = sp.host_inputs(8, s, CAPACITY)
    pos = torch.arange(8, 8 + s, dtype=torch.int32)[None].expand(3, s).contiguous()
    return (torch.zeros(1, s, cfg.hidden_size, dtype=torch.float16), pos, torch.from_numpy(write),
            torch.from_numpy(mask), st["k_cache"], st["v_cache"], st["conv_state"], st["rec_state"])


def export_decoder(args):
    import coreai_torch as ct
    from coreai_models.export.mlir_ops import register_custom_torch_lowering, remove_functionalization
    t0 = time.monotonic()
    cfg, model = build_decoder(args.weights)
    quantized = 0
    if args.weight_bits == 4:
        model, quantized = _quantize(model, cfg)
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    timings = {"build": round(time.monotonic() - t0, 1)}
    for name, s, mode in (("main", 1, "step"), ("prefill", 64, "chunk")):
        sp.set_scan(types.SimpleNamespace(model=model.m), mode)
        t = time.monotonic()
        with torch.no_grad():
            ep = torch.export.export(model, example(cfg, s), strict=False).run_decompositions(ct.get_decomp_table())
        remove_functionalization(ep)
        if len(ep.graph_signature.user_inputs_to_mutate) != 4:
            raise SystemExit(f"{name}: {len(ep.graph_signature.user_inputs_to_mutate)} mutated inputs")
        converter.add_exported_program(ep, input_names=sp.STATIC_INPUTS, output_names=["logits"],
                                       state_names=sp.STATES, entrypoint_name=name)
        timings[name] = round(time.monotonic() - t, 1)
        del ep
    register_custom_torch_lowering(converter)
    t = time.monotonic()
    program = converter.to_coreai()
    program.optimize()
    args.output.mkdir(parents=True)
    program.save_asset(args.output / "model.aimodel")
    timings["convert"] = round(time.monotonic() - t, 1)
    size = sum(p.stat().st_size for p in (args.output / "model.aimodel").rglob("*") if p.is_file())
    report = {"layers": cfg.num_hidden_layers, "weightBits": args.weight_bits, "quantizedLinears": quantized,
              "capacity": CAPACITY, "chunk": 64, "bytes": size, "timings": timings, "peakRSSGB": sp.peak_gb()}
    json.dump(report, open(args.output / "export.json", "w"), indent=1)
    print(json.dumps(report))


def _quantize(model, cfg):
    """spike.quantize's recipe (int4 block 32, every decoder Linear) with this module's example inputs."""
    import re
    from coreai_opt.quantization import Quantizer
    from coreai_opt.quantization.config import ModuleQuantizerConfig, QuantizerConfig
    from coreai_opt.quantization.spec.granularity import PerBlockGranularity
    from coreai_opt.quantization.spec.qscheme import QuantizationScheme
    from coreai_opt.quantization.spec.spec import QuantizationSpec
    spec = QuantizationSpec(dtype=torch.int4, qscheme=QuantizationScheme.SYMMETRIC,
                            granularity=PerBlockGranularity(axis=1, block_size=32))
    config = ModuleQuantizerConfig(op_input_spec=None, op_output_spec=None, op_state_spec={"weight": spec})
    selected = {f"^{re.escape(path)}$": config for path, child in model.named_modules()
                if isinstance(child, nn.Linear) and (path.startswith("m.layers.") or path == "head")}
    quantizer = Quantizer(model, QuantizerConfig(global_config=None, module_name_configs=selected,
                                                 execution_mode="eager"))
    with torch.no_grad():
        prepared = quantizer.prepare(tuple(t.clone() for t in example(cfg, 1)))
        return quantizer.finalize(prepared).eval(), len(selected)


# MARK: run

def log_softmax(x):
    x = x - x.max()
    return x - np.log(np.exp(x).sum())



def guarded():
    """A call counter that checks MPSGraph temp and the disk every 16 calls, and a watchdog thread every 10 s."""
    temp = Path(subprocess.run(["getconf", "DARWIN_USER_TEMP_DIR"], capture_output=True, text=True).stdout.strip())
    mps = temp / "com.apple.MetalPerformanceShadersGraph"
    free0 = shutil.disk_usage("/").free

    def guard():
        mine = sum(f.stat().st_size for d in mps.glob(f"mpsgraph-{os.getpid()}-*") for f in d.rglob("*") if f.is_file())
        drop = free0 - shutil.disk_usage("/").free
        if mine > 20 * 2**30 or drop > 30 * 2**30:
            raise SystemExit(f"guard: MPSGraph temp {mine / 2**30:.1f} GB, free disk down {drop / 2**30:.1f} GB")

    def watchdog():
        while True:
            time.sleep(10)
            try:
                guard()
            except SystemExit as e:
                print(e, flush=True)
                sp.release_lock()
                os._exit(3)
    threading.Thread(target=watchdog, daemon=True).start()
    return guard


def run(args):
    import coreai.runtime as rt
    ref = np.load(args.ref)
    meta = json.load(open(args.weights / "extract.json"))
    image_id = meta["tokens_ids"]["image_token_id"]
    ids, positions, delta = ref["input_ids"], ref["position_ids"], int(ref["rope_delta"].ravel()[0])
    gen_ids, gen_logits = ref["gen_ids"], ref["gen_logits"].astype(np.float32)
    T, N = len(ids), len(gen_ids)
    if T + N > CAPACITY:
        raise SystemExit(f"{T} + {N} tokens exceed the {CAPACITY} KV rows")
    embed = np.load(args.weights / "embed.npy", mmap_mode="r")
    report = {"promptTokens": T, "steps": N}
    guard = guarded()
    calls = [0]
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())

    def call(fn, feeds, state=None):
        calls[0] += 1
        if calls[0] % 16 == 1:
            guard()

        async def go():
            inputs = {k: rt.NDArray(np.ascontiguousarray(v)) for k, v in feeds.items()}
            o = await (fn(inputs, state) if state is not None else fn(inputs))
            return {k: v.numpy().copy() for k, v in o.items()}
        return asyncio.run(go())

    sp.take_lock()
    try:
        # 1. vision, one picture at a time
        t = time.monotonic()
        vision = asyncio.run(rt.AIModel.load(args.vision / "model.aimodel", gpu)).load_function("main")
        report["visionLoadSeconds"] = round(time.monotonic() - t, 1)
        rows, offset, times = [], 0, []
        for _, h, w in ref["image_grid_thw"]:
            n = int(h * w)
            t = time.monotonic()
            out = call(vision, {"patches": g.pad_patches(ref["pixel_values"][offset:offset + n], PATCHES),
                                "grid": np.array([h, w], dtype=np.int32)})["image_embeds"]
            times.append(time.monotonic() - t)
            rows.append(out[:n // 4].astype(np.float32))
            offset += n
        image = np.concatenate(rows)
        report["vision"] = {"seconds": [round(x, 2) for x in times], "cosine": g.cosine(image, ref["image_embeds"]),
                            "worstRowCosine": float(g.row_cosines(image, ref["image_embeds"]).min()),
                            "relL2": g.rel_l2(image, ref["image_embeds"])}
        print("vision", report["vision"], flush=True)
        # 2. the prompt: embeddings with the image rows spliced in
        x = embed[ids].astype(np.float16)
        where = np.nonzero(ids == image_id)[0]
        if len(where) != len(image):
            raise SystemExit(f"{len(where)} image pads for {len(image)} image rows")
        x[where] = image.astype(np.float16)
        t = time.monotonic()
        decoder = asyncio.run(rt.AIModel.load(args.decoder / "model.aimodel", gpu))
        fns = {n: decoder.load_function(n) for n in ("main", "prefill")}
        report["decoderLoadSeconds"] = round(time.monotonic() - t, 1)
        desc = fns["main"].desc
        shapes = {n: (list(desc.state_descriptor(n).shape), str(desc.state_descriptor(n).dtype)) for n in desc.state_names}

        def fresh():
            return {n: rt.NDArray(np.zeros(s, dtype=np.float32 if "32" in d else np.float16)) for n, (s, d) in shapes.items()}

        def feeds(rows_x, pos3, start):
            s = rows_x.shape[0]
            _, write, mask = sp.host_inputs(start, s, CAPACITY)
            return {"inputs_embeds": rows_x[None].astype(np.float16), "positions": pos3.astype(np.int32),
                    "write": write, "mask": mask}

        def prompt(state):
            i, last, chunk_times = 0, None, []
            while i + 64 <= T:
                t = time.monotonic()
                last = call(fns["prefill"], feeds(x[i:i + 64], positions[:, i:i + 64], i), state)["logits"][0, -1]
                chunk_times.append(time.monotonic() - t)
                i += 64
            while i < T:
                last = call(fns["main"], feeds(x[i:i + 1], positions[:, i:i + 1], i), state)["logits"][0, -1]
                i += 1
            return last, chunk_times

        def step(state, token, k):
            p = T + k + delta
            return call(fns["main"], feeds(embed[[token]].astype(np.float16), np.full((3, 1), p, np.int32), T + k),
                        state)["logits"][0, -1]

        # 3. prompt logits, then teacher-forced steps on the reference's tokens
        state = fresh()
        t = time.monotonic()
        last, chunk_times = prompt(state)
        report["promptSeconds"] = round(time.monotonic() - t, 1)
        def compare(lg, k):
            """Against the reference's step k (raw logits or log-probabilities: both compared as log-softmax)."""
            mine = log_softmax(lg.astype(np.float64))
            ref_lp = log_softmax(gen_logits[k].astype(np.float64))
            p = np.exp(ref_lp)
            top_mine, top_ref = set(np.argsort(mine)[-5:]), set(np.argsort(ref_lp)[-5:])
            return {"agree": int(mine.argmax()) == int(gen_ids[k]), "kl": float((p * (ref_lp - mine)).sum()),
                    "top5": len(top_mine & top_ref) / 5}

        rows, step_times = [compare(last, 0)], []
        for k in range(N - 1):
            t = time.monotonic()
            lg = step(state, int(gen_ids[k]), k)
            step_times.append(time.monotonic() - t)
            rows.append(compare(lg, k + 1))
        agree = [r["agree"] for r in rows]
        kls = [r["kl"] for r in rows]
        report["teacherForced"] = {"reference": str(ref["reference"]) if "reference" in ref else "transformers-fp32",
                                   "argmaxAgree": f"{sum(agree)}/{len(agree)}",
                                   "disagreeAt": [i for i, a in enumerate(agree) if not a],
                                   "klMean": round(float(np.mean(kls)), 5), "klMax": round(float(np.max(kls)), 5),
                                   "top5OverlapMean": round(float(np.mean([r["top5"] for r in rows])), 3)}
        warm = sorted(chunk_times[1:] or chunk_times)
        report["speed"] = {"prefillMsPerChunk": round(1000 * warm[len(warm) // 2], 1) if warm else None,
                           "stepMs": round(1000 * sorted(step_times)[len(step_times) // 2], 1) if step_times else None}
        print("teacher-forced", report["teacherForced"], report["speed"], flush=True)
        # 4. free greedy from a fresh prompt
        state = fresh()
        last, _ = prompt(state)
        out = [int(last.argmax())]
        for k in range(N - 1):
            out.append(int(step(state, out[-1], k).argmax()))
        same = next((i for i, (a, b) in enumerate(zip(out, gen_ids)) if a != int(b)), None)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(meta["snapshot"])
        report["greedy"] = {"identicalTokens": same is None, "firstDifference": same,
                            "text": tok.decode(out, skip_special_tokens=True),
                            "referenceText": tok.decode(gen_ids, skip_special_tokens=True)}
        print("greedy", report["greedy"], flush=True)
    finally:
        sp.release_lock()
    report["guard"] = "passed"
    out_path = args.decoder / "run-vlm.json"
    json.dump(report, open(out_path, "w"), indent=1)
    print(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("export-vision")
    v.add_argument("--weights", type=Path, required=True)
    v.add_argument("--output", type=Path, required=True)
    d = sub.add_parser("export-decoder")
    d.add_argument("--weights", type=Path, required=True)
    d.add_argument("--output", type=Path, required=True)
    d.add_argument("--weight-bits", type=int, choices=[16, 4], default=16)
    c = sub.add_parser("check-vision")
    c.add_argument("--weights", type=Path, required=True)
    c.add_argument("--vision", type=Path, required=True)
    c.add_argument("--pictures", type=Path, nargs="+", required=True)
    r = sub.add_parser("run")
    r.add_argument("--weights", type=Path, required=True)
    r.add_argument("--vision", type=Path, required=True)
    r.add_argument("--decoder", type=Path, required=True)
    r.add_argument("--ref", type=Path, required=True)
    args = ap.parse_args()
    {"export-vision": export_vision, "check-vision": check_vision, "export-decoder": export_decoder, "run": run}[args.cmd](args)


if __name__ == "__main__":
    main()

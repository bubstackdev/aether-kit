#!/usr/bin/env python3
"""Qwen3.6-27B hybrid-prefill spike (spike 1 of the 27B describer plan): does a Core AI bundle with a one-token `main`
function and a static S-token `prefill` function, sharing the four hybrid states, load on this macOS, agree with the
one-token loop, and how fast are both?

Real dimensions (hidden 5120, 48 GDN value heads over 16 key heads, gated full attention every 4th layer), the first
N real layers (from extract_mlx_layers.py), fp16 weights and KV, fp32 recurrent state. Text positions only (plain
RoPE): M-RoPE and images are later slices. Reuses the community authoring port in Apple's coreai-models
(`_reference/coreai-models`, BSD) through PYTHONPATH, unmodified.

    COREAI_MODELS_SRC=<_reference/coreai-models/python/src> Forge/.venv/bin/python -I Spikes/qwen36-prefill/spike.py \
        export --weights <extract dir> --layers 4 --scan chunk|metal --chunk 64 --capacity 1024 --output <new dir>
    ... spike.py run --weights <extract dir> --bundle <dir> [--tokens 320] [--compute gpu]

`run` holds /tmp/atelier-gpu.lock (the shared lease's directory form) while it touches the GPU.
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import resource
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

# `-I` ignores PYTHONPATH: the coreai-models checkout comes from COREAI_MODELS_SRC (its python/src folder).
if os.environ.get("COREAI_MODELS_SRC"):
    sys.path.insert(0, os.environ["COREAI_MODELS_SRC"])
    # coreai_models.export's __init__ pulls the whole pipeline (datasets, …); only mlir_ops is needed: a bare package.
    _export = types.ModuleType("coreai_models.export")
    _export.__path__ = [os.path.join(os.environ["COREAI_MODELS_SRC"], "coreai_models", "export")]
    sys.modules["coreai_models.export"] = _export
LOCK = "/tmp/atelier-gpu.lock"
SUBCHUNK = 16  # the in-graph scan's piece size (0: the whole block at once)
STATES = ["keyCache", "valueCache", "convState", "recState"]
# convState is stored taps-major, [L,1,K-1,conv_dim]: as [L,1,conv_dim,K-1] (6-byte fp16 rows) Core AI returned it
# unchanged from `main` (2026-11-10), while the IR wrote it correctly. Eager callers of the zoo layout set this False.
CONV_T = True
INPUTS = ["inputs_embeds", "position_ids"]
OUTPUTS = ["hidden"]


def zoo():
    import coreai_models.models.macos.qwen3_5 as q
    from coreai_models.primitives.macos.cache import SSMState
    if not getattr(SSMState, "_spike_cast", False):
        # The port returns the new recurrent state in the activations' dtype (fp16); the spike keeps recState fp32,
        # so each write is cast to the state's own dtype (the recurrence itself already runs in fp32).
        update = SSMState.update_states
        SSMState.update_states = lambda self, i, new: update(self, i, new.to(self.states.dtype))
        SSMState._spike_cast = True
        # The scans return the new state in the query's dtype; given fp32 operands they keep it fp32 (no per-token
        # fp16 rounding of the recurrence), and only the output goes back to the activations' dtype.
        for name in ("_gated_delta_step", "_gated_delta_chunk"):
            original = getattr(q, name)

            def wide(query, key, value, g, beta, initial_state, *rest, _f=original, _chunk=name.endswith("chunk"), **kw):
                args = [t.float() for t in (query, key, value, g, beta)]
                state = initial_state.float()
                s = query.shape[2]
                # The doubling inverse loses the state at 64-token blocks even in fp32 (layer 0: |state| 2e6 vs 32
                # exact; 16 is exact to 1e-6, 32 to 3e-4), so a long block is scanned as SUBCHUNK-token pieces in
                # sequence inside the same graph: the projections still see the whole block.
                step = SUBCHUNK if _chunk and SUBCHUNK and s > SUBCHUNK else s
                outs = []
                for i in range(0, s, step):
                    out, state = _f(*(a[:, :, i:i + step] for a in args), state, *rest, **kw)
                    outs.append(out)
                out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
                return out.to(query.dtype), state
            setattr(q, name, wide)
    return q


def config(weights: Path, layers: int, capacity: int):
    q = zoo()
    text = json.load(open(weights / "extract.json"))["textConfig"]
    return q.qwen3_5_config_from_hf(types.SimpleNamespace(**text), max_context_length=capacity, num_layers=layers)


def build(weights: Path, layers: int, capacity: int, dtype=torch.float16):
    from safetensors.torch import load_file
    q = zoo()
    cfg = config(weights, layers, capacity)
    core = q.Qwen3_5DecodeCore(cfg)
    raw = load_file(str(weights / "layers.safetensors"))
    prefix = "model.language_model."
    state = {"model." + k[len(prefix):]: v for k, v in raw.items()
             if k.startswith(prefix) and int(k.split(".")[3]) < layers}
    missing, unexpected = core.load_state_dict(state, strict=False)
    missing = [m for m in missing if m != "model.norm.weight" and not m.endswith("embed_tokens.weight")]
    if missing or unexpected:
        raise SystemExit(f"missing {missing[:8]} unexpected {unexpected[:8]}")
    with torch.no_grad():
        core.model.norm.weight.zero_()  # (1 + 0): the cut model has no final norm of its own
    core.model.reset_buffers()
    core.last_token_only = True
    for layer in core.model.layers:  # the recurrent state stays fp32 (spike F: fp16 state was slower and drifts)
        if not layer.is_full:
            layer.linear_attn.A_log.data = layer.linear_attn.A_log.data.float()
    return cfg, core.to(dtype).eval()


def set_scan(core, mode: str, kernel=None):
    for layer in core.model.layers:
        if layer.is_full:
            continue
        la = layer.linear_attn
        la.use_loopfree_step = mode == "step"
        la.use_loopfree_chunk = mode == "chunk"
        la.use_metal_chunk = mode == "metal"
        if mode == "metal":
            from coreai_models.models.macos.qwen3_5_gdn_metal import MetalGDNChunk
            la.metal_chunk = MetalGDNChunk(kernel, use_qk_l2_norm=True, chunk_max=kernel.chunk_max)


REC_DTYPE = torch.float32
STATIC_INPUTS = ["inputs_embeds", "positions", "write", "mask"]


class StaticCore(torch.nn.Module):
    """Every input has a fixed shape (the Python runtime re-specializes, and MPSGraph writes temp files, for each new
    shape: 2026-11-09 it wrote 369 GB). Per call: `inputs_embeds` [1,S,H], `positions` [1,S] int32 (RoPE),
    `write` [C,S] fp16 one-hot (cache row c takes query row s), `mask` [1,1,S,C] fp16 additive (0 or -inf-ish).
    The full-attention layers rewrite their cache as keep * cache + write @ new; the GDN layers are the port's code."""

    def __init__(self, core):
        super().__init__()
        self.m = core.model

    def attend(self, a, x, cos, sin, k_cache, v_cache, i, write, mask):
        b, s, _ = x.shape
        H, HKV, D = a.n_heads, a.n_kv_heads, a.head_dim
        q, gate = a.q_proj(x).view(b, s, H, D * 2).chunk(2, dim=-1)
        gate = gate.reshape(b, s, H * D)
        q = a.q_norm(q).transpose(1, 2)
        k = a.k_norm(a.k_proj(x).view(b, s, HKV, D)).transpose(1, 2)
        v = a.v_proj(x).view(b, s, HKV, D).transpose(1, 2)
        q, k = zoo().apply_rope(q, k, cos, sin)
        keep = 1 - write.sum(dim=-1, keepdim=True)                     # [C,1]
        if s == 1:
            # Core AI lowers `write @ v` with a one-row query in the wrong layout (2026-11-10: the value for row 5 of 4
            # KV heads landed in rows 20-23, keys were right); a broadcast multiply avoids the K=1 matmul.
            col = write.reshape(1, 1, -1, 1)                           # [1,1,C,1]
            keys = k_cache[i] * keep + col * k
            values = v_cache[i] * keep + col * v
        else:
            keys = k_cache[i] * keep + write @ k                       # [1,HKV,C,D]
            values = v_cache[i] * keep + write @ v
        k_cache[i].copy_(keys)
        v_cache[i].copy_(values)
        r = H // HKV
        keys = keys.unsqueeze(2).expand(b, HKV, r, keys.shape[2], D).reshape(b, H, keys.shape[2], D)
        values = values.unsqueeze(2).expand(b, HKV, r, values.shape[2], D).reshape(b, H, values.shape[2], D)
        out = torch.nn.functional.scaled_dot_product_attention(q, keys, values, attn_mask=mask, scale=D ** -0.5)
        out = out.transpose(1, 2).reshape(b, s, H * D) * torch.sigmoid(gate)
        return a.o_proj(out)

    def forward(self, inputs_embeds, positions, write, mask, k_cache, v_cache, conv_state, rec_state):
        cos, sin = self.m.rope_cos_sin(positions)
        h, fi, li = inputs_embeds, 0, 0
        for layer in self.m.layers:
            normed = layer.input_layernorm(h)
            if layer.is_full:
                r = self.attend(layer.self_attn, normed, cos, sin, k_cache, v_cache, fi, write, mask)
                fi += 1
            else:
                conv_in = conv_state[li].transpose(-1, -2) if CONV_T else conv_state[li]
                r, new_conv, new_rec = layer.linear_attn(normed, conv_in, rec_state[li])
                if inputs_embeds.shape[1] < new_conv.shape[-1]:
                    # At S=1 the new window is the old one shifted plus one new column: a view of the state itself,
                    # and Core AI drops that write (2026-11-10: convState came back unchanged). Rebuild it as a fresh
                    # tensor: the old taps through a fixed shift matrix, the new column masked in.
                    k1 = new_conv.shape[-1]
                    shift = torch.diag(torch.ones(k1 - 1, dtype=conv_in.dtype), -1)   # [c0,c1,c2] @ shift = [c1,c2,0]
                    last = torch.zeros(k1, dtype=conv_in.dtype); last[-1] = 1
                    new_conv = conv_in @ shift + new_conv * last
                new_conv = new_conv.transpose(-1, -2) if CONV_T else new_conv
                # `copy_` into the conv state lowers to a write Core AI never applies (2026-11-10: convState stayed
                # zero through `main` and `prefill`, in either layout, while the IR wrote it); the zoo's own
                # mutable_slice_update (SSMState) is the path its stateful exports use.
                from coreai_models.primitives.macos.cache import SSMState
                SSMState(conv_state).update_states(li, new_conv.to(conv_state.dtype))
                rec_state[li].copy_(new_rec.to(rec_state.dtype))
                li += 1
            h = h + r
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        return self.m.norm(h)[:, -1:, :]


def host_inputs(start: int, s: int, capacity: int):
    """positions, write, mask for query rows start..start+s-1 (numpy, the run's feeds)."""
    pos = np.arange(start, start + s, dtype=np.int32)
    write = np.zeros((capacity, s), np.float16)
    write[pos, np.arange(s)] = 1
    mask = np.where(np.arange(capacity)[None, :] <= pos[:, None], 0, -30000).astype(np.float16)[None, None]
    return pos[None], write, mask


def example(cfg, s: int, capacity: int, offset: int, dtype=torch.float16):
    q = zoo()
    st = q.build_decode_state(cfg, max_seq_len=capacity, dtype=dtype)
    st["rec_state"] = st["rec_state"].to(REC_DTYPE)
    if CONV_T:
        st["conv_state"] = st["conv_state"].transpose(-1, -2).contiguous()
    pos, write, mask = host_inputs(offset, s, capacity)
    ex = (torch.zeros(1, s, cfg.hidden_size, dtype=dtype), torch.from_numpy(pos), torch.from_numpy(write),
          torch.from_numpy(mask), st["k_cache"], st["v_cache"], st["conv_state"], st["rec_state"])
    return ex, None


def export(args):
    import coreai_torch as ct
    from coreai_models.export.mlir_ops import register_custom_torch_lowering, remove_functionalization
    global REC_DTYPE
    REC_DTYPE = torch.float16 if args.rec == "fp16" else torch.float32
    global SUBCHUNK
    SUBCHUNK = args.subchunk
    if args.output.exists():
        raise SystemExit("output exists")
    t0 = time.monotonic()
    cfg, core = build(args.weights, args.layers, args.capacity)
    static = StaticCore(core).eval()
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    kernel = None
    if args.scan == "metal":
        from coreai_models.models.macos.qwen3_5_gdn_metal import build_gdn_chunk_kernel
        kernel = build_gdn_chunk_kernel(chunk_max=args.chunk)
        kernel.chunk_max = args.chunk
        converter.register_custom_kernels([kernel])
    timings = {"build": round(time.monotonic() - t0, 1)}
    for name, s, mode in (("main", 1, args.main_scan), ("prefill", args.chunk, args.scan)):
        set_scan(core, mode, kernel)
        ex, dyn = example(cfg, s, args.capacity, offset=8)
        t = time.monotonic()
        with torch.no_grad():
            ep = torch.export.export(static, ex, strict=False)
            ep = ep.run_decompositions(ct.get_decomp_table())
        remove_functionalization(ep)
        mutated = len(ep.graph_signature.user_inputs_to_mutate)
        expected = 4 if any(l.is_full for l in core.model.layers) else 2  # GDN-only cuts never write the KV cache
        if mutated != expected:
            raise SystemExit(f"{name}: {mutated} mutated inputs, expected {expected} states")
        dims = [list(n.meta["val"].shape) for n in ep.graph.nodes if n.op == "placeholder" and "val" in n.meta]
        if any(not isinstance(d, int) for shape in dims for d in shape):
            raise SystemExit(f"{name}: a dynamic dimension slipped in: {dims}")
        converter.add_exported_program(ep, input_names=STATIC_INPUTS, output_names=OUTPUTS, state_names=STATES,
                                       entrypoint_name=name)
        timings[f"export_{name}"] = round(time.monotonic() - t, 1)
        del ep
    register_custom_torch_lowering(converter)
    t = time.monotonic()
    program = converter.to_coreai()
    program.optimize()
    args.output.mkdir(parents=True)
    program.save_asset(args.output / "model.aimodel")
    timings["convert"] = round(time.monotonic() - t, 1)
    size = sum(p.stat().st_size for p in (args.output / "model.aimodel").rglob("*") if p.is_file())
    report = {"layers": args.layers, "kinds": "".join("a" if l.is_full else "l" for l in core.model.layers),
              "scan": args.scan, "subchunk": args.subchunk, "mainScan": args.main_scan, "recState": args.rec, "chunk": args.chunk, "capacity": args.capacity, "bytes": size,
              "timings": timings, "peakRSSGB": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 1),
              "hostOS": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()}
    json.dump(report, open(args.output / "export.json", "w"), indent=1)
    print(json.dumps(report))


# MARK: run

def take_lock():
    while True:
        try:
            os.mkdir(LOCK)
            open(os.path.join(LOCK, "pid"), "w").write(str(os.getpid()))
            return
        except FileExistsError:
            time.sleep(5)


def release_lock():
    try:
        if open(os.path.join(LOCK, "pid")).read().strip() == str(os.getpid()):
            shutil.rmtree(LOCK)
    except FileNotFoundError:
        pass


def reference(cfg, core, embeds: np.ndarray, capacity: int, chunk: int):
    """torch fp32 on CPU, the one-token loop (exact recurrence): last-row hidden after every token, final states."""
    q = zoo()
    ref = core.float()
    set_scan(ref, "step")
    st = q.build_decode_state(cfg, max_seq_len=capacity, dtype=torch.float32)
    rows = []
    with torch.no_grad():
        for t in range(len(embeds)):
            h = ref(torch.from_numpy(embeds[t:t + 1][None].astype(np.float32)),
                    torch.arange(t + 1, dtype=torch.int32)[None],
                    st["k_cache"], st["v_cache"], st["conv_state"], st["rec_state"])
            rows.append(h[0, -1].numpy().copy())
    return np.stack(rows), {k: v.numpy().copy() for k, v in st.items()}


def cosine(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def run(args):
    import coreai.runtime as rt
    meta = json.load(open(args.bundle / "export.json"))
    cfg, core = build(args.weights, meta["layers"], meta["capacity"])
    embeds = np.load(args.weights / "embeds.npy")[:args.tokens]
    chunk, capacity = meta["chunk"], meta["capacity"]
    t = time.monotonic()
    ref_rows, ref_states = reference(cfg, core, embeds, capacity, chunk)
    report = {**meta, "tokens": len(embeds), "referenceSeconds": round(time.monotonic() - t, 1)}
    del core
    gc.collect()

    options = (rt.SpecializationOptions.cpu_only() if args.compute == "cpu" else
               rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
    take_lock()
    try:
        t = time.monotonic()
        model = asyncio.run(rt.AIModel.load(args.bundle / "model.aimodel", options))
        report["loadSeconds"] = round(time.monotonic() - t, 1)
        fns = {}
        for name in ("main", "prefill"):
            t = time.monotonic()
            try:
                fns[name] = model.load_function(name)
                report[f"loadFunction_{name}"] = round(time.monotonic() - t, 1)
            except Exception as e:  # the go/no-go: does a prefill function load at all here?
                report[f"loadFunction_{name}"] = f"FAILED: {type(e).__name__}: {e}"[:400]
        desc = fns["main"].desc
        shapes = {n: (list(desc.state_descriptor(n).shape), desc.state_descriptor(n).dtype) for n in desc.state_names}
        report["states"] = {n: [s, str(d)] for n, (s, d) in shapes.items()}

        def fresh():
            out = {}
            for n, (s, d) in shapes.items():
                out[n] = rt.NDArray(np.zeros(s, dtype=np.float32 if "32" in str(d) else np.float16))
            return out

        temp = Path(subprocess.run(["getconf", "DARWIN_USER_TEMP_DIR"], capture_output=True, text=True).stdout.strip())
        mps = temp / "com.apple.MetalPerformanceShadersGraph"
        free0 = shutil.disk_usage("/").free

        def guard():
            """Stop before the disk does: MPSGraph temp for this process under 20 GB, free disk within 30 GB of start."""
            mine = sum(f.stat().st_size for d in mps.glob(f"mpsgraph-{os.getpid()}-*") for f in d.rglob("*") if f.is_file())
            drop = free0 - shutil.disk_usage("/").free
            if mine > 20 * 2**30 or drop > 30 * 2**30:
                raise SystemExit(f"guard: MPSGraph temp {mine / 2**30:.1f} GB, free disk down {drop / 2**30:.1f} GB; stopping")

        import threading

        def watchdog():
            while True:
                time.sleep(10)
                try:
                    guard()
                except SystemExit as e:
                    print(e, flush=True)
                    release_lock()
                    os._exit(3)
        threading.Thread(target=watchdog, daemon=True).start()
        calls = [0]

        def call(fn, feeds, state):
            calls[0] += 1
            if calls[0] % 16 == 1:
                guard()
            async def go():
                o = await fn({k: rt.NDArray(np.ascontiguousarray(v)) for k, v in feeds.items()}, state)
                return {k: v.numpy().copy() for k, v in o.items()}
            return asyncio.run(go())

        def feeds(lo, hi):
            pos, write, mask = host_inputs(lo, hi - lo, capacity)
            return {"inputs_embeds": embeds[lo:hi][None].astype(np.float16), "positions": pos, "write": write, "mask": mask}

        # A: the one-token loop through `main`.
        state = fresh()
        rows_a, times_a = [], []
        for i in range(len(embeds)):
            t = time.monotonic()
            rows_a.append(call(fns["main"], feeds(i, i + 1), state)["hidden"][0, -1])
            times_a.append(time.monotonic() - t)
        rows_a = np.stack(rows_a).astype(np.float32)
        states_a = {n: state[n].numpy().copy() for n in state}
        warm = sorted(times_a[8:])
        report["main"] = {"msPerTokenMedian": round(1000 * warm[len(warm) // 2], 2),
                          "lastRowCosine": cosine(rows_a[-1], ref_rows[-1]),
                          "worstRowCosine": min(cosine(rows_a[i], ref_rows[i]) for i in range(len(embeds))),
                          "nan": bool(np.isnan(rows_a).any())}
        # B: `prefill` chunks, the remainder through `main`.
        if "prefill" in fns and not isinstance(report.get("loadFunction_prefill"), str):
            state = fresh()
            times_b, rows_b, i = [], {}, 0
            while i + chunk <= len(embeds):
                t = time.monotonic()
                rows_b[i + chunk - 1] = call(fns["prefill"], feeds(i, i + chunk), state)["hidden"][0, -1]
                times_b.append(time.monotonic() - t)
                i += chunk
            while i < len(embeds):
                rows_b[i] = call(fns["main"], feeds(i, i + 1), state)["hidden"][0, -1]
                i += 1
            states_b = {n: state[n].numpy().copy() for n in state}
            warm_b = sorted(times_b[1:]) or times_b
            last = len(embeds) - 1
            report["prefill"] = {
                "msPerChunkMedian": round(1000 * warm_b[len(warm_b) // 2], 1),
                "firstChunkMs": round(1000 * times_b[0], 1),
                "tokensPerSecond": round(chunk / warm_b[len(warm_b) // 2], 0),
                "chunkEndCosineVsReference": {k: round(cosine(v, ref_rows[k]), 6) for k, v in rows_b.items() if (k + 1) % chunk == 0},
                "chunkEndCosineVsMainLoop": {k: round(cosine(v, rows_a[k]), 6) for k, v in rows_b.items() if (k + 1) % chunk == 0},
                "lastRowCosineVsReference": cosine(rows_b[last], ref_rows[last]),
                "recStateCosineVsMainLoop": cosine(states_b["recState"], states_a["recState"]),
                "recStateCosineVsReference": cosine(states_b["recState"], ref_states["rec_state"]),
                "nan": bool(any(np.isnan(v).any() for v in rows_b.values())),
            }
    finally:
        release_lock()
    report["guard"] = "passed"
    print(json.dumps(report, indent=1))
    out = args.bundle / f"run-{args.compute}.json"
    json.dump(report, open(out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--weights", type=Path, required=True)
    e.add_argument("--layers", type=int, default=4)
    e.add_argument("--scan", choices=["chunk", "metal"], default="chunk")
    e.add_argument("--chunk", type=int, default=64)
    e.add_argument("--capacity", type=int, default=1024)
    e.add_argument("--rec", choices=["fp32", "fp16"], default="fp32")
    e.add_argument("--subchunk", type=int, default=16)
    e.add_argument("--main-scan", choices=["step", "chunk"], default="step")
    e.add_argument("--output", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--weights", type=Path, required=True)
    r.add_argument("--bundle", type=Path, required=True)
    r.add_argument("--tokens", type=int, default=320)
    r.add_argument("--compute", choices=["gpu", "cpu"], default="gpu")
    args = ap.parse_args()
    export(args) if args.cmd == "export" else run(args)


if __name__ == "__main__":
    main()

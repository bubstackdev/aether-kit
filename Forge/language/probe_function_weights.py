#!/usr/bin/env python3
"""Toy decoder layouts for Core AI's per-function weight handling (development tooling only).

    Forge/.venv/bin/python -I Forge/language/probe_function_weights.py layout <main|image|two|twin|embeds> --output <new .aimodel>
    Forge/.venv/bin/python -I Forge/language/probe_function_weights.py export-memory <one|two> --output <new .aimodel>

`layout` writes a small asset (144 MiB of fp16 weights) shaped like the Qwen3-VL text decoder: an embedding table
tied to the head, blocks of two matrices, and `withImage`'s splice and DeepStack adds. Layouts:
  main    `main(input_ids)` only
  image   `withImage(input_ids, image_embeds, deepstack)` only, q from 1 (text steps send one zero image row)
  two     `main` + `withImage`, today's Qwen3-VL decoder layout
  twin    `main` + `main2`, the identical graph under two names
  embeds  `main(inputs_embeds, image_mask, deepstack)` + `embed(input_ids)`, the lookup as its own function
  aligned `main(input_ids, image_embeds [q, hidden], deepstack [levels, q, hidden])`, rows aligned with the
          submission and used where the token is an image pad (the proposed contract)
`Scripts/probe-function-weights.swift` runs them on the GPU (docs/qwen3-vl-single-decoder.md).

`export-memory` converts a larger toy (2.03 GiB) as one program or two over the same weights, and prints RSS and
phys_footprint after each stage of the exporter's pipeline, with a cpu_only load at the end like the self-check.
`--free-eager` frees the eager module after conversion, before saving.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import gc
import os
import threading
import time
from pathlib import Path

import psutil
import torch
import torch.nn.functional as F

IMAGE, LEVELS = 7, 2


class Toy(torch.nn.Module):
    def __init__(self, vocab: int, hidden: int, inner: int, layers: int) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden, dtype=torch.float16)
        self.w1 = torch.nn.ParameterList(torch.nn.Parameter(torch.empty(inner, hidden, dtype=torch.float16)) for _ in range(layers))
        self.w2 = torch.nn.ParameterList(torch.nn.Parameter(torch.empty(hidden, inner, dtype=torch.float16)) for _ in range(layers))
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for p in self.parameters():         # distinct values, so the converter cannot merge blobs
                p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float16) * 0.02)

    def blocks(self, x, row=None, pick=None, deepstack=None, mask=None):
        for i in range(len(self.w1)):
            x = x + F.linear(F.silu(F.linear(x, self.w1[i])), self.w2[i])
            if deepstack is not None and i < LEVELS:
                if mask is not None:
                    x = x + deepstack[i][None] * mask
                else:
                    extra = F.embedding(row, deepstack[i])[None].to(x.dtype)
                    x = x + torch.where(pick, extra, torch.zeros_like(extra))
        return F.linear(x[:, -1:], self.embed.weight)

    def forward(self, input_ids, image_embeds=None, deepstack=None):
        x = self.embed(input_ids)
        if image_embeds is None:
            return self.blocks(x)
        # The Qwen3-VL splice: pad rows take the next image row, by a running count over the pad mask.
        is_image = input_ids[0] == IMAGE
        row = torch.clamp(torch.cumsum(is_image.to(torch.int32), 0, dtype=torch.int32) - 1, min=0)
        pick = is_image[None, :, None]
        x = torch.where(pick, F.embedding(row, image_embeds)[None].to(x.dtype), x)
        return self.blocks(x, row, pick, deepstack)


class WithImage(torch.nn.Module):
    def __init__(self, inner: Toy) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, input_ids, image_embeds, deepstack):
        return self.inner(input_ids, image_embeds, deepstack)


class Aligned(torch.nn.Module):
    """The proposed contract: image rows aligned with the submission, used where the token is an image pad."""

    def __init__(self, inner: Toy) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, input_ids, image_embeds, deepstack):
        pick = (input_ids[0] == IMAGE)[None, :, None]
        x = torch.where(pick, image_embeds[None], self.inner.embed(input_ids))
        for i in range(len(self.inner.w1)):
            x = x + F.linear(F.silu(F.linear(x, self.inner.w1[i])), self.inner.w2[i])
            if i < LEVELS:
                x = x + torch.where(pick, deepstack[i][None], torch.zeros_like(x))
        return F.linear(x[:, -1:], self.inner.embed.weight)


class FromEmbeds(torch.nn.Module):
    def __init__(self, inner: Toy) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, inputs_embeds, image_mask, deepstack):
        return self.inner.blocks(inputs_embeds, deepstack=deepstack, mask=image_mask)


class Embed(torch.nn.Module):
    def __init__(self, inner: Toy) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, input_ids):
        return self.inner.embed(input_ids)


def convert(toy: Toy, entries: list[str], hidden: int, stage=lambda name: None):
    import coreai_torch as ct

    q = torch.export.Dim("q", min=1, max=512)
    q2 = torch.export.Dim("q", min=2, max=512)                 # today's withImage lower bound
    n = torch.export.Dim("n", min=1, max=512)
    ids = torch.arange(8, dtype=torch.int32)[None] + 100
    wids = torch.arange(12, dtype=torch.int32)[None] + 100
    wids[0, 3:8] = IMAGE
    image = (wids, torch.zeros(5, hidden, dtype=torch.float16), torch.zeros(LEVELS, 5, hidden, dtype=torch.float16))
    decompositions = ct.get_decomp_table()
    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    with torch.no_grad():
        for entry in entries:
            if entry in ("main", "main2"):
                ep = torch.export.export(toy, (ids,), dynamic_shapes=({1: q},))
                names = (["input_ids"], ["logits"])
            elif entry in ("withImage", "withImage-q1"):
                ep = torch.export.export(WithImage(toy), image,
                                         dynamic_shapes=({1: q if entry == "withImage-q1" else q2}, {0: n}, {1: n}))
                names, entry = (["input_ids", "image_embeds", "deepstack"], ["logits"]), "withImage"
            elif entry == "aligned":
                a = (wids, torch.zeros(12, hidden, dtype=torch.float16), torch.zeros(LEVELS, 12, hidden, dtype=torch.float16))
                ep = torch.export.export(Aligned(toy), a, dynamic_shapes=({1: q}, {0: q}, {1: q}))
                names, entry = (["input_ids", "image_embeds", "deepstack"], ["logits"]), "main"
            elif entry == "fromEmbeds":
                e = (torch.zeros(1, 8, hidden, dtype=torch.float16), torch.zeros(1, 8, 1, dtype=torch.float16),
                     torch.zeros(LEVELS, 8, hidden, dtype=torch.float16))
                ep = torch.export.export(FromEmbeds(toy), e, dynamic_shapes=({1: q}, {1: q}, {1: q}))
                names, entry = (["inputs_embeds", "image_mask", "deepstack"], ["logits"]), "main"
            else:
                ep = torch.export.export(Embed(toy), (ids,), dynamic_shapes=({1: q},))
                names = (["input_ids"], ["inputs_embeds"])
            converter.add_exported_program(ep.run_decompositions(decompositions), input_names=names[0],
                                           output_names=names[1], state_names=[], entrypoint_name=entry)
            del ep
    stage("exported programs")
    program = converter.to_coreai()
    stage("to_coreai")
    program.optimize()
    stage("optimize")
    return program, (ids, image)


LAYOUTS = {"main": ["main"], "image": ["withImage-q1"], "two": ["main", "withImage"], "twin": ["main", "main2"],
           "embeds": ["fromEmbeds", "embed"], "aligned": ["aligned"]}


class _RusageV2(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f, ctypes.c_uint64) for f in (
        "user", "system", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size", "resident_size",
        "phys_footprint", "proc_start_abstime", "proc_exit_abstime", "child_user", "child_system",
        "child_pkg_idle_wkups", "child_interrupt_wkups", "child_pageins", "child_elapsed_abstime",
        "diskio_bytesread", "diskio_byteswritten")]


def phys_footprint() -> int:
    usage = _RusageV2()
    ctypes.CDLL("/usr/lib/libSystem.B.dylib").proc_pid_rusage(os.getpid(), 2, ctypes.byref(usage))
    return usage.phys_footprint


def export_memory(programs: str, output: Path, free_eager: bool) -> None:
    process, start, peak = psutil.Process(), time.time(), [0]

    def sample():
        while True:
            peak[0] = max(peak[0], process.memory_info().rss)
            time.sleep(0.05)

    def stage(name):
        rss = process.memory_info().rss
        print(f"{name:26s} rss {rss / 2**30:6.2f} GiB (peak {max(peak[0], rss) / 2**30:6.2f})  "
              f"footprint {phys_footprint() / 2**30:6.2f} GiB  {time.time() - start:6.1f} s", flush=True)
        peak[0] = rss

    threading.Thread(target=sample, daemon=True).start()
    hidden = 2048
    toy = Toy(8192, hidden, 8192, 32).eval()
    stage(f"eager module ({sum(p.numel() * 2 for p in toy.parameters()) / 2**30:.2f} GiB)")
    program, (ids, image) = convert(toy, ["main", "withImage-q1"] if programs == "two" else ["withImage-q1"], hidden, stage)
    if free_eager:
        del toy                                 # the converter's staged programs went with `convert`
        gc.collect()
        stage("eager module freed")
    program.save_asset(output)
    stage("save_asset")

    async def load():
        import coreai.runtime as rt

        model = await rt.AIModel.load(output, rt.SpecializationOptions.cpu_only())
        function = model.load_function("withImage")
        await function({"input_ids": rt.NDArray(ids.numpy()), "image_embeds": rt.NDArray(image[1][:1].numpy()),
                        "deepstack": rt.NDArray(image[2][:, :1].numpy())}, {})

    asyncio.run(load())
    stage("cpu_only load and call")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["layout", "export-memory"])
    parser.add_argument("variant")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--free-eager", action="store_true", help="export-memory: free the eager module before saving")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output already exists; choose a new path")
    if args.mode == "export-memory":
        if args.variant not in ("one", "two"):
            raise SystemExit("export-memory takes one or two")
        export_memory(args.variant, args.output, args.free_eager)
        return
    if args.variant not in LAYOUTS:
        raise SystemExit(f"layout is one of {sorted(LAYOUTS)}")
    program, _ = convert(Toy(8192, 1024, 4096, 8).eval(), LAYOUTS[args.variant], 1024)
    program.save_asset(args.output)


if __name__ == "__main__":
    main()

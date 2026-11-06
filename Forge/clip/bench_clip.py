#!/usr/bin/env python3
"""The text-batch and precision measurements behind the CLIP export's shape choices (recorded in export-report.json).

    Forge/.venv/bin/python -I Forge/clip/bench_clip.py --output <new dir> [--batches 1 2 4 8 16] [--runs 20]
    Forge/.venv/bin/python -I Forge/clip/bench_clip.py --time-bundles <bundle> [<bundle> ...] [--runs 50]

Exports the text graph at each static batch (float16), plus one asset holding the batch-1 and batch-8 functions
together (to see whether Core AI shares the weights between functions), then loads each on the GPU and times a
warm call. Prints and writes `bench.json`: asset bytes, load seconds, median call seconds and seconds per text row.
GPU: a few seconds per graph, one at a time. `--time-bundles` instead times each exported bundle's graphs (warm median,
interleaved across bundles so GPU contention hits all of them alike) and prints the result.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clip_graphs as cg  # noqa: E402


def export(functions: dict[str, int], path: Path, snapshot: Path) -> None:
    import coreai_torch as ct

    converter = ct.TorchConverter(mode=ct.TorchConverter.Mode.RELEASE)
    for name, batch in functions.items():
        graph = cg.TextGraph(cg.build_model(snapshot), torch.float16).eval()
        example = (torch.full((batch, cg.TEXT_LENGTH), cg.EOS, dtype=torch.int32),)
        with torch.no_grad():
            ep = torch.export.export(graph, example, strict=False).run_decompositions(ct.get_decomp_table())
        converter.add_exported_program(ep, input_names=["tokens"], output_names=["text_embedding"], state_names=[],
                                       entrypoint_name=name)
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(path)


def time_bundles(bundles: list[Path], runs: int) -> dict:
    import coreai.runtime as rt

    options = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    image = rt.NDArray(cg.synthetic_tensor(1).astype(np.float16))
    calls = {}
    for bundle in bundles:
        m = json.loads((bundle / ".aether/aether.json").read_text())
        for role, spec in m["graphs"].items():
            model = asyncio.run(rt.AIModel.load(bundle / spec["file"].split(":", 1)[1], options))
            entries = spec.get("buckets", {}).get("entries", [{"value": 1, "function": spec["function"]}])
            for e in entries:
                fn = model.load_function(e["function"])
                if role == "imageEncoder":
                    feeds = {"image": image}
                else:
                    ids = np.full((e["value"], cg.TEXT_LENGTH), cg.EOS, dtype=np.int32)
                    ids[:, 0], ids[:, 1:6] = cg.BOS, 320
                    feeds = {"tokens": rt.NDArray(ids)}
                calls[(str(bundle), f"{role}.{e['function']}")] = (model, fn, feeds)
    times = {k: [] for k in calls}
    for _, fn, feeds in calls.values():
        asyncio.run(fn(feeds, {}))
    for _ in range(runs):
        for k, (_, fn, feeds) in calls.items():
            t = time.monotonic()
            asyncio.run(fn(feeds, {}))
            times[k].append(time.monotonic() - t)
    out = {}
    for (bundle, name), ts in times.items():
        out.setdefault(bundle, {})[name] = {"medianSeconds": round(statistics.median(ts), 5),
                                            "minSeconds": round(min(ts), 5)}
    return out


def time_function(path: Path, function: str, batch: int, runs: int) -> dict:
    import coreai.runtime as rt

    options = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    t0 = time.monotonic()
    model = asyncio.run(rt.AIModel.load(path, options))
    fn = model.load_function(function)
    load = time.monotonic() - t0
    ids = np.full((batch, cg.TEXT_LENGTH), cg.EOS, dtype=np.int32)
    ids[:, 0], ids[:, 1:6] = cg.BOS, 320

    async def call():
        return await fn({"tokens": rt.NDArray(ids)}, {})

    asyncio.run(call())
    times = []
    for _ in range(runs):
        t = time.monotonic()
        asyncio.run(call())
        times.append(time.monotonic() - t)
    median = statistics.median(times)
    return {"function": function, "batch": batch, "loadSeconds": round(load, 3), "medianSeconds": round(median, 5),
            "secondsPerRow": round(median / batch, 5)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path)
    ap.add_argument("--time-bundles", type=Path, nargs="+")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--snapshot", type=Path, default=cg.DEFAULT_SNAPSHOT)
    args = ap.parse_args()
    if args.time_bundles:
        print(json.dumps(time_bundles(args.time_bundles, args.runs), indent=1))
        return
    if args.output is None:
        ap.error("--output is required")
    if args.output.exists():
        raise SystemExit("output directory already exists; choose a new directory")
    args.output.mkdir(parents=True)
    torch.set_grad_enabled(False)
    results = {"static": [], "multiFunction": None}
    for batch in args.batches:
        path = args.output / f"text_b{batch}.aimodel"
        export({"main": batch}, path, args.snapshot)
        r = time_function(path, "main", batch, args.runs)
        r["bytes"] = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        results["static"].append(r)
        print(json.dumps(r), file=sys.stderr, flush=True)
    path = args.output / "text_b1_b8.aimodel"
    export({"b1": 1, "b8": 8}, path, args.snapshot)
    results["multiFunction"] = {"bytes": sum(p.stat().st_size for p in path.rglob("*") if p.is_file()),
                                "functions": [time_function(path, "b1", 1, args.runs),
                                              time_function(path, "b8", 8, args.runs)]}
    print(json.dumps(results["multiFunction"]), file=sys.stderr, flush=True)
    (args.output / "bench.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()

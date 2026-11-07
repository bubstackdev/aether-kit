#!/usr/bin/env python3
"""The AdcSR x4 gate: the exported Core AI graph (GPU) against the upstream CPU float32 references.

    scripts/gpu-lease -- Forge/.venv/bin/python -I Forge/adcsr/check_adcsr.py --bundle <bundle dir> \
        --reference <reference dir> --report <new check-report.json>

The graph file, function, role names and compute placement all come from the bundle's own manifest. For every
reference case (reference_adcsr.py) the gate checks:

- parity, per tile: each tile's input through the graph, its raw output against upstream's on that tile, PSNR >= 40 dB
  (docs/M4.md's super-resolution bar; raw [-1,1] outputs, peak-to-peak 2);
- parity, whole image: the GPU tiles stitched with the driver's rule (`adcsr_graphs.stitch`), colour-matched, mapped
  and clamped as the manifest's `postprocess` says, against the reference's tiled final image, PSNR >= 40 dB on [0,1];
- fidelity, whole image (cases whose sides are multiples of 16): the same GPU result against upstream's own
  whole-image inference (test.py), held to the same 40 dB bar and reported separately. It measures the tiling, not
  the export: AdcSR's attention and GroupNorm see the whole input, so tiles of a photo larger than one tile miss it
  (reference_adcsr.py);
- a repeat run of the first tile that is bit-identical.

The graph's descriptor is also checked against the manifest's assertions: lr float32 [1,3,128,128], sr [1,3,512,512].
Each case's GPU result is written beside the report (`<report stem>/<case>/sr.f32`, `sr.png`). The exit status is
nonzero when a parity check fails; the fidelity verdict is the report's `fidelityPassed`. Runs one graph on the GPU,
one tile at a time, for about a minute: wrap it in the GPU lease.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adcsr_graphs as ag  # noqa: E402

TILE_BAR = IMAGE_BAR = FIDELITY_BAR = 40.0


class Graph:
    def __init__(self, path: Path, function: str, compute: str) -> None:
        import coreai.runtime as rt
        self.rt = rt
        options = (rt.SpecializationOptions.cpu_only() if compute == "cpu" else
                   rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
        self.model = asyncio.run(rt.AIModel.load(path, options))
        self.function = self.model.load_function(function)

    def descriptor(self) -> dict:
        d = self.function.desc
        describe = lambda x: {"dtype": str(x.dtype), "shape": list(x.shape)}  # noqa: E731
        return {"inputs": {n: describe(d.input_descriptor(n)) for n in d.input_names},
                "outputs": {n: describe(d.output_descriptor(n)) for n in d.output_names}}

    def __call__(self, **feeds: np.ndarray) -> dict[str, np.ndarray]:
        rt = self.rt

        async def go():
            out = await self.function({k: rt.NDArray(np.ascontiguousarray(v)) for k, v in feeds.items()}, {})
            return {k: v.numpy().copy() for k, v in out.items()}
        return asyncio.run(go())


def rounded(value: float) -> float | str:
    return round(value, 3) if np.isfinite(value) else "inf"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise SystemExit("report already exists; choose a new path")
    manifest = json.loads((args.bundle / ".aether/aether.json").read_text())
    index = json.loads((args.reference / "references.json").read_text())
    spec = manifest["graphs"]["upscaler"]
    lr_name, sr_name = spec["inputs"]["image"], spec["outputs"]["image"]
    tiling, sr_post = manifest["preprocess"]["image"]["tiling"], manifest["postprocess"]["sr"]
    if (tiling["tile"] != [ag.TILE, ag.TILE] or tiling["overlap"] != ag.OVERLAP or tiling["scale"] != ag.SCALE
            or sr_post["range"] != list(ag.RANGE) or sr_post["colorMatch"] != "host"):
        raise SystemExit("the manifest's tiling or postprocess differs from the harness's rule")
    dump = args.report.with_suffix("")
    checks: list[dict] = []

    def record(group: str, name: str, value: float, bar: float, passed: bool, **extra) -> None:
        checks.append({"group": group, "check": name, "value": value, "bar": bar, "passed": bool(passed), **extra})
        print(f"[check] {'PASS' if passed else 'FAIL'} {group} {name}: {value} (bar {bar})", file=sys.stderr,
              flush=True)

    started = time.monotonic()
    t0 = time.monotonic()
    graph = Graph(args.bundle / spec["file"].split(":", 1)[1], spec["function"], spec["compute"])
    load_seconds = round(time.monotonic() - t0, 2)
    descriptor = graph.descriptor()
    want_in, want_out = index["tile"], index["srTile"]
    ok = (descriptor["inputs"].get(lr_name, {}).get("dtype", "").endswith("float32")
          and descriptor["inputs"][lr_name]["shape"] == want_in
          and descriptor["outputs"].get(sr_name, {}).get("shape") == want_out)
    record("parity", "descriptor", float(ok), 1.0, ok, descriptor=descriptor)

    timings, repeat_done = [], False
    for entry in index["cases"]:
        case = json.loads((args.reference / entry["name"] / "case.json").read_text())
        folder = args.reference / case["name"]
        h, w = case["size"][1], case["size"][0]
        lr = ag.read_f32(folder / "input.f32", case["shapes"]["input"])
        tiles, worst = [], float("inf")
        for tile in case["tiles"]:
            i = tile["index"]
            x = ag.read_f32(folder / f"tiles/{i}.lr.f32", want_in)
            if not np.array_equal(x, ag.crop_tile(lr, tile["x0"], tile["y0"])):
                raise SystemExit(f"{case['name']} tile {i}: the recorded crop differs from the tiling rule's")
            want = ag.read_f32(folder / f"tiles/{i}.sr.f32", want_out)
            t0 = time.monotonic()
            got = graph(**{lr_name: x})[sr_name].astype(np.float32).reshape(want_out)
            timings.append(round(time.monotonic() - t0, 4))
            if not np.isfinite(got).all():
                raise SystemExit(f"{case['name']} tile {i}: non-finite graph output")
            p = ag.psnr_raw(got, want)
            worst = min(worst, p)
            record("parity", f"{case['name']}.tile{i}.psnr", rounded(p), TILE_BAR, p >= TILE_BAR,
                   maxAbs=float(np.abs(got - want).max()))
            if not repeat_done:
                again = graph(**{lr_name: x})[sr_name].astype(np.float32).reshape(want_out)
                same = np.array_equal(again, got)
                record("parity", f"{case['name']}.tile{i}.repeat.exact", float(same), 1.0, same)
                repeat_done = True
            tiles.append((tile["x0"], tile["y0"], got))
        final = ag.postprocess(ag.stitch(tiles, w, h), lr)
        want_final = ag.read_f32(folder / "sr.f32", case["shapes"]["sr"])
        p = ag.psnr(final, want_final)
        u8 = ag.psnr(ag.to_uint8(final).astype(np.float32), ag.to_uint8(want_final).astype(np.float32), 255.0)
        record("parity", f"{case['name']}.image.psnr", rounded(p), IMAGE_BAR, p >= IMAGE_BAR,
               uint8PSNR=rounded(u8), tiles=len(tiles), worstTilePSNR=rounded(worst))
        if case.get("tiledVsFull") is not None:
            full = ag.read_f32(folder / "full.f32", case["shapes"]["sr"])
            f = ag.psnr(final, full)
            record("fidelity", f"{case['name']}.tiledVsFull.psnr", rounded(f), FIDELITY_BAR, f >= FIDELITY_BAR,
                   tiles=len(tiles), cpuTiledVsFull=case["tiledVsFull"]["psnr"])
        (dump / case["name"]).mkdir(parents=True, exist_ok=True)
        ag.write_f32(dump / case["name"] / "sr.f32", final)
        ag.write_png(dump / case["name"] / "sr.png", ag.to_uint8(final))
    del graph
    parity = all(c["passed"] for c in checks if c["group"] == "parity")
    fidelity = all(c["passed"] for c in checks if c["group"] == "fidelity")
    args.report.write_text(json.dumps({"passed": parity, "fidelityPassed": fidelity,
                                       "seconds": round(time.monotonic() - started, 1), "loadSeconds": load_seconds,
                                       "tileSeconds": {"median": float(np.median(timings)), "max": max(timings),
                                                       "count": len(timings)},
                                       "precision": manifest.get("precision"),
                                       "bundle": str(args.bundle), "reference": str(args.reference),
                                       "aetherJSONSHA256": ag.sha256_file(args.bundle / ".aether/aether.json"),
                                       "checks": checks}, indent=2) + "\n")
    if not parity:
        raise SystemExit("parity check failed")


if __name__ == "__main__":
    main()

"""Run the exported TripoSR stages on the GPU over preprocessed images, for gate_triposr.py.

usage: USE_OS_COREAI=1 run_images.py <export-dir> <inputs-dir> <results-dir> [--res 256] [--slab 16] [--tag fp16]
GPU (under the lock): for each <stem>.npy, the scene codes (first call and warm median), then the 256³ density grid
three ways: `field` in chunks of 65,536 points, `field-262144` in chunks of 262,144, and `grid-<res>x<slab>` (the
gather-free slab function). CPU: marching cubes on the slab grid (PyMCubes), vertex colours through `field`, an OBJ
(trimesh), and <stem>-mesh.npz for the gate and the USDZ writer. Grids are saved as float16 to keep the results small.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import coreai.runtime as rt

ap = argparse.ArgumentParser()
ap.add_argument("exports")
ap.add_argument("inputs")
ap.add_argument("results")
ap.add_argument("--res", type=int, default=256)
ap.add_argument("--slab", type=int, default=16)
ap.add_argument("--tag", default="fp16")
ap.add_argument("--threshold", type=float, default=25.0)
a = ap.parse_args()
E, R = Path(a.exports), Path(a.results)
R.mkdir(parents=True, exist_ok=True)
RADIUS = 0.87
dt = np.float16 if a.tag == "fp16" else np.float32
lin = np.linspace(0, 1, a.res, dtype=np.float32) * 2 * RADIUS - RADIUS


async def load(name, opts):
    model = await rt.AIModel.load(E / f"{name}.aimodel", opts)
    return model, model.load_function("main")


async def main():
    opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    t0 = time.perf_counter()
    loaded = {n: await load(n, opts) for n in
              (f"scene-{a.tag}", f"field-{a.tag}", f"field-262144-{a.tag}", f"grid-{a.res}x{a.slab}-{a.tag}")}
    report = {"tag": a.tag, "res": a.res, "slab": a.slab, "loadSeconds": time.perf_counter() - t0, "images": {}}
    scene = loaded[f"scene-{a.tag}"][1]
    fields = {65536: loaded[f"field-{a.tag}"][1], 262144: loaded[f"field-262144-{a.tag}"][1]}
    grid_fn = loaded[f"grid-{a.res}x{a.slab}-{a.tag}"][1]
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], -1)

    for path in sorted(Path(a.inputs).glob("*.npy")):
        stem, rep = path.stem, {}
        image = rt.NDArray(np.load(path).astype(dt))
        times = []
        for _ in range(4):
            t = time.perf_counter()
            codes = (await scene({"image": image}))["scene_codes"].numpy()
            times.append(time.perf_counter() - t)
        rep["sceneSeconds"] = {"first": times[0], "warmMedian": float(np.median(times[1:]))}
        np.save(R / f"{stem}-codes.npy", codes)
        codes_in = rt.NDArray(codes.astype(dt))

        async def query(fn, chunk_points):
            o = await fn({"scene_codes": codes_in, "points": rt.NDArray(chunk_points.astype(dt))})
            return o["density"].numpy()[:, 0].astype(np.float32), o["color"].numpy().astype(np.float32)

        rep["gridSeconds"] = {}
        for chunk, fn in fields.items():
            dens = np.empty(len(points), np.float32)
            t = time.perf_counter()
            for s in range(0, len(points), chunk):
                block = points[s:s + chunk]
                n = len(block)
                if n < chunk:
                    block = np.concatenate([block, np.zeros((chunk - n, 3), np.float32)])
                dens[s:s + n] = (await query(fn, block))[0][:n]
            rep["gridSeconds"][f"field{chunk}"] = time.perf_counter() - t
            np.save(R / f"{stem}-grid-field{chunk}.npy", dens.astype(np.float16))
        slab = np.empty((a.res, a.res, a.res), np.float32)
        xs = rt.NDArray(lin.astype(dt))
        t = time.perf_counter()
        for k in range(0, a.res, a.slab):
            o = await grid_fn({"scene_codes": codes_in, "xs": xs, "zs": rt.NDArray(lin[k:k + a.slab].astype(dt))})
            slab[:, :, k:k + a.slab] = o["density"].numpy().astype(np.float32)
        rep["gridSeconds"]["slab"] = time.perf_counter() - t
        # exp(...) in fp16 overflows above 65,504 deep inside dense objects (fp32 references reach about 1e5); an inf
        # there makes marching cubes interpolate NaN vertices, so clamp to fp16's largest finite value.
        rep["infiniteDensities"] = int(np.isinf(slab).sum())
        slab = np.minimum(slab, np.finfo(np.float16).max)
        dens = slab.reshape(-1)
        np.save(R / f"{stem}-grid-slab.npy", dens.astype(np.float16))

        import mcubes
        import trimesh
        t = time.perf_counter()
        v, f = mcubes.marching_cubes(slab - a.threshold, 0.0)
        rep["marchingCubesSeconds"] = time.perf_counter() - t
        v = v[:, [2, 1, 0]] / (a.res - 1.0) * 2 * RADIUS - RADIUS          # upstream's axis swap and scaling
        f = f[:, [2, 1, 0]]
        t = time.perf_counter()
        colors = []
        for s in range(0, len(v), 65536):
            block = v[s:s + 65536].astype(np.float32)
            n = len(block)
            if n < 65536:
                block = np.concatenate([block, np.zeros((65536 - n, 3), np.float32)])
            colors.append((await query(fields[65536], block))[1][:n])
        rep["vertexColorSeconds"] = time.perf_counter() - t
        col = np.concatenate(colors) if colors else np.zeros((0, 3), np.float32)
        np.savez(R / f"{stem}-mesh.npz", vertices=v.astype(np.float32), faces=f.astype(np.int32), colors=col)
        trimesh.Trimesh(vertices=v, faces=f, vertex_colors=(col.clip(0, 1) * 255).astype(np.uint8)).export(
            R / f"{stem}-mesh.obj")
        best = min(rep["gridSeconds"].values())
        rep["totalSeconds"] = rep["sceneSeconds"]["warmMedian"] + best + rep["marchingCubesSeconds"] \
            + rep["vertexColorSeconds"]
        rep["mesh"] = {"vertices": int(len(v)), "faces": int(len(f))}
        report["images"][stem] = rep
        print(stem, json.dumps(rep), flush=True)
    (R / f"run-images-{a.res}-{a.tag}.json").write_text(json.dumps(report, indent=1))


asyncio.run(main())

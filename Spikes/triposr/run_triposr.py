"""Run the exported TripoSR stages on the GPU, compare with eager fp32 references, extract a mesh, write OBJ.

usage: USE_OS_COREAI=1 run_triposr.py <export-dir> <results-dir> [--res 256] [--tag fp16]
GPU part (under the lock): scene once (+ warm repeats), field over the res³ grid in 65536-point chunks.
CPU part (after the GPU work, still inside this process): marching cubes (PyMCubes), vertex colors via the GPU field,
OBJ via trimesh. The reference mesh comes from the eager density grid written by `eager_grid.py`, if present.
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
ap.add_argument("results")
ap.add_argument("--res", type=int, default=256)
ap.add_argument("--tag", default="fp16")
ap.add_argument("--threshold", type=float, default=25.0)
a = ap.parse_args()
E, R = Path(a.exports), Path(a.results)
R.mkdir(parents=True, exist_ok=True)
CH = 65536
RADIUS = 0.87
np16 = np.float16 if a.tag == "fp16" else np.float32
rep = {"tag": a.tag, "res": a.res}


def rel(x, y):
    return float(np.abs(x.astype(np.float64) - y).max() / (np.abs(y).max() + 1e-12))


async def main():
    opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    t0 = time.perf_counter()
    scene_m = await rt.AIModel.load(E / f"scene-{a.tag}.aimodel", opts)
    field_m = await rt.AIModel.load(E / f"field-{a.tag}.aimodel", opts)
    rep["loadSeconds"] = time.perf_counter() - t0
    scene, field = scene_m.load_function("main"), field_m.load_function("main")
    image = np.load(E / "input-image.npy").astype(np16)
    times = []
    for _ in range(4):
        t = time.perf_counter()
        codes = (await scene({"image": rt.NDArray(image)}))["scene_codes"].numpy()
        times.append(time.perf_counter() - t)
    rep["sceneSeconds"] = {"first": times[0], "warmMedian": float(np.median(times[1:]))}
    ref_codes = np.load(E / "ref-scene-codes.npy")
    rep["sceneCodesRelErr"] = rel(codes, ref_codes)
    rep["sceneCodesCosine"] = float(np.dot(codes.ravel().astype(np.float64), ref_codes.ravel())
                                    / (np.linalg.norm(codes.astype(np.float64)) * np.linalg.norm(ref_codes)))
    np.save(R / f"scene-codes-{a.tag}.npy", codes)
    codes_in = rt.NDArray(codes.astype(np16))

    async def query(pts):
        o = await field({"scene_codes": codes_in, "points": rt.NDArray(pts.astype(np16))})
        return o["density"].numpy()[:, 0].astype(np.float32), o["color"].numpy().astype(np.float32)

    pts = np.load(E / "ref-points.npy")
    d, c = await query(pts)
    rep["fieldDensityRelErr"] = rel(d, np.load(E / "ref-density.npy")[:, 0])
    rep["fieldColorMaxErr"] = float(np.abs(c - np.load(E / "ref-color.npy")).max())

    # Density grid exactly as TSR.extract_mesh builds it: linspace(0, 1, res)³ (ij) scaled to (-r, r).
    lin = np.linspace(0, 1, a.res, dtype=np.float32) * 2 * RADIUS - RADIUS
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], -1)
    dens = np.empty(len(grid), np.float32)
    t = time.perf_counter()
    for s in range(0, len(grid), CH):
        chunk = grid[s:s + CH]
        if len(chunk) < CH:
            chunk = np.concatenate([chunk, np.zeros((CH - len(chunk), 3), np.float32)])
        dd, _ = await query(chunk)
        dens[s:s + CH] = dd[:min(CH, len(grid) - s)]
    rep["gridSeconds"] = time.perf_counter() - t
    rep["gridPoints"] = len(grid)
    np.save(R / f"density-grid-{a.res}-{a.tag}.npy", dens)

    import mcubes
    import trimesh
    t = time.perf_counter()
    level = (dens - a.threshold).reshape(a.res, a.res, a.res)
    v, f = mcubes.marching_cubes(level, 0.0)
    rep["marchingCubesSeconds"] = time.perf_counter() - t
    v = v[:, [2, 1, 0]] / (a.res - 1.0) * 2 * RADIUS - RADIUS     # same axis swap and scaling as upstream
    t = time.perf_counter()
    colors = []
    for s in range(0, len(v), CH):
        chunk = v[s:s + CH].astype(np.float32)
        n = len(chunk)
        if n < CH:
            chunk = np.concatenate([chunk, np.zeros((CH - n, 3), np.float32)])
        colors.append((await query(chunk))[1][:n])
    rep["vertexColorSeconds"] = time.perf_counter() - t
    col = np.concatenate(colors) if colors else np.zeros((0, 3), np.float32)
    mesh = trimesh.Trimesh(vertices=v, faces=f[:, [2, 1, 0]], vertex_colors=(col * 255).astype(np.uint8))
    t = time.perf_counter()
    mesh.export(R / f"mesh-{a.res}-{a.tag}.obj")
    rep["objExportSeconds"] = time.perf_counter() - t
    rep["mesh"] = {"vertices": int(len(v)), "faces": int(len(f)), "watertight": bool(mesh.is_watertight),
                   "bounds": mesh.bounds.tolist()}
    refp = R / f"density-grid-{a.res}-eager.npy"
    if refp.exists():
        ref = np.load(refp)
        rep["gridDensityRelErr"] = rel(dens, ref)
        inside, inside_ref = dens > a.threshold, ref > a.threshold
        rep["occupancyIoU"] = float((inside & inside_ref).sum() / max(1, (inside | inside_ref).sum()))
        rv, rf = mcubes.marching_cubes((ref - a.threshold).reshape(a.res, a.res, a.res), 0.0)
        rep["refMesh"] = {"vertices": int(len(rv)), "faces": int(len(rf))}
    (R / f"run-{a.res}-{a.tag}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


asyncio.run(main())

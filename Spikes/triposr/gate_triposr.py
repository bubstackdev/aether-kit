"""TripoSR go/no-go gate: Core AI fp16 against eager fp32 on the same preprocessed images (CPU only).

usage: gate_triposr.py <references-dir> <results-dir> <out.json> [--usdz-log usdchecker.log] [--iou-only]

Per image, against the eager references from reference_images.py:
  - occupancy IoU at threshold 25 for each Core AI grid (field65536, field262144, slab) — every one must be ≥ 0.98;
  - symmetric Chamfer distance between the Core AI mesh (slab grid) and the eager mesh: the mean of both directions'
    nearest-neighbour distances over 100,000 surface samples each, relative to the eager mesh's bounding-box
    diagonal — must be < 1%;
  - time to a coloured mesh after load (scene warm + fastest grid + marching cubes + vertex colours) — must be < 3 s.
The USDZ criterion passes when the usdchecker log given says "Success!". Exit status 0 only if everything passes.
--iou-only records the IoUs and non-finite counts alone (for a run whose meshes can't be compared, such as the first
run's, whose miscompiled grid gave NaN vertices); it is not a gate.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

IOU_MIN, CHAMFER_MAX, SECONDS_MAX, THRESHOLD, SAMPLES = 0.98, 0.01, 3.0, 25.0, 100_000

ap = argparse.ArgumentParser()
ap.add_argument("references")
ap.add_argument("results")
ap.add_argument("out")
ap.add_argument("--usdz-log")
ap.add_argument("--iou-only", action="store_true")
a = ap.parse_args()
refs, res = Path(a.references), Path(a.results)
run = json.loads(next(res.glob("run-images-*.json")).read_text())


def surface_samples(vertices, faces, seed):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    points, _ = trimesh.sample.sample_surface(mesh, SAMPLES, seed=seed)
    return points


summary = {"criteria": {"iouMin": IOU_MIN, "chamferMaxOfDiagonal": CHAMFER_MAX, "secondsMax": SECONDS_MAX},
           "images": {}}
passed = True
for stem, rep in sorted(run["images"].items()):
    ref_grid = np.load(refs / f"{stem}-ref-grid.npy") > THRESHOLD
    ious, nonfinite = {}, {}
    for variant in ("field65536", "field262144", "slab"):
        values = np.load(res / f"{stem}-grid-{variant}.npy").astype(np.float32)
        nonfinite[variant] = int((~np.isfinite(values)).sum())
        grid = values > THRESHOLD
        ious[variant] = float((grid & ref_grid).sum() / max(1, (grid | ref_grid).sum()))
    if a.iou_only:
        summary["images"][stem] = {"iou": ious, "nonfinite": nonfinite}
        print(stem, json.dumps(summary["images"][stem]))
        continue
    ref = np.load(refs / f"{stem}-ref-mesh.npz")
    ours = np.load(res / f"{stem}-mesh.npz")
    p_ref = surface_samples(ref["vertices"], ref["faces"], 0)
    p_ours = surface_samples(ours["vertices"], ours["faces"], 1)
    d1, _ = cKDTree(p_ref).query(p_ours)
    d2, _ = cKDTree(p_ours).query(p_ref)
    diagonal = float(np.linalg.norm(ref["vertices"].max(0) - ref["vertices"].min(0)))
    chamfer = float((d1.mean() + d2.mean()) / 2 / diagonal)
    ok = min(ious.values()) >= IOU_MIN and chamfer < CHAMFER_MAX and rep["totalSeconds"] < SECONDS_MAX
    passed &= ok
    summary["images"][stem] = {"iou": ious, "chamferOfDiagonal": chamfer, "seconds": rep["totalSeconds"],
                               "gridSeconds": rep["gridSeconds"], "vertices": rep["mesh"]["vertices"],
                               "refVertices": int(len(ref["vertices"])), "pass": ok}
    print(stem, json.dumps(summary["images"][stem]))
if a.iou_only:
    summary["gate"] = "not judged (--iou-only)"
    Path(a.out).write_text(json.dumps(summary, indent=1))
    sys.exit(0)
if a.usdz_log:
    usdz_ok = "Success!" in Path(a.usdz_log).read_text()
    summary["usdzcheckerPass"] = usdz_ok
    passed &= usdz_ok
else:
    summary["usdzcheckerPass"] = None
    passed = False
summary["gate"] = "pass" if passed else "fail"
Path(a.out).write_text(json.dumps(summary, indent=1))
print("gate:", summary["gate"])
sys.exit(0 if passed else 1)

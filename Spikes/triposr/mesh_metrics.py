"""Metrics beside the gate (CPU only): how much of the Chamfer distance is sampling noise, how far Core AI's mesh vertices
sit from the eager mesh's, the scene codes' agreement, and the face winding.

usage: mesh_metrics.py <references-dir> <results-dir> <out.json>
  selfChamferBaseline   the gate's Chamfer between two different surface samplings of the same eager mesh, relative to
                        its bounding-box diagonal: the floor any mesh comparison with 100,000 samples can reach
  vertexToRefVertexP99, vertexToRefVertexMax
                        each Core AI vertex's distance to the nearest eager vertex, relative to the diagonal
  codesCosine, codesMaxRelErr
                        Core AI fp16 scene codes against eager fp32
  signedVolume, watertight
                        a positive signed volume on a closed mesh means outward-facing (counter-clockwise) faces
"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

refs, res, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])


def samples(mesh, seed):
    return trimesh.sample.sample_surface(trimesh.Trimesh(vertices=mesh["vertices"], faces=mesh["faces"], process=False),
                                         100_000, seed=seed)[0]


out = {}
for path in sorted(res.glob("*-mesh.npz")):
    stem = path.name[:-len("-mesh.npz")]
    ref, ours = np.load(refs / f"{stem}-ref-mesh.npz"), np.load(path)
    a, b = samples(ref, 0), samples(ref, 1)
    diagonal = float(np.linalg.norm(ref["vertices"].max(0) - ref["vertices"].min(0)))
    baseline = (cKDTree(a).query(b)[0].mean() + cKDTree(b).query(a)[0].mean()) / 2 / diagonal
    codes = np.load(res / f"{stem}-codes.npy").astype(np.float64)
    ref_codes = np.load(refs / f"{stem}-ref-codes.npy").astype(np.float64)
    distance, _ = cKDTree(ref["vertices"]).query(ours["vertices"])
    triangles = ours["vertices"][ours["faces"]]
    signed_volume = float(np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6)
    mesh = trimesh.Trimesh(vertices=ours["vertices"], faces=ours["faces"], process=False)
    out[stem] = {"selfChamferBaseline": float(baseline),
                 "codesCosine": float((codes * ref_codes).sum() / np.linalg.norm(codes) / np.linalg.norm(ref_codes)),
                 "codesMaxRelErr": float(np.abs(codes - ref_codes).max() / np.abs(ref_codes).max()),
                 "vertexToRefVertexP99": float(np.percentile(distance, 99) / diagonal),
                 "vertexToRefVertexMax": float(distance.max() / diagonal),
                 "signedVolume": signed_volume, "watertight": bool(mesh.is_watertight)}
    print(stem, json.dumps(out[stem]))
out_path.write_text(json.dumps(out, indent=1) + "\n")

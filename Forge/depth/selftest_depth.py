#!/usr/bin/env python3
"""Eager CPU self-test of the export graph, before any Core AI conversion.

    Forge/.venv/bin/python -I Forge/depth/selftest_depth.py [--reference <reference dir>] [--report <json>]

- With the export patches applied and the backbone in float32, `DepthGraph` equals upstream's own
  `DepthAnything3Net.forward` bit for bit (so the graph's omissions and the RoPE substitution change nothing).
- The exported precision (float16 backbone, float32 head), run eagerly, is within the gate's bars (check_depth.py).
- After the patches are undone, upstream's RoPE forward is restored and its frequency cache holds no traced tensors.
- `--sample-manifest`'s manifest names the `depth-estimator@1` roles (MANIFEST.md §1.3) and matches the graph's I/O.
With `--reference`, the cases are reference_depth.py's (and upstream's float32 output must reproduce them exactly);
otherwise one seeded synthetic image. Exits nonzero when a check fails. CPU only; a few seconds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import depth_graphs as dg  # noqa: E402
import export_depth  # noqa: E402
from check_depth import CONFIDENCE_BAR, DEPTH_BAR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--snapshot", type=Path, default=dg.DEFAULT_SNAPSHOT)
    ap.add_argument("--upstream", type=Path, default=dg.DEFAULT_UPSTREAM)
    args = ap.parse_args()
    dg.verify_sources(args.snapshot, args.upstream)
    torch.set_grad_enabled(False)
    checks: list[dict] = []

    def record(name: str, value: float, bar: float, passed: bool) -> None:
        checks.append({"check": name, "value": value, "bar": bar, "passed": bool(passed)})
        print(f"[selftest] {'PASS' if passed else 'FAIL'} {name}: {value} (bar {bar})", file=sys.stderr, flush=True)

    cases = []
    if args.reference:
        index = json.loads((args.reference / "references.json").read_text())
        for entry in index["cases"]:
            folder = args.reference / entry["name"]
            cases.append((entry["name"], dg.read_f32(folder / "input.f32", index["shapes"]["input"]),
                          dg.read_f32(folder / "depth.f32", index["shapes"]["depth"])))
    else:
        cases.append(("synthetic-3", dg.synthetic_tensor(3), None))

    upstream = dg.build_net(args.upstream, args.snapshot)
    _, _, rope = dg.load_upstream(args.upstream)
    original_forward = rope.RotaryPositionEmbedding2D.forward
    with dg.export_patches(args.upstream):
        exact = dg.DepthGraph(dg.build_net(args.upstream, args.snapshot), torch.float32).eval()
        mixed = dg.DepthGraph(dg.build_net(args.upstream, args.snapshot), torch.float16).eval()
        results = []
        for name, x, recorded in cases:
            image = torch.from_numpy(x)
            results.append((name, recorded, image, exact(image.half()), mixed(image.half())))
    restored = rope.RotaryPositionEmbedding2D.forward is original_forward
    record("rope.restored", float(restored), 1.0, restored)
    for name, recorded, image, (d32, c32), (d16, c16) in results:
        want_d, want_c = dg.upstream_depth(upstream, image)
        same = torch.equal(d32, want_d) and torch.equal(c32, want_c)
        record(f"{name}.float32.exact", float(same), 1.0, same)
        if recorded is not None:
            again = np.array_equal(want_d.numpy(), recorded)
            record(f"{name}.reference.reproduced", float(again), 1.0, again)
        e, c = dg.rel_l2(d16.numpy(), want_d.numpy()), dg.rel_l2(c16.numpy(), want_c.numpy())
        record(f"{name}.mixed.depth.relL2", round(e, 6), DEPTH_BAR, e <= DEPTH_BAR)
        record(f"{name}.mixed.confidence.relL2", round(c, 6), CONFIDENCE_BAR, c <= CONFIDENCE_BAR)
    caches = [len(m.frequency_cache) for m in upstream.modules() if isinstance(m, rope.RotaryPositionEmbedding2D)]
    clean = all(isinstance(t, torch.Tensor) and not type(t).__name__.startswith(("Fake", "Functional"))
                for m in upstream.modules() if isinstance(m, rope.RotaryPositionEmbedding2D)
                for pair in m.frequency_cache.values() for t in pair)
    record("rope.cache.eager", float(clean and bool(caches)), 1.0, clean and bool(caches))

    sample = export_depth.sample_manifest("macos")
    graph = sample["graphs"]["estimator"]
    roles = (sample["driver"] == "depth-estimator@1" and graph["inputs"] == {"image": "image"}
             and graph["outputs"] == {"depth": "depth", "confidence": "depth_conf"}
             and export_depth.INPUTS == ["image"] and export_depth.OUTPUTS == ["depth", "depth_conf"]
             and sample["postprocess"]["kind"] == "depth")
    record("manifest.roles", float(roles), 1.0, roles)
    passed = all(c["passed"] for c in checks)
    if args.report:
        args.report.write_text(json.dumps({"passed": passed, "checks": checks}, indent=2) + "\n")
    if not passed:
        raise SystemExit("self-test failed")


if __name__ == "__main__":
    main()

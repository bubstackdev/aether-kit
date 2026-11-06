#!/usr/bin/env python3
"""Eager CPU self-test of the export graph and the tiling rule, before any Core AI conversion.

    Forge/.venv/bin/python -I Forge/adcsr/selftest_adcsr.py [--reference <reference dir>] [--report <json>]

- `TileGraph` in float32 equals upstream test.py's network bit for bit on a tile (the wrapper changes nothing), and
  the float16 export keeps exactly the decoder's last two up blocks and head in float32 (`FP32_FROM`).
- The one-tile path is upstream's own inference: for a 128 x 128 image, the tile's output equals test.py's
  whole-image output exactly, and the stitched, colour-matched image is within float32 rounding of test.py's.
- Tiling known answers: tile starts for extents below, at and above the tile; the edge clamp; the feather ramp, whose
  overlapping halves sum to 1 at the nominal overlap; constant tiles blend to the constant, one tile to itself; the colour match uses
  torch's unbiased std; the quantization truncates.
- `--sample-manifest`'s manifest names the `image-to-image-tiled@1` roles (MANIFEST.md §1.3), the graph's I/O, the
  tiling rule and the `superRes` postprocess; the NOTICE ends with the license's Attachment A, verbatim (11 uses).
With `--reference`, upstream's float32 output must also reproduce the first case's recorded tile 0 exactly. Exits
nonzero when a check fails. CPU only; under a minute.
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
import adcsr_graphs as ag  # noqa: E402
import export_adcsr  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--snapshot", type=Path, default=ag.DEFAULT_SNAPSHOT)
    args = ap.parse_args()
    ag.verify_sources(args.snapshot)
    torch.set_grad_enabled(False)
    checks: list[dict] = []

    def record(name: str, value, bar, passed: bool) -> None:
        checks.append({"check": name, "value": value, "bar": bar, "passed": bool(passed)})
        print(f"[selftest] {'PASS' if passed else 'FAIL'} {name}: {value} (bar {bar})", file=sys.stderr, flush=True)

    # Tiling known answers (no model needed).
    starts = {e: ag.tile_starts(e) for e in (75, 128, 129, 141, 203, 240, 256, 320)}
    want = {75: [0], 128: [0], 129: [0, 1], 141: [0, 13], 203: [0, 75], 240: [0, 112], 256: [0, 112, 128],
            320: [0, 112, 192]}
    record("tiling.starts", starts == want, True, starts == want)
    grid = ag.tile_grid(256, 141)
    ok = grid == [(0, 0), (112, 0), (128, 0), (0, 13), (112, 13), (128, 13)]
    record("tiling.grid.rowMajor", ok, True, ok)
    lr = np.arange(3 * 5 * 7, dtype=np.float32).reshape(1, 3, 5, 7)
    crop = ag.crop_tile(lr, 0, 0)
    ok = (crop.shape == (1, 3, ag.TILE, ag.TILE) and np.array_equal(crop[:, :, :5, :7], lr)
          and np.array_equal(crop[:, :, 5:, :7], np.repeat(lr[:, :, 4:5], ag.TILE - 5, axis=2))
          and np.array_equal(crop[:, :, :, 7:], np.repeat(crop[:, :, :, 6:7], ag.TILE - 7, axis=3)))
    record("tiling.edgeClamp", ok, True, ok)
    ramp = ag.feather_ramp()
    e = ag.OVERLAP * ag.SCALE
    ok = (ramp.shape == (ag.SR_TILE,) and ramp[0] == np.float32(1) / np.float32(e + 1) and ramp[e] == 1
          and np.allclose(ramp[-e:] + ramp[:e], 1.0, atol=1e-6) and np.array_equal(ramp, ramp[::-1]))
    record("tiling.featherRamp", ok, True, ok)
    rng = np.random.default_rng(1)
    same = rng.standard_normal((1, 3, ag.SR_TILE, ag.SR_TILE)).astype(np.float32)
    w, h = 203, 141
    const = np.full((1, 3, ag.SR_TILE, ag.SR_TILE), 0.25, dtype=np.float32)
    out = ag.stitch([(x0, y0, const) for x0, y0 in ag.tile_grid(w, h)], w, h)
    ok = out.shape == (1, 3, h * ag.SCALE, w * ag.SCALE) and np.allclose(out, 0.25, atol=1e-6)
    record("tiling.blend.constant", float(np.abs(out - 0.25).max()), 1e-6, ok)
    single = ag.stitch([(0, 0, same)], 100, 75)
    ok = np.allclose(single, same[:, :, :300, :400], atol=1e-6)
    record("tiling.blend.singleTile", float(np.abs(single - same[:, :, :300, :400]).max()), 1e-6, ok)
    sr = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)
    small = rng.standard_normal((1, 3, 2, 2)).astype(np.float32)
    cm = ag.color_match(sr, small)
    ok = (np.allclose(cm.mean(axis=(2, 3)), small.mean(axis=(2, 3)), atol=1e-5)
          and np.allclose(cm.std(axis=(2, 3), ddof=1), small.std(axis=(2, 3), ddof=1), atol=1e-5))
    record("colorMatch.unbiased", ok, True, ok)
    unit = np.array([0.0, 0.5, 0.999, 1.0], dtype=np.float32).reshape(1, 1, 1, 4).repeat(3, axis=1)
    ok = ag.to_uint8(unit)[0, :, 0].tolist() == [0, 127, 254, 255]
    record("quantize.truncate", ok, True, ok)
    u8 = np.array([[[0, 128, 255]]], dtype=np.uint8)
    ok = np.allclose(ag.lr_tensor(u8).ravel(), [-1.0, 128 / 255 * 2 - 1, 1.0], atol=1e-7)
    record("preprocess.range", ok, True, ok)

    # The graph against upstream.
    upstream = ag.build_upstream(args.snapshot)
    params = ag.parameter_count(upstream)
    record("upstream.parameters", params, 456073683, params == 456073683)
    graph = ag.TileGraph(ag.build_upstream(args.snapshot), torch.float32).eval()
    case = ag.CASES[0]
    lr = ag.lr_tensor(ag.case_lr(case))
    want = upstream(torch.from_numpy(lr))
    got = graph(torch.from_numpy(lr))
    record("graph.float32.exact", bool(torch.equal(got, want)), True, torch.equal(got, want))
    mixed = ag.TileGraph(ag.build_upstream(args.snapshot), torch.float16)
    split = ag.precision_split(upstream, torch.float16)
    ok = (len(mixed.low) == ag.FP32_FROM and len(mixed.low) + len(mixed.high) == len(upstream)
          and {p.dtype for p in mixed.low.parameters()} == {torch.float16}
          and {p.dtype for p in mixed.high.parameters()} == {torch.float32}
          and split["float32"] == ["up2", "up3", "conv_norm_out", "conv_act", "conv_out"]
          and split["lowParameters"] + split["float32Parameters"] == params)
    record("graph.float16.split", split["float32"], ag.STAGES[ag.FP32_FROM:], ok)
    del mixed
    full_raw, full = ag.upstream_full(upstream, lr)
    raw, tiles = ag.upscale_tiled(lambda t: upstream(torch.from_numpy(t)).numpy(), lr)
    exact = len(tiles) == 1 and np.array_equal(tiles[0][2], full_raw)
    record("oneTile.equalsUpstream", exact, True, exact)
    p = ag.psnr(ag.postprocess(raw, lr), full)
    record("oneTile.final.psnr", round(p, 2), 120.0, p >= 120.0)
    if args.reference:
        index = json.loads((args.reference / "references.json").read_text())
        first = index["cases"][0]["name"]
        folder = args.reference / first
        x = ag.read_f32(folder / "tiles/0.lr.f32", index["tile"])
        again = np.array_equal(upstream(torch.from_numpy(x)).numpy(), ag.read_f32(folder / "tiles/0.sr.f32",
                                                                                  index["srTile"]))
        record(f"{first}.reference.reproduced", again, True, again)

    sample = export_adcsr.sample_manifest("macos")
    g = sample["graphs"]["upscaler"]
    t = sample["preprocess"]["image"]["tiling"]
    roles = (sample["driver"] == "image-to-image-tiled@1" and g["inputs"] == {"image": "lr"}
             and g["outputs"] == {"image": "sr"} and export_adcsr.INPUTS == ["lr"] and export_adcsr.OUTPUTS == ["sr"]
             and sample["postprocess"]["kind"] == "superRes"
             and sample["postprocess"]["sr"]["range"] == [-1.0, 1.0] and sample["postprocess"]["sr"]["colorMatch"] == "host"
             and t["tile"] == [ag.TILE, ag.TILE] and t["overlap"] == max(4, ag.TILE // 8) and t["scale"] == ag.SCALE
             and sample["preprocess"]["image"]["mean"] == [0.5] * 3 and sample["preprocess"]["image"]["std"] == [0.5] * 3)
    record("manifest.roles", roles, True, roles)
    license_text = (ag.DEFAULT_BASE_LICENSE / "LICENSE-MODEL").read_text()
    text = export_adcsr.notice("float16", license_text)
    restrictions = license_text[license_text.rindex("Attachment A"):].strip()
    bullets = [line for line in restrictions.splitlines() if line.startswith("- ")]
    ok = (text.rstrip().endswith(restrictions) and len(bullets) == 11 and all(line in text for line in bullets)
          and "6. The Output You Generate" not in text and "NOTICE TO DEPLOYERS AND USERS" in text)
    record("notice.attachmentA.verbatim", len(bullets), 11, ok)
    passed = all(c["passed"] for c in checks)
    if args.report:
        args.report.write_text(json.dumps({"passed": passed, "checks": checks}, indent=2) + "\n")
    if not passed:
        raise SystemExit("self-test failed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The SD-Turbo graph parity check: each exported Core AI graph (GPU) against the CPU float32 Diffusers reference.

    Forge/.venv/bin/python -I Forge/sdturbo/check_sdturbo.py --bundle <bundle dir> --reference <reference dir> \
        --report <new check-report.json>

Teacher-forced, per graph, with the reference's own inputs fed in (the 0.0.x spike's bars):
- text encoder: cosine >= 0.99999 against the reference's text embeddings;
- U-Net: every step's epsilon cosine >= 0.9999;
- VAE decoder: decoding the reference's final latents, RMSE <= 0.01 on [-1, 1].
Host recipe (exact): the bundle's schedule tables equal the reference's, and Kit's seeded noise equals its noise. The
whole pipeline from the seed is the Swift gate's (`latentDiffusionMatchesDiffusers`). Exits nonzero when a bar fails.
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
import sdturbo_graphs as sg  # noqa: E402

TEXT_BAR, EPSILON_BAR, DECODER_RMSE_BAR = 0.99999, 0.9999, 0.01


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


class Graph:
    def __init__(self, path: Path, function: str) -> None:
        import coreai.runtime as rt
        self.rt = rt
        options = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
        self.model = asyncio.run(rt.AIModel.load(path, options))
        self.function = self.model.load_function(function)

    def __call__(self, **feeds: np.ndarray) -> dict[str, np.ndarray]:
        rt = self.rt

        async def go():
            out = await self.function({k: rt.NDArray(np.ascontiguousarray(v)) for k, v in feeds.items()}, {})
            return {k: v.numpy().copy() for k, v in out.items()}
        return asyncio.run(go())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise SystemExit("report already exists; choose a new path")
    manifest = json.loads((args.bundle / ".aether/aether.json").read_text())
    ref = json.loads((args.reference / "reference.json").read_text())
    graphs = {role: (args.bundle / g["file"].split(":", 1)[1], g["function"]) for role, g in manifest["graphs"].items()}
    checks: list[dict] = []

    def record(name: str, value: float, bar: float, passed: bool, **extra) -> None:
        checks.append({"check": name, "value": value, "bar": bar, "passed": bool(passed), **extra})
        print(f"[check] {'PASS' if passed else 'FAIL'} {name}: {value} (bar {bar})", file=sys.stderr, flush=True)

    for case in ref["cases"]:
        table = manifest["schedule"]["tables"][str(case["steps"])]
        record(f"{case['id']}.schedule.exact", float(table == case["schedule"]), 1.0, table == case["schedule"])
        noise = sg.seeded_noise(case["seed"], int(np.prod(sg.latent_shape(ref["size"]))))
        same = np.array_equal(noise, np.load(args.reference / case["id"] / "noise.npy").ravel())
        record(f"{case['id']}.noise.exact", float(same), 1.0, same)

    started = time.monotonic()
    text = Graph(*graphs["textEncoder"])
    for case in ref["cases"]:
        got = text(input_ids=np.array([case["ids"]], dtype=np.int32))["text_embeds"]
        c = cosine(got, np.load(args.reference / case["id"] / "text_embeds.npy"))
        record(f"{case['id']}.text.cosine", round(c, 9), TEXT_BAR, c >= TEXT_BAR)
    del text
    unet = Graph(*graphs["denoiser"])
    for case in ref["cases"]:
        embeds = np.load(args.reference / case["id"] / "text_embeds.npy")
        for k in range(case["steps"]):
            step = np.load(args.reference / case["id"] / f"step{k}.npz")
            got = unet(sample=step["sample"], timestep=np.array([step["timestep"]], dtype=np.float32),
                       text_embeds=embeds)["epsilon"]
            c = cosine(got, step["epsilon"])
            record(f"{case['id']}.step{k}.epsilon.cosine", round(c, 9), EPSILON_BAR, c >= EPSILON_BAR,
                   rmse=rmse(got, step["epsilon"]))
    del unet
    vae = Graph(*graphs["vaeDecoder"])
    for case in ref["cases"]:
        got = vae(latents=np.load(args.reference / case["id"] / "final.npy"))["image"]
        e = rmse(got, np.load(args.reference / case["id"] / "image.npy"))
        record(f"{case['id']}.decoder.rmse", e, DECODER_RMSE_BAR, e <= DECODER_RMSE_BAR)
    del vae
    passed = all(c["passed"] for c in checks)
    args.report.write_text(json.dumps({"passed": passed, "seconds": round(time.monotonic() - started, 1),
                                       "bundle": str(args.bundle), "reference": str(args.reference),
                                       "checks": checks}, indent=2) + "\n")
    if not passed:
        raise SystemExit("parity check failed")


if __name__ == "__main__":
    main()

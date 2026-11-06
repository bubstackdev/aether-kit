#!/usr/bin/env python3
"""The FLUX.2 [klein] parity gate: the exported Core AI graphs (GPU) against the CPU float32 Diffusers reference.

    Forge/.venv/bin/python -I Forge/flux2/check_flux2.py --bundle <bundle dir> --reference <reference dir> \
        --report <new check-report.json>

Bars (the ones 0.0.x qualified this recipe at):
- host recipe: Kit's prompt rule (`prompt_ids`) equals the pipeline's padded ids and mask exactly; Kit's seeded noise
  and schedule (empirical μ) equal the reference's noise, sigmas and timestep inputs exactly;
- text: every real conditioning row has cosine >= 0.999 against the reference (textEncoder graph); the padding rows'
  minimum is recorded, not gated (see below);
- denoiser, teacher-forced: for every step and segment, each output (carries, cond, velocity) cosine >= 0.999 against
  the reference's own tensors at that boundary, with the reference's inputs fed in;
- VAE decoder: decoding the reference's final latents, PSNR >= 35 dB against the reference image (on [0, 1]);
- VAE encoder: encoding the reference image, latent cosine >= 0.999 against `_encode_vae_image`;
- final image, closed loop from the reference's conditioning rows (4 x 3 segments -> host float32 Euler -> graph VAE,
  from the seed): SSIM >= 0.94 against the reference image (PSNR reported);
- whole pipeline (the same loop from the graph's own conditioning rows): SSIM recorded, not gated. All 512 rows
  condition the denoiser, and the padding rows (small norms, after <|endoftext|>) move under any precision change:
  0.0.x measured 0.81-0.999 for this path and 0.865-0.999 for Diffusers' own bf16 pipeline against float32, so the
  path measures the padding rows' sensitivity more than conversion error (0.0.x gated it the same way).
One gate: exits nonzero when any bar fails. Refuses to start while Swift tests run (xctest), unless `--wait`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import flux2_graphs as fg  # noqa: E402

COSINE_BAR = 0.999
PSNR_BAR = 35.0
SSIM_BAR = 0.94


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def row_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float64), b.astype(np.float64)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """On [0, 1] images (inputs are decoder outputs in [-1, 1])."""
    x, y = (a.astype(np.float64) + 1) / 2, (b.astype(np.float64) + 1) / 2
    mse = float(np.mean((x - y) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM (Wang et al. 2004: Gaussian window sigma 1.5, 11 taps, K1 0.01, K2 0.03) over RGB on [0, 1]."""
    from scipy.ndimage import gaussian_filter
    x, y = (a.astype(np.float64) + 1) / 2, (b.astype(np.float64) + 1) / 2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    blur = lambda z: gaussian_filter(z, 1.5, truncate=3.5, mode="reflect")  # noqa: E731
    values = []
    for c in range(x.shape[0]):
        mx, my = blur(x[c]), blur(y[c])
        sxx, syy, sxy = blur(x[c] * x[c]) - mx * mx, blur(y[c] * y[c]) - my * my, blur(x[c] * y[c]) - mx * my
        values.append(((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2)))
    return float(np.mean(values))


class Graph:
    def __init__(self, path: Path, function: str) -> None:
        import coreai.runtime as rt
        self.rt = rt
        options = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
        t0 = time.monotonic()
        self.model = asyncio.run(rt.AIModel.load(path, options))
        self.function = self.model.load_function(function)
        self.load_seconds = round(time.monotonic() - t0, 1)

    def __call__(self, **feeds: np.ndarray) -> dict[str, np.ndarray]:
        rt = self.rt

        async def go():
            out = await self.function({k: rt.NDArray(np.ascontiguousarray(v)) for k, v in feeds.items()}, {})
            return {k: v.numpy().copy() for k, v in out.items()}
        return asyncio.run(go())

    def close(self) -> None:
        del self.function, self.model


def ancestors() -> set[int]:
    pids, pid = set(), os.getpid()
    while pid > 1:
        pids.add(pid)
        out = subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        pid = int(out) if out else 1
    return pids


def wait_for_swift_tests(wait: bool) -> None:
    mine = ancestors()          # a shell that merely mentions the pattern is not a test run
    while True:
        lines = subprocess.run(["pgrep", "-fl", "xctest|swiftpm-testing-helper"], capture_output=True,
                               text=True).stdout.splitlines()
        busy = "\n".join(line for line in lines if line.split()[0].isdigit() and int(line.split()[0]) not in mine)
        if not busy.strip():
            return
        if not wait:
            raise SystemExit(f"Swift test processes are running; not loading GPU graphs:\n{busy}")
        print(f"[check] waiting for: {busy.strip()}", file=sys.stderr, flush=True)
        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--wait", action="store_true", help="wait for running Swift tests instead of refusing")
    args = parser.parse_args()
    if args.report.exists():
        raise SystemExit("report already exists; choose a new path")
    started = time.monotonic()
    manifest = json.loads((args.bundle / "aether.json").read_text())
    ref = json.loads((args.reference / "reference.json").read_text())
    tokens = json.loads((args.reference / "tokens.json").read_text())
    graphs = manifest["graphs"]
    files = {role: args.bundle / g["file"].split(":", 1)[1] for role, g in graphs.items()}
    load = lambda role: Graph(files[role], graphs[role]["function"])  # noqa: E731
    checks: list[dict] = []

    def record(name: str, value: float, bar: float, passed: bool | None = None, **extra) -> None:
        ok = (value >= bar) if passed is None else passed
        checks.append({"check": name, "value": value, "bar": bar, "passed": bool(ok), **extra})
        print(f"[check] {'PASS' if ok else 'FAIL'} {name}: {value} (bar {bar})", file=sys.stderr, flush=True)

    # Host recipe: tokens, noise, schedule (exact).
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bundle / "tokenizer")
    ids, mask = fg.prompt_ids(tokenizer, ref["prompt"])
    exact = ids == tokens["padded"] and mask == tokens["mask"]
    record("tokens.exact", float(exact), 1.0, exact, real=sum(mask))
    rows = columns = ref["height"] // fg.PIXELS_PER_TOKEN
    n = rows * columns
    noise = fg.seeded_noise(ref["seed"], n * fg.LATENT_CHANNELS).reshape(n, fg.LATENT_CHANNELS)
    exact = np.array_equal(noise, np.load(args.reference / "noise.npy"))
    record("noise.exact", float(exact), 1.0, exact)
    sigmas, timesteps, mu = fg.schedule(ref["steps"], n)
    exact = sigmas == ref["sigmas"] and timesteps == ref["timesteps"]
    record("schedule.exact", float(exact), 1.0, exact, mu=mu)
    rope_text = fg.text_rotary(fg.MAX_TOKENS).numpy()
    rope_image = fg.image_rotary(rows, columns).numpy()
    ropes = dict(rope_text=rope_text, rope_image=rope_image)

    wait_for_swift_tests(args.wait)
    timings = {}

    # Text encoder.
    text = load("textEncoder")
    timings["loadText"] = text.load_seconds
    t0 = time.monotonic()
    out = text(input_ids=np.array([ids], dtype=np.int32), position_ids=np.arange(fg.MAX_TOKENS, dtype=np.int32)[None],
               attention_mask=np.array([mask], dtype=np.int32))
    timings["text"] = round(time.monotonic() - t0, 2)
    text_embeds = out["text_embeds"]
    ref_text = np.load(args.reference / "text_embeds.npy")
    text.close()
    if text_embeds.shape != ref_text.shape:
        raise SystemExit(f"text_embeds {text_embeds.shape} vs reference {ref_text.shape}")
    per_row = row_cosines(text_embeds.astype(np.float32), ref_text)
    real = sum(mask)
    record("text.realRowCosineMin", round(float(per_row[:real].min()), 6), COSINE_BAR,
           paddingRowCosineMin=round(float(per_row[real:].min()), 6) if real < per_row.size else None,
           cosine=round(cosine(text_embeds, ref_text), 7), rows=real, finite=bool(np.isfinite(text_embeds).all()))

    # Denoiser segments: teacher-forced per step and segment, then the closed loop.
    segments = [load(f"denoiser{i}") for i in range(len(fg.SEGMENTS))]
    timings["loadDenoisers"] = round(sum(s.load_seconds for s in segments), 1)
    teacher = []
    for s in range(ref["steps"]):
        r = dict(np.load(args.reference / f"step{s}.npz"))
        feeds = [
            dict(latents=r["latents_in"].astype(np.float16), text_embeds=ref_text.astype(np.float16),
                 sigma=r["timestep"].astype(np.float32), **ropes),
            dict(image_1=r["image_1"], text_1=r["text_1"], cond=r["cond"], **ropes),
            dict(image_2=r["image_2"], text_2=r["text_2"], cond=r["cond"], **ropes),
        ]
        expected = [dict(image_1=r["image_1"], text_1=r["text_1"], cond=r["cond"]),
                    dict(image_2=r["image_2"], text_2=r["text_2"]), dict(velocity=r["velocity"])]
        row = {"step": s}
        for i, (segment, feed, want) in enumerate(zip(segments, feeds, expected)):
            got = segment(**feed)
            for name, value in want.items():
                c = cosine(got[name], value)
                row[f"denoiser{i}.{name}"] = round(c, 6)
                checks.append({"check": f"teacher.step{s}.denoiser{i}.{name}", "value": round(c, 6), "bar": COSINE_BAR,
                               "passed": c >= COSINE_BAR})
        teacher.append(row)
        print(f"[check] teacher step {s}: " + json.dumps({k: v for k, v in row.items() if k != "step"}), file=sys.stderr,
              flush=True)

    step_seconds = []

    def closed_loop(conditioning: np.ndarray) -> np.ndarray:
        latents = noise.copy()
        for s in range(ref["steps"]):
            t0 = time.monotonic()
            sigma = np.array([timesteps[s]], dtype=np.float32)
            o0 = segments[0](latents=latents.astype(np.float16), text_embeds=conditioning.astype(np.float16),
                             sigma=sigma, **ropes)
            o1 = segments[1](image_1=o0["image_1"], text_1=o0["text_1"], cond=o0["cond"], **ropes)
            o2 = segments[2](image_2=o1["image_2"], text_2=o1["text_2"], cond=o0["cond"], **ropes)
            delta = np.float32(sigmas[s + 1]) - np.float32(sigmas[s])
            latents = (latents + (delta * o2["velocity"].astype(np.float32)).astype(np.float32)).astype(np.float32)
            step_seconds.append(round(time.monotonic() - t0, 3))
        return latents

    latents = closed_loop(ref_text)
    whole = closed_loop(text_embeds)
    for segment in segments:
        segment.close()
    ref_final = np.load(args.reference / f"step{ref['steps'] - 1}.npz")["latents_out"]
    closed_latent_cosine = round(cosine(latents, ref_final), 6)

    # VAE decoder and encoder.
    vae = load("vaeDecoder")
    ref_image = np.load(args.reference / "image.npy")
    decoded = vae(latents=ref_final.reshape(rows, columns, fg.LATENT_CHANNELS).astype(np.float16))["image"]
    record("vae.psnrDB", round(psnr(decoded, ref_image), 3), PSNR_BAR)
    t0 = time.monotonic()
    image = vae(latents=latents.reshape(rows, columns, fg.LATENT_CHANNELS).astype(np.float16))["image"]
    timings["decode"] = round(time.monotonic() - t0, 2)
    whole_image = vae(latents=whole.reshape(rows, columns, fg.LATENT_CHANNELS).astype(np.float16))["image"]
    vae.close()
    encoder = load("vaeEncoder")
    encoded = encoder(reference_pixels=ref_image.astype(np.float32))["reference_latents"]
    encoder.close()
    record("vaeEncoder.latentCosine", round(cosine(encoded, np.load(args.reference / "encoder_latents.npy")), 6),
           COSINE_BAR, shape=list(encoded.shape))
    record("image.ssim", round(ssim(image, ref_image), 6), SSIM_BAR, psnrDB=round(psnr(image, ref_image), 3),
           finalLatentCosine=closed_latent_cosine)
    whole_ssim = round(ssim(whole_image, ref_image), 6)
    whole_record = {"ssim": whole_ssim, "psnrDB": round(psnr(whole_image, ref_image), 3),
                    "finalLatentCosine": round(cosine(whole, ref_final), 6), "gated": False}
    print(f"[check] RECORDED wholePipeline: {whole_record}", file=sys.stderr, flush=True)
    from PIL import Image

    def save_png(pixels: np.ndarray, path: Path) -> None:
        Image.fromarray(np.clip(np.round((pixels.transpose(1, 2, 0) / 2 + 0.5) * 255), 0, 255).astype(np.uint8)).save(path)
    png = args.report.with_suffix(".png")
    save_png(image, png)
    save_png(whole_image, args.report.with_name(args.report.stem + "-whole.png"))

    teacher_checks = [c for c in checks if c["check"].startswith("teacher.")]
    velocity = [c["value"] for c in teacher_checks if c["check"].endswith(".velocity")]
    value = lambda name, key="value": next(c[key] for c in checks if c["check"] == name)  # noqa: E731
    summary = {
        "bundle": str(args.bundle), "reference": str(args.reference),
        "passed": all(c["passed"] for c in checks),
        "counts": {"checks": len(checks), "failed": sum(not c["passed"] for c in checks)},
        "minima": {"textRealRowCosine": value("text.realRowCosineMin"),
                   "textPaddingRowCosine": value("text.realRowCosineMin", "paddingRowCosineMin"),
                   "teacherCosine": min(c["value"] for c in teacher_checks),
                   "teacherVelocityCosine": min(velocity),
                   "vaePSNRdB": value("vae.psnrDB"), "vaeEncoderLatentCosine": value("vaeEncoder.latentCosine"),
                   "imageSSIM": value("image.ssim"), "imagePSNRdB": value("image.ssim", "psnrDB"),
                   "closedLoopLatentCosine": closed_latent_cosine},
        "wholePipeline": whole_record,
        "timingsSeconds": {**timings, "closedLoopSteps": step_seconds, "total": round(time.monotonic() - started, 1)},
        "hostPeakRSSGB": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),
        "image": str(png),
    }
    args.report.write_text(json.dumps({**summary, "teacher": teacher, "checks": checks}, indent=2) + "\n")
    print(json.dumps(summary, indent=1))
    if not summary["passed"]:
        raise SystemExit("FLUX.2 parity gate failed")


if __name__ == "__main__":
    main()

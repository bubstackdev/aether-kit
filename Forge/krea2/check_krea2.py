#!/usr/bin/env python3
"""The Krea 2 parity gate: the exported Core AI graphs (GPU) against the CPU float32 Diffusers reference.

    Forge/.venv/bin/python -I Forge/krea2/check_krea2.py --bundle <bundle dir> --reference <reference dir> \
        --report <new check-report.json>

Bars (FLUX.2's template):
- tokens: Kit's prompt rule (`prompt_ids`) equals the pipeline's real token ids exactly; Kit's seeded noise and
  schedule equal the reference's noise and sigmas exactly;
- text: every real conditioning row's cosine >= 0.999 against the reference (textEncoder graph);
- denoiser, teacher-forced: for every step and segment, each output (carries, cond, velocity) cosine >= 0.999 against
  the reference's own tensors at that boundary, with the reference's inputs fed in;
- VAE: decoding the reference's final latents, PSNR >= 35 dB against the reference image (on [0, 1]);
- final image, closed loop (graph text -> 8 x 4 segments -> host float32 Euler -> graph VAE, from the seed): SSIM >=
  0.94 against the reference image (PSNR reported);
- refine, when the bundle has `vaeEncoder` and the reference recorded one: the encoder graph on `image.npy` against the
  source VAE's latents, cosine >= 0.999; then closed loop from the graph's latents (Kit's seeded refine noise, exact;
  `(1 - sigma) * x0 + sigma * noise` on the host; the remaining steps; graph VAE): SSIM >= 0.94 against
  `refine_image.npy`. A bundle or reference without them skips the refine checks and says so.
- references, when the reference was made with `--references` (and the bundle has `vision`): per case, Kit's host
  prompt (`reference_prompt`: ids, extension ids, M-RoPE) exactly; the vision graph per image against the oracle's
  merged tokens and DeepStack, cosine >= 0.999; `encodeWithImages` fed the graph's own vision outputs against the
  oracle conditioning, mean row cosine >= 0.999 and worst row >= 0.99 (the 0.0.x bars). The closed loop then runs from
  the chosen case's graph conditioning, and the teacher-forced steps from its oracle conditioning.
- masked denoisers (a `textMask` input): every denoiser call takes the conditioning padded with zero rows to the
  fixed length (1116) and `text_mask` (1 real, 0 padding);
  text carries are compared on the real rows. The closed-loop image is also saved as `<report>.npy` (float32 [3, H, W]
  on [-1, 1]) so two bundles' images can be compared (`--compare-image`: PSNR and SSIM against another check's .npy).
One gate: exits nonzero when any bar fails. Refuses to start while the M2 Swift gates run (xctest).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import krea2_graphs as kg  # noqa: E402

COSINE_BAR = 0.999
REFERENCE_WORST_ROW_BAR = 0.99
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


class Segments:
    """The four denoiser segments at one function: exact length (no mask), or a masked function's fixed text length,
    with the conditioning and text carries padded with zero rows and `text_mask` added to every call."""

    def __init__(self, manifest: dict, files: dict, rows: int) -> None:
        g = manifest["graphs"]["denoiser0"]
        phases = g.get("functions") or {"step": g["function"]}
        self.rows, self.length = rows, None
        if "textMask" not in g["inputs"]:
            self.function = phases["step"]
        else:
            if rows > kg.MASK_TEXT_ROWS:
                raise SystemExit(f"{rows} conditioning rows exceed the denoisers' {kg.MASK_TEXT_ROWS}")
            self.length = kg.MASK_TEXT_ROWS
            self.function = phases["step"]
            self.mask = kg.text_mask(rows, self.length).numpy()
        self.graphs = [Graph(files[f"denoiser{i}"], self.function) for i in range(4)]
        self.load_seconds = round(sum(s.load_seconds for s in self.graphs), 1)

    def pad(self, x: np.ndarray) -> np.ndarray:
        if self.length is None:
            return x
        return np.concatenate([x, np.zeros((self.length - x.shape[0], *x.shape[1:]), dtype=x.dtype)])

    def __call__(self, index: int, **feeds: np.ndarray) -> dict[str, np.ndarray]:
        for name in list(feeds):
            if name == "text_embeds" or name.startswith("text_"):
                feeds[name] = self.pad(feeds[name])
        if self.length is not None:
            feeds["text_mask"] = self.mask
        out = self.graphs[index](**feeds)
        return {k: (v[:self.rows] if k.startswith("text_") else v) for k, v in out.items()}

    def close(self) -> None:
        for graph in self.graphs:
            graph.close()


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
    import os
    pids, pid = set(), os.getpid()
    while pid > 1:
        pids.add(pid)
        out = subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        pid = int(out) if out else 1
    return pids


def wait_for_m2_gates(wait: bool) -> None:
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


def reference_checks(args, manifest, files, tokenizer, references, record, timings) -> np.ndarray:
    """Every recorded case through Kit's host prompt, the vision graph and `encodeWithImages`; returns the chosen
    case's graph conditioning."""
    block = manifest["krea2"]["references"]
    text_graph = manifest["graphs"]["textEncoder"]
    vision = Graph(files["vision"], manifest["graphs"]["vision"]["function"])
    text = Graph(files["textEncoder"], text_graph["functions"][block["function"]])
    timings["loadVision"], timings["loadReferenceText"] = vision.load_seconds, text.load_seconds
    capacity, slots = kg.REFERENCE_PATCH_CAPACITY, kg.REFERENCE_MAX_IMAGE_TOKENS
    chosen = None
    for cid, case in references["cases"].items():
        d = args.reference / "cases" / cid
        grids = [tuple(g) for g in json.loads((d / "grids.json").read_text())]
        prompt = kg.reference_prompt(tokenizer, case["prompt"], grids)
        host = (prompt["ids"] == np.load(d / "ids.npy").tolist()
                and prompt["graphIds"] == np.load(d / "graph_ids.npy").tolist()
                and np.array_equal(np.array(prompt["rope"], dtype=np.int32), np.load(d / "rope.npy")))
        record(f"references.{cid}.host.exact", float(host), 1.0, host, rows=case["rows"])
        patches, oracle_image = np.load(d / "patches.npy"), np.load(d / "image_embeds.npy")
        oracle_deep = np.load(d / "deepstack_embeds.npy")
        width = oracle_image.shape[1]
        image = np.zeros((slots, width), dtype=np.float32)
        deep = np.zeros((oracle_deep.shape[0], slots, width), dtype=np.float32)
        start, slot = 0, 0
        t0 = time.monotonic()
        for t, h, w in grids:
            n = t * h * w
            feed = np.zeros((capacity, patches.shape[1]), dtype=np.float16)
            feed[:n] = patches[start:start + n]
            out = vision(patches=feed, grid=np.array([h, w], dtype=np.int32))
            k = n // 4
            image[slot:slot + k] = out["image_embeds"][:k]
            deep[:, slot:slot + k] = out["deepstack_embeds"][:, :k]
            start, slot = start + n, slot + k
        timings[f"vision.{cid}"] = round(time.monotonic() - t0, 2)
        record(f"references.{cid}.vision.cosine", round(cosine(image[:slot], oracle_image), 6), COSINE_BAR,
               rowCosineMin=round(float(row_cosines(image[:slot], oracle_image).min()), 6))
        record(f"references.{cid}.deepstack.cosine", round(cosine(deep[:, :slot], oracle_deep), 6), COSINE_BAR)
        feeds = dict(input_ids=np.array([prompt["graphIds"]], dtype=np.int32),
                     rope_positions=np.array(prompt["rope"], dtype=np.int32), image_embeds=image,
                     deepstack_embeds=deep)
        t0 = time.monotonic()
        got = text(**feeds)["text_embeds"].astype(np.float32)
        timings[f"referenceText.{cid}"] = round(time.monotonic() - t0, 2)
        want = np.load(d / "conditioning.npy")
        if got.shape != want.shape:
            raise SystemExit(f"{cid}: conditioning {got.shape} vs oracle {want.shape}")
        rows = row_cosines(got, want)
        image_rows = np.array(prompt["ids"][kg.DROP_ROWS:]) == kg.IMAGE_PAD_ID
        record(f"references.{cid}.text.meanRowCosine", round(float(rows.mean()), 6), COSINE_BAR,
               worstRow=round(float(rows.min()), 6), worstRowIndex=int(rows.argmin()),
               worstImageRow=round(float(rows[image_rows].min()), 6), worstTextRow=round(float(rows[~image_rows].min()), 6),
               cosine=round(cosine(got, want), 7), rows=int(rows.size))
        record(f"references.{cid}.text.worstRowCosine", round(float(rows.min()), 6), REFERENCE_WORST_ROW_BAR)
        if cid == references["case"]:
            chosen = got.astype(np.float16)
    vision.close()
    text.close()
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--wait", action="store_true", help="wait for running Swift tests instead of refusing")
    parser.add_argument("--compare-image", type=Path,
                        help="another check's closed-loop image (.npy): PSNR and SSIM against this run's, reported")
    args = parser.parse_args()
    if args.report.exists():
        raise SystemExit("report already exists; choose a new path")
    started = time.monotonic()
    manifest = json.loads((args.bundle / ".aether/aether.json").read_text())
    ref = json.loads((args.reference / "reference.json").read_text())
    tokens = json.loads((args.reference / "tokens.json").read_text())
    files = {role: args.bundle / g["file"].split(":", 1)[1] for role, g in manifest["graphs"].items()}
    functions = {role: g.get("function") or g["functions"]["step"] for role, g in manifest["graphs"].items()}
    checks: list[dict] = []

    def record(name: str, value: float, bar: float, passed: bool | None = None, **extra) -> None:
        ok = (value >= bar) if passed is None else passed
        checks.append({"check": name, "value": value, "bar": bar, "passed": bool(ok), **extra})
        print(f"[check] {'PASS' if ok else 'FAIL'} {name}: {value} (bar {bar})", file=sys.stderr, flush=True)

    # Host recipe: tokens, noise, schedule (exact).
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bundle / "tokenizer")
    references = ref.get("references")
    ids = (np.load(args.reference / "cases" / references["case"] / "ids.npy").tolist() if references
           else kg.prompt_ids(tokenizer, ref["prompt"]))
    record("tokens.exact", float(ids == tokens["real"]), 1.0, ids == tokens["real"], count=len(ids))
    rows = columns = ref["height"] // kg.PIXELS_PER_TOKEN
    n = rows * columns
    noise = kg.seeded_noise(ref["seed"], n * kg.LATENT_CHANNELS).reshape(n, kg.LATENT_CHANNELS)
    ref_noise = np.load(args.reference / "noise.npy")
    record("noise.exact", float(np.array_equal(noise, ref_noise)), 1.0, np.array_equal(noise, ref_noise))
    sigmas, timesteps = kg.schedule(ref["steps"])
    exact = sigmas == ref["sigmas"] and timesteps == ref["timesteps"]
    record("schedule.exact", float(exact), 1.0, exact)
    rope = kg.rotary_table(rows, columns)
    rope = rope.numpy()

    wait_for_m2_gates(args.wait)
    timings = {}

    # Text encoder.
    ref_text = np.load(args.reference / "text_embeds.npy")
    if references:
        if "vision" not in files:
            raise SystemExit("the reference has reference images but the bundle has no vision graph")
        text_embeds = reference_checks(args, manifest, files, tokenizer, references, record, timings)
        per_row = row_cosines(text_embeds.astype(np.float32), ref_text)
        record("text.rowCosineMin", round(float(per_row.min()), 6), REFERENCE_WORST_ROW_BAR,
               cosine=round(cosine(text_embeds, ref_text), 7), rows=int(per_row.size),
               case=references["case"], note="the chosen case's graph conditioning, as the closed loop uses it")
    else:
        text = Graph(files["textEncoder"], functions["textEncoder"])
        timings["loadText"] = text.load_seconds
        length = len(ids)
        t0 = time.monotonic()
        out = text(input_ids=np.array([ids], dtype=np.int32), position_ids=np.arange(length, dtype=np.int32)[None])
        timings["text"] = round(time.monotonic() - t0, 2)
        text_embeds = out["text_embeds"]
        text.close()
        if text_embeds.shape != ref_text.shape:
            raise SystemExit(f"text_embeds {text_embeds.shape} vs reference {ref_text.shape}")
        per_row = row_cosines(text_embeds.astype(np.float32), ref_text)
        record("text.rowCosineMin", round(float(per_row.min()), 6), COSINE_BAR,
               cosine=round(cosine(text_embeds, ref_text), 7), rows=int(per_row.size),
               maxAbs=float(np.abs(ref_text).max()))

    # Denoiser segments: teacher-forced per step and segment, then the closed loop.
    segments = Segments(manifest, files, ref_text.shape[0])
    timings["loadDenoisers"] = segments.load_seconds
    teacher = []
    step_seconds = []
    for s in range(ref["steps"]):
        r = dict(np.load(args.reference / f"step{s}.npz"))
        t = r["timestep"].astype(np.float32)
        feeds = [
            dict(latents=r["latents_in"].astype(np.float16), text_embeds=ref_text.astype(np.float16), sigma=t,
                 rope_image=rope),
            dict(image_1=r["image_1"], text_1=r["text_1"], cond=r["cond"], rope_image=rope),
            dict(image_2=r["image_2"], text_2=r["text_2"], cond=r["cond"], rope_image=rope),
            dict(image_3=r["image_3"], text_3=r["text_3"], cond=r["cond"], rope_image=rope),
        ]
        expected = [dict(image_1=r["image_1"], text_1=r["text_1"], cond=r["cond"]),
                    dict(image_2=r["image_2"], text_2=r["text_2"]), dict(image_3=r["image_3"], text_3=r["text_3"]),
                    dict(velocity=r["velocity"])]
        row = {"step": s}
        for i, (feed, want) in enumerate(zip(feeds, expected)):
            got = segments(i, **feed)
            for name, value in want.items():
                c = cosine(got[name], value)
                row[f"denoiser{i}.{name}"] = round(c, 6)
                checks.append({"check": f"teacher.step{s}.denoiser{i}.{name}", "value": round(c, 6), "bar": COSINE_BAR,
                               "passed": c >= COSINE_BAR})
        teacher.append(row)
        print(f"[check] teacher step {s}: " + json.dumps({k: v for k, v in row.items() if k != "step"}), file=sys.stderr,
              flush=True)

    latents = noise.copy()
    for s in range(ref["steps"]):
        t0 = time.monotonic()
        sigma = np.array([timesteps[s]], dtype=np.float32)
        o0 = segments(0, latents=latents.astype(np.float16), text_embeds=text_embeds, sigma=sigma, rope_image=rope)
        o1 = segments(1, image_1=o0["image_1"], text_1=o0["text_1"], cond=o0["cond"], rope_image=rope)
        o2 = segments(2, image_2=o1["image_2"], text_2=o1["text_2"], cond=o0["cond"], rope_image=rope)
        o3 = segments(3, image_3=o2["image_3"], text_3=o2["text_3"], cond=o0["cond"], rope_image=rope)
        delta = np.float32(sigmas[s + 1]) - np.float32(sigmas[s])
        latents = (latents + (delta * o3["velocity"].astype(np.float32)).astype(np.float32)).astype(np.float32)
        step_seconds.append(round(time.monotonic() - t0, 3))
    # Refine (K7): the encoder graph, then the remaining steps from the start step, closed loop.
    refine_latents = None
    refine = ref.get("refine")
    if refine and "vaeEncoder" in files and (args.reference / "encoder_latents.npy").exists():
        encoder = Graph(files["vaeEncoder"], functions["vaeEncoder"])
        timings["loadEncoder"] = encoder.load_seconds
        picture = np.load(args.reference / "image.npy")
        t0 = time.monotonic()
        encoded = encoder(image=picture.astype(np.float16))["latents"].astype(np.float32)
        timings["encode"] = round(time.monotonic() - t0, 2)
        encoder.close()
        x0 = encoded.reshape(n, kg.LATENT_CHANNELS)
        ref_x0 = np.load(args.reference / "encoder_latents.npy")
        record("vaeEncoder.latentCosine", round(cosine(x0, ref_x0), 6), COSINE_BAR,
               rowCosineMin=round(float(row_cosines(x0, ref_x0).min()), 6))
        refine_noise = kg.seeded_noise(refine["seed"], n * kg.LATENT_CHANNELS).reshape(n, kg.LATENT_CHANNELS)
        same = np.array_equal(refine_noise, np.load(args.reference / "refine_noise.npy"))
        record("refine.noise.exact", float(same), 1.0, same)
        start = refine["startStep"]
        sigma = np.float32(sigmas[start])
        record("refine.sigma.exact", float(abs(float(sigma) - refine["startSigma"]) < 1e-6), 1.0,
               abs(float(sigma) - refine["startSigma"]) < 1e-6, sigma=float(sigma))
        refine_latents = ((np.float32(1) - sigma) * x0 + sigma * refine_noise).astype(np.float32)
        start_cosine = cosine(refine_latents, np.load(args.reference / "refine_start.npy"))
        for s in range(start, ref["steps"]):
            t = np.array([timesteps[s]], dtype=np.float32)
            o0 = segments(0, latents=refine_latents.astype(np.float16), text_embeds=text_embeds, sigma=t, rope_image=rope)
            o1 = segments(1, image_1=o0["image_1"], text_1=o0["text_1"], cond=o0["cond"], rope_image=rope)
            o2 = segments(2, image_2=o1["image_2"], text_2=o1["text_2"], cond=o0["cond"], rope_image=rope)
            o3 = segments(3, image_3=o2["image_3"], text_3=o2["text_3"], cond=o0["cond"], rope_image=rope)
            delta = np.float32(sigmas[s + 1]) - np.float32(sigmas[s])
            refine_latents = (refine_latents + (delta * o3["velocity"].astype(np.float32))).astype(np.float32)
        refine["startLatentCosine"] = round(start_cosine, 6)
    else:
        print("[check] refine: skipped (no vaeEncoder graph or no refine reference)", file=sys.stderr, flush=True)
    segments.close()
    ref_final = np.load(args.reference / f"step{ref['steps'] - 1}.npz")["latents_out"]
    closed_latent_cosine = round(cosine(latents, ref_final), 6)

    # VAE decoder.
    vae = Graph(files["vaeDecoder"], functions["vaeDecoder"])
    ref_image = np.load(args.reference / "image.npy")
    decoded = vae(latents=ref_final.reshape(rows, columns, kg.LATENT_CHANNELS).astype(np.float16))["image"]
    record("vae.psnrDB", round(psnr(decoded, ref_image), 3), PSNR_BAR)
    t0 = time.monotonic()
    image = vae(latents=latents.reshape(rows, columns, kg.LATENT_CHANNELS).astype(np.float16))["image"]
    timings["decode"] = round(time.monotonic() - t0, 2)
    refined = None
    if refine_latents is not None:
        refined = vae(latents=refine_latents.reshape(rows, columns, kg.LATENT_CHANNELS).astype(np.float16))["image"]
    vae.close()
    if refined is not None:
        ref_refined = np.load(args.reference / "refine_image.npy")
        record("refine.ssim", round(ssim(refined, ref_refined), 6), SSIM_BAR, psnrDB=round(psnr(refined, ref_refined), 3),
               startLatentCosine=refine["startLatentCosine"],
               ssimToSource=round(ssim(refined, np.load(args.reference / "image.npy")), 6))
    record("image.ssim", round(ssim(image, ref_image), 6), SSIM_BAR, psnrDB=round(psnr(image, ref_image), 3),
           finalLatentCosine=closed_latent_cosine)
    from PIL import Image
    pixels = np.clip(np.round((image.transpose(1, 2, 0) / 2 + 0.5) * 255), 0, 255).astype(np.uint8)
    png = args.report.with_suffix(".png")
    Image.fromarray(pixels).save(png)
    np.save(args.report.with_suffix(".npy"), image.astype(np.float32))
    compared = None
    if args.compare_image is not None:
        other = np.load(args.compare_image)
        compared = {"image": str(args.compare_image), "psnrDB": round(psnr(image, other), 3),
                    "ssim": round(ssim(image, other), 6), "maxAbs": float(np.abs(image - other).max())}
        print(f"[check] against {args.compare_image}: {compared}", file=sys.stderr, flush=True)

    teacher_checks = [c for c in checks if c["check"].startswith("teacher.")]
    velocity = [c["value"] for c in teacher_checks if c["check"].endswith(".velocity")]
    summary = {
        "variant": ref["variant"], "bundle": str(args.bundle), "reference": str(args.reference),
        "passed": all(c["passed"] for c in checks),
        "counts": {"checks": len(checks), "failed": sum(not c["passed"] for c in checks)},
        "minima": {"textRowCosine": next(c["value"] for c in checks if c["check"] == "text.rowCosineMin"),
                   **({"referenceMeanRowCosine": min(c["value"] for c in checks if c["check"].endswith("meanRowCosine")),
                       "referenceWorstRowCosine": min(c["value"] for c in checks
                                                      if c["check"].endswith("worstRowCosine")),
                       "referenceVisionCosine": min(c["value"] for c in checks if c["check"].endswith("vision.cosine")),
                       "referenceDeepstackCosine": min(c["value"] for c in checks
                                                       if c["check"].endswith("deepstack.cosine"))}
                      if references else {}),
                   "teacherCosine": min(c["value"] for c in teacher_checks),
                   "teacherVelocityCosine": min(velocity),
                   "vaePSNRdB": next(c["value"] for c in checks if c["check"] == "vae.psnrDB"),
                   "imageSSIM": next(c["value"] for c in checks if c["check"] == "image.ssim"),
                   "imagePSNRdB": next(c["psnrDB"] for c in checks if c["check"] == "image.ssim"),
                   "closedLoopLatentCosine": closed_latent_cosine,
                   **({"vaeEncoderLatentCosine": next(c["value"] for c in checks if c["check"] == "vaeEncoder.latentCosine"),
                       "refineSSIM": next(c["value"] for c in checks if c["check"] == "refine.ssim")}
                      if refined is not None else {"refine": "skipped"})},
        "denoisers": {"textMask": segments.length is not None, "function": segments.function,
                      "textLength": segments.length, "conditioningRows": segments.rows},
        **({"comparedImage": compared} if compared else {}),
        "timingsSeconds": {**timings, "closedLoopSteps": step_seconds, "total": round(time.monotonic() - started, 1)},
        "hostPeakRSSGB": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),
        "image": str(png),
    }
    args.report.write_text(json.dumps({**summary, "teacher": teacher, "checks": checks}, indent=2) + "\n")
    print(json.dumps(summary, indent=1))
    if not summary["passed"]:
        raise SystemExit("Krea 2 parity gate failed")


if __name__ == "__main__":
    main()

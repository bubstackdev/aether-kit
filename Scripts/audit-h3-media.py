#!/usr/bin/env python3
"""Media audit of one H3 render: the decoded frames and PCM a `aether-probe h3-profile-run --render` directory holds.

CPU only; reads the render directory and the dump it names, writes nothing there. Checks, each with a stated bar:

  frames   count = the geometry's snapped frame count (17n + 5 at or above the requested count); shape [F, H, W, 3] uint8
           at the dump's canvas; no constant, black (mean < 8) or white (mean > 247) frame; frozen pairs (mean absolute
           difference < 0.25) at most 10% of transitions. Recorded: per-frame mean and spread, saturated-pixel share,
           temporal difference and luma jumps above 25 (flicker).
  audio    PCM [2, N] float32 at 32 kHz, finite, N = 800 x round(F / 24 x 40) per channel, duration within 1/40 s of
           F / 24; clipped samples (|x| >= 0.999) at most 0.1%; DC offset below 0.01; not entirely silent (20 ms windows
           under -60 dBFS). Recorded: peak and RMS in dBFS, silent-window share and the L/R correlation.

`--clips DIR` also writes a 16-bit WAV of the PCM and, when ffmpeg is installed, an H.264/AAC MP4 at 24 fps for the
subjective review. `--reference RENDER_DIR` compares against another render of the same geometry (for example Turbo4
against Turbo8 on the same seed and prompt): per-frame PSNR and luma SSIM (11x11 Gaussian, sigma 1.5), and the PCM's
per-channel correlation and difference level. With `--clips`, it also writes the reference's MP4 and a side-by-side
(reference left). Usage: python -I audit-h3-media.py RENDER_DIR [--reference RENDER_DIR] [--clips NEW_DIR] [--output REPORT.json]
"""
import argparse, hashlib, json, math, shutil, subprocess, sys, wave
from pathlib import Path

import numpy as np

FPS, SAMPLE_RATE, HOP, LATENTS_PER_SECOND = 24, 32_000, 800, 40


def tensor(directory, manifest, name, dtype):
    entry = manifest["tensors"][name]
    data = (directory / entry["file"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise SystemExit(f"{name} changed on disk")
    assert entry["dtype"] == dtype, (name, entry["dtype"])
    return np.frombuffer(data, dtype={"uint8": np.uint8, "float32": "<f4"}[dtype]).reshape(entry["shape"])


def compare(frames, ref_frames, pcm, ref_pcm):
    from scipy.ndimage import gaussian_filter
    psnr, ssim = [], []
    for a, b in zip(frames.astype(np.float64), ref_frames.astype(np.float64)):
        mse = ((a - b) ** 2).mean()
        psnr.append(99.0 if mse == 0 else 10 * math.log10(255 ** 2 / mse))
        la, lb = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]), (0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2])
        blur = lambda x: gaussian_filter(x, 1.5, truncate=3.5)
        ma, mb = blur(la), blur(lb)
        va, vb, cov = blur(la * la) - ma * ma, blur(lb * lb) - mb * mb, blur(la * lb) - ma * mb
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        ssim.append(float((((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2)))[5:-5, 5:-5].mean()))
    stats = lambda v: dict(min=round(min(v), 4), median=round(float(np.median(v)), 4), mean=round(float(np.mean(v)), 4))
    diff = (pcm.astype(np.float64) - ref_pcm.astype(np.float64))
    db = lambda x: round(20 * math.log10(max(x, 1e-12)), 2)
    return dict(framePSNR=stats(psnr), frameLumaSSIM=stats(ssim),
                audioCorrelation=[round(float(np.corrcoef(a, b)[0, 1]), 4) for a, b in zip(pcm, ref_pcm)],
                audioDifferenceRMSDBFS=[db(float(np.sqrt((d ** 2).mean()))) for d in diff],
                audioRMSDBFS=dict(render=[db(float(np.sqrt((c.astype(np.float64) ** 2).mean()))) for c in pcm],
                                  reference=[db(float(np.sqrt((c.astype(np.float64) ** 2).mean()))) for c in ref_pcm]))


def check(report, name, passed, value, bar):
    report["checks"].append(dict(name=name, passed=bool(passed), value=value, bar=bar))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("render", type=Path)
    ap.add_argument("--clips", type=Path, help="a new directory for the review WAV and MP4")
    ap.add_argument("--reference", type=Path, help="another render of the same geometry to compare against")
    ap.add_argument("--output", type=Path, help="also write the report here")
    args = ap.parse_args()
    manifest = json.loads((args.render / "manifest.json").read_text())
    dump = json.loads((Path(manifest["fields"]["dump"]) / "manifest.json").read_text())["fields"]
    width, height, requested = int(dump["width"]), int(dump["height"]), int(dump["frames"])
    expected_frames = requested + (-(requested - 5)) % 17
    report = dict(schema=1, render=str(args.render), profile_sha256=manifest["fields"]["profile_sha256"],
                  geometry=dict(width=width, height=height, requestedFrames=requested, frames=expected_frames, fps=FPS,
                                durationSeconds=expected_frames / FPS), checks=[], stats={})

    frames = tensor(args.render, manifest, "frames", "uint8")
    check(report, "frameShape", list(frames.shape) == [expected_frames, height, width, 3], list(frames.shape), [expected_frames, height, width, 3])
    f = frames.astype(np.float32)
    means, spreads = f.mean(axis=(1, 2, 3)), f.std(axis=(1, 2, 3))
    luma = (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]).mean(axis=(1, 2))
    diffs = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2, 3)) if len(f) > 1 else np.zeros(0)
    check(report, "noConstantFrame", int((spreads < 2).sum()) == 0, int((spreads < 2).sum()), 0)
    check(report, "noBlackFrame", int((means < 8).sum()) == 0, int((means < 8).sum()), 0)
    check(report, "noWhiteFrame", int((means > 247).sum()) == 0, int((means > 247).sum()), 0)
    frozen = int((diffs < 0.25).sum())
    check(report, "frozenTransitionsAtMost10Percent", frozen <= 0.1 * max(len(diffs), 1), frozen, f"<= {0.1 * max(len(diffs), 1):.1f}")
    report["stats"]["frames"] = dict(
        meanRange=[round(float(means.min()), 2), round(float(means.max()), 2)], spreadRange=[round(float(spreads.min()), 2), round(float(spreads.max()), 2)],
        saturatedShare=round(float(((frames == 0) | (frames == 255)).mean()), 5),
        temporalDifference=dict(min=round(float(diffs.min()), 3), median=round(float(np.median(diffs)), 3), max=round(float(diffs.max()), 3)) if len(diffs) else None,
        lumaJumpsAbove25=int((np.abs(np.diff(luma)) > 25).sum()))

    pcm = tensor(args.render, manifest, "pcm", "float32")
    latents = round(expected_frames / FPS * LATENTS_PER_SECOND)
    check(report, "pcmShape", list(pcm.shape) == [2, HOP * latents], list(pcm.shape), [2, HOP * latents])
    check(report, "pcmFinite", bool(np.isfinite(pcm).all()), int((~np.isfinite(pcm)).sum()), 0)
    duration = pcm.shape[1] / SAMPLE_RATE
    check(report, "durationMatchesVideo", abs(duration - expected_frames / FPS) <= 1 / LATENTS_PER_SECOND,
          round(duration - expected_frames / FPS, 5), f"|d| <= {1 / LATENTS_PER_SECOND}")
    clipped = float((np.abs(pcm) >= 0.999).mean())
    check(report, "clippingAtMost0.1Percent", clipped <= 0.001, round(clipped, 6), 0.001)
    dc = [round(float(c.mean()), 5) for c in pcm]
    check(report, "dcOffsetBelow0.01", all(abs(d) < 0.01 for d in dc), dc, 0.01)
    window = SAMPLE_RATE // 50
    usable = pcm[:, : pcm.shape[1] // window * window].reshape(2, -1, window)
    window_db = 20 * np.log10(np.sqrt((usable.astype(np.float64) ** 2).mean(axis=(0, 2))) + 1e-12)
    silent = float((window_db < -60).mean())
    check(report, "notSilent", silent < 1.0, round(silent, 4), "< 1.0")
    db = lambda x: round(20 * math.log10(max(x, 1e-12)), 2)
    report["stats"]["audio"] = dict(
        sampleRate=SAMPLE_RATE, channels=int(pcm.shape[0]), samples=int(pcm.shape[1]), seconds=round(duration, 4),
        peakDBFS=[db(float(np.abs(c).max())) for c in pcm], rmsDBFS=[db(float(np.sqrt((c.astype(np.float64) ** 2).mean()))) for c in pcm],
        silentWindowShare=round(silent, 4),
        channelCorrelation=round(float(np.corrcoef(pcm[0], pcm[1])[0, 1]), 4) if pcm[0].std() > 0 and pcm[1].std() > 0 else None)
    report["passed"] = all(c["passed"] for c in report["checks"])

    if args.reference:
        ref_manifest = json.loads((args.reference / "manifest.json").read_text())
        ref_frames = tensor(args.reference, ref_manifest, "frames", "uint8")
        ref_pcm = tensor(args.reference, ref_manifest, "pcm", "float32")
        if ref_frames.shape != frames.shape or ref_pcm.shape != pcm.shape:
            raise SystemExit("reference render has another geometry")
        report["reference"] = dict(render=str(args.reference), comparison=compare(frames, ref_frames, pcm, ref_pcm))

    if args.clips:
        args.clips.mkdir(parents=True, exist_ok=False)
        wav = args.clips / "audio.wav"
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
            w.writeframes((np.clip(pcm.T, -1, 1) * 32767).round().astype("<i2").tobytes())
        clips = dict(wav=str(wav))
        if shutil.which("ffmpeg"):
            mp4 = args.clips / "render.mp4"
            command = ["ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(FPS), "-i", "-",
                       "-i", str(wav), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-c:a", "aac", "-b:a", "192k", "-shortest", str(mp4)]
            subprocess.run(command, input=frames.tobytes(), check=True)
            clips["mp4"] = str(mp4)
        if args.reference and shutil.which("ffmpeg"):
            ref_mp4, pair = args.clips / "reference.mp4", args.clips / "reference-left.mp4"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(FPS),
                            "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(ref_mp4)], input=ref_frames.tobytes(), check=True)
            subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(ref_mp4), "-i", str(clips["mp4"]), "-filter_complex", "hstack=inputs=2",
                            "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "copy", str(pair)], check=True)
            clips |= dict(referenceMp4=str(ref_mp4), sideBySide=str(pair))
        report["clips"] = clips

    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    sys.stdout.write(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

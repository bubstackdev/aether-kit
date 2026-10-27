#!/usr/bin/env python3
"""Paired, interleaved H3 render timings: two aether-probe builds (A = current path, B = candidate) render the same dump
in ABBA order, each render under the GPU lock as its own owned process. Records each render's stage and step times
from the probe's report, the load average before and after, the sampled Metal peak, and the final latents' hashes
(optionally checked against a trajectory capture's final-step hashes). Raw tensors are deleted after recording.

Usage: time-h3-render-paired.py --a PROBE --b PROBE --profile BUNDLE --dump DUMP --work NEW_DIR [--pairs N] [--capture CAPTURE_RUN]
"""
import argparse, hashlib, json, os, statistics, subprocess, time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load():
    out = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout.strip("{} \n").split()
    return [float(x) for x in out[:3]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True); ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--profile", type=Path, required=True); ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True); ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--capture", type=Path, help="trajectory capture run whose final-step latent hashes every render must equal")
    ap.add_argument("--gpu-lock", type=Path, default=Path.home() / ".cache/aether-kit/with-gpu-lock.sh")
    args = ap.parse_args()
    work = args.work.resolve(); work.mkdir(parents=True, exist_ok=False)
    expected = None
    if args.capture:
        hashes = json.loads((args.capture / "capture/manifest.json").read_text())["report"]["hashes"]
        last = max(int(k.split("_")[0][4:]) for k in hashes if k.startswith("step"))
        expected = {k: hashes[f"step{last}_latents_out_{k}"] for k in ("video", "audio")}
    builds = {"A": args.a.resolve(), "B": args.b.resolve()}
    runs = []
    order = []
    for i in range(args.pairs):
        order += ["A", "B", "B", "A"] if i % 2 == 0 else ["B", "A", "A", "B"]
    order = order[:2 * args.pairs]
    for n, variant in enumerate(order):
        out = work / f"run-{n:02d}-{variant}"
        before = load(); start = time.monotonic()
        proc = subprocess.run([str(args.gpu_lock), f"h3-speed: paired render {n} {variant}", str(builds[variant]), "h3-profile-run", str(args.profile),
                               str(args.dump), "-", str(out), "--render"], capture_output=True, text=True)
        elapsed = time.monotonic() - start; after = load()
        m = json.loads((out / "manifest.json").read_text())
        timings = {t["name"]: t["seconds"] for t in m["report"]["timings"]}
        latents = {k: m["tensors"][f"final_{k}"]["sha256"] for k in ("video", "audio")}
        record = dict(run=n, variant=variant, exit=proc.returncode, wallIncludingLockWaitSeconds=round(elapsed, 2), loadBefore=before, loadAfter=after,
                      renderSeconds=timings.get("render"), stages={k: v for k, v in timings.items() if k.startswith(("stage.", "step."))},
                      metalPeakGiB=round(m["report"]["sampledPeak"]["metalAllocatedBytes"] / 2**30, 3), finalLatents=latents,
                      matchesCapture=None if expected is None else latents == expected)
        runs.append(record)
        print(json.dumps(dict(run=n, variant=variant, render=record["renderSeconds"], denoising=record["stages"].get("stage.denoising"),
                              matches=record["matchesCapture"])), flush=True)
        for raw in out.glob("*.raw"):
            raw.unlink()
    summary = {}
    for v in ("A", "B"):
        rs = [r for r in runs if r["variant"] == v]
        summary[v] = dict(n=len(rs), renderMedian=statistics.median(r["renderSeconds"] for r in rs),
                          denoisingMedian=statistics.median(r["stages"]["stage.denoising"] for r in rs),
                          decodingVideoMedian=statistics.median(r["stages"]["stage.decodingVideo"] for r in rs),
                          metalPeakGiB=max(r["metalPeakGiB"] for r in rs), probeSHA256=sha(builds[v]))
    report = dict(schema=1, profileSHA256=sha(args.profile), dump=str(args.dump), order=order, runs=runs, summary=summary,
                  allMatchCapture=None if expected is None else all(r["matchesCapture"] for r in runs))
    (work / "paired.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

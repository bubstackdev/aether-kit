#!/usr/bin/env python3
"""Own the schema-3 H3 trajectory gate for one task: a captured native Turbo trajectory, an exact replay, and
original-precision CPU references at the captured contexts.

Stages run strictly in order, each an owned process group with a wall budget, joined before the next starts:

  dump            aether-probe h3-profile-dump (CPU): token ids, layout, step plan, seeded latents (+ first frame, FL2VA).
  capture         aether-probe h3-profile-run --render --capture=STEPS (GPU, under ~/.cache/aether-kit/with-gpu-lock.sh):
                  a complete H3VideoSession render whose opt-in denoiser observer hashes every trajectory boundary
                  (refined text; per step input latents, packed sequence, all 50 block outputs, heads, updated latents)
                  and saves the selected steps' CPU-reference contexts.
  replay          the same render with --replay=CAPTURE: every boundary hash must equal the capture's. A difference
                  means the trajectory is nondeterministic; the gate stops there.
  oracle          prepare-h3-profile-oracle.py --part trajectory (CPU, off the GPU lock, 8 threads): per selected step,
                  packing, blocks 0 and 49, heads and both scheduler updates with the original modules at original
                  precision, full tensors; float32 blocks as a diagnostic only.
  gate            every required comparison finite and within the predefined relative-L2 limit; replay identical;
                  pins unchanged. Then the capture's tensors are deleted (manifests and hashes are kept).

CPU stages wait for --minimum-free-percent free memory and are stopped above --cpu-rss-limit-gb resident.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

LIMIT = 0.02
BOUNDARIES_PER_STEP = 2 + 1 + 50 + 2 + 2   # latents in, packed, blocks, heads, latents out


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def memory_free_percent():
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=60).stdout
    match = re.search(r"System-wide memory free percentage:\s*(\d+)%", out)
    return int(match.group(1)) if match else None


def group_rss_bytes(pgid):
    out = subprocess.run(["ps", "-o", "rss=,pgid=", "-A"], capture_output=True, text=True, timeout=30).stdout
    return sum(int(f[0]) * 1024 for f in (line.split() for line in out.splitlines()) if len(f) == 2 and f[1] == str(pgid))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["t2va", "fl2va"], required=True)
    ap.add_argument("--steps", default="all", help="steps to reference: all or S,S,... (every step is hashed and replayed)")
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--python", type=Path, required=True)
    ap.add_argument("--staging", type=Path, required=True)
    ap.add_argument("--prompt", type=Path, required=True)
    ap.add_argument("--keyframe", type=Path, help="first frame (FL2VA), exactly WIDTHxHEIGHT")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=544)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1043)
    ap.add_argument("--recipe", default="turbo8")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--minimum-free-percent", type=int, default=40)
    ap.add_argument("--cpu-rss-limit-gb", type=float, default=20.0)
    ap.add_argument("--keep-captures", action="store_true")
    ap.add_argument("--gpu-lock", type=Path, default=Path.home() / ".cache/aether-kit/with-gpu-lock.sh")
    args = ap.parse_args()
    if (args.task == "fl2va") != (args.keyframe is not None):
        ap.error("--keyframe is required for fl2va and refused for t2va")
    root = Path(__file__).resolve().parent.parent
    probe = root / ".build/arm64-apple-macosx/release/aether-probe"
    oracle = root / "Scripts/prepare-h3-profile-oracle.py"
    work = args.work.resolve()
    (work / "controllers").mkdir(parents=True, exist_ok=True)
    staging = args.staging.resolve()
    files = [probe, oracle, root / "Scripts/h3_export_lib.py", Path(__file__).resolve(), root / "Package.swift",
             args.profile.resolve(), args.prompt.resolve(), staging / "oracle-scripts/vendor/convert_minimax_h3_to_diffusers.py"]
    files += [args.keyframe.resolve()] if args.keyframe else []
    for directory in ["Sources/AetherKit", "Sources/AetherAssets", "Sources/AetherCoreAI", "Sources/AetherH3",
                      "Sources/AetherH3CoreAI", "Sources/AetherH3Tokenizers", "Sources/AetherProbe"]:
        files.extend(sorted(p for p in (root / directory).rglob("*") if p.is_file()))
    pins = {str(p): sha(p) for p in files}
    digest = hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()
    pins_path = work / "pins.json"
    if pins_path.exists():
        if json.loads(pins_path.read_text())["filesSHA256"] != pins:
            ap.error(f"{work} was started with different pins; use a new --work directory")
    else:
        pins_path.write_text(json.dumps(dict(schema=1, createdAt=now(), filesSHA256=pins, digest=digest), indent=2) + "\n")

    env = dict(os.environ, H3_STAGING=str(staging), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    t = args.task
    d = lambda name: str(work / name)
    dump = [str(probe), "h3-profile-dump", str(args.profile), str(args.prompt), str(args.width), str(args.height), str(args.frames),
            str(args.seed), args.recipe, d(f"dump-{t}")] + ([str(args.keyframe)] if args.keyframe else [])
    stages = [
        (f"dump-{t}", "cpu", 600, dump),
        (f"capture-{t}", "gpu", 7200, [str(probe), "h3-profile-run", str(args.profile), d(f"dump-{t}"), "-", d(f"capture-{t}"), "--render", f"--capture={args.steps}"]),
        (f"replay-{t}", "gpu", 7200, [str(probe), "h3-profile-run", str(args.profile), d(f"dump-{t}"), "-", d(f"replay-{t}"), "--render", f"--replay={d(f'capture-{t}')}"]),
        (f"oracle-{t}", "cpu", 6 * 3600, [str(args.python), "-I", "-B", str(oracle), "--part", "trajectory", "--dump", d(f"dump-{t}"), "--run", d(f"capture-{t}"),
                                          "--steps", args.steps, "--output", d(f"oracle-{t}"), "--threads", str(args.threads)]),
        # Diagnostic only, never an acceptance criterion: native blocks against the same modules in float32.
        (f"float32-{t}", "cpu", 3600, [str(args.python), "-I", "-B", str(oracle), "--part", "trajectory", "--float32-diagnostic", "--dump", d(f"dump-{t}"),
                                       "--run", d(f"capture-{t}"), "--steps", args.steps, "--output", d(f"float32-{t}"), "--threads", str(args.threads)]),
    ]
    optional = {f"float32-{t}"}
    records = {}
    for name, kind, budget, command in stages:
        record_path = work / "controllers" / f"{name}.json"
        output = work / name
        if record_path.exists():
            previous = json.loads(record_path.read_text())
            if previous.get("passed") and previous.get("pinsDigest") == digest and output.exists():
                records[name] = previous
                print(f"{name}: already passed", flush=True)
                continue
        if output.exists():
            output.rename(work / f"{name}.failed-{int(time.time())}")
        if kind == "gpu":
            command = [str(args.gpu_lock), f"h3-trajectory: {name}", *command]
        record = dict(schema=1, stage=name, kind=kind, command=command, budgetSeconds=budget, pinsDigest=digest,
                      gpuLock=kind == "gpu", passed=False, joined=False, timedOut=False)
        if kind == "cpu":
            waited = time.monotonic()
            while True:
                free = memory_free_percent()
                if free is not None and free >= args.minimum_free_percent:
                    break
                if time.monotonic() - waited > 4 * 3600:
                    record.update(error=f"memory free {free}% below {args.minimum_free_percent}% for 4 h")
                    record_path.write_text(json.dumps(record, indent=2) + "\n")
                    return 1
                time.sleep(60)
            record.update(memoryFreePercentAtStart=free, memoryWaitSeconds=round(time.monotonic() - waited, 1),
                          rssLimitBytes=int(args.cpu_rss_limit_gb * (1 << 30)))
        record["startedAt"] = now()
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"{name}: start {record['startedAt']}", flush=True)
        start, peak = time.monotonic(), 0
        with (work / "controllers" / f"{name}.stdout.log").open("wb") as out, (work / "controllers" / f"{name}.stderr.log").open("wb") as err:
            proc = subprocess.Popen(command, stdout=out, stderr=err, env=env, start_new_session=True)
            record["pid"] = proc.pid
            record_path.write_text(json.dumps(record, indent=2) + "\n")
            try:
                while proc.poll() is None:
                    if time.monotonic() - start > budget - 10:
                        record["timedOut"] = True
                        break
                    if kind == "cpu":
                        rss = group_rss_bytes(proc.pid)
                        peak = max(peak, rss)
                        if rss > record["rssLimitBytes"]:
                            record["rssLimitExceeded"] = True
                            break
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            except KeyboardInterrupt:
                record["interrupted"] = True
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        record.update(joined=True, exitCode=proc.returncode, endedAt=now(), elapsedSeconds=round(time.monotonic() - start, 3))
        if kind == "cpu":
            record["sampledPeakGroupRSSBytes"] = peak
        record["sourcePinsUnchanged"] = all(sha(p) == v for p, v in pins.items())
        try:
            m = json.loads((output / "manifest.json").read_text())
            report = m.get("report")
            record["outputError"] = report.get("error") if isinstance(report, dict) else None
        except (OSError, ValueError) as error:
            record["outputError"] = f"manifest unreadable: {error}"
        record["passed"] = (proc.returncode == 0 and not record["timedOut"] and not record.get("rssLimitExceeded")
                            and not record.get("interrupted") and record["sourcePinsUnchanged"] and record.get("outputError") is None)
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"{name}: passed={record['passed']} exit={proc.returncode} {record['elapsedSeconds']} s", flush=True)
        records[name] = record
        if not record["passed"] and name not in optional:
            return 1

    def manifest(path):
        return json.loads((work / path / "manifest.json").read_text())
    failures = []
    dump_fields = manifest(f"dump-{t}")["fields"]
    steps_total = int(dump_fields["steps"])
    expected_hashes = 1 + steps_total * BOUNDARIES_PER_STEP
    capture_hashes = manifest(f"capture-{t}/capture")["report"]["hashes"]
    capture, replay = manifest(f"capture-{t}"), manifest(f"replay-{t}")
    if len(capture_hashes) != expected_hashes or capture["fields"].get("trajectory_hashes") != str(expected_hashes):
        failures.append(f"capture holds {len(capture_hashes)} boundary hashes, expected {expected_hashes}")
    if replay["fields"].get("replay_mismatches") != "0" or replay["fields"].get("trajectory_hashes") != str(expected_hashes):
        failures.append(f"replay: {replay['fields'].get('replay_mismatches')} mismatches over {replay['fields'].get('trajectory_hashes')} hashes")
    oracle_report = manifest(f"oracle-{t}")["report"]
    selected = list(range(steps_total)) if args.steps == "all" else [int(s) for s in args.steps.split(",")]
    required = [f"step{s}/{n}" for s in selected for n in
                ("packed", "block00", "block49", "heads_video", "heads_audio", "latents_out_video", "latents_out_audio")]
    comparisons = oracle_report.get("comparisons", {})
    for key in required:
        row = comparisons.get(key)
        if row is None:
            failures.append(f"{key}: missing")
        elif not (row["finite"] and math.isfinite(row["relative_l2"]) and row["relative_l2"] <= LIMIT):
            failures.append(f"{key}: relative L2 {row['relative_l2']} (finite {row['finite']})")
    layout = oracle_report.get("layout", {})
    if not layout or not all(layout.values()):
        failures.append(f"layout checks {layout}")
    pins_unchanged = all(sha(p) == v for p, v in pins.items())
    if not pins_unchanged:
        failures.append("pins changed during the gate")
    open_items = []
    for n, m in (("capture", capture), ("replay", replay)):
        metal = ((m.get("report") or {}).get("sampledPeak") or {}).get("metalAllocatedBytes")
        if metal and metal > 18 * (1 << 30):
            open_items.append(f"{n}: sampled Metal peak {metal / (1 << 30):.2f} GiB exceeds 18 GiB")
    gate = dict(schema=1, gate=f"h3-profile-trajectory-{t}", completedAt=now(), passed=not failures, failures=failures, openItems=open_items,
                predefinedRelativeL2Limit=LIMIT, task=t, referencedSteps=selected,
                canvas=dict(width=args.width, height=args.height, frames=int(dump_fields["frames"]), rows=int(dump_fields["rows"]),
                            tokens=int(dump_fields["tokens"]), seed=args.seed, recipe=args.recipe, anchors=dump_fields.get("anchors")),
                profileSHA256=dump_fields["profile_sha256"],
                boundaryHashes=dict(expected=expected_hashes, captured=len(capture_hashes), replayMismatches=replay["fields"].get("replay_mismatches"),
                                    captureHashesSHA256=hashlib.sha256(json.dumps(capture_hashes, sort_keys=True).encode()).hexdigest(),
                                    finalLatents={k: capture_hashes.get(f"step{steps_total - 1}_latents_out_{k}") for k in ("video", "audio")}),
                comparisons={k: comparisons[k] for k in required if k in comparisons},
                float32Diagnostics=(manifest(f"float32-{t}")["report"].get("float32_diagnostics")
                                    if records.get(f"float32-{t}", {}).get("passed") else dict(unavailable=records.get(f"float32-{t}", {}).get("outputError") or "stage failed")),
                oracleTimings={k: v for k, v in oracle_report.items() if k.endswith("_seconds")}, layoutChecks=layout,
                oracleFields=manifest(f"oracle-{t}")["fields"],
                nativeReports={"capture": capture.get("report"), "replay": replay.get("report")}, controllers=records,
                pinsDigest=digest, pinsUnchanged=pins_unchanged, filesSHA256=pins,
                scope=("Native Turbo trajectory through H3VideoSession at one canvas: exact replay of every boundary hash, and "
                       "original-precision CPU references at captured contexts (packing, blocks 0 and 49, heads, scheduler "
                       "updates) for the selected steps. Not independent CPU trajectory parity, CPU checks at every block, "
                       "decoded-media acceptance, lifecycle or memory-limit qualification."))
    (work / f"gate-{t}.json").write_text(json.dumps(gate, indent=2) + "\n")
    if not args.keep_captures and gate["passed"]:
        removed = 0
        for directory in (work / f"capture-{t}", work / f"capture-{t}/capture", work / f"replay-{t}"):
            for raw in directory.glob("*.raw"):
                removed += raw.stat().st_size
                raw.unlink()
        gate["deletedCaptureBytes"] = removed
        (work / f"gate-{t}.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps(dict(passed=gate["passed"], failures=failures, openItems=open_items, gate=str(work / f"gate-{t}.json")), indent=2), flush=True)
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

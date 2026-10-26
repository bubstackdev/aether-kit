#!/usr/bin/env python3
"""Own the schema-3 H3 graph gate: each native graph against the original CPU modules, at one canvas.

Stages run strictly in order, each as an owned process group with a wall budget, joined before the next starts:

  dump-t2va, dump-fl2va    aether-probe h3-profile-dump (CPU): token ids, layout, step plan, seeded latents, probe hidden.
  oracle-text              prepare-h3-profile-oracle.py --part text (CPU): the original Qwen3-VL tap and refined text.
  native-t2va, native-fl2va  aether-probe h3-profile-run --graphs (GPU, under ~/.cache/aether-kit/with-gpu-lock.sh):
                           conditioner tap, prepare text/pack, the native step-0 chain of all 50 blocks (block 0 reads
                           the packed sequence; blocks 0 and 49 and block 49's input are saved), heads on the block-49
                           output, one tile window, audio decoder;
                           the FL2VA dump runs the first-frame encoder only. Each later graph consumes the CPU oracle's
                           own inputs, so every comparison isolates one graph.
  oracle-graphs-t2va, oracle-graphs-fl2va  --part graphs (CPU): the same graphs with the original modules.
  compare-*                --part compare (CPU): relative L2, cosine and max |diff| for every shared tensor.
  gate                     Every required tensor present, finite, same shape and within the predefined relative-L2
                           limit; layout and tokenizer checks true; all pins unchanged since the first stage.

CPU stages never hold the GPU lock. Before each one starts, `memory_pressure` must report at least
--minimum-free-percent free; its process group's resident memory is sampled and the stage is stopped if it exceeds
--cpu-rss-limit-gb. A stage whose record passed with the current pins is not repeated; a failed stage's directory is
renamed aside so a retry writes a new one. Inputs (profile, checkpoint, adapter, staging) are read-only.
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
import sys
import time

LIMIT = 0.02
REQUIRED = {
    "compare-t2va-text": ["tap", "refined"],
    "compare-t2va-graphs": ["packed", "block00", "block49", "heads_video", "heads_audio", "tile", "pcm"],
    "compare-fl2va": ["condition"],
}


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
    total = 0
    for line in out.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1] == str(pgid):
            total += int(fields[0]) * 1024
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", type=Path, required=True, help="schema-3 bundle.json (read in place)")
    ap.add_argument("--python", type=Path, required=True, help="oracle interpreter (torch, pinned Diffusers 7ecb1e1b)")
    ap.add_argument("--staging", type=Path, required=True, help="H3_STAGING: oracle-scripts/vendor and exports/h3-fl2va/converted")
    ap.add_argument("--prompt", type=Path, required=True)
    ap.add_argument("--keyframe", type=Path, required=True, help="first frame for the FL2VA dump, exactly WIDTHxHEIGHT")
    ap.add_argument("--work", type=Path, required=True, help="durable working directory (stage outputs and records)")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=544)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1043)
    ap.add_argument("--recipe", default="turbo8")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--blocks", default="0,49")
    ap.add_argument("--minimum-free-percent", type=int, default=40)
    ap.add_argument("--cpu-rss-limit-gb", type=float, default=20.0)
    ap.add_argument("--gpu-lock", type=Path, default=Path.home() / ".cache/aether-kit/with-gpu-lock.sh")
    ap.add_argument("--stop-after", help="stop after this stage")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    probe = root / ".build/arm64-apple-macosx/release/aether-probe"
    oracle = root / "Scripts/prepare-h3-profile-oracle.py"
    work = args.work.resolve()
    (work / "controllers").mkdir(parents=True, exist_ok=True)
    staging = args.staging.resolve()

    files = [probe, oracle, root / "Scripts/h3_export_lib.py", Path(__file__).resolve(), root / "Package.swift",
             args.profile.resolve(), args.prompt.resolve(), args.keyframe.resolve(),
             staging / "oracle-scripts/vendor/convert_minimax_h3_to_diffusers.py",
             staging / "exports/h3-fl2va/converted/vae/config.json", staging / "exports/h3-fl2va/converted/audio-vae/config.json"]
    for directory in ["Sources/AetherKit", "Sources/AetherAssets", "Sources/AetherCoreAI", "Sources/AetherH3",
                      "Sources/AetherH3CoreAI", "Sources/AetherH3Tokenizers", "Sources/AetherProbe"]:
        files.extend(sorted(p for p in (root / directory).rglob("*") if p.is_file()))
    pins = {str(p): sha(p) for p in files}
    pins_digest = hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()
    first_pins = work / "pins.json"
    if first_pins.exists():
        if json.loads(first_pins.read_text())["filesSHA256"] != pins:
            ap.error(f"{work} was started with different pins; use a new --work directory")
    else:
        first_pins.write_text(json.dumps(dict(schema=1, createdAt=now(), filesSHA256=pins, digest=pins_digest), indent=2) + "\n")

    env = dict(os.environ, H3_STAGING=str(staging), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    py = [str(args.python), "-I", "-B", str(oracle)]
    d = lambda name: str(work / name)
    canvas = [str(args.width), str(args.height), str(args.frames), str(args.seed), args.recipe]
    stages = [
        ("dump-t2va", "cpu", 600, [str(probe), "h3-profile-dump", str(args.profile), str(args.prompt), *canvas, d("dump-t2va")]),
        ("dump-fl2va", "cpu", 600, [str(probe), "h3-profile-dump", str(args.profile), str(args.prompt), *canvas, d("dump-fl2va"), str(args.keyframe)]),
        ("oracle-text", "cpu", 7200, py + ["--part", "text", "--dump", d("dump-t2va"), "--prompt", str(args.prompt), "--output", d("oracle-text"), "--threads", str(args.threads)]),
        ("native-t2va", "gpu", 7200, [str(probe), "h3-profile-run", str(args.profile), d("dump-t2va"), d("oracle-text"), d("native-t2va"), "--graphs"]),
        ("native-fl2va", "gpu", 7200, [str(probe), "h3-profile-run", str(args.profile), d("dump-fl2va"), "-", d("native-fl2va"), "--graphs"]),
        ("oracle-graphs-t2va", "cpu", 6 * 3600, py + ["--part", "graphs", "--dump", d("dump-t2va"), "--run", d("native-t2va"), "--text", d("oracle-text"), "--output", d("oracle-graphs-t2va"), "--threads", str(args.threads), "--blocks", args.blocks]),
        ("oracle-graphs-fl2va", "cpu", 3600, py + ["--part", "graphs", "--dump", d("dump-fl2va"), "--run", d("native-fl2va"), "--text", d("oracle-text"), "--output", d("oracle-graphs-fl2va"), "--threads", str(args.threads)]),
        ("compare-t2va-text", "cpu", 1800, py + ["--part", "compare", "--text", d("oracle-text"), "--run", d("native-t2va"), "--output", d("compare-t2va-text")]),
        ("compare-t2va-graphs", "cpu", 1800, py + ["--part", "compare", "--text", d("oracle-graphs-t2va"), "--run", d("native-t2va"), "--output", d("compare-t2va-graphs")]),
        ("compare-fl2va", "cpu", 1800, py + ["--part", "compare", "--text", d("oracle-graphs-fl2va"), "--run", d("native-fl2va"), "--output", d("compare-fl2va")]),
    ]

    records = {}
    for name, kind, budget, command in stages:
        record_path = work / "controllers" / f"{name}.json"
        output = work / name
        if record_path.exists():
            previous = json.loads(record_path.read_text())
            if previous.get("passed") and previous.get("pinsDigest") == pins_digest and output.exists():
                records[name] = previous
                print(f"{name}: already passed", flush=True)
                if name == args.stop_after:
                    return 0
                continue
        if output.exists():
            output.rename(work / f"{name}.failed-{int(time.time())}")
        if kind == "gpu":
            command = [str(args.gpu_lock), f"h3-graph-gate: {name}", *command]
        record = dict(schema=1, stage=name, kind=kind, command=command, budgetSeconds=budget, pinsDigest=pins_digest,
                      gpuLock=kind == "gpu", passed=False, joined=False, timedOut=False)
        if kind == "cpu":
            # CPU stages wait for memory headroom instead of taking the GPU lock.
            waited = time.monotonic()
            while True:
                free = memory_free_percent()
                if free is not None and free >= args.minimum_free_percent:
                    break
                if time.monotonic() - waited > 4 * 3600:
                    record.update(error=f"memory free {free}% stayed below {args.minimum_free_percent}% for 4 h")
                    record_path.write_text(json.dumps(record, indent=2) + "\n")
                    return 1
                time.sleep(60)
            record.update(memoryFreePercentAtStart=free, memoryWaitSeconds=round(time.monotonic() - waited, 1),
                          rssLimitBytes=int(args.cpu_rss_limit_gb * (1 << 30)))
        record["startedAt"] = now()
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"{name}: start {record['startedAt']}", flush=True)
        start = time.monotonic()
        peak = 0
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
        record["sourcePinsUnchanged"] = all(sha(p) == digest for p, digest in pins.items())
        manifest = output / "manifest.json"
        try:
            m = json.loads(manifest.read_text())
            record["outputManifestSHA256"] = sha(manifest)
            record["outputError"] = m.get("report", {}).get("error") if isinstance(m.get("report"), dict) else None
        except (OSError, ValueError) as error:
            record["outputError"] = f"manifest unreadable: {error}"
        record["passed"] = (proc.returncode == 0 and not record["timedOut"] and not record.get("rssLimitExceeded")
                            and not record.get("interrupted") and record["sourcePinsUnchanged"] and record.get("outputError") is None)
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        print(f"{name}: passed={record['passed']} exit={proc.returncode} {record['elapsedSeconds']} s", flush=True)
        records[name] = record
        if not record["passed"]:
            return 1
        if name == args.stop_after:
            return 0

    # Gate: predefined limit on every required comparison, plus the oracle's own layout/tokenizer checks.
    def manifest(name):
        return json.loads((work / name / "manifest.json").read_text())
    comparisons, failures = {}, []
    for stage, names in REQUIRED.items():
        rows = manifest(stage)["report"]["compare"]
        for tensor in names:
            row = rows.get(tensor)
            key = f"{stage}/{tensor}"
            comparisons[key] = row
            if row is None:
                failures.append(f"{key}: missing")
            elif "error" in row:
                failures.append(f"{key}: {row['error']}")
            elif not (row["finite"] and math.isfinite(row["relative_l2"]) and row["relative_l2"] <= LIMIT):
                failures.append(f"{key}: relative L2 {row['relative_l2']} (finite {row['finite']})")
    text_report = manifest("oracle-text")["report"]
    graphs_report = manifest("oracle-graphs-t2va")["report"]
    layout = graphs_report.get("layout", {})
    expected_inputs = {"block00": "packed", "block49": "context49"}
    if graphs_report.get("block_inputs") != expected_inputs or graphs_report.get("heads_input") != "native block49 output":
        failures.append(f"oracle-graphs-t2va: blocks/heads did not read the real step-0 contexts: {graphs_report.get('block_inputs')}, {graphs_report.get('heads_input')}")
    if text_report.get("token_ids_equal_transformers") is not True:
        failures.append("oracle-text: Swift token ids differ from Transformers")
    if not layout or not all(layout.values()):
        failures.append(f"oracle-graphs-t2va: layout checks {layout}")
    native = {n: manifest(n) for n in ("native-t2va", "native-fl2va")}
    for n, m in native.items():
        if m["fields"].get("profile_sha256") != manifest("dump-t2va")["fields"]["profile_sha256"]:
            failures.append(f"{n}: profile differs from the dump")
    pins_unchanged = all(sha(p) == digest for p, digest in pins.items())
    if not pins_unchanged:
        failures.append("pins changed during the gate")
    dump_fields = manifest("dump-t2va")["fields"]
    # Not acceptance criteria here (no memory limit is defined for this gate); recorded so they stay visible.
    open_items = []
    for n, m in native.items():
        metal = ((m.get("report") or {}).get("sampledPeak") or {}).get("metalAllocatedBytes")
        if metal and metal > 18 * (1 << 30):
            open_items.append(f"{n}: sampled Metal peak {metal / (1 << 30):.2f} GiB exceeds the 18 GiB bound earlier H3 probes used; "
                              "memory limits for the profile path are not yet defined (open memory-limit item)")
    gate = dict(schema=1, gate="h3-profile-graphs", completedAt=now(), passed=not failures, failures=failures, openItems=open_items,
                predefinedRelativeL2Limit=LIMIT, canvas=dict(width=args.width, height=args.height, frames=int(dump_fields["frames"]),
                rows=int(dump_fields["rows"]), tokens=int(dump_fields["tokens"]), seed=args.seed, recipe=args.recipe),
                profileSHA256=dump_fields["profile_sha256"], comparisons=comparisons,
                tokenizer=dict(tokenIdsEqualTransformers=text_report.get("token_ids_equal_transformers"), tapSeconds=text_report.get("tap_seconds")),
                layoutChecks=layout, oracleReports={n: manifest(n)["report"] for n in ("oracle-graphs-t2va", "oracle-graphs-fl2va")},
                oracleFields={n: manifest(n)["fields"] for n in ("oracle-text", "oracle-graphs-t2va", "oracle-graphs-fl2va")},
                nativeReports={n: m["report"] for n, m in native.items()}, controllers=records,
                pinsDigest=pins_digest, pinsUnchanged=pins_unchanged, filesSHA256=pins,
                scope=("Isolated native graphs (conditioner tap, prepare text/pack, blocks 0 and 49 at their real step-0 contexts "
                       "from the native chain, heads on the native block-49 output, one 256 px tile window, fixed audio decoder, "
                       "first-frame encoder) against the original-precision CPU modules on identical input bytes. Step 0 only "
                       "(one timestep cardinality). "
                       "Not a denoising trajectory, decoded clip, lifecycle, memory-limit or performance qualification."))
    (work / "gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps(dict(passed=gate["passed"], failures=failures, gate=str(work / "gate.json")), indent=2), flush=True)
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

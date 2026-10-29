#!/usr/bin/env python3
"""Own the H3 memory gate: complete renders under 20 ms memory timelines, held to the 18 GiB bar.

For each task (T2VA, FL2VA) and canvas, as owned process groups joined in order:

  dump-TASK-CANVAS     aether-probe h3-profile-dump (CPU).
  render-TASK-CANVAS   aether-probe h3-memory-render (GPU, under ~/.cache/aether-kit/with-gpu-lock.sh): a complete
                       H3VideoSession render; the timeline is split at the session's stage events.

The gate attributes every sample to its stage (a stage includes memory the previous stage has not yet returned),
records each stage's Metal and footprint peak, and fails if any render's Metal peak exceeds the bar. It writes the
measured resource entry for `H3MeasuredResources` (stage peaks are the maximum over every render; operation bytes are
the largest footprint excess over Metal during a render), which `H3ProfileEngine.measuredAdmission` turns into stage
reservations covering every measured peak. Metal is MTLDevice.currentAllocatedSize; footprint is the process's
physical footprint; they overlap and cannot be added. Sampling every 20 ms can miss shorter peaks.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

BAR = 18 << 30                 # today's region: at most 960x544's pixels and 124 frames
EXTENDED_BAR = 24 << 30        # beyond it (a larger canvas or a longer clip), decided 2026-10-29
EXTENDED_PHYSICAL = 64 << 30   # physical memory a device needs to use an extended-bar entry
MARGIN = 256 << 20
STAGES = ["encoding", "encodingKeyframe", "denoising", "decodingVideo", "decodingAudio"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scrub(value):
    """A record for the public repository: this Mac's home directory and volumes become placeholders."""
    text = json.dumps(value)
    text = text.replace(str(Path.home()), "<home>")
    for volume in sorted(Path("/Volumes").glob("*"), key=lambda v: -len(str(v))):
        text = text.replace(str(volume), "<volume>")
    return json.loads(text)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def stage_of(phase):
    if phase.startswith("stage."):
        name = phase.split(".")[1]
        return name if name in STAGES else None
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--prompt", type=Path, required=True)
    ap.add_argument("--keyframes", nargs="+", type=Path, required=True, help="first frames, one per canvas, named by their exact size")
    ap.add_argument("--canvases", default="960x544,544x960")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1043)
    ap.add_argument("--recipe", default="turbo8")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--gpu-lock", type=Path, default=Path.home() / ".cache/aether-kit/with-gpu-lock.sh")
    ap.add_argument("--spike-capture", type=float, metavar="GIB",
                    help="pass --spike-capture=GIB to every render: a one-shot thread sample, vmmap and footprint the first time a "
                         "denoising sample's Metal allocation exceeds GIB (written beside the render as render-*.spike/)")
    ap.add_argument("--carry-peaks", nargs="*", type=Path, default=[], metavar="RECORD",
                    help="earlier gate records whose per-stage peaks the measured entry must also cover (each is cited in the record)")
    ap.add_argument("--record", help="also write the gate record to docs/evidence/RECORD (a new file name); the measured entry cites it")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    # Each gate is held to one bar, chosen by its region: an extended entry needs a larger canvas or a longer clip.
    pixels = max(int(w) * int(h) for w, h in (c.split("x") for c in args.canvases.split(",")))
    snapped = 17 * ((max(0, args.frames - 5) + 16) // 17) + 5
    bar = EXTENDED_BAR if pixels > 960 * 544 or snapped > 124 else BAR
    probe = root / ".build/release/aether-probe"   # both build engines link .build/release
    work = args.work.resolve()
    (work / "controllers").mkdir(parents=True, exist_ok=True)
    # Render pins: everything a render depends on. This script's evaluation is pinned separately (scriptSHA256), so
    # a revised evaluation can be rerun over the same joined renders without re-rendering.
    files = [probe, root / "Package.swift", args.profile.resolve(), args.prompt.resolve(), *[k.resolve() for k in args.keyframes]]
    for directory in ["Sources/AetherKit", "Sources/AetherCoreAI", "Sources/AetherH3", "Sources/AetherH3CoreAI", "Sources/AetherProbe"]:
        files.extend(sorted(p for p in (root / directory).rglob("*") if p.is_file() and p.name != "h3-measured-resources.json"))
    pins = {str(p): sha(p) for p in files}
    digest = hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()
    pins_path = work / "pins.json"
    accepted = {digest}
    if pins_path.exists():
        stored = json.loads(pins_path.read_text())
        previous = dict(stored["filesSHA256"])
        if stored.get("schema") == 1:
            # Schema-1 pins also held this script and the measured-resource data; renders made under them stay valid
            # when every render input is unchanged.
            accepted.add(hashlib.sha256(json.dumps(stored["filesSHA256"], sort_keys=True).encode()).hexdigest())
            previous = {k: v for k, v in previous.items() if k != str(Path(__file__).resolve()) and not k.endswith("h3-measured-resources.json")}
        if previous != pins:
            ap.error(f"{work} was started with different render pins; use a new --work directory")
        stored.update(schema=2, filesSHA256=pins, digest=digest, acceptedDigests=sorted(accepted))
        pins_path.write_text(json.dumps(stored, indent=2) + "\n")
    else:
        pins_path.write_text(json.dumps(dict(schema=2, createdAt=now(), filesSHA256=pins, digest=digest, acceptedDigests=[digest]), indent=2) + "\n")
    accepted |= set(json.loads(pins_path.read_text()).get("acceptedDigests", []))
    keyframes = {}
    for k in args.keyframes:
        keyframes[k.stem.split("-")[-1]] = k.resolve()

    runs, records = [], {}
    for canvas in args.canvases.split(","):
        width, height = canvas.split("x")
        for task in ("t2va", "fl2va"):
            if task == "fl2va" and canvas not in keyframes:
                ap.error(f"no first frame for {canvas}")
            stem = f"{task}-{canvas}"
            dump = [str(probe), "h3-profile-dump", str(args.profile), str(args.prompt), width, height, str(args.frames), str(args.seed), args.recipe,
                    str(work / f"dump-{stem}")] + ([str(keyframes[canvas])] if task == "fl2va" else [])
            render = [str(args.gpu_lock), f"h3-memory-gate: {stem}", str(probe), "h3-memory-render", str(args.profile), str(work / f"dump-{stem}"),
                      str(work / f"render-{stem}")] + ([f"--spike-capture={args.spike_capture}"] if args.spike_capture else [])
            for name, command, budget in ((f"dump-{stem}", dump, 600), (f"render-{stem}", render, 7200)):
                record_path = work / "controllers" / f"{name}.json"
                if record_path.exists():
                    previous = json.loads(record_path.read_text())
                    if previous.get("passed") and previous.get("pinsDigest") in accepted:
                        records[name] = previous
                        continue
                output = work / name
                if output.exists():
                    output.rename(work / f"{name}.failed-{int(time.time())}")
                record = dict(schema=1, stage=name, command=command, budgetSeconds=budget, pinsDigest=digest, startedAt=now(),
                              gpuLock=name.startswith("render"), passed=False, timedOut=False)
                print(f"{name}: start", flush=True)
                start = time.monotonic()
                with (work / "controllers" / f"{name}.stdout.log").open("wb") as out, (work / "controllers" / f"{name}.stderr.log").open("wb") as err:
                    proc = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
                    try:
                        proc.wait(timeout=budget)
                    except subprocess.TimeoutExpired:
                        record["timedOut"] = True
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                record.update(exitCode=proc.returncode, joined=True, elapsedSeconds=round(time.monotonic() - start, 3), endedAt=now(),
                              sourcePinsUnchanged=all(sha(p) == v for p, v in pins.items()))
                record["passed"] = proc.returncode == 0 and not record["timedOut"] and record["sourcePinsUnchanged"]
                record_path.write_text(json.dumps(record, indent=2) + "\n")
                print(f"{name}: passed={record['passed']} {record['elapsedSeconds']} s", flush=True)
                records[name] = record
                if not record["passed"]:
                    return 1
            runs.append((task, canvas, work / f"render-{stem}" / "memory.json"))

    # Admission covers, per stage, the largest sample of max(Metal allocation, physical footprint): Metal can be
    # reserved without being touched, and footprint keeps memory Metal has already returned, so each sample takes the
    # larger. Because a stage's footprint already holds everything resident (the session's carried CPU state
    # included), operation bytes are only that carried state, sized from the clip's tensors: two copies of the
    # latents, the conditioning tap, one decoded window as float CHW and RGB8, and the PCM; rounded up to 64 MiB.
    failures, renders = [], {}
    stage_peaks, operation = {}, 0
    identities, architectures = set(), set()
    for task, canvas, path in runs:
        timeline = json.loads(path.read_text())
        fields, samples = timeline["fields"], timeline["samples"]
        identities.add(fields["execution_identity"]); architectures.add(fields["device_architecture"])
        if fields["task"] != task or f"{fields['width']}x{fields['height']}" != canvas:
            raise RuntimeError(f"{path}: timeline is for {fields['task']} {fields['width']}x{fields['height']}, expected {task} {canvas}")
        per_stage = {}
        for s in samples:
            stage = stage_of(s["phase"])
            if stage:
                p = per_stage.setdefault(stage, dict(metalPeakBytes=0, footprintPeakBytes=0, residentPeakBytes=0))
                p["metalPeakBytes"] = max(p["metalPeakBytes"], s["metal"]); p["footprintPeakBytes"] = max(p["footprintPeakBytes"], s["footprint"])
                p["residentPeakBytes"] = max(p["residentPeakBytes"], s["metal"], s["footprint"])
        in_render = [s for s in samples if s["phase"] not in ("start", "baseline")]
        metal_peak = max(s["metal"] for s in in_render)
        dump = json.loads((work / f"dump-{task}-{canvas}" / "manifest.json").read_text())
        width, height, frames = (int(dump["fields"][k]) for k in ("width", "height", "frames"))
        na = round(frames / 24 * 40)
        tensor_bytes = lambda name: (lambda e: 4 * math.prod(e["shape"]) if e["dtype"] == "float32" else 0)(dump["tensors"][name])
        carried = (2 * (tensor_bytes("latents_video") + tensor_bytes("latents_audio")) + int(dump["fields"]["tokens"]) * 5120 * 2
                   + 22 * width * height * (3 * 4 + 3) + 2 * 800 * na * 4)
        carried = -(-carried // (64 << 20)) * (64 << 20)
        renders[f"{task}-{canvas}"] = dict(metalPeakBytes=metal_peak, metalPeakGiB=round(metal_peak / (1 << 30), 3),
                                           footprintPeakGiB=round(max(s["footprint"] for s in in_render) / (1 << 30), 3),
                                           carriedStateBytes=carried,
                                           stages={k: dict(v, metalPeakGiB=round(v["metalPeakBytes"] / (1 << 30), 3),
                                                           footprintPeakGiB=round(v["footprintPeakBytes"] / (1 << 30), 3),
                                                           residentPeakGiB=round(v["residentPeakBytes"] / (1 << 30), 3)) for k, v in per_stage.items()},
                                           withinBar=metal_peak <= bar, timelineSHA256=sha(path))
        if metal_peak > bar:
            failures.append(f"{task} {canvas}: Metal peak {metal_peak / (1 << 30):.2f} GiB exceeds the {bar >> 30} GiB bar")
        for k, v in per_stage.items():
            stage_peaks[k] = max(stage_peaks.get(k, 0), v["residentPeakBytes"])
        operation = max(operation, carried)
    if len(identities) != 1 or len(architectures) != 1:
        failures.append(f"renders disagree on identity/device: {identities} {architectures}")
    # Peaks observed by earlier gates that this entry must still cover (for example a transient this gate did not see).
    # Only a passed gate on the same device architecture can be carried; each carried record is pinned by its SHA-256.
    carried_peaks, carried_records, carried_stage_bytes = [], [], {}
    for record_path in args.carry_peaks:
        earlier = json.loads(record_path.read_text())
        earlier_device = earlier["measuredResources"]["device_architecture"]
        if not earlier.get("passed"):
            failures.append(f"carried record {record_path.name} did not pass")
        if earlier_device not in architectures:
            failures.append(f"carried record {record_path.name} is for device {earlier_device}, not {architectures}")
        carried_records.append(dict(record=record_path.name, sha256=sha(record_path), passed=earlier.get("passed"), deviceArchitecture=earlier_device,
                                    executionIdentity=earlier["measuredResources"]["execution_identity"]))
        for render_name, render in earlier["renders"].items():
            for k, v in render["stages"].items():
                carried_stage_bytes[k] = max(carried_stage_bytes.get(k, 0), v["residentPeakBytes"])
                if v["residentPeakBytes"] > stage_peaks.get(k, 0):
                    carried_peaks.append(dict(stage=k, bytes=v["residentPeakBytes"], render=render_name, record=record_path.name,
                                              recordSHA256=sha(record_path),
                                              executionIdentity=earlier["measuredResources"]["execution_identity"], replaced=stage_peaks.get(k, 0)))
                    stage_peaks[k] = v["residentPeakBytes"]
    spikes = {}
    for task, canvas, path in runs:
        trigger = path.parent.with_name(path.parent.name + ".spike") / "trigger.json"
        spikes[f"{task}-{canvas}"] = json.loads(trigger.read_text()) if trigger.exists() else None
    worst = max(stage_peaks.values()) + MARGIN + operation
    if worst > bar:
        failures.append(f"admission estimate {worst / (1 << 30):.2f} GiB (largest stage + margin + operation) exceeds the {bar >> 30} GiB bar")
    pins_unchanged = all(sha(p) == v for p, v in pins.items())
    if not pins_unchanged:
        failures.append("pins changed during the gate")
    identity = identities.pop() if len(identities) == 1 else None
    architecture = architectures.pop() if len(architectures) == 1 else None
    maximum_frames = int(json.loads(runs[0][2].read_text())["fields"]["frames"])
    # The entries cite the record this run writes (docs/evidence/RECORD); without --record they are not bundleable.
    evidence = f"docs/evidence/{args.record}" if args.record else "unrecorded"

    def measured_entry(stage_bytes, operation_bytes, tasks=None):
        e = dict(execution_identity=identity, device_architecture=architecture, canvases=args.canvases.split(","),
                 maximum_frames=maximum_frames, stage_peak_bytes={k: stage_bytes[k] for k in STAGES if k in stage_bytes},
                 operation_bytes=operation_bytes, evidence=evidence)
        if tasks:
            e["tasks"] = tasks
        if bar != BAR:
            e["memory_bar_bytes"] = bar
        return e
    entry = measured_entry(stage_peaks, operation)
    # One entry per task, so an FL2VA entry can be re-measured and replaced without touching T2VA's. Carried peaks
    # apply to the combined entry only.
    per_task = {}
    for task in sorted({t for t, _, _ in runs}):
        peaks, carried_bytes = {}, 0
        for name, render in renders.items():
            if name.startswith(task + "-"):
                for k, v in render["stages"].items():
                    peaks[k] = max(peaks.get(k, 0), v["residentPeakBytes"])
                carried_bytes = max(carried_bytes, render["carriedStateBytes"])
        # Carried records cover every task: an earlier record's peak for a stage this task uses must still be reserved.
        for k, carried in carried_stage_bytes.items():
            if k in peaks:
                peaks[k] = max(peaks[k], carried)
        task_worst = max(peaks.values()) + MARGIN + carried_bytes
        if task_worst > bar:
            failures.append(f"{task} admission estimate {task_worst / (1 << 30):.2f} GiB exceeds the {bar >> 30} GiB bar")
        per_task[task] = dict(entry=measured_entry(peaks, carried_bytes, [task]), admissionWorstCaseBytes=task_worst)
    bundle = json.loads(args.profile.read_text())
    gate = dict(schema=1, gate="h3-memory", completedAt=now(), passed=not failures, failures=failures,
                spikeCapture=dict(thresholdGiB=args.spike_capture, triggers=spikes) if args.spike_capture else None,
                carriedRecords=carried_records, carriedStagePeaks=carried_peaks,
                memoryBarBytes=bar, minimumPhysicalMemoryBytes=EXTENDED_PHYSICAL if bar != BAR else 0,
                memoryBar=(f"Metal allocation peak over a complete render must not exceed {bar >> 30} GiB, for every task and canvas; "
                           "18 GiB up to 960x544's pixels and 124 frames, 24 GiB beyond (with at least 64 GiB of physical memory)"),
                profileAdapter=bundle.get("adapter"), profileLineage=bundle.get("lineage"),
                marginBytes=MARGIN, admissionWorstCaseBytes=worst, renders=renders, measuredResources=entry, measuredEntriesByTask=per_task,
                controllers=records,
                pinsDigest=digest, pinsUnchanged=pins_unchanged, filesSHA256=pins, scriptSHA256=sha(Path(__file__).resolve()),
                scope=("Complete H3VideoSession renders (conditioner, first-frame encoder for FL2VA, eight Turbo steps, video and audio decode) "
                       "at the requested canvases and length; per-stage peaks attribute residual memory from the previous stage to the next stage."))
    (work / "gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    if args.record:
        record = root / "docs/evidence" / args.record
        if record.exists():
            raise SystemExit(f"{record} already exists; choose a new record name")
        # The repository copy is scrubbed of local paths (the work directory keeps the raw gate.json).
        record.write_text(json.dumps(scrub(gate), indent=2) + "\n")
    print(json.dumps(dict(passed=gate["passed"], failures=failures, worstGiB=round(worst / (1 << 30), 3),
                          renders={k: v["metalPeakGiB"] for k, v in renders.items()}), indent=2), flush=True)
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

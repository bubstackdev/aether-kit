#!/usr/bin/env python3
"""Own a bounded H3 eight-step denoising probe, preserving source pins and joined status.

Preparation is CPU-only. Native execution requires an explicit --run plus a
passing preparation report from the same executable, sources and manifests.
This script does not schedule a future GPU window or export/copy model weights.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--phases", type=Path, required=True)
    parser.add_argument("--cpu-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chain", type=Path, required=True)
    parser.add_argument("--modulations", type=Path, required=True)
    parser.add_argument("--references", default="-")
    parser.add_argument("--prepared", type=Path, help="Preparation output directory; required for --capture/--run")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    output = args.output.resolve()
    exe = root / ".build/arm64-apple-macosx/release/aether-probe"
    bundle, phases, cpu = (p.resolve() for p in (args.bundle, args.phases, args.cpu_record))
    suffixes = ["-controller.json", "-provenance.json", "-stdout.json", "-stderr.log"]
    if output.exists() or any(Path(str(output) + s).exists() for s in suffixes):
        parser.error("output and controller artifacts must be new")
    output.parent.mkdir(parents=True, exist_ok=True)
    chain = args.chain.resolve()
    paths = [exe, bundle, phases, cpu, chain, root / "Package.swift", Path(__file__).resolve()]
    if args.references != "-": paths.append(Path(args.references).resolve())
    for block in range(50):
        asset = args.modulations.resolve() / f"modulation-{block}.aimodel"
        paths.extend([asset / "main.hash", asset / "main.mlirb"])
    for directory in ["Sources/AetherH3", "Sources/AetherH3CoreAI", "Sources/AetherProbe",
                      "Tests/AetherH3Tests", "Tests/AetherH3CoreAITests", "Tests/AetherProbeTests"]:
        paths.extend(sorted(p for p in (root / directory).rglob("*") if p.suffix in [".swift", ".json"]))
    pins = {str(p): sha(p) for p in paths}
    if args.run or args.capture:
        if args.prepared is None:
            parser.error("--capture/--run needs --prepared PREPARATION_OUTPUT_DIRECTORY")
        prepared_path = args.prepared.resolve()
        ready = json.loads((prepared_path / "report.json").read_text())
        prepared_pins = json.loads(Path(str(prepared_path) + "-provenance.json").read_text())["filesSHA256"]
        prepared_controller = json.loads(Path(str(prepared_path) + "-controller.json").read_text())
        if not (prepared_controller.get("passed") and prepared_controller.get("joined")
                and prepared_controller.get("exitCode") == 0 and prepared_controller.get("timedOut") is False
                and prepared_controller.get("sourcePinsUnchanged") and prepared_controller.get("gpuWorkRequested") is False
                and ready.get("passed") and ready.get("stage") == "ready"
                and ready.get("selectedFilesVerified") and ready.get("nativeExecuted") is False
                and ready.get("bundleSHA256") == sha(bundle)
                and ready.get("phaseSHA256") == sha(phases)
                and ready.get("cpuRecordSHA256") == sha(cpu) and ready.get("chainSHA256") == sha(chain)
                and (not args.run or ready.get("readyForQualification"))
                and pins == prepared_pins):
            parser.error("preparation must match the current executable, sources and manifests")
    elif args.prepared is not None:
        parser.error("--prepared applies only to --capture/--run")
    provenance = Path(str(output) + "-provenance.json")
    provenance.write_text(json.dumps({"schema": 1, "filesSHA256": pins}, indent=2) + "\n")
    if args.run and args.references == "-": parser.error("--run requires CPU references")
    if args.capture and args.references != "-": parser.error("--capture must stay unqualified without references")
    command = [str(exe), "qualify-h3-denoising", str(bundle), str(phases), str(cpu), str(chain),
               str(args.modulations.resolve()), args.references if args.references == "-" else str(Path(args.references).resolve()), str(output),
               "--run" if args.run else "--capture" if args.capture else "--prepare-only"]
    budget = 1800 if args.run or args.capture else 300
    controller = dict(schema=1, startedAt=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      command=command, budgetSeconds=budget, gpuWorkRequested=args.run or args.capture,
                      passed=False, joined=False, timedOut=False)
    controller_path = Path(str(output) + "-controller.json")

    def save():
        controller_path.write_text(json.dumps(controller, indent=2) + "\n")

    start = time.monotonic()
    with Path(str(output) + "-stdout.json").open("xb") as stdout, Path(str(output) + "-stderr.log").open("xb") as stderr:
        proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
        controller["pid"] = proc.pid
        save()
        print(json.dumps({"pid": proc.pid, "budgetSeconds": budget, "gpuWorkRequested": args.run or args.capture}), flush=True)
        try:
            proc.wait(timeout=budget - 10)
        except subprocess.TimeoutExpired:
            controller["timedOut"] = True
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            controller.update(joined=True, exitCode=proc.returncode, elapsedSeconds=time.monotonic() - start)
            controller["sourcePinsUnchanged"] = all(sha(p) == digest for p, digest in pins.items())
            save()
    try:
        report = json.loads((output / "report.json").read_text())
        controller["passed"] = (proc.returncode == 0 and not controller["timedOut"]
                                and controller["sourcePinsUnchanged"]
                                and report.get("passed") is (not args.capture)
                                and report.get("stage") == ("complete" if args.run else "capturedAwaitingCPUReferences" if args.capture else "ready")
                                and report.get("nativeExecuted") is (args.run or args.capture)
                                and (not args.capture or report.get("captureComplete")))
    except (OSError, ValueError) as error:
        controller["reportError"] = str(error)
    save()
    print(json.dumps(controller, indent=2), flush=True)
    return 0 if controller["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

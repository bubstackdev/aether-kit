"""Run a numerical gate then one render inside a shared twenty-minute window.

Each child controller owns and joins its native process group. This coordinator
does not launch the render after a failed gate and forwards interruption to the
active controller. It never accepts output quality automatically.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    started = time.monotonic()
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("phase-binary", "phase-input", "render-binary", "render-input", "output", "ffmpeg", "cache"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--budget", type=int, default=1200)
    args = ap.parse_args()
    if not 31 <= args.budget <= 1200:
        ap.error("shared wall budget must be 31..1200 seconds")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    controller = Path(__file__).with_name("run-h3-check.py")
    report = dict(schema=1, passed=False, quality_accepted=False, wall_budget_seconds=args.budget,
                  scope="Numerical gate plus finished-render execution inside one shared GPU window; output quality requires separate review",
                  coordinator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  controller_sha256=hashlib.sha256(controller.read_bytes()).hexdigest(), stages=[])
    child = None

    def interrupted(_signal, _frame):
        raise InterruptedError("qualified-render coordinator interrupted")

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for kind, binary, source in [("phase", args.phase_binary, args.phase_input),
                                      ("render", args.render_binary, args.render_input)]:
            remaining = int(args.budget - (time.monotonic() - started)) - 1
            budget = min(180, remaining - 12) if kind == "phase" else remaining
            if budget < 11:
                raise TimeoutError("insufficient shared wall budget for another native stage")
            directory = output / kind
            command = [sys.executable, "-B", str(controller), "--kind", kind,
                       "--binary", str(binary.resolve()), "--input", str(source.resolve()),
                       "--output", str(directory), "--budget", str(budget)]
            if kind == "render":
                command += ["--ffmpeg", str(args.ffmpeg.resolve()), "--cache", str(args.cache.resolve())]
            print(f"H3 qualified render: starting {kind}, stage budget {budget}s", flush=True)
            child = subprocess.Popen(command)
            while child.poll() is None:
                if time.monotonic() - started >= args.budget - 2:
                    raise TimeoutError("shared window deadline reached")
                try:
                    child.wait(timeout=0.25)
                except subprocess.TimeoutExpired:
                    pass
            record_path = directory / "controller.json"
            record = json.loads(record_path.read_text())
            report["stages"].append(dict(kind=kind, controller=str(record_path),
                controller_sha256=hashlib.sha256(record_path.read_bytes()).hexdigest(),
                passed=record["passed"], elapsed_seconds=record["elapsed_seconds"],
                binary_sha256=record["binary_sha256"], input_sha256=record["input_sha256"]))
            (output / "coordinator.json").write_text(json.dumps(report, indent=2) + "\n")
            if child.returncode != 0 or not record["passed"] or not record["child_joined"]:
                raise RuntimeError(f"{kind} failed; no further stage will run")
            child = None
        report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
    finally:
        if child is not None and child.poll() is None:
            # The tested controller handles TERM by stopping and joining its own
            # native group, including any FFmpeg child, within its cleanup reserve.
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                report["controller_cleanup_pending"] = True
                # Its own preassigned deadline still bounds native work. Do not
                # kill the controller and orphan the native group it owns.
        report["elapsed_seconds"] = time.monotonic() - started
        if report["elapsed_seconds"] > args.budget:
            report["passed"] = False
            report["budget_exceeded"] = True
        (output / "coordinator.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=report["passed"], report=str(output / "coordinator.json"))), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

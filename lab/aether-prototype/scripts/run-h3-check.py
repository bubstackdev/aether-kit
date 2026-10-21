"""Own one H3 check/render and its media subprocesses under a wall budget.

Adapter/decoder/phase probes allow at most 300 seconds; complete renders at most
1200. Ten seconds are reserved for termination/join. Nothing starts until the
caller explicitly invokes this script in the native execution environment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


DRIVERS = {"labrender", "h3_generate", "h3_benchmark", "h3_turbo_probe", "h3_decode_probe", "h3_phase_probe"}


def conflicts(own_pid=None):
    listing = subprocess.run(["ps", "-axo", "pid=,comm="], capture_output=True, text=True, check=True, timeout=1)
    found = []
    for line in listing.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2 and int(fields[0]) != own_pid and Path(fields[1]).name in DRIVERS:
            found.append(dict(pid=int(fields[0]), executable=fields[1]))
    return found


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["adapter", "decoder", "phase", "render"], required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--ffmpeg", type=Path)
    args = parser.parse_args()
    maximum = 1200 if args.kind == "render" else 300
    if not 11 <= args.budget <= maximum:
        parser.error(f"budget must be 11..{maximum} seconds, including cleanup")
    if args.kind == "render" and (args.ffmpeg is None or not args.ffmpeg.is_file()):
        parser.error("a render requires --ffmpeg for its finished MP4")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary, source = args.binary.resolve(), args.input.resolve()
    command = [str(binary), str(source), str(output / "report.json")]
    if args.kind == "render":
        command += [str(output / "clip"), "--ffmpeg=" + str(args.ffmpeg.resolve())]
        if args.cache:
            command += ["--cache=" + str(args.cache.resolve())]
    record = dict(schema=1, passed=False, kind=args.kind, command=command,
                  wall_budget_seconds=args.budget, cleanup_reserve_seconds=10,
                  binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope="One explicitly invoked native process group; application report/exit and wall budget checked, no full environment capture")
    child = None

    def interrupted(_signal, _frame):
        raise InterruptedError("H3 controller received termination")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        record["conflicts_before"] = conflicts()
        if record["conflicts_before"]:
            raise RuntimeError("Another known renderer is active")
        with (output / "stdout.log").open("w") as out, (output / "stderr.log").open("w") as err:
            child = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
            (output / "progress.json").write_text(json.dumps(dict(pid=child.pid, process_group=child.pid, command=command)))
            sample_at = started
            while child.poll() is None:
                if time.monotonic() - started >= args.budget - 10:
                    raise TimeoutError("Wall budget reached; cleanup reserve begins")
                if time.monotonic() >= sample_at:
                    active = conflicts(child.pid)
                    if active:
                        record["conflicts_during_run"] = active
                        raise RuntimeError("Another known renderer started")
                    print(f"H3 {args.kind}: {time.monotonic() - started:.0f} seconds", flush=True)
                    sample_at = time.monotonic() + 15
                try:
                    child.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
            report = json.loads((output / "report.json").read_text())
            record["application_passed"] = report["passed"]
            record["passed"] = child.returncode == 0 and report["passed"]
            if args.kind == "render" and not (output / "clip/clip.mp4").is_file():
                raise RuntimeError("Render exited without a finished MP4")
    except Exception as error:
        record["passed"] = False
        record["error"] = str(error)
    finally:
        if child is not None:
            # Terminate the owned group on failure, including an in-flight mux
            # if the native parent already died. Never signal other renderers.
            if not record["passed"]:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            child.wait()
            record["exit_status"] = child.returncode
        record["child_joined"] = child is None or child.poll() is not None
        record["elapsed_seconds"] = time.monotonic() - started
        if record["elapsed_seconds"] > args.budget:
            record["passed"] = False
            record["budget_exceeded"] = True
        (output / "controller.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(dict(passed=record["passed"], report=str(output / "controller.json"))), flush=True)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

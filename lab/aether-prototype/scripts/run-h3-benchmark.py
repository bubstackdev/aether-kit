"""Own one bounded native benchmark child and preserve its exit independently of RPCs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import signal
import subprocess
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def host_state(pid=None):
    commands = {"memory": ["/usr/bin/vm_stat"], "thermal": ["/usr/bin/pmset", "-g", "therm"]}
    if pid is not None:
        commands["process"] = ["/bin/ps", "-p", str(pid), "-o", "pid=,etime=,stat=,rss=,pcpu="]
    result = {"utc": datetime.now(timezone.utc).isoformat()}
    for name, command in commands.items():
        p = subprocess.run(command, capture_output=True, text=True, timeout=5)
        result[name] = {"status": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    return result


def conflicting_processes(own_child=None):
    """Known native render drivers only; never stop another owner's process."""
    result = subprocess.run(["/bin/ps", "-axo", "pid=,comm="], capture_output=True, text=True, timeout=5, check=True)
    conflicts = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2 and int(fields[0]) != own_child and Path(fields[1]).name in {"labrender", "h3_generate", "h3_benchmark"}:
            conflicts.append({"pid": int(fields[0]), "executable": fields[1]})
    return conflicts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--deadline", type=int, default=900)
    args = parser.parse_args()
    if not 3 <= args.steps <= 50 or not 1 <= args.deadline <= 900:
        parser.error("steps must be 3..50 and deadline 1..900 seconds")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary, bundle = args.binary.resolve(), args.bundle.resolve()
    command = [str(binary), str(bundle), str(output / "report.json"), str(args.steps)]
    record = {"schema": 1, "passed": False, "command": command, "binary_sha256": sha(binary), "bundle_sha256": sha(bundle), "controller_sha256": sha(Path(__file__)), "deadline_seconds": args.deadline, "host_before": host_state(), "host_samples": [], "scope": "One untraced benchmark child; explicit wall deadline, exit status and cleanup. No cache purge, clock control or GPU concurrency introduced."}
    started = time.monotonic()
    child = None

    def stop(_signal, _frame):
        raise InterruptedError("benchmark controller received termination")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        record["conflicts_before"] = conflicting_processes()
        if record["conflicts_before"]:
            raise RuntimeError(f"Known render driver already active: {record['conflicts_before']}")
        with (output / "stdout.log").open("w") as out, (output / "stderr.log").open("w") as err:
            child = subprocess.Popen(command, stdout=out, stderr=err)
            (output / "progress.json").write_text(json.dumps({"pid": child.pid, "command": command, "started_utc": datetime.now(timezone.utc).isoformat()}))
            sample_at = started
            while child.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed > args.deadline:
                    raise TimeoutError("native benchmark exceeded its wall deadline")
                if time.monotonic() >= sample_at:
                    conflicts = conflicting_processes(child.pid)
                    if conflicts:
                        record["conflicts_during_run"] = conflicts
                        raise RuntimeError(f"Known render driver started during benchmark: {conflicts}")
                    record["host_samples"].append(host_state(child.pid))
                    sample_at = time.monotonic() + 30
                    print(f"benchmark child {child.pid}: {elapsed:.0f} seconds", flush=True)
                try:
                    child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            record["exit_status"] = child.returncode
            report = json.loads((output / "report.json").read_text())
            record["passed"] = child.returncode == 0 and report["passed"]
    except Exception as error:
        record["error"] = str(error)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if child is not None:
            record["exit_status"] = child.returncode
        record["child_joined"] = child is None or child.poll() is not None
        record["elapsed_seconds"] = time.monotonic() - started
        record["host_after"] = host_state()
        (output / "controller.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"passed": record["passed"], "report": str(output / "controller.json")}), flush=True)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

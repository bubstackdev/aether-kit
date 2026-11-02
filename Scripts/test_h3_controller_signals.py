#!/usr/bin/env python3
"""Tests that each H3 qualification controller stops its detached workload when it is terminated.

Each controller runs in a scratch copy of the repository whose aether-probe is a stub that records its process group
and sleeps; no model, GPU or network is used. Run: python3 -I Scripts/test_h3_controller_signals.py"""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import h3_controller  # noqa: E402
# With AETHER_TEST_STUB_IGNORE_TERM the stub ignores SIGTERM, so a controller's cleanup sits in its grace period.
STUB = '#!/bin/sh\n[ -n "$AETHER_TEST_STUB_IGNORE_TERM" ] && trap "" TERM\necho $$ > "$AETHER_TEST_STUB_PGID"\nexec sleep 60\n'


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class ControllerTerminationTests(unittest.TestCase):
    """The local-line review's M1: SIGTERM or SIGHUP to a controller must not leave its workload running."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="h3-controller-signals-"))
        self.stubs = []
        (self.root / "Scripts").mkdir()
        for name in ("prepare-h3-profile-oracle.py", "h3_export_lib.py"):
            (self.root / "Scripts" / name).write_text("")
        shutil.copy(SCRIPTS / "h3_controller.py", self.root / "Scripts" / "h3_controller.py")
        (self.root / "Package.swift").write_text("")
        for build in (".build/release", ".build/arm64-apple-macosx/release"):
            (self.root / build).mkdir(parents=True)
            probe = self.root / build / "aether-probe"
            probe.write_text(STUB)
            probe.chmod(0o755)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        for name in ("bundle.json", "phases.json", "cpu.json", "chain.json", "profile.json", "prompt.txt", "first-960x544.png"):
            (self.inputs / name).write_text("{}")
        for block in range(50):
            asset = self.inputs / "modulations" / f"modulation-{block}.aimodel"
            asset.mkdir(parents=True)
            (asset / "main.hash").write_text("")
            (asset / "main.mlirb").write_text("")
        for path in ("staging/oracle-scripts/vendor/convert_minimax_h3_to_diffusers.py",
                     "staging/exports/h3-fl2va/converted/vae/config.json", "staging/exports/h3-fl2va/converted/audio-vae/config.json"):
            (self.inputs / path).parent.mkdir(parents=True, exist_ok=True)
            (self.inputs / path).write_text("{}")

    def tearDown(self):
        for pgid in self.stubs:   # only the stub groups this test started, should a check have failed
            if group_alive(pgid):
                os.killpg(pgid, signal.SIGKILL)
        shutil.rmtree(self.root, ignore_errors=True)

    def terminate(self, script, arguments, sig=signal.SIGTERM, again=()):
        """Starts `script` with `arguments`, waits for its stub workload, sends `sig` to the controller alone and
        returns the stub's process group once the controller has exited. With `again`, the stub ignores SIGTERM, so the
        controller is still in its cleanup grace period when those further signals arrive a second later: they must
        not cut the cleanup short, so the controller exits only after its SIGKILL at the grace deadline."""
        shutil.copy(SCRIPTS / script, self.root / "Scripts" / script)
        pgid_file = self.root / f"stub-{script}.pgid"
        env = dict(os.environ, AETHER_TEST_STUB_PGID=str(pgid_file))
        if again:
            env["AETHER_TEST_STUB_IGNORE_TERM"] = "1"
        controller = subprocess.Popen([sys.executable, "-I", str(self.root / "Scripts" / script), *arguments], env=env,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            deadline = time.monotonic() + 30
            while not (pgid_file.exists() and pgid_file.read_text().strip()):
                self.assertIsNone(controller.poll(), f"{script} exited before starting its workload")
                self.assertLess(time.monotonic(), deadline, f"{script} never started its workload")
                time.sleep(0.1)
            pgid = int(pgid_file.read_text())
            self.stubs.append(pgid)
            self.assertTrue(group_alive(pgid))
            controller.send_signal(sig)
            if again:
                time.sleep(1)
                self.assertIsNone(controller.poll(), f"{script} finished its cleanup without waiting for the workload")
            for extra in again:
                if controller.poll() is None:
                    controller.send_signal(extra)
                    time.sleep(0.2)
            controller.wait(timeout=40)
        finally:
            if controller.poll() is None:
                controller.kill()
                controller.wait()
        self.assertNotEqual(controller.returncode, 0)
        self.assertFalse(group_alive(pgid), f"{script}'s workload outlived the controller after signal {sig}")
        return pgid

    def i(self, name):
        return str(self.inputs / name)

    def test_phases(self):
        output = self.root / "out/phases"
        self.terminate("qualify-h3-phases.py", ["--bundle", self.i("bundle.json"), "--phases", self.i("phases.json"),
                                                "--cpu-record", self.i("cpu.json"), "--output", str(output), "--prepare-only"])
        record = json.loads(Path(str(output) + "-controller.json").read_text())
        self.assertTrue(record["interrupted"] and record["joined"] and not record["passed"])

    def test_denoising(self):
        output = self.root / "out/denoising"
        self.terminate("qualify-h3-denoising.py", ["--bundle", self.i("bundle.json"), "--phases", self.i("phases.json"),
                                                   "--cpu-record", self.i("cpu.json"), "--chain", self.i("chain.json"),
                                                   "--modulations", self.i("modulations"), "--output", str(output), "--prepare-only"],
                       sig=signal.SIGHUP)
        record = json.loads(Path(str(output) + "-controller.json").read_text())
        self.assertTrue(record["interrupted"] and record["joined"] and not record["passed"])

    def staged(self, work):
        return ["--profile", self.i("profile.json"), "--python", sys.executable, "--staging", self.i("staging"),
                "--prompt", self.i("prompt.txt"), "--work", str(work), "--minimum-free-percent", "0"]

    def interrupted_records(self, work):
        records = [json.loads(p.read_text()) for p in (work / "controllers").glob("*.json")]
        return [r for r in records if r.get("interrupted")]

    def test_profile_graphs(self):
        work = self.root / "graphs"
        self.terminate("qualify-h3-profile-graphs.py", self.staged(work) + ["--keyframe", self.i("first-960x544.png")])
        self.assertTrue(self.interrupted_records(work))

    def test_profile_trajectory(self):
        work = self.root / "trajectory"
        self.terminate("qualify-h3-profile-trajectory.py", ["--task", "t2va"] + self.staged(work))
        self.assertTrue(self.interrupted_records(work))

    def test_memory_gate(self):
        work = self.root / "memory"
        self.terminate("qualify-h3-memory.py", ["--profile", self.i("profile.json"), "--prompt", self.i("prompt.txt"),
                                                "--keyframes", self.i("first-960x544.png"), "--canvases", "960x544",
                                                "--work", str(work)])
        records = self.interrupted_records(work)
        self.assertTrue(records and all(r["joined"] and not r["passed"] for r in records))


    # A second (and third) signal during the cleanup grace period: SIGTERM, SIGINT and SIGHUP alike.
    def test_phases_second_signal(self):
        output = self.root / "out/phases-twice"
        self.terminate("qualify-h3-phases.py", ["--bundle", self.i("bundle.json"), "--phases", self.i("phases.json"),
                                                "--cpu-record", self.i("cpu.json"), "--output", str(output), "--prepare-only"],
                       again=(signal.SIGTERM,))
        record = json.loads(Path(str(output) + "-controller.json").read_text())
        self.assertTrue(record["interrupted"] and record["joined"] and not record["passed"])

    def test_denoising_second_signal(self):
        output = self.root / "out/denoising-twice"
        self.terminate("qualify-h3-denoising.py", ["--bundle", self.i("bundle.json"), "--phases", self.i("phases.json"),
                                                   "--cpu-record", self.i("cpu.json"), "--chain", self.i("chain.json"),
                                                   "--modulations", self.i("modulations"), "--output", str(output), "--prepare-only"],
                       sig=signal.SIGINT, again=(signal.SIGINT, signal.SIGHUP))
        record = json.loads(Path(str(output) + "-controller.json").read_text())
        self.assertTrue(record["interrupted"] and record["joined"] and not record["passed"])

    def test_profile_graphs_second_signal(self):
        work = self.root / "graphs-twice"
        self.terminate("qualify-h3-profile-graphs.py", self.staged(work) + ["--keyframe", self.i("first-960x544.png")],
                       again=(signal.SIGHUP,))
        self.assertTrue(self.interrupted_records(work))

    def test_profile_trajectory_second_signal(self):
        work = self.root / "trajectory-twice"
        self.terminate("qualify-h3-profile-trajectory.py", ["--task", "t2va"] + self.staged(work), again=(signal.SIGTERM,))
        self.assertTrue(self.interrupted_records(work))

    def test_memory_gate_second_signal(self):
        work = self.root / "memory-twice"
        self.terminate("qualify-h3-memory.py", ["--profile", self.i("profile.json"), "--prompt", self.i("prompt.txt"),
                                                "--keyframes", self.i("first-960x544.png"), "--canvases", "960x544",
                                                "--work", str(work)], again=(signal.SIGTERM, signal.SIGTERM))
        records = self.interrupted_records(work)
        self.assertTrue(records and all(r["joined"] and not r["passed"] for r in records))


def reset_controller_state():
    h3_controller.pending.clear()
    h3_controller._state.update(exiting=False, starting=False, stopping=False)


class StopTests(unittest.TestCase):
    """h3_controller.stop() waits for the whole group, not just its leader."""

    def setUp(self):
        reset_controller_state()
        self.directory = Path(tempfile.mkdtemp(prefix="h3-controller-stop-"))
        self.procs = []

    def tearDown(self):
        for proc in self.procs:   # only groups this test started, should a check have failed
            if group_alive(proc.pid):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.poll()
        reset_controller_state()
        shutil.rmtree(self.directory, ignore_errors=True)

    def start(self, script):
        proc = h3_controller.start(["/bin/sh", "-c", script])
        self.procs.append(proc)
        time.sleep(0.5)   # let the shells set their traps
        return proc

    def test_a_leader_that_exits_on_sigterm_does_not_end_the_wait(self):
        # lockf leads a GPU wrapper's group and exits at once on SIGTERM while its child runs on.
        proc = self.start('(trap "" TERM; exec sleep 30) & wait')
        started = time.monotonic()
        self.assertTrue(h3_controller.stop(proc, grace=2))
        self.assertFalse(group_alive(proc.pid))
        self.assertGreaterEqual(time.monotonic() - started, 1.9, "the TERM-ignoring child was killed before the grace period")

    def test_a_wrapper_releases_before_the_group_is_killed_even_after_a_second_signal(self):
        # Like with-gpu-lock.sh: the leader exits on SIGTERM; the wrapper shell stops its TERM-ignoring command with its
        # own SIGKILL a second later, then releases. A second signal meanwhile must not kill the wrapper first.
        released = self.directory / "released"
        wrapper = self.directory / "wrapper.sh"
        wrapper.write_text("""trap 'kill -TERM $c; sleep 1; kill -KILL $c; wait $c; echo released > "$1"; exit 130' TERM\n"""
                           """sh -c 'trap "" TERM; exec sleep 30' &\nc=$!\nwait $c\n""")
        proc = self.start(f'sh {wrapper} {released} & wait')
        previous = {sig: signal.getsignal(sig) for sig in h3_controller.SIGNALS}
        h3_controller.install()
        try:
            timer = threading.Timer(0.3, lambda: os.kill(os.getpid(), signal.SIGTERM))
            timer.start()
            started = time.monotonic()
            joined = h3_controller.stop(proc, grace=5)
            timer.join()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        self.assertTrue(joined and not group_alive(proc.pid))
        self.assertTrue(released.exists(), "the wrapper was killed before it released")
        self.assertLess(time.monotonic() - started, 5, "the group should empty once the wrapper releases")
        self.assertTrue(h3_controller.interrupted(), "the second signal is queued, not lost")

    def test_an_interrupted_controller_starts_nothing(self):
        marker = self.directory / "started"
        h3_controller.pending.append(signal.SIGTERM)
        with self.assertRaises(KeyboardInterrupt):
            h3_controller.start(["/bin/sh", "-c", f"touch {marker}"])
        time.sleep(0.3)
        self.assertFalse(marker.exists())

if __name__ == "__main__":
    unittest.main()

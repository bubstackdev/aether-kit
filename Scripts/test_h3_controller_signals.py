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
import time
import unittest

SCRIPTS = Path(__file__).resolve().parent
STUB = '#!/bin/sh\necho $$ > "$AETHER_TEST_STUB_PGID"\nexec sleep 60\n'


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

    def terminate(self, script, arguments, sig=signal.SIGTERM):
        """Starts `script` with `arguments`, waits for its stub workload, sends `sig` to the controller alone and
        returns the stub's process group once the controller has exited."""
        shutil.copy(SCRIPTS / script, self.root / "Scripts" / script)
        pgid_file = self.root / f"stub-{script}.pgid"
        env = dict(os.environ, AETHER_TEST_STUB_PGID=str(pgid_file))
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
            controller.wait(timeout=30)
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


if __name__ == "__main__":
    unittest.main()

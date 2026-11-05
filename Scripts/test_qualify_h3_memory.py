#!/usr/bin/env python3
"""Tests for Scripts/qualify-h3-memory.py's supervision and resume rules, with harmless sleepers and no GPU work.

Run: python3 -I Scripts/test_qualify_h3_memory.py"""
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("qualify_h3_memory", os.path.join(os.path.dirname(os.path.abspath(__file__)), "qualify-h3-memory.py"))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class TaskSelectionTests(unittest.TestCase):
    """0.0.7 review: --tasks selects renders, and an entry claims only what its own renders measured."""

    def test_every_task_by_default_and_first_last_only_with_a_last_keyframe(self):
        self.assertEqual(gate.planned_renders("t2va,fl2va,fl2va-firstlast", ["704x704"], {"704x704": "last.png"}),
                         [("t2va", "704x704"), ("fl2va", "704x704"), ("fl2va-firstlast", "704x704")])
        self.assertEqual(gate.planned_renders("t2va,fl2va,fl2va-firstlast", ["960x544", "544x960"], {}),
                         [("t2va", "960x544"), ("fl2va", "960x544"), ("t2va", "544x960"), ("fl2va", "544x960")])

    def test_first_last_without_fl2va_is_refused(self):
        with self.assertRaisesRegex(ValueError, "fl2va-firstlast without fl2va"):
            gate.planned_renders("fl2va-firstlast", ["704x704"], {"704x704": "last.png"})
        with self.assertRaisesRegex(ValueError, "fl2va-firstlast without fl2va"):
            gate.planned_renders("t2va,fl2va-firstlast", ["704x704"], {"704x704": "last.png"})

    def test_a_selection_with_no_render_is_refused(self):
        # fl2va-firstlast needs a last keyframe, so with fl2va filtered too nothing is left; an empty or unknown list refuses.
        with self.assertRaisesRegex(ValueError, "must name"):
            gate.planned_renders("", ["704x704"], {})
        with self.assertRaisesRegex(ValueError, "must name"):
            gate.planned_renders("t2va,flva", ["704x704"], {})
        # A valid selection that still leaves nothing (no canvas) refuses before anything runs, never at runs[0].
        with self.assertRaisesRegex(ValueError, "leaves no render"):
            gate.planned_renders("t2va,fl2va", [], {})

    def test_an_entry_claims_a_task_only_from_its_own_render_on_every_canvas(self):
        self.assertEqual(gate.claimed_tasks([("fl2va", "704x704"), ("fl2va-firstlast", "704x704")], ["704x704"]), ["fl2va"])
        self.assertEqual(gate.claimed_tasks([("fl2va-firstlast", "704x704")], ["704x704"]), [])
        self.assertEqual(gate.claimed_tasks([("t2va", "960x544"), ("fl2va", "960x544"), ("t2va", "544x960")], ["960x544", "544x960"]), ["t2va"])
        self.assertEqual(gate.claimed_tasks([("t2va", "704x704"), ("fl2va", "704x704")], ["704x704"]), ["fl2va", "t2va"])


class SupervisionTests(unittest.TestCase):
    """Review M7: an interrupted gate stops and joins the workload's own process group."""

    def run_sleeper(self, interrupt_after=None, budget=30, script="import time; time.sleep(30)"):
        # The sleeper writes its process-group id first, so the test can check the whole group afterwards.
        record = dict(timedOut=False)
        with tempfile.TemporaryDirectory() as directory:
            pgid_path = Path(directory) / "pgid"
            command = [sys.executable, "-I", "-c", f"import os; open({str(pgid_path)!r}, 'w').write(str(os.getpgrp())); {script}"]
            timer = threading.Timer(interrupt_after, lambda: os.kill(os.getpid(), signal.SIGINT)) if interrupt_after else None
            raised = None
            with open(os.devnull, "wb") as out:
                if timer:
                    timer.start()
                try:
                    gate.run_owned(command, out, out, budget, record)
                except KeyboardInterrupt as error:
                    raised = error
                finally:
                    if timer:
                        timer.cancel()
            pgid = int(pgid_path.read_text()) if pgid_path.exists() else None
        return record, raised, pgid

    def test_interrupt_stops_and_joins_the_workload_group(self):
        start = time.monotonic()
        record, raised, pgid = self.run_sleeper(interrupt_after=1)
        self.assertIsInstance(raised, KeyboardInterrupt)
        self.assertTrue(record["interrupted"] and record["joined"])
        self.assertIsNotNone(record["exitCode"])
        self.assertIsNotNone(pgid)
        self.assertFalse(group_alive(pgid), "the workload outlived the interrupted gate")
        self.assertLess(time.monotonic() - start, 20)

    def test_budget_stops_the_workload_group(self):
        record, raised, pgid = self.run_sleeper(budget=1)
        self.assertIsNone(raised)
        self.assertTrue(record["timedOut"] and record["joined"])
        self.assertFalse(group_alive(pgid))

    def test_a_completed_workload_reports_its_exit_code(self):
        record, raised, _ = self.run_sleeper(script="raise SystemExit(3)")
        self.assertIsNone(raised)
        self.assertEqual(record["exitCode"], 3)
        self.assertFalse(record["timedOut"] or record.get("interrupted"))


class ResumeTests(unittest.TestCase):
    """Review L4: a resumed gate reuses a stage only for the same workload."""

    def test_a_stage_resumes_only_for_the_same_command(self):
        command = ["probe", "h3-profile-dump", "profile", "prompt", "960", "544", "120", "1043", "turbo8", "dump"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dump.json"
            self.assertIsNone(gate.resumable(path, command, {"digest"}))
            path.write_text(json.dumps(dict(passed=True, pinsDigest="digest", command=command)))
            self.assertIsNotNone(gate.resumable(path, command, {"digest"}))
            for changed in (7, 6, 8):   # seed, frames, recipe
                other = list(command)
                other[changed] = "x"
                self.assertIsNone(gate.resumable(path, other, {"digest"}), other)
            self.assertIsNone(gate.resumable(path, command, {"other-digest"}))
            path.write_text(json.dumps(dict(passed=False, pinsDigest="digest", command=command)))
            self.assertIsNone(gate.resumable(path, command, {"digest"}))

    def test_a_dump_must_match_the_requested_workload(self):
        fields = dict(recipe="turbo8", seed="1043", frames="124")
        gate.check_dump(fields, "t2va", "960x544", "turbo8", 1043, 120)
        for recipe, seed, frames in (("turbo4", 1043, 120), ("turbo8", 7, 120), ("turbo8", 1043, 125)):
            with self.assertRaises(RuntimeError):
                gate.check_dump(fields, "t2va", "960x544", recipe, seed, frames)

    def test_snapping_matches_h3_geometry(self):
        self.assertEqual([gate.snapped_frames(n) for n in (120, 121, 124, 125, 141, 243)], [124, 124, 124, 141, 141, 243])


if __name__ == "__main__":
    unittest.main()

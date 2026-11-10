#!/usr/bin/env python3
"""Hermetic tests for scripts/h3-memory-gate (no GPU, no swift test: the runner is faked).

    python3 -I scripts/test_h3_memory_gate.py
"""
import importlib.machinery
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
_loader = importlib.machinery.SourceFileLoader("h3_memory_gate", str(HERE / "h3-memory-gate"))
spec = importlib.util.spec_from_loader("h3_memory_gate", _loader)
gate = importlib.util.module_from_spec(spec)
_loader.exec_module(gate)
FIXTURE = HERE.parent / "Tests/AetherCoreTests/Fixtures/h3/minimax-h3.json"
GIB = 1 << 30


class Matrix(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(FIXTURE.read_text())

    def test_the_full_matrix_follows_each_profiles_recipes_and_keyframes(self):
        renders = gate.plan_matrix(self.manifest)
        names = [r["name"] for r in renders]
        # Standard: 4 profiles × 2 recipes × 3 tasks; square: 1 × 1 × 3; large: 2 × 1 × 3.
        self.assertEqual(len(renders), 24 + 3 + 6)
        self.assertEqual(names[:3], ["960x544x124-turbo8-t2va", "960x544x124-turbo8-fl2va1", "960x544x124-turbo8-fl2va2"])
        self.assertNotIn("1280x704x124-turbo4-t2va", names)
        self.assertIn("960x544x243-turbo4-fl2va2", names)

    def test_filters_and_text_only_profiles(self):
        renders = gate.plan_matrix(self.manifest, profiles=["704x704x124"], tasks=["fl2va1", "fl2va2"])
        self.assertEqual([r["name"] for r in renders], ["704x704x124-turbo8-fl2va1", "704x704x124-turbo8-fl2va2"])
        del self.manifest["h3"]["profiles"][4]["keyframeEncoder"]
        self.assertEqual([r["task"] for r in gate.plan_matrix(self.manifest, profiles=["704x704x124"])], ["t2va"])
        with self.assertRaises(SystemExit):
            gate.plan_matrix(self.manifest, profiles=["640x480x124"])

    def test_each_render_holds_the_gpu_lease_in_its_own_process(self):
        render = gate.plan_matrix(self.manifest)[0]
        command = gate.render_command(Path("/x"), Path("/r/a.json"), render, ".build-ci")
        self.assertEqual(command[:2], [str(gate.REPO / "scripts/gpu-lease"), "--"])
        self.assertIn("AETHER_GATES=1", command)
        self.assertIn("AETHER_H3_GATE_RENDER=960x544x124:turbo8:t2va", command)
        self.assertEqual(command[-2:], ["--filter", "MiniMaxH3ProfileGateTests"])
        self.assertIn("--no-parallel", command)
        prune = gate.prune_command()
        self.assertEqual(prune[:2], [str(gate.REPO / "scripts/gpu-lease"), "--"])
        self.assertEqual(prune[-4:], ["prune", "--bucket", "swiftpm-testing-helper", "--yes"])


class Run(unittest.TestCase):
    def run_gate(self, records, renders, free, outcome, **options):
        ran = []

        def runner(command, out):
            ran.append(command)
            if "prune" in command:
                return 0
            record = Path(next(a for a in command if a.startswith("AETHER_H3_GATE_RECORD=")).split("=", 1)[1])
            ok = outcome(record.stem)
            record.write_text(json.dumps({"passed": ok}))
            return 0 if ok else 1

        status = gate.run_matrix(renders, Path("/export"), records, scratch=".build-ci", minimum_free_gib=100,
                                 runner=runner, free_bytes=free, log=lambda *_: None, **options)
        return status, ran

    def test_runs_in_order_records_and_prunes_at_the_end(self):
        renders = gate.plan_matrix(json.loads(FIXTURE.read_text()), profiles=["960x544x124"], recipes=["turbo8"])
        with tempfile.TemporaryDirectory() as t:
            records = Path(t)
            status, ran = self.run_gate(records, renders, lambda: 500 * GIB, lambda name: not name.endswith("fl2va2"))
            self.assertEqual(status, 1)
            self.assertEqual(len(ran), 4)
            self.assertIn("prune", ran[-1])
            summary = json.loads((records / "matrix.json").read_text())
            self.assertEqual([r["passed"] for r in summary["renders"]], [True, True, False])
            # A rerun skips what passed and runs the failure again.
            status, ran = self.run_gate(records, renders, lambda: 500 * GIB, lambda name: True)
            self.assertEqual(status, 0)
            self.assertEqual(len(ran), 2)

    def test_stops_below_the_free_disk_floor_and_still_prunes(self):
        renders = gate.plan_matrix(json.loads(FIXTURE.read_text()), profiles=["960x544x124"], recipes=["turbo8"])
        frees = iter([500 * GIB, 99 * GIB, 500 * GIB])
        with tempfile.TemporaryDirectory() as t:
            status, ran = self.run_gate(Path(t), renders, lambda: next(frees), lambda name: True)
            self.assertEqual(status, 2)
            self.assertEqual(len(ran), 2)
            self.assertIn("prune", ran[-1])
            self.assertIn("below 100", json.loads((Path(t) / "matrix.json").read_text())["stopped"])

    def test_keep_cache_skips_the_prune(self):
        renders = gate.plan_matrix(json.loads(FIXTURE.read_text()), profiles=["704x704x124"], tasks=["t2va"])
        with tempfile.TemporaryDirectory() as t:
            _, ran = self.run_gate(Path(t), renders, lambda: 500 * GIB, lambda name: True, keep_cache=True)
            self.assertEqual(len(ran), 1)


if __name__ == "__main__":
    unittest.main(verbosity=1)

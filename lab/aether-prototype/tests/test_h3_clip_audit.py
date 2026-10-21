"""CPU-only acceptance checks for the shared gate/render wall window."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("clip_audit", Path(__file__).parents[1] / "scripts/verify-h3-clip.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class QualifiedWindowAudit(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="aether-window-audit-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "run-01"
        self.path = self.run / "coordinator.json"
        self.build = self.root / "render/manifest.json"
        self.children, self.identities = {}, {}
        self.record = dict(passed=True, elapsed_seconds=730.0, wall_budget_seconds=1200, stages=[])
        for key, filename in [("coordinator_sha256", "run-h3-qualified-render.py"),
                              ("controller_sha256", "run-h3-check.py")]:
            (self.root / filename).write_text("CPU audit fixture; never executed")
            self.record[key] = audit.sha(self.root / filename)
        for kind, seconds, budget in [("phase", 20.0, 180), ("render", 700.0, 1177)]:
            self.children[kind] = dict(passed=True, child_joined=True, exit_status=0,
                elapsed_seconds=seconds, wall_budget_seconds=budget,
                binary_sha256=hashlib.sha256((kind + "-binary").encode()).hexdigest(),
                input_sha256=hashlib.sha256((kind + "-input").encode()).hexdigest())
            self.identities[kind] = dict(binary_sha256=self.children[kind]["binary_sha256"],
                                         bundle_sha256=self.children[kind]["input_sha256"])
        self.publish()

    def publish(self):
        self.record["stages"] = []
        for kind in ["phase", "render"]:
            child = self.children[kind]
            p = self.run / kind / "controller.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(child))
            build = self.root / kind / "manifest.json"
            build.parent.mkdir(parents=True, exist_ok=True)
            build.write_text(json.dumps(self.identities[kind]))
            self.record["stages"].append(dict(kind=kind, controller=str(p), controller_sha256=audit.sha(p),
                passed=child["passed"], elapsed_seconds=child["elapsed_seconds"],
                binary_sha256=child["binary_sha256"], input_sha256=child["input_sha256"]))
        self.path.write_text(json.dumps(self.record))

    def check(self):
        return audit.check_qualified_window(self.path, self.run / "render", self.build)

    def test_complete_window_uses_total_time(self):
        result = self.check()
        self.assertTrue(result["passed"])
        self.assertEqual(result["wall_seconds"], 730.0)

    def test_green_render_cannot_hide_overbudget_total(self):
        self.record["elapsed_seconds"] = 1200.1
        self.publish()
        with self.assertRaisesRegex(ValueError, "shared 20-minute"):
            self.check()

    def test_failed_gate_rejects_successful_render(self):
        self.children["phase"]["passed"] = False
        self.publish()
        with self.assertRaisesRegex(ValueError, "failed or was not joined"):
            self.check()

    def test_unjoined_child_is_rejected(self):
        self.children["render"]["child_joined"] = False
        self.publish()
        with self.assertRaisesRegex(ValueError, "failed or was not joined"):
            self.check()

    def test_changed_phase_identity_is_rejected(self):
        self.identities["phase"]["bundle_sha256"] = "0" * 64
        self.publish()
        with self.assertRaisesRegex(ValueError, "frozen build"):
            self.check()

    def test_changed_controller_bytes_are_rejected(self):
        p = self.run / "phase/controller.json"
        p.write_text(p.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "controller hash"):
            self.check()

    def test_impossible_stage_total_is_rejected(self):
        self.record["elapsed_seconds"] = 719.0
        self.publish()
        with self.assertRaisesRegex(ValueError, "Stage wall times"):
            self.check()

    def test_controller_still_cleaning_up_is_rejected(self):
        self.record["controller_cleanup_pending"] = True
        self.publish()
        with self.assertRaisesRegex(ValueError, "did not finish"):
            self.check()


if __name__ == "__main__":
    unittest.main()

"""Hermetic tests for the H3 profile table, the manifest builder, extend_h3.py and record_envelopes_h3.py.

    python3 -I Forge/h3/test_profiles_h3.py

Standard library only (no torch, no checkpoint, no export): scripts/ci.sh runs it.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import extend_h3  # noqa: E402
import profiles_h3 as ph  # noqa: E402
import record_envelopes_h3 as rec  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[2] / "Tests/AetherCoreTests/Fixtures/h3/minimax-h3.json"


class ProfileTable(unittest.TestCase):
    def test_every_profile_and_its_memory_class(self):
        table = {p.key: p for p in ph.PROFILES}
        self.assertEqual(list(table), ["960x544x124", "544x960x124", "960x544x243", "544x960x243", "704x704x124",
                                       "1280x704x124", "704x1280x124"])
        self.assertEqual(table["960x544x243"].video_rows, 72 * 510)
        self.assertEqual(table["960x544x243"].audio_latents, 405)
        self.assertEqual(table["704x704x124"].video_rows, 37 * 484)
        self.assertEqual(table["1280x704x124"].video_rows, 37 * 880)
        for key in ("960x544x124", "544x960x124", "704x704x124"):
            self.assertEqual(table[key].memory, {"barBytes": 18 << 30, "minimumPhysicalMemoryBytes": 0}, key)
        for key in ("960x544x243", "544x960x243", "1280x704x124", "704x1280x124"):
            self.assertEqual(table[key].memory, {"barBytes": 24 << 30, "minimumPhysicalMemoryBytes": 64 << 30}, key)
        self.assertEqual(table["1280x704x124"].recipes, ("turbo8",))
        self.assertEqual(table["960x544x243"].recipes, ("turbo8", "turbo4"))
        self.assertEqual(ph.AUDIO_LATENTS, [207, 405])

    def test_first_and_last_fit_the_row_limit_on_every_profile(self):
        # Text rows: the prompt plus, per keyframe, a label and the canvas's merged vision tokens (H/32 · W/32).
        for p in ph.PROFILES:
            rows = 2 * (p.keyframe_rows + 8) + 64 + 2 * p.keyframe_rows + 2 * p.audio_latents + p.video_rows
            self.assertLessEqual(rows, ph.ROWS[1], p.key)


class Manifest(unittest.TestCase):
    def test_the_sample_manifest_is_the_swift_fixture(self):
        self.assertEqual(ph.sample_manifest(), json.loads(FIXTURE.read_text()),
                         "regenerate: python3 -I Forge/h3/profiles_h3.py --sample-manifest > " + str(FIXTURE))

    def test_profiles_carry_their_shapes_recipes_and_memory(self):
        document = ph.sample_manifest()
        profiles = {p["key"]: p for p in document["h3"]["profiles"]}
        self.assertEqual(len(profiles), 7)
        p = profiles["960x544x243"]
        self.assertEqual(p["audioDecoder"], "audioDecoder405")
        self.assertEqual(p["keyframeEncoder"], "videoEncoderTail960x544")
        self.assertEqual(p["shapes"]["body"], {"table": [[6, 3, ph.HIDDEN]]})
        self.assertIn("audioDecoder405", p["shapes"])
        for role in p["shapes"]:
            self.assertIn(role, document["graphs"], role)

    def test_a_profile_without_its_graphs_is_left_out_or_text_only(self):
        meta = ph.sample_meta()
        del meta["audio-decoder"]["fixed"]["405"]
        meta["video-encoder"]["tails"] = {k: {} for k in ("544x960", "960x544")}
        document = ph.manifest(meta, {})
        profiles = {p["key"]: p for p in document["h3"]["profiles"]}
        self.assertEqual(sorted(profiles), sorted(["960x544x124", "544x960x124", "704x704x124", "1280x704x124", "704x1280x124"]))
        self.assertNotIn("keyframeEncoder", profiles["704x704x124"])
        self.assertNotIn("videoEncoderTail704x704", profiles["704x704x124"]["shapes"])
        self.assertIn("keyframeEncoder", profiles["960x544x124"])
        self.assertNotIn("audioDecoder405", document["graphs"])


def fake_export(root: Path) -> Path:
    """A tiny export: the sample meta restricted to the first two profiles' graphs, and a few bundle files."""
    meta = ph.sample_meta()
    del meta["audio-decoder"]["fixed"]["405"]
    meta["video-encoder"]["tails"] = {k: {} for k in ("544x960", "960x544")}
    (root / "work").mkdir(parents=True)
    (root / "work/build.json").write_text(json.dumps(meta))
    (root / "work/converted/vae").mkdir(parents=True)
    (root / "work/converted/vae/config.json").write_text("{}")
    bundle = root / "bundle"
    files = {}
    for rel, text in {"weights/body/block-00.bin": "weights", "models/body.aimodel/main.mlirb": "graph", "LICENSE": "license"}.items():
        (bundle / rel).parent.mkdir(parents=True, exist_ok=True)
        (bundle / rel).write_text(text)
        files[rel] = {"bytes": len(text), "sha256": extend_h3.sha256_file(bundle / rel)}
    restricted = [p for p in ph.PROFILES if p.key in ("960x544x124", "544x960x124")]
    document = ph.manifest(meta, files, profiles=restricted)
    (bundle / ".aether").mkdir()
    (bundle / ".aether/aether.json").write_text(json.dumps(document, indent=2))
    (bundle / "SHA256SUMS").write_text("x  LICENSE\n")
    (bundle / "export-report.json").write_text("{}")
    return root


class Extend(unittest.TestCase):
    def test_manifest_only_links_everything_and_never_writes_the_source(self):
        with tempfile.TemporaryDirectory() as t:
            source, output = fake_export(Path(t) / "source"), Path(t) / "extended"
            before = (source / "bundle/.aether/aether.json").read_bytes()
            extend_h3.main(["--source", str(source), "--output", str(output), "--manifest-only", "--minimum-free-gb", "0"])
            self.assertEqual((source / "bundle/.aether/aether.json").read_bytes(), before)
            for rel in ("weights/body/block-00.bin", "models/body.aimodel/main.mlirb", "LICENSE"):
                self.assertTrue(os.path.samefile(source / "bundle" / rel, output / "bundle" / rel), rel)
            self.assertTrue(os.path.samefile(source / "work/converted/vae/config.json", output / "work/converted/vae/config.json"))
            for rel in extend_h3.REWRITTEN:
                self.assertEqual((output / "bundle" / rel).stat().st_nlink, 1, rel)
            self.assertEqual((output / "work/build.json").stat().st_nlink, 1)
            document = json.loads((output / "bundle/.aether/aether.json").read_text())
            keys = [p["key"] for p in document["h3"]["profiles"]]
            self.assertEqual(keys, ["960x544x124", "544x960x124", "704x704x124", "1280x704x124", "704x1280x124"])
            self.assertEqual({f["path"] for f in document["assets"]["main"]["files"]},
                             {"weights/body/block-00.bin", "models/body.aimodel/main.mlirb", "LICENSE"})
            sums = (output / "bundle/SHA256SUMS").read_text()
            self.assertIn("  .aether/aether.json", sums)
            self.assertIn(extend_h3.sha256_file(output / "bundle/.aether/aether.json"), sums)
            # A rerun keeps the links and gives the same manifest.
            extend_h3.main(["--source", str(source), "--output", str(output), "--manifest-only", "--minimum-free-gb", "0"])
            self.assertEqual(json.loads((output / "bundle/.aether/aether.json").read_text()), document)

    def test_refuses_an_output_inside_the_source_and_a_low_disk(self):
        with tempfile.TemporaryDirectory() as t:
            source = fake_export(Path(t) / "source")
            with self.assertRaises(SystemExit):
                extend_h3.main(["--source", str(source), "--output", str(source / "inner"), "--manifest-only"])
            with self.assertRaises(SystemExit):
                extend_h3.main(["--source", str(source), "--output", str(Path(t) / "x"), "--manifest-only",
                                "--minimum-free-gb", "1e9"])


def render(profile, recipe, task, peak=10 << 30, held=8 << 30, passed=True, export="e" * 64, architecture="h17c"):
    key = profile + rec.TASKS[task]
    return {"schema": rec.SCHEMA, "render": {"profile": profile, "recipe": recipe, "task": task}, "admissionKey": key,
            "export": export, "architecture": architecture, "marginBytes": 256 << 20, "passed": passed,
            "phases": {ph_: {"peakBytes": peak, "heldBytes": held} for ph_ in rec.PHASES}}


class Record(unittest.TestCase):
    def test_envelopes_take_the_largest_phase_over_each_keys_renders(self):
        manifest = ph.sample_manifest()
        records = [render("960x544x124", "turbo8", "t2va", peak=9 << 30), render("960x544x124", "turbo4", "t2va", peak=10 << 30),
                   render("960x544x124", "turbo8", "fl2va1"), render("960x544x124", "turbo8", "fl2va2", passed=False)]
        for i, r in enumerate(records):
            r["_file"] = f"{i}.json"
        found, skipped = rec.envelopes(manifest, records)
        self.assertEqual([e["key"] for e in found], ["960x544x124", "960x544x124/fl2va1"])
        self.assertEqual(found[0]["recipes"], ["turbo4", "turbo8"])
        self.assertEqual(found[0]["perStep"]["peakBytes"], 10 << 30)
        self.assertEqual(found[0]["export"], "e" * 64)
        self.assertTrue(skipped and skipped[0].startswith("960x544x124/fl2va2"))

    def test_refuses_mixed_exports_and_a_peak_over_the_bar(self):
        manifest = ph.sample_manifest()
        mixed = [render("960x544x124", "turbo8", "t2va"), render("960x544x124", "turbo8", "fl2va1", export="f" * 64)]
        for i, r in enumerate(mixed):
            r["_file"] = f"{i}.json"
        with self.assertRaises(SystemExit):
            rec.envelopes(manifest, mixed)
        over = [render("960x544x124", "turbo8", "t2va", peak=18 << 30)]
        over[0]["_file"] = "0.json"
        with self.assertRaises(SystemExit):
            rec.envelopes(manifest, over)

    def test_writes_a_fresh_manifest_and_its_checksum_and_refuses_a_linked_one(self):
        with tempfile.TemporaryDirectory() as t:
            bundle, renders = Path(t) / "bundle", Path(t) / "renders"
            (bundle / ".aether").mkdir(parents=True)
            renders.mkdir()
            path = bundle / ".aether/aether.json"
            path.write_text(json.dumps(ph.sample_manifest()))
            (bundle / "SHA256SUMS").write_text("0  .aether/aether.json\n1  LICENSE\n")
            (renders / "a.json").write_text(json.dumps(render("704x704x124", "turbo8", "t2va")))
            rec.main(["--bundle", str(bundle), "--renders", str(renders)])
            document = json.loads(path.read_text())
            self.assertEqual([e["key"] for e in document["h3"]["envelopes"]], ["704x704x124"])
            self.assertIn(rec.sha256_file(path) + "  .aether/aether.json", (bundle / "SHA256SUMS").read_text())
            os.link(path, Path(t) / "other.json")
            with self.assertRaises(SystemExit):
                rec.main(["--bundle", str(bundle), "--renders", str(renders)])


if __name__ == "__main__":
    unittest.main(verbosity=1)

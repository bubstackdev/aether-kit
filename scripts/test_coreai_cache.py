#!/usr/bin/env python3
"""Tests for scripts/coreai-cache. Run: python3 -I scripts/test_coreai_cache.py (scripts/ci.sh runs it).

Every case builds a synthetic cache tree in a temporary directory; nothing here reads or writes the real cache. Bucket
names and hashes are made up."""
import fcntl
import importlib.machinery
import importlib.util
import io
import os
import sys
import tempfile
import time
import unittest

sys.dont_write_bytecode = True  # the script has no .py suffix: its bytecode would land beside it as a stray file
_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coreai-cache")
_loader = importlib.machinery.SourceFileLoader("coreai_cache", _path)
spec = importlib.util.spec_from_loader("coreai_cache", _loader)
cache = importlib.util.module_from_spec(spec)
_loader.exec_module(cache)

BUILD, OTHER = "26B123", "25A354"
MODEL_A, MODEL_B = "a" * 64, "b" * 64
GPU, CPU = "1" * 64, "C" * 64
DAY = 86400


class Tree:
    """A synthetic Core AI cache root under a temporary directory."""

    def __init__(self, base):
        self.base = base
        self.root = os.path.join(base, "Caches", "coreai-cache")
        for name in ("sync", "tmp"):
            os.makedirs(os.path.join(self.root, name))

    def entry(self, bucket, model=MODEL_A, options=GPU, build=BUILD, size=4096, age_days=0.0, complete=True):
        path = os.path.join(self.root, build, bucket, model, options)
        package = os.path.join(path, "model.aimodelx")
        os.makedirs(package if complete else path)
        if complete:
            with open(os.path.join(package, "main-this.odix"), "wb") as handle:
                handle.write(b"\0" * 64)
            with open(os.path.join(package, "resources.bin"), "wb") as handle:
                handle.write(b"\1" * size)
        stamp = time.time() - age_days * DAY
        if complete:
            os.utime(os.path.join(package, "main-this.odix"), (stamp, stamp))
        os.utime(path, (stamp, stamp))
        return path

    def lock(self, build=BUILD, name="F" * 64):
        directory = os.path.join(self.root, "sync", build)
        os.makedirs(directory, exist_ok=True)
        open(os.path.join(directory, name + ".lock"), "wb").close()

    def run(self, *argv, os_build=BUILD):
        out = io.StringIO()
        status = cache.main(["--root", self.root, "--os-build", os_build] + list(argv), out=out)
        return status, out.getvalue()


def export(directory, name, digest_hex):
    path = os.path.join(directory, name)
    os.makedirs(path)
    with open(os.path.join(path, "main.hash"), "wb") as handle:
        handle.write(bytes.fromhex(digest_hex))
    open(os.path.join(path, "main.mlirb"), "wb").close()
    return path


class CacheTestCase(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.tree = Tree(os.path.realpath(self._temporary.name))
        self.warnings, self._warn = [], cache.warn
        cache.warn = self.warnings.append

    def tearDown(self):
        cache.warn = self._warn
        self._temporary.cleanup()


class RootTests(CacheTestCase):
    def test_a_directory_not_named_coreai_cache_is_refused(self):
        other = os.path.join(self.tree.base, "elsewhere")
        os.makedirs(other)
        self.assertEqual(cache.main(["--root", other, "list"], out=io.StringIO()), cache.EX_NOT_A_CACHE)

    def test_a_root_with_unexpected_children_is_refused(self):
        os.makedirs(os.path.join(self.tree.root, "Documents"))
        status, _ = self.tree.run("prune", "--bucket", "tool", "--yes")
        self.assertEqual(status, cache.EX_NOT_A_CACHE)
        self.assertTrue(os.path.isdir(os.path.join(self.tree.root, "Documents")))

    def test_a_symlinked_root_is_refused(self):
        link = os.path.join(self.tree.base, "link", "coreai-cache")
        os.makedirs(os.path.dirname(link))
        os.symlink(self.tree.root, link)
        self.assertEqual(cache.main(["--root", link, "list"], out=io.StringIO()), cache.EX_NOT_A_CACHE)

    def test_an_empty_cache_lists(self):
        status, out = self.tree.run("list")
        self.assertEqual(status, 0)
        self.assertIn("(no entries)", out)


class ListTests(CacheTestCase):
    def test_list_reports_each_bucket_with_its_entries(self):
        self.tree.entry("tool-one", MODEL_A, GPU)
        self.tree.entry("tool-one", MODEL_B, CPU)
        self.tree.entry("tool-two", MODEL_A, GPU, build=OTHER)
        self.tree.lock()
        status, out = self.tree.run("list", "--entries")
        self.assertEqual(status, 0)
        self.assertRegex(out, r"tool-one\s+\S+ \S+\s+2 entries")
        self.assertRegex(out, r"tool-two\s+\S+ \S+\s+1 entry")
        self.assertIn("%s  (another OS build)" % OTHER, out)
        self.assertIn("%s/%s" % (MODEL_B, CPU), out)
        self.assertIn("sync: %s (1 locks)" % BUILD, out)

    def test_list_marks_an_entry_a_process_holds(self):
        path = self.tree.entry("tool")
        descriptor = os.open(os.path.join(path, "model.aimodelx"), os.O_RDONLY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH)
            self.assertIn("in use", self.tree.run("list", "--entries")[1])
        finally:
            os.close(descriptor)
        self.assertNotIn("in use", self.tree.run("list", "--entries")[1])

    def test_list_deletes_nothing(self):
        path = self.tree.entry("tool")
        self.tree.run("list", "--entries")
        self.assertTrue(os.path.isdir(path))


class PruneSelectorTests(CacheTestCase):
    def test_prune_needs_a_selector(self):
        path = self.tree.entry("tool")
        self.assertEqual(self.tree.run("prune", "--yes")[0], cache.EX_USAGE)
        self.assertTrue(os.path.isdir(path))

    def test_a_dry_run_deletes_nothing(self):
        path = self.tree.entry("tool")
        status, out = self.tree.run("prune", "--bucket", "tool")
        self.assertEqual(status, 0)
        self.assertIn("would delete", out)
        self.assertTrue(os.path.isdir(path))

    def test_bucket_deletes_only_that_bucket(self):
        gone = self.tree.entry("test-runner", MODEL_A, GPU)
        gone_too = self.tree.entry("test-runner", MODEL_B, CPU)
        kept = self.tree.entry("app", MODEL_A, GPU)
        status, out = self.tree.run("prune", "--bucket", "test-runner", "--yes")
        self.assertEqual(status, 0, out)
        self.assertFalse(os.path.exists(gone))
        self.assertFalse(os.path.exists(gone_too))
        self.assertTrue(os.path.isdir(kept))
        self.assertIn("freed", out)
        # Under the running build the emptied directories stay (a process may be writing into them).
        self.assertTrue(os.path.isdir(os.path.join(self.tree.root, BUILD, "test-runner")))

    def test_bucket_is_repeatable(self):
        one = self.tree.entry("one")
        two = self.tree.entry("two")
        three = self.tree.entry("three")
        self.tree.run("prune", "--bucket", "one", "--bucket", "two", "--yes")
        self.assertEqual([os.path.exists(p) for p in (one, two, three)], [False, False, True])

    def test_keep_wins_over_every_selector(self):
        kept = self.tree.entry("app", build=OTHER, age_days=90)
        gone = self.tree.entry("tool", build=OTHER, age_days=90)
        status, _ = self.tree.run("prune", "--other-os-builds", "--older-than", "30", "--bucket", "app",
                                  "--bucket", "tool", "--keep", "app", "--yes")
        self.assertEqual(status, 0)
        self.assertTrue(os.path.isdir(kept))
        self.assertFalse(os.path.exists(gone))

    def test_older_than_reads_the_last_use(self):
        old = self.tree.entry("tool", MODEL_A, GPU, age_days=40)
        recent = self.tree.entry("tool", MODEL_B, GPU, age_days=2)
        # Compiled long ago but loaded yesterday: Core AI updates the .odix access time on a hit.
        used = self.tree.entry("tool", MODEL_A, CPU, age_days=40)
        odix = os.path.join(used, "model.aimodelx", "main-this.odix")
        os.utime(odix, (time.time() - DAY, os.lstat(odix).st_mtime))
        self.tree.run("prune", "--older-than", "30", "--yes")
        self.assertEqual([os.path.exists(p) for p in (old, recent, used)], [False, True, True])

    def test_other_os_builds_spares_the_running_build(self):
        current = self.tree.entry("tool", build=BUILD)
        previous = self.tree.entry("tool", build=OTHER)
        self.tree.lock(BUILD)
        self.tree.lock(OTHER)
        status, out = self.tree.run("prune", "--other-os-builds", "--yes")
        self.assertEqual(status, 0, out)
        self.assertTrue(os.path.isdir(current))
        self.assertFalse(os.path.exists(previous))
        # The emptied directories of the other build go, and its locks.
        self.assertFalse(os.path.exists(os.path.join(self.tree.root, OTHER)))
        self.assertFalse(os.path.exists(os.path.join(self.tree.root, "sync", OTHER)))
        self.assertTrue(os.path.isdir(os.path.join(self.tree.root, "sync", BUILD)))

    def test_selectors_combine(self):
        match = self.tree.entry("tool", MODEL_A, build=OTHER, age_days=40)
        wrong_bucket = self.tree.entry("app", MODEL_A, build=OTHER, age_days=40)
        too_young = self.tree.entry("tool", MODEL_B, build=OTHER, age_days=1)
        running = self.tree.entry("tool", MODEL_A, build=BUILD, age_days=40)
        self.tree.run("prune", "--bucket", "tool", "--older-than", "30", "--other-os-builds", "--yes")
        self.assertEqual([os.path.exists(p) for p in (match, wrong_bucket, too_young, running)],
                         [False, True, True, True])


class StaleTests(CacheTestCase):
    def setUp(self):
        super().setUp()
        self.sources = os.path.join(self.tree.base, "models")
        os.makedirs(self.sources)

    def test_stale_deletes_entries_no_source_export_has(self):
        export(self.sources, "kept.aimodel", MODEL_A)
        referenced = self.tree.entry("tool", MODEL_A, GPU)
        stale = self.tree.entry("tool", MODEL_B, GPU)
        status, out = self.tree.run("prune", "--stale", "--source", self.sources, "--yes")
        self.assertEqual(status, 0, out)
        self.assertTrue(os.path.isdir(referenced))
        self.assertFalse(os.path.exists(stale))

    def test_stale_follows_symlinked_exports_and_compiled_models(self):
        elsewhere = os.path.join(self.tree.base, "elsewhere")
        os.makedirs(elsewhere)
        os.symlink(export(elsewhere, "linked.aimodel", MODEL_A), os.path.join(self.sources, "linked.aimodel"))
        export(self.sources, "device.aimodelc", MODEL_B)
        os.symlink(self.sources, os.path.join(self.sources, "loop"))
        self.assertEqual(cache.source_hashes([self.sources]), {MODEL_A, MODEL_B})

    def test_stale_needs_a_source(self):
        self.assertEqual(self.tree.run("prune", "--stale")[0], cache.EX_USAGE)
        self.assertEqual(self.tree.run("prune", "--bucket", "x", "--source", self.sources)[0], cache.EX_USAGE)


class SafetyTests(CacheTestCase):
    def test_an_entry_in_use_is_skipped(self):
        path = self.tree.entry("tool")
        descriptor = os.open(os.path.join(path, "model.aimodelx"), os.O_RDONLY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH)
            status, out = self.tree.run("prune", "--bucket", "tool", "--yes")
        finally:
            os.close(descriptor)
        self.assertEqual(status, 0)
        self.assertIn("(in use)", out)
        self.assertTrue(os.path.isdir(path))

    def test_symlinks_inside_the_root_are_never_followed(self):
        outside = os.path.join(self.tree.base, "precious")
        os.makedirs(os.path.join(outside, GPU, "model.aimodelx"))
        with open(os.path.join(outside, "keep.txt"), "w") as handle:
            handle.write("keep")
        bucket = os.path.join(self.tree.root, BUILD, "tool")
        os.makedirs(bucket)
        os.symlink(outside, os.path.join(bucket, MODEL_A))                  # a model-hash level symlink
        os.symlink(outside, os.path.join(self.tree.root, BUILD, "linked"))  # a bucket-level symlink
        status, out = self.tree.run("prune", "--bucket", "tool", "--bucket", "linked", "--yes")
        self.assertEqual(status, 0, out)
        self.assertTrue(os.path.isfile(os.path.join(outside, "keep.txt")))
        self.assertTrue(os.path.isdir(os.path.join(outside, GPU, "model.aimodelx")))
        self.assertTrue(os.path.islink(os.path.join(bucket, MODEL_A)))

    def test_a_symlinked_entry_package_is_not_deleted_through(self):
        outside = os.path.join(self.tree.base, "precious")
        os.makedirs(outside)
        with open(os.path.join(outside, "keep.txt"), "w") as handle:
            handle.write("keep")
        path = os.path.join(self.tree.root, BUILD, "tool", MODEL_A, GPU)
        os.makedirs(path)
        os.symlink(outside, os.path.join(path, "model.aimodelx"))
        self.tree.run("prune", "--bucket", "tool", "--yes")
        self.assertTrue(os.path.isfile(os.path.join(outside, "keep.txt")))

    def test_inside_refuses_paths_that_leave_the_root(self):
        os.makedirs(os.path.join(self.tree.root, BUILD))
        root = cache.validate_root(self.tree.root)
        self.assertTrue(cache.inside(root, os.path.join(root, BUILD)))
        self.assertFalse(cache.inside(root, os.path.join(root, "..", "elsewhere")))
        self.assertFalse(cache.inside(root, root + "-sibling/x"))

    def test_a_recent_incomplete_entry_is_skipped_and_an_old_one_deleted(self):
        recent = self.tree.entry("tool", MODEL_A, complete=False)
        old = self.tree.entry("tool", MODEL_B, complete=False, age_days=1)
        status, out = self.tree.run("prune", "--bucket", "tool", "--yes")
        self.assertEqual(status, 0, out)
        self.assertTrue(os.path.isdir(recent))
        self.assertFalse(os.path.exists(old))
        self.assertIn("may still be compiling", out)

    def test_a_leftover_of_an_interrupted_prune_is_finished(self):
        self.tree.entry("tool")
        aside = os.path.join(self.tree.root, BUILD, "tool", cache.ASIDE_PREFIX + "X")
        os.makedirs(os.path.join(aside, "model.aimodelx"))
        status, out = self.tree.run("prune", "--bucket", "nothing-matches", "--yes")
        self.assertEqual(status, 0, out)
        self.assertFalse(os.path.exists(aside))
        self.assertIn("interrupted prune", out)

    def test_a_kept_bucket_keeps_its_leftovers_too(self):
        aside = os.path.join(self.tree.root, BUILD, "app", cache.ASIDE_PREFIX + "X")
        os.makedirs(aside)
        self.tree.run("prune", "--bucket", "tool", "--keep", "app", "--yes")
        self.assertTrue(os.path.isdir(aside))

    def test_a_sync_build_with_more_than_locks_is_left(self):
        self.tree.lock(OTHER)
        with open(os.path.join(self.tree.root, "sync", OTHER, "notes.txt"), "w") as handle:
            handle.write("x")
        status, _ = self.tree.run("prune", "--other-os-builds", "--yes")
        self.assertEqual(status, cache.EX_FAILED)
        self.assertTrue(os.path.isfile(os.path.join(self.tree.root, "sync", OTHER, "notes.txt")))
        self.assertIn("not only locks", self.warnings[0])


if __name__ == "__main__":
    unittest.main()

"""Launcher boundary tests; synthetic files are never official evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import local_p1 as m


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.repo = self.base / "repo"
        self.repo.mkdir()
        self.source = self.base / "source"
        self.source.mkdir()
        assets = {}
        for name in ("model.safetensors", "config.json", "generation_config.json", "merges.txt",
                     "tokenizer.json", "tokenizer_config.json", "vocab.json"):
            raw = ("fixture:" + name).encode()
            (self.source / name).write_bytes(raw)
            assets[name] = hashlib.sha256(raw).hexdigest()
        self.lock = {"assets": assets, "weight_bytes": (self.source / "model.safetensors").stat().st_size}
        path = self.repo / m.LOCK
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(self.lock))
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        m.git(self.repo, "add", ".")
        m.git(self.repo, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
        self.head = m.git(self.repo, "rev-parse", "HEAD")
        self.root = self.base / "work"
        self.args = argparse.Namespace(repo=str(self.repo), source=str(self.source), work_dir=str(self.root))
        for name, value in (("HEAD", self.head), ("WEIGHT_SHA", assets["model.safetensors"])):
            p = patch.object(m, name, value)
            p.start()
            self.addCleanup(p.stop)

    def test_reject_wrong_source_before_mutations(self):
        (self.source / "config.json").write_text("wrong")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            m.prepare(self.args)
        self.assertFalse(self.root.exists())

    def test_reject_lfs_pointer_size(self):
        (self.source / "model.safetensors").write_text("version https://git-lfs.github.com/spec/v1\n")
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            m.verify_source(self.source, self.lock)

    def test_no_reuse_work_directory(self):
        self.root.mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            m.prepare(self.args)

    def test_symlink_asset_rejected(self):
        p = self.source / "config.json"
        real = self.source / "real.json"
        p.rename(real)
        try:
            p.symlink_to(real)
        except OSError:
            self.skipTest("OS disallows symlink fixture")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            m.verify_source(self.source, self.lock)

    def test_real_worktree_isolation_and_prepared_status(self):
        before = {p.name: m.sha(p) for p in self.source.iterdir()}
        # A user's dirty checkout must remain untouched; the new worktree is clean.
        (self.repo / m.LOCK).write_text("user uncommitted work")
        original_execute = m.execute
        calls = []
        def execute(args, cwd, log, offline=False):
            calls.append([str(a) for a in args])
            if args[0] == "git":
                original_execute(args, cwd, log, offline)
            else:
                log.write_text("MOCK runtime operation; no official evidence")
        with patch.object(m, "execute", side_effect=execute), patch.object(m, "runtime_check"), \
             patch.object(m.shutil, "disk_usage", return_value=argparse.Namespace(free=100 * 1024**3)):
            m.prepare(self.args)
        self.assertEqual(m.git(self.repo, "rev-parse", "HEAD"), self.head)
        self.assertEqual((self.repo / m.LOCK).read_text(), "user uncommitted work")
        self.assertEqual(m.git(self.root / "checkout", "rev-parse", "HEAD"), self.head)
        self.assertEqual(before, {p.name: m.sha(p) for p in self.source.iterdir()})
        self.assertEqual(json.loads((self.root / "prepared.json").read_text())["status"], "PREPARED_NOT_P1")
        self.assertFalse(any("qualify" in c or "smoke" in c for c in calls))

    def test_setup_failure_preserved_no_ready_record(self):
        with patch.object(m, "execute", side_effect=ValueError("fixture install failure")), \
             patch.object(m.shutil, "disk_usage", return_value=argparse.Namespace(free=100 * 1024**3)):
            with self.assertRaisesRegex(ValueError, "install failure"):
                m.prepare(self.args)
        self.assertTrue((self.root / "preparation_failure.json").exists())
        self.assertFalse((self.root / "prepared.json").exists())

    def test_command_failure_not_masked(self):
        with self.assertRaisesRegex(ValueError, "exited 7"):
            m.execute([m.sys.executable, "-c", "raise SystemExit(7)"], self.base, self.base / "fail.log")

    def test_run_failure_retains_unique_attempts_and_never_trains(self):
        self.root.mkdir()
        m.write(self.root / "prepared.json", {"status": "PREPARED_NOT_P1", "target_head": self.head})
        calls = []
        def fail(args, *a, **kw):
            calls.append([str(x) for x in args])
            raise ValueError("fixture native parity failure")
        with patch.object(m, "check_checkout"), patch.object(m, "runtime_check"), patch.object(m, "execute", side_effect=fail):
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, "parity failure"):
                    m.run_p1(self.args)
        attempts = list(self.root.glob("p1-*"))
        self.assertEqual(len(attempts), 2)
        self.assertTrue(all((p / "launcher_failure.json").exists() for p in attempts))
        self.assertTrue(all(not (p / "completed.json").exists() for p in attempts))
        self.assertTrue(all("qualify" in c and "smoke" not in c for c in calls))

    def test_changed_checkout_rejected(self):
        (self.repo / m.LOCK).write_text("changed")
        with self.assertRaisesRegex(ValueError, "Tracked checkout"):
            m.check_checkout(self.repo)

    def test_unsupported_host_stops_before_install(self):
        with patch.object(m.platform, "system", return_value="Darwin"):
            with self.assertRaisesRegex(ValueError, "Windows/Linux"):
                m.prepare(self.args)
        self.assertFalse(self.root.exists())

    def test_unvalidated_receipt_cannot_be_reported_as_pass(self):
        self.root.mkdir()
        m.write(self.root / "prepared.json", {"status": "PREPARED_NOT_P1", "target_head": self.head})
        with patch.object(m, "check_checkout"), patch.object(m, "execute"), \
             patch.object(m, "runtime_check", side_effect=[None, ValueError("invalid receipt")]):
            with self.assertRaisesRegex(ValueError, "invalid receipt"):
                m.run_p1(self.args)
        attempt = next(self.root.glob("p1-*"))
        self.assertFalse((attempt / "completed.json").exists())
        self.assertTrue((attempt / "launcher_failure.json").exists())


if __name__ == "__main__":
    unittest.main()

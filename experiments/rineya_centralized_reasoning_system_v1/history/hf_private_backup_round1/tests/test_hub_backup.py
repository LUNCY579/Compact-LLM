"""Local fixtures only. Never invoke a real HF subprocess or network upload."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.rineya_centralized_reasoning_system_v1 import hub_backup as h


class FakeHF:
    def __init__(self, private=True):
        self.private = private
        self.files = {}
        self.uploads = 0
        self.unknown = False
        self.info_calls = 0

    def info(self):
        self.info_calls += 1
        if not self.private:
            raise h.BackupError("private visibility not verified")
        return {"id": h.REPO_ID, "private": True, "sha": "a" * 40, "siblings": [{"rfilename": name} for name in self.files]}

    def upload(self, folder, content):
        self.uploads += 1
        for path in folder.rglob("*"):
            if path.is_file():
                self.files[path.relative_to(folder).as_posix()] = path.read_bytes()
        if self.unknown:
            raise h.BackupError("response unavailable")

    def download(self, remote, revision, directory, maximum_bytes):
        assert revision == "a" * 40
        assert len(self.files[remote]) <= maximum_bytes
        path = directory / remote
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.files[remote])
        return path


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "source"
        self.package = self.root / h.PACKAGE
        self.package.mkdir(parents=True)
        (self.package / "fixture.json").write_text(json.dumps({"status": "quarantined", "external_path": "C:/must/not/read/secrets.json"}), encoding="utf-8")
        (self.package / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
        (self.root / "AGENTS.md").write_text("preserve prior work\n", encoding="utf-8")
        self.head = lambda _: "b" * 40

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, **kwargs):
        return h.prepare(self.root, head_reader=self.head, **kwargs)

    def prepared(self):
        result = self.prepare()
        return h.local_root(self.root), result["content_sha256"]

    def test_inventory_preserves_paths_bytes_hash_and_no_training_promotion(self):
        inv = h.inventory(self.root, "primary")
        self.assertEqual(inv["included_files"], 3)
        row = next(r for r in inv["files"] if r["path"].endswith("fixture.json"))
        data = (self.package / "fixture.json").read_bytes()
        self.assertEqual((row["bytes"], row["sha256"]), (len(data), h.sha(data)))
        self.assertFalse(row["training_eligible"])
        self.assertEqual(inv["content_sha256"], h.digest(inv["files"]))

    def test_never_reads_database_tensor_cache_or_opaque_archive(self):
        names = ("host.sqlite3", "activation.npy", "source.tar.gz")
        for name in names:
            (self.package / name).write_bytes(b"do not open")
        cache = self.package / "__pycache__"
        cache.mkdir()
        (cache / "bad.pyc").write_bytes(b"do not open")
        original = h.read_stable
        reads = []
        def record(root, path, expected=None):
            reads.append(path.name)
            return original(root, path, expected)
        with patch.object(h, "read_stable", side_effect=record):
            inv = h.inventory(self.root, "primary")
        self.assertFalse(set(names) & set(reads))
        self.assertNotIn("bad.pyc", reads)
        self.assertTrue(any("ambiguity" in e["reason"] for e in inv["exclusions"]))

    def test_sensitive_name_and_actual_token_rejected(self):
        for name in (".env", "credentials.json", "id_rsa"):
            path = self.package / name
            path.write_text("private\n", encoding="utf-8")
            with self.assertRaises(h.BackupError):
                h.inventory(self.root, "primary")
            path.unlink()
        (self.package / "innocent.json").write_bytes(b'{"value":"' + b"hf_" + b"a" * 30 + b'"}')
        with self.assertRaises(h.BackupError):
            h.inventory(self.root, "primary")

    def test_credential_filename_refused_without_echoing_token(self):
        secret = "hf_" + "a" * 30
        (self.package / (secret + ".json")).write_text("{}", encoding="utf-8")
        with self.assertRaises(h.BackupError) as captured:
            h.inventory(self.root, "primary")
        self.assertNotIn(secret, str(captured.exception))

    def test_unknown_binary_suffix_and_disguised_binary_refused(self):
        for name, data in (("unknown.dat", b"data"), ("binary.json", b"\x00payload"), ("fake.png", b"not a png")):
            path = self.package / name
            path.write_bytes(data)
            with self.assertRaises(h.BackupError):
                h.inventory(self.root, "primary")
            path.unlink()

    def test_path_escape_and_source_link_refused(self):
        outside = Path(self.temp.name) / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        with self.assertRaises(h.BackupError):
            h.read_stable(self.root, outside)
        link = self.package / "escape.txt"
        try:
            link.symlink_to(outside)
        except OSError:
            # Windows may forbid creating symlinks; mocked lstat still tests refusal.
            fake = copy.copy(outside.lstat())
            with patch.object(Path, "lstat", return_value=fake), patch.object(h.stat, "S_ISLNK", return_value=True):
                with self.assertRaises(h.BackupError):
                    h.no_link(outside)
        else:
            with self.assertRaises(h.BackupError):
                h.inventory(self.root, "primary")

    def test_hardlink_refused(self):
        try:
            (self.package / "hardlink.txt").hardlink_to(self.package / "source.py")
        except OSError:
            self.skipTest("hardlinks unavailable")
        with self.assertRaises(h.BackupError):
            h.inventory(self.root, "primary")

    def test_safe_size_and_count_bounds(self):
        with patch.object(h, "MAX_FILE_BYTES", 2):
            with self.assertRaises(h.BackupError):
                h.inventory(self.root, "primary")
        with patch.object(h, "MAX_FILES", 1):
            with self.assertRaises(h.BackupError):
                h.inventory(self.root, "primary")

    def test_payload_digest_excludes_backup_state_receipts_and_head(self):
        first = self.prepare()
        local = h.local_root(self.root)
        (local / "receipt.json").write_text("different receipt", encoding="utf-8")
        (self.package / "HF_PRIVATE_BACKUP_RECEIPT_001.json").write_text("receipt only", encoding="utf-8")
        second = h.prepare(self.root, head_reader=lambda _: "c" * 40)
        self.assertEqual(first["content_sha256"], second["content_sha256"])
        self.assertEqual(second["status"], "LOCAL_SAME_CONTENT_NOOP")
        (self.package / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
        third = self.prepare()
        self.assertNotEqual(first["content_sha256"], third["content_sha256"])
        self.assertTrue(Path(first["prepared_path"]).exists())

    def test_stale_inventory_and_mutable_exception_refused(self):
        inv = h.inventory(self.root, "primary")
        row = next(r for r in inv["files"] if r["path"].endswith("source.py"))
        (self.package / "source.py").write_text("changed\n", encoding="utf-8")
        with self.assertRaises(h.BackupError):
            h.read_stable(self.root, self.package / "source.py", row)
        with self.assertRaises(h.BackupError):
            self.prepare(exception=self.root)
        with self.assertRaises(h.BackupError):
            self.prepare(exception=self.root, exception_sha256="0" * 64)

    def test_exception_has_separate_prefix_and_exact_freeze(self):
        frozen = h.inventory(self.root, "exception")["content_sha256"]
        result = self.prepare(exception=self.root, exception_sha256=frozen)
        manifest = h.verify_prepared(Path(result["prepared_path"]), result["content_sha256"])
        self.assertEqual({r["path"].split("/")[0] for r in manifest["files"]}, {"primary", "exception"})

    def test_added_source_after_inventory_refused(self):
        original = h.read_stable
        injected = False
        def mutate(root, path, expected=None):
            nonlocal injected
            data = original(root, path, expected)
            if expected is not None and not injected:
                injected = True
                (self.package / "new.json").write_text("{}", encoding="utf-8")
            return data
        with patch.object(h, "read_stable", side_effect=mutate):
            with self.assertRaises(h.BackupError):
                self.prepare()

    def test_prepared_tamper_and_extra_file_refused(self):
        local, content = self.prepared()
        frozen = local / "snapshots" / content
        (frozen / "surprise.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(h.BackupError):
            h.verify_prepared(frozen, content)
        (frozen / "surprise.json").unlink()
        with (frozen / "snapshots" / content / "archive.tar").open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaises(h.BackupError):
            h.verify_prepared(frozen, content)

    def test_remote_visibility_gate_no_write(self):
        local, content = self.prepared()
        client = FakeHF(private=False)
        with self.assertRaises(h.BackupError):
            h.upload_local(local, content, client)
        self.assertEqual(client.uploads, 0)

    def test_successful_upload_then_verified_remote_noop(self):
        local, content = self.prepared()
        client = FakeHF()
        self.assertEqual(h.upload_local(local, content, client)["status"], "VERIFIED")
        self.assertEqual(h.upload_local(local, content, client)["status"], "VERIFIED_REMOTE_SAME_CONTENT_NOOP")
        self.assertEqual(client.uploads, 1)
        self.assertEqual(h.read_state(local, content)["remote_commit"], "a" * 40)

    def test_unknown_response_requires_explicit_reconcile_not_repeat(self):
        local, content = self.prepared()
        client = FakeHF()
        client.unknown = True
        with self.assertRaises(h.BackupError):
            h.upload_local(local, content, client)
        self.assertEqual(h.read_state(local, content)["status"], "UNKNOWN")
        with self.assertRaises(h.BackupError):
            h.upload_local(local, content, client)
        self.assertEqual(client.uploads, 1)
        self.assertEqual(h.reconcile_local(local, content, client)["status"], "VERIFIED")
        self.assertEqual(h.upload_local(local, content, client)["status"], "VERIFIED_REMOTE_SAME_CONTENT_NOOP")
        self.assertEqual(client.uploads, 1)

    def test_unknown_absent_reconciled_before_separate_retry(self):
        local, content = self.prepared()
        client = FakeHF()
        h.atomic_json(h.state_path(local, content), {"content_sha256": content, "status": "UNKNOWN"})
        with self.assertRaises(h.BackupError):
            h.upload_local(local, content, client)
        self.assertEqual(h.reconcile_local(local, content, client)["status"], "RECONCILED_ABSENT")
        self.assertEqual(h.upload_local(local, content, client)["status"], "VERIFIED")
        self.assertEqual(client.uploads, 1)

    def test_remote_partial_or_changed_snapshot_never_overwritten(self):
        local, content = self.prepared()
        client = FakeHF()
        client.files[f"snapshots/{content}/manifest.json"] = b"incomplete"
        with self.assertRaises(h.BackupError):
            h.upload_local(local, content, client)
        self.assertEqual(client.uploads, 0)
        client.files.clear()
        h.upload_local(local, content, client)
        client.files[f"snapshots/{content}/archive.tar"] += b"changed"
        with self.assertRaises(h.BackupError):
            h.upload_local(local, content, client)
        self.assertEqual(client.uploads, 1)

    def test_hfcli_fixed_private_dataset_and_forbidden_flags_absent(self):
        calls = []
        client = h.HFCLI()
        client.run = lambda args, timeout=120: calls.append(args) or ""
        client.upload(Path("prepared"), "0" * 64)
        args = calls[0]
        self.assertEqual(args[:2], ["upload", h.REPO_ID])
        self.assertEqual(args[args.index("--type") + 1], "dataset")
        self.assertIn("--private", args)
        self.assertFalse({"--delete", "--every", "--token", "--no-private"} & set(args))
        client.run = lambda args, timeout=120: json.dumps({"id": h.REPO_ID, "private": False, "sha": "a" * 40})
        with self.assertRaises(h.BackupError):
            client.info()

    def test_download_inventory_bounds_before_transfer(self):
        client = h.HFCLI()
        calls = []
        def run(args, timeout=120):
            calls.append(args)
            return json.dumps([{"file": "snapshots/data/manifest.json", "size": "50G"}])
        client.run = run
        with self.assertRaises(h.BackupError):
            client.download("snapshots/data/manifest.json", "a" * 40, self.root, 8192)
        self.assertEqual(len(calls), 1)
        self.assertIn("--dry-run", calls[0])

    def test_remote_target_appearing_immediately_before_write_refused(self):
        local, content = self.prepared()
        client = FakeHF()
        original = client.info
        def changed():
            info = original()
            if client.info_calls == 2:
                info["siblings"] = [{"rfilename": f"snapshots/{content}/manifest.json"}]
            return info
        client.info = changed
        with self.assertRaises(h.BackupError):
            h.upload_local(local, content, client)
        self.assertEqual(client.uploads, 0)

    def test_source_operation_lock_blocks_parallel_write(self):
        local = h.local_root(self.root)
        with h.lock(local):
            with self.assertRaises(h.BackupError):
                with h.lock(local):
                    self.fail("lock allowed simultaneous write")
        self.assertFalse((local / "operation.lock").exists())


if __name__ == "__main__":
    unittest.main()

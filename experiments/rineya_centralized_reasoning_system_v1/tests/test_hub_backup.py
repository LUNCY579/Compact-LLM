"""Local fixtures only. Never invoke a real HF subprocess or network upload."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
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
        self.head = "a" * 40
        self.race = False
        self.commit_calls = []

    def info(self):
        self.info_calls += 1
        if not self.private:
            raise h.BackupError("private visibility not verified")
        return {"id": h.REPO_ID, "private": True, "sha": self.head, "siblings": [{"rfilename": name} for name in self.files]}

    def upload(self, folder, content, parent_commit):
        self.uploads += 1
        request = {"folder": str(folder), "content_sha256": content, "parent_commit": parent_commit, "repo_id": h.REPO_ID, "repo_type": "dataset", "visibility_required": "private"}
        h.sdk_commit(request, self, lambda **kw: SimpleNamespace(**kw))
        if self.unknown:
            raise h.BackupError("response unavailable")

    def repo_info(self, **kw):
        assert kw == {"repo_id": h.REPO_ID, "repo_type": "dataset", "revision": "main", "files_metadata": False}
        return SimpleNamespace(id=h.REPO_ID, private=self.private, sha=self.head, siblings=[SimpleNamespace(rfilename=name) for name in self.files])

    def create_commit(self, **kw):
        self.commit_calls.append(kw)
        assert kw["repo_id"] == h.REPO_ID and kw["repo_type"] == "dataset"
        assert kw["create_pr"] is False and kw["run_as_future"] is False
        assert len(kw["operations"]) == 3
        if self.race:
            self.head = "c" * 40
            archive = next(op.path_in_repo for op in kw["operations"] if op.path_in_repo.endswith("archive.tar"))
            self.files[archive] = b"concurrent bytes must remain"
        if kw["parent_commit"] != self.head:
            raise h.BackupError("expected-parent conflict")
        for operation in kw["operations"]:
            self.files[operation.path_in_repo] = operation.path_or_fileobj
        self.head = "b" * 40
        return SimpleNamespace(oid=self.head)

    def download(self, remote, revision, directory, maximum_bytes):
        assert revision == self.head
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
        self.primary_patch = patch.object(h, "PRIMARY", self.root)
        self.exception_patch = patch.object(h, "EXCEPTION", self.root)
        self.primary_patch.start()
        self.exception_patch.start()

    def tearDown(self):
        self.exception_patch.stop()
        self.primary_patch.stop()
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
        self.assertEqual(h.read_state(local, content)["remote_commit"], "b" * 40)

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
        local, content = self.prepared()
        frozen = local / "snapshots" / content
        client = h.HFCLI()
        response = SimpleNamespace(returncode=0, stdout=json.dumps({"status": "COMMITTED", "remote_commit": "b" * 40, "parent_commit": "a" * 40}))
        with patch.object(h.subprocess, "run", return_value=response) as called:
            client.upload(frozen, content, "a" * 40)
        args = called.call_args.args[0]
        self.assertEqual(args[:2], [str(h.HERMES_PYTHON), "-c"])
        self.assertEqual(args[2], h.SDK_BRIDGE)
        request = h.strict_json(called.call_args.kwargs["input"])
        self.assertEqual(request["repo_id"], h.REPO_ID)
        self.assertEqual(request["repo_type"], "dataset")
        self.assertEqual(request["visibility_required"], "private")
        self.assertEqual(request["parent_commit"], "a" * 40)
        self.assertNotIn("token", request)
        self.assertFalse({"--delete", "--every", "--token", "--no-private"} & set(args))
        client.run = lambda args, timeout=120: json.dumps({"id": h.REPO_ID, "private": False, "sha": "a" * 40})
        with self.assertRaises(h.BackupError):
            client.info()

    def test_complete_manifest_credential_bypass_refused_before_write(self):
        local, content = self.prepared()
        path = local / "snapshots" / content / "snapshots" / content / "manifest.json"
        manifest = h.strict_json(path.read_bytes())
        credential = "hf_" + "a" * 30
        manifest["unchecked_metadata"] = credential
        path.write_bytes(h.canonical(manifest))
        client = FakeHF()
        with self.assertRaises(h.BackupError) as captured:
            h.upload_local(local, content, client)
        self.assertNotIn(credential, str(captured.exception))
        self.assertEqual(client.uploads, 0)
        self.assertEqual(client.commit_calls, [])

    def test_manifest_schema_and_provenance_are_bound(self):
        local, content = self.prepared()
        frozen = local / "snapshots" / content
        path = frozen / "snapshots" / content / "manifest.json"
        original = path.read_bytes()
        for mutate in (lambda m: m.update(unchecked="extra"), lambda m: m["sources"][0].update(git_head_provenance_only="c" * 40), lambda m: m["sources"][0].update(root="C:/outside"), lambda m: m["sources"][0].update(included_files=True), lambda m: m["files"][0].update(qualification="approved")):
            manifest = h.strict_json(original)
            mutate(manifest)
            path.write_bytes(h.canonical(manifest))
            with self.assertRaises(h.BackupError):
                h.verify_prepared(frozen, content)
        path.write_bytes(original)
        self.assertEqual(h.verify_prepared(frozen, content)["content_sha256"], content)
        with self.assertRaises(h.BackupError):
            h.strict_json('{"field":1,"field":2}')

    def test_inside_write_concurrency_preserves_remote_and_requires_reconcile(self):
        local, content = self.prepared()
        client = FakeHF()
        client.race = True
        with self.assertRaises(h.BackupError):
            h.upload_local(local, content, client)
        archive = f"snapshots/{content}/archive.tar"
        self.assertEqual(client.files, {archive: b"concurrent bytes must remain"})
        self.assertEqual(h.read_state(local, content)["status"], "UNKNOWN")
        self.assertEqual(client.commit_calls[0]["parent_commit"], "a" * 40)
        with self.assertRaises(h.BackupError):
            h.upload_local(local, content, client)
        self.assertEqual(client.uploads, 1)
        with self.assertRaises(h.BackupError):
            h.reconcile_local(local, content, client)
        self.assertEqual(client.files[archive], b"concurrent bytes must remain")
        self.assertEqual(h.read_state(local, content)["status"], "UNKNOWN")

    def test_invalid_remote_inventory_never_proves_absence_or_writes(self):
        local, content = self.prepared()
        client = FakeHF()
        broken = [[{}, {"rfilename": None}], [{"rfilename": "same"}, {"rfilename": "same"}], [{"rfilename": "../escape"}], [{"rfilename": "a//b"}], [{"rfilename": "a", "path": "b"}], [{"rfilename": "A"}, {"rfilename": "a"}], [{"rfilename": "a"}, {"rfilename": "a/b"}], [{"rfilename": "bad\\path"}], ["not-an-entry"]]
        for siblings in broken:
            client.info = lambda siblings=siblings: {"id": h.REPO_ID, "private": True, "sha": "a" * 40, "siblings": siblings}
            with self.assertRaises(h.BackupError):
                h.upload_local(local, content, client)
            self.assertEqual(client.uploads, 0)
            self.assertEqual(h.read_state(local, content)["status"], "UNATTEMPTED")

    def test_sdk_source_mutation_after_validation_is_refused(self):
        local, content = self.prepared()
        frozen = local / "snapshots" / content
        request = {"folder": str(frozen), "content_sha256": content, "parent_commit": "a" * 40, "repo_id": h.REPO_ID, "repo_type": "dataset", "visibility_required": "private"}
        client = FakeHF()
        original = h.transport_bytes
        def mutate(root, path, maximum, expected_hash):
            if path.name == "README.md":
                path.write_text("changed after validation", encoding="utf-8")
            return original(root, path, maximum, expected_hash)
        with patch.object(h, "transport_bytes", side_effect=mutate):
            with self.assertRaises(h.BackupError):
                h.sdk_commit(request, client, lambda **kw: SimpleNamespace(**kw))
        self.assertEqual(client.commit_calls, [])

    def test_sdk_exact_three_byte_additions_and_atomic_parent(self):
        local, content = self.prepared()
        client = FakeHF()
        h.upload_local(local, content, client)
        call = client.commit_calls[0]
        self.assertEqual(set(call), {"repo_id", "repo_type", "revision", "operations", "commit_message", "parent_commit", "create_pr", "run_as_future"})
        self.assertEqual({op.path_in_repo for op in call["operations"]}, {"README.md", f"snapshots/{content}/manifest.json", f"snapshots/{content}/archive.tar"})
        self.assertTrue(all(type(op.path_or_fileobj) is bytes for op in call["operations"]))
        self.assertEqual(call["parent_commit"], "a" * 40)

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

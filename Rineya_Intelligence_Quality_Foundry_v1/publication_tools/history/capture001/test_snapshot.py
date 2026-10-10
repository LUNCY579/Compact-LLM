"""Synthetic, bounded fixtures only; never inspect the real foundry payloads.

Run from this directory: python -m unittest -v test_snapshot
"""

import gzip
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import snapshot as publication


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="publication-tests-")
        self.root = Path(self.temporary.name)
        self.source = self.root / "foundry"
        self.source.mkdir()
        self.archive = self.root / "archive"

    def tearDown(self):
        self.temporary.cleanup()

    def put(self, relative, data):
        path = self.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def rows(self, name):
        with gzip.open(self.archive / name, "rb") as stream:
            return [json.loads(line) for line in stream]

    def rewrite_catalog(self, name, rows):
        data = b"".join(publication._json_bytes(row) + b"\n" for row in rows)
        with (self.archive / name).open("wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
                stream.write(data)
        compressed = (self.archive / name).read_bytes()
        manifest = json.loads((self.archive / "manifest.json").read_bytes())
        manifest["catalogs"][name] = {
            "compressed_sha256": hashlib.sha256(compressed).hexdigest(), "compressed_size": len(compressed),
            "sha256": hashlib.sha256(data).hexdigest(), "size": len(data), "records": len(rows),
        }
        (self.archive / "manifest.json").write_bytes(publication._json_bytes(manifest) + b"\n")

    def simple_archive(self):
        self.put("README.md", b"byte exact\r\n")
        publication.snapshot(self.source, self.archive)

    def test_raw_corpus_pilot_exception_and_historical_method_sources(self):
        policy = publication.Policy()
        self.assertIsNone(policy.reason("candidates/pilot-v3/train.jsonl"))
        self.assertIsNone(policy.reason("releases/pilot-v2/train_sft.jsonl"))
        self.assertEqual(policy.reason("cycles/old/before/source__TRAIN.jsonl"),
                         "possible_raw_corpus_outside_owned_pilot")
        self.assertIsNone(policy.reason("cycles/old/train_audit_rows.jsonl"))
        self.assertIsNone(policy.reason("cycles/old/check_metadata_contract_v6.py"))
        self.assertIsNone(policy.reason("cycles/old/FINAL_DELIVERY_MANIFEST.json"))
        self.assertEqual(policy.reason("CHECK/check_metadata_contract_v6.py"),
                         "protected_subtree_or_filename")
        self.assertEqual(policy.reason("cycles/old/pytorch_model.bin"), "model_weight_filename")
        self.assertIsNone(policy.reason("cycles/old/CODE.bin"))

    def test_new_policy_omissions_verify(self):
        self.put("evidence/pytorch_model.bin", b"synthetic model filename")
        self.put("cycles/old/source_TRAIN.jsonl", b"synthetic raw corpus filename")
        report = publication.snapshot(self.source, self.archive)
        self.assertEqual(report["omissions_by_reason"], {
            "model_weight_filename": 1, "possible_raw_corpus_outside_owned_pilot": 1})
        publication.verify(self.archive)

    def test_protected_physical_root_and_extra_rejected_before_inspection(self):
        logical = "_supplement/accidental_checkout_v10/SOURCE_PIN_v10.json"
        with mock.patch.object(publication, "_check_ancestors", side_effect=AssertionError("stat attempted")):
            with self.assertRaises(publication.ArchiveError):
                publication.snapshot(self.root / "CHECK", self.archive)
            with self.assertRaises(publication.ArchiveError):
                publication.snapshot(self.source, self.archive,
                    extras=[(logical, self.root / "privatebank" / "SOURCE_PIN_v10.json")])

    def test_exact_program_file_never_opens_descendants(self):
        blocked = publication.PROGRAM_FILES[0] + "/body.jsonl"
        self.assertEqual(publication.Policy().reason(blocked), "unreferenced_program_subtree")
        self.put(blocked, b"must not read")
        original = publication._source_stat
        def guard(path):
            if path.name == "body.jsonl":
                raise AssertionError("blocked descendant statted")
            return original(path)
        with mock.patch.object(publication, "_source_stat", side_effect=guard):
            report = publication.snapshot(self.source, self.archive)
        self.assertEqual(report["counts"]["occurrences"], 0)
        publication.verify(self.archive)

    def test_partial_object_migration_rolls_back_without_orphans(self):
        self.put("evidence/payload.txt", random.Random(72).randbytes(96 * 1024))
        original = publication.os.rename
        calls = 0
        def fail_second(source, target):
            nonlocal calls
            calls += 1
            self.assertEqual(Path(source).anchor, Path(target).anchor)
            if calls == 2:
                raise OSError("synthetic second migration failure")
            return original(source, target)
        with mock.patch.object(publication.os, "rename", side_effect=fail_second):
            report = publication.snapshot(self.source, self.archive, part_max_bytes=1024)
        self.assertEqual(report["omissions_by_reason"], {"unreadable_or_changed_file": 1})
        self.assertEqual(report["counts"]["unique_objects"], 0)
        publication.verify(self.archive)

    def test_second_compression_failure_preserves_verifiable_omission(self):
        self.put("evidence/payload.txt", b"synthetic")
        original = publication._stream_source
        calls = 0
        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise publication.SourceChanged("synthetic second read failure")
            return original(*args, **kwargs)
        with mock.patch.object(publication, "_stream_source", side_effect=fail_second):
            report = publication.snapshot(self.source, self.archive)
        self.assertEqual(report["omissions_by_reason"], {"source_changed": 1})
        publication.verify(self.archive)

    def test_unsafe_name_is_inert_omission_evidence(self):
        class Entry:
            name = "a:b.txt"
            path = "never inspected"
            def stat(self, **kwargs):
                raise AssertionError("unsafe name statted")
        class Entries:
            def __init__(self):
                self.values = iter([Entry()])
            def __next__(self):
                return next(self.values)
            def close(self):
                pass
        original_scandir = publication.os.scandir
        def source_names_only(path):
            return Entries() if Path(path) == self.source else original_scandir(path)
        with mock.patch.object(publication.os, "scandir", side_effect=source_names_only):
            report = publication.snapshot(self.source, self.archive)
        self.assertEqual(report["omissions_by_reason"], {"unsafe_logical_path": 1})
        publication.verify(self.archive)

    def test_dedup_preserves_roles_and_exact_restore(self):
        data = b'{"content":"same bytes", "role":"assistant"}\r\n'
        expected = {
            "candidates/pilot/CHOSEN.jsonl": data,
            "packets/pilot/REJECTED.jsonl": data,
            "releases/pilot/VIEW.jsonl": data,
            "README.md": b"readable\r\n\x00unchanged\n",
            "docs/design.md": b"design\n",
        }
        for relative, content in expected.items():
            self.put(relative, content)
        report = publication.snapshot(self.source, self.archive)
        self.assertEqual(report["counts"]["occurrences"], 5)
        self.assertEqual(report["counts"]["unique_objects"], 3)
        rows = self.rows("paths.jsonl.gz")
        role_rows = [row for row in rows if row["path"] in list(expected)[:3]]
        self.assertEqual(len({row["object"] for row in role_rows}), 1)
        self.assertEqual({row["path"] for row in rows}, set(expected))
        result = publication.verify(self.archive)
        target = self.root / "restored"
        publication.restore(self.archive, target, result["paths_catalog_sha256"])
        for relative, content in expected.items():
            self.assertEqual((target / relative).read_bytes(), content)
            self.assertEqual((self.source / relative).read_bytes(), content)
        self.assertEqual((self.archive / "source/README.md").read_bytes(), expected["README.md"])
        self.assertFalse((self.archive / "source/candidates").exists())

    def test_parts_bounded_and_object_bytes_deterministic(self):
        payload = random.Random(723).randbytes(96 * 1024)
        self.put("evidence/payload.jsonl", payload)
        first = publication.snapshot(self.source, self.archive, part_max_bytes=1024)
        parts = [row for row in self.rows("objects.jsonl.gz") if row["type"] == "part"]
        self.assertGreater(len(parts), 10)
        self.assertTrue(all(0 < row["size"] <= 1024 for row in parts))
        second_archive = self.root / "second-archive"
        publication.snapshot(self.source, second_archive, part_max_bytes=1024)
        for row in parts:
            left = (self.archive / row["path"]).read_bytes()
            self.assertEqual(left, (second_archive / row["path"]).read_bytes())
            self.assertEqual(hashlib.sha256(left).hexdigest(), row["sha256"])
        self.assertEqual(publication.verify(self.archive)["counts"], first["counts"])
        target = self.root / "restored"
        publication.restore(self.archive, target)
        self.assertEqual((target / "evidence/payload.jsonl").read_bytes(), payload)

    def test_corrupt_compressed_part_is_rejected(self):
        self.simple_archive()
        row = next(row for row in self.rows("objects.jsonl.gz") if row["type"] == "part")
        part = self.archive / row["path"]
        value = bytearray(part.read_bytes())
        value[-5] ^= 0x80
        part.write_bytes(value)
        with self.assertRaises(publication.ArchiveError):
            publication.verify(self.archive)

    def test_missing_and_unknown_objects_are_rejected(self):
        self.simple_archive()
        row = next(row for row in self.rows("objects.jsonl.gz") if row["type"] == "part")
        part = self.archive / row["path"]
        contents = part.read_bytes()
        part.unlink()
        with self.assertRaises(publication.ArchiveError):
            publication.verify(self.archive)
        part.write_bytes(contents)
        unknown = self.archive / "objects/unknown"
        unknown.mkdir()
        with self.assertRaises(publication.ArchiveError):
            publication.verify(self.archive)

    def test_missing_catalog_and_compressed_catalog_corruption(self):
        self.simple_archive()
        path = self.archive / "paths.jsonl.gz"
        data = path.read_bytes()
        path.unlink()
        with self.assertRaises(publication.ArchiveError):
            publication.verify(self.archive)
        path.write_bytes(data + b"unexpected")
        with self.assertRaises(publication.ArchiveError):
            publication.verify(self.archive)

    def test_traversal_absolute_drive_and_alias_paths_are_rejected_before_restore(self):
        self.simple_archive()
        original = self.rows("paths.jsonl.gz")
        for path in ("../escape", "/absolute", "C:/absolute", "folder\\escape", "a/../b", "a//b", "NUL.txt", "trailing."):
            with self.subTest(path=path):
                row = dict(original[0], path=path)
                self.rewrite_catalog("paths.jsonl.gz", [row])
                destination = self.root / "never-created"
                with self.assertRaises(publication.ArchiveError):
                    publication.restore(self.archive, destination)
                self.assertFalse(destination.exists())

    def test_duplicate_catalog_paths_and_case_aliases_rejected(self):
        self.simple_archive()
        row = self.rows("paths.jsonl.gz")[0]
        for duplicate in (dict(row), dict(row, path=row["path"].swapcase())):
            self.rewrite_catalog("paths.jsonl.gz", [row, duplicate])
            with self.assertRaises(publication.ArchiveError):
                publication.verify(self.archive)

    def test_directory_case_aliases_and_ancestor_collisions_rejected(self):
        self.simple_archive()
        row = self.rows("paths.jsonl.gz")[0]
        for pair in (("a/file", "a/file/child"), ("a/file/child", "a/file"), ("a/one", "A/two")):
            self.rewrite_catalog("paths.jsonl.gz", [dict(row, path=name, materialized=False) for name in pair])
            with self.assertRaises(publication.ArchiveError):
                publication.verify(self.archive)

    def test_changed_source_recorded_and_not_claimed_frozen(self):
        path = self.put("evidence/live.jsonl", b"original bytes\n")
        original_reader = publication._stream_source
        reads = 0

        def changing_reader(*args, **kwargs):
            nonlocal reads
            result = original_reader(*args, **kwargs)
            reads += 1
            if reads == 1:
                path.write_bytes(b"changed bytes and size\n")
            return result

        with mock.patch.object(publication, "_stream_source", side_effect=changing_reader):
            report = publication.snapshot(self.source, self.archive)
        self.assertEqual(report["counts"]["occurrences"], 0)
        self.assertEqual(report["counts"]["unique_objects"], 0)
        self.assertEqual(report["omissions_by_reason"], {"source_changed": 1})
        self.assertFalse(report["atomic_source_freeze_claim"])
        self.assertEqual(report["capture_issues"], 1)
        self.assertEqual(publication.verify(self.archive)["counts"]["occurrences"], 0)

    def test_matching_stats_still_require_matching_second_hash(self):
        path = self.put("evidence/live.jsonl", b"original bytes\n")
        original_reader = publication._stream_source
        reads = 0

        def mismatching_reader(*args, **kwargs):
            nonlocal reads
            result = original_reader(*args, **kwargs)
            reads += 1
            return ("0" * 64, result[1]) if reads == 2 else result

        with mock.patch.object(publication, "_stream_source", side_effect=mismatching_reader):
            report = publication.snapshot(self.source, self.archive)
        self.assertEqual(reads, 2)
        self.assertEqual(report["omissions_by_reason"], {"source_changed": 1})
        self.assertEqual(path.read_bytes(), b"original bytes\n")
        publication.verify(self.archive)

    def test_protected_paths_are_gated_before_any_entry_stat(self):
        names = ["CHECK_payload.jsonl", "FINAL", "eval", "spent", "privatebank", "private", "sealed_payloads", "program/other"]

        class NeverStatEntry:
            def __init__(self, name):
                self.name = name
                self.path = str(self.source) if hasattr(self, "source") else name

            def stat(self, **kwargs):
                raise AssertionError("protected entry was statted")

        class Entries:
            def __init__(self):
                self.entries = iter(NeverStatEntry(name) for name in names)

            def __next__(self):
                return next(self.entries)

            def close(self):
                pass

        omitted = []
        with mock.patch.object(publication.os, "scandir", return_value=Entries()):
            files = list(publication._walk_source(self.source, publication.Policy(), lambda path, reason: omitted.append((path, reason))))
        self.assertEqual(files, [])
        self.assertEqual(len(omitted), len(names))

    def test_safe_manifests_check_methods_and_exact_program_allowlist(self):
        allowed = ["SEAL.json", "FINAL_MANIFEST.json", "fixtures/CHECK_metadata_manifest.json", "registry_tools/CHECK_method.py",
                   publication.PROGRAM_PREFIX + "/design.md", *publication.PROGRAM_FILES]
        for name in allowed:
            self.put(name, b"safe method or metadata\n")
        for name in ("CHECK/raw.jsonl", "FINAL_MANIFEST.json/raw.jsonl"):
            # The second basename is already a file, so exercise the policy directly.
            self.assertEqual(publication.Policy().reason(name), "protected_subtree_or_filename")
        blocked = "program/20261010_resource_planning_quality_v1/raw_sibling.jsonl"
        self.put(blocked, b"excluded\n")
        self.put("privatebank/SEAL.json", b"must not inspect\n")
        report = publication.snapshot(self.source, self.archive)
        self.assertEqual({row["path"] for row in self.rows("paths.jsonl.gz")}, set(allowed))
        self.assertIn("unreferenced_program_subtree", report["omissions_by_reason"])
        self.assertFalse((self.archive / "source" / blocked).exists())
        self.assertTrue((self.archive / "source" / publication.PROGRAM_FILES[0]).is_file())
        publication.verify(self.archive)

    def test_four_supplements_are_distinct_logical_occurrences(self):
        extras = []
        for name in sorted(publication.SUPPLEMENT_NAMES):
            path = self.root / name
            path.write_bytes(b"metadata identical bytes\n")
            extras.append(("_supplement/accidental_checkout_v10/" + name, path))
        report = publication.snapshot(self.source, self.archive, extras=extras)
        self.assertEqual(report["counts"]["supplement_files"], 4)
        self.assertEqual(report["counts"]["unique_objects"], 1)
        self.assertEqual({row["path"] for row in self.rows("paths.jsonl.gz")}, {name for name, _ in extras})
        target = self.root / "restored"
        publication.restore(self.archive, target)
        for logical, physical in extras:
            self.assertEqual((target / logical).read_bytes(), physical.read_bytes())

    def test_secrets_omitted_without_redacting_and_chunk_boundary_detected(self):
        token = b"ghp_" + b"aB3dE5gH7jK9mN1pQ2sT4vW6yZ8cD0fG2hJ4kL6"
        content = b"x" * (publication.CHUNK - 15) + b" " + token + b"\n"
        path = self.put("evidence/token.txt", content)
        self.put("evidence/key.txt", b"-----BEGIN PRIVATE KEY-----\nsynthetic fixture\n")
        self.put("evidence/aws.txt", b"AKIA0123456789ABCDEF\n")
        self.put("evidence/openai.txt", b"sk-proj-aB3dE5gH7jK9mN1pQ2sT4vW6yZ8cD0fG2hJ4kL6\n")
        self.put("docs/example.md", b"sk-" + b"x" * 64 + b" is a placeholder\n")
        report = publication.snapshot(self.source, self.archive)
        self.assertEqual(report["omissions_by_reason"], {"secret_signature": 4})
        self.assertEqual(report["counts"]["occurrences"], 1)
        self.assertEqual(path.read_bytes(), content)
        self.assertNotIn(token.decode(), (self.archive / "report.json").read_text())
        publication.verify(self.archive)

    def test_mirror_catalog_binding_and_external_pin(self):
        self.simple_archive()
        result = publication.verify(self.archive)
        with self.assertRaises(publication.ArchiveError):
            publication.verify(self.archive, "0" * 64)
        self.assertTrue(publication.verify(self.archive, result["paths_catalog_sha256"])["verified"])
        mirror = self.archive / "source/README.md"
        contents = mirror.read_bytes()
        mirror.write_bytes(b"changed")
        with self.assertRaises(publication.ArchiveError):
            publication.verify(self.archive)
        mirror.unlink()
        with self.assertRaises(publication.ArchiveError):
            publication.verify(self.archive)
        mirror.write_bytes(contents)
        (self.archive / "source/extra.txt").write_bytes(b"unknown")
        with self.assertRaises(publication.ArchiveError):
            publication.verify(self.archive)

    def test_archive_and_restore_are_exclusive_and_source_is_never_output(self):
        self.simple_archive()
        with self.assertRaises(FileExistsError):
            publication.snapshot(self.source, self.archive)
        with self.assertRaises(publication.ArchiveError):
            publication.snapshot(self.source, self.source / "archive")
        self.assertFalse((self.source / "archive").exists())
        destination = self.root / "occupied"
        destination.mkdir()
        sentinel = destination / "README.md"
        sentinel.write_bytes(b"do not overwrite")
        with self.assertRaises(publication.ArchiveError):
            publication.restore(self.archive, destination)
        self.assertEqual(sentinel.read_bytes(), b"do not overwrite")
        with self.assertRaises(publication.ArchiveError):
            publication.restore(self.archive, self.archive / "inside")

    def test_symlink_source_omitted_and_archive_symlink_rejected(self):
        self.simple_archive()
        source_link = self.source / "linked.txt"
        try:
            source_link.symlink_to(self.source / "README.md")
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable to this Windows account")
        second = self.root / "second"
        report = publication.snapshot(self.source, second)
        self.assertEqual(report["omissions_by_reason"], {"symlink_or_reparse_point": 1})
        mirror = self.archive / "source/README.md"
        mirror.unlink()
        mirror.symlink_to(self.source / "README.md")
        with self.assertRaises(publication.ArchiveError):
            publication.verify(self.archive)

    def test_catalog_concatenated_gzip_members_and_bounded_raw_reader(self):
        self.put("README.md", b"small\n")
        self.put("docs/design.md", b"different\n")
        publication.snapshot(self.source, self.archive)
        rows = self.rows("paths.jsonl.gz")
        data = b"".join(publication._json_bytes(row) + b"\n" for row in rows)
        compressed = b"".join(gzip.compress(publication._json_bytes(row) + b"\n", mtime=0) for row in rows)
        path = self.archive / "paths.jsonl.gz"
        path.write_bytes(compressed)
        manifest = json.loads((self.archive / "manifest.json").read_bytes())
        manifest["catalogs"]["paths.jsonl.gz"].update({"compressed_sha256": hashlib.sha256(compressed).hexdigest(), "compressed_size": len(compressed), "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)})
        (self.archive / "manifest.json").write_bytes(publication._json_bytes(manifest) + b"\n")
        publication.verify(self.archive)
        physical = self.root / "large-part"
        physical.write_bytes(b"a" * (publication.CHUNK + 10))
        reader = publication.MultiPartReader(self.root, [("large-part", hashlib.sha256(physical.read_bytes()).hexdigest(), publication.CHUNK + 10)])
        with reader:
            self.assertEqual(len(reader.read()), publication.CHUNK)
            self.assertEqual(reader.read(), b"a" * 10)
            self.assertEqual(reader.read(), b"")


if __name__ == "__main__":
    unittest.main()

"""Synthetic-only regression test SOURCE for the reviewed backup implementation.

These tests have not been executed as part of drafting this package. Every payload
is invented here and written under TemporaryDirectory; no production inventory or
production payload is opened. Run only after reviewing backup.py and this file.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import backup


INVENTORY_SHA256 = "1" * 64
TRAIN_ROLE = "train_candidate_or_train_audit"
TRAIN_PATH = "synthetic_package/data/train.jsonl"
# Deliberately invented high-entropy signature fixture; never a real credential.
SYNTHETIC_TOKEN = "hf_0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def synthetic_row(path: str, data: bytes, role: str = TRAIN_ROLE) -> dict:
    return {
        "path": path,
        "bytes": len(data),
        "backup_role": role,
        "training_eligible": False,
    }


def synthetic_selection(rows: list[dict]) -> dict:
    return backup.prepare_selection(
        {"status": "METADATA_ONLY_NOT_PREPARED", "files": rows},
        INVENTORY_SHA256,
    )


class SyntheticBackupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="rineya-synthetic-test-")
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.source = self.work / "synthetic-source"
        self.source.mkdir()

    def write_fixture(self, relative: str, data: bytes) -> Path:
        path = self.source.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def capture_fixture(self, rows: list[dict], name: str = "output") -> tuple[dict, Path]:
        output = self.work / name
        outcome = backup.capture(
            synthetic_selection(rows), output,
            source_root=self.source, test_fixture=True,
        )
        return outcome, output

    def assert_failed_without_success_artifacts(self, outcome: dict, output: Path) -> None:
        self.assertEqual(outcome["status"], "FAILED")
        self.assertFalse((output / "manifest.json").exists())
        self.assertFalse((output / "backup.tar.gz").exists())
        self.assertTrue((output / "START.json").is_file())
        self.assertTrue((output / "OUTCOME.json").is_file())
        self.assertTrue((output / "attempts.jsonl").is_file())
        self.assertTrue((output / "archive.partial.tar.gz").is_file())
        recorded = json.loads((output / "OUTCOME.json").read_text(encoding="utf-8"))
        self.assertEqual(recorded, outcome)
        self.assertFalse(recorded["upload_attempted"])
        self.assertEqual(recorded["intended_exit_code"], 2)
        self.assertEqual(recorded["observed_exit_code"], "UNKNOWN")

    def read_success(self, outcome: dict, output: Path) -> tuple[dict, dict[str, bytes]]:
        self.assertEqual(outcome["status"], "SUCCESS", outcome)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        verified = backup.verify_archive(output / "backup.tar.gz", manifest)
        self.assertTrue(verified["verified"])
        with tarfile.open(output / "backup.tar.gz", "r:gz") as archive:
            data = {}
            for member in archive.getmembers():
                self.assertTrue(member.isfile())
                stream = archive.extractfile(member)
                self.assertIsNotNone(stream)
                with stream:
                    data[member.name] = stream.read()
        return manifest, data

    def test_path_gate_is_pure_and_rejects_forbidden_paths_before_stat(self) -> None:
        forbidden = [
            "../synthetic/train.jsonl",
            "/synthetic/train.jsonl",
            "C:/synthetic/train.jsonl",
            "synthetic\\data\\train.jsonl",
            "synthetic/.env",
            "synthetic/evaluation/train.jsonl",
            "synthetic/evaluator_outputs/train.jsonl",
            "synthetic/holdout/train.jsonl",
            "synthetic/sealed_payload/train.jsonl",
            "synthetic/private_bank/train.jsonl",
        ]
        with mock.patch.object(Path, "stat", side_effect=AssertionError("stat is forbidden")), \
                mock.patch("os.stat", side_effect=AssertionError("stat is forbidden")), \
                mock.patch("builtins.open", side_effect=AssertionError("open is forbidden")):
            for path in forbidden:
                with self.subTest(path=path):
                    self.assertIsNotNone(backup.path_gate(path, TRAIN_ROLE, 1))
            self.assertIsNotNone(backup.path_gate(TRAIN_PATH, "undeclared_test_role", 1))
            self.assertIsNotNone(backup.path_gate(TRAIN_PATH, TRAIN_ROLE, -1))
            self.assertIsNotNone(backup.path_gate(TRAIN_PATH, TRAIN_ROLE, True))
            self.assertIsNone(backup.path_gate(TRAIN_PATH, TRAIN_ROLE, 1))

    def test_file_id_is_canonical_and_role_sensitive(self) -> None:
        expected = hashlib.sha256(json.dumps(
            [TRAIN_PATH, TRAIN_ROLE], ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        self.assertEqual(backup.file_id(TRAIN_PATH, TRAIN_ROLE), expected)
        self.assertNotEqual(backup.file_id(TRAIN_PATH, TRAIN_ROLE),
                            backup.file_id(TRAIN_PATH, "source_document_or_metadata"))

    def test_blocked_selected_path_is_rejected_before_source_stat_or_open(self) -> None:
        row = synthetic_selection([
            synthetic_row("synthetic_package/holdout/train.jsonl", b"{}\n")
        ])["rows"][0]
        with mock.patch.object(backup, "_check_ancestors", side_effect=AssertionError("lstat is forbidden")), \
                mock.patch.object(Path, "resolve", side_effect=AssertionError("resolve is forbidden")), \
                mock.patch.object(Path, "lstat", side_effect=AssertionError("lstat is forbidden")), \
                mock.patch("os.open", side_effect=AssertionError("open is forbidden")):
            with self.assertRaisesRegex(backup.BackupError, "protected_path"):
                backup.scan_file(self.source, row)

    def test_stale_prospective_gate_is_rejected_before_any_payload_scan(self) -> None:
        selection = synthetic_selection([synthetic_row(TRAIN_PATH, b"{}\n")])
        row = selection["rows"][0]
        row["path"] = "synthetic_package/holdout/train.jsonl"
        row["file_id"] = backup.file_id(row["path"], row["backup_role"])
        # Retain the old ALLOWED_PROSPECTIVELY gate to represent stale selection.
        output = self.work / "stale-gate-output"
        with mock.patch.object(backup, "scan_file", side_effect=AssertionError("must never read")) as scan:
            outcome = backup.capture(selection, output, source_root=self.source, test_fixture=True)
            scan.assert_not_called()
        self.assert_failed_without_success_artifacts(outcome, output)
        self.assertEqual(outcome["fatal"]["reason"], "prospective_gate_mismatch")
        self.assertEqual(set(outcome["counts"].values()), {"UNKNOWN"})

    def test_selection_preserves_all_102_declared_train_roles(self) -> None:
        roles = sorted(backup.TRAIN_ROLES)
        self.assertGreaterEqual(len(roles), 2)
        rows = [
            synthetic_row(f"synthetic_package/data/train_{index:03d}.jsonl", b"{}\n",
                          roles[0] if index < 46 else roles[1])
            for index in range(102)
        ]
        selection = synthetic_selection(rows)
        selected = selection["rows"]
        self.assertEqual(len(selected), 102)
        self.assertEqual([row["backup_role"] for row in selected],
                         [row["backup_role"] for row in rows])
        self.assertEqual(sum(row["backup_role"] == roles[0] for row in selected), 46)
        self.assertEqual(sum(row["backup_role"] == roles[1] for row in selected), 56)
        self.assertEqual({row["file_id"] for row in selected},
                         {backup.file_id(row["path"], row["backup_role"]) for row in rows})
        self.assertTrue(all(row["training_eligible"] is False for row in selected))
        self.assertTrue(all(row["selection_kind"] == "declared_train" for row in selected))
        self.assertEqual(selection["inventory_sha256"], INVENTORY_SHA256)
        self.assertEqual(selection["status"], "PROSPECTIVE_NOT_CAPTURED")

    def test_raw_newlines_crlf_and_non_ascii_bytes_are_unchanged(self) -> None:
        payloads = {
            TRAIN_PATH: b'{"text":"line one"}\r\n{"text":"line two"}\n\n',
            "synthetic_package/data/train_utf8.jsonl":
                '{"text":"\u96ea \u2603 caf\u00e9"}\r\n'.encode("utf-8"),
        }
        for path, data in payloads.items():
            self.write_fixture(path, data)
        outcome, output = self.capture_fixture(
            [synthetic_row(path, data) for path, data in payloads.items()])
        _, archived = self.read_success(outcome, output)
        self.assertEqual(archived, {"source/" + path: data for path, data in payloads.items()})
        self.assertEqual(outcome["counts"]["captured_files"], 2)
        self.assertEqual(outcome["counts"]["verified_files"], 2)

    def test_gzip_train_is_scanned_and_original_compressed_bytes_are_archived(self) -> None:
        path = "synthetic_package/data/train.jsonl.gz"
        cleartext = b'{"text":"synthetic harmless fixture"}\r\n\n'
        payload = gzip.compress(cleartext, mtime=1234567)
        self.write_fixture(path, payload)
        outcome, output = self.capture_fixture([synthetic_row(path, payload)])
        _, archived = self.read_success(outcome, output)
        self.assertEqual(archived["source/" + path], payload)
        self.assertEqual(gzip.decompress(archived["source/" + path]), cleartext)

    def test_secret_inside_gzip_blocks_every_success_artifact(self) -> None:
        path = "synthetic_package/data/train.jsonl.gz"
        # This is an invented credential-shaped string, never a real credential.
        synthetic_token = SYNTHETIC_TOKEN
        cleartext = ('{"text":"synthetic fixture","api_key":"' + synthetic_token + '"}\n').encode()
        payload = gzip.compress(cleartext, mtime=0)
        self.write_fixture(path, payload)
        outcome, output = self.capture_fixture([synthetic_row(path, payload)])
        self.assert_failed_without_success_artifacts(outcome, output)
        self.assertEqual({failure["path"] for failure in outcome["failures"]}, {path})
        self.assertTrue(any("secret" in failure["reason"].lower()
                            for failure in outcome["failures"]), outcome)
        for name in ("START.json", "OUTCOME.json", "attempts.jsonl"):
            self.assertNotIn(synthetic_token.encode("utf-8"), (output / name).read_bytes())

    def test_second_pass_gzip_scan_blocks_secret_despite_mocked_approved_descriptor(self) -> None:
        path = "synthetic_package/data/train.jsonl.gz"
        malicious_cleartext = ('{"text":"' + SYNTHETIC_TOKEN + '"}\n').encode("utf-8")
        harmless_cleartext = ('{"text":"' + "A" * len(SYNTHETIC_TOKEN) + '"}\n').encode("utf-8")
        split = len(b'{"text":"') + 20

        def two_members(cleartext: bytes) -> bytes:
            return (gzip.compress(cleartext[:split], compresslevel=0, mtime=0) +
                    gzip.compress(cleartext[split:], compresslevel=0, mtime=0))

        # Stored blocks guarantee equal compressed sizes. Splitting the token
        # across gzip members prevents raw bytes from containing its signature.
        harmless = two_members(harmless_cleartext)
        malicious = two_members(malicious_cleartext)
        self.assertEqual(len(harmless), len(malicious))
        self.assertNotIn(SYNTHETIC_TOKEN.encode("utf-8"), malicious)
        self.assertEqual(gzip.decompress(malicious), malicious_cleartext)
        fixture = self.write_fixture(path, harmless)
        original_scan = backup.scan_file
        changed = False

        def approve_then_substitute(root: Path, row: dict) -> dict:
            nonlocal changed
            descriptor = original_scan(root, row)
            fixture.write_bytes(malicious)
            changed = True
            # Deliberately approve even the substituted digest and stat metadata.
            # The actual archive read must independently scan decoded bytes.
            descriptor["sha256"] = hashlib.sha256(malicious).hexdigest()
            descriptor["stat_fingerprint"] = list(backup._fingerprint(fixture.stat()))
            return descriptor

        with mock.patch.object(backup, "scan_file", side_effect=approve_then_substitute):
            outcome, output = self.capture_fixture([synthetic_row(path, harmless)])
        self.assertTrue(changed)
        self.assert_failed_without_success_artifacts(outcome, output)
        self.assertTrue(any(failure["stage"] == "second_read" and
                            failure["reason"] == "secret_signature_huggingface_token"
                            for failure in outcome["failures"]), outcome)

    def test_malformed_truncated_and_bad_crc_gzip_have_explicit_failures(self) -> None:
        path = "synthetic_package/data/train.jsonl.gz"
        valid = gzip.compress(b'{"text":"synthetic gzip validity fixture"}\n', mtime=0)
        bad_crc = bytearray(valid)
        bad_crc[-8] ^= 1
        cases = {
            "malformed": (b"SYNTHETIC_NOT_GZIP", "invalid_gzip_crc_or_encoding"),
            "truncated": (valid[:-8], "gzip_truncated_or_missing_eof"),
            "bad-crc": (bytes(bad_crc), "invalid_gzip_crc_or_encoding"),
            "trailing-data": (valid + b"SYNTHETIC_NOT_A_MEMBER", "invalid_gzip_crc_or_encoding"),
        }
        for name, (payload, reason) in cases.items():
            with self.subTest(case=name):
                self.write_fixture(path, payload)
                outcome, output = self.capture_fixture([synthetic_row(path, payload)], name + "-output")
                self.assert_failed_without_success_artifacts(outcome, output)
                self.assertEqual(outcome["failures"], [{
                    "file_id": backup.file_id(path, TRAIN_ROLE),
                    "path": path,
                    "stage": "first_read",
                    "reason": reason,
                }])

    def test_gzip_expansion_bound_is_an_explicit_failure_without_truncation(self) -> None:
        path = "synthetic_package/data/train.jsonl.gz"
        payload = gzip.compress(b"synthetic expansion bytes\n" * 32, mtime=0)
        self.write_fixture(path, payload)
        with mock.patch.object(backup, "MAX_EXPANDED_BYTES", 32):
            outcome, output = self.capture_fixture([synthetic_row(path, payload)])
        self.assert_failed_without_success_artifacts(outcome, output)
        self.assertEqual(outcome["failures"][0]["stage"], "first_read")
        self.assertEqual(outcome["failures"][0]["reason"], "gzip_expansion_bound_exceeded")

    def test_concatenated_gzip_scans_secret_in_later_member(self) -> None:
        path = "synthetic_package/data/train.jsonl.gz"
        first = gzip.compress(b'{"text":"synthetic harmless first member"}\n', mtime=0)
        second = gzip.compress(('{"text":"' + SYNTHETIC_TOKEN + '"}\n').encode("utf-8"), mtime=0)
        payload = first + second
        self.write_fixture(path, payload)
        outcome, output = self.capture_fixture([synthetic_row(path, payload)])
        self.assert_failed_without_success_artifacts(outcome, output)
        self.assertEqual(outcome["failures"][0]["reason"], "secret_signature_huggingface_token")

    def test_gzipped_text_metadata_preserves_compressed_and_decoded_bytes(self) -> None:
        train = b"{}\n"
        path = "synthetic_package/README.txt.gz"
        cleartext = b"Synthetic metadata.\r\nLine endings stay exact.\n\n"
        payload = gzip.compress(cleartext, mtime=24681357)
        self.write_fixture(TRAIN_PATH, train)
        self.write_fixture(path, payload)
        rows = [synthetic_row(TRAIN_PATH, train), synthetic_row(path, payload, backup.METADATA_ROLE)]
        outcome, output = self.capture_fixture(rows)
        manifest, archived = self.read_success(outcome, output)
        self.assertEqual(archived["source/" + path], payload)
        self.assertEqual(gzip.decompress(archived["source/" + path]), cleartext)
        metadata = next(row for row in manifest["files"] if row["path"] == path)
        self.assertEqual(metadata["backup_role"], backup.METADATA_ROLE)
        self.assertFalse(metadata["training_eligible"])

    def test_gzipped_text_metadata_decoded_secret_blocks_capture_and_keeps_scanning(self) -> None:
        train = b"{}\n"
        path = "synthetic_package/README.txt.gz"
        payload = gzip.compress(("Synthetic metadata token: " + SYNTHETIC_TOKEN + "\r\n").encode("utf-8"), mtime=0)
        self.write_fixture(TRAIN_PATH, train)
        self.write_fixture(path, payload)
        rows = [synthetic_row(TRAIN_PATH, train), synthetic_row(path, payload, backup.METADATA_ROLE)]
        outcome, output = self.capture_fixture(rows)
        self.assert_failed_without_success_artifacts(outcome, output)
        self.assertEqual({failure["path"] for failure in outcome["failures"]}, {path})
        self.assertEqual(outcome["failures"][0]["reason"], "secret_signature_huggingface_token")
        attempts = [json.loads(line) for line in
                    (output / "attempts.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual({attempt["path"] for attempt in attempts}, {TRAIN_PATH, path})
        self.assertTrue(any(attempt["path"] == TRAIN_PATH and
                            attempt["status"] == "VERIFIED_FIRST_READ" for attempt in attempts))

    def test_same_size_mutation_after_scan_is_detected_before_success(self) -> None:
        before = b'{"text":"alpha"}\n'
        after = b'{"text":"omega"}\n'
        self.assertEqual(len(before), len(after))
        path = self.write_fixture(TRAIN_PATH, before)
        original_scan = backup.scan_file
        changed = False

        def scan_then_mutate(root: Path, row: dict) -> dict:
            nonlocal changed
            descriptor = original_scan(root, row)
            if not changed:
                path.write_bytes(after)
                changed = True
            return descriptor

        with mock.patch.object(backup, "scan_file", side_effect=scan_then_mutate):
            outcome, output = self.capture_fixture([synthetic_row(TRAIN_PATH, before)])
        self.assertTrue(changed)
        self.assert_failed_without_success_artifacts(outcome, output)
        self.assertTrue(outcome["failures"], outcome)

    def test_all_missing_selected_rows_have_explicit_first_pass_failures(self) -> None:
        rows = [synthetic_row(f"synthetic_package/data/train_missing_{index}.jsonl", b"{}\n")
                for index in range(3)]
        outcome, output = self.capture_fixture(rows)
        self.assert_failed_without_success_artifacts(outcome, output)
        self.assertEqual({failure["path"] for failure in outcome["failures"]},
                         {row["path"] for row in rows})
        self.assertEqual(outcome["counts"]["selected_files"], 3)
        self.assertEqual(outcome["counts"]["captured_files"], 0)
        self.assertEqual(outcome["counts"]["failed_attempts"], 3)
        attempts = [json.loads(line) for line in
                    (output / "attempts.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual({attempt["file_id"] for attempt in attempts},
                         {backup.file_id(row["path"], row["backup_role"]) for row in rows})
        self.assertTrue(all(attempt["stage"] == "first_read" and
                            attempt["status"] == "FAILED" for attempt in attempts))

    def test_blocked_and_missing_rows_do_not_hide_later_failures(self) -> None:
        rows = [
            synthetic_row("synthetic_package/holdout/train.jsonl", b"{}\n"),
            synthetic_row("synthetic_package/data/train_missing.jsonl", b"{}\n"),
        ]
        outcome, output = self.capture_fixture(rows)
        self.assert_failed_without_success_artifacts(outcome, output)
        self.assertEqual({failure["path"] for failure in outcome["failures"]},
                         {row["path"] for row in rows})
        self.assertEqual(outcome["counts"]["failed_attempts"], 2)

    def test_declared_size_change_is_a_failure(self) -> None:
        declared = b'{"text":"original"}\n'
        self.write_fixture(TRAIN_PATH, declared + b"\n")
        outcome, output = self.capture_fixture([synthetic_row(TRAIN_PATH, declared)])
        self.assert_failed_without_success_artifacts(outcome, output)
        self.assertEqual({failure["path"] for failure in outcome["failures"]}, {TRAIN_PATH})

    def test_archive_is_deterministic_for_identical_source_bytes(self) -> None:
        payload = b'{"text":"deterministic synthetic bytes"}\r\n'
        self.write_fixture(TRAIN_PATH, payload)
        rows = [synthetic_row(TRAIN_PATH, payload)]
        first, first_output = self.capture_fixture(rows, "first-output")
        second, second_output = self.capture_fixture(rows, "second-output")
        first_manifest, _ = self.read_success(first, first_output)
        second_manifest, _ = self.read_success(second, second_output)
        first_bytes = (first_output / "backup.tar.gz").read_bytes()
        second_bytes = (second_output / "backup.tar.gz").read_bytes()
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(hashlib.sha256(first_bytes).hexdigest(),
                         hashlib.sha256(second_bytes).hexdigest())
        self.assertEqual(first_manifest["content_sha256"], second_manifest["content_sha256"])
        self.assertEqual(first_manifest["files"][0]["sha256"],
                         hashlib.sha256(payload).hexdigest())

    def test_second_pass_hash_detects_mutation_despite_equal_stat_fingerprint(self) -> None:
        before = b'{"text":"alpha"}\n'
        after = b'{"text":"omega"}\n'
        path = self.write_fixture(TRAIN_PATH, before)
        original_scan = backup.scan_file

        def scan_then_mutate(root: Path, row: dict) -> dict:
            descriptor = original_scan(root, row)
            path.write_bytes(after)
            return descriptor

        # Deliberately defeat the stat-based shortcut, proving the independent
        # hash of bytes passed to tar.addfile still detects the changed content.
        with mock.patch.object(backup, "_fingerprint", return_value=(1, 2, len(before), 3, 4)), \
                mock.patch.object(backup, "scan_file", side_effect=scan_then_mutate):
            outcome, output = self.capture_fixture([synthetic_row(TRAIN_PATH, before)])
        self.assert_failed_without_success_artifacts(outcome, output)
        self.assertTrue(any(failure["stage"] == "second_read" and
                            failure["reason"] == "independent_hash_or_size_mismatch"
                            for failure in outcome["failures"]), outcome)

    def test_receipts_record_actual_pid_and_reproducible_configuration_pins(self) -> None:
        payload = b'{"text":"synthetic receipt fixture"}\n'
        self.write_fixture(TRAIN_PATH, payload)
        selection = synthetic_selection([synthetic_row(TRAIN_PATH, payload)])
        output = self.work / "receipt-output"
        outcome = backup.capture(selection, output, source_root=self.source, test_fixture=True)
        manifest, _ = self.read_success(outcome, output)
        start = json.loads((output / "START.json").read_text(encoding="utf-8"))
        recorded_outcome = json.loads((output / "OUTCOME.json").read_text(encoding="utf-8"))

        def canonical(value: object) -> bytes:
            return json.dumps(value, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"), allow_nan=False).encode("utf-8")

        expected_selection_pin = hashlib.sha256(canonical(selection)).hexdigest()
        expected_tool_pin = hashlib.sha256(Path(backup.__file__).read_bytes()).hexdigest()
        expected_source_root = str(self.source.resolve(strict=True))
        expected_config_pin = hashlib.sha256(canonical({
            "format": backup.FORMAT,
            "source_root": expected_source_root,
            "destination": backup.DESTINATION,
            "namespace": backup.NAMESPACE,
            "claims": backup.CLAIMS,
            "selection": selection,
            "max_expanded_bytes": backup.MAX_EXPANDED_BYTES,
            "max_expansion_ratio": backup.MAX_EXPANSION_RATIO,
        })).hexdigest()
        for receipt in (start, recorded_outcome):
            with self.subTest(event=receipt["event"]):
                self.assertEqual(receipt["pid"], os.getpid())
                self.assertEqual(receipt["source_root"], expected_source_root)
                self.assertEqual(receipt["selection_sha256"], expected_selection_pin)
                self.assertEqual(receipt["inventory_sha256"], INVENTORY_SHA256)
                self.assertEqual(receipt["tool_sha256"], expected_tool_pin)
                self.assertEqual(receipt["config_sha256"], expected_config_pin)
        self.assertEqual(start["source_mode"], "synthetic_fixture")
        self.assertEqual(start["counts"], {name: "UNKNOWN" for name in
                                          ("selected_files", "captured_files", "failed_attempts", "verified_files")})
        self.assertEqual(start["observed_exit_code"], "UNKNOWN")
        self.assertEqual(recorded_outcome, outcome)
        self.assertEqual(recorded_outcome["intended_exit_code"], 0)
        self.assertEqual(recorded_outcome["observed_exit_code"], "UNKNOWN")
        self.assertFalse(recorded_outcome["upload_attempted"])
        self.assertEqual(recorded_outcome["claims"], manifest["claims"])
        for claims in (selection["claims"], manifest["claims"], recorded_outcome["claims"]):
            self.assertFalse(claims["training_ready"])
            self.assertFalse(claims["training_authorized"])
            self.assertFalse(claims["training_eligible"])
            self.assertEqual(claims["rights_use"], "UNKNOWN")
        attempts = [json.loads(line) for line in
                    (output / "attempts.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(all(attempt["pid"] == os.getpid() for attempt in attempts))

    def test_synthetic_fixture_root_cannot_overlap_production_root(self) -> None:
        selection = synthetic_selection([synthetic_row(TRAIN_PATH, b"{}\n")])
        # The scan hook prevents payload reads even if this guard regresses.
        with mock.patch.object(backup, "scan_file", side_effect=AssertionError("must never read")) as scan:
            for index, root in enumerate((backup.SOURCE_ROOT,
                                          backup.SOURCE_ROOT / "synthetic_must_not_open",
                                          backup.SOURCE_ROOT.parent)):
                with self.subTest(root=str(root)):
                    output = self.work / f"overlap-output-{index}"
                    with self.assertRaisesRegex(backup.BackupError, "synthetic_fixture_must_be_disjoint"):
                        backup.capture(selection, output, source_root=root, test_fixture=True)
                    self.assertFalse(output.exists())
            scan.assert_not_called()

    def test_output_traversal_is_rejected_before_stat_scan_or_output_writes(self) -> None:
        selection = synthetic_selection([synthetic_row(TRAIN_PATH, b"{}\n")])
        output = self.work / "owned" / ".." / "outside"
        with mock.patch.object(backup, "_check_ancestors", side_effect=AssertionError("must never stat")) as ancestors, \
                mock.patch.object(Path, "resolve", side_effect=AssertionError("must never resolve")) as resolve, \
                mock.patch.object(Path, "mkdir", side_effect=AssertionError("must never create output")) as mkdir, \
                mock.patch.object(backup, "_write_json", side_effect=AssertionError("must never write receipt")) as write, \
                mock.patch.object(backup, "scan_file", side_effect=AssertionError("must never read")) as scan:
            with self.assertRaisesRegex(backup.BackupError, "unsafe_output_path"):
                backup.capture(selection, output, source_root=self.source, test_fixture=True)
            ancestors.assert_not_called()
            resolve.assert_not_called()
            mkdir.assert_not_called()
            write.assert_not_called()
            scan.assert_not_called()
        self.assertFalse((self.work / "outside").exists())
        self.assertFalse((self.work / "owned").exists())

    def test_source_traversal_is_rejected_before_stat_scan_or_output_writes(self) -> None:
        selection = synthetic_selection([synthetic_row(TRAIN_PATH, b"{}\n")])
        roots = (
            self.source / "synthetic-sibling" / "..",
            Path("synthetic-relative-sibling") / ".." / "synthetic-relative-root",
            backup.SOURCE_ROOT.parent / "synthetic-nonexistent-sibling" / ".." / backup.SOURCE_ROOT.name,
        )
        with mock.patch.object(backup, "_check_ancestors", side_effect=AssertionError("must never stat")) as ancestors, \
                mock.patch.object(Path, "resolve", side_effect=AssertionError("must never resolve")) as resolve, \
                mock.patch.object(Path, "mkdir", side_effect=AssertionError("must never create output")) as mkdir, \
                mock.patch.object(backup, "_write_json", side_effect=AssertionError("must never write receipt")) as write, \
                mock.patch.object(backup, "scan_file", side_effect=AssertionError("must never read")) as scan:
            for index, root in enumerate(roots):
                with self.subTest(root=str(root)):
                    with self.assertRaisesRegex(backup.BackupError, "unsafe_source_root"):
                        backup.capture(selection, self.work / f"source-traversal-output-{index}",
                                       source_root=root, test_fixture=True)
            ancestors.assert_not_called()
            resolve.assert_not_called()
            mkdir.assert_not_called()
            write.assert_not_called()
            scan.assert_not_called()
        for index in range(len(roots)):
            self.assertFalse((self.work / f"source-traversal-output-{index}").exists())

    def test_symlink_output_ancestor_is_rejected_when_platform_supports_symlinks(self) -> None:
        selection = synthetic_selection([synthetic_row(TRAIN_PATH, b"{}\n")])
        target = self.work / "synthetic-output-target"
        target.mkdir()
        alias = self.work / "synthetic-output-alias"
        try:
            alias.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"Synthetic directory symlink unavailable on this platform: {error}")
        with mock.patch.object(Path, "mkdir", side_effect=AssertionError("must never create output")) as mkdir, \
                mock.patch.object(backup, "_write_json", side_effect=AssertionError("must never write receipt")) as write, \
                mock.patch.object(backup, "scan_file", side_effect=AssertionError("must never read")) as scan:
            with self.assertRaisesRegex(backup.BackupError, "symlink_or_reparse_point"):
                backup.capture(selection, alias / "capture", source_root=self.source, test_fixture=True)
            mkdir.assert_not_called()
            write.assert_not_called()
            scan.assert_not_called()
        self.assertFalse((target / "capture").exists())

    def test_reparse_output_ancestor_is_rejected_before_scan_or_output_creation(self) -> None:
        selection = synthetic_selection([synthetic_row(TRAIN_PATH, b"{}\n")])
        ancestor = self.work / "synthetic-reparse-output-parent"
        ancestor.mkdir()
        real_lstat = Path.lstat

        def lstat_with_reparse(path: Path, *args: object, **kwargs: object) -> object:
            if path == ancestor:
                return SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_file_attributes=0x400)
            return real_lstat(path, *args, **kwargs)

        with mock.patch.object(Path, "lstat", autospec=True, side_effect=lstat_with_reparse), \
                mock.patch.object(Path, "mkdir", side_effect=AssertionError("must never create output")) as mkdir, \
                mock.patch.object(backup, "_write_json", side_effect=AssertionError("must never write receipt")) as write, \
                mock.patch.object(backup, "scan_file", side_effect=AssertionError("must never read")) as scan:
            with self.assertRaisesRegex(backup.BackupError, "symlink_or_reparse_point"):
                backup.capture(selection, ancestor / "capture", source_root=self.source, test_fixture=True)
            mkdir.assert_not_called()
            write.assert_not_called()
            scan.assert_not_called()
        self.assertFalse((ancestor / "capture").exists())

    def test_describe_cli_rejects_external_selection_before_stat_or_read(self) -> None:
        stderr = io.StringIO()
        with mock.patch.object(backup, "_check_ancestors", side_effect=AssertionError("must never stat")) as ancestors, \
                mock.patch.object(Path, "resolve", side_effect=AssertionError("must never resolve")) as resolve, \
                mock.patch.object(backup, "_local_json", side_effect=AssertionError("must never read configuration")) as local_json, \
                mock.patch.object(backup, "_hash_file", side_effect=AssertionError("must never hash source")) as hash_file, \
                mock.patch("sys.stderr", stderr):
            result = backup.main(["describe", "--selection", str(self.work / "outside.json")])
            ancestors.assert_not_called()
            resolve.assert_not_called()
            local_json.assert_not_called()
            hash_file.assert_not_called()
        self.assertEqual(result, 1)
        failure = json.loads(stderr.getvalue())
        self.assertEqual(failure["status"], "FAILED_BEFORE_CAPTURE")
        self.assertEqual(failure["reason"], "local_configuration_or_artifact_outside_owned_directory")
        self.assertFalse(failure["capture_executed"])
        self.assertFalse(failure["upload_attempted"])

    def test_review_gate_rejects_external_configuration_before_stat_read_or_hash(self) -> None:
        with mock.patch.object(backup, "_check_ancestors", side_effect=AssertionError("must never stat")) as ancestors, \
                mock.patch.object(Path, "resolve", side_effect=AssertionError("must never resolve")) as resolve, \
                mock.patch.object(backup, "_local_json", side_effect=AssertionError("must never read configuration")) as local_json, \
                mock.patch.object(backup, "_hash_file", side_effect=AssertionError("must never hash source")) as hash_file:
            with self.assertRaisesRegex(backup.BackupError,
                                       "local_configuration_or_artifact_outside_owned_directory"):
                backup._review_gate(self.work / "outside-review.json", self.work / "outside-selection.json")
            ancestors.assert_not_called()
            resolve.assert_not_called()
            local_json.assert_not_called()
            hash_file.assert_not_called()

    def write_test_archive(self, path: Path, members: list[tuple[tarfile.TarInfo, bytes]]) -> None:
        with tarfile.open(path, "w:gz") as archive:
            for info, payload in members:
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload) if info.isfile() else None)

    def test_archive_verifier_rejects_extra_duplicate_traversal_and_nonregular_members(self) -> None:
        payload = b'{"text":"synthetic tar fixture"}\n'
        self.write_fixture(TRAIN_PATH, payload)
        outcome, output = self.capture_fixture([synthetic_row(TRAIN_PATH, payload)])
        manifest, _ = self.read_success(outcome, output)
        # Clear compressed-file pins so each malformed tar reaches its member
        # gate. The original identity and content descriptors remain intact.
        unpinned = copy.deepcopy(manifest)
        unpinned.pop("archive_sha256")
        unpinned.pop("archive_size")
        member_name = "source/" + TRAIN_PATH
        regular = tarfile.TarInfo(member_name)
        extra = tarfile.TarInfo("source/synthetic_package/data/train_extra.jsonl")
        traversal = tarfile.TarInfo("source/../outside.jsonl")
        directory = tarfile.TarInfo(member_name)
        directory.type = tarfile.DIRTYPE
        symlink = tarfile.TarInfo(member_name)
        symlink.type = tarfile.SYMTYPE
        symlink.linkname = "synthetic_target"
        cases = {
            "extra": ([(regular, payload), (extra, b"{}\n")], "extra_duplicate_or_nonregular"),
            "duplicate": ([(regular, payload), (copy.copy(regular), payload)], "extra_duplicate_or_nonregular"),
            "traversal": ([(regular, payload), (traversal, b"{}\n")], "unsafe_logical_path"),
            "directory": ([(directory, b"")], "extra_duplicate_or_nonregular"),
            "symlink": ([(symlink, b"")], "extra_duplicate_or_nonregular"),
            "missing": ([], "missing_member_or_archive_changed"),
            "wrong_size": ([(regular, payload + b"\n")], "archive_member_size_mismatch"),
            "wrong_hash": ([(regular, b"x" * len(payload))], "archive_member_hash_mismatch"),
        }
        for name, (members, reason) in cases.items():
            with self.subTest(case=name):
                archive_path = self.work / f"malformed-{name}.tar.gz"
                self.write_test_archive(archive_path, members)
                with self.assertRaisesRegex(backup.BackupError, reason):
                    backup.verify_archive(archive_path, unpinned)

    def test_archive_verifier_reads_gzip_footer_and_rejects_trailing_tar_content(self) -> None:
        payload = b'{"text":"synthetic footer fixture"}\n'
        self.write_fixture(TRAIN_PATH, payload)
        outcome, output = self.capture_fixture([synthetic_row(TRAIN_PATH, payload)])
        manifest, _ = self.read_success(outcome, output)
        unpinned = copy.deepcopy(manifest)
        unpinned.pop("archive_sha256")
        unpinned.pop("archive_size")
        original = (output / "backup.tar.gz").read_bytes()

        damaged = bytearray(original)
        damaged[-8] ^= 1  # Alter only the gzip CRC, after all source bytes.
        corrupt_footer = self.work / "corrupt-footer.tar.gz"
        corrupt_footer.write_bytes(damaged)
        with self.assertRaisesRegex(backup.BackupError, "invalid_or_unreadable_archive"):
            backup.verify_archive(corrupt_footer, unpinned)

        trailing = self.work / "nonzero-trailing-tar.tar.gz"
        trailing.write_bytes(gzip.compress(gzip.decompress(original) + b"SYNTHETIC_TRAILING_BYTES", mtime=0))
        with self.assertRaisesRegex(backup.BackupError, "nonzero_trailing_tar_bytes"):
            backup.verify_archive(trailing, unpinned)

    def test_archive_verifier_requires_two_tar_eof_blocks_even_with_valid_gzip(self) -> None:
        payload = b'{"text":"synthetic tar EOF fixture"}\n'
        self.write_fixture(TRAIN_PATH, payload)
        outcome, output = self.capture_fixture([synthetic_row(TRAIN_PATH, payload)])
        manifest, _ = self.read_success(outcome, output)
        unpinned = copy.deepcopy(manifest)
        unpinned.pop("archive_sha256")
        unpinned.pop("archive_size")
        tar_bytes = gzip.decompress((output / "backup.tar.gz").read_bytes())
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as archive:
            member = archive.getmembers()[0]
            last_member_end = member.offset_data + ((member.size + 511) // 512) * 512
        for eof_bytes in (0, 512):
            with self.subTest(eof_bytes=eof_bytes):
                malformed = self.work / f"missing-tar-eof-{eof_bytes}.tar.gz"
                truncated_tar = tar_bytes[:last_member_end + eof_bytes]
                malformed.write_bytes(gzip.compress(truncated_tar, mtime=0))
                with self.assertRaisesRegex(backup.BackupError, "missing_two_tar_eof_blocks"):
                    backup.verify_archive(malformed, unpinned)

    def test_archive_verifier_checks_nonzero_prefetched_eof_padding(self) -> None:
        payload = b'{"text":"synthetic tar prefetched-padding fixture"}\n'
        self.write_fixture(TRAIN_PATH, payload)
        outcome, output = self.capture_fixture([synthetic_row(TRAIN_PATH, payload)])
        manifest, _ = self.read_success(outcome, output)
        unpinned = copy.deepcopy(manifest)
        unpinned.pop("archive_sha256")
        unpinned.pop("archive_size")
        tar_bytes = bytearray(gzip.decompress((output / "backup.tar.gz").read_bytes()))
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as archive:
            member = archive.getmembers()[0]
            last_member_end = member.offset_data + ((member.size + 511) // 512) * 512
        # Leave the first EOF block intact; poison bytes already prefetched by
        # tarfile after it decides that no further members exist.
        marker = b"SYNTHETIC_HIDDEN_NONZERO"
        start = last_member_end + 512 + 32
        tar_bytes[start:start + len(marker)] = marker
        malformed = self.work / "nonzero-prefetched-eof-padding.tar.gz"
        malformed.write_bytes(gzip.compress(bytes(tar_bytes), mtime=0))
        with self.assertRaisesRegex(backup.BackupError, "nonzero_trailing_tar_bytes"):
            backup.verify_archive(malformed, unpinned)

    def test_symlink_payload_is_rejected_when_platform_supports_symlinks(self) -> None:
        target = self.write_fixture("synthetic_package/data/real.jsonl", b"{}\n")
        link = self.source.joinpath(*TRAIN_PATH.split("/"))
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"Synthetic symlink unavailable on this platform: {error}")
        outcome, output = self.capture_fixture([synthetic_row(TRAIN_PATH, b"{}\n")])
        self.assert_failed_without_success_artifacts(outcome, output)
        self.assertTrue(any("symlink" in failure["reason"].lower()
                            for failure in outcome["failures"]), outcome)

    def test_duplicate_selection_is_fatal_and_counts_remain_unknown(self) -> None:
        selection = synthetic_selection([synthetic_row(TRAIN_PATH, b"{}\n")])
        selection["rows"].append(copy.deepcopy(selection["rows"][0]))
        output = self.work / "duplicate-output"
        outcome = backup.capture(selection, output, source_root=self.source, test_fixture=True)
        self.assert_failed_without_success_artifacts(outcome, output)
        self.assertEqual(set(outcome["counts"].values()), {"UNKNOWN"})
        self.assertTrue(outcome["fatal"]["reason"])

    def test_interruption_preserves_failure_receipt_and_unknown_counts(self) -> None:
        selection = synthetic_selection([synthetic_row(TRAIN_PATH, b"{}\n")])
        output = self.work / "interrupted-output"
        with mock.patch.object(backup, "scan_file", side_effect=KeyboardInterrupt):
            outcome = backup.capture(selection, output, source_root=self.source, test_fixture=True)
        self.assert_failed_without_success_artifacts(outcome, output)
        self.assertEqual(set(outcome["counts"].values()), {"UNKNOWN"})
        self.assertEqual(outcome["fatal"]["stage"], "first_read")
        self.assertEqual(outcome["fatal"]["reason"], "unexpected_KeyboardInterrupt")


if __name__ == "__main__":
    unittest.main()

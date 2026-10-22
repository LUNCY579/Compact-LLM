"""Offline fixtures for exact identity/text contact and immutable audit outputs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if __package__:
    from . import audit_open_exclusions as audit
else:
    import audit_open_exclusions as audit


def row(row_id, text, split=None, **extra):
    value = {"id": row_id, "messages": [{"role": "user", "content": text}], **extra}
    if split:
        value.update(split=split, source_split=split)
    return value


class OpenExclusionAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.candidate = self.root / "candidate-v1"; self.candidate.mkdir()
        self.rows = {"train": [row("train-A", "Name the steward chosen by this amber ledger.", "train", parent_id="training-world")],
                     "dev": [row("dev-A", "Identify the role attached to mint observatory record.", "dev", parent_id="development-world")]}
        self.bank_rows = {
            "process-transition-probe-001": [row("dose-A", "Quartz ciphers orbit lunar lanterns silently.", source_id="source-dose", parent_id="parent-dose")],
            "select-views-001": [row("select-A", "Saffron submarines chart velvet horizons.", source_id="source-select")],
            "arithmetic-monitor-open": [row("monitor-A", "What is 10% of 100?", group_id="monitor-group")],
        }
        self.paths = {name: self.root / (name + ".jsonl") for name in self.bank_rows}
        self.expected_hashes = {}
        self.write_candidate(); self.write_banks()
        for context in (patch.object(audit, "OPEN_BANK_PATHS", self.paths),
                        patch.object(audit, "EXPECTED_BANK_ROWS", {name: 1 for name in self.paths}),
                        patch.object(audit, "EXPECTED_BANK_SHA256", self.expected_hashes)):
            context.start(); self.addCleanup(context.stop)

    def write_candidate(self):
        files = {}
        for split, rows in self.rows.items():
            name = split.upper() + ".jsonl"
            path = self.candidate / name
            path.write_bytes(b"".join(audit.canonical(r) + b"\n" for r in rows))
            files[name] = audit.sha(path)
        manifest = {"schema": 1, "record": "BINDING_SIGNAL_DATA_READINESS",
                    "files_sha256": files, "rows": {k: len(v) for k, v in self.rows.items()}}
        (self.candidate / "BUILD.json").write_bytes(audit.canonical(manifest) + b"\n")

    def write_banks(self):
        for name, rows in self.bank_rows.items():
            self.paths[name].write_bytes(b"".join(audit.canonical(r) + b"\n" for r in rows))
            self.expected_hashes[name] = audit.sha(self.paths[name])

    def audit(self):
        return audit.audit_candidate(self.candidate)

    def contacts(self, result, split="train", bank="process-transition-probe-001"):
        return result["comparisons"][split][bank]["collisions"]

    def test_clean_audit_pins_every_input_and_grants_no_execution_permission(self):
        result = self.audit()
        self.assertEqual(result["status"], "PASS_OPEN_EXCLUSIONS_BUILD_ONLY")
        self.assertEqual(result["candidate_contact_rows"], {"train": 0, "dev": 0})
        self.assertEqual(len(result["input_files"]), 7)
        for item in result["input_files"].values():
            self.assertEqual(item["sha256"], audit.sha(Path(item["path"])))
        for flag in ("training_permission", "evaluation_permission", "quality_claim", "release_permitted",
                     "candidate_files_modified", "filtering_performed", "semantic_contamination_certificate"):
            self.assertIs(result[flag], False)

    def test_cross_field_parent_group_and_nested_source_identity_contact(self):
        self.rows["train"][0]["parent_id"] = "monitor-group"
        self.rows["dev"][0]["provenance"] = {"source_id": "source-dose"}
        self.write_candidate()
        result = self.audit()
        self.assertEqual(result["status"], "BLOCKED_ON_OPEN_EXCLUSION_CONTACT")
        self.assertEqual(result["candidate_contact_rows"], {"train": 1, "dev": 1})
        contacts = self.contacts(result, bank="arithmetic-monitor-open")
        self.assertIn("identifier_contact", contacts[0]["reasons"])
        self.assertEqual(contacts[0]["identifier_matches"][0]["bank_field"], "group_id")
        self.assertEqual(self.contacts(result, "dev")[0]["identifier_matches"][0]["candidate_field"], "provenance.source_id")

    def test_direct_id_contact_is_reported_without_filtering(self):
        self.rows["train"][0]["id"] = "dose-A"
        self.write_candidate()
        before = {p: audit.sha(p) for p in (*self.candidate.iterdir(), *self.paths.values())}
        result = self.audit()
        self.assertIn("panel_id", self.contacts(result)[0]["reasons"])
        self.assertEqual(result["candidate_rows"]["train"], 1)
        self.assertEqual(before, {p: audit.sha(p) for p in before})
        self.assertNotIn("filtered_rows", result)

    def test_unicode_case_and_whitespace_normalized_prompt_match(self):
        self.bank_rows["process-transition-probe-001"][0]["messages"] = [{"role": "user", "content": "ＭＩＬＫ  STATION\nClerk?"}]
        self.rows["train"][0]["messages"] = [{"role": "user", "content": "milk station clerk?"}]
        self.write_candidate(); self.write_banks()
        result = self.audit()
        self.assertIn("normalized_exact_user_prompt", self.contacts(result)[0]["reasons"])
        self.assertEqual(self.contacts(result)[0]["normalized_prompt_bank_ids"], ["dose-A"])

    def test_exact_turn_match_preserves_full_prompt_boundaries(self):
        self.bank_rows["process-transition-probe-001"][0]["messages"] = [
            {"role": "user", "content": "An unrelated opening turn."},
            {"role": "assistant", "content": "Acknowledged."},
            {"role": "user", "content": "Silver fish?"}]
        self.rows["train"][0]["messages"] = [{"role": "user", "content": "Silver fish?"}]
        self.write_candidate(); self.write_banks()
        reasons = self.contacts(self.audit())[0]["reasons"]
        self.assertIn("normalized_exact_user_turn", reasons)
        self.assertNotIn("normalized_exact_user_prompt", reasons)

    def test_shared_eight_gram_is_caught_with_different_full_stem(self):
        self.bank_rows["process-transition-probe-001"][0]["messages"] = [
            {"role": "user", "content": "one two three four five six seven eight red ending"}]
        self.rows["train"][0]["messages"] = [
            {"role": "user", "content": "one two three four five six seven eight blue finale"}]
        self.write_candidate(); self.write_banks()
        reasons = self.contacts(self.audit())[0]["reasons"]
        self.assertIn("user_8gram", reasons)
        self.assertNotIn("digit_blinded_panel_stem", reasons)

    def test_digit_changed_short_arithmetic_stem_is_caught(self):
        self.rows["train"][0]["messages"] = [{"role": "user", "content": "Record entry: What is 25% of 400?"}]
        self.write_candidate()
        contact = self.contacts(self.audit(), bank="arithmetic-monitor-open")[0]
        self.assertIn("digit_blinded_panel_stem", contact["reasons"])
        self.assertEqual(contact["digit_blinded_stem_bank_ids"], ["monitor-A"])
        self.assertNotIn("normalized_exact_user_prompt", contact["reasons"])

    def test_shared_system_and_assistant_content_are_excluded(self):
        common = "Quartz ciphers orbit lunar lanterns silently. What is 10% of 100?"
        self.rows["train"][0]["messages"] = [
            {"role": "system", "content": common},
            *self.rows["train"][0]["messages"], {"role": "assistant", "content": common}]
        self.write_candidate()
        self.assertEqual(self.audit()["status"], "PASS_OPEN_EXCLUSIONS_BUILD_ONLY")

    def test_changed_candidate_bytes_fail_the_frozen_build_hash(self):
        with (self.candidate / "TRAIN.jsonl").open("ab") as stream:
            stream.write(b" ")
        with self.assertRaisesRegex(audit.AuditError, "BUILD hash mismatch"):
            self.audit()

    def test_build_row_counts_and_split_identity_are_verified(self):
        manifest = json.loads((self.candidate / "BUILD.json").read_text())
        manifest["rows"]["train"] = 2
        (self.candidate / "BUILD.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(audit.AuditError, "row counts"):
            self.audit()
        self.rows["train"][0]["source_split"] = "dev"; self.write_candidate()
        with self.assertRaisesRegex(audit.AuditError, "split identity"):
            self.audit()

    def test_empty_or_duplicate_bank_cannot_establish_no_contact(self):
        path = self.paths["process-transition-probe-001"]
        original = path.read_bytes(); path.write_bytes(b"")
        self.expected_hashes["process-transition-probe-001"] = audit.sha(path)
        with self.assertRaisesRegex(audit.AuditError, "empty candidate/bank"):
            self.audit()
        path.write_bytes(original + original)
        self.expected_hashes["process-transition-probe-001"] = audit.sha(path)
        with self.assertRaisesRegex(audit.AuditError, "duplicate row ID"):
            self.audit()

    def test_all_three_complete_banks_are_required(self):
        self.bank_rows["process-transition-probe-001"].append(row("dose-extra", "Completely separate sailing route."))
        self.write_banks()
        with self.assertRaisesRegex(audit.AuditError, "complete open-bank row inventories"):
            self.audit()

    def test_changed_open_bank_bytes_do_not_silently_redefine_exclusion_scope(self):
        with self.paths["select-views-001"].open("ab") as stream:
            stream.write(b" ")
        with self.assertRaisesRegex(audit.AuditError, "frozen SHA256"):
            self.audit()

    def test_report_is_create_only_and_does_not_touch_candidate(self):
        result = self.audit(); output = self.candidate / "OPEN_EXCLUSION_AUDIT.json"
        dataset_sha = audit.sha(self.candidate / "TRAIN.jsonl")
        audit.write_report(output, result)
        saved = output.read_bytes()
        with self.assertRaises(FileExistsError):
            audit.write_report(output, {"status": "changed"})
        self.assertEqual(output.read_bytes(), saved)
        self.assertEqual(audit.sha(self.candidate / "TRAIN.jsonl"), dataset_sha)


if __name__ == "__main__":
    unittest.main()

"""Synthetic contract tests; this fixture is not a historical scorer substitute."""
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

# Direct execution works with -I -S without importing any external package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired_correctness as audit


class SyntheticRule:
    """Small exact/reference and abstention rule for adapter contract tests only."""

    @staticmethod
    def first_line(reply):
        return reply.splitlines()[0].strip() if reply.splitlines() else ""

    @staticmethod
    def canonical(reply):
        return reply.strip().lower().rstrip(".!?")

    @classmethod
    def abstains(cls, reply):
        return cls.canonical(reply) == "abstain"

    @classmethod
    def score_row(cls, row_type, reply, reference):
        first = cls.first_line(reply)
        if row_type == "boundary":
            return cls.abstains(first), cls.abstains(first)
        return first == reference, cls.canonical(first) == cls.canonical(reference)


def fixture():
    rows = [{"id": key, "family": family, "row_type": "answerable",
             "reference": "right", "prompt": [{"role": "user", "content": key}]}
            for key, family in (("both", "one"), ("loss", "one"),
                                ("gain", "two"), ("neither", "two"))]
    rows.append({"id": "boundary", "family": "boundary", "row_type": "boundary",
                 "reference": "ABSTAIN", "prompt": [{"role": "user", "content": "unknown"}]})
    outputs = {"SPARSE": {"both": "right", "loss": "right", "gain": "wrong",
                          "neither": "wrong", "boundary": "ABSTAIN"},
               "TH25": {"both": "right", "loss": "wrong", "gain": "right",
                        "neither": "wrong", "boundary": "ABSTAIN"}}
    arms = []
    for name, replies in outputs.items():
        predictions = []
        for row in rows:
            reply = replies[row["id"]]
            prediction = {"id": row["id"], "family": row["family"],
                          "row_type": row["row_type"], "prediction": reply,
                          "credited": SyntheticRule.score_row(row["row_type"], reply, row["reference"])[1],
                          "generated_tokens": 2}
            if row["row_type"] == "answerable":
                prediction["attempt"] = not SyntheticRule.abstains(reply)
            predictions.append(prediction)
        arms.append({"arm": name, "predictions": predictions, "eos": [2, 3],
                     "summary": {"boundary_abstain": 1, "n_boundary": 1,
                                 "answerable_attempt": 4, "answerable_correct": 2,
                                 "n_answerable": 4}})
    run = {"schema": 1, "instrument": audit.INSTRUMENT,
           "bank_sha256": "synthetic-bank", "rule_module_sha256": "synthetic-rule",
           "runtime": {"python": "synthetic-python", "torch": "synthetic-torch",
                       "transformers": "synthetic-transformers", "device": "cpu",
                       "dtype": "float32", "decoding": "greedy", "max_new_tokens": 48,
                       "thinking_disabled": True, "tf32": False}, "arms": arms}
    return rows, [run]


def evaluate(rows, runs):
    return audit.audit_documents(rows, runs, bank_sha256="synthetic-bank",
                                 scorer_sha256="synthetic-rule", rule=SyntheticRule)


def split_receipts(runs):
    left, right = deepcopy(runs[0]), deepcopy(runs[0])
    left["arms"], right["arms"] = [left["arms"][0]], [right["arms"][1]]
    return [left, right]


class PairingTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.runs = fixture()

    def test_four_correctness_cells_and_descriptive_contract(self):
        report = evaluate(self.rows, self.runs)
        self.assertEqual({p["id"]: p["outcome"] for p in report["pairs"]},
                         {"both": "both_correct", "loss": "loss", "gain": "gain",
                          "neither": "both_incorrect"})
        summary = report["summary"]
        for cell in ("both_correct", "both_incorrect", "source_correct_losses", "source_incorrect_gains"):
            self.assertEqual(summary[cell], 1)
        self.assertEqual(summary["source_correct"], 2)
        self.assertEqual(summary["candidate_correct"], 2)
        self.assertEqual(summary["candidate_minus_source_correct"], 0)
        self.assertEqual(summary["source_correct_loss_rate"], 0.5)
        self.assertEqual(summary["source_incorrect_gain_rate"], 0.5)
        self.assertEqual(report["model_calls"], 0)
        self.assertIs(report["confirmation_claim"], False)
        self.assertEqual(report["stored_flag_mismatches"], [])
        self.assertEqual(report["stored_summary_mismatches"], [])
        self.assertEqual(report["coverage"]["bank_rows"], 5)
        self.assertEqual(report["coverage"]["boundary_rows"], 1)
        self.assertEqual(report["coverage"]["answerable_pairs"], 4)
        self.assertEqual(report["per_family"]["one"]["source_correct_losses"], 1)
        self.assertEqual(report["per_family"]["two"]["source_incorrect_gains"], 1)

    def test_pairing_is_by_id_independent_of_all_row_orders(self):
        expected = evaluate(self.rows, self.runs)
        self.rows.reverse()
        self.runs[0]["arms"][0]["predictions"] = self.runs[0]["arms"][0]["predictions"][2:] + self.runs[0]["arms"][0]["predictions"][:2]
        self.runs[0]["arms"][1]["predictions"].reverse()
        self.assertEqual(evaluate(self.rows, self.runs), expected)

    def test_separate_receipts_are_equivalent(self):
        self.assertEqual(evaluate(self.rows, split_receipts(self.runs)), evaluate(self.rows, self.runs))

    def test_duplicate_bank_ids_rejected_including_boundary(self):
        for row in self.rows:
            with self.subTest(id=row["id"]):
                with self.assertRaisesRegex(audit.AuditError, "bank: duplicate id"):
                    evaluate(self.rows + [deepcopy(row)], self.runs)

    def test_duplicate_prediction_ids_rejected_including_boundary(self):
        for arm_index in (0, 1):
            for row_index in (0, 4):
                runs = deepcopy(self.runs)
                predictions = runs[0]["arms"][arm_index]["predictions"]
                predictions.append(deepcopy(predictions[row_index]))
                with self.subTest(arm=arm_index, row=row_index):
                    with self.assertRaisesRegex(audit.AuditError, "duplicate id"):
                        evaluate(self.rows, runs)

    def test_missing_or_extra_prediction_ids_rejected(self):
        for arm_index in (0, 1):
            for row_index in (0, 4):
                runs = deepcopy(self.runs)
                runs[0]["arms"][arm_index]["predictions"].pop(row_index)
                with self.subTest(arm=arm_index, missing=row_index):
                    with self.assertRaisesRegex(audit.AuditError, "incompatible IDs; missing=\\['"):
                        evaluate(self.rows, runs)
            runs = deepcopy(self.runs)
            extra = deepcopy(runs[0]["arms"][arm_index]["predictions"][0])
            extra["id"] = "extra"
            runs[0]["arms"][arm_index]["predictions"].append(extra)
            with self.subTest(arm=arm_index, extra=True):
                with self.assertRaisesRegex(audit.AuditError, "extra=\\['extra'\\]"):
                    evaluate(self.rows, runs)

    def test_duplicate_arms_rejected_within_and_across_receipts(self):
        for split in (False, True):
            runs = deepcopy(self.runs)
            if split:
                runs.append(deepcopy(runs[0]))
            else:
                runs[0]["arms"].append(deepcopy(runs[0]["arms"][0]))
            with self.subTest(split=split):
                with self.assertRaisesRegex(audit.AuditError, "duplicate arm"):
                    evaluate(self.rows, runs)

    def test_missing_or_identical_requested_arms_rejected(self):
        self.runs[0]["arms"].pop()
        with self.assertRaisesRegex(audit.AuditError, "missing requested"):
            evaluate(self.rows, self.runs)
        with self.assertRaisesRegex(audit.AuditError, "source and candidate must differ"):
            audit.audit_documents(self.rows, self.runs, bank_sha256="synthetic-bank",
                                  scorer_sha256="synthetic-rule", rule=SyntheticRule,
                                  source_arm="SPARSE", candidate_arm="SPARSE")

    def test_zero_denominators_are_null(self):
        for all_correct in (False, True):
            runs = deepcopy(self.runs)
            for arm in runs[0]["arms"]:
                for pred in arm["predictions"]:
                    if pred["row_type"] == "answerable":
                        pred["prediction"], pred["credited"] = ("right" if all_correct else "wrong"), all_correct
                arm["summary"]["answerable_correct"] = 4 if all_correct else 0
            summary = evaluate(self.rows, runs)["summary"]
            with self.subTest(all_correct=all_correct):
                field = "source_incorrect_gain_rate" if all_correct else "source_correct_loss_rate"
                denominator = "source_incorrect_gain_denominator" if all_correct else "source_correct_loss_denominator"
                self.assertIsNone(summary[field])
                self.assertEqual(summary[denominator], 0)

    def test_stored_flags_and_summaries_are_reported_not_used_for_correctness(self):
        source = self.runs[0]["arms"][0]
        source["predictions"][0]["credited"] = False
        source["predictions"][0]["attempt"] = False
        source["predictions"][4]["credited"] = False
        source["summary"]["answerable_correct"] = 999
        report = evaluate(self.rows, self.runs)
        self.assertEqual(report["summary"]["source_correct"], 2)
        self.assertEqual(report["summary"]["source_attempts"], 4)
        self.assertEqual({(m["id"], m["field"]) for m in report["stored_flag_mismatches"]},
                         {("both", "credited"), ("both", "attempt"), ("boundary", "credited")})
        self.assertEqual(report["stored_summary_mismatches"],
                         [{"arm": "SPARSE", "field": "answerable_correct", "stored": 999, "rescored": 2}])

    def test_first_line_punctuation_and_token_cap_diagnostics(self):
        prediction = self.runs[0]["arms"][1]["predictions"][0]
        prediction["prediction"] = "RIGHT.\nsecond line is ignored by this synthetic rule"
        prediction["generated_tokens"] = 48
        report = evaluate(self.rows, self.runs)
        pair = next(p for p in report["pairs"] if p["id"] == "both")
        self.assertEqual(pair["outcome"], "both_correct")
        self.assertIs(pair["candidate"]["strict_correct"], False)
        self.assertIs(pair["candidate"]["correct"], True)
        self.assertIs(pair["candidate"]["multiline"], True)
        self.assertIs(pair["candidate"]["at_token_cap"], True)
        self.assertIs(pair["canonical_first_line_changed"], False)
        self.assertEqual(report["format_diagnostics"]["changed_text_same_canonical_pairs"], 1)
        self.assertEqual(report["format_diagnostics"]["canonical_credit_beyond_strict"]["TH25"], 1)

    def test_second_line_correct_answer_does_not_gain_synthetic_credit(self):
        prediction = self.runs[0]["arms"][1]["predictions"][2]
        prediction["prediction"], prediction["credited"] = "wrong\nright", False
        self.runs[0]["arms"][1]["summary"]["answerable_correct"] = 1
        report = evaluate(self.rows, self.runs)
        self.assertEqual(next(p for p in report["pairs"] if p["id"] == "gain")["outcome"], "both_incorrect")

    def test_abstention_changes_attempt_denominator(self):
        pred = self.runs[0]["arms"][1]["predictions"][2]
        pred.update(prediction="ABSTAIN", attempt=False, credited=False)
        self.runs[0]["arms"][1]["summary"].update(answerable_attempt=3, answerable_correct=1)
        report = evaluate(self.rows, self.runs)
        self.assertEqual(report["summary"]["candidate_attempts"], 3)
        self.assertEqual(report["summary"]["paired_rows"], 4)


class ProvenanceTests(unittest.TestCase):
    def test_bank_and_scorer_hash_mismatch_rejected(self):
        for field in ("bank_sha256", "rule_module_sha256"):
            rows, runs = fixture()
            runs[0][field] = "other"
            with self.subTest(field=field):
                with self.assertRaisesRegex(audit.AuditError, "sha256 provenance"):
                    evaluate(rows, runs)

    def test_missing_and_invalid_runtime_fields_rejected(self):
        invalid = {"python": "", "torch": None, "transformers": 7, "device": False,
                   "dtype": "float16", "decoding": "sample", "max_new_tokens": True,
                   "thinking_disabled": 1, "tf32": 0}
        for field in audit.RUNTIME_KEYS:
            for missing in (False, True):
                rows, runs = fixture()
                if missing:
                    runs[0]["runtime"].pop(field)
                else:
                    runs[0]["runtime"][field] = invalid[field]
                with self.subTest(field=field, missing=missing):
                    with self.assertRaises(audit.AuditError):
                        evaluate(rows, runs)

    def test_runtime_difference_between_receipts_rejected(self):
        rows, runs = fixture()
        runs = split_receipts(runs)
        runs[1]["runtime"]["torch"] = "other-runtime-version"
        with self.assertRaisesRegex(audit.AuditError, "across receipts"):
            evaluate(rows, runs)

    def test_invalid_or_incompatible_eos_rejected(self):
        for eos in (None, [], [2, 2], [True], [-1], ["2"], [2, 4]):
            rows, runs = fixture()
            runs[0]["arms"][1]["eos"] = eos
            with self.subTest(eos=eos):
                with self.assertRaisesRegex(audit.AuditError, "EOS provenance"):
                    evaluate(rows, runs)

    def test_receipt_schema_and_instrument_rejected(self):
        for field, bad in (("schema", True), ("schema", 2), ("instrument", "other")):
            rows, runs = fixture()
            runs[0][field] = bad
            with self.subTest(field=field, bad=bad):
                with self.assertRaisesRegex(audit.AuditError, "schema/instrument"):
                    evaluate(rows, runs)

    def test_missing_prediction_provenance_rejected(self):
        for field in ("credited", "attempt", "generated_tokens", "prediction", "family", "row_type"):
            rows, runs = fixture()
            runs[0]["arms"][0]["predictions"][0].pop(field)
            with self.subTest(field=field):
                with self.assertRaises(audit.AuditError):
                    evaluate(rows, runs)

    def test_nonboolean_flags_and_invalid_token_counts_rejected(self):
        for field, values in (("credited", (0, 1, None, "true")),
                              ("attempt", (0, 1, None, "false")),
                              ("generated_tokens", (-1, 49, True, 1.5))):
            for value in values:
                rows, runs = fixture()
                runs[0]["arms"][0]["predictions"][0][field] = value
                with self.subTest(field=field, value=value):
                    with self.assertRaises(audit.AuditError):
                        evaluate(rows, runs)

    def test_incompatible_prediction_metadata_rejected_including_boundary(self):
        for row_index in (0, 4):
            for field in ("family", "row_type"):
                rows, runs = fixture()
                runs[0]["arms"][1]["predictions"][row_index][field] = "other"
                with self.subTest(field=field, row=row_index):
                    with self.assertRaisesRegex(audit.AuditError, "incompatible row metadata"):
                        evaluate(rows, runs)

    def test_missing_bank_provenance_and_noncanonical_ids_rejected(self):
        for field in ("reference", "family", "prompt", "row_type"):
            rows, runs = fixture()
            rows[0].pop(field)
            with self.subTest(field=field):
                with self.assertRaises(audit.AuditError):
                    evaluate(rows, runs)
        for value in (None, "", " both", "both ", 1):
            rows, runs = fixture()
            rows[0]["id"] = value
            with self.subTest(id=value):
                with self.assertRaisesRegex(audit.AuditError, "canonical nonblank"):
                    evaluate(rows, runs)

    def test_missing_summary_and_boolean_summary_values_rejected(self):
        for missing in (False, True):
            rows, runs = fixture()
            summary = runs[0]["arms"][0]["summary"]
            if missing:
                summary.pop("answerable_correct")
            else:
                summary["answerable_correct"] = True
            with self.subTest(missing=missing):
                with self.assertRaisesRegex(audit.AuditError, "missing/invalid summary"):
                    evaluate(rows, runs)

    def test_duplicate_json_keys_and_nonfinite_constants_rejected(self):
        for raw in ('{"id":"one","id":"two"}', '{"value":NaN}', '{"value":Infinity}'):
            with self.subTest(raw=raw):
                with self.assertRaises(audit.AuditError):
                    audit.decode(raw)

    def test_audit_paths_rejects_unpinned_scorer_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bank, scorer = root / "bank.jsonl", root / "rule.py"
            bank.write_text(json.dumps(fixture()[0][0]) + "\n", encoding="utf-8")
            scorer.write_text("raise AssertionError('this must not execute')\n", encoding="utf-8")
            with self.assertRaisesRegex(audit.AuditError, "frozen scorer unavailable/incompatible"):
                audit.audit_paths(bank, [], scorer)

    def test_malformed_historical_result_shapes_fail_clearly(self):
        # Mock the upstream scoring adapter solely to exercise result shape guards.
        # The temporary empty module is explicitly synthetic, never historical evidence.
        malformed = [None, [], {"instrument": "CONFIRM-002", "h_bank_sha256": "BANK", "descriptive": None},
                     {"instrument": "CONFIRM-002", "h_bank_sha256": "BANK", "descriptive": []},
                     {"instrument": "CONFIRM-002", "h_bank_sha256": "BANK",
                      "descriptive": {"answerable_correct": [], "answerable_attempt": {}}}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bank, scorer, result = root / "bank.jsonl", root / "rule.py", root / "result.json"
            bank.write_bytes(b"{}\n")
            scorer.write_bytes(b"# empty synthetic module for result validation only\n")
            bank_hash = audit.digest(bank.read_bytes())
            report = {"summary": {"source_correct": 2, "candidate_correct": 2,
                                  "source_attempts": 4, "candidate_attempts": 4}}
            for document in malformed:
                if isinstance(document, dict):
                    document["h_bank_sha256"] = bank_hash
                result.write_text(json.dumps(document), encoding="utf-8")
                with self.subTest(document=document):
                    with patch.object(audit, "FROZEN_RULE_SHA256", audit.digest(scorer.read_bytes())), \
                            patch.object(audit, "audit_documents", return_value=deepcopy(report)):
                        with self.assertRaises(audit.AuditError):
                            audit.audit_paths(bank, [], scorer, result_path=result)


class CliWriteBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.evidence = self.root / "evidence"
        self.evidence.mkdir()
        self.bank = self.evidence / "bank.jsonl"
        self.scorer = self.evidence / "rule.py"
        self.run = self.evidence / "run.json"
        self.bank.write_text("{}\n", encoding="utf-8")
        self.scorer.write_text("# deliberately unpinned synthetic fixture\n", encoding="utf-8")
        self.run.write_text("{}\n", encoding="utf-8")
        self.output = self.root / "derived.json"
        self.args = ["--bank", str(self.bank), "--runs", str(self.run),
                     "--scorer", str(self.scorer), "--output", str(self.output)]

    def invoke(self, args=None):
        out, error = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(error):
            code = audit.main(args or self.args)
        return code, out.getvalue(), error.getvalue()

    def test_failed_audit_creates_no_output(self):
        code, out, error = self.invoke()
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("AUDIT_FAILED", error)
        self.assertFalse(self.output.exists())

    def test_existing_output_is_preserved_when_write_is_attempted(self):
        self.output.write_bytes(b"existing evidence must survive\n")
        report = {"summary": {}, "stored_flag_mismatches": []}
        with patch.object(audit, "audit_paths", return_value=report):
            code, out, error = self.invoke()
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("AUDIT_FAILED", error)
        self.assertEqual(self.output.read_bytes(), b"existing evidence must survive\n")

    def test_output_inside_evidence_directory_is_blocked_before_audit(self):
        output = self.evidence / "derived" / "report.json"
        args = self.args[:-1] + [str(output)]
        with patch.object(audit, "audit_paths") as run:
            code, out, error = self.invoke(args)
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("outside every original evidence/scorer directory", error)
        run.assert_not_called()
        self.assertFalse(output.exists())

    def test_successful_mocked_audit_writes_create_only_lf_json(self):
        # This tests CLI serialization only, never substitutes a historical scorer.
        report = {"summary": {"paired_rows": 4}, "stored_flag_mismatches": []}
        with patch.object(audit, "audit_paths", return_value=report):
            code, out, error = self.invoke()
        self.assertEqual(code, 0)
        self.assertEqual(error, "")
        self.assertEqual(json.loads(self.output.read_bytes()), report)
        self.assertNotIn(b"\r", self.output.read_bytes())
        self.assertEqual(json.loads(out)["summary"], report["summary"])


if __name__ == "__main__":
    unittest.main()

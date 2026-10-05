"""CPU-only schedule and adversarial draft checks; no framework or model."""
from collections import Counter
import copy
from decimal import Decimal
import json
from pathlib import Path
import sys
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
import study_schedule as study


def resign(body):
    body["sha256"] = study.digest({k: v for k, v in body.items() if k != "sha256"})
    return body


class StudyScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body = study.build_registration()
        cls.occurrences = json.loads((study.BUNDLE / "training_occurrences.json").read_text())["row_ids"]
        cls.pools = {arm: {row["row_id"]: row for row in
                         [json.loads(line) for line in (study.BUNDLE / ("token_pool_" + arm + ".jsonl")).read_text().splitlines()]}
                     for arm in ("M1", "M2")}
        cls.records = {row["row_id"]: row for row in
                       [json.loads(line) for line in (study.BUNDLE / "candidate_records.jsonl").read_text().splitlines()]}

    def test_full_cpu_audit_and_framework_free_import(self):
        result = study.validate_registration(self.body)
        self.assertEqual("PASS_CPU_STUDY_SCHEDULE_AUDIT_NOT_LAUNCH_PERMISSION", result["status"])
        self.assertFalse(result["native_launch_authorized"])
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)

    def test_explicit_LR_values_independent_decimal_reference(self):
        expected = [0.00005, 0.0001] + [float(Decimal("0.0001") * Decimal(27 - u) / Decimal(25)) for u in range(3, 28)]
        self.assertEqual(expected, study.learning_rates())
        self.assertEqual(27, len(expected))
        self.assertEqual(0.000096, expected[2])
        self.assertEqual(0.000004, expected[-2])
        self.assertEqual(0.0, expected[-1])

    def test_independent_chunk_walk_covers_exact_source_and_measured_arrays(self):
        for arm in ("M1", "M2"):
            chunks = self.body["arms"][arm]["resume_chunks"]
            flattened = [row for chunk in chunks for row in chunk["occurrence_row_ids"]]
            self.assertEqual(self.occurrences[arm], flattened)
            self.assertEqual([(1, 180), (181, 360), (361, 540)],
                             [(c["first_occurrence_ordinal"], c["last_occurrence_ordinal"]) for c in chunks])
            self.assertEqual([(1, 9), (10, 18), (19, 27)],
                             [(c["first_update_ordinal"], c["last_update_ordinal"]) for c in chunks])
            for chunk in chunks:
                ids = chunk["occurrence_row_ids"]
                self.assertEqual(180, len(ids))
                self.assertEqual(sum(len(self.pools[arm][row]["input_ids"]) for row in ids), chunk["input_tokens"])
                self.assertEqual(sum(sum(label != -100 for label in self.pools[arm][row]["labels"]) for row in ids), chunk["supervised_tokens"])
                self.assertEqual(dict(Counter(self.records[row]["parent_id"] for row in ids)), chunk["parent_exposures"])
            self.assertEqual(study.EXPECTED_TOTALS[arm][0], sum(c["input_tokens"] for c in chunks))

    def test_final_zero_LR_update_consumes_all20_sequences(self):
        for arm in ("M1", "M2"):
            update = self.body["arms"][arm]["updates"][-1]
            self.assertEqual(0.0, update["learning_rate"])
            self.assertEqual(20, update["sequences"])
            self.assertEqual(self.occurrences[arm][520:540], update["occurrence_row_ids"])
            self.assertEqual((521, 540), (update["first_occurrence_ordinal"], update["last_occurrence_ordinal"]))

    def test_shared_parent_vector_seed_inventory_loss_and_closed_M0(self):
        self.assertEqual(self.body["arms"]["M1"]["totals"]["parent_exposures"], self.body["arms"]["M2"]["totals"]["parent_exposures"])
        self.assertEqual(101, self.body["seed"])
        self.assertEqual(112, self.body["adaptation"]["trainable_tensors"])
        self.assertEqual(1146880, self.body["adaptation"]["trainable_parameters"])
        self.assertEqual("completion_plus_EOS_mean_per_sequence_then_mean_over_accumulation", self.body["adaptation"]["loss"])
        self.assertEqual(0, self.body["arms"]["M0"]["optimizer_updates"])
        self.assertEqual(0, self.body["arms"]["M0"]["trainable_parameters"])

    def test_native7_scope_does_not_clear_study_gates_or_enable_STATE1_warmstart(self):
        self.assertIn("short_mechanics_only", self.body["native7_binding"]["scope"])
        self.assertFalse(self.body["resume_contract"]["qualification_STATE1_warmstart_allowed"])
        self.assertFalse(self.body["native_launch_authorized"])
        self.assertTrue(all(value is False for value in self.body["permissions"].values()))
        for token in ("sustained", "language", "long_context", "evaluation", "scorer", "statistics", "cost", "authorization"):
            self.assertTrue(any(token in gate for gate in self.body["pending_gates"]), token)

    def test_missing_final_batch_rejected_even_when_resigned(self):
        body = copy.deepcopy(self.body)
        body["arms"]["M1"]["resume_chunks"][-1]["occurrence_row_ids"] = body["arms"]["M1"]["resume_chunks"][-1]["occurrence_row_ids"][:-20]
        with self.assertRaisesRegex(ValueError, "full updates"):
            study.validate_registration(resign(body))

    def test_reordered_source_occurrence_rejected_even_with_new_chunkhash(self):
        body = copy.deepcopy(self.body)
        chunk = body["arms"]["M2"]["resume_chunks"][0]
        chunk["occurrence_row_ids"][0], chunk["occurrence_row_ids"][1] = chunk["occurrence_row_ids"][1], chunk["occurrence_row_ids"][0]
        chunk["occurrence_order_sha256"] = study.digest(chunk["occurrence_row_ids"])
        with self.assertRaisesRegex(ValueError, "source occurrence order"):
            study.validate_registration(resign(body))

    def test_wrong_global_LR_resume_ordinal_rejected(self):
        body = copy.deepcopy(self.body)
        body["arms"]["M1"]["updates"][9]["learning_rate"] = 0.00005
        with self.assertRaisesRegex(ValueError, "global LR ordinal"):
            study.validate_registration(resign(body))

    def test_forged_chunk_counts_or_parent_vector_rejected(self):
        for field in ("input_tokens", "supervised_tokens", "parent_exposures"):
            body = copy.deepcopy(self.body)
            chunk = body["arms"]["M1"]["resume_chunks"][0]
            if field == "parent_exposures":
                chunk[field][next(iter(chunk[field]))] += 1
            else:
                chunk[field] += 1
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "tokens or parent vector"):
                study.validate_registration(resign(body))

    def test_laundered_launch_and_qualification_warmstart_rejected(self):
        for kind in ("launch", "STATE1"):
            body = copy.deepcopy(self.body)
            if kind == "launch":
                body["native_launch_authorized"] = True
            else:
                body["resume_contract"]["qualification_STATE1_warmstart_allowed"] = True
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "closed draft"):
                study.validate_registration(resign(body))

    def test_modified_source_pin_rejected(self):
        body = copy.deepcopy(self.body)
        body["source_and_decision_pins"][next(iter(body["source_and_decision_pins"]))] = "0" * 64
        with self.assertRaisesRegex(ValueError, "byte pins changed"):
            study.validate_registration(resign(body))

    def test_prompt_mask_EOS_padding_and_bool_token_negative_controls(self):
        row = next(iter(self.pools["M1"].values()))
        for kind in ("mask", "EOS", "padding", "bool"):
            bad = copy.deepcopy(row)
            if kind == "mask":
                bad["labels"][0] = bad["input_ids"][0]
            elif kind == "EOS":
                bad["completion_ids"][-1] = 123
            elif kind == "padding":
                bad["attention_mask"][-1] = 0
            else:
                bad["input_ids"][0] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                study.validate_token_row(bad)


if __name__ == "__main__":
    unittest.main()

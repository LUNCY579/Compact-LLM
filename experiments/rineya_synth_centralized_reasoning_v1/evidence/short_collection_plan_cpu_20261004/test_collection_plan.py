"""CPU plan invariants; no framework, model, scoring or native child."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collection_plan as plan


def row(index, length=10):
    return {"row_id": plan.digest([index]), "prompt_sha256": plan.digest(["text", index]),
        "prompt_ids": [5] * length, "prompt_tokens": length, "attention_mask": [1] * length}


class PlanControls(unittest.TestCase):
    def test_full_bound_is_sum_of_each_actual_forward_prefix(self):
        rows = [row(1, 10), row(2, 480)]
        batch = plan.batches(rows, "atomic_RS_DEV")[0]
        prefixes = sum(r["prompt_tokens"] + step for r in rows for step in range(32))
        self.assertEqual(prefixes, batch["maximum_capture_input_positions"])
        self.assertEqual(prefixes + 20, batch["maximum_total_input_positions"])
        self.assertEqual(66, batch["maximum_total_forwards"])
        self.assertTrue(all(r["prompt_tokens"] + 32 <= 512 for r in batch["rows"]))

    def test_boundaries_preserve_all_rows_order_and_partial_last_batch(self):
        rows = [row(i) for i in range(130)]
        batches = plan.batches(rows, "task_guard_DEV")
        self.assertEqual([64, 64, 2], [len(b["rows"]) for b in batches])
        self.assertEqual(rows, [r for b in batches for r in b["rows"]])
        self.assertEqual([(0, 64), (64, 128), (128, 130)], [(b["start"], b["stop"]) for b in batches])
        self.assertEqual([1, 2, 3], [b["ordinal"] for b in batches])

    def test_batches_do_not_alias_the_source(self):
        rows = [row(1)]
        output = plan.batches(rows, "atomic_RS_DEV")
        output[0]["rows"][0]["prompt_ids"][0] = 99
        self.assertEqual(5, rows[0]["prompt_ids"][0])

    def test_private_fields_duplicate_ids_overlong_and_bad_masks_are_rejected(self):
        bad = row(1); bad["answer"] = [2]
        masked = row(1); masked["attention_mask"][0] = 0
        for rows in ([bad], [row(1), row(1)], [row(1, 481)], [masked]):
            with self.subTest(rows=rows[0]["row_id"]):
                with self.assertRaises(ValueError): plan.batches(rows, "atomic_RS_DEV")

    def test_source_population_is_explicit_and_plan_never_authorizes_execution(self):
        with self.assertRaises(ValueError): plan.batches([row(1)], "language")
        with self.assertRaises(ValueError): plan.batches([], "atomic_RS_DEV")
        batch = plan.batches([row(1)], "atomic_RS_DEV")[0]
        self.assertFalse(batch["native_launch_authorized"])
        self.assertFalse(batch["scoring_authorized"])

    def test_actual_source_plan_matches_denominators_and_prompt_only_inventory(self):
        result = plan.build()
        self.assertEqual(1152, result["rows"])
        self.assertEqual({"atomic_RS_DEV": 6, "task_guard_DEV": 12},
                         {kind: len(items) for kind, items in result["batches"].items()})
        for items in result["batches"].values():
            for batch in items:
                self.assertEqual(plan.digest(batch["rows"]), batch["source_packet_sha256"])
                self.assertEqual(2050, batch["maximum_total_forwards"])
                self.assertTrue(all(set(r) == {"row_id", "prompt_sha256", "prompt_ids", "prompt_tokens", "attention_mask"}
                                    for r in batch["rows"]))
        self.assertFalse(result["native_launch_authorized"])
        self.assertFalse(result["language_qualification_established"])
        self.assertFalse(result["prior_correct_enrollment_performed"])


if __name__ == "__main__": unittest.main()

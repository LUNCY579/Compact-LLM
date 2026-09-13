"""Frozen-source parity and semantic regressions for the pure failure classifier."""
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import unittest

from experiments.rab001_exploration import headroom_audit as headroom


class FailureStratumTests(unittest.TestCase):
    def test_all_150_frozen_rows_match_both_saved_policy_partitions(self):
        saved_path = headroom.OUT / "HEADROOM_AUDIT.json"
        before = hashlib.sha256(saved_path.read_bytes()).hexdigest()
        expected = json.loads(saved_path.read_text())
        frozen = json.loads((headroom.ROOT / "outputs/retention-a329eb7b/B28/retention.json").read_text())
        labels = json.loads((headroom.OUT / "PROBE_LABEL_AUDIT.json").read_text())
        panel = {row["id"]: row for row in (json.loads(line) for line in
                 (headroom.ROOT / "data/monitor_probe.jsonl").read_text().splitlines())}
        self.assertEqual(len(frozen["rows"]), 150)
        for policy in headroom.retention.BARE_POLICIES:
            with self.subTest(policy=policy):
                targets = headroom.retention.build_overlay(labels, policy)["targets"]
                ids, by_class, correct = defaultdict(list), defaultdict(Counter), set()
                for row in frozen["rows"]:
                    rid, source = row["id"], row["source"]
                    category = headroom.failure_stratum(source["raw"], source["terminated"], targets[rid])
                    if category is None:
                        correct.add(rid)
                    else:
                        ids[category].append(rid)
                        by_class[panel[rid]["reply_class"]][category] += 1
                saved = expected["policies"][policy]
                self.assertEqual({name: sorted(members) for name, members in ids.items()}, saved["ids"])
                self.assertEqual({name: len(members) for name, members in ids.items()}, saved["buckets"])
                self.assertEqual({name: dict(counts) for name, counts in by_class.items()}, saved["by_reply_class"])
                self.assertEqual(correct, set(frozen["reference_ids_by_policy"][policy]))
        self.assertEqual(hashlib.sha256(saved_path.read_bytes()).hexdigest(), before)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("experiments.architecture_benchmark_001.run", sys.modules)

    def test_parsed_value_mismatch_precedes_permissive_number_matching(self):
        target = {"kind": "number", "value": "1", "bare_allowed": True}
        # The permissive matcher sees the numerator 1, but the grammar correctly parses 1/2.
        self.assertTrue(headroom.asserts_target("1/2", target))
        self.assertEqual(headroom.failure_stratum("1/2", True, target), "wrong_answer")

    def test_nontermination_precedes_correct_text_and_semantic_matching(self):
        target = {"kind": "number", "value": "1", "bare_allowed": True}
        self.assertEqual(headroom.failure_stratum("1", False, target), "runaway")
        self.assertIsNone(headroom.failure_stratum("1", True, target))

    def test_compound_duration_is_total_not_component(self):
        target = {"kind": "duration", "value": "1500", "bare_allowed": True, "bare_unit": "second"}
        self.assertEqual(headroom.duration_seconds("1 hour and 25 minutes"), {Fraction(5100)})
        self.assertEqual(headroom.failure_stratum("1 hour and 25 minutes", True, target), "wrong_answer")
        self.assertEqual(headroom.failure_stratum("The duration is 1 hour and 25 minutes.", True, target), "wrong_answer")
        correct = {**target, "value": "5100"}
        self.assertEqual(headroom.failure_stratum("The duration is 1 hour and 25 minutes.", True, correct), "format_only")

    def test_clock_semantics_distinguish_format_from_wrong_meridiem(self):
        target = {"kind": "clock", "minute_of_day": 1135}
        self.assertEqual(headroom.failure_stratum("It ends at 6:55 PM.", True, target), "format_only")
        self.assertEqual(headroom.failure_stratum("It ends at 6:55 AM.", True, target), "wrong_answer")


if __name__ == "__main__":
    unittest.main()

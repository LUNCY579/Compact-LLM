"""Metadata/statistical controls; no model efficacy or protected inputs."""
import copy
import unittest

from experiments.rineya_quality_queue_v1.pattern_audit import (
    evaluation_parents, paired_process, parse_pruning_table, retained_summary, training_parents,
)


class PatternAuditTests(unittest.TestCase):
    def test_original_retention_excludes_newly_correct_union_members(self):
        item = {"source_correct_total": 47, "lost_total": 3, "arm_correct_total": 72,
                "arm_correct_outside_source": 28, "reference_size": 75, "status": "FAIL",
                "per_class": {"example": {"retained": 72}}}
        value = {"by_bare_policy": {"permissive": item, "strict": item},
                 "probe_rows": 150, "probe_sha256": "source", "reference_digest": "scorer",
                 "samples": [{}] * 24}
        result = retained_summary(value)
        self.assertEqual(result["policies"]["permissive"]["retained"], 44)
        self.assertEqual(result["policies"]["permissive"]["union_per_class_retained_sum"], 72)
        self.assertFalse(result["complete_retention_identity_rederivation"])
        item["arm_correct_outside_source"] = 27
        with self.assertRaises(ValueError):
            retained_summary(value)

    def test_net_gain_keeps_repairs_and_damages_separate(self):
        before = {"views_sha256": "bank", "implementation_digest": "scorer",
                  "predictions": [{"id": "a", "canonical": False}, {"id": "b", "canonical": True},
                                  {"id": "c", "canonical": False}]}
        after = copy.deepcopy(before)
        for row, correct in zip(after["predictions"], (True, False, True)):
            row["canonical"] = correct
        result = paired_process(before, after)
        self.assertEqual((result["repairs"], result["damages"], result["net"]), (2, 1, 1))
        after["predictions"][0]["id"] = "different"
        with self.assertRaises(ValueError):
            paired_process(before, after)

    def test_bank_or_scorer_change_cannot_be_called_paired(self):
        before = {"views_sha256": "a", "implementation_digest": "s", "predictions": []}
        with self.assertRaises(ValueError):
            paired_process(before, dict(before, views_sha256="b"))
        with self.assertRaises(ValueError):
            paired_process(before, dict(before, implementation_digest="other"))

    def test_actual_build_seed_has_priority_over_wrong_plan_metadata(self):
        parents, seed, method = training_parents({"selected_parent_ids": ["p", "q"], "selection_seed": 23},
                                                 {"selection_seed": 17})
        self.assertEqual(seed, 23)
        self.assertEqual(parents, {"p", "q"})
        self.assertEqual(method, "EXPLICIT_BUILD_PARENT_IDS")

    def test_derived_transition_ids_exclude_replay_and_keep_derivation_explicit(self):
        row = "RRFR2-" + "a" * 24 + "-step7-valid"
        parents, seed, method = training_parents({"transition_draw": {"selection_seed": 29}},
            {"row_order_per_exposure": [row, "rehearsal:owned-record"]})
        self.assertEqual(parents, {"RRFR2-" + "a" * 24})
        self.assertEqual(seed, 29)
        self.assertEqual(method, "PLAN_TRANSITION_ID_PREFIX_DERIVATION")
        with self.assertRaises(ValueError):
            training_parents({"seed": 17}, {"row_order_per_exposure": ["unknown-layout"]})
        with self.assertRaises(ValueError):
            training_parents({"seed": 17}, {"row_order_per_exposure": [row, row]})

    def test_pruning_anchor_is_not_an_independent_new_confirmation(self):
        rows = []
        for seed in (17, 23, 29):
            for arm, retained in (("ctrl", 38), ("k25", 39)):
                rows.append(f"| {seed} | {arm} | 230 | 3/3 | 70 | 60 | {retained} | 150 | 0.5 | 8 |")
        result = parse_pruning_table("\n".join(rows))
        self.assertEqual([row["replication_role"] for row in result],
                         ["selection_anchor", "prospective_new_draw", "prospective_new_draw"])
        self.assertEqual(result[0]["remaining_original_losses"], 8)
        self.assertTrue(all(not row["dominates_frozen_source"] for row in result))
        with self.assertRaises(ValueError):
            parse_pruning_table("\n".join(rows[:-1]))

    def test_evaluation_parent_denominator_requires_one_view_per_parent(self):
        base = "RRP-process:RRFR2-" + "b" * 24
        process = {"predictions": [{"id": base + "-step1-valid"}]}
        self.assertEqual(evaluation_parents(process), {"RRFR2-" + "b" * 24})
        process["predictions"].append({"id": base + "-step2-valid"})
        with self.assertRaisesRegex(ValueError, "multiple views"):
            evaluation_parents(process)

    def test_build_and_trained_plan_rosters_must_agree(self):
        parent = "RRFR2-" + "c" * 24
        build = {"selected_parent_ids": [parent], "selection_seed": 17}
        plan = {"row_order_per_exposure": [parent + "-step0-valid"]}
        self.assertEqual(training_parents(build, plan)[0], {parent})
        build["selected_parent_ids"] = ["RRFR2-" + "d" * 24]
        with self.assertRaisesRegex(ValueError, "roster mismatch"):
            training_parents(build, plan)

"""Counterexamples and invariances using explicit synthetic judgement metadata."""
import copy
import unittest

from experiments.rineya_quality_queue_v1.source_preservation import source_preservation_tally as tally


def source_preservation_tally(rows, outcomes):
    # Fixture declarations only; production must supply its separately frozen cells.
    cells = {tuple(row[k] for k in ("source_id", "family_id", "context_bin", "answerable"))
             for row in rows}
    declared = [dict(zip(("source_id", "family_id", "context_bin", "answerable"), cell))
                for cell in sorted(cells)]
    return tally(rows, outcomes, declared)


def view(correct=True, **changes):
    result = {"strict_correct": correct, "semantic_correct": correct, "refused": False,
              "justified_refusal": False, "unsupported_answer": False,
              "terminated": True, "truncated": False,
              "output_tokens": 8}
    result.update(changes)
    return result


def enrollment(case, source="common", answerable=True, correct=True):
    return {"case_id": case, "source_id": source, "family_id": "fixture-family", "context_bin": "short",
            "answerable": answerable, "baseline_strict_correct": correct, "baseline_semantic_correct": correct}


def pair(case, baseline=None, candidate=None, status="COMPLETE"):
    return {"case_id": case, "status": status,
            "baseline": view() if baseline is None and status == "COMPLETE" else baseline,
            "candidate": view() if candidate is None and status == "COMPLETE" else candidate}


class SourcePreservationTests(unittest.TestCase):
    def test_relative_retention_win_does_not_establish_absolute_source_preservation(self):
        rows = [enrollment(str(i)) for i in range(4)]
        magnitude = [pair(str(i), view(), view(i != 0)) for i in range(4)]
        scalar = [pair(str(i), view(), view(i == 3)) for i in range(4)]
        m = source_preservation_tally(rows, magnitude)
        s = source_preservation_tally(rows, scalar)
        self.assertGreater(m["totals"]["candidate_semantic_correct"], s["totals"]["candidate_semantic_correct"])
        self.assertEqual(m["totals"]["semantic_kept"], 3)
        self.assertEqual(m["totals"]["semantic_losses"], 1)
        self.assertEqual(m["observed_preservation"], "OBSERVED_DAMAGE")

    def test_pooled_gain_does_not_hide_rare_source_loss(self):
        enrolled = [enrollment("a", correct=False), enrollment("b", correct=False), enrollment("rare", "rare")]
        outcomes = [pair("a", view(False), view()), pair("b", view(False), view()),
                    pair("rare", view(), view(False))]
        result = source_preservation_tally(enrolled, outcomes)
        totals = result["totals"]
        self.assertEqual(totals["candidate_semantic_correct"] - totals["baseline_semantic_correct"], 1)
        self.assertEqual(result["sources"]["rare"]["fixture-family"]["short"]["answerable"]["semantic_losses"], 1)
        self.assertEqual(result["damage_case_ids"]["rare"], ["strict_correct_to_wrong", "semantic_correct_to_wrong"])
        self.assertEqual(result["observed_preservation"], "OBSERVED_DAMAGE")
        self.assertEqual(result["quality_advancement"], "HOLD")

    def test_strict_and_semantic_axes_and_new_refusal_are_independent(self):
        result = source_preservation_tally([enrollment("x")],
            [pair("x", view(), view(semantic_correct=False, refused=True))])
        self.assertEqual(result["totals"]["strict_losses"], 0)
        self.assertEqual(result["totals"]["semantic_losses"], 1)
        self.assertEqual(result["totals"]["new_answerable_refusals"], 1)
        self.assertEqual(result["totals"]["length"]["candidate"]["semantic"]["wrong"]["tokens"], 8)

    def test_unanswerable_regression_is_visible_despite_answerable_repairs(self):
        rows = [enrollment("a", correct=False), enrollment("b", correct=False), enrollment("u", answerable=False)]
        outcomes = [pair("a", view(False), view()), pair("b", view(False), view()),
                    pair("u", view(refused=True, justified_refusal=True), view(False, unsupported_answer=True))]
        result = source_preservation_tally(rows, outcomes)
        bucket = result["sources"]["common"]["fixture-family"]["short"]["unanswerable"]
        self.assertEqual(bucket["semantic_losses"], 1)
        self.assertEqual(bucket["new_unsupported_answers"], 1)
        self.assertEqual(bucket["baseline_justified_refusals"], 1)
        self.assertEqual(bucket["candidate_justified_refusals"], 0)
        self.assertEqual(bucket["justified_refusal_losses"], 1)
        self.assertIn("u", result["damage_case_ids"])

    def test_missing_and_failed_pairs_preserve_the_enrolled_denominator(self):
        rows = [enrollment(x) for x in ("ok", "missing", "failed", "uncertain", "unmeasured")]
        outcomes = [pair("ok"), pair("failed", status="RUN_FAILED"),
                    pair("uncertain", status="ADJUDICATION_UNRESOLVED"),
                    pair("unmeasured", status="NOT_MEASURED")]
        result = source_preservation_tally(rows, outcomes)
        self.assertEqual(result["totals"]["expected_pairs"], 5)
        self.assertEqual(result["totals"]["complete_pairs"], 1)
        self.assertEqual(result["totals"]["frozen_semantic_successes"], 5)
        self.assertEqual(result["totals"]["unresolved_semantic_source_successes"], 4)
        self.assertEqual(result["totals"]["semantic_kept"], 1)
        self.assertEqual(result["unresolved_case_ids"], ["failed", "missing", "uncertain", "unmeasured"])
        self.assertEqual(result["unresolved_source_success_case_ids"]["semantic"], result["unresolved_case_ids"])
        self.assertEqual(result["observed_preservation"], "UNRESOLVED_PAIRS")

    def test_termination_loss_and_lengths_are_not_rescued_by_correctness(self):
        result = source_preservation_tally([enrollment("x")],
            [pair("x", view(output_tokens=5), view(terminated=False, truncated=True, output_tokens=20))])
        self.assertEqual(result["totals"]["semantic_losses"], 0)
        self.assertEqual(result["totals"]["natural_termination_losses"], 1)
        self.assertEqual(result["totals"]["new_truncations"], 1)
        self.assertEqual(result["totals"]["length"]["candidate"]["strict"]["correct"],
                         {"count": 1, "tokens": 20, "minimum": 20, "maximum": 20})

    def test_zero_observed_damage_is_not_quality_evidence_or_authority(self):
        result = source_preservation_tally([enrollment("x")], [pair("x")])
        self.assertEqual(result["observed_preservation"], "NO_OBSERVED_DAMAGE_IN_SUPPLIED_PAIRS")
        self.assertEqual(result["quality_advancement"], "HOLD")
        for key in ("native_execution_authorized", "protected_bank_access_authorized",
                    "promotion_permitted", "method_selected_for_experiment",
                    "quality_improvement_established", "scientific_independence_verified"):
            self.assertIs(result[key], False)

    def test_required_cells_are_not_inferred_from_observed_outcomes(self):
        rows = [enrollment("x")]
        cells = [{k: rows[0][k] for k in ("source_id", "family_id", "context_bin", "answerable")},
                 {"source_id": "missing-source", "family_id": "fixture-family",
                  "context_bin": "long", "answerable": True}]
        result = tally(rows, [pair("x")], cells)
        self.assertEqual(result["observed_preservation"], "UNMEASURED_REQUIRED_CELLS")
        self.assertEqual(result["empty_required_cells"], [cells[1]])
        self.assertEqual(result["totals"]["expected_pairs"], 1)
        with self.assertRaises(ValueError):
            tally(rows, [pair("x")], cells + [cells[0]])

    def test_no_source_success_cannot_become_zero_loss_preservation(self):
        result = source_preservation_tally([enrollment("x", correct=False)], [pair("x", view(False), view())])
        self.assertEqual(result["totals"]["semantic_repairs"], 1)
        self.assertEqual(result["observed_preservation"], "UNMEASURED_SOURCE_CORRECT_CELLS")
        self.assertEqual(len(result["empty_source_correct_cells"]), 2)

    def test_unknown_baseline_membership_is_not_retroactively_filled(self):
        result = source_preservation_tally([enrollment("x", correct=None)], [pair("x")])
        self.assertEqual(result["totals"]["frozen_semantic_successes"], 0)
        self.assertEqual(result["totals"]["unknown_semantic_baseline_memberships"], 1)
        self.assertEqual(result["observed_preservation"], "UNKNOWN_BASELINE_MEMBERSHIP")
        self.assertEqual(result["unknown_baseline_membership_case_ids"]["semantic"], ["x"])
        with self.assertRaises(ValueError):
            source_preservation_tally([enrollment("x")], [pair("x", view(False), view())])

    def test_reordering_is_invariant_and_inputs_are_unchanged(self):
        rows = [enrollment("b"), enrollment("a", "rare", False)]
        outcomes = [pair("a", view(refused=True), view(False, unsupported_answer=True)), pair("b")]
        originals = copy.deepcopy((rows, outcomes))
        result = source_preservation_tally(rows, outcomes)
        self.assertEqual(result, source_preservation_tally(list(reversed(rows)), list(reversed(outcomes))))
        self.assertEqual((rows, outcomes), originals)

    def test_context_damage_is_visible_within_one_source_and_family(self):
        rows = [enrollment("short", correct=False), enrollment("long")]
        rows[1]["context_bin"] = "long"
        result = source_preservation_tally(rows,
            [pair("short", view(False), view()), pair("long", view(), view(False))])
        family = result["sources"]["common"]["fixture-family"]
        self.assertEqual(family["short"]["answerable"]["semantic_repairs"], 1)
        self.assertEqual(family["long"]["answerable"]["semantic_losses"], 1)
        self.assertEqual(result["totals"]["candidate_semantic_correct"], result["totals"]["baseline_semantic_correct"])
        self.assertIn("long", result["damage_case_ids"])

    def test_duplicate_unmatched_malformed_and_partial_judgements_are_refused(self):
        bad_pairs = [[pair("x"), pair("x")], [pair("other")],
                     [pair("x", view(output_tokens=True))],
                     [pair("x", view(terminated=True, truncated=True))],
                     [pair("x", view(justified_refusal=True))],
                     [pair("x", view(), None, "RUN_FAILED")]]
        for outcomes in bad_pairs:
            with self.subTest(outcomes=outcomes), self.assertRaises(ValueError):
                source_preservation_tally([enrollment("x")], outcomes)
        for rows in ([], [enrollment("x"), enrollment("x")], [enrollment("x", answerable=1)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                source_preservation_tally(rows, [])

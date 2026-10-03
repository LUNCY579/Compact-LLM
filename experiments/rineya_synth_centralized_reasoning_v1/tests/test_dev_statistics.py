"""Stdlib-only arithmetic tests over synthetic binary fixtures, not model results."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import unittest

LOCATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LOCATION))
import dev_statistics as stats


def panel():
    rows = []
    for skill in ("R", "S", "RS"):
        for index in range(32):
            component = "panel_%s_%02d" % (skill, index)
            for condition in (0, 1):
                for rendering in ("familiar", "heldout_ledger"):
                    rows.append({"row_id": component + "_%d_" % condition + rendering,
                                 "component_id": component, "skill": skill,
                                 "condition": condition, "rendering": rendering})
    return rows


def guards(count=96, *, bindings=True, extra_wrong=False):
    domains, m0 = [], {}
    scorer = {}
    for kind in ("task", "language", "long_context"):
        name = "guard_" + kind
        ids = [name + "_r%03d" % i for i in range(count)]
        components = [name + "_c%03d" % i for i in range(count)]
        m0.update({row: True for row in ids})
        if extra_wrong and kind == "task":
            ids += ["guard_task_wrong_same_parent", "guard_task_wrong_new_parent"]
            components += [components[0], "guard_task_never_correct_parent"]
            m0.update({row: False for row in ids[-2:]})
        domains.append({"domain_id": name, "class": kind, "row_ids": ids, "component_ids": components})
        if bindings:
            scorer[name] = {"scorer_sha256": "a" * 64, "labels_sha256": "b" * 64,
                            "qualification_receipt_sha256": "c" * 64, "qualified": True}
    enrollment = stats.freeze_guard_enrollment(domains, m0, baseline_id="pinned_M0_fixture",
                                               scorer_bindings=scorer)
    candidates = {arm: dict(m0) for arm in ("M1", "M2")}
    return enrollment, candidates


def positive_scores(rows):
    return {"M0": {row["row_id"]: True for row in rows},
            "M1": {row["row_id"]: False for row in rows},
            "M2": {row["row_id"]: True for row in rows}}


def analyze(rows, scores, enrollment, candidates):
    return stats.analyze_dev_screen(rows, scores, enrollment, candidates, training_seed=101,
                                    expected_enrollment_sha256=enrollment["sha256"],
                                    expected_panel_metadata_sha256=stats._digest(sorted(rows,key=lambda row:row["row_id"])))


class ExactBoundTests(unittest.TestCase):
    def test_zero_failure_formula_matches_exact_inversion(self):
        for n in (1, 10, 94, 100, 100000):
            for alpha in (.05, .025, .05 / 6):
                with self.subTest(n=n, alpha=alpha):
                    value = stats.one_sided_binomial_upper(0, n, alpha)
                    self.assertAlmostEqual(1 - alpha ** (1 / n), value, places=14)
                    self.assertAlmostEqual(alpha, (1 - value) ** n, delta=2e-11)
        self.assertAlmostEqual(.2588655508930523, stats.one_sided_binomial_upper(0, 10, .05), places=14)

    def test_known_nonzero_clopper_pearson_bounds_use_binomial_tail(self):
        for failures, expected in ((1, .3941633024365047), (2, .5069013010632024)):
            value = stats.one_sided_binomial_upper(failures, 10, .05)
            self.assertAlmostEqual(expected, value, places=12)
            # Independent small-n polynomial, with no beta quantile or module CDF.
            probability = sum(math.comb(10, k) * value ** k * (1 - value) ** (10 - k)
                              for k in range(failures + 1))
            self.assertAlmostEqual(.05, probability, places=13)
        self.assertEqual(1, stats.one_sided_binomial_upper(10, 10, .05))

    def test_more_domains_widen_bounds_and_zero_sample_is_not_a_pass(self):
        self.assertGreater(stats.one_sided_binomial_upper(0, 100, .05 / 12),
                           stats.one_sided_binomial_upper(0, 100, .05 / 6))
        self.assertGreater(stats.one_sided_binomial_upper(0, 93, .05 / 6), .05)
        self.assertLess(stats.one_sided_binomial_upper(0, 94, .05 / 6), .05)
        for arguments in ((0, 0, .05), (True, 10, .05), (11, 10, .05), (0, 10, 0), (0, 10, True)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                stats.one_sided_binomial_upper(*arguments)


class AggregationAndBootstrapTests(unittest.TestCase):
    def test_four_rows_are_averaged_once_per_component(self):
        rows = panel()[:8]
        scores = {row["row_id"]: index in (0, 1, 4) for index, row in enumerate(rows)}
        result = stats.component_accuracies(rows, scores)
        self.assertEqual({"panel_R_00": {"skill": "R", "accuracy": .5},
                          "panel_R_01": {"skill": "R", "accuracy": .25}}, result)
        with self.assertRaisesRegex(ValueError, "four complete"):
            stats.component_accuracies(rows[:-1], {row["row_id"]: True for row in rows[:-1]})

    def test_paired_resampling_identical_arms_is_exact_zero_despite_heterogeneity(self):
        rows = panel()
        binary = {row["row_id"]: index % 5 == 0 for index, row in enumerate(rows)}
        component = stats.component_accuracies(rows, binary)
        previous = random.getstate()
        result = stats.paired_cluster_bootstrap(component, component)
        self.assertEqual(previous, random.getstate())
        self.assertEqual({"M2_minus_M1": 0, "CI95": [0, 0]}, result["atomic_R_S"])
        self.assertEqual({"M2_minus_M1": 0, "CI95": [0, 0]}, result["RS"])
        self.assertEqual(10000, result["resamples"])
        self.assertEqual(.95, result["confidence"])

    def test_skill_macro_endpoint_and_RS_are_separate_and_seed_repeatable(self):
        rows = panel()
        first = {row["row_id"]: row["skill"] == "S" for row in rows}
        second = {row["row_id"]: row["skill"] in ("R", "RS") for row in rows}
        a, b = stats.component_accuracies(rows, first), stats.component_accuracies(rows, second)
        result = stats.paired_cluster_bootstrap(a, b)
        self.assertEqual(result, stats.paired_cluster_bootstrap(a, b))
        self.assertEqual({"M2_minus_M1": 0, "CI95": [0, 0]}, result["atomic_R_S"])
        self.assertEqual({"M2_minus_M1": 1, "CI95": [1, 1]}, result["RS"])

    def test_fewer_components_are_not_replaced_by_more_variant_rows(self):
        rows = panel()
        scores = stats.component_accuracies(rows, {row["row_id"]: True for row in rows})
        reduced = dict(scores)
        reduced.pop(next(iter(reduced)))
        with self.assertRaisesRegex(ValueError, "quotas"):
            stats.paired_cluster_bootstrap(reduced, reduced)


class DevelopmentDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = panel()
        cls.scores = positive_scores(cls.rows)

    def test_positive_screen_requires_all_endpoints_and_retention_and_stays_DEV_only(self):
        enrolled, candidates = guards()
        result = analyze(self.rows, self.scores, enrolled, candidates)
        self.assertEqual("ADVANCE_DEV_SCREEN", result["status"])
        self.assertTrue(result["advancement_requirements_met"])
        self.assertEqual(3, result["declared_domain_count"])
        self.assertEqual(6, len(result["retention"]))
        self.assertTrue(all(record["per_comparison_alpha"] == .05 / 6 for record in result["retention"]))
        self.assertTrue(result["single_training_seed_cannot_confirm_transfer_or_broad_retention"])
        self.assertFalse(result["scorer_qualification_independently_verified_by_this_module"])
        for name in ("native_launch_authorized", "training_authorized", "protected_bank_access",
                     "confirmation_eligible", "broad_retention_certified", "transfer_confirmation_established"):
            self.assertFalse(result[name])

    def test_zero_losses_with_small_n_are_inconclusive_not_advanced(self):
        enrolled, candidates = guards(count=93)
        result = analyze(self.rows, self.scores, enrolled, candidates)
        self.assertEqual("INCONCLUSIVE", result["status"])
        self.assertFalse(result["advancement_requirements_met"])
        self.assertTrue(all(record["observed_component_failures"] == 0 for record in result["retention"]))
        self.assertTrue(any(reason.startswith("INSUFFICIENT_RETENTION_PRECISION") for reason in result["issues"]))

    def test_prior_correct_losses_are_parent_failures_and_repairs_cannot_offset(self):
        enrolled, candidates = guards(count=500, extra_wrong=True)
        candidates["M2"]["guard_task_r000"] = False
        candidates["M2"]["guard_task_wrong_same_parent"] = True
        candidates["M2"]["guard_task_wrong_new_parent"] = True
        result = analyze(self.rows, self.scores, enrolled, candidates)
        task = next(record for record in result["retention"] if record["domain_id"] == "guard_task" and record["arm"] == "M2")
        self.assertEqual(500, task["eligible_components"])
        self.assertEqual(1, task["observed_component_failures"])
        self.assertEqual(["guard_task_c000"], task["lost_components"])
        self.assertEqual(["guard_task_r000"], task["lost_prior_correct_rows"])
        self.assertEqual(2, len(task["repaired_previously_incorrect_rows"]))
        self.assertLess(task["exact_one_sided_upper_failure_rate"], .05)
        self.assertFalse(task["passed"])  # The zero-observed-loss condition is independent of the upper bound.
        self.assertEqual("DO_NOT_ADVANCE", result["status"])

    def test_loss_of_any_correct_variant_counts_component_once(self):
        enrolled, candidates = guards()
        domains = copy.deepcopy(enrolled["declared_domains"])
        domains[0]["row_ids"] += ["guard_language_r000_second"]
        domains[0]["component_ids"] += [domains[0]["component_ids"][0]]
        m0 = dict(enrolled["m0_scores"], guard_language_r000_second=True)
        enrolled = stats.freeze_guard_enrollment(domains, m0, baseline_id=enrolled["baseline_id"],
                                                 scorer_bindings=enrolled["scorer_bindings"])
        candidates = {arm: dict(m0) for arm in ("M1", "M2")}
        candidates["M2"]["guard_language_r000"] = False
        candidates["M2"]["guard_language_r000_second"] = False
        result = analyze(self.rows, self.scores, enrolled, candidates)
        loss = next(record for record in result["retention"] if record["domain_id"] == "guard_language" and record["arm"] == "M2")
        self.assertEqual(1, loss["observed_component_failures"])
        self.assertEqual(2, len(loss["lost_prior_correct_rows"]))
        self.assertEqual(96, loss["eligible_components"])

    def test_binary_scores_do_not_qualify_missing_or_legacy_language_grader(self):
        enrolled, candidates = guards(bindings=False)
        result = analyze(self.rows, self.scores, enrolled, candidates)
        self.assertEqual("INCONCLUSIVE", result["status"])
        self.assertTrue(result["computational_screen_only"])
        self.assertFalse(result["advancement_requirements_met"])
        enrolled, candidates = guards()
        bindings = copy.deepcopy(enrolled["scorer_bindings"])
        bindings["guard_language"]["qualified"] = False
        enrolled = stats.freeze_guard_enrollment(enrolled["declared_domains"], enrolled["m0_scores"],
            baseline_id=enrolled["baseline_id"], scorer_bindings=bindings)
        result = analyze(self.rows, self.scores, enrolled, candidates)
        self.assertEqual("INCONCLUSIVE", result["status"])
        self.assertTrue(result["computational_screen_only"])

    def test_missing_classes_unknown_class_and_zero_eligible_remain_inconclusive(self):
        enrolled, candidates = guards()
        for remove_class in ("language", "long_context"):
            domains = [domain for domain in enrolled["declared_domains"] if domain["class"] != remove_class]
            ids = {row for domain in domains for row in domain["row_ids"]}
            m0 = {row: value for row, value in enrolled["m0_scores"].items() if row in ids}
            names = {domain["domain_id"] for domain in domains}
            binding = {name: value for name, value in enrolled["scorer_bindings"].items() if name in names}
            frozen = stats.freeze_guard_enrollment(domains, m0, baseline_id="M0", scorer_bindings=binding)
            result = analyze(self.rows, self.scores, frozen, {arm: dict(m0) for arm in ("M1", "M2")})
            self.assertEqual("INCONCLUSIVE", result["status"])
            self.assertFalse(result["advancement_requirements_met"])
        domains = copy.deepcopy(enrolled["declared_domains"])
        domains[0]["class"] = "unknown_class"
        unknown = stats.freeze_guard_enrollment(domains, enrolled["m0_scores"], baseline_id="M0",
                                               scorer_bindings=enrolled["scorer_bindings"])
        self.assertEqual("INCONCLUSIVE", analyze(self.rows, self.scores, unknown, candidates)["status"])
        scores = {row: False for row in enrolled["m0_scores"]}
        empty = stats.freeze_guard_enrollment(enrolled["declared_domains"], scores, baseline_id="M0",
                                             scorer_bindings=enrolled["scorer_bindings"])
        result = analyze(self.rows, self.scores, empty, candidates)
        self.assertEqual("INCONCLUSIVE", result["status"])
        self.assertTrue(all(record["exact_one_sided_upper_failure_rate"] is None for record in result["retention"]))

    def test_missing_outputs_do_not_reduce_denominators_or_fake_zero_loss(self):
        enrolled, candidates = guards()
        scores = copy.deepcopy(self.scores)
        scores["M2"].pop(next(iter(scores["M2"])))
        result = analyze(self.rows, scores, enrolled, candidates)
        self.assertEqual("INCONCLUSIVE", result["status"])
        self.assertIsNone(result["endpoints"])
        candidates["M2"].pop("guard_task_r000")
        result = analyze(self.rows, self.scores, enrolled, candidates)
        task = next(record for record in result["retention"] if record["domain_id"] == "guard_task" and record["arm"] == "M2")
        self.assertEqual(96, task["eligible_components"])
        self.assertIsNone(task["observed_component_failures"])
        self.assertIsNone(task["exact_one_sided_upper_failure_rate"])
        self.assertEqual("INCONCLUSIVE", result["status"])
        result = analyze(self.rows, {"M0": self.scores["M0"]}, enrolled, {})
        self.assertEqual("INCONCLUSIVE", result["status"])

    def test_atomic_and_RS_gates_cannot_substitute_for_each_other(self):
        enrolled, candidates = guards()
        scores = copy.deepcopy(self.scores)
        for row in self.rows:
            if row["skill"] == "RS": scores["M2"][row["row_id"]] = False
        result = analyze(self.rows, scores, enrolled, candidates)
        self.assertTrue(result["atomic_pass"])
        self.assertFalse(result["RS_pass"])
        self.assertEqual("DO_NOT_ADVANCE", result["status"])
        for row in self.rows:
            scores["M2"][row["row_id"]] = row["skill"] == "RS"
        result = analyze(self.rows, scores, enrolled, candidates)
        self.assertFalse(result["atomic_pass"])
        self.assertTrue(result["RS_pass"])
        self.assertEqual("DO_NOT_ADVANCE", result["status"])

    def test_positive_atomic_CI_below_five_percentage_points_does_not_advance(self):
        enrolled, candidates = guards()
        scores = copy.deepcopy(self.scores)
        for row in self.rows:
            component_index = int(row["component_id"].split("_")[-1])
            scores["M2"][row["row_id"]] = row["skill"] == "RS" or (
                component_index < 2 and row["condition"] == 0 and row["rendering"] == "familiar")
        result = analyze(self.rows, scores, enrolled, candidates)
        endpoint = result["endpoints"]["atomic_R_S"]
        self.assertGreater(endpoint["CI95"][0], 0)
        self.assertEqual(.015625, endpoint["M2_minus_M1"])
        self.assertFalse(result["atomic_pass"])
        self.assertTrue(result["RS_pass"])
        self.assertTrue(result["retention_pass"])
        self.assertEqual("DO_NOT_ADVANCE", result["status"])


class IntegrityTests(unittest.TestCase):
    def test_guard_membership_is_frozen_and_rehashed_changes_are_rejected(self):
        enrolled, candidates = guards()
        for mutation in (lambda value:value["domains"][0]["eligible_components"].clear(),
                         lambda value:value.update(candidate_outputs_used=True),
                         lambda value:value.update(baseline_id="changed")):
            forged = copy.deepcopy(enrolled)
            mutation(forged)
            if forged["baseline_id"] != "changed":
                forged["sha256"] = stats._digest({key:value for key,value in forged.items() if key != "sha256"})
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                analyze(panel(), positive_scores(panel()), forged, candidates)

    def test_no_guard_rows_or_components_can_leak_into_acquisition_population(self):
        rows = panel()
        enrolled, candidates = guards()
        for field, value in (("row_ids", rows[0]["row_id"]), ("component_ids", rows[0]["component_id"])):
            domains = copy.deepcopy(enrolled["declared_domains"])
            old = domains[0][field][0]
            domains[0][field][0] = value
            m0 = dict(enrolled["m0_scores"])
            if field == "row_ids": m0[value] = m0.pop(old)
            frozen = stats.freeze_guard_enrollment(domains, m0, baseline_id="M0",
                                                   scorer_bindings=enrolled["scorer_bindings"])
            with self.assertRaisesRegex(ValueError, "overlaps"):
                analyze(rows, positive_scores(rows), frozen, {arm:dict(m0) for arm in ("M1", "M2")})

    def test_boolean_scores_only_unknown_ids_and_unregistered_seed_are_rejected(self):
        rows = panel()
        enrolled, candidates = guards()
        for altered in (dict(positive_scores(rows), M3={}), positive_scores(rows)):
            if "M3" not in altered: altered["M2"][rows[0]["row_id"]] = 1
            with self.assertRaises(ValueError): analyze(rows, altered, enrolled, candidates)
        altered = positive_scores(rows)
        altered["M2"]["undeclared_row"] = True
        with self.assertRaisesRegex(ValueError, "undeclared score"):
            analyze(rows, altered, enrolled, candidates)
        with self.assertRaisesRegex(ValueError, "seed 101"):
            stats.analyze_dev_screen(rows, positive_scores(rows), enrolled, candidates, training_seed=102,
                                    expected_enrollment_sha256=enrolled["sha256"],
                                    expected_panel_metadata_sha256=stats.freeze_panel_metadata(rows)["panel_metadata_sha256"])

    def test_post_candidate_cohort_filter_cannot_replace_pre_frozen_membership(self):
        enrolled, candidates = guards(count=500)
        expected = enrolled["sha256"]
        candidates["M2"]["guard_task_r000"] = False
        domains = copy.deepcopy(enrolled["declared_domains"])
        task = next(domain for domain in domains if domain["domain_id"] == "guard_task")
        index = task["row_ids"].index("guard_task_r000")
        del task["row_ids"][index]
        del task["component_ids"][index]
        scores = dict(enrolled["m0_scores"])
        scores.pop("guard_task_r000")
        narrowed = stats.freeze_guard_enrollment(domains, scores, baseline_id=enrolled["baseline_id"],
                                                scorer_bindings=enrolled["scorer_bindings"])
        trimmed = {arm:{row:value for row,value in values.items() if row in scores}
                   for arm,values in candidates.items()}
        with self.assertRaisesRegex(ValueError, "pre-candidate frozen"):
            stats.analyze_dev_screen(panel(), positive_scores(panel()), narrowed, trimmed, training_seed=101,
                                    expected_enrollment_sha256=expected,
                                    expected_panel_metadata_sha256=stats.freeze_panel_metadata(panel())["panel_metadata_sha256"])

    def test_post_score_regrouping_cannot_change_cluster_bootstrap_units(self):
        rows = panel()
        frozen = stats.freeze_panel_metadata(rows)
        changed = copy.deepcopy(rows)
        # Swap the same condition/rendering row between two parents. Four-row
        # completeness and skill counts remain valid, but cluster units change.
        changed[0]["component_id"], changed[4]["component_id"] = changed[4]["component_id"], changed[0]["component_id"]
        enrolled, candidates = guards()
        with self.assertRaisesRegex(ValueError, "pre-score frozen"):
            stats.analyze_dev_screen(changed, positive_scores(changed), enrolled, candidates, training_seed=101,
                expected_enrollment_sha256=enrolled["sha256"],
                expected_panel_metadata_sha256=frozen["panel_metadata_sha256"])
        self.assertFalse(frozen["student_scores_used"])

    def test_scorer_bindings_require_explicit_label_and_receipt_hashes(self):
        enrolled, _ = guards()
        bad = copy.deepcopy(enrolled["scorer_bindings"])
        del bad["guard_task"]["labels_sha256"]
        with self.assertRaisesRegex(ValueError, "hash metadata"):
            stats.freeze_guard_enrollment(enrolled["declared_domains"], enrolled["m0_scores"], baseline_id="M0",
                                          scorer_bindings=bad)

    def test_module_import_is_framework_free_and_policy_is_closed(self):
        code = ("import sys;sys.path.insert(0,sys.argv[1]);import dev_statistics as s;"
                "assert 'torch' not in sys.modules and 'transformers' not in sys.modules;"
                "p=s.policy();assert p['native_launch_authorized'] is False;print('CPU_ONLY')")
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(LOCATION)],
                                text=True, capture_output=True, check=True, timeout=30)
        self.assertEqual("CPU_ONLY", result.stdout.strip())


if __name__ == "__main__":
    unittest.main()

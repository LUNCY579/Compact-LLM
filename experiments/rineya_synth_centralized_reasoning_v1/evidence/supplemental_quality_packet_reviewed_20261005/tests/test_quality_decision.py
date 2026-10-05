"""OFFLINE/MOCKED receipts and outcomes: these tests are not model evidence."""

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import unittest

from quality_decision import POLICY_ID, evaluate_quality


def outcome(correct=False, **extra):
    return {"correct": correct, "format_correct": True, "abstained": False,
            "capped": False, **extra}


def receipt(**extra):
    return {"status": "COMPLETE", "completed_updates": 27,
            "consumed_sequences": 540, "final_step": 27, "failed": False, **extra}


def fixture(size=100, long_context=True):
    """Exact three-point boundary using only deterministic synthetic fixtures."""
    rows, baseline = [], {}
    for role in ("R", "S", "RS"):
        for index in range(size):
            key = f"{role}-{index}"
            rows.append({"key": key, "parent_id": key, "source": f"atomic-{role}",
                         "role": role, "stratum": "fixture", "rendering": "canonical",
                         "prompt_hash": hashlib.sha256(key.encode()).hexdigest()})
            baseline[key] = outcome(index < size // 2)
    for source in ("arithmetic", "retrieval", "abstention", "dialogue") + (("long_context",) if long_context else ()):
        for index in range(2):
            key = f"guard-{source}-{index}"
            rows.append({"key": key, "parent_id": key, "source": source,
                         "role": "retention", "stratum": "retention-fixture",
                         "rendering": "canonical", "prompt_hash": hashlib.sha256(key.encode()).hexdigest()})
            baseline[key] = outcome(index == 0, abstained=source == "abstention")
    treatment = deepcopy(baseline)
    for role in ("R", "S", "RS"):
        for index in range(size // 2, min(size, size // 2 + 3)):
            treatment[f"{role}-{index}"]["correct"] = True
    return baseline, {arm: {"rows": deepcopy(treatment), "completion": receipt()}
                      for arm in ("M1", "M2")}, {"rows": rows}, {
        "policy_id": POLICY_ID, "long_context_qualified": long_context,
        "bootstrap_samples": 20, "bootstrap_seed": 101}


def exact(value):
    return Fraction(value["numerator"], value["denominator"])


class QualityDecisionTests(unittest.TestCase):
    def setUp(self):
        self.baseline, self.candidates, self.cohort, self.policy = fixture()

    def evaluate(self):
        return evaluate_quality(self.baseline, self.candidates, self.cohort, self.policy)

    def test_exact_boundary_and_m1_tie_preference(self):
        decision = self.evaluate()
        self.assertEqual(decision["selected_arm"], "M1")
        self.assertFalse(decision["M2_advantage_claim"])
        for arm in ("M1", "M2"):
            self.assertEqual(exact(decision["candidates"][arm]["skills"]["macro_RS_atomic_gain"]), Fraction(3, 100))
            self.assertEqual(exact(decision["candidates"][arm]["skills"]["RS"]["gain"]), Fraction(3, 100))
            self.assertTrue(decision["candidates"][arm]["eligible"])
        self.assertTrue(decision["decision_is_engineering_screen"])
        self.assertEqual(decision["evidence_kind"], "mechanical_fixture")
        self.assertFalse(decision["native_execution_permitted"])
        self.assertFalse(decision["quality_claim_permitted"])
        self.assertFalse(decision["promotion_permitted"])

    def test_below_boundary_fails(self):
        for arm in ("M1", "M2"):
            self.candidates[arm]["rows"]["R-52"]["correct"] = False
        decision = self.evaluate()
        self.assertIsNone(decision["selected_arm"])
        self.assertFalse(decision["candidates"]["M1"]["checks"]["macro_R_S_gain_at_least_3pp"])

    def test_neither_atomic_skill_may_decline(self):
        rows = self.candidates["M1"]["rows"]
        for index in range(50, 53):
            rows[f"R-{index}"]["correct"] = False
        rows["R-0"]["correct"] = False
        for index in range(53, 57):
            rows[f"S-{index}"]["correct"] = True
        decision = self.evaluate()
        arm = decision["candidates"]["M1"]
        self.assertEqual(exact(arm["skills"]["macro_RS_atomic_gain"]), Fraction(3, 100))
        self.assertFalse(arm["checks"]["R_does_not_decline"])
        self.assertFalse(arm["eligible"])
        self.assertEqual(decision["selected_arm"], "M2")

    def test_repairs_cannot_cancel_damage(self):
        rows = self.candidates["M1"]["rows"]
        rows["guard-arithmetic-0"]["correct"] = False
        rows["guard-arithmetic-1"]["correct"] = True
        decision = self.evaluate()
        source = decision["candidates"]["M1"]["retention"]["arithmetic"]
        self.assertEqual((source["retained"], source["damaged"], source["rescued"]), (0, 1, 1))
        self.assertFalse(decision["candidates"]["M1"]["eligible"])
        self.assertEqual(decision["candidates"]["M1"]["follow_up"]["intervention"], "PRESERVATION_REPLAY_SUCCESSOR")

    def test_missing_source_correct_support_is_undefined(self):
        self.baseline["guard-dialogue-0"]["correct"] = False
        for arm in ("M1", "M2"):
            self.candidates[arm]["rows"]["guard-dialogue-0"]["correct"] = False
        decision = self.evaluate()
        for arm in ("M1", "M2"):
            self.assertEqual(decision["candidates"][arm]["retention"]["dialogue"]["status"], "UNDEFINED")
            self.assertFalse(decision["candidates"][arm]["eligible"])
        self.assertIsNone(decision["selected_arm"])

    def test_missing_entire_retention_source_is_undefined(self):
        keys = {"guard-retrieval-0", "guard-retrieval-1"}
        self.cohort["rows"] = [row for row in self.cohort["rows"] if row["key"] not in keys]
        for key in keys:
            del self.baseline[key]
            for candidate in self.candidates.values():
                del candidate["rows"][key]
        decision = self.evaluate()
        self.assertEqual(decision["candidates"]["M1"]["retention"]["retrieval"]["status"], "UNDEFINED")
        self.assertIsNone(decision["selected_arm"])

    def test_long_context_absence_blocks_broad_claim(self):
        baseline, candidates, cohort, policy = fixture(long_context=False)
        policy["long_context_qualified"] = True  # A flag cannot invent enrollment.
        decision = evaluate_quality(baseline, candidates, cohort, policy)
        self.assertEqual(decision["selected_arm"], "M1")
        self.assertEqual(decision["broad_claim_status"], "BLOCKED")
        self.assertEqual(decision["candidates"]["M1"]["long_context"]["guard_status"], "UNDEFINED")

    def test_unqualified_long_context_blocks_broad_claim(self):
        self.policy["long_context_qualified"] = False
        self.assertEqual(self.evaluate()["broad_claim_status"], "BLOCKED")

    def test_enrolled_long_context_damage_blocks_scoped_screen(self):
        self.candidates["M1"]["rows"]["guard-long_context-0"]["correct"] = False
        self.assertFalse(self.evaluate()["candidates"]["M1"]["eligible"])

    def test_m2_advantage_requires_exact_three_points(self):
        rows = self.candidates["M2"]["rows"]
        for role in ("R", "S"):
            for index in range(53, 56):
                rows[f"{role}-{index}"]["correct"] = True
        decision = self.evaluate()
        self.assertEqual(exact(decision["M2_minus_M1_macro_R_S"]), Fraction(3, 100))
        self.assertTrue(decision["M2_advantage_claim"])
        self.assertTrue(decision["M2_engineering_advantage_signal"])
        self.assertEqual(decision["M2_advantage_signal_scope"],
                         "MECHANICAL_ENGINEERING_FIXTURE_ONLY_NO_MODEL_QUALITY_CLAIM")
        self.assertFalse(decision["quality_claim_permitted"])
        self.assertEqual(decision["selected_arm"], "M2")

    def test_ineligible_m2_cannot_claim_advantage(self):
        rows = self.candidates["M2"]["rows"]
        for role in ("R", "S"):
            for index in range(53, 56):
                rows[f"{role}-{index}"]["correct"] = True
        rows["guard-abstention-0"]["correct"] = False
        decision = self.evaluate()
        self.assertFalse(decision["M2_advantage_claim"])
        self.assertEqual(decision["selected_arm"], "M1")

    def test_complete_both_arms_before_selection(self):
        del self.candidates["M2"]
        decision = self.evaluate()
        self.assertFalse(decision["all_prespecified_arms_complete"])
        self.assertIsNone(decision["selected_arm"])
        self.assertEqual(decision["candidates"]["M2"]["status"], "BLOCKED")

    def test_missing_receipt_blocks_selection(self):
        del self.candidates["M2"]["completion"]
        self.assertIsNone(self.evaluate()["selected_arm"])

    def test_failed_interrupted_or_wrong_final_runs_block_selection(self):
        for change in ({"failed": True}, {"status": "INTERRUPTED"}, {"completed_updates": 26},
                       {"completed_updates": 28}, {"consumed_sequences": 539}, {"final_step": 9},
                       {"evaluated_step": 9}, {"completed_updates": True}):
            with self.subTest(change=change):
                self.candidates["M2"]["completion"] = receipt(**change)
                self.assertIsNone(self.evaluate()["selected_arm"])

    def test_missing_or_extra_outcome_refused(self):
        for kind in ("baseline", "missing", "extra"):
            with self.subTest(kind=kind):
                baseline, candidates, cohort, policy = fixture()
                if kind == "baseline":
                    del baseline["R-0"]
                elif kind == "missing":
                    del candidates["M1"]["rows"]["S-0"]
                else:
                    candidates["M2"]["rows"]["unfrozen"] = outcome()
                with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                    evaluate_quality(baseline, candidates, cohort, policy)

    def test_duplicate_ids_prompt_hash_or_rendering_refused(self):
        for field in ("key", "prompt_hash", "rendering"):
            with self.subTest(field=field):
                baseline, candidates, cohort, policy = fixture()
                if field == "rendering":
                    cohort["rows"][1]["parent_id"] = cohort["rows"][0]["parent_id"]
                else:
                    cohort["rows"][1][field] = cohort["rows"][0][field]
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    evaluate_quality(baseline, candidates, cohort, policy)

    def test_parent_metadata_drift_refused(self):
        self.cohort["rows"][1]["parent_id"] = self.cohort["rows"][0]["parent_id"]
        self.cohort["rows"][1]["rendering"] = "alternate"
        self.cohort["rows"][1]["source"] = "unexpected"
        with self.assertRaisesRegex(ValueError, "parent identity drift"):
            self.evaluate()

    def test_unknown_transition_variants_can_keep_parent_grouping(self):
        self.cohort["rows"][1]["parent_id"] = self.cohort["rows"][0]["parent_id"]
        self.cohort["rows"][1]["rendering"] = "missing-fact"
        self.cohort["rows"][1]["stratum"] = "UNKNOWN-transition"
        decision = self.evaluate()
        self.assertEqual(decision["baseline"]["skills"]["R"]["parent_count"], 99)
        self.assertIn("UNKNOWN-transition", decision["baseline"]["behavior"]["by_stratum"])

    def test_baseline_optional_checkpoint_metadata_uses_unchanged_source(self):
        self.baseline["R-0"]["checkpoint_step"] = 0
        self.assertEqual(self.evaluate()["selected_arm"], "M1")
        self.baseline["R-0"]["checkpoint_step"] = 27
        with self.assertRaisesRegex(ValueError, "unchanged step-0"):
            self.evaluate()

    def test_optional_bound_outcome_identity_drift_refused(self):
        self.candidates["M1"]["rows"]["R-0"]["prompt_hash"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "identity drift"):
            self.evaluate()

    def test_step9_outcomes_refused_even_with_final_receipt(self):
        self.candidates["M1"]["rows"]["R-0"]["checkpoint_step"] = 9
        with self.assertRaisesRegex(ValueError, "final step-27"):
            self.evaluate()

    def test_boolean_flags_are_strict(self):
        for field in ("correct", "format_correct", "abstained", "capped"):
            with self.subTest(field=field):
                self.candidates["M1"]["rows"]["R-0"][field] = 1
                with self.assertRaisesRegex(ValueError, "boolean"):
                    self.evaluate()
                self.candidates["M1"]["rows"]["R-0"][field] = True if field in ("correct", "format_correct") else False
        self.policy["long_context_qualified"] = 1
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.evaluate()

    def test_frozen_thresholds_and_guard_membership_cannot_weaken(self):
        for value in (0, float("nan"), float("inf"), "-1", "1/50"):
            with self.subTest(value=value):
                self.policy["macro_gain_threshold"] = value
                with self.assertRaisesRegex(ValueError, "3/100"):
                    self.evaluate()
        del self.policy["macro_gain_threshold"]
        self.policy["required_retention_sources"] = ["arithmetic"]
        with self.assertRaisesRegex(ValueError, "four-source"):
            self.evaluate()

    def test_real_root_policy_schema_and_denominator_freeze(self):
        policy_path = Path(__file__).resolve().parents[1] / "QUALITY_SCREEN_POLICY_001.json"
        real_policy = json.loads(policy_path.read_text(encoding="utf-8"))
        baseline, candidates, cohort, _ = fixture(size=32)
        dev_rows = [row for row in cohort["rows"] if row["role"] != "retention"]
        cohort["rows"] = [row for row in cohort["rows"] if row["role"] == "retention"]
        for original in dev_rows:
            for condition, rendering in ((0, "familiar"), (0, "heldout_ledger"),
                                          (1, "familiar"), (1, "heldout_ledger")):
                key = f"{original['key']}-c{condition}-{rendering}"
                row = {**original, "key": key, "condition": condition, "rendering": rendering,
                       "prompt_hash": hashlib.sha256(key.encode()).hexdigest()}
                cohort["rows"].append(row)
                baseline[key] = deepcopy(baseline[original["key"]])
                for candidate in candidates.values():
                    candidate["rows"][key] = deepcopy(candidate["rows"][original["key"]])
            del baseline[original["key"]]
            for candidate in candidates.values():
                del candidate["rows"][original["key"]]
        decision = evaluate_quality(baseline, candidates, cohort, real_policy)
        self.assertEqual(decision["policy_id"], real_policy["policy_id"])
        self.assertEqual(decision["selected_arm"], "M1")
        self.assertFalse(decision["native_execution_permitted"])
        self.assertEqual(decision["candidates"]["M1"]["skills"]["R"]["accuracy_interval"]["seed"], 20261005)
        mutated_policy = {**real_policy, "expected_dev_rows": 383}
        with self.assertRaisesRegex(ValueError, "expected_dev_rows"):
            evaluate_quality(baseline, candidates, cohort, mutated_policy)

    def test_real_policy_substantive_fields_cannot_weaken(self):
        weakening = {"minimum_macro_atomic_gain": 0, "minimum_rs_transfer_gain": "nan",
                     "minimum_m2_advantage": "inf", "minimum_gain_fraction": [0, 100],
                     "atomic_skill_weights": {"R": [1, 1], "S": [0, 1]},
                     "no_atomic_skill_decline": False, "maximum_damaged_prior_correct_cases": 1,
                     "repairs_can_cancel_damage": True, "both_candidates_must_complete": False,
                     "candidate_order": ["M2", "M1"], "checkpoint_selection": "best_intermediate",
                     "native_scoring_authorized": True, "training_authorized": 0,
                     "promotion_authorized": True}
        for name, value in weakening.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    evaluate_quality(self.baseline, self.candidates, self.cohort, {**self.policy, name: value})

    def test_condition_rendering_variants_share_parent_without_collision(self):
        # Mirrors compiler identity even in a smaller generic fixture cohort.
        for index, pair in enumerate(((0, "familiar"), (0, "heldout_ledger"),
                                      (1, "familiar"), (1, "heldout_ledger"))):
            self.cohort["rows"][index].update(parent_id="R-crossed-parent", condition=pair[0], rendering=pair[1])
        decision = self.evaluate()
        parent = next(parent for parent in decision["baseline"]["skills"]["R"]["parents"]
                      if parent["parent_id"] == "R-crossed-parent")
        self.assertEqual(parent["row_count"], 4)

    def test_duplicate_condition_rendering_within_parent_refused(self):
        self.cohort["rows"][0].update(parent_id="R-crossed-parent", condition=0, rendering="familiar")
        self.cohort["rows"][1].update(parent_id="R-crossed-parent", condition=0, rendering="familiar")
        with self.assertRaisesRegex(ValueError, "duplicate parent/condition/rendering"):
            self.evaluate()

    def test_condition_identity_drift_and_noninteger_condition_refused(self):
        self.cohort["rows"][0]["condition"] = 0
        self.candidates["M1"]["rows"]["R-0"]["condition"] = 1
        with self.assertRaisesRegex(ValueError, "identity drift: condition"):
            self.evaluate()
        self.candidates["M1"]["rows"]["R-0"]["condition"] = False
        with self.assertRaisesRegex(ValueError, "identity drift: condition"):
            self.evaluate()
        del self.candidates["M1"]["rows"]["R-0"]["condition"]
        self.cohort["rows"][0]["condition"] = False
        with self.assertRaisesRegex(ValueError, "condition.*nonnegative integer"):
            self.evaluate()

    def test_actual_policy_rejects_synthetic_unique_rendering_substitute(self):
        baseline, candidates, cohort, policy = fixture(size=4)
        policy["schema"] = "synth-quality-engineering-screen-v1"
        for role in ("R", "S", "RS"):
            for index in range(4):
                row = next(row for row in cohort["rows"] if row["key"] == f"{role}-{index}")
                row.update(parent_id=f"{role}-parent", condition=index // 2, rendering=f"v{index}")
        with self.assertRaisesRegex(ValueError, "frozen 2x2"):
            evaluate_quality(baseline, candidates, cohort, policy)

    def test_actual_policy_requires_explicit_condition(self):
        baseline, candidates, cohort, policy = fixture(size=4)
        policy["schema"] = "synth-quality-engineering-screen-v1"
        with self.assertRaisesRegex(ValueError, "explicit integer condition"):
            evaluate_quality(baseline, candidates, cohort, policy)

    def test_parent_variants_are_not_independent_rows(self):
        baseline, candidates, cohort, policy = fixture(size=10)
        for index in range(9):
            row = cohort["rows"][index]
            row["parent_id"] = "R-heavy-parent"
            row["rendering"] = f"variant-{index}"
            baseline[row["key"]]["correct"] = False
            for arm in candidates.values():
                arm["rows"][row["key"]]["correct"] = True
        baseline["R-9"]["correct"] = False
        for arm in candidates.values():
            arm["rows"]["R-9"]["correct"] = False
        decision = evaluate_quality(baseline, candidates, cohort, policy)
        metric = decision["candidates"]["M1"]["skills"]["R"]
        self.assertEqual(metric["correct"], 9)
        self.assertEqual(metric["row_count"], 10)
        self.assertEqual(metric["parent_count"], 2)
        self.assertEqual(exact(metric["accuracy"]), Fraction(1, 2))
        self.assertEqual(metric["gain_interval"]["unit"], "parent")
        self.assertEqual(metric["gain_interval"]["parent_count"], 2)
        self.assertFalse(metric["gain_interval"]["powered_claim"])

    def test_behavior_dimensions_and_uncertainty_remain_separate(self):
        row = self.candidates["M1"]["rows"]["R-0"]
        row.update(format_correct=False, abstained=True, capped=True)
        decision = self.evaluate()
        metrics = decision["candidates"]["M1"]["behavior"]["overall"]
        self.assertEqual(metrics["format_new_failures"], 1)
        self.assertEqual(metrics["capped"], 1)
        self.assertEqual(metrics["abstention_transitions"]["started"], 1)
        self.assertTrue(decision["candidates"]["M1"]["eligible"])
        self.assertEqual(decision["uncertainty_status"], "EXPLORATORY_ENGINEERING_SCREEN_NOT_POWERED")

    def test_follow_up_routes_without_automatic_launch(self):
        self.policy["diagnosed_poor_reference_resolution"] = True
        decision = self.evaluate()
        self.assertEqual(decision["candidates"]["M1"]["follow_up"]["intervention"],
                         "VERIFIED_INTERMEDIATE_STATE_SUPERVISION_COMPARISON")
        self.policy["diagnosed_poor_reference_resolution"] = False
        for index in range(50, 53):
            self.candidates["M1"]["rows"][f"RS-{index}"]["correct"] = False
        decision = self.evaluate()
        follow_up = decision["candidates"]["M1"]["follow_up"]
        self.assertEqual(follow_up["intervention"], "CONDITION_COVERAGE_AND_FRESH_TRANSFER_SUCCESSOR")
        self.assertFalse(follow_up["auto_launch"])
        self.assertTrue(follow_up["separate_prospective_registration_required"])
        self.assertEqual(decision["candidates"]["M2"]["follow_up"]["intervention"],
                         "REPLICATE_ON_TWO_INDEPENDENT_TRAINING_DRAWS")

    def test_bootstrap_is_deterministic_and_function_is_pure(self):
        inputs = deepcopy((self.baseline, self.candidates, self.cohort, self.policy))
        first = self.evaluate()
        second = self.evaluate()
        self.assertEqual(first, second)
        self.assertEqual(inputs, (self.baseline, self.candidates, self.cohort, self.policy))

    def test_unregistered_arm_or_missing_skill_refused(self):
        self.candidates["M3"] = deepcopy(self.candidates["M1"])
        with self.assertRaisesRegex(ValueError, "prespecified M1/M2"):
            self.evaluate()
        del self.candidates["M3"]
        self.cohort["rows"] = [row for row in self.cohort["rows"] if row["role"] != "RS"]
        with self.assertRaisesRegex(ValueError, "R, S, and RS"):
            self.evaluate()


if __name__ == "__main__":
    unittest.main()

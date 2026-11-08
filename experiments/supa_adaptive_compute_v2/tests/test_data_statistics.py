"""CPU-only data/scorer/statistical checks; no model or sealed data access."""
from __future__ import annotations

from collections import Counter
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.stats import binomtest

from experiments.supa_adaptive_compute_v2.shared.datasets import (
    FAMILIES, ROLES, TRACKS, alignment_prompts, build_banks, diagnostic_rows,
    generate_rows, load_frozen_dataset, semantic_answer, validate_rows,
)
from experiments.supa_adaptive_compute_v2.shared.hashing import canonical_sha256
from experiments.supa_adaptive_compute_v2.shared.scoring import canonical_answer, score_response
from experiments.supa_adaptive_compute_v2.shared.statistics import (
    comparison_annotations, exact_mcnemar, paired_analysis, paired_binary_statistics, power_planning,
)


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.banks = {(track, role): generate_rows(track, role) for track in TRACKS for role in ROLES}

    def test_all_banks_balanced_disjoint_deterministic_and_independently_solved(self):
        scenarios, prompts, latents = set(), set(), set()
        for (track, role), rows in self.banks.items():
            with self.subTest(track=track, role=role):
                count = 1024 if role == "confirmatory_dev" else 128
                counts = validate_rows(rows)
                self.assertEqual(counts["examples"], count)
                self.assertEqual(counts["scenarios"], count)
                self.assertEqual(counts["families"], {f: count // 8 for f in FAMILIES})
                self.assertEqual(rows, generate_rows(track, role))
                if role == "confirmatory_dev":
                    self.assertEqual(set(counts["templates"].values()), {32})
                    self.assertEqual(sum(r["template_unseen_in_engineering"] for r in rows), 256)
                else:
                    self.assertFalse(any(r["template_unseen_in_engineering"] for r in rows))
                for row in rows:
                    self.assertNotIn(row["scenario_id"], scenarios)
                    self.assertNotIn(row["prompt_sha256"], prompts)
                    self.assertNotIn(row["latent_sha256"], latents)
                    scenarios.add(row["scenario_id"]); prompts.add(row["prompt_sha256"]); latents.add(row["latent_sha256"])

    def test_every_template_rejects_consistently_corrupted_gold_and_scorer(self):
        templates = {}
        for row in self.banks["native_armr", "confirmatory_dev"]:
            templates.setdefault(row["template_id"], row)
        self.assertEqual(len(templates), 32)
        for template, source in templates.items():
            with self.subTest(template=template):
                row = copy.deepcopy(source)
                row["scorer"] = {"type": "exact", "answers": ["deliberately_wrong"]}
                row["answer"] = "deliberately_wrong"
                row["latent_sha256"] = canonical_sha256({"facts": row["facts"], "question": row["question"], "answer": row["answer"]})
                self.assertTrue(score_response(row["answer"], row["scorer"])["correct"])
                with self.assertRaisesRegex(ValueError, "independent rendered-fact oracle"):
                    validate_rows([row])

    def test_manual_semantic_oracles_cover_all_32_templates(self):
        # Hand-authored cases and answers, independent of random builders/scorers.
        cases = [
            ("relational.total_order", ["A is taller than B.", "C is shorter than B."], "Who is the shortest?", "C"),
            ("relational.spatial_chain", ["A is 4 meters east of B.", "B is 9 meters west of C."], "What is A's signed eastward displacement from C, in meters?", "-5"),
            ("relational.ancestor_chain", ["A is a parent of B.", "B is a parent of C.", "C is a parent of D."], "Who is the great-grandparent of D?", "A"),
            ("relational.nested_containment", ["Object A is inside box B.", "Box B is inside crate C.", "Crate C is inside room D."], "Which named room contains object A?", "D"),
            ("counterfactual.intervened_causal_chain", ["Device A: lamp equals relay; relay equals switch.", "The switch is initially 0.", "Intervene to set relay to 1, leaving the switch unchanged."], "What is the lamp value?", "1"),
            ("counterfactual.replacement_arithmetic", ["Workshop A makes 4 boxes with 9 items per box.", "Counterfactually replace the items per box by 3; keep the box count fixed."], "What would the total be?", "12"),
            ("counterfactual.overridden_rule", ["A badge normally opens the door.", "An emergency override blocks everyone when active.", "The badge holder arrives; counterfactually the override is inactive."], "Does it open?", "yes"),
            ("counterfactual.changed_route", ["The original route from A goes to B, then C.", "Replace only the first edge A->B by A->D.", "The route from D continues to E."], "After replacement, where do exactly two hops from A end?", "E"),
            ("compositional.successive_maps", ["Map F sends A to B, and D to E.", "Map G sends B to C, and E to A."], "What is G(F(D))?", "A"),
            ("compositional.string_transform", ["The input at station A is abcdef.", "First reverse the string; then remove the first two characters of the reversed string."], "What remains?", "dcba"),
            ("compositional.affine_composition", ["At station A, F(t)=3*t+7 and G(t)=t-4."], "What is G(F(5))?", "18"),
            ("compositional.list_pipeline", ["The list in job A is 5,8,3,2,7,4,1.", "Keep even values, double them, then sort ascending."], "Return JSON.", [4, 8, 16]),
            ("mathematical.inventory_balance", ["Shop A starts with 12 pens, receives 5, and sells 9."], "How many remain?", "8"),
            ("mathematical.unit_rate", ["Machine A makes 7 units per minute at a constant rate."], "How many units does it make in 4 minutes?", "28"),
            ("mathematical.integer_remainder", ["Bin A holds 23 counters.", "Put as many complete groups of 5 counters aside as possible."], "How many remain?", "3"),
            ("mathematical.fraction_of_quantity", ["Farm A harvested 36 apples and sold 2/6 of them."], "How many sold?", "12"),
            ("symbolic_logic.boolean_expression", ["For system A, P=true, Q=true, R=false."], "Evaluate (P AND NOT Q) OR R.", "false"),
            ("symbolic_logic.set_intersection", ["For job A, set A=[1, 4, 8] and set B=[2, 4, 8]."], "Return intersection.", [4, 8]),
            ("symbolic_logic.exclusive_or", ["In circuit A, input P is true and input Q is true."], "What is the XOR output?", "false"),
            ("symbolic_logic.quantified_rule", ["Every A is a B.", "No B is a C.", "Object D is a A."], "Can D be a C under these rules?", "no"),
            ("retrieval_multihop.two_table_join", ["Employee A belongs to team B.", "Employee D belongs to team E.", "Team B uses room C.", "Team E uses room D."], "Which room does employee A's team use?", "C"),
            ("retrieval_multihop.dated_update", ["On day 9, parcel A moved to D.", "On day 2, parcel A was in B.", "On day 7, parcel A moved to C."], "Where after latest event?", "D"),
            ("retrieval_multihop.linked_records", ["Record A points to B.", "Record B points to C.", "Record C has value 9.", "Record D has value 4."], "Follow two pointers from A. What value?", "9"),
            ("retrieval_multihop.distractor_retrieval", ["Record A has code 121.", "Record B has code 422."], "What is the code for record B?", "422"),
            ("ordinary_qa_preservation.appointment_lookup", ["A's dental appointment is at 3:45.", "B's appointment is tomorrow."], "At what time is A's dental appointment?", "3:45"),
            ("ordinary_qa_preservation.object_location", ["A put the keys in the drawer and left them there.", "B put a book on the floor."], "Where are A's keys?", "drawer"),
            ("ordinary_qa_preservation.weather_action", ["For trip A, the forecast is snow.", "The packing rule is umbrella for rain, boots for snow, and sunhat for sun."], "Which item?", "boots"),
            ("ordinary_qa_preservation.menu_lookup", ["Cafe A's menu lists soup at 7 credits, salad at 11 credits, and tea at 4 credits."], "Price of salad?", "11"),
            ("instruction_evidence.json_fields", ["Order A has quantity 9."], "Return order and quantity.", {"order": "A", "quantity": 9}),
            ("instruction_evidence.ordered_output", ["Tokens: A B C."], "Output the second token followed by the first.", "B A"),
            ("instruction_evidence.insufficient_evidence", ["A owns 3 blue marbles.", "B owns 7 red marbles.", "No other facts are supplied."], "How many red marbles does A own?", "unknown"),
            ("instruction_evidence.conflicting_evidence", ["Two equally authoritative records describe the same current state of locker A.", "Record 1: locker A is open.", "Record 2: locker A is closed.", "Neither record has priority."], "Return conflict or consistent.", "conflict"),
        ]
        self.assertEqual(len({c[0] for c in cases}), 32)
        for template, facts, question, expected in cases:
            with self.subTest(template=template):
                self.assertEqual(semantic_answer({"template_id": template, "facts": facts, "question": question}), expected)

    def test_diagnostics_are_real_linked_transformations_with_semantic_changes(self):
        for track in TRACKS:
            originals = self.banks[track, "confirmatory_dev"]
            sources = {r["example_id"]: r for r in originals}
            diagnostics = diagnostic_rows(originals)
            self.assertEqual(len(diagnostics), 128)
            self.assertEqual(len({r["scenario_id"] for r in diagnostics}), 128)
            counts = Counter(r["variant"] for r in diagnostics)
            self.assertEqual(counts, dict(paraphrase=30, irrelevant_context=28, reordered_evidence=32,
                                         renamed_entities=28, changed_relation_direction=4, longer_chain=4,
                                         counterfactual_change=2))
            validate_rows(originals + diagnostics)
            for row in diagnostics:
                source = sources[row["source_example_id"]]
                self.assertEqual(row["scenario_id"], source["scenario_id"])
                self.assertEqual(row["split"], source["split"])
                self.assertFalse(row["primary_eligible"])
                self.assertNotEqual(row["facts"], source["facts"])
                if row["variant"] == "reordered_evidence":
                    self.assertEqual(row["facts"], source["facts"][::-1])
                if row["variant"] == "longer_chain":
                    self.assertEqual(len(row["facts"]), len(source["facts"]) + 1)
                    self.assertIn("4 generations", row["question"])
                if row["variant"] == "counterfactual_change":
                    self.assertEqual(int(row["answer"]), 1-int(source["answer"]))
                if row["variant"] not in {"renamed_entities", "counterfactual_change"}:
                    self.assertEqual(row["answer"], source["answer"])

    def test_fit_holdout_scenario_boundary_and_unseen_template(self):
        rows = self.banks["cross_scale", "calibration"]
        self.assertEqual(Counter(r["alignment_partition"] for r in rows), {"fit": 96, "holdout": 32})
        changed = copy.deepcopy(rows[0]); changed["alignment_partition"] = "holdout"
        with self.assertRaisesRegex(ValueError, "fit/holdout"):
            validate_rows([changed])
        changed = copy.deepcopy(self.banks["cross_scale", "confirmatory_dev"][3]); changed["split"] = "architecture_dev"
        with self.assertRaisesRegex(ValueError, "template crosses"):
            validate_rows([changed])

    def test_new_seed_never_reuses_immutable_ids_for_changed_examples(self):
        first = generate_rows("native_armr", "calibration", seed=1)
        second = generate_rows("native_armr", "calibration", seed=2)
        self.assertFalse({r["example_id"] for r in first} & {r["example_id"] for r in second})

    def test_prompt_hash_schema_and_duplicate_guards(self):
        row = copy.deepcopy(self.banks["native_armr", "confirmatory_dev"][0])
        for field, value in (("prompt_sha256", "0"*64), ("template_id", "relational.fake"), ("prompt", "unbound prompt"), ("question", "")):
            changed = {**row, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_rows([changed])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_rows([row, row])

    def test_freeze_reload_answer_free_alignment_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory(prefix="adaptive_banks_") as directory:
            root = Path(directory) / "banks"
            manifests = build_banks(root)
            for track, manifest in manifests.items():
                for role in (*ROLES, "confirmatory_diagnostics"):
                    rows = load_frozen_dataset(Path(manifest), track=track, role=role)
                    self.assertEqual(len(rows), 1024 if role == "confirmatory_dev" else 128)
                with self.assertRaises(ValueError):
                    load_frozen_dataset(Path(manifest), track="native_armr" if track == "cross_scale" else "cross_scale", role="calibration")
            fit = alignment_prompts(Path(manifests["cross_scale"]), partition="fit")
            holdout = alignment_prompts(Path(manifests["cross_scale"]), partition="holdout")
            self.assertEqual((len(fit), len(holdout)), (96, 32))
            self.assertFalse({r["scenario_id"] for r in fit} & {r["scenario_id"] for r in holdout})
            for row in fit + holdout:
                self.assertEqual(set(row), {"example_id", "scenario_id", "family", "prompt", "prompt_sha256"})
            with self.assertRaises(FileExistsError):
                build_banks(root)
            data_path = root / "native_armr/calibration.jsonl"
            data_path.write_bytes(data_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "dataset hash"):
                load_frozen_dataset(Path(manifests["native_armr"]), track="native_armr", role="calibration")


class ScorerTests(unittest.TestCase):
    def test_exact_is_deterministic_and_rejects_extra_or_contradictory_answers(self):
        scorer = {"type": "exact", "answers": ["42"]}
        for value in ("42", " 42.\n", "４２"):
            self.assertTrue(score_response(value, scorer)["correct"])
            self.assertEqual(score_response(value, scorer), score_response(value, scorer))
        for value in ("41 or 42", "The answer is 42", "42 43", "0; 42", "[42]", "42 because 6*7", "42\nwrong", "142", ""):
            with self.subTest(value=value):
                self.assertFalse(score_response(value, scorer)["correct"])

    def test_json_rejects_duplicates_nonfinite_coercions_extra_fields_and_text(self):
        scorer = {"type": "json_exact", "value": {"n": 1, "values": [2, 3]}}
        self.assertTrue(score_response('{"values": [2, 3], "n": 1}', scorer)["correct"])
        for value in ('{"n":true,"values":[2,3]}', '{"n":1.0,"values":[2,3]}', '{"n":1,"n":1,"values":[2,3]}',
                      '{"n":NaN,"values":[2,3]}', '{"n":Infinity,"values":[2,3]}', '{"n":1e999,"values":[2,3]}',
                      '{"n":1,"values":[3,2]}', '{"n":1,"values":[2,3],"extra":0}',
                      '```json\n{"n":1,"values":[2,3]}\n```', '{"n":1,"values":[2,3]} trailing'):
            with self.subTest(value=value):
                self.assertFalse(score_response(value, scorer)["correct"])

    def test_invalid_scorer_configuration_fails_closed(self):
        for scorer in ({"type": "numeric", "value": 1}, {"type": "exact", "answers": ["a", "b"]},
                       {"type": "exact", "answers": [""]}, {"type": "exact", "answers": [1]},
                       {"type": "json_exact", "value": float("nan")}, {"type": "json_exact", "value": {1: "a"}}):
            with self.subTest(scorer=scorer), self.assertRaises(ValueError):
                canonical_answer({"scorer": scorer})


class StatisticsTests(unittest.TestCase):
    def test_exact_mcnemar_against_scipy_binomtest(self):
        for n in (1, 2, 5, 16, 64, 128, 1024):
            for wins in sorted({0, 1, n//4, n//2, n-1, n}):
                losses = n-wins
                left, right = [False]*wins+[True]*losses, [True]*wins+[False]*losses
                result = exact_mcnemar(left, right)
                expected = binomtest(wins, n, .5, alternative="two-sided").pvalue
                with self.subTest(n=n, wins=wins):
                    self.assertTrue(math.isclose(result["p_value_two_sided"], expected, rel_tol=2e-10, abs_tol=1e-300))
                    self.assertEqual(result["candidate_only_correct"], wins)
                    self.assertEqual(result["baseline_only_correct"], losses)
        self.assertEqual(exact_mcnemar([True, False], [True, False])["p_value_two_sided"], 1.)

    def test_scenario_bootstrap_independent_reference_and_cluster_estimand(self):
        result = paired_binary_statistics([0, 0, 0, 0], [1, 1, 1, 0], scenario_ids=["a", "a", "a", "b"], seed=123, bootstrap_samples=1000)
        indices = np.random.default_rng(123).integers(0, 2, size=(1000, 2))
        estimates = np.array([1., 0.])[indices].mean(axis=1)
        self.assertEqual(result["scenario_bootstrap_ci"]["lower"], np.quantile(estimates, .025))
        self.assertEqual(result["scenario_bootstrap_ci"]["upper"], np.quantile(estimates, .975))
        self.assertEqual(result["independent_scenarios"], 2)
        self.assertEqual(result["accuracy_difference"], .75)
        self.assertEqual(result["scenario_macro_difference"], .5)
        self.assertEqual(result["mcnemar"]["inference_status"], "descriptive_only_correlated_variants")
        self.assertEqual(result["primary_test"], "exact_cluster_sign_flip")

    def test_paired_rates_wins_losses_and_undefined_rer(self):
        result = paired_binary_statistics([1, 0, 0, 1], [1, 1, 1, 0], bootstrap_samples=100)
        self.assertEqual((result["paired_wins"], result["paired_losses"]), (2, 1))
        self.assertEqual(result["absolute_error_reduction"], .25)
        self.assertEqual(result["relative_error_reduction"], .5)
        perfect = paired_binary_statistics([1, 1], [1, 1], bootstrap_samples=100)
        self.assertIsNone(perfect["relative_error_reduction"])
        self.assertIsNotNone(perfect["degenerate_ci_caution"])

    def test_binary_and_cluster_inputs_reject_nonfinite_and_missing_values(self):
        for left, right in (([], []), ([1], [1, 0]), ([float("nan")], [0]), ([1], [float("inf")]), ([2], [0]), (["1"], [0])):
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                paired_binary_statistics(left, right, bootstrap_samples=10)
        for ids in ([None], [""], [" "]):
            with self.assertRaises(ValueError):
                paired_binary_statistics([0], [1], scenario_ids=ids, bootstrap_samples=10)
        with self.assertRaises(ValueError):
            paired_binary_statistics([0], [1], bootstrap_samples=0)

    def test_paired_coverage_metadata_safety_and_finite_cost_guards(self):
        a = dict(example_id="x", scenario_id="s", family="relational", score=0, variant="original", response="a", gold_nll=2., wall_seconds=1.)
        b = {**a, "score": 1, "response": "b", "gold_nll": 1.}
        result = paired_analysis([a], [b], expected_example_ids=["x"], bootstrap_samples=10)
        self.assertEqual(result["overall"]["nll"]["candidate_minus_baseline"], -1.)
        self.assertEqual(result["overall"]["output_identity"]["rate"], 0.)
        for baseline, candidate, expected in (([], [b], ["x"]), ([a], [b], ["x", "y"]), ([a], [b], ["x", "x"]),
                                               ([a, a], [b], None), ([a], [{**b, "family": "other"}], None),
                                               ([a], [{**b, "safety_stopped": True}], None),
                                               ([a], [{**b, "gold_nll": float("nan")}], None),
                                               ([a], [{**b, "wall_seconds": float("inf")}], None),
                                               ([a], [{**b, "generation_dense_matmul_flops_estimate": -1}], None)):
            with self.subTest(candidate=candidate, expected=expected), self.assertRaises(ValueError):
                paired_analysis(baseline, candidate, expected_example_ids=expected, bootstrap_samples=10)
        with self.assertRaisesRegex(ValueError, "diagnostics"):
            paired_analysis([{**a, "variant": "paraphrase"}], [{**b, "variant": "paraphrase"}], bootstrap_samples=10)

    def test_holm_known_reference_and_power_planning(self):
        adjusted = comparison_annotations({"a": .01, "b": .04, "c": .03}, analysis_role="confirmatory")
        self.assertEqual([adjusted[k]["holm_p"] for k in "abc"], [.03, .06, .06])
        plan = power_planning()
        self.assertEqual(len(plan["rows"]), 48)
        selected = {r["discordance"]: r for r in plan["rows"] if r["n_independent_scenarios"] == 1024 and r["effect"] == "2pp"}
        self.assertAlmostEqual(selected[.1]["approximate_mde_at_target_power"], .027686, places=5)
        self.assertGreater(selected[.05]["approximate_power"], selected[.1]["approximate_power"])
        for row in plan["rows"]:
            self.assertTrue(0 <= row["approximate_power"] <= 1)
            self.assertGreater(row["approximate_n_for_target_power"], 0)
        for kwargs in ({"alpha": float("nan")}, {"baseline_accuracy": 1.}, {"target_power": .4}):
            with self.assertRaises(ValueError):
                power_planning(**kwargs)


if __name__ == "__main__":
    unittest.main()

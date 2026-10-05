"""Independent CPU schedule accounting and deliberate corruption controls.

Default arrays below are artificial mechanics fixtures, not Titan token counts.
Exact-local tokenizer integration is opt-in and never admits native training.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from balanced_corpus import build_balanced_plan, compile_balanced_records
from cpu_tokenizer import ASSISTANT_PREFIX, EOS_ID, load_local_tokenizer
from matching_schedule import freeze_enrollment, match_schedule, validate_schedule
from rs_contract import digest, parse_visible


def first_depth(spec):
    name = spec["lookup"]["requested"][0]
    visited = set()
    while type(name) is str:
        if name in visited or name not in spec["lookup"]["links"]:
            raise AssertionError("fixture requested chain is unresolved")
        visited.add(name)
        name = spec["lookup"]["links"][name]
    return len(visited)


def fixtures():
    plan = build_balanced_plan()
    enrollment = freeze_enrollment(plan)
    selected = set(enrollment["parents"])
    rows = [row for row in compile_balanced_records(plan) if row["parent_id"] in selected]
    return plan, enrollment, rows


def artificial_arrays(rows):
    """Fixed 64-position reversible-character targets, no tokenizer claim."""
    arrays = {}
    for row in rows:
        target = row["target"]
        completion = [1000 + ord(char) for char in target] + [EOS_ID]
        prompt_ids = [271] * (64 - len(completion))
        messages = row["messages"]
        prompt = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        prompt += ASSISTANT_PREFIX
        full = prompt + target + "<|im_end|>\n"
        arrays[row["row_id"]] = {
            "prompt_ids": prompt_ids, "completion_ids": completion,
            "input_ids": prompt_ids + completion, "labels": [-100] * len(prompt_ids) + completion,
            "attention_mask": [1] * 64, "prompt_tokens": len(prompt_ids),
            "supervised_tokens": len(completion), "total_tokens": 64,
            "prompt_text_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "full_template_sha256": hashlib.sha256(full.encode()).hexdigest()}
    return arrays


def mechanics_decisions():
    decisions = json.loads((PACKAGE / "PILOT_DECISIONS.json").read_bytes())
    decisions["budget"]["minimum_consumed_per_arm_per_seed"] = 20000
    decisions["budget"]["ceiling_per_trained_arm_per_seed"] = 100000
    return decisions


def rehash(report):
    report["sha256"] = digest({key: value for key, value in report.items() if key != "sha256"})


def independent_v1_residual_lower_bound(rows, arrays, enrollment):
    """Exact dual certificate over token differences and marginal histograms.

    An arbitrary dual multiplier is valid as a lower bound after this algebraic
    check; finding it never supplies proof. No optimizer or production matching
    helper is used here. Dropping all other constraints only enlarges feasibility.
    """
    quads = [(parent_id, ci) for parent_id in enrollment["parents"] for ci in range(4)]
    grouped = {key: [row for row in rows if (row["parent_id"], row["condition_index"]) == key]
               for key in quads}
    matrix = []
    for skill in ("R", "S"):
        for stratum in range(1, 6):
            column = []
            for quad in quads:
                count = 0
                for row in grouped[quad]:
                    spec = parse_visible(row["messages"])
                    row_skill = "S" if "selection" in spec else "R"
                    value = spec["selection"]["k"] if row_skill == "S" else first_depth(spec)
                    if (row_skill, value) == (skill, stratum):
                        count += 1 if row["arm"] == "M1" else -1
                column.append(count)
            matrix.append(column)
    costs = [sum(arrays[row["row_id"]]["total_tokens"] * (1 if row["arm"] == "M2" else -1)
                 for row in grouped[quad]) for quad in quads]
    dual = list(map(Fraction, (20, 10, 0, -10, -20, 2, 0, -2, -4, -6)))
    remainder = [Fraction(costs[j]) - sum(Fraction(matrix[i][j]) * dual[i] for i in range(10))
                 for j in range(len(quads))]
    # If H*w=0, c*w=(H^T*y+r)*w=r*w. A positive coefficient is minimized
    # at lower bound1, a negative coefficient at upper bound6. This inequality
    # holds for real weights too, and therefore for all integer schedules.
    return sum(value * (1 if value >= 0 else 6) for value in remainder)


class EnrollmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan, cls.enrollment, cls.rows = fixtures()

    def test_enrollment_is_twenty_train_parents_and_two_per_atomic_stratum(self):
        parents = self.enrollment["parents"]
        self.assertEqual(20, len(parents))
        self.assertEqual(20, len(set(parents)))
        self.assertEqual(self.plan["sha256"], self.enrollment["plan_sha256"])
        strata = Counter()
        for parent_id in parents:
            self.assertEqual("train", self.plan["assignments"][parent_id]["split"])
            spec = self.plan["parents"][parent_id]["spec"]
            strata[("S", spec["selection"]["k"]) if "selection" in spec
                   else ("R", first_depth(spec))] += 1
        self.assertEqual({(skill, stratum): 2 for skill in ("R", "S") for stratum in range(1, 6)}, dict(strata))
        self.assertEqual(640, len(self.rows))

    def test_enrollment_precedes_arrays_and_is_dictionary_order_independent(self):
        reversed_plan = copy.deepcopy(self.plan)
        for field in ("parents", "sources", "assignments", "condition_plans"):
            reversed_plan[field] = dict(reversed(list(reversed_plan[field].items())))
        self.assertEqual(self.enrollment, freeze_enrollment(reversed_plan))
        self.assertEqual(self.enrollment, freeze_enrollment(copy.deepcopy(self.plan)))

    def test_enrollment_chooses_source_hash_rank_without_token_feedback(self):
        expected = []
        for skill in ("R", "S"):
            for stratum in range(1, 6):
                candidates = []
                for parent_id, parent in self.plan["parents"].items():
                    if self.plan["assignments"][parent_id]["split"] != "train":
                        continue
                    spec = parent["spec"]
                    actual_skill = "S" if "selection" in spec else "R"
                    actual_stratum = spec["selection"]["k"] if actual_skill == "S" else first_depth(spec)
                    if (actual_skill, actual_stratum) == (skill, stratum):
                        candidates.append(parent_id)
                candidates.sort(key=lambda parent_id: digest({"seed": self.plan["seed"], "pool": parent_id}))
                expected.extend(candidates[:2])
        self.assertEqual(expected, self.enrollment["parents"])
        self.assertEqual(digest({key: value for key, value in self.enrollment.items() if key != "sha256"}),
                         self.enrollment["sha256"])

    def test_insufficient_train_strata_and_noninteger_counts_fail_closed(self):
        with self.assertRaises(ValueError):
            freeze_enrollment(build_balanced_plan(parent_count=2))
        for count in (True, False, 0, -1, 2.0, "2", None):
            with self.subTest(count=count), self.assertRaises(ValueError):
                freeze_enrollment(self.plan, parents_per_stratum=count)


@unittest.skipUnless(importlib.util.find_spec("scipy"), "optional CPU optimizer unavailable")
class ScheduleCheckerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan, cls.enrollment, cls.rows = fixtures()
        cls.arrays = artificial_arrays(cls.rows)
        cls.decisions = mechanics_decisions()
        cls.schedule = match_schedule(cls.rows, cls.arrays, cls.enrollment, cls.decisions)
        if cls.schedule["status"] != "MATCHED_CPU_SCHEDULE":
            raise AssertionError(f"mechanics matching fixture failed: {cls.schedule}")

    def assert_rejected(self, report=None, rows=None, arrays=None, enrollment=None, decisions=None):
        with self.assertRaises(ValueError):
            validate_schedule(self.schedule if report is None else report,
                              self.rows if rows is None else rows,
                              self.arrays if arrays is None else arrays,
                              self.enrollment if enrollment is None else enrollment,
                              self.decisions if decisions is None else decisions)

    def test_independent_counts_exposures_strata_and_position_controls(self):
        validate_schedule(self.schedule, self.rows, self.arrays, self.enrollment, self.decisions)
        mapping = {row["row_id"]: row for row in self.rows}
        totals, skills, exposures, histograms = {}, {}, {}, {}
        positions, opportunities, slots = {}, {}, {}
        for arm in ("M1", "M2"):
            ids = self.schedule["schedules"][arm]
            self.assertEqual(0, len(ids) % 20)
            selected = [mapping[row_id] for row_id in ids]
            self.assertTrue(all(row["arm"] == arm and row["split"] == "train" for row in selected))
            totals[arm] = sum(self.arrays[row_id]["total_tokens"] for row_id in ids)
            skills[arm] = Counter()
            exposures[arm] = Counter()
            histograms[arm] = {"R": Counter(), "S": Counter()}
            for skill in ("R", "S"):
                positions[arm, skill], opportunities[arm, skill], slots[arm, skill] = Counter(), defaultdict(float), 0
            for row in selected:
                skill, spec = row["skill"], row["spec"]
                skills[arm][skill] += self.arrays[row["row_id"]]["total_tokens"]
                exposures[arm][row["parent_id"]] += 1
                histograms[arm][skill][spec["selection"]["k"] if skill == "S" else first_depth(spec)] += 1
                answer = json.loads(row["target"])["answer"]
                positions[arm, skill].update(answer)
                slots[arm, skill] += len(answer)
                for position in range(len(spec["initial"])):
                    opportunities[arm, skill][position] += len(answer) / len(spec["initial"])
            budget = self.decisions["budget"]
            self.assertLessEqual(budget["minimum_consumed_per_arm_per_seed"], totals[arm])
            self.assertLessEqual(totals[arm], budget["ceiling_per_trained_arm_per_seed"])
            for skill in ("R", "S"):
                self.assertLessEqual(abs(skills[arm][skill] - totals[arm] / 2), 512)
                for position, opportunity in opportunities[arm, skill].items():
                    self.assertLessEqual(abs(positions[arm, skill][position] - opportunity),
                                         .04 * slots[arm, skill] + 1e-8)
        self.assertEqual(len(self.schedule["schedules"]["M1"]), len(self.schedule["schedules"]["M2"]))
        self.assertEqual(exposures["M1"], exposures["M2"])
        self.assertEqual(histograms["M1"], histograms["M2"])
        self.assertLessEqual(abs(totals["M1"] - totals["M2"]), 512)
        for skill in ("R", "S"):
            for position in set(positions["M1", skill]) | set(positions["M2", skill]):
                self.assertLessEqual(abs(positions["M1", skill][position] - positions["M2", skill][position]),
                                     .02 * min(slots["M1", skill], slots["M2", skill]) + 1e-8)
        expected_quads = {f"{parent_id}:{ci}" for parent_id in self.enrollment["parents"] for ci in range(4)}
        self.assertEqual(expected_quads, set(self.schedule["quad_weights"]))
        for key, weight in self.schedule["quad_weights"].items():
            self.assertIs(type(weight), int)
            self.assertGreaterEqual(weight, 1)
            self.assertLessEqual(weight, 6)
            parent_id, ci = key.rsplit(":", 1)
            for arm in ("M1", "M2"):
                counter = Counter(self.schedule["schedules"][arm])
                for row in self.rows:
                    if row["parent_id"] == parent_id and row["condition_index"] == int(ci) and row["arm"] == arm:
                        self.assertEqual(weight, counter[row["row_id"]])

    def test_independent_checker_uses_no_optimizer_import_or_solver(self):
        import builtins
        original_import = builtins.__import__
        def blocked_import(name, *args, **kwargs):
            if name.split(".", 1)[0] in ("scipy", "numpy"):
                raise AssertionError("checker imported an optimizer")
            return original_import(name, *args, **kwargs)
        with patch.object(builtins, "__import__", side_effect=blocked_import):
            validate_schedule(self.schedule, self.rows, self.arrays, self.enrollment, self.decisions)

    def test_rehashed_schedule_row_count_order_and_unregistered_ids_rejected(self):
        for fault in ("missing", "replace", "order", "unregistered"):
            report = copy.deepcopy(self.schedule)
            ids = report["schedules"]["M1"]
            if fault == "missing":
                ids.pop()
            elif fault == "replace":
                ids[0] = ids[-1]
            elif fault == "order":
                ids[0], ids[-1] = ids[-1], ids[0]
            else:
                ids[0] = "0" * 64
            rehash(report)
            with self.subTest(fault=fault):
                self.assert_rejected(report=report)

    def test_rehashed_quad_weight_faults_rejected(self):
        for value in (0, 7, True, 1.5):
            report = copy.deepcopy(self.schedule)
            report["quad_weights"][next(iter(report["quad_weights"]))] = value
            rehash(report)
            with self.subTest(value=value):
                self.assert_rejected(report=report)

    def test_rehashed_reported_totals_updates_bindings_and_native_flags_rejected(self):
        for fault in ("input_total", "supervised_total", "skill_total", "updates", "array_binding",
                      "position", "stratum", "training_flag", "launch_flag"):
            report = copy.deepcopy(self.schedule)
            if fault == "input_total": report["metrics"]["totals"]["M1"]["nonpadding_tokens"] += 1
            elif fault == "supervised_total": report["metrics"]["totals"]["M1"]["supervised_tokens"] += 1
            elif fault == "skill_total": report["metrics"]["totals"]["M1"]["skill_tokens"]["R"] += 1
            elif fault == "updates": report["optimizer_updates_per_arm"] += 1
            elif fault == "array_binding": report["arrays_sha256"] = "0" * 64
            elif fault == "position":
                counts = report["metrics"]["answer_position_counts"]["M1"]["R"]
                counts[next(iter(counts))] += 1
            elif fault == "stratum":
                counts = report["metrics"]["strata"]["M1"]["S_k"]
                counts[next(iter(counts))] += 1
            elif fault == "training_flag": report["training_admitted"] = True
            else: report["native_launch_authorized"] = True
            rehash(report)
            with self.subTest(fault=fault):
                self.assert_rejected(report=report)

    def test_counts_masks_eos_ids_and_array_pool_faults_rejected(self):
        first_id = next(iter(self.arrays))
        for fault in ("total", "supervised", "prompt", "label", "attention", "eos", "bool", "missing", "extra"):
            arrays = copy.deepcopy(self.arrays)
            row = arrays[first_id]
            if fault == "total": row["total_tokens"] += 1
            elif fault == "supervised": row["supervised_tokens"] += 1
            elif fault == "prompt": row["prompt_tokens"] += 1
            elif fault == "label": row["labels"][0] = row["input_ids"][0]
            elif fault == "attention": row["attention_mask"][0] = 0
            elif fault == "eos": row["input_ids"][-1] = 17
            elif fault == "bool": row["input_ids"][0] = True
            elif fault == "missing": del arrays[first_id]
            else: arrays["extra"] = copy.deepcopy(row)
            with self.subTest(fault=fault):
                self.assert_rejected(arrays=arrays)

    def test_train_role_atomic_labels_targets_and_row_pool_faults_rejected(self):
        for fault in ("role", "skill", "answer", "duplicate", "missing"):
            rows = copy.deepcopy(self.rows)
            if fault == "role": rows[0]["split"] = "dev"
            elif fault == "skill": rows[0]["skill"] = "RS"
            elif fault == "answer": rows[0]["target"] = '{"answer":"UNKNOWN"}'
            elif fault == "duplicate": rows.append(copy.deepcopy(rows[0]))
            else: rows.pop()
            with self.subTest(fault=fault):
                self.assert_rejected(rows=rows)

    def test_parent_enrollment_and_policy_mutations_rejected(self):
        for fault in ("parent", "duplicate", "hash"):
            enrollment = copy.deepcopy(self.enrollment)
            if fault == "parent": enrollment["parents"][0] = "unregistered"
            elif fault == "duplicate": enrollment["parents"][0] = enrollment["parents"][-1]
            else: enrollment["plan_sha256"] = "0" * 64
            rehash(enrollment)
            with self.subTest(fault=fault):
                self.assert_rejected(enrollment=enrollment)
        decisions = copy.deepcopy(self.decisions)
        decisions["budget"]["ceiling_per_trained_arm_per_seed"] = True
        self.assert_rejected(decisions=decisions)

    def test_impossible_tiny_budget_returns_blocked_without_a_partial_schedule(self):
        decisions = copy.deepcopy(self.decisions)
        decisions["budget"]["minimum_consumed_per_arm_per_seed"] = 1
        decisions["budget"]["ceiling_per_trained_arm_per_seed"] = 1
        result = match_schedule(self.rows, self.arrays, self.enrollment, decisions)
        self.assertEqual("BLOCKED_MATCHING", result["status"])
        self.assertFalse(result.get("schedules"))


@unittest.skipUnless(os.environ.get("SYNTH_CRS_LOCAL_TOKENIZER_DIR") and importlib.util.find_spec("scipy"),
                     "set SYNTH_CRS_LOCAL_TOKENIZER_DIR for pinned local matching integration")
class ExactLocalMatchingIntegrationTests(unittest.TestCase):
    def test_actual_tokenized_v1_pool_has_proven_residual_infeasibility(self):
        _, enrollment, rows = fixtures()
        tokenizer = load_local_tokenizer(model_dir=os.environ["SYNTH_CRS_LOCAL_TOKENIZER_DIR"])
        before = tokenizer.verify_binding()
        arrays = {row["row_id"]: tokenizer.encode_example(row["messages"], row["target"]) for row in rows}
        decisions = json.loads((PACKAGE / "PILOT_DECISIONS.json").read_bytes())
        result = match_schedule(rows, arrays, enrollment, decisions)
        bound = independent_v1_residual_lower_bound(rows, arrays, enrollment)
        self.assertEqual(Fraction(736), bound)
        self.assertGreater(bound, decisions["budget"]["maximum_between_arm_residual_tokens"])
        self.assertEqual("BLOCKED_MATCHING", result["status"])
        self.assertEqual("solver_reported_infeasible", result["reason"])
        self.assertEqual(2, result["solver"]["status_code"])
        self.assertFalse(result.get("schedules"))
        self.assertFalse(result["training_admitted"])
        self.assertEqual(before, tokenizer.verify_binding())


if __name__ == "__main__":
    unittest.main()

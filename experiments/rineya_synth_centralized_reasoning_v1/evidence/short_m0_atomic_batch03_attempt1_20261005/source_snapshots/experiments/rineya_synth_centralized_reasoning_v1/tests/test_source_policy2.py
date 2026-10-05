"""Source-policy 2 adversarial semantics and preserved V1 evidence continuity.

These are CPU source/verification checks. No tokenizer, optimizer, model or GPU
is required; saved V1 token arrays are public, explicitly unadmitted artifacts.
"""
from __future__ import annotations

from collections import Counter
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
for location in (PACKAGE, PACKAGE / "tests"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from balanced_corpus import (build_balanced_plan, compile_balanced_records,
                             condition_worlds, validate_plan)
from matching_schedule import freeze_enrollment
from rs_contract import canonical, digest, format_answer, parse_visible, render, validate_spec
from rs_verifier import verify
from test_matching_schedule import independent_v1_residual_lower_bound

ARCHIVE = PACKAGE / "data" / "crossed_v1_blocked_20261003"


def walk(spec, start):
    """Direct test oracle: walk visible links, with no production chain helper."""
    nodes = []
    while type(start) is str:
        if start in nodes or start not in spec["lookup"]["links"]:
            raise ValueError("unresolved test route")
        nodes.append(start)
        start = spec["lookup"]["links"][start]
    return nodes, start


def adversarial_source():
    return {
        "initial": [1, 2, 4, 8, 16, 32],
        "lookup": {
            "requested": ["q0", "other"],
            "links": {
                "q0": "q1", "q1": "q2", "q2": 0, "other": 1,
                "x0": "x1", "x1": "x2", "x2": 5,
                "shortcut": 4,
                "same_old0": "same_old1", "same_old1": "same_old2", "same_old2": 0,
                "cycle": "cycle", "broken": "missing"}}}


class MatchedDepthRebindingTests(unittest.TestCase):
    def test_same_depth_head_is_selected_and_internal_shortcuts_are_never_selected(self):
        base = adversarial_source()
        original = copy.deepcopy(base)
        for seed in range(64):
            with self.subTest(seed=seed):
                worlds = condition_worlds(base, seed, source_policy=2)
                rebound = next(world for world in worlds if world["name"] == "request_rebound")
                spec = rebound["spec"]
                self.assertEqual("x0", spec["lookup"]["requested"][0])
                self.assertEqual(base["lookup"]["requested"][1:], spec["lookup"]["requested"][1:])
                self.assertEqual(base["lookup"]["links"], spec["lookup"]["links"])
                self.assertEqual(3, len(walk(spec, "x0")[0]))
                self.assertEqual(5, walk(spec, "x0")[1])
                self.assertEqual(["lookup.requested"], rebound["changed_fields"])
                self.assertEqual("change", rebound["effect"])
                self.assertTrue(verify(render(spec), '{"answer":[1,5]}')["accepted"])
                self.assertFalse(verify(render(spec), '{"answer":[0,1]}')["accepted"])
        self.assertEqual(original, base)

    def test_only_shorter_novel_routes_do_not_silently_replace_same_depth_requirement(self):
        base = adversarial_source()
        del base["lookup"]["links"]["x0"]
        self.assertTrue(all(len(walk(base, alias)[0]) < 3 for alias in ("x1", "x2", "shortcut")))
        for seed in range(8):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                condition_worlds(base, seed, source_policy=2)
        # Same-depth aliases resolving to a prior terminal cannot supply novelty.
        self.assertEqual(3, len(walk(base, "same_old0")[0]))
        self.assertEqual(0, walk(base, "same_old0")[1])

    def test_v1_retains_its_original_shorter_alias_behavior(self):
        base = adversarial_source()
        selected = set()
        for seed in range(64):
            spec = next(world["spec"] for world in condition_worlds(base, seed, source_policy=1)
                        if world["name"] == "request_rebound")
            selected.add(spec["lookup"]["requested"][0])
        self.assertTrue(selected & {"x1", "x2", "shortcut"})

    def test_unknown_or_noninteger_policy_versions_fail_closed(self):
        for value in (True, False, 0, 3, 2.0, "2", None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    condition_worlds(adversarial_source(), 1, source_policy=value)
                with self.assertRaises(ValueError):
                    build_balanced_plan(source_policy=value)


class SourceCatalogAndPreservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan1 = build_balanced_plan(source_policy=1)
        cls.plan2 = build_balanced_plan(source_policy=2)

    def test_registered_v2_entire_catalog_is_reproducible_frozen_and_distinct_from_v1(self):
        registration = json.loads((PACKAGE / "SOURCE_POLICY_002.json").read_bytes())
        self.assertEqual(2, registration["source_policy"])
        self.assertEqual(64, registration["candidate_parents"])
        self.assertEqual(20261003, registration["seed"])
        self.assertTrue(registration["registered_before_v2_solver_attempt"])
        self.assertFalse(registration["token_budget_changed"])
        self.assertFalse(registration["training_admitted"])
        self.assertFalse(registration["native_launch_authorized"])
        self.assertEqual("synth-balanced-corpus-2", self.plan2["version"])
        self.assertNotEqual(self.plan1["sha256"], self.plan2["sha256"])
        self.assertEqual(self.plan2, build_balanced_plan(source_policy=2))
        validate_plan(self.plan2)
        for source in self.plan2["sources"].values():
            self.assertEqual(2, source["version"])
            self.assertEqual(2, source["content"]["source_policy"])
            self.assertEqual(digest(source["content"]), source["sha256"])

    def test_generated_x_chain_and_rebinding_preserve_depth_and_change_endpoint(self):
        strata = Counter()
        for parent_id, parent in self.plan2["parents"].items():
            spec = parent["spec"]
            skill = validate_spec(spec)
            stratum = spec["selection"]["k"] if skill == "S" else len(walk(spec, spec["lookup"]["requested"][0])[0])
            if self.plan2["assignments"][parent_id]["split"] == "train":
                strata[skill, stratum] += 1
            if skill != "R":
                continue
            self.assertEqual(stratum, len(walk(spec, "x0_0")[0]))
            old_terminals = {walk(spec, name)[1] for name in spec["lookup"]["requested"]}
            world = next(w for w in self.plan2["condition_plans"][parent_id] if w["name"] == "request_rebound")
            changed = world["spec"]
            route, new_terminal = walk(changed, changed["lookup"]["requested"][0])
            self.assertEqual(stratum, len(route))
            self.assertNotIn(new_terminal, old_terminals)
            expected = {walk(spec, name)[1] for name in spec["lookup"]["requested"][1:]} | {new_terminal}
            self.assertTrue(verify(render(changed), format_answer(sorted(expected)))["accepted"])
            self.assertFalse(verify(render(changed), format_answer(sorted(old_terminals)))["accepted"])
            self.assertEqual(len(old_terminals), len(expected))
        self.assertTrue(all(strata[skill, value] >= 2 for skill in ("R", "S") for value in range(1, 6)))
        enrollment = freeze_enrollment(self.plan2)
        self.assertEqual(20, len(enrollment["parents"]))
        self.assertEqual(self.plan2["sha256"], enrollment["plan_sha256"])
        self.assertFalse(enrollment["training_admitted"])

    def test_all_registered_v2_rows_have_atomic_visible_independently_verified_targets(self):
        rows = compile_balanced_records(self.plan2)
        self.assertEqual(2048, len(rows))
        for row in rows:
            self.assertIn(validate_spec(parse_visible(row["messages"])), ("R", "S"))
            self.assertTrue(verify(row["messages"], row["target"])["accepted"])

    def test_v1_archive_file_hashes_plan_records_and_exact_736_certificate_continue_to_match(self):
        receipt = json.loads((ARCHIVE / "CPU_BUNDLE_RECEIPT.json").read_bytes())
        self.assertFalse(receipt["training_admitted"])
        self.assertFalse(receipt["native_launch_authorized"])
        self.assertEqual(0, receipt["model_loads"])
        self.assertEqual(0, receipt["training_steps"])
        for relative, expected in receipt["file_sha256"].items():
            path = (ARCHIVE / relative).resolve()
            self.assertTrue(path.is_relative_to(ARCHIVE.resolve()))
            self.assertEqual(expected, hashlib.sha256(path.read_bytes()).hexdigest(), relative)
        preserved_plan = json.loads((ARCHIVE / "plan.json").read_bytes())
        self.assertEqual(canonical(self.plan1), canonical(preserved_plan))
        self.assertEqual(receipt["plan_sha256"], self.plan1["sha256"])
        self.assertEqual(self.plan1, build_balanced_plan())
        enrollment = json.loads((ARCHIVE / "enrollment.json").read_bytes())
        self.assertEqual(canonical(enrollment), canonical(freeze_enrollment(self.plan1)))
        rows = json.loads((ARCHIVE / "candidate_records.json").read_bytes())
        selected = [row for row in compile_balanced_records(self.plan1) if row["parent_id"] in enrollment["parents"]]
        self.assertEqual(canonical(selected), canonical(rows))
        arrays = json.loads((ARCHIVE / "arrays.json").read_bytes())
        self.assertEqual(receipt["arrays_sha256"], digest(arrays))
        self.assertEqual(Fraction(736), independent_v1_residual_lower_bound(rows, arrays, enrollment))
        diagnostic = json.loads((ARCHIVE / "constraint_diagnostic.json").read_bytes())
        self.assertEqual(736, diagnostic["certificate"]["exact_lower_bound_input_tokens"])
        self.assertEqual(512, diagnostic["registered_budget_residual_max"])
        self.assertEqual("BLOCKED_MATCHING", diagnostic["observed_matching_result"]["status"])


if __name__ == "__main__":
    unittest.main()

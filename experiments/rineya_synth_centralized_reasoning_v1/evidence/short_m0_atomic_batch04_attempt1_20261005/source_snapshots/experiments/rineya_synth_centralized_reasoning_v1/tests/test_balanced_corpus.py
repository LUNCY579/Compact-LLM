"""Independent CPU checks for crossed atomic R/S corpus construction.

These tests use hand-specified reference answers and an independent bit-mask
oracle. They do not establish token matching, native admission, or transfer.
"""
from __future__ import annotations

import copy
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from balanced_corpus import (audit_balance, build_balanced_plan, choose_target,
                             compile_balanced_records, condition_worlds,
                             family_key, surface_view,
                             validate_balanced_records)
from rs_contract import (canonical, digest, format_answer, parse_answer,
                         parse_visible, render, validate_spec)
from rs_verifier import verify


def r_world(shared=False):
    """Two length-three requests, with a deliberately unused broken alias."""
    links = {"q": "q_mid", "q_mid": "q_end", "q_end": 1,
             "r": "r_mid", "r_mid": "r_end", "r_end": 4,
             "other": 2, "unused": "missing"}
    if shared:
        links["r_mid"] = "q_end"
        del links["r_end"]
    return {"initial": [1, 3, 7, 15, 31, 63],
            "lookup": {"links": links, "requested": ["q", "r"]}}


def s_world():
    return {"initial": [1, 3, 7, 15, 31, 63],
            "selection": {"k": 2, "target": 34}}


def brute_answers(spec):
    """Small test oracle: direct references and exhaustive bit masks.

    No compiler target, solver output, or verifier solution list is used to
    determine a reference answer. Equal-valued items retain distinct indices.
    """
    eligible = set(range(len(spec["initial"])))
    if "lookup" in spec:
        eligible = set()
        links = spec["lookup"]["links"]
        for requested in spec["lookup"]["requested"]:
            seen = set()
            while type(requested) is str:
                if requested in seen or requested not in links:
                    return set()
                seen.add(requested)
                requested = links[requested]
            eligible.add(requested)
    if "selection" not in spec:
        return {tuple(sorted(eligible))}
    selection = spec["selection"]
    answers = set()
    for mask in range(1 << len(spec["initial"])):
        chosen = tuple(i for i in range(len(spec["initial"])) if mask & (1 << i))
        if (len(chosen) == selection["k"] and set(chosen) <= eligible and
                sum(spec["initial"][i] for i in chosen) == selection["target"]):
            answers.add(chosen)
    return answers


def requested_depth(spec, name):
    depth = 0
    seen = set()
    while type(name) is str:
        if name in seen or name not in spec["lookup"]["links"]:
            raise AssertionError("test expected a solved chain")
        seen.add(name)
        depth += 1
        name = spec["lookup"]["links"][name]
    return depth


def independent_reindex(spec, item_map):
    """Apply the declared old-index -> new-index map without compiler code."""
    result = copy.deepcopy(spec)
    for old_index, new_index in enumerate(item_map):
        result["initial"][new_index] = spec["initial"][old_index]
    if "lookup" in spec:
        result["lookup"]["links"] = {
            name: item_map[target] if type(target) is int else target
            for name, target in spec["lookup"]["links"].items()}
    return result


def remap_answers(answers, item_map):
    return {tuple(sorted(item_map[i] for i in answer)) for answer in answers}


def rebind_row_hash(row):
    row["row_id"] = digest({key: value for key, value in row.items() if key != "row_id"})


def changed_public_fields(before, after):
    fields = []
    if before["initial"] != after["initial"]:
        fields.append("initial")
    if "lookup" in before:
        for name in ("links", "requested"):
            if before["lookup"][name] != after["lookup"][name]:
                fields.append("lookup." + name)
    if "selection" in before:
        for name in ("k", "target"):
            if before["selection"][name] != after["selection"][name]:
                fields.append("selection." + name)
    return set(fields)


class ConditionSemanticsTests(unittest.TestCase):
    def test_hand_specified_answers_do_not_come_from_the_compiler(self):
        self.assertEqual({(1, 4)}, brute_answers(r_world()))
        self.assertEqual({(1,)}, brute_answers(r_world(shared=True)))
        self.assertEqual({(1, 4)}, brute_answers(s_world()))
        changed = s_world()
        changed["selection"]["target"] = 22
        self.assertEqual({(2, 3)}, brute_answers(changed))
        changed["selection"] = {"k": 3, "target": 35}
        self.assertEqual({(0, 1, 4)}, brute_answers(changed))

    def test_r_depth_bidirectionality_and_real_answer_dependence(self):
        for base in (r_world(), r_world(shared=True)):
            with self.subTest(shared=base == r_world(shared=True)):
                original = copy.deepcopy(base)
                worlds = condition_worlds(base, seed=17)
                self.assertEqual(original, base)
                self.assertEqual({"depth_previous", "depth_next", "endpoint_rebound", "request_rebound"},
                                 {world["name"] for world in worlds})
                self.assertEqual(4, len(worlds))
                before = brute_answers(base)
                first = base["lookup"]["requested"][0]
                for world in worlds:
                    spec = world["spec"]
                    self.assertEqual({"name", "spec", "changed_fields", "effect"}, set(world))
                    self.assertEqual("R", validate_spec(spec))
                    self.assertEqual(base["initial"], spec["initial"])
                    self.assertEqual(len(base["lookup"]["requested"]),
                                     len(spec["lookup"]["requested"]))
                    self.assertTrue(world["changed_fields"])
                    self.assertEqual(changed_public_fields(base, spec), set(world["changed_fields"]))
                    self.assertNotEqual(base, spec)
                    after = brute_answers(spec)
                    self.assertTrue(after)
                    if world["name"] in ("depth_previous", "depth_next"):
                        delta = -1 if world["name"] == "depth_previous" else 1
                        self.assertEqual(requested_depth(base, first) + delta,
                                         requested_depth(spec, first))
                        self.assertEqual("preserve", world["effect"])
                        self.assertEqual(before, after)
                    else:
                        self.assertEqual("change", world["effect"])
                        self.assertTrue(before.isdisjoint(after))
                    for answer in before:
                        accepted = verify(render(spec), format_answer(list(answer)))["accepted"]
                        self.assertEqual(world["effect"] == "preserve", accepted)

    def test_s_target_and_cardinality_shifts_are_solved_and_answer_sensitive(self):
        base = s_world()
        original = copy.deepcopy(base)
        worlds = condition_worlds(base, seed=31)
        self.assertEqual(original, base)
        self.assertEqual({"target_low", "target_high", "k_previous", "k_next"},
                         {world["name"] for world in worlds})
        before = brute_answers(base)
        for world in worlds:
            with self.subTest(condition=world["name"]):
                spec = world["spec"]
                self.assertEqual({"name", "spec", "changed_fields", "effect"}, set(world))
                self.assertEqual("S", validate_spec(spec))
                self.assertEqual(base["initial"], spec["initial"])
                self.assertEqual("change", world["effect"])
                self.assertTrue(world["changed_fields"])
                self.assertEqual(changed_public_fields(base, spec), set(world["changed_fields"]))
                if world["name"] == "target_low":
                    self.assertLess(spec["selection"]["target"], 34)
                    self.assertEqual(2, spec["selection"]["k"])
                elif world["name"] == "target_high":
                    self.assertGreater(spec["selection"]["target"], 34)
                    self.assertEqual(2, spec["selection"]["k"])
                else:
                    self.assertEqual(1 if world["name"] == "k_previous" else 3,
                                     spec["selection"]["k"])
                self.assertTrue(brute_answers(spec))
                self.assertTrue(before.isdisjoint(brute_answers(spec)))
                self.assertFalse(verify(render(spec), '{"answer":[1,4]}')["accepted"])

    def test_witness_choice_accepts_alternatives_and_is_not_always_lexicographic(self):
        spec = {"initial": [0] * 6, "selection": {"k": 2, "target": 0}}
        expected = brute_answers(spec)
        self.assertEqual(15, len(expected))
        selected = set()
        for seed in range(128):
            answer = choose_target(spec, seed)
            self.assertEqual(answer, choose_target(copy.deepcopy(spec), seed))
            self.assertIs(type(answer), list)
            self.assertIn(tuple(sorted(answer)), expected)
            selected.add(tuple(sorted(answer)))
        self.assertGreater(len(selected), 1)
        self.assertNotEqual({min(expected)}, selected)
        for answer in expected:
            self.assertTrue(verify(render(spec), format_answer(list(answer)))["accepted"])
        self.assertFalse(verify(render(spec), '{"answer":[0,0]}')["accepted"])

    def test_unsupported_conditions_fail_instead_of_substituting_easier_ones(self):
        unsolved_r = r_world()
        unsolved_r["lookup"]["requested"] = ["unused"]
        unsolved_s = s_world()
        unsolved_s["selection"]["target"] = 40000
        flat_s = {"initial": [2] * 6, "selection": {"k": 2, "target": 4}}
        composed = r_world()
        composed["selection"] = {"k": 1, "target": 3}
        for spec in (None, unsolved_r, unsolved_s, flat_s, composed):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                condition_worlds(spec, seed=17)
        for spec in (unsolved_r, unsolved_s):
            with self.subTest(target=spec), self.assertRaises(ValueError):
                choose_target(spec, 17)

    def test_registered_boundary_cycles_are_explicit_and_answer_sensitive(self):
        for k, target in ((1, 31), (5, 113)):
            base = s_world()
            base["selection"] = {"k": k, "target": target}
            before = brute_answers(base)
            self.assertTrue(before)
            worlds = {world["name"]: world for world in condition_worlds(base, seed=47)}
            previous_k = 1 + ((k - 2) % 5)
            next_k = 1 + (k % 5)
            self.assertEqual(previous_k, worlds["k_previous"]["spec"]["selection"]["k"])
            self.assertEqual(next_k, worlds["k_next"]["spec"]["selection"]["k"])
            for name in ("k_previous", "k_next"):
                self.assertTrue(before.isdisjoint(brute_answers(worlds[name]["spec"])))

    def test_depth_boundary_cycles_preserve_the_exact_reference_answer(self):
        for depth in (1, 5):
            base = r_world()
            links = base["lookup"]["links"]
            if depth == 1:
                links["q"] = 1
            else:
                links["q_end"] = "q_extra1"
                links["q_extra1"] = "q_extra2"
                links["q_extra2"] = 1
            self.assertEqual(depth, requested_depth(base, "q"))
            worlds = {world["name"]: world for world in condition_worlds(base, seed=53)}
            self.assertEqual(1 + ((depth - 2) % 5),
                             requested_depth(worlds["depth_previous"]["spec"], "q"))
            self.assertEqual(1 + (depth % 5),
                             requested_depth(worlds["depth_next"]["spec"], "q"))
            for name in ("depth_previous", "depth_next"):
                self.assertEqual(brute_answers(base), brute_answers(worlds[name]["spec"]))


class SurfaceAndFamilyTests(unittest.TestCase):
    def test_rotations_move_values_references_and_all_alternative_answers(self):
        repeated = {"initial": [1, 1, 2, 2, 3, 3],
                    "selection": {"k": 2, "target": 4}}
        composed = r_world()
        composed["selection"] = {"k": 1, "target": 3}
        for base in (r_world(), repeated, composed):
            original = copy.deepcopy(base)
            n = len(base["initial"])
            for template in range(4):
                for rotation in range(n):
                    with self.subTest(skill=validate_spec(base), template=template, rotation=rotation):
                        view = surface_view(base, template, rotation)
                        self.assertEqual({"spec", "messages", "item_map"}, set(view))
                        item_map = view["item_map"]
                        self.assertEqual(list(range(n)), sorted(item_map))
                        self.assertTrue(all(type(i) is int for i in item_map))
                        self.assertEqual(independent_reindex(base, item_map), view["spec"])
                        self.assertEqual(view["spec"], parse_visible(view["messages"]))
                        self.assertEqual(remap_answers(brute_answers(base), item_map),
                                         brute_answers(view["spec"]))
                        for answer in brute_answers(view["spec"]):
                            self.assertTrue(verify(view["messages"], format_answer(list(answer)))["accepted"])
            self.assertEqual(original, base)

    def test_full_rotation_orbit_balances_each_source_witness_position(self):
        for spec, answer in ((r_world(), [1, 4]), (s_world(), [1, 4])):
            counts = Counter()
            for rotation in range(len(spec["initial"])):
                view = surface_view(spec, 0, rotation)
                moved = [view["item_map"][index] for index in answer]
                self.assertTrue(verify(view["messages"], format_answer(moved))["accepted"])
                counts.update(moved)
            self.assertEqual({i: len(answer) for i in range(len(spec["initial"]))}, dict(counts))

    def test_family_keys_ignore_alias_item_reindexing_and_irrelevant_values(self):
        base = r_world(shared=True)
        renamed = copy.deepcopy(base)
        names = {name: "renamed_" + str(i) for i, name in enumerate(base["lookup"]["links"])}
        names["missing"] = "absent"
        renamed["lookup"] = {
            "links": {names[name]: names[target] if type(target) is str else target
                      for name, target in base["lookup"]["links"].items()},
            "requested": [names[name] for name in reversed(base["lookup"]["requested"])]}
        renamed["initial"] = [-100, 100, 0, 9, 9, -9]
        renamed["lookup"]["requested"].append(renamed["lookup"]["requested"][0])
        self.assertEqual(family_key(base), family_key(renamed))
        for rotation in range(6):
            self.assertEqual(family_key(base), family_key(surface_view(base, 0, rotation)["spec"]))
        selected = s_world()
        for rotation in range(6):
            self.assertEqual(family_key(selected),
                             family_key(surface_view(selected, 0, rotation)["spec"]))
        changed = s_world()
        changed["selection"]["target"] = 22
        self.assertNotEqual(family_key(selected), family_key(changed))

    def test_family_key_records_shared_alias_usage_beyond_terminal_equality(self):
        shared = {"initial": [1, 3, 7, 15, 31, 63],
                  "lookup": {"links": {"q": "tail", "r": "tail", "tail": 1},
                             "requested": ["q", "r"]}}
        separate = copy.deepcopy(shared)
        separate["lookup"]["links"] = {"q": "q_tail", "r": "r_tail", "q_tail": 1, "r_tail": 1}
        self.assertEqual(brute_answers(shared), brute_answers(separate))
        self.assertNotEqual(family_key(shared), family_key(separate))

    def test_malformed_surface_arguments_are_rejected(self):
        for template, rotation in ((True, 0), (4, 0), (-1, 0), (0, True), (0, -1)):
            with self.subTest(template=template, rotation=rotation), self.assertRaises(ValueError):
                surface_view(s_world(), template, rotation)


class CrossedCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = build_balanced_plan(seed=20261003, parent_count=16)
        cls.records = compile_balanced_records(cls.plan)

    def test_plan_and_records_are_reproducible_detached_and_content_hashed(self):
        self.assertEqual(self.plan, build_balanced_plan(seed=20261003, parent_count=16))
        self.assertEqual(self.records, compile_balanced_records(self.plan))
        self.assertEqual(self.plan["sha256"],
                         digest({key: value for key, value in self.plan.items() if key != "sha256"}))
        self.assertEqual(set(self.plan["parents"]), set(self.plan["assignments"]))
        self.assertEqual(set(self.plan["parents"]), set(self.plan["condition_plans"]))
        self.assertIs(type(self.plan["attempts"]), list)
        for pid, parent in self.plan["parents"].items():
            source = self.plan["sources"][parent["source_ids"][0]]
            self.assertEqual(source["sha256"], digest(source["content"]))
            self.assertEqual(parent["spec"], source["content"]["base_spec"])
            self.assertEqual(self.plan["condition_plans"][pid], source["content"]["conditions"])
            for world in self.plan["condition_plans"][pid]:
                self.assertEqual(changed_public_fields(parent["spec"], world["spec"]),
                                 set(world["changed_fields"]))
        copied = copy.deepcopy(self.records)
        copied[0]["messages"][1]["content"] = "outside mutation"
        self.assertNotEqual(copied[0]["messages"], self.records[0]["messages"])

    def test_default_plan_condition_metadata_reports_only_actual_field_changes(self):
        plan = build_balanced_plan()
        for pid, parent in plan["parents"].items():
            for world in plan["condition_plans"][pid]:
                with self.subTest(parent=pid, condition=world["name"]):
                    self.assertEqual(changed_public_fields(parent["spec"], world["spec"]),
                                     set(world["changed_fields"]))

    def test_every_condition_rendering_cell_has_equal_parent_exposure(self):
        grouped = defaultdict(list)
        for row in self.records:
            grouped[(row["parent_id"], row["arm"])].append(row)
        self.assertEqual(32 * len(self.plan["parents"]), len(self.records))
        for pid in self.plan["parents"]:
            for arm in ("M1", "M2"):
                rows = grouped[(pid, arm)]
                self.assertEqual(16, len(rows))
                self.assertEqual({(i, t): 1 for i in range(4) for t in range(4)},
                                 dict(Counter((r["condition_index"], r["template"]) for r in rows)))
                for row in rows:
                    self.assertEqual((row["condition_index"] * 4 + row["template"]) % len(row["spec"]["initial"]),
                                     row["rotation"])
                    original = (self.plan["parents"][pid]["spec"] if arm == "M1" else
                                self.plan["condition_plans"][pid][row["condition_index"]]["spec"])
                    self.assertEqual(independent_reindex(original, row["item_map"]), row["spec"])
                    if arm == "M2":
                        self.assertEqual(self.plan["condition_plans"][pid][row["condition_index"]]["name"],
                                         row["condition"])

    def test_targets_are_valid_in_original_and_visible_space_without_prefix_requirement(self):
        original_witnesses = defaultdict(set)
        for row in self.records:
            original = (self.plan["parents"][row["parent_id"]]["spec"] if row["arm"] == "M1" else
                        self.plan["condition_plans"][row["parent_id"]][row["condition_index"]]["spec"])
            candidate = parse_answer(row["target"], len(row["spec"]["initial"]))
            self.assertIs(type(candidate), list)
            inverse = {new: old for old, new in enumerate(row["item_map"])}
            source_answer = tuple(sorted(inverse[index] for index in candidate))
            original_witnesses[(row["parent_id"], row["arm"], row["condition_index"])].add(source_answer)
            self.assertIn(source_answer, brute_answers(original))
            self.assertIn(tuple(sorted(candidate)), brute_answers(row["spec"]))
            self.assertEqual(row["spec"], parse_visible(row["messages"]))
            self.assertEqual(verify(row["messages"], row["target"]), row["verification"])
            self.assertTrue(row["verification"]["accepted"])
            self.assertEqual(validate_spec(row["spec"]), row["skill"])
            self.assertIn(row["skill"], ("R", "S"))
        # Re-solving lexicographically after each rotation can produce valid
        # targets yet systematically favor low visible indices. Freeze each
        # original-space witness before rendering and move that same witness.
        self.assertEqual({1}, {len(answers) for answers in original_witnesses.values()})

    def test_family_source_and_planned_condition_footprints_never_cross_splits(self):
        ownership = {}
        for pid, parent in self.plan["parents"].items():
            assignment = self.plan["assignments"][pid]
            self.assertIn(assignment["split"], ("train", "dev", "eval_candidate"))
            tokens = [("component", assignment["component_id"]), ("family", parent["family_id"]),
                      ("shape", family_key(parent["spec"]))]
            tokens.extend(("shape", family_key(world["spec"]))
                          for world in self.plan["condition_plans"][pid])
            for sid in parent["source_ids"]:
                tokens.extend((("source", sid), ("content", self.plan["sources"][sid]["sha256"])))
            for token in tokens:
                self.assertTrue(token not in ownership or ownership[token] == assignment["split"])
                ownership[token] = assignment["split"]
        for row in self.records:
            self.assertEqual(self.plan["assignments"][row["parent_id"]]["split"], row["split"])
            self.assertEqual(self.plan["assignments"][row["parent_id"]]["component_id"], row["component_id"])

    def test_balance_report_matches_independent_counts_and_declares_residual_strata(self):
        report = audit_balance(self.records)
        self.assertEqual(len(self.records), report["record_count"])
        self.assertTrue(report["full_crossing"])
        self.assertTrue(report["matched_parent_exposures"])
        self.assertTrue(report["no_RS_in_training"])
        for arm in ("M1", "M2"):
            expected = Counter(r["parent_id"] for r in self.records if r["arm"] == arm)
            self.assertEqual(dict(expected), report["parent_exposures"][arm])
            for skill in ("R", "S"):
                rows = [r for r in self.records if r["arm"] == arm and r["skill"] == skill]
                positions = Counter(str(index) for row in rows
                                    for index in parse_answer(row["target"], len(row["spec"]["initial"])))
                self.assertEqual(dict(positions), report["answer_position_counts_by_arm_skill"][arm + ":" + skill])
                strata = report["strata_by_arm_skill"][arm + ":" + skill]
                expected_strata = {
                    "n": Counter(str(len(row["spec"]["initial"])) for row in rows),
                    "answer_size": Counter(str(len(parse_answer(row["target"], len(row["spec"]["initial"]))))
                                           for row in rows),
                    "solution_multiplicity": Counter(str(len(brute_answers(row["spec"]))) for row in rows)}
                if skill == "S":
                    expected_strata["k"] = Counter(str(row["spec"]["selection"]["k"]) for row in rows)
                else:
                    expected_strata["requested_count"] = Counter(str(len(row["spec"]["lookup"]["requested"]))
                                                                for row in rows)
                    expected_strata["first_requested_depth"] = Counter(str(requested_depth(
                        row["spec"], row["spec"]["lookup"]["requested"][0])) for row in rows)
                    expected_strata["max_requested_depth"] = Counter(str(max(
                        requested_depth(row["spec"], name) for name in row["spec"]["lookup"]["requested"]))
                        for row in rows)
                self.assertEqual({key: dict(value) for key, value in expected_strata.items()}, strata)
        self.assertEqual(dict(Counter(r["arm"] + ":" + r["condition"] + ":" + str(r["template"])
                                      for r in self.records)), report["condition_template_counts"])

    def test_audit_detects_missing_crossing_and_exposure_cells(self):
        incomplete = self.records[1:]
        report = audit_balance(incomplete)
        self.assertFalse(report["full_crossing"])
        self.assertFalse(report["matched_parent_exposures"])

    def test_record_and_visible_gold_forgery_fail_even_with_a_rebound_row_hash(self):
        original = self.records[0]
        mutations = [("arm", "M0"), ("skill", "RS"), ("template", True), ("rotation", True),
                     ("condition_index", True), ("split", "protected_final"),
                     ("component_id", "0" * 64), ("source_hash", "0" * 64),
                     ("base_spec_hash", "0" * 64), ("target", '{"answer":"UNKNOWN"}'),
                     ("item_map", [0] * len(original["spec"]["initial"])),
                     ("verification", {"accepted": True})]
        for field, value in mutations:
            rows = copy.deepcopy(self.records)
            rows[0][field] = value
            rebind_row_hash(rows[0])
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_balanced_records(rows, self.plan)
        rows = copy.deepcopy(self.records)
        rows[0]["teacher_hint"] = "private gold"
        rebind_row_hash(rows[0])
        with self.assertRaises(ValueError):
            validate_balanced_records(rows, self.plan)

    def test_forged_atomic_label_cannot_make_composition_trainable(self):
        rows = copy.deepcopy(self.records)
        row = next(row for row in rows if row["skill"] == "R")
        resolved = next(iter(brute_answers(row["spec"])))
        row["spec"]["selection"] = {"k": 1, "target": row["spec"]["initial"][resolved[0]]}
        row["messages"] = render(row["spec"], row["template"])
        row["target"] = format_answer([resolved[0]])
        row["verification"] = verify(row["messages"], row["target"])
        row["split"] = "train"
        row["skill"] = "R"
        rebind_row_hash(row)
        self.assertTrue(row["verification"]["accepted"])
        self.assertEqual("RS", validate_spec(parse_visible(row["messages"])))
        self.assertFalse(audit_balance(rows)["no_RS_in_training"])
        with self.assertRaises(ValueError):
            validate_balanced_records(rows, self.plan)

    def test_rebound_plan_hash_does_not_hide_world_source_or_split_changes(self):
        pid = next(iter(self.plan["parents"]))
        sid = self.plan["parents"][pid]["source_ids"][0]
        for scope in ("source", "condition", "assignment", "parent", "unknown"):
            plan = copy.deepcopy(self.plan)
            if scope == "source":
                plan["sources"][sid]["content"]["base_spec"]["initial"][0] += 1
                plan["sources"][sid]["sha256"] = digest(plan["sources"][sid]["content"])
            elif scope == "condition":
                plan["condition_plans"][pid][0]["spec"]["initial"][0] += 1
            elif scope == "assignment":
                plan["assignments"][pid]["split"] = "protected_final"
            elif scope == "parent":
                plan["parents"][pid]["spec"]["initial"][0] += 1
            else:
                plan["teacher_hint"] = "private gold"
            plan["sha256"] = digest({key: value for key, value in plan.items() if key != "sha256"})
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                compile_balanced_records(plan)

    def test_invalid_plan_arguments_and_partial_record_collections_fail_closed(self):
        for count in (None, True, 0, 257):
            with self.subTest(count=count), self.assertRaises(ValueError):
                build_balanced_plan(parent_count=count)
        for seed in (True, -1, 2**32, None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                build_balanced_plan(seed=seed, parent_count=16)
        for rows in (None, {}, [], self.records[1:], self.records + [self.records[0]]):
            with self.subTest(rows=type(rows).__name__), self.assertRaises(ValueError):
                validate_balanced_records(rows, self.plan)


if __name__ == "__main__":
    unittest.main()

"""CPU compiler provenance, leakage, condition dependence and export audits.

These checks use local finite fixtures only.  No tokenizer budget, model
qualification, training result, protected bank, network or GPU is involved.
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

from rs_contract import digest, format_answer, parse_visible, render, validate_spec
from rs_verifier import solve_visible, verify
from task_compiler import (compile_records, condition_variants, export_training,
                           fixture_catalog, register_parents, structural_key,
                           validate_records)


def source(spec, sid):
    content = {"base_spec": copy.deepcopy(spec)}
    return {"source_id": sid, "version": 1, "content": content,
            "sha256": digest(content), "provenance": "local_test_fixture",
            "license": "repository_terms", "attribution": "CPU audit fixture"}


def parent(spec, pid, family=None, extra_sources=()):
    return {"parent_id": pid, "family_id": family or f"family_{pid}",
            "source_ids": [f"source_{pid}", *extra_sources], "spec": copy.deepcopy(spec)}


def selection(offset=0):
    values = [offset + j for j in (1, 2, 4, 8, 16, 32)]
    return {"initial": values, "selection": {"k": 2, "target": values[0] + values[1]}}


def reference():
    return {"initial": [1, 2, 4, 8, 16, 32],
            "lookup": {"links": {"head": "leaf", "leaf": 0, "other": 5,
                                   "unused": "unused", "broken": "absent"},
                       "requested": ["head", "other", "head"]}}


def rs_catalog(training):
    spec = reference()
    spec["selection"] = {"k": 1, "target": 1}
    sources = [source(spec, "source_rs")]
    parents = [parent(spec, "rs")]
    for seed in range(100):
        catalog = register_parents(sources, parents, seed)
        if (catalog["assignments"]["rs"]["split"] == "train") == training:
            return catalog
    raise AssertionError("deterministic split search failed")


def rebind_row_id(row):
    row["row_id"] = digest({key: value for key, value in row.items() if key != "row_id"})


class CatalogIsolationTests(unittest.TestCase):
    def test_fixture_catalog_is_reproducible_balanced_and_detached(self):
        catalog = fixture_catalog()
        self.assertEqual(catalog, fixture_catalog())
        self.assertNotEqual(catalog["sha256"], fixture_catalog(seed=1004)["sha256"])
        signatures = Counter(validate_spec(p["spec"]) for p in catalog["parents"].values())
        self.assertEqual({"R": 24, "S": 24}, signatures)
        self.assertEqual({"train", "dev", "eval_fixture"},
                         {a["split"] for a in catalog["assignments"].values()})
        original = fixture_catalog(parent_count=2)
        pid = next(iter(original["parents"]))
        sid = original["parents"][pid]["source_ids"][0]
        original["parents"][pid]["spec"]["initial"][0] += 1
        self.assertNotEqual(original["parents"][pid]["spec"],
                            original["sources"][sid]["content"]["base_spec"])

    def test_family_and_source_dependencies_are_transitively_connected(self):
        specs = [selection(i * 9) for i in range(4)]
        sources = [source(spec, f"source_p{i}") for i, spec in enumerate(specs)]
        shared = source(selection(60), "shared_source")
        duplicate_a = source(selection(65), "identical_content_a")
        duplicate_b = copy.deepcopy(duplicate_a)
        duplicate_b["source_id"] = "identical_content_b"
        sources.extend((shared, duplicate_a, duplicate_b))
        parents = [parent(specs[0], "p0", "same_family"),
                   parent(specs[1], "p1", "same_family", ["shared_source"]),
                   parent(specs[2], "p2", extra_sources=["shared_source", "identical_content_a"]),
                   parent(specs[3], "p3", extra_sources=["identical_content_b"])]
        for seed in (0, 1, 1003):
            catalog = register_parents(sources, parents, seed)
            self.assertEqual(1, len({a["component_id"] for a in catalog["assignments"].values()}))
            self.assertEqual(1, len({a["split"] for a in catalog["assignments"].values()}))
            reversed_catalog = register_parents(list(reversed(sources)), list(reversed(parents)), seed)
            self.assertEqual(catalog, reversed_catalog)

    def test_alias_renaming_repeated_requests_and_distractors_are_not_fresh_worlds(self):
        original = reference()
        renamed = copy.deepcopy(original)
        renamed["initial"] = [-100, 100, 0, 0, 0, 0]
        renamed["lookup"] = {"links": {"start": "end", "end": 0, "second": 5,
                                          "noise": 3},
                               "requested": ["second", "start"]}
        self.assertEqual(structural_key(original), structural_key(renamed))
        specs = [original, renamed]
        catalog = register_parents([source(s, f"source_p{i}") for i, s in enumerate(specs)],
                                   [parent(s, f"p{i}") for i, s in enumerate(specs)])
        self.assertEqual(catalog["assignments"]["p0"], catalog["assignments"]["p1"])
        selected = selection()
        permuted = copy.deepcopy(selected)
        permuted["initial"].reverse()
        self.assertEqual(structural_key(selected), structural_key(permuted))

    def test_condition_world_collisions_are_isolated_before_compilation(self):
        # A parent can be the declared target intervention of another parent.
        base = selection()
        changed = dict((name, spec) for name, spec, _, _ in condition_variants(base))["target_changed"]
        catalog = register_parents([source(base, "source_a"), source(changed, "source_b")],
                                   [parent(base, "a"), parent(changed, "b")])
        self.assertEqual(catalog["assignments"]["a"], catalog["assignments"]["b"])
        validate_records(compile_records(catalog), catalog)

    def test_duplicate_source_ids_parent_ids_and_missing_dependencies_are_rejected(self):
        base = selection()
        sources, parents = [source(base, "source_a")], [parent(base, "a")]
        invalid = [(sources * 2, parents), (sources, parents * 2),
                   (sources, [dict(parents[0], source_ids=["absent"])]),
                   (sources, [dict(parents[0], source_ids=["source_a", "source_a"])])]
        for source_list, parent_list in invalid:
            with self.subTest(sources=len(source_list), parents=parent_list), self.assertRaises(ValueError):
                register_parents(source_list, parent_list)

    def test_registry_unknown_fields_and_malformed_dependencies_fail_with_value_error(self):
        base = selection()
        valid_sources, valid_parents = [source(base, "source_a")], [parent(base, "a")]
        invalid_sources = [None, [], {}, dict(valid_sources[0], teacher_hint="gold"),
                           dict(valid_sources[0], version=True),
                           dict(valid_sources[0], sha256="0" * 64),
                           dict(valid_sources[0], content={"bad": object()})]
        for bad_source in invalid_sources:
            with self.subTest(source=repr(bad_source)), self.assertRaises(ValueError):
                register_parents([bad_source], valid_parents)
        for refs in (None, [], "source_a", [[]], [{}], [1], [True], ["absent"]):
            bad_parent = dict(valid_parents[0], source_ids=refs)
            with self.subTest(refs=refs), self.assertRaises(ValueError):
                register_parents(valid_sources, [bad_parent])
        for bad_parent in (None, [], dict(valid_parents[0], teacher_hint="gold"),
                           dict(valid_parents[0], spec=None)):
            with self.subTest(parent=bad_parent), self.assertRaises(ValueError):
                register_parents(valid_sources, [bad_parent])

    def test_fixture_arguments_are_bounded_and_boolean_is_not_integer(self):
        for count in (None, True, 1, 3, 0, 258, 2.0):
            with self.subTest(count=count), self.assertRaises(ValueError):
                fixture_catalog(parent_count=count)
        for seed in (True, -1, 2**32, None, 1.0):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                fixture_catalog(seed=seed)

    def test_registered_parent_requires_a_matching_grounded_source_world(self):
        base = selection()
        mismatch = selection(10)
        with self.assertRaises(ValueError):
            register_parents([source(base, "source_a")], [parent(mismatch, "a")])
        # Merely rebinding a content hash does not turn unrelated content into
        # the explicit local base world required by this fixture compiler.
        changed_source = source(base, "source_a")
        changed_source["content"] = {"unrelated": mismatch}
        changed_source["sha256"] = digest(changed_source["content"])
        with self.assertRaises(ValueError):
            register_parents([changed_source], [parent(base, "a")])


class ConditionDependenceTests(unittest.TestCase):
    def test_atomic_variants_have_real_answer_dependence_and_preserved_witnesses(self):
        for base in (reference(), selection()):
            before = solve_visible(render(base))
            before_answers = {tuple(a) for a in before["solutions"]}
            variants = condition_variants(base)
            self.assertEqual(4, len(variants))
            self.assertEqual(base, variants[0][1])
            self.assertEqual(([], "preserve"), (variants[0][2], variants[0][3]))
            for name, changed, fields, effect in variants:
                with self.subTest(skill=validate_spec(base), variant=name):
                    after = solve_visible(render(changed))
                    self.assertEqual("solved", after["status"])
                    if name != "baseline":
                        self.assertTrue(fields)
                        self.assertNotEqual(base, changed)
                    after_answers = {tuple(a) for a in after["solutions"]}
                    if effect == "change":
                        self.assertTrue(before_answers.isdisjoint(after_answers))
                        self.assertTrue(all(not verify(render(changed), format_answer(a))["accepted"]
                                            for a in before["solutions"]))
                    else:
                        # Preserve means previous correct witnesses remain valid;
                        # adding an item can legitimately add further solutions.
                        self.assertTrue(before_answers <= after_answers)
            self.assertEqual(base, variants[0][1])

    def test_item_added_preserves_witnesses_without_claiming_solution_set_invariance(self):
        base = {"initial": [0, 1, 2, 3, 4, 5], "selection": {"k": 1, "target": 0}}
        variants = {name: spec for name, spec, _, _ in condition_variants(base)}
        self.assertEqual([[0]], solve_visible(render(base))["solutions"])
        self.assertEqual([[0], [6]], solve_visible(render(variants["item_added"]))["solutions"])

    def test_amplification_requires_atomic_solved_and_supported_interventions(self):
        unsolved_r = reference()
        unsolved_r["lookup"]["requested"] = ["broken"]
        unsolved_s = selection()
        unsolved_s["selection"]["target"] = 40000
        composed = reference()
        composed["selection"] = {"k": 1, "target": 1}
        no_unused_endpoint = {"initial": [1, 2, 3, 4, 5, 6],
                              "lookup": {"links": {f"a{i}": i for i in range(6)},
                                         "requested": [f"a{i}" for i in range(6)]}}
        identical_values = {"initial": [2] * 6, "selection": {"k": 2, "target": 4}}
        for spec in (None, unsolved_r, unsolved_s, composed, no_unused_endpoint, identical_values):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                condition_variants(spec)


class CompilationBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = fixture_catalog(parent_count=12)
        cls.records = compile_records(cls.catalog)

    def test_compilation_is_reproducible_and_arms_match_parent_and_skill_allocations(self):
        self.assertEqual(self.records, compile_records(self.catalog))
        counts = {arm: Counter((r["parent_id"], r["split"], r["skill"])
                              for r in self.records if r["arm"] == arm)
                  for arm in ("M1", "M2")}
        self.assertEqual(counts["M1"], counts["M2"])
        self.assertEqual({4}, set(counts["M1"].values()))
        grouped = defaultdict(list)
        for row in self.records:
            grouped[row["parent_id"]].append(row)
            self.assertTrue(verify(row["messages"], row["candidate"])["accepted"])
            self.assertEqual(row["spec"], parse_visible(row["messages"]))
        for pid, rows in grouped.items():
            surface = [r for r in rows if r["arm"] == "M1"]
            self.assertEqual({0, 1, 2, 3}, {r["variant"]["template"] for r in surface})
            self.assertEqual(4, len({digest(r["messages"]) for r in surface}))
            self.assertEqual({digest(self.catalog["parents"][pid]["spec"])},
                             {r["spec_hash"] for r in surface})
            self.assertEqual(1, len({r["candidate"] for r in surface}))
            self.assertEqual(1, len({digest(r["verification"]["outcome"]) for r in surface}))

    def test_exports_are_atomic_train_only_and_contain_only_visible_messages_and_target(self):
        for arm in ("M1", "M2"):
            exported = export_training(self.records, self.catalog, arm)
            expected = [r for r in self.records if r["arm"] == arm and r["split"] == "train"]
            self.assertEqual(len(expected), len(exported))
            self.assertTrue(exported)
            for row, original in zip(exported, expected):
                self.assertEqual({"messages", "target"}, set(row))
                self.assertEqual(original["messages"], row["messages"])
                self.assertEqual(original["candidate"], row["target"])
                self.assertIn(validate_spec(parse_visible(row["messages"])), ("R", "S"))
                self.assertTrue(verify(row["messages"], row["target"])["accepted"])
            exported[0]["messages"][1]["content"] = "changed outside compiler"
            self.assertNotEqual(exported[0]["messages"], expected[0]["messages"])
        for arm in ("M0", "EVAL_RS", None, [], "M3"):
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                export_training(self.records, self.catalog, arm)

    def test_duplicate_records_unknown_fields_and_bad_containers_are_rejected(self):
        for rows in (None, {}, [], [None], [copy.deepcopy(self.records[0])] * 2):
            with self.subTest(rows=repr(rows)[:120]), self.assertRaises(ValueError):
                validate_records(rows, self.catalog)
        original = self.records[0]
        for scope in ("row", "variant", "spec", "message"):
            row = copy.deepcopy(original)
            if scope == "row":
                row["teacher_hint"] = "gold"
            elif scope == "variant":
                row["variant"]["teacher_hint"] = "gold"
            elif scope == "spec":
                row["spec"]["teacher_hint"] = "gold"
            else:
                row["messages"][1]["teacher_hint"] = "gold"
            rebind_row_id(row)
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                validate_records([row], self.catalog)

    def test_record_metadata_world_candidate_and_messages_cannot_be_rebound(self):
        original = self.records[0]
        mutations = [("source_ids", ["absent"]), ("source_hashes", ["0" * 64]),
                     ("parent_spec_hash", "0" * 64), ("spec_hash", "0" * 64),
                     ("component_id", "0" * 64), ("family_id", "unrelated_family"),
                     ("parent_id", "absent"), ("split", "invalid_split"),
                     ("seed", original["seed"] + 1), ("generator", "other_generator"),
                     ("skill", "RS"), ("schema_version", True),
                     ("candidate", '{"answer":"UNKNOWN"}'), ("verification", {"accepted": True}),
                     ("messages", None), ("spec", None), ("variant", []), ("arm", "M0")]
        for field, value in mutations:
            row = copy.deepcopy(original)
            row[field] = value
            rebind_row_id(row)
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_records([row], self.catalog)
        row = copy.deepcopy(original)
        row["spec"]["initial"][0] += 1
        row["spec_hash"] = digest(row["spec"])
        row["messages"] = render(row["spec"], row["variant"]["template"])
        rebind_row_id(row)
        with self.assertRaises(ValueError):
            validate_records([row], self.catalog)

    def test_boolean_integer_substitution_cannot_reuse_stale_binding(self):
        catalog = fixture_catalog(seed=1, parent_count=4)
        original = compile_records(catalog)[0]
        self.assertEqual(1, original["seed"])
        for field in ("seed", "accepted", "index"):
            row = copy.deepcopy(original)
            if field == "seed":
                row["seed"] = True
            elif field == "accepted":
                row["verification"]["accepted"] = 1
            else:
                # Choose an index 0 or 1 so Python equality would collapse its
                # boolean substitute while canonical JSON keeps them distinct.
                for index_row in compile_records(catalog):
                    solutions = index_row["verification"]["outcome"]["solutions"]
                    match = next(((i, j) for i, solution in enumerate(solutions)
                                  for j, value in enumerate(solution) if value in (0, 1)), None)
                    if match is not None:
                        row = copy.deepcopy(index_row)
                        i, j = match
                        value = row["verification"]["outcome"]["solutions"][i][j]
                        row["verification"]["outcome"]["solutions"][i][j] = bool(value)
                        break
                else:
                    self.fail("regression fixture lacks a zero/one solution index")
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_records([row], catalog)

    def test_non_json_record_values_and_invalid_unicode_fail_with_value_error(self):
        row = copy.deepcopy(self.records[0])
        row["verification"] = {"accepted": object()}
        with self.assertRaises(ValueError):
            validate_records([row], self.catalog)
        row = copy.deepcopy(self.records[0])
        row["messages"][1]["content"] = "\ud800"
        with self.assertRaises(ValueError):
            validate_records([row], self.catalog)
        catalog = copy.deepcopy(self.catalog)
        pid = next(iter(catalog["parents"]))
        catalog["assignments"][pid]["split"] = object()
        with self.assertRaises(ValueError):
            compile_records(catalog)

    def test_catalog_source_parent_hash_split_and_unknown_metadata_mutations_rejected(self):
        pid = next(iter(self.catalog["parents"]))
        sid = self.catalog["parents"][pid]["source_ids"][0]
        for scope in ("source_content", "source_hash", "source_unknown", "parent_spec",
                      "parent_sources", "parent_unknown", "assignment", "catalog_hash", "catalog_unknown"):
            catalog = copy.deepcopy(self.catalog)
            if scope == "source_content":
                catalog["sources"][sid]["content"]["injected"] = "new source"
            elif scope == "source_hash":
                catalog["sources"][sid]["sha256"] = "0" * 64
            elif scope == "source_unknown":
                catalog["sources"][sid]["teacher_hint"] = "gold"
            elif scope == "parent_spec":
                catalog["parents"][pid]["spec"]["initial"][0] += 1
            elif scope == "parent_sources":
                catalog["parents"][pid]["source_ids"] = ["absent"]
            elif scope == "parent_unknown":
                catalog["parents"][pid]["teacher_hint"] = "gold"
            elif scope == "assignment":
                catalog["assignments"][pid]["split"] = "invalid_split"
            elif scope == "catalog_hash":
                catalog["sha256"] = "0" * 64
            else:
                catalog["teacher_hint"] = "gold"
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                compile_records(catalog)

    def test_rs_composition_is_held_out_and_export_rejects_forged_atomic_labels(self):
        heldout = rs_catalog(training=False)
        rows = compile_records(heldout)
        self.assertEqual(1, len(rows))
        self.assertEqual("EVAL_RS", rows[0]["arm"])
        self.assertEqual("RS", rows[0]["skill"])
        self.assertNotEqual("train", rows[0]["split"])
        self.assertEqual([], export_training(rows, heldout, "M1"))
        training = rs_catalog(training=True)
        with self.assertRaises(ValueError):
            compile_records(training)
        # Rebind every public lineage field after forging a training arm.  A
        # checksum alone cannot protect RS isolation against such an attacker.
        for catalog in (heldout, training):
            for skill in ("RS", "R", "S"):
                row = copy.deepcopy(rows[0])
                row.update(catalog["assignments"]["rs"])
                row["seed"] = catalog["seed"]
                row["arm"] = "M1"
                row["skill"] = skill
                row["variant"] = {"kind": "surface", "name": "surface_0", "template": 0,
                                  "changed_fields": [], "answer_effect": "preserve"}
                rebind_row_id(row)
                with self.subTest(split=row["split"], skill=skill), self.assertRaises(ValueError):
                    validate_records([row], catalog)
                with self.subTest(export_split=row["split"], skill=skill), self.assertRaises(ValueError):
                    export_training([row], catalog, "M1")


if __name__ == "__main__":
    unittest.main()

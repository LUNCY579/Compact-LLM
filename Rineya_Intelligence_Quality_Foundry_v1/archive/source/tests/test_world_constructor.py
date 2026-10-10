"""Pre-birth tests only: allocation, gate validation/denial, source primitives.

No test constructs a full dataset world, writes a registry, renders, or solves.
"""
import ast
from collections import Counter
import copy
import hashlib
import itertools
from pathlib import Path
import random
import unittest
from unittest.mock import patch

from mechanisms.distributed_dossier_v1 import world_constructor as constructor


class AllocationTests(unittest.TestCase):
    def test_complete_preassigned_allocation(self):
        slots = constructor.allocation(90210)
        self.assertEqual(len(slots), 54)
        self.assertEqual(len({s.parent_id for s in slots}), 54)
        self.assertEqual(len({s.reservation_id for s in slots}), 54)
        self.assertEqual(len({s.content_family_id for s in slots}), 54)
        self.assertEqual(len({s.stream_key for s in slots}), 54)
        self.assertEqual(Counter(s.role for s in slots), {"PROSPECTIVE_TRAIN": 36, "EXPOSED_DEV": 18})
        self.assertEqual(Counter(s.intended_status for s in slots), {"DETERMINATE": 36, "UNDERDETERMINED": 9, "INCONSISTENT": 9})
        self.assertEqual(Counter(s.format for s in slots), {"JSON": 18, "LINE": 18, "TOKEN": 18})
        for slot in slots:
            self.assertEqual(slot.format, constructor.FORMATS[(slot.mechanism_index + slot.structure_index + (slot.structure_index + 2) * (slot.level - 1) + slot.case_index) % 3])
            self.assertEqual(slot.role, "EXPOSED_DEV" if slot.structure_index == 2 else "PROSPECTIVE_TRAIN")
            self.assertEqual(set(slot.cell_metadata()), {"mechanism", "structure_id", "structure_index", "level", "case_index", "role", "format"})
            self.assertTrue(constructor.schema.ID.fullmatch(slot.parent_id))
            self.assertTrue(constructor.schema.REGISTRY_ID.fullmatch(slot.content_family_id))

    def test_status_formats_within_mechanism_and_train(self):
        slots = constructor.allocation(5)
        for mechanism in constructor.MECHANISMS:
            for status in ("UNDERDETERMINED", "INCONSISTENT"):
                subset = [s for s in slots if s.mechanism == mechanism and s.intended_status == status]
                self.assertEqual(Counter(s.format for s in subset), {"JSON": 1, "LINE": 1, "TOKEN": 1})
                train = [s.format for s in subset if s.role == "PROSPECTIVE_TRAIN"]
                self.assertEqual(len(set(train)), 2)

    def test_recipe_slot_not_status_siblings(self):
        slots = constructor.allocation(77)
        for mi, si, level in itertools.product(range(3), range(3), (1, 2)):
            cell = [s for s in slots if (s.mechanism_index, s.structure_index, s.level) == (mi, si, level)]
            self.assertEqual([s.case_index for s in cell], [0, 1, 2])
            self.assertEqual(len({s.content_family_id for s in cell}), 3)
            self.assertEqual(len({s.stream_key for s in cell}), 3)
        self.assertEqual(constructor.allocation(77), slots)
        self.assertNotEqual(constructor.allocation(78)[0].parent_id, slots[0].parent_id)

    def test_global_random_is_untouched(self):
        before = random.getstate()
        constructor.allocation(-10)
        a = constructor.isolated_rng(12, "cell", "facts")
        x = a.getrandbits(256)
        constructor.isolated_rng(12, "unrelated", "labels").getrandbits(256)
        self.assertEqual(constructor.isolated_rng(12, "cell", "facts").getrandbits(256), x)
        self.assertNotEqual(constructor.isolated_rng(12, "other", "facts").getrandbits(256), x)
        self.assertEqual(random.getstate(), before)
        for seed in (True, 1.0, "7", None):
            with self.assertRaises(ValueError): constructor.allocation(seed)


class BirthGateTests(unittest.TestCase):
    def test_missing_authorization_never_reaches_gate_or_source(self):
        with patch.object(constructor, "verify_birth_gates", side_effect=AssertionError("gate entered")), patch.object(constructor, "_Source", side_effect=AssertionError("world born")):
            for consent in (None, True, "FROZEN", ""):
                with self.assertRaises(constructor.BirthGateError):
                    constructor.construct_worlds(seed=1, expected_pins={}, birth_authorization=consent)

    def test_missing_external_pins_never_reaches_source(self):
        with patch.object(constructor, "_Source", side_effect=AssertionError("world born")):
            for pins in ({}, None, {key: "0" * 64 for key in constructor.PIN_FIELDS if key != "fixture_reviewed_inventory_pin_sha256"}):
                with self.assertRaises(constructor.BirthGateError):
                    constructor.construct_worlds(seed=1, expected_pins=pins, birth_authorization=constructor.BIRTH_CONSENT)

    def test_wrong_external_bytes_never_reaches_source(self):
        pins = {key: "0" * 64 for key in constructor.PIN_FIELDS}
        with patch.object(constructor, "_Source", side_effect=AssertionError("world born")):
            with self.assertRaises(constructor.BirthGateError):
                constructor.construct_worlds(seed=1, expected_pins=pins, birth_authorization=constructor.BIRTH_CONSENT)

    def test_current_published_gate_pins_read_only(self):
        # Root has confirmed these independent reviews. Replays exact current
        # bytes for mechanics validation; grants no full-world birth consent.
        paths = {"catalogue_freeze_receipt_sha256": constructor.MODULE.parent / "catalogue_freeze_receipt.json",
                 "fixture_reviewed_inventory_pin_sha256": constructor.fixture_exclusions.REVIEWED_INVENTORY_PIN_PATH,
                 "fixture_inventory_sha256": constructor.fixture_exclusions.INVENTORY_PATH,
                 "fixture_exclusions_sha256": Path(constructor.fixture_exclusions.__file__),
                 "fixture_completeness_seal_sha256": constructor.fixture_exclusions.COMPLETENESS_SEAL_PATH}
        if not all(p.exists() for p in paths.values()):
            self.skipTest("independently reviewed birth artifacts not published")
        pins = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
        with patch.object(constructor, "_Source", side_effect=AssertionError("world born")):
            gates = constructor.verify_birth_gates(pins)
        self.assertEqual(gates["decision"], "BIRTH_PIN_GATES_VERIFIED")
        self.assertFalse(gates["solver_called"])
        self.assertFalse(gates["registry_written"])
        self.assertEqual(gates["external_pins"], pins)

    def test_forbidden_imports_absent(self):
        tree = ast.parse(constructor.MODULE.read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): imported.extend(a.name for a in node.names)
            if isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
                imported.extend(a.name for a in node.names)
        self.assertFalse(any("independent_solver" in name or "renderer" in name or "registry" in name for name in imported))
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("solve", "solve_world", "render", "reserve_parent") for n in ast.walk(tree)))


class SourcePrimitiveTests(unittest.TestCase):
    def test_false_guard_and_pre_action_swap_copy(self):
        before = {"m": 0, "x": 1, "y": 0}
        self.assertEqual(constructor.apply_program({"program_id": "P7", "guard": "m1", "action": "flip_x"}, before), before)
        self.assertEqual(constructor.apply_program({"program_id": "P7", "guard": "always", "action": "swap_mx"}, before), {"m": 1, "x": 0, "y": 0})
        self.assertEqual(constructor.apply_program({"program_id": "P7", "guard": "x1", "action": "copy_xy"}, before), {"m": 0, "x": 1, "y": 1})
        self.assertEqual(before, {"m": 0, "x": 1, "y": 0})

    def test_distinct_menu_functions_all_eight_states(self):
        for count in (3, 4):
            for seed in range(20):
                candidates, latent = constructor.sample_operator_menu(constructor.isolated_rng(seed, "bench-menu"), count=count)
                self.assertEqual(len(candidates), count)
                self.assertEqual(len({p["program_id"] for p in candidates}), count)
                tables = [constructor.truth_table(p) for p in candidates]
                self.assertEqual(len(set(tables)), count)
                self.assertTrue(all(len(t) == 8 for t in tables))
                self.assertIn(latent, [p["program_id"] for p in candidates])

    def test_latent_program_and_calibration_are_source_only(self):
        target = {"program_id": "OpaqueSource", "guard": "y1", "action": "swap_mx"}
        for seed in range(20):
            rng = constructor.isolated_rng(seed, "calibration")
            candidates, latent = constructor.sample_operator_menu(rng, count=4, latent_program=target)
            before = copy.deepcopy(candidates)
            selected = next(p for p in candidates if p["program_id"] == latent)
            observations = constructor.author_bench_examples(candidates, latent, rng)
            self.assertEqual(constructor.truth_table(selected), constructor.truth_table(target))
            self.assertEqual(candidates, before)
            for example in observations:
                self.assertEqual(example["output"], constructor.apply_program(selected, example["input"]))
            compatible = [p for p in candidates if all(constructor.apply_program(p, e["input"]) == e["output"] for e in observations)]
            self.assertEqual([p["program_id"] for p in compatible], [latent])
            for dropped in range(len(observations)):
                reduced = observations[:dropped] + observations[dropped + 1:]
                self.assertGreater(len([p for p in candidates if all(constructor.apply_program(p, e["input"]) == e["output"] for e in reduced)]), 1)

    def test_menu_and_bench_authoring_are_reproducible(self):
        before = random.getstate()
        first = constructor.sample_operator_menu(constructor.isolated_rng(45, "menu"), count=4)
        second = constructor.sample_operator_menu(constructor.isolated_rng(45, "menu"), count=4)
        self.assertEqual(first, second)
        self.assertEqual(random.getstate(), before)

    def test_primitive_invalid_inputs(self):
        good = {"program_id": "P", "guard": "always", "action": "flip_m"}
        for program, state in ((dict(good, answer=1), {"m": 0, "x": 0, "y": 0}),
                               (dict(good, guard="m2"), {"m": 0, "x": 0, "y": 0}),
                               (dict(good, action="copy_my"), {"m": 0, "x": 0, "y": 0}),
                               (dict(good, program_id=None), {"m": 0, "x": 0, "y": 0}),
                               (good, {"m": True, "x": 0, "y": 0}), (good, {"m": 0, "x": 0})):
            with self.assertRaises(ValueError): constructor.apply_program(program, state)
        for count in (2, 5, True):
            with self.assertRaises(ValueError): constructor.sample_operator_menu(random.Random(1), count=count)
        with self.assertRaises(ValueError):
            constructor.author_bench_examples([good, dict(good, program_id="Q"), dict(good, program_id="S")], "P", random.Random(1))


if __name__ == "__main__":
    unittest.main()

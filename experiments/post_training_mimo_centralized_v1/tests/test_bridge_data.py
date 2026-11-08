"""Synthetic mechanics tests; no model or production bank is opened."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.post_training_mimo_centralized_v1.scripts import bridge_data as b


def parent(family, seed=71, role="TRAIN", catalog="familiar"):
    for i in range(10000):
        item = b.allocate_parent(b.generate_spec(family, seed, i, catalog), seed, i)
        if item.role == role:
            return item
    raise AssertionError("test fixture failed to find role")


def registration():
    return {
        "status": "FROZEN_FAST_LANE_DATA_ONLY", "selection_seed": 20260927,
        "authorization": {"dataset_artifact_generation": True, "native_execution": False,
                          "check_rendering": False, "final_rendering": False},
        "families": list(b.FAMILIES),
        "counts": {role: {"acquisition_parents": 8 if role == "CHECK" else 4} for role in b.ROLES},
    }


class IdentityTests(unittest.TestCase):
    def test_addition_and_multiplication_commute_recursively(self):
        left = {"family": b.FAMILIES[0], "expression": b._op("add", 1, b._op("mul", 2, 3))}
        right = {"family": b.FAMILIES[0], "expression": b._op("add", b._op("mul", 3, 2), 1)}
        self.assertEqual(b.allocate_parent(left), b.allocate_parent(right))

    def test_subtraction_does_not_commute(self):
        a = b.allocate_parent({"family": b.FAMILIES[0], "expression": b._op("sub", 1, 2)})
        z = b.allocate_parent({"family": b.FAMILIES[0], "expression": b._op("sub", 2, 1)})
        self.assertNotEqual(a.canonical_parent_sha256, z.canonical_parent_sha256)

    def test_sets_sort_deduplicate_and_commute(self):
        left = {"family": b.FAMILIES[2], "expression": b._op("union", [2, 1, 2], [3, 2])}
        right = {"family": b.FAMILIES[2], "expression": b._op("union", [2, 3], [1, 2])}
        self.assertEqual(b.allocate_parent(left), b.allocate_parent(right))

    def test_difference_and_text_program_order_remain_meaningful(self):
        left = {"family": b.FAMILIES[2], "expression": b._op("difference", [1], [2])}
        right = {"family": b.FAMILIES[2], "expression": b._op("difference", [2], [1])}
        self.assertNotEqual(b.allocate_parent(left).parent_id, b.allocate_parent(right).parent_id)
        spec = b.generate_spec(b.FAMILIES[3], 1, 0)
        changed = copy.deepcopy(spec)
        changed["program"].reverse()
        self.assertNotEqual(b.allocate_parent(spec).parent_id, b.allocate_parent(changed).parent_id)

    def test_all_five_roles_and_domain_hash_formula(self):
        seen = set()
        for i in range(300):
            p = b.allocate_parent(b.generate_spec(b.FAMILIES[0], 1, i), 1, i)
            seen.add(p.role)
            raw = p.canonical_json.encode()
            self.assertEqual(p.canonical_parent_sha256, b.sha256(raw))
            digest = b.sha256(b.ROLE_DOMAIN + raw)
            self.assertEqual(p.split_sha256, digest)
            bucket = int(digest[:16], 16) % 10000
            self.assertEqual(p.role, next(role for ceiling, role in b.ROLE_BUCKETS if bucket < ceiling))
        self.assertEqual(seen, set(b.ROLES))

    def test_metadata_and_surface_variants_cannot_resplit_parent(self):
        p = parent(b.FAMILIES[1])
        p2 = b.allocate_parent(json.loads(p.canonical_json), seed=999, candidate_index=1234)
        self.assertEqual((p.parent_id, p.role), (p2.parent_id, p2.role))
        rows = [b.make_row(p, "a" * 40, variant) for variant in (0, 1)]
        self.assertEqual(rows[0]["parent_id"], rows[1]["parent_id"])
        self.assertEqual(rows[0]["role"], rows[1]["role"])
        self.assertEqual(rows[0]["target"], rows[1]["target"])
        self.assertNotEqual(rows[0]["prompt"], rows[1]["prompt"])

    def test_catalogs_have_disjoint_operation_structures(self):
        for family in b.FAMILIES:
            catalogs = []
            for catalog in ("familiar", "check_composition", "final_composition"):
                catalogs.append({b.allocate_parent(b.generate_spec(family, 3, i, catalog)).structure_id for i in range(8)})
            self.assertTrue(catalogs[0].isdisjoint(catalogs[1]))
            self.assertTrue(catalogs[0].isdisjoint(catalogs[2]))
            self.assertTrue(catalogs[1].isdisjoint(catalogs[2]))

    def test_sealed_apis_fail_even_on_known_correct_input(self):
        for role in ("CHECK", "FINAL"):
            p = parent(b.FAMILIES[0], role=role)
            for fn, args in ((b.render_prompt, (p,)), (b.reference_answer, (p,)),
                             (b.verify_answer, (p, '{"answer":0}')), (b.make_row, (p, "a" * 40))):
                with self.assertRaises(PermissionError):
                    fn(*args)

    def test_forged_role_or_identity_is_host_error(self):
        p = parent(b.FAMILIES[0], role="CHECK")
        for forged in (replace(p, role="TRAIN"), replace(p, parent_id="bridge-fake"),
                       replace(p, canonical_parent_sha256="0" * 64), replace(p, structure_id="wrong")):
            with self.assertRaises(ValueError):
                b.verify_answer(forged, "invalid response")

    def test_invalid_specs_and_bool_inputs_are_rejected(self):
        examples = [{"family": "unknown"}, {"family": b.FAMILIES[0], "expression": True},
                    {"family": b.FAMILIES[0], "expression": 51},
                    {"family": b.FAMILIES[0], "expression": b._op("div", 1, 2)},
                    {"family": b.FAMILIES[2], "expression": [True, 2]}]
        for spec in examples:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                b.allocate_parent(spec)
        for bad in (True, -1, 2**64):
            with self.assertRaises(ValueError):
                b.generate_spec(b.FAMILIES[0], bad, 0)

    def test_allocate_never_renders_or_solves(self):
        with patch.object(b, "render_prompt", side_effect=AssertionError("rendered before split")), \
                patch.object(b, "reference_answer", side_effect=AssertionError("solved before split")), \
                patch.object(b, "verify_answer", side_effect=AssertionError("verified before split")):
            parents, _ = b.prepare_parents({role: 8 if role == "CHECK" else 4 for role in b.ROLES}, 33)
        self.assertEqual(len(parents), 24)


class VerifierTests(unittest.TestCase):
    def test_positive_and_semantic_mutations_over_fresh_specs(self):
        checked = 0
        for family in b.FAMILIES:
            for catalog in ("familiar", "check_composition", "final_composition"):
                for i in range(70):
                    p = b.allocate_parent(b.generate_spec(family, 202, i, catalog), 202, i)
                    if p.role not in b.OPEN_ROLES:
                        continue
                    answer = b.reference_answer(p)
                    self.assertTrue(b.verify_answer(p, b.json_bytes({"answer": answer}).decode()))
                    self.assertFalse(b.verify_answer(p, b.json_bytes({"answer": b.wrong_answer(answer)}).decode()))
                    checked += 1
        self.assertGreater(checked, 650)

    def test_verifier_does_not_call_or_trust_reference(self):
        for family in b.FAMILIES:
            p = parent(family)
            response = b.json_bytes({"answer": b.reference_answer(p)}).decode()
            with patch.object(b, "reference_answer", side_effect=AssertionError("shared solver")):
                self.assertTrue(b.verify_answer(p, response))

    def test_mutated_reference_is_caught_at_row_generation(self):
        p = parent(b.FAMILIES[0])
        bad = b.reference_answer(p) + 1
        with patch.object(b, "reference_answer", return_value=bad), self.assertRaises(RuntimeError):
            b.make_row(p, "a" * 40)

    def test_malformed_envelopes_and_nonfinite_values_rejected(self):
        invalid = ["", "null", "[]", "0", "true", '{}', '{"answer":0,"extra":0}',
                   '{"answer":0,"answer":0}', '{"answer":0,"ans\\u0077er":0}',
                   '{"answer":NaN}', '{"answer":Infinity}', '{"answer":-Infinity}',
                   '{"answer":0} trailing', '```json\n{"answer":0}\n```',
                   '{"Answer":0}', '{"answer":0,}', '\ufeff{"answer":0}', " " * 4097]
        for family in b.FAMILIES:
            p = parent(family)
            for response in invalid:
                with self.subTest(family=family, response=response):
                    self.assertFalse(b.verify_answer(p, response))

    def test_boolean_and_float_are_not_integer_answers(self):
        for family in b.FAMILIES[:3]:
            p = parent(family)
            for answer in (True, False, 1.0, [True], [1.0], [False, False, False], [0.0, 0.0, 0.0]):
                self.assertFalse(b.verify_answer(p, b.json_bytes({"answer": answer}).decode()))

    def test_correct_answer_with_extra_or_duplicate_keys_fails(self):
        for family in b.FAMILIES:
            p = parent(family)
            answer = json.dumps(b.reference_answer(p))
            self.assertFalse(b.verify_answer(p, '{"answer":' + answer + ',"extra":null}'))
            self.assertFalse(b.verify_answer(p, '{"answer":' + answer + ',"answer":' + answer + '}'))
            self.assertTrue(b.verify_answer(p, ' \n { "answer": ' + answer + " } \t"))

    def test_set_order_duplicates_and_out_of_domain_fail(self):
        for i in range(100):
            p = b.allocate_parent(b.generate_spec(b.FAMILIES[2], 122, i), 122, i)
            if p.role not in b.OPEN_ROLES:
                continue
            answer = b.reference_answer(p)
            if len(answer) >= 2:
                for mutation in (list(reversed(answer)), answer + [answer[-1]], answer + [1000]):
                    self.assertFalse(b.verify_answer(p, json.dumps({"answer": mutation})))
                return
        self.fail("nontrivial set fixture not found")

    def test_text_is_exact_no_stripping_or_case_relaxation(self):
        p = parent(b.FAMILIES[3])
        answer = b.reference_answer(p)
        for mutation in (answer + " ", " " + answer, answer + "\n", [answer]):
            self.assertFalse(b.verify_answer(p, json.dumps({"answer": mutation})))

    def test_empty_set_and_empty_text_are_valid_answers(self):
        for family, spec in (
            (b.FAMILIES[2], {"family": b.FAMILIES[2], "expression": []}),
            (b.FAMILIES[3], {"family": b.FAMILIES[3], "tokens": ["fern"], "program": [{"op": "drop", "n": 3}], "separator": "|"}),
        ):
            p = b.allocate_parent(spec)
            # These public mechanics fixtures use fixed identities, never role overrides.
            if p.role in b.OPEN_ROLES:
                answer = [] if family == b.FAMILIES[2] else ""
                self.assertEqual(b.reference_answer(p), answer)
                self.assertTrue(b.verify_answer(p, json.dumps({"answer": answer})))

    def test_wrong_types_null_objects_and_long_payloads_fail(self):
        for family in b.FAMILIES:
            p = parent(family)
            for value in (None, {}, {"x": 1}, [[1]], "x" * 5000):
                self.assertFalse(b.verify_answer(p, json.dumps({"answer": value})))

    def test_host_corruption_is_not_a_model_failure(self):
        p = parent(b.FAMILIES[0])
        broken = replace(p, canonical_json='{"family":"unknown"}')
        with self.assertRaises(ValueError):
            b.verify_answer(broken, "wrong answer")


class ExportTests(unittest.TestCase):
    def test_counts_lineage_hashes_controls_and_unmeasured_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reg = root / "registration.json"
            reg.write_bytes(b.json_bytes(registration()))
            receipt = b.build_dataset(reg, root / "data", "a" * 40, variants=2)
            self.assertEqual(receipt["parent_counts"], {role: 8 if role == "CHECK" else 4 for role in b.ROLES})
            self.assertEqual(receipt["open_row_counts"], {role: 8 for role in b.ROLES[:3]})
            self.assertEqual(receipt["controls"]["positive_reference_answers_verified"], 24)
            self.assertFalse(receipt["e1_launch_ready"])
            self.assertIsNone(receipt["cost"]["dataset_token_count"])
            self.assertEqual(receipt["cost"]["actual_training_tokens"], 0)
            for name, data in receipt["files"].items():
                raw = (root / "data" / name).read_bytes()
                self.assertEqual(b.sha256(raw), data["sha256"])
                self.assertEqual(len(raw), data["bytes"])
            all_ids = set()
            for role in b.ROLES[:3]:
                rows = [json.loads(line) for line in (root / "data" / f"{role}.jsonl").read_text().splitlines()]
                self.assertEqual(len({row["parent_id"] for row in rows}), 4)
                self.assertTrue(all_ids.isdisjoint(row["parent_id"] for row in rows))
                all_ids.update(row["parent_id"] for row in rows)
                for row in rows:
                    self.assertEqual(row["role"], role)
                    self.assertEqual(row["prompt_sha256"], b.sha256(row["prompt"].encode()))
                    self.assertEqual(row["target_sha256"], b.sha256(row["target"].encode()))
                    self.assertEqual(row["canonical_parent_sha256"], b.sha256(b.json_bytes(row["spec"])))
                    self.assertIsNone(row["tokenized_prompt_sha256"])
                    self.assertIsNone(row["prompt_tokens"])
                    self.assertIsNone(row["target_tokens"])
                    self.assertEqual(row["tokenization_status"], "UNMEASURED")
                    self.assertFalse(row["training_eligibility"])
                    self.assertEqual(row["training_role_eligible"], role == "TRAIN")
                    self.assertEqual(row["license_provenance"]["redistribution_license"], "UNRESOLVED_NO_NEW_LICENSE_GRANT")
            reservations = [json.loads(line) for line in (root / "data" / "SEALED_RESERVATIONS.jsonl").read_text().splitlines()]
            self.assertEqual(len(reservations), 12)
            self.assertTrue(all_ids.isdisjoint(row["parent_id"] for row in reservations))
            allowed = {"parent_id", "role", "canonical_parent_sha256", "split_sha256", "status"}
            self.assertTrue(all(set(row) == allowed for row in reservations))
            self.assertTrue(all(row["role"] in {"CHECK", "FINAL"} for row in reservations))

    def test_output_is_write_once_and_registration_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reg = root / "registration.json"
            reg.write_bytes(b.json_bytes(registration()))
            b.build_dataset(reg, root / "data", "a" * 40)
            before = {p.name: p.read_bytes() for p in (root / "data").iterdir()}
            with self.assertRaises(FileExistsError):
                b.build_dataset(reg, root / "data", "b" * 40)
            self.assertEqual(before, {p.name: p.read_bytes() for p in (root / "data").iterdir()})
            invalid = registration()
            invalid["authorization"]["dataset_artifact_generation"] = False
            reg.write_bytes(b.json_bytes(invalid))
            with self.assertRaises(ValueError):
                b.build_dataset(reg, root / "other", "a" * 40)
            self.assertFalse((root / "other").exists())

    def test_regeneration_is_byte_identical_except_measurement_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reg = root / "registration.json"
            reg.write_bytes(b.json_bytes(registration()))
            r1 = b.build_dataset(reg, root / "one", "a" * 40)
            r2 = b.build_dataset(reg, root / "two", "a" * 40)
            self.assertEqual(r1["files"], r2["files"])

    def test_build_never_passes_sealed_parents_to_solvers(self):
        original = b.reference_answer
        calls = []
        def guarded(p):
            self.assertIn(p.role, b.OPEN_ROLES)
            calls.append(p.parent_id)
            return original(p)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reg = root / "registration.json"
            reg.write_bytes(b.json_bytes(registration()))
            with patch.object(b, "reference_answer", side_effect=guarded):
                b.build_dataset(reg, root / "data", "a" * 40)
        self.assertEqual(len(calls), 12)

    def test_final_null_remains_unresolved_and_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reg = registration()
            reg["counts"]["FINAL"]["acquisition_parents"] = None
            path = root / "registration.json"
            path.write_bytes(b.json_bytes(reg))
            receipt = b.build_dataset(path, root / "data", "a" * 40)
            self.assertEqual(receipt["parent_counts"]["FINAL"], 0)
            self.assertEqual(receipt["final_count_status"], "UNRESOLVED_UNOPENED")

    def test_bad_registration_variants_and_source_revisions(self):
        invalids = []
        reg = registration()
        reg["status"] = "PROPOSED"
        invalids.append(reg)
        for field in ("check_rendering", "final_rendering"):
            reg = registration()
            reg["authorization"][field] = True
            invalids.append(reg)
        for value in (True, -4, 1, 2.0):
            reg = registration()
            reg["counts"]["TRAIN"]["acquisition_parents"] = value
            invalids.append(reg)
        reg = registration()
        reg["counts"]["CHECK"]["acquisition_parents"] = 4
        invalids.append(reg)
        reg = registration()
        reg["counts"]["TRAIN"]["per_family"] = 20
        invalids.append(reg)
        reg = registration()
        reg["parent_policy"] = {"buckets": {"TRAIN": [0, 10000]}}
        invalids.append(reg)
        for reg in invalids:
            with self.subTest(reg=reg), self.assertRaises(ValueError):
                b.validate_registration(reg)
        p = parent(b.FAMILIES[0])
        with self.assertRaises(ValueError):
            b.make_row(p, "latest")
        with self.assertRaises(ValueError):
            b.render_prompt(p, True)

    def test_verification_failure_creates_no_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reg = root / "registration.json"
            reg.write_bytes(b.json_bytes(registration()))
            with patch.object(b, "verify_answer", return_value=False), self.assertRaises(RuntimeError):
                b.build_dataset(reg, root / "data", "a" * 40)
            self.assertFalse((root / "data").exists())


if __name__ == "__main__":
    unittest.main()

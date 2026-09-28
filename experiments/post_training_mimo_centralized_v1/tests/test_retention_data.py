"""Synthetic retention mechanics only; no production bank or model is opened."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.post_training_mimo_centralized_v1.scripts import retention_data as r


def parent(domain="retrieval", role="TRAIN", seed=17):
    for index in range(10000):
        candidate = r.allocate_parent(r.generate_spec(domain, seed, index), seed, index)
        if candidate.role == role:
            return candidate
    raise AssertionError("could not find synthetic fixture role")


def registration():
    return {
        "status": "FROZEN_FAST_LANE_RETENTION_ONLY", "selection_seed": 913,
        "domains": list(r.DOMAINS),
        "authorization": {"dataset_artifact_generation": True, "native_execution": False,
                          "training": False, "check_rendering": False, "final_rendering": False},
        "counts": {role: {domain: 4 if role in {"TRAIN", "DEV", "CHECK"} else 0
                          for domain in r.DOMAINS} for role in r.ROLES},
    }


def answer(value):
    return r.json_bytes({"answer": value}).decode()


class ParentTests(unittest.TestCase):
    def test_reordered_and_duplicate_facts_share_identity_and_role(self):
        p = parent()
        spec = json.loads(p.canonical_json)
        spec["records"].reverse()
        spec["records"].append(copy.deepcopy(spec["records"][0]))
        q = r.allocate_parent(spec, seed=99, candidate_index=999)
        self.assertEqual((p.parent_id, p.role, p.canonical_json), (q.parent_id, q.role, q.canonical_json))

    def test_query_and_domain_derivatives_cannot_resplit_world(self):
        p = parent()
        spec = json.loads(p.canonical_json)
        expected = r.reference_answer(p)
        spec["queries"].reverse()
        q = r.allocate_parent(spec)
        self.assertEqual((p.parent_id, p.role), (q.parent_id, q.role))
        self.assertEqual(r.reference_answer(q), expected[::-1])
        spec["domain"] = "answerability_abstention"
        twin = r.allocate_parent(spec)
        self.assertEqual((p.parent_id, p.role), (twin.parent_id, twin.role))
        self.assertEqual(r.reference_answer(twin, "answerable"), expected[::-1])
        self.assertIsNone(r.reference_answer(twin, "missing_evidence"))
        spec["queries"][0] = {key: spec["records"][-1][key] for key in ("entity", "field")}
        if spec["queries"][0] != spec["queries"][1]:
            self.assertEqual(r.allocate_parent(spec).parent_id, p.parent_id)

    def test_instruction_directive_order_duplicates_and_templates_stay_together(self):
        p = parent("instruction_following")
        spec = json.loads(p.canonical_json)
        spec["directives"].reverse()
        spec["directives"].append(copy.deepcopy(spec["directives"][0]))
        q = r.allocate_parent(spec)
        self.assertEqual((p.parent_id, p.role), (q.parent_id, q.role))
        self.assertEqual(r.reference_answer(p), r.reference_answer(q))
        for template in r.FORMATS:
            spec["template"] = template
            sibling = r.allocate_parent(spec)
            self.assertEqual((p.parent_id, p.role), (sibling.parent_id, sibling.role))

    def test_contradictory_records_and_incomplete_parents_are_rejected(self):
        spec = json.loads(parent().canonical_json)
        bad = copy.deepcopy(spec)
        bad["records"].append({**bad["records"][0], "value": "Contradiction"})
        with self.assertRaisesRegex(ValueError, "contradictory"):
            r.allocate_parent(bad)
        bad = copy.deepcopy(spec)
        query = bad["queries"][0]
        bad["records"] = [record for record in bad["records"] if any(record[key] != query[key] for key in query)]
        with self.assertRaisesRegex(ValueError, "missing queried"):
            r.allocate_parent(bad)

    def test_contradictory_instruction_priority_is_rejected(self):
        spec = json.loads(parent("instruction_following").canonical_json)
        spec["directives"].append({**spec["directives"][0], "text": "Conflicting order"})
        with self.assertRaisesRegex(ValueError, "contradictory"):
            r.allocate_parent(spec)

    def test_role_formula_all_roles_and_tampered_host_metadata(self):
        seen = set()
        for index in range(150):
            p = r.allocate_parent(r.generate_spec("retrieval", 2, index), 2, index)
            seen.add(p.role)
            spec = json.loads(p.canonical_json)
            digest = r.sha256(r.ROLE_DOMAIN + r._semantic_bytes(spec))
            self.assertEqual(p.split_sha256, digest)
            bucket = int(digest[:16], 16) % 10000
            self.assertEqual(p.role, next(role for high, role in r.ROLE_BUCKETS if bucket < high))
        self.assertEqual(seen, set(r.ROLES))
        p = parent(role="CHECK")
        for q in (replace(p, role="TRAIN"), replace(p, parent_id="fake"), replace(p, stratum="fake")):
            with self.assertRaises(ValueError):
                r.reference_answer(q)

    def test_closed_roles_never_render_solve_verify_or_make_rows(self):
        for role in ("TRAIN_SELECTION", "CHECK", "FINAL"):
            p = parent(role=role)
            for fn, args in ((r.reference_answer, (p,)), (r.render_prompt, (p,)),
                             (r.verify_answer, (p, answer(None))), (r.make_row, (p, "a" * 40))):
                with self.assertRaises(PermissionError):
                    fn(*args)


class SemanticsTests(unittest.TestCase):
    def test_generated_answers_and_independent_verifier_all_domains(self):
        verified = 0
        for domain in r.DOMAINS:
            for index in range(80):
                p = r.allocate_parent(r.generate_spec(domain, 33, index), 33, index)
                if p.role not in r.OPEN_ROLES:
                    continue
                variants = ("answerable", "missing_evidence") if domain == "answerability_abstention" else ("standard",)
                for variant in variants:
                    target = answer(r.reference_answer(p, variant))
                    with patch.object(r, "reference_answer", side_effect=AssertionError("verifier called reference")), \
                         patch.object(r, "_visible_records", side_effect=AssertionError("verifier reused evidence execution")):
                        self.assertTrue(r.verify_answer(p, target, variant))
                    verified += 1
        self.assertGreater(verified, 150)

    def test_wrong_entity_field_order_or_case_is_not_evidence(self):
        p = parent()
        spec = json.loads(p.canonical_json)
        correct = r.reference_answer(p)
        self.assertNotEqual(correct[0], correct[1])
        self.assertFalse(r.verify_answer(p, answer(correct[::-1])))
        self.assertFalse(r.verify_answer(p, answer([value.lower() for value in correct])))
        query = spec["queries"][0]
        wrong_field = next(record["value"] for record in spec["records"]
                           if record["entity"] == query["entity"] and record["field"] != query["field"])
        wrong_entity = next(record["value"] for record in spec["records"]
                            if record["entity"] != query["entity"] and record["field"] == query["field"])
        for unsupported in (wrong_field, wrong_entity):
            self.assertNotEqual(unsupported, correct[0])
            self.assertFalse(r.verify_answer(p, answer([unsupported, correct[1]])))

    def test_missing_evidence_twin_never_uses_hidden_parent_value(self):
        p = parent("answerability_abstention")
        complete = r.reference_answer(p, "answerable")
        self.assertIsNone(r.reference_answer(p, "missing_evidence"))
        self.assertFalse(r.verify_answer(p, answer(complete), "missing_evidence"))
        self.assertFalse(r.verify_answer(p, answer(None), "answerable"))
        missing_prompt = r.render_prompt(p, "missing_evidence")
        self.assertNotIn(complete[0], missing_prompt)
        self.assertIn(complete[1], missing_prompt)
        full_lines = r.render_prompt(p, "answerable").splitlines()
        missing_lines = missing_prompt.splitlines()
        self.assertEqual([line for line in full_lines if complete[0] not in line], missing_lines)
        self.assertTrue(r.verify_answer(p, answer(None), "missing_evidence"))
        with self.assertRaises(ValueError):
            r.verify_answer(p, answer(None), "standard")

    def test_instruction_priority_and_exact_output_have_manual_oracle(self):
        spec = {"domain": "instruction_following", "template": r.FORMATS[0], "directives": [
            {"priority": "final", "slot": "message", "text": "Stay exactly Here"},
            {"priority": "base", "slot": "message", "text": "Wrong base"},
            {"priority": "task", "slot": "recipient", "text": "Cedar Room"},
            {"priority": "base", "slot": "recipient", "text": "Wrong recipient"},
            {"priority": "task", "slot": "message", "text": "Wrong task"},
            {"priority": "base", "slot": "note", "text": "Forbidden note"},
        ]}
        # Vary a literal distractor until an open role occurs; this is a test fixture only.
        for i in range(100):
            spec["directives"][-1]["text"] = "Forbidden note " + str(i)
            p = r.allocate_parent(spec)
            if p.role in r.OPEN_ROLES:
                break
        expected = "To: Cedar Room\nMessage: Stay exactly Here"
        self.assertEqual(r.reference_answer(p), expected)
        self.assertTrue(r.verify_answer(p, answer(expected)))
        for incorrect in (expected.replace("Cedar Room", "Wrong recipient"),
                          expected.replace("Stay exactly Here", "Wrong base"),
                          expected.replace("Stay exactly Here", "Wrong task"),
                          expected.lower(), expected + " Forbidden note", expected.replace("\n", " ")):
            self.assertFalse(r.verify_answer(p, answer(incorrect)))

    def test_strict_json_types_duplicate_keys_nonfinite_and_extras(self):
        p = parent()
        good = answer(r.reference_answer(p))
        for bad in ('{"answer":true}', '{"answer":1}', '{"answer":null}', '{"answer":NaN}',
                    '{"answer":[true,false]}', '{"answer":[1,2]}', '{"answer":{}}',
                    good[:-1] + ',"extra":1}', '{"answer":null,"answer":null}',
                    good + " extra", "```json\n" + good + "\n```", "[" + good + "]", ""):
            self.assertFalse(r.verify_answer(p, bad))
        twin = parent("answerability_abstention")
        self.assertFalse(r.verify_answer(twin, '{"answer":false}', "missing_evidence"))
        self.assertFalse(r.verify_answer(twin, '{"answer":0}', "missing_evidence"))
        self.assertFalse(r.verify_answer(twin, '{"answer":"null"}', "missing_evidence"))
        bad_spec = r.generate_spec("retrieval", 1, 0)
        bad_spec["records"][0]["value"] = True
        with self.assertRaises(ValueError):
            r.allocate_parent(bad_spec)


class ExportTests(unittest.TestCase):
    def test_allocation_precedes_render_and_score_and_is_deterministic(self):
        counts = r.validate_registration(registration())
        with patch.object(r, "render_prompt", side_effect=AssertionError("early render")), \
             patch.object(r, "reference_answer", side_effect=AssertionError("early answer")), \
             patch.object(r, "verify_answer", side_effect=AssertionError("early score")):
            parents, summary = r.prepare_parents(counts, 913)
            again, again_summary = r.prepare_parents(counts, 913)
        self.assertEqual((parents, summary), (again, again_summary))
        self.assertEqual(len(parents), 36)
        self.assertEqual(len({p.parent_id for p in parents}), 36)
        self.assertEqual({p.role for p in parents}, {"TRAIN", "DEV", "CHECK"})
        for role in ("TRAIN", "DEV", "CHECK"):
            for domain in r.DOMAINS:
                selected = [p for p in parents if p.role == role and json.loads(p.canonical_json)["domain"] == domain]
                self.assertEqual(len(selected), 4)
                self.assertEqual(len({p.stratum for p in selected}), 4)

    def test_export_hash_only_check_twins_stay_together_and_write_once(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            reg = base / "registration.json"
            reg.write_bytes(r.json_bytes(registration()))
            out = base / "out"
            real_row = r.make_row
            rendered_roles = []
            def guarded_row(p, source_revision, variant="standard"):
                self.assertIn(p.role, r.OPEN_ROLES)
                rendered_roles.append(p.role)
                return real_row(p, source_revision, variant)
            with patch.object(r, "make_row", side_effect=guarded_row):
                receipt = r.build_dataset(reg, out, "a" * 40)
            self.assertEqual(len(rendered_roles), 32)
            self.assertEqual(receipt["open_row_counts"], {"TRAIN": 16, "DEV": 16})
            role_parents = {}
            for role in r.OPEN_ROLES:
                rows = [json.loads(line) for line in (out / (role + ".jsonl")).read_text().splitlines()]
                role_parents[role] = {row["parent_id"] for row in rows}
                self.assertEqual(len(role_parents[role]), 12)
                twins = {}
                for row in rows:
                    self.assertFalse(row["training_eligibility"])
                    self.assertIsNone(row["source_correct_membership"])
                    self.assertEqual(row["cost"]["model_calls"], 0)
                    if row["domain"] == "answerability_abstention":
                        twins.setdefault(row["parent_id"], []).append(row)
                self.assertEqual(len(twins), 4)
                for pair in twins.values():
                    self.assertEqual({row["variant_id"] for row in pair}, {"answerable", "missing_evidence"})
                    self.assertEqual(len({row["split_sha256"] for row in pair}), 1)
            sealed = [json.loads(line) for line in (out / "SEALED_RESERVATIONS.jsonl").read_text().splitlines()]
            self.assertEqual(len(sealed), 12)
            role_parents["CHECK"] = {row["parent_id"] for row in sealed}
            for item in sealed:
                self.assertEqual(set(item), {"parent_id", "role", "canonical_parent_sha256", "split_sha256", "status"})
                self.assertEqual(item["role"], "CHECK")
                self.assertEqual(item["status"], "HASH_ONLY_UNOPENED")
            self.assertTrue(role_parents["TRAIN"].isdisjoint(role_parents["DEV"]))
            self.assertTrue(role_parents["TRAIN"].isdisjoint(role_parents["CHECK"]))
            self.assertTrue(role_parents["DEV"].isdisjoint(role_parents["CHECK"]))
            self.assertFalse((out / "FINAL.jsonl").exists())
            for name, file_info in receipt["files"].items():
                self.assertEqual(r.sha256((out / name).read_bytes()), file_info["sha256"])
            for key in ("source_correct_enrollment_qualified", "retention_dataset_qualified",
                        "language_preservation_qualified", "long_context_preservation_qualified", "e1_launch_ready"):
                self.assertFalse(receipt[key])
            prior = {path.name: path.read_bytes() for path in out.iterdir()}
            with self.assertRaises(FileExistsError):
                r.build_dataset(reg, out, "a" * 40)
            self.assertEqual(prior, {path.name: path.read_bytes() for path in out.iterdir()})
            second = base / "second"
            r.build_dataset(reg, second, "a" * 40)
            for name in ("TRAIN.jsonl", "DEV.jsonl", "SEALED_RESERVATIONS.jsonl"):
                self.assertEqual((out / name).read_bytes(), (second / name).read_bytes())

    def test_registration_fail_closed_before_writes(self):
        mutations = []
        for key, value in (("native_execution", True), ("training", True), ("check_rendering", True),
                           ("final_rendering", True), ("dataset_artifact_generation", 1)):
            bad = registration()
            bad["authorization"][key] = value
            mutations.append(bad)
        for role, value in (("FINAL", 4), ("TRAIN_SELECTION", 4), ("TRAIN", 1), ("DEV", True)):
            bad = registration()
            bad["counts"][role]["retrieval"] = value
            mutations.append(bad)
        bad = registration()
        bad["status"] = "PROPOSED_UNQUALIFIED"
        mutations.append(bad)
        bad = registration()
        bad["model_outcome_based_selection"] = True
        mutations.append(bad)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reg, output = root / "registration.json", root / "out"
            for bad in mutations:
                reg.write_bytes(r.json_bytes(bad))
                with self.assertRaises(ValueError):
                    r.build_dataset(reg, output, "a" * 40)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

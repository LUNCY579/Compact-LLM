"""Tiny stdlib-only oracle/integrity tests; no full artifact build or model use."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import generate as g


class BindingSignalTests(unittest.TestCase):
    def setUp(self):
        self.dataset = g.generate(train_worlds=2, dev_worlds=2)

    def test_deterministic_balanced_grouped_partitions(self):
        self.assertEqual(self.dataset, g.generate(train_worlds=2, dev_worlds=2))
        report = g.audit(self.dataset)
        self.assertEqual(report["rows"], {"train": 24, "dev": 24})
        self.assertEqual(report["worlds"], {"train": 2, "dev": 2})
        self.assertTrue(all(set(counts.values()) == {6} for counts in report["category_counts"].values()))
        self.assertEqual(sum(row["reference"] == "UNKNOWN" for row in self.dataset["train"]), 12)
        self.assertTrue(report["all_related_views_grouped"])

    def test_oracle_reads_facts_not_condition_or_template_labels(self):
        world = {"roles": ["r1", "r2", "r3", "r4", "r5", "r6"],
                 "people": ["p1", "p2", "p3", "p4", "p5", "p6"], "action": "inspect a form",
                 "assignments": [{"role": "r1", "status": "occupied", "person": "p4"},
                                 {"role": "r2", "status": "vacant", "person": None}]}
        self.assertEqual(g.oracle(world, {"kind": "person", "target_role": "r1", "condition": "missing"}), "p4")
        self.assertEqual(g.oracle(world, {"kind": "person", "target_role": "r2"}), "UNKNOWN")
        self.assertEqual(g.oracle(world, {"kind": "person", "target_role": "r3"}), "UNKNOWN")
        self.assertEqual(g.oracle(world, {"kind": "role", "target_role": "r2"}), "r2")
        self.assertEqual(g.oracle(world, {"kind": "role", "target_role": "r3"}), "r3")

    def test_counterfactual_context_always_contains_vacancy_and_unbound_names(self):
        for rows in self.dataset.values():
            for row in rows:
                world = row["world"]
                facts = g.validate_world(world)
                self.assertEqual(sum(f["status"] == "vacant" for f in facts.values()), 1)
                self.assertEqual(len(set(world["roles"]) - set(facts)), 1)
                assigned = {f["person"] for f in facts.values() if f["status"] == "occupied"}
                self.assertEqual(len(set(world["people"]) - assigned), 2)

    def test_wrong_reference_and_wrong_rendered_answer_are_refused(self):
        for field in ("reference", "assistant"):
            data = copy.deepcopy(self.dataset)
            if field == "reference":
                data["train"][0]["reference"] = "UNKNOWN"
            else:
                data["train"][0]["messages"][-1]["content"] = "UNKNOWN"
            with self.assertRaisesRegex(ValueError, "wrong answer"):
                g.audit(data)

    def test_ambiguous_binding_is_refused_by_oracle(self):
        row = self.dataset["train"][0]
        world = copy.deepcopy(row["world"])
        world["assignments"].append({"role": row["query"]["target_role"], "status": "occupied",
                                     "person": world["people"][-1]})
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            g.oracle(world, row["query"])

    def test_all_four_families_decode_from_policy_text_without_gold_input(self):
        for family, template in g.TEMPLATES.items():
            world = g.make_world(template["split"], 0)
            world["template_family"] = family
            for query in g.queries(world):
                decoded = g.decode_policy_input(g.render_user(world, query))
                self.assertEqual(decoded["template_family"], family)
                self.assertEqual(decoded["answer"], g.oracle(world, query))
                self.assertEqual(decoded["query"], {"kind": query["kind"], "target_role": query["target_role"]})

    def test_renderer_omission_rejected_even_when_regeneration_agrees(self):
        render = g.render_user

        def omit_target_assignment(world, query):
            text = render(world, query)
            if query["condition"] != "occupied":
                return text
            fact = g.validate_world(world)[query["target_role"]]
            return "\n".join(line for line in text.splitlines()
                             if not (fact["role"] in line and fact["person"] in line))

        with patch.object(g, "render_user", side_effect=omit_target_assignment):
            faulty = g.generate(train_worlds=1, dev_worlds=1)
            row = faulty["train"][0]
            self.assertEqual(row["messages"], g.make_row(row["world"], row["query"])["messages"])
            with self.assertRaisesRegex(ValueError, "omitted or changed assignment"):
                g.audit(faulty)

    def test_renderer_question_swap_rejected_even_when_regeneration_agrees(self):
        render = g.render_user

        def swap_kind(world, query):
            return render(world, dict(query, kind="role" if query["kind"] == "person" else "person"))

        with patch.object(g, "render_user", side_effect=swap_kind):
            faulty = g.generate(train_worlds=1, dev_worlds=1)
            row = faulty["train"][0]
            self.assertEqual(row["messages"], g.make_row(row["world"], row["query"])["messages"])
            with self.assertRaisesRegex(ValueError, "question kind differs"):
                g.audit(faulty)

    def test_renderer_duty_swap_rejected_even_when_regeneration_agrees(self):
        render = g.render_user

        def swap_duty(world, query):
            other = next(role for role in world["roles"] if role != query["target_role"])
            return render(world, dict(query, target_role=other))

        with patch.object(g, "render_user", side_effect=swap_duty):
            faulty = g.generate(train_worlds=1, dev_worlds=1)
            row = faulty["train"][0]
            self.assertEqual(row["messages"], g.make_row(row["world"], row["query"])["messages"])
            with self.assertRaisesRegex(ValueError, "duty-to-role designation differs"):
                g.audit(faulty)

    def test_duplicate_missing_view_and_cross_split_world_are_refused(self):
        data = copy.deepcopy(self.dataset)
        data["train"].append(copy.deepcopy(data["train"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            g.audit(data)
        data = copy.deepcopy(self.dataset); data["train"].pop()
        with self.assertRaisesRegex(ValueError, "twelve"):
            g.audit(data)
        data = copy.deepcopy(self.dataset)
        row = copy.deepcopy(data["train"][0]); row["split"] = row["source_split"] = "dev"
        row["id"] += "-moved"; data["dev"].append(row)
        with self.assertRaisesRegex(ValueError, "crossed partitions"):
            g.audit(data)

    def test_renaming_a_duplicate_world_does_not_create_independent_data(self):
        data = copy.deepcopy(self.dataset)
        world = copy.deepcopy(data["train"][0]["world"])
        world["base_world_id"] += "-renamed"
        world["index"] += 100
        data["train"].extend(g.make_row(world, query) for query in g.queries(world))
        with self.assertRaisesRegex(ValueError, "duplicate semantic world"):
            g.audit(data)

    def test_entity_and_template_family_leakage_are_refused(self):
        data = copy.deepcopy(self.dataset)
        # Change every derivative consistently so the dedicated pool gate catches it.
        target_id = data["dev"][0]["base_world_id"]
        for row in data["dev"]:
            if row["base_world_id"] == target_id:
                row["world"]["people"][-1] = g.PEOPLE["train"][0]
        with self.assertRaisesRegex(ValueError, "entity pool"):
            g.audit(data)
        data = copy.deepcopy(self.dataset)
        for row in data["dev"]:
            row["world"]["template_family"] = "train_club_register"
            row["template_family"] = "train_club_register"
        with self.assertRaisesRegex(ValueError, "template family"):
            g.audit(data)

    def test_template_content_overlap_detected_even_with_distinct_family_names(self):
        train_family = self.dataset["train"][0]["template_family"]
        dev_family = self.dataset["dev"][0]["template_family"]
        leaked = copy.deepcopy(g.TEMPLATES[train_family]); leaked["split"] = "dev"
        with patch.dict(g.TEMPLATES, {dev_family: leaked}):
            data = {split: [g.make_row(row["world"], row["query"]) for row in rows]
                    for split, rows in self.dataset.items()}
            with self.assertRaisesRegex(ValueError, "template-content overlap"):
                g.audit(data)

    def test_create_only_tiny_artifacts_and_no_token_budget_claim(self):
        with tempfile.TemporaryDirectory(prefix="binding-signal-fixture-") as directory:
            out = Path(directory) / "tiny"
            manifest = g.write_dataset(out, self.dataset)
            self.assertEqual(manifest["status"], "BUILD_ONLY_BLOCKED_EXTERNAL_COVERAGE")
            self.assertFalse(manifest["training_permission"])
            self.assertFalse(manifest["tokenizer_compiled"])
            self.assertEqual(manifest["human_data_review"], "NOT_PERFORMED")
            self.assertEqual(manifest["license"], "UNSPECIFIED_INTERNAL_RESEARCH")
            self.assertFalse(manifest["release_permitted"])
            self.assertTrue(manifest["independent_policy_input_semantics_verified"])
            self.assertEqual(len((out / "TRAIN.jsonl").read_text().splitlines()), 24)
            self.assertEqual(json.loads((out / "BUILD.json").read_text())["input_eval_rows_read"], 0)
            with self.assertRaises(ValueError):
                g.write_dataset(out, self.dataset)

    def test_complete_default_bank_has_no_approved_literal_contacts(self):
        dataset = g.generate()
        report = g.audit(dataset)
        self.assertEqual(report["rows"], {"train": 768, "dev": 384})
        self.assertEqual(report["worlds"], {"train": 64, "dev": 32})
        self.assertEqual(report["root_seed"], 23)
        self.assertEqual(report["world_namespace"], "BINDV2")
        self.assertTrue(all(row["parent_id"].startswith("BINDV2-") for rows in dataset.values() for row in rows))
        self.assertEqual(report["status"], "BUILD_ONLY_BLOCKED_EXTERNAL_COVERAGE")
        self.assertFalse(report["external_exclusions_complete"])
        self.assertTrue(all(not contacts for fields in report["approved_literal_exclusions"]["contacts"].values()
                            for contacts in fields.values()))

    def test_generic_proper_noun_exclusions_are_not_waived(self):
        text = "Which What Name Give Work Case State According Harbor"
        dataset = {"train": [{"id": "literal-test", "messages": [{"role": "user", "content": text}]}], "dev": []}
        before = copy.deepcopy(dataset)
        contacts = g.literal_contacts(dataset, g.approved_manifest())
        self.assertEqual({row["term"] for row in contacts["train"]["forbidden_proper_nouns"]}, set(text.split()))
        self.assertEqual(dataset, before)

    def test_literal_domain_matching_keeps_phrase_boundaries_and_unicode(self):
        manifest = {field: [] for field in g.LITERAL_FIELDS}
        manifest["forbidden_domains"] = ["museum"]
        dataset = {"train": [
            {"id": "hit", "messages": [{"role": "user", "content": "The ＭＵＳＥＵＭ."}]},
            {"id": "not-a-hit", "messages": [{"role": "user", "content": "museumlike"}]}], "dev": []}
        contact = g.literal_contacts(dataset, manifest)["train"]["forbidden_domains"]
        self.assertEqual(contact, [{"term": "museum", "row_ids": ["hit"], "row_count": 1}])

    def test_literal_contact_blocks_output_without_dropping_rows(self):
        with patch.dict(g.PEOPLE, {"train": ("Which", "What", "Name", "Give", "Work", "Case")}):
            faulty = g.generate(train_worlds=1, dev_worlds=1)
            before = copy.deepcopy(faulty)
            with tempfile.TemporaryDirectory(prefix="binding-v2-refusal-") as directory:
                out = Path(directory) / "refused"
                with self.assertRaisesRegex(ValueError, "approved literal exclusion contact"):
                    g.write_dataset(out, faulty)
                self.assertFalse(out.exists())
            self.assertEqual(faulty, before)


if __name__ == "__main__":
    unittest.main()

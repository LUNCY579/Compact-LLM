"""Synthetic source/packet controls only; no authored bank, tokenizer or native reads."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from experiments.post_training_mimo_centralized_v1.scripts import language_eval_inputs as language


def wrapper(part, kind, document):
    return {"source_path": f"sources/language_dev_cal_v1/{kind}_{part}.json",
            "raw_bytes": language.canonical(document) + b"\n"}


def spec_documents():
    documents = []
    for part, strata in (("A", language.STRATA[:4]), ("B", language.STRATA[4:])):
        worlds = []
        for stratum in strata:
            for role, prefix, count in (("DEV", "LDEV", 8), ("CAL", "LCAL", 2)):
                for ordinal in range(1, count + 1):
                    parent = f"{prefix}-{stratum}-{ordinal:02d}"
                    worlds.append({
                        "parent_id": parent, "role": role, "stratum": stratum,
                        "conversation_form": "single" if ordinal <= (4 if role == "DEV" else 1) else "multi",
                        "author_id": language.AUTHORS[part],
                        "semantic_world": {"actors": ["a colleague", "the user"], "facts": ["Synthetic world " + parent],
                            "timeline": ["The room changed before the request."], "constraints": ["Use only supplied facts."],
                            "potential_confusions": ["Old and new room assignments."], "exclusion_tags": ["synthetic test only"],
                            "task_intent": "Restate the current room.", "decisive_structure": "An explicit update overrides an earlier fact."},
                    })
        documents.append(wrapper(part, "SPECS", {"schema_version": 1, "stage": "SPEC_ONLY_NO_RENDERED_TEXT",
                                                "author_id": language.AUTHORS[part], "worlds": worlds}))
    return documents


def payload_documents(specs):
    documents = []
    for part, author in language.AUTHORS.items():
        worlds = []
        owned = [spec for spec in specs if spec["author_id"] == author]
        for spec in owned:
            messages = [{"role": "user", "content": "The current room is blue. State the current room."}]
            if spec["conversation_form"] == "multi":
                messages = [{"role": "user", "content": "The room is green."},
                            {"role": "assistant", "content": "I understand the room is green."},
                            {"role": "user", "content": "It changed to blue. State the current room."}]
            controls = []
            if spec["role"] == "CAL":
                for number, control in enumerate(language.CONTROL_IDS):
                    controls.append({"control_id": control, "response": f"Synthetic candidate wording {number}.",
                                     "author_intent": "intended_acceptable" if number < 2 else control,
                                     "intent_rationale": "Unqualified intended distinction for a synthetic control.",
                                     "qualified_reference_scores": None})
            worlds.append({"parent_id": spec["parent_id"], "world_spec_sha256": spec["world_spec_sha256"],
                           "messages": messages, "semantic_contract": {
                               "must_include": ["The current blue room."], "must_avoid": ["The earlier green room."],
                               "acceptable_variation": ["Equivalent brief phrasing."],
                               "dimensions": {dimension: "Judge this dimension from the supplied conversation." for dimension in language.DIMENSIONS}},
                           "dev_exemplar": "The room is blue." if spec["role"] == "DEV" else None,
                           "cal_controls": controls, "authorship": "assistant_source_candidate"})
        documents.append(wrapper(part, "PAYLOADS", {"schema_version": 1, "stage": "RENDERED_CANDIDATES_UNQUALIFIED",
            "spec_file_sha256": owned[0]["spec_file_sha256"], "author_id": author, "worlds": worlds}))
    return documents


def change_document(documents, index, edit):
    result = deepcopy(documents)
    document = json.loads(result[index]["raw_bytes"])
    edit(document)
    result[index]["raw_bytes"] = language.canonical(document) + b"\n"
    return result


def rating_import(packet, records=()):
    return {"schema_version": 1, "packet_sha256": packet["coordinator_mapping"]["rating_view_sha256"],
            "provenance": dict.fromkeys(language.PROVENANCE_FIELDS), "ratings": list(records)}


def rehash_packet(packet):
    """Simulate a self-consistent edited packet, without access to its source rows."""
    mapping = packet["coordinator_mapping"]
    for item in packet["rating_view"]:
        mapping["entries"][item["id"]]["rating_item_sha256"] = language.digest(item)
    mapping["rating_view_sha256"] = language.digest(packet["rating_view"])
    mapping["mapping_sha256"] = language.digest({key: value for key, value in mapping.items() if key != "mapping_sha256"})


class LanguageInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents = spec_documents()
        cls.specs = language.validate_specs(cls.documents)
        cls.payloads = payload_documents(cls.specs)
        cls.rows = language.validate_payloads(cls.specs, cls.payloads)
        cls.packet = language.build_cal_packet(cls.rows, seed=20260928, template_id="TEMPLATE_A")

    def test_exact_spec_population_roles_forms_and_original_world_hash(self):
        self.assertEqual(len(self.specs), 80)
        self.assertEqual(Counter(spec["role"] for spec in self.specs), {"DEV": 64, "CAL": 16})
        self.assertEqual(Counter((spec["role"], spec["conversation_form"]) for spec in self.specs),
                         {("DEV", "single"): 32, ("DEV", "multi"): 32, ("CAL", "single"): 8, ("CAL", "multi"): 8})
        for stratum in language.STRATA:
            self.assertEqual(Counter(spec["role"] for spec in self.specs if spec["stratum"] == stratum), {"DEV": 8, "CAL": 2})
        self.assertEqual(self.specs[0]["parent_id"], "LDEV-explanation-01")
        for spec in self.specs:
            original = {key: spec[key] for key in language.WORLD_FIELDS}
            self.assertEqual(spec["world_spec_sha256"], language.digest(original))
            document = next(doc for doc in self.documents if doc["source_path"] == spec["spec_source_path"])
            self.assertEqual(spec["spec_file_sha256"], hashlib.sha256(document["raw_bytes"]).hexdigest())

    def test_spec_role_id_author_form_and_extra_rendered_field_rejected(self):
        for field, value in (("role", "CHECK"), ("parent_id", "LDEV-explanation-09"),
                             ("author_id", language.AUTHORS["B"]), ("conversation_form", "multi"),
                             ("messages", [{"role": "user", "content": "premature rendering"}])):
            documents = change_document(self.documents, 0, lambda doc: doc["worlds"][0].update({field: value}))
            with self.subTest(field=field), self.assertRaises(ValueError):
                language.validate_specs(documents)

    def test_spec_duplicate_missing_semantic_clone_and_empty_content_rejected(self):
        edits = [lambda doc: doc["worlds"].pop(),
                 lambda doc: doc["worlds"].__setitem__(1, deepcopy(doc["worlds"][0])),
                 lambda doc: doc["worlds"][1].update(semantic_world=deepcopy(doc["worlds"][0]["semantic_world"])),
                 lambda doc: doc["worlds"][0]["semantic_world"].update(facts=[])]
        for edit in edits:
            with self.assertRaises(ValueError):
                language.validate_specs(change_document(self.documents, 0, edit))

    def test_document_labels_duplicate_keys_and_nonfinite_json_rejected(self):
        for label in ("../SPECS_A.json", "sources/language_dev_cal_v1/CHECK.json", "C:/SPECS_A.json"):
            documents = deepcopy(self.documents)
            documents[0]["source_path"] = label
            with self.subTest(label=label), self.assertRaises(ValueError):
                language.validate_specs(documents)
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}'):
            documents = deepcopy(self.documents)
            documents[0]["raw_bytes"] = raw
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                language.validate_specs(documents)

    def test_payload_population_source_binding_and_authorship_preserved(self):
        self.assertEqual(len(self.rows), 80)
        self.assertEqual(sum(len(row["cal_controls"]) for row in self.rows), 96)
        self.assertEqual(sum(row["dev_exemplar"] is not None for row in self.rows), 64)
        self.assertTrue(all(row["authorship"] == "assistant_source_candidate" for row in self.rows))
        for row in self.rows:
            self.assertEqual(row["row_sha256"], language.digest({key: value for key, value in row.items() if key != "row_sha256"}))
        self.assertEqual(self.rows, language.validate_payloads(list(reversed(self.specs)), list(reversed(self.payloads))))

    def test_altered_enriched_spec_with_old_hash_is_not_trusted(self):
        specs = deepcopy(self.specs)
        specs[0]["semantic_world"]["facts"][0] = "Altered after specification freeze."
        with self.assertRaisesRegex(ValueError, "semantic world hash drift"):
            language.validate_payloads(specs, self.payloads)

    def test_payload_raw_spec_hash_and_world_hash_drift_rejected(self):
        for edit in (lambda doc: doc.update(spec_file_sha256="a" * 64),
                     lambda doc: doc["worlds"][0].update(world_spec_sha256="b" * 64)):
            with self.assertRaises(ValueError):
                language.validate_payloads(self.specs, change_document(self.payloads, 0, edit))
        changed_specs = deepcopy(self.documents)
        changed_specs[0]["raw_bytes"] += b"\n"
        with self.assertRaises(ValueError):
            language.validate_payloads(language.validate_specs(changed_specs), self.payloads)

    def test_conversation_boundaries_system_roles_and_ending_rejected(self):
        variants = [[{"role": "system", "content": "forbidden"}],
                    [{"role": "assistant", "content": "wrong start"}],
                    [{"role": "user", "content": ""}],
                    [{"role": "user", "content": "first"}, {"role": "assistant", "content": "wrong end"}]]
        for messages in variants:
            with self.assertRaises(ValueError):
                language.validate_payloads(self.specs, change_document(
                    self.payloads, 0, lambda doc: doc["worlds"][0].update(messages=messages)))
        with self.assertRaises(ValueError):
            language.validate_payloads(self.specs, change_document(self.payloads, 0,
                lambda doc: doc["worlds"][4].update(messages=[{"role": "user", "content": "too short for multi"}])))

    def test_contract_guidance_requires_four_strings_not_reference_scores(self):
        edits = [lambda doc: doc["worlds"][0]["semantic_contract"]["dimensions"].update(adequacy=2),
                 lambda doc: doc["worlds"][0]["semantic_contract"]["dimensions"].pop("fluency"),
                 lambda doc: doc["worlds"][0]["semantic_contract"].update(must_include=[])]
        for edit in edits:
            with self.assertRaises(ValueError):
                language.validate_payloads(self.specs, change_document(self.payloads, 0, edit))

    def test_cal_controls_reject_duplicates_wrong_intent_or_qualified_scores(self):
        edits = [lambda doc: doc["worlds"][8]["cal_controls"][0].update(qualified_reference_scores={name: 2 for name in language.DIMENSIONS}),
                 lambda doc: doc["worlds"][8]["cal_controls"][2].update(author_intent="intended_acceptable"),
                 lambda doc: doc["worlds"][8]["cal_controls"][1].update(control_id="positive_a"),
                 lambda doc: doc["worlds"][8]["cal_controls"][1].update(response=doc["worlds"][8]["cal_controls"][0]["response"]),
                 lambda doc: doc["worlds"][8].update(dev_exemplar="Not allowed on CAL")]
        for edit in edits:
            with self.assertRaises(ValueError):
                language.validate_payloads(self.specs, change_document(self.payloads, 0, edit))

    def test_packet_dose_blinding_and_one_repeat_per_world(self):
        view, mapping = self.packet["rating_view"], self.packet["coordinator_mapping"]
        self.assertEqual(len(view), 112)
        self.assertEqual(len({item["id"] for item in view}), 112)
        self.assertTrue(all(set(item) == {"id", "messages", "response", "instructions"} for item in view))
        entries = mapping["entries"]
        self.assertEqual(sum(not entry["is_repeat"] for entry in entries.values()), 96)
        self.assertEqual(sum(entry["is_repeat"] for entry in entries.values()), 16)
        self.assertEqual(set(Counter(entry["parent_id"] for entry in entries.values()).values()), {7})
        self.assertEqual(set(Counter(entry["parent_id"] for entry in entries.values() if entry["is_repeat"]).values()), {1})
        by_id = {item["id"]: item for item in view}
        for key, entry in entries.items():
            self.assertEqual(entry["rating_item_sha256"], language.digest(by_id[key]))
            if entry["is_repeat"]:
                original = by_id[entry["repeat_of_id"]]
                self.assertEqual(by_id[key]["messages"], original["messages"])
                self.assertEqual(by_id[key]["response"], original["response"])
        serialized = json.dumps(view)
        for hidden in ("parent_id", "control_id", "author_intent", "is_repeat", "qualified_reference_scores", "LDEV-", "LCAL-"):
            self.assertNotIn(hidden, serialized)

    def test_packets_deterministic_and_only_registered_templates(self):
        self.assertEqual(self.packet, language.build_cal_packet(self.rows, seed=20260928, template_id="TEMPLATE_A"))
        other = language.build_cal_packet(self.rows, seed=20260929, template_id="TEMPLATE_B")
        self.assertNotEqual(self.packet["rating_view"], other["rating_view"])
        for seed, template in ((20260929, "TEMPLATE_A"), (True, "TEMPLATE_A"), (20260928, "REVIEWER_1")):
            with self.assertRaises(ValueError):
                language.build_cal_packet(self.rows, seed=seed, template_id=template)

    def test_mutated_normalized_rows_cannot_build_packet(self):
        rows = deepcopy(self.rows)
        rows[8]["cal_controls"][0]["response"] = "Altered candidate after binding."
        with self.assertRaisesRegex(ValueError, "row hash drift"):
            language.build_cal_packet(rows, seed=20260928, template_id="TEMPLATE_A")

    def test_empty_import_keeps_all_missing_and_undefined_repeat_agreement(self):
        report = language.validate_ratings(self.packet, rating_import(self.packet), rows=self.rows)
        self.assertEqual((report["expected_items"], report["supplied_items"], report["absent_items"]), (112, 0, 112))
        self.assertEqual(len(report["ratings"]), 112)
        self.assertTrue(all(set(row["scores"].values()) == {"missing"} for row in report["ratings"]))
        self.assertIsNone(report["repeat_diagnostic"]["agreement_fraction"])
        self.assertFalse(report["grading_qualified"])
        self.assertFalse(report["source_correct_membership"])
        self.assertFalse(report["e1_authorized"])

    def test_partial_scores_unjudgeable_and_provenance_are_preserved(self):
        scores = {"adequacy": 2, "relevance": "unjudgeable", "fluency": "missing", "role_consistency": 1}
        inputs = rating_import(self.packet, [{"id": self.packet["rating_view"][0]["id"], "scores": scores}])
        inputs["provenance"]["grader_id"] = "synthetic-test-grader"
        inputs["provenance"]["qualification_claim"] = True
        report = language.validate_ratings(self.packet, inputs, rows=self.rows)
        self.assertEqual(report["ratings"][0]["scores"], scores)
        self.assertEqual(report["absent_items"], 111)
        self.assertEqual(report["provenance"], inputs["provenance"])
        self.assertEqual(report["dimension_counts_including_repeats"]["relevance"]["unjudgeable"], 1)
        self.assertFalse(report["grading_qualified"])

    def test_all_full_scores_and_repeat_disagreement_remain_diagnostic(self):
        inputs = rating_import(self.packet, [{"id": item["id"], "scores": dict.fromkeys(language.DIMENSIONS, 2)}
                                           for item in self.packet["rating_view"]])
        report = language.validate_ratings(self.packet, inputs, rows=self.rows)
        self.assertEqual(report["repeat_diagnostic"]["exact_vector_agreements"], 16)
        repeated_id = next(key for key, entry in self.packet["coordinator_mapping"]["entries"].items() if entry["is_repeat"])
        next(row for row in inputs["ratings"] if row["id"] == repeated_id)["scores"]["adequacy"] = 0
        report = language.validate_ratings(self.packet, inputs, rows=self.rows)
        self.assertEqual(report["repeat_diagnostic"]["fully_scored_pairs"], 16)
        self.assertEqual(report["repeat_diagnostic"]["exact_vector_agreements"], 15)
        self.assertFalse(report["reference_labels_qualified"])
        self.assertFalse(report["grading_qualified"])

    def test_invalid_ratings_extra_duplicate_id_dimension_and_score_rejected(self):
        record = {"id": self.packet["rating_view"][0]["id"], "scores": dict.fromkeys(language.DIMENSIONS, 2)}
        for score in (True, 2.0, -1, 3, None, "2", "unknown"):
            changed = deepcopy(record)
            changed["scores"]["adequacy"] = score
            with self.subTest(score=score), self.assertRaises(ValueError):
                language.validate_ratings(self.packet, rating_import(self.packet, [changed]), rows=self.rows)
        for records in ([record, record], [{**record, "id": "unknown"}], [{**record, "expected_label": 2}],
                        [{**record, "scores": {"adequacy": 2}}]):
            with self.assertRaises(ValueError):
                language.validate_ratings(self.packet, rating_import(self.packet, records), rows=self.rows)

    def test_packet_content_mapping_or_import_identity_drift_rejected(self):
        for change in ("response", "extra", "mapping", "order"):
            packet = deepcopy(self.packet)
            if change == "response":
                packet["rating_view"][0]["response"] += " Edited."
            elif change == "extra":
                packet["rating_view"][0]["author_intent"] = "leak"
            elif change == "mapping":
                next(iter(packet["coordinator_mapping"]["entries"].values()))["is_repeat"] = True
            else:
                packet["rating_view"].reverse()
            with self.subTest(change=change), self.assertRaises(ValueError):
                language.validate_ratings(packet, rating_import(self.packet), rows=self.rows)
        inputs = rating_import(self.packet)
        inputs["packet_sha256"] = "a" * 64
        with self.assertRaises(ValueError):
            language.validate_ratings(self.packet, inputs, rows=self.rows)

    def test_rehashed_extra_mapping_fields_or_false_repeat_provenance_are_rejected(self):
        for change in ("extra", "source", "repeat_source"):
            packet = deepcopy(self.packet)
            mapping = packet["coordinator_mapping"]
            entry = next(value for value in mapping["entries"].values() if value["is_repeat"])
            if change == "extra":
                entry["qualified"] = True
            elif change == "source":
                entry["spec_source_path"] = "sources/language_dev_cal_v1/CHECK.json"
            else:
                entry["payload_file_sha256"] = "a" * 64
            mapping["mapping_sha256"] = language.digest({key: value for key, value in mapping.items() if key != "mapping_sha256"})
            with self.subTest(change=change), self.assertRaises(ValueError):
                language.validate_ratings(packet, rating_import(packet), rows=self.rows)

    def test_rating_import_requires_fresh_source_rows_without_fallback(self):
        with self.assertRaises(TypeError):
            language.validate_ratings(self.packet, rating_import(self.packet))
        changed = change_document(self.payloads, 0, lambda doc: doc["worlds"][8]["cal_controls"][0].update(
            response="A new source revision requires a newly bound packet."))
        fresh_rows = language.validate_payloads(language.validate_specs(self.documents), changed)
        with self.assertRaisesRegex(ValueError, "freshly validated source rows"):
            language.validate_ratings(self.packet, rating_import(self.packet), rows=fresh_rows)

    def test_rehashed_nonrepeat_response_or_invalid_display_schema_is_rejected(self):
        for change in ("response", "nonstring", "system_message", "blank_message"):
            packet = deepcopy(self.packet)
            mapping = packet["coordinator_mapping"]["entries"]
            repeated_ids = {entry["repeat_of_id"] for entry in mapping.values() if entry["is_repeat"]}
            item = next(value for value in packet["rating_view"] if not mapping[value["id"]]["is_repeat"]
                        and value["id"] not in repeated_ids)
            if change == "response":
                item["response"] = "Unbound replacement response."
            elif change == "nonstring":
                item["response"] = 17
            elif change == "system_message":
                item["messages"][0]["role"] = "system"
            else:
                item["messages"][0]["content"] = " "
            rehash_packet(packet)
            with self.subTest(change=change), self.assertRaises(ValueError):
                language.validate_ratings(packet, rating_import(packet), rows=self.rows)

    def test_fully_rehashed_response_or_same_parent_prior_conversation_cannot_replace_source(self):
        for change in ("response_and_control_hash", "whole_parent_prior_context"):
            packet = deepcopy(self.packet)
            entries = packet["coordinator_mapping"]["entries"]
            if change == "response_and_control_hash":
                repeated_ids = {entry["repeat_of_id"] for entry in entries.values() if entry["is_repeat"]}
                item = next(value for value in packet["rating_view"] if not entries[value["id"]]["is_repeat"]
                            and value["id"] not in repeated_ids)
                item["response"] = "Rehashed replacement not found in the freshly validated source."
                entry = entries[item["id"]]
                entry["control_sha256"] = language.digest({"control_id": entry["control_id"],
                    "response": item["response"], "author_intent": entry["author_intent"],
                    "intent_rationale": entry["intent_rationale"], "qualified_reference_scores": None})
            else:
                parent = next(entry["parent_id"] for entry in entries.values() if entry["parent_id"].endswith("-02"))
                for item in packet["rating_view"]:
                    if entries[item["id"]]["parent_id"] == parent:
                        item["messages"][1]["content"] = "Rewritten prior assistant context shared by all seven displayed records."
            rehash_packet(packet)
            # All local packet hashes/topology agree; only binding to fresh source
            # records can establish that the content still matches the source.
            language._packet(packet)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "freshly validated source rows"):
                language.validate_ratings(packet, rating_import(packet), rows=self.rows)

    def test_fresh_import_has_no_framework_or_environment_side_effect(self):
        command = ("import os,sys; before=dict(os.environ); "
                   "from experiments.post_training_mimo_centralized_v1.scripts import language_eval_inputs; "
                   "assert dict(os.environ)==before; "
                   "assert not any(name in sys.modules for name in ('torch','transformers','tokenizers','safetensors')); "
                   "assert not any('tokenize_bridge' in name for name in sys.modules)")
        result = subprocess.run([sys.executable, "-B", "-c", command], cwd=Path(__file__).resolve().parents[3],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

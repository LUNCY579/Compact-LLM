"""Small failure-oriented controls; synthetic arrays are not tokenization proof."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
import unittest
from unittest.mock import patch

from experiments.post_training_mimo_centralized_v1.scripts import e1_train_reader as reader


def pin(raw):
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def array(row_id, component, *, source=None, prompt=4, prior=0):
    ids = [10] * prompt + [11, 151645]
    row = {"row_id": row_id, "parent_id": "parent-" + row_id, "family": "family-" + component,
           "role": "TRAIN", "training_role_eligible": True, "training_eligibility": False,
           "prompt_tokens": prompt, "completion_tokens": 2, "nonpadding_tokens": len(ids),
           "padding_tokens": 0, "completion_eos_id": 151645, "input_ids": ids,
           "labels": [-100] * prompt + ids[prompt:], "attention_mask": [1] * len(ids),
           "completion_mask": [0] * prompt + [1, 1], "input_token_ids_sha256": reader.digest(ids),
           "tokenized_prompt_sha256": reader.digest(ids[:prompt]), "completion_token_ids_sha256": reader.digest(ids[prompt:]),
           "source_row_sha256": reader.digest(source or {"row_id": row_id}), "target_json_tokens": 1}
    if component == "language_replay":
        row.update(component=component, prior_assistant_turns=prior, prior_turns_supervised=False)
    return row


def language_case():
    source = {"id": "language-1", "system_policy_version": "fixture",
              "messages": [{"role": "system", "content": "System"}, {"role": "user", "content": "First"},
                           {"role": "assistant", "content": "Earlier reply"}, {"role": "user", "content": "Second"},
                           {"role": "assistant", "content": "Last reply"}],
              "metadata": {"split": "train", "base_template_family": "language-1", "reply_class": "reply",
                           "preservation_replay_for": "rineya_v3", "preservation_source_version": "v2.0"}}
    row = array("language-1", "language_replay", source=source, prior=1)
    row.update(parent_id="v2.0:language-1", family="reply")
    original = deepcopy(source)
    del original["metadata"]["preservation_replay_for"]
    del original["metadata"]["preservation_source_version"]
    lineage = {"row_id": row["row_id"], "cluster": row["parent_id"], "reply_class": row["family"],
               "source_record_modified": False, "training_eligibility": False,
               "original_v2_canonical_sha256": reader.digest(original)}
    profile = {"row_id": row["row_id"], "parent_cluster": row["parent_id"], "reply_class": row["family"],
               "prior_assistant_turns": 1, "all_prior_turns_masked": True, "prompt_tokens": row["prompt_tokens"],
               "completion_tokens": row["completion_tokens"], "nonpadding_tokens": row["nonpadding_tokens"],
               "prompt_token_ids_sha256": row["tokenized_prompt_sha256"],
               "completion_token_ids_sha256": row["completion_token_ids_sha256"], "supervised_positions": 2}
    return source, row, lineage, profile


class Case:
    def __init__(self, root):
        self.root = Path(root)
        source, lang, lineage, profile = language_case()
        self.arrays = {"acquisition": [array("acq-1", "acquisition"), array("acq-2", "acquisition", prompt=5)],
                       "deterministic_replay": [array("replay-1", "deterministic_replay", prompt=6)],
                       "language_replay": [lang]}
        self.source, self.lineage, self.profile = source, [lineage], [profile]
        self.components = {component: {"arrays": component + "/TRAIN.tokenized.jsonl"} for component in self.arrays}
        self.components["language_replay"]["source"] = "language_replay/TRAIN_CANDIDATES.jsonl"
        self.files, self.binding = {}, {"rows": [], "locations": {}}
        self.write()
        keys = [row["key"] for row in self.binding["rows"]]
        self.plan = {"ordered_rows": deepcopy(self.binding["rows"]), "steps": []}
        self.add_step([keys[2], keys[0]])
        self.add_step([keys[3], keys[1]])

    def write(self):
        self.files = {}; self.binding = {"rows": [], "locations": {}}
        for component, arrays in self.arrays.items():
            relative = self.components[component]["arrays"]
            path = self.root / relative; path.parent.mkdir(parents=True, exist_ok=True)
            raw = b""; offset = 0
            for number, row in enumerate(arrays, 1):
                line = reader.canonical(row) + b"\n"
                key = component + ":" + row["row_id"]
                self.binding["rows"].append({"key": key, "component": component, "role": "TRAIN", "parent_id": row["parent_id"],
                                             "nonpadding_tokens": row["nonpadding_tokens"], "completion_tokens": row["completion_tokens"]})
                self.binding["locations"][key] = {"file": relative, "line": number, "byte_offset": offset,
                    "raw_line_sha256": hashlib.sha256(line).hexdigest(), "source_row_sha256": row["source_row_sha256"],
                    "input_token_ids_sha256": row["input_token_ids_sha256"]}
                raw += line; offset += len(line)
            path.write_bytes(raw); self.files[relative] = pin(raw)
        relative = self.components["language_replay"]["source"]
        path = self.root / relative; path.parent.mkdir(parents=True, exist_ok=True)
        raw = reader.canonical(self.source) + b"\n"; path.write_bytes(raw); self.files[relative] = pin(raw)

    def add_step(self, keys):
        by_key = {row["key"]: row for row in self.binding["rows"]}
        self.plan["steps"].append({"optimizer_step": len(self.plan["steps"]) + 1, "row_keys": list(keys), "learning_rate": 0.00001,
            "checkpoint_after": False, "dev_evaluation_after": False,
            "component_rows": {component: sum(by_key[key]["component"] == component for key in keys) for component in self.components},
            "nonpadding_tokens": sum(by_key[key]["nonpadding_tokens"] for key in keys),
            "completion_tokens": sum(by_key[key]["completion_tokens"] for key in keys)})

    def open(self):
        return reader.BoundTrainReader.for_test(self.root, components=self.components, files=self.files, plan=self.plan,
            binding=self.binding, lineage=self.lineage, profile=self.profile, expected_prior=1)


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.case = Case(self.temp.name)

    def rejects(self, code, function):
        with self.assertRaises(reader.ReaderError) as context:
            function()
        self.assertEqual(context.exception.code, code)

    def mutate_array(self, component, mutator):
        mutator(self.case.arrays[component][0])
        self.case.write()
        self.case.plan["ordered_rows"] = deepcopy(self.case.binding["rows"])

    def test_actual_rows_follow_frozen_schedule_and_are_not_resident(self):
        bound = self.case.open()
        items = list(bound.iter_plan())
        expected = [key for step in self.case.plan["steps"] for key in step["row_keys"]]
        self.assertEqual([item["schedule"]["row_key"] for item in items], expected)
        self.assertEqual([item["schedule"]["sequence_index"] for item in items], list(range(4)))
        self.assertEqual([item["schedule"]["microbatch_index"] for item in items], [1, 2, 1, 2])
        self.assertEqual(sum(item["row"]["completion_tokens"] for item in items), 8)
        self.assertEqual(bound.receipt()["retained_token_rows"], 0)
        self.assertTrue(bound.receipt()["fixture_only"])
        self.assertEqual(bound.receipt()["maximum_simultaneous_file_handles"], 2)
        self.assertEqual(bound.receipt()["maximum_lookup_file_handles"], 1)
        self.assertEqual(bound.receipt()["persistent_file_handles"], 0)
        self.assertLess(bound.receipt()["retained_metadata_bytes"], 100000)
        for field in ("native_qualified", "native_launch_authorized", "training_authorized", "e1_authorized", "native_executor", "CHECK_FINAL_access"):
            self.assertIs(bound.receipt()[field], False)

    def test_returned_row_and_external_schedule_mutation_do_not_change_reader(self):
        bound = self.case.open(); key = self.case.plan["steps"][0]["row_keys"][0]
        row = bound.get(key); row["labels"][0] = 999
        self.case.plan["steps"][0]["row_keys"].reverse()
        self.case.profile[0]["prior_assistant_turns"] = 100
        self.assertEqual(bound.get(key)["labels"][0], -100)
        self.assertEqual(next(bound.iter_plan())["schedule"]["row_key"], key)
        self.assertEqual(bound.get("language_replay:language-1")["prior_assistant_turns"], 1)
        with self.assertRaises(TypeError):
            bound._index[key] = bound._index[key]

    def test_optimizer_boundary_iteration_and_invalid_partial_cursor(self):
        bound = self.case.open(); items = list(bound.iter_plan(start_optimizer_step=2))
        self.assertEqual([item["schedule"]["sequence_index"] for item in items], [2, 3])
        self.assertEqual(list(bound.iter_plan(start_optimizer_step=3)), [])
        for value in (0, 4, True, 1.5):
            self.rejects("START_STEP", lambda: list(bound.iter_plan(start_optimizer_step=value)))

    def test_reordered_bound_population_is_rejected_even_with_identical_keys(self):
        self.case.binding["rows"][0:2] = reversed(self.case.binding["rows"][0:2])
        self.case.plan["ordered_rows"] = deepcopy(self.case.binding["rows"])
        self.rejects("ORDERED_ROWS", self.case.open)

    def test_reordered_plan_population_and_wrong_component_schedule_key(self):
        original = deepcopy(self.case.plan)
        self.case.plan["ordered_rows"].reverse()
        self.rejects("ORDERED_ROWS", self.case.open)
        self.case.plan = original
        self.case.plan["steps"][0]["row_keys"][0] = "acquisition:replay-1"
        self.rejects("PLAN_KEY", self.case.open)
        self.case = Case(self.temp.name)
        self.mutate_array("language_replay", lambda row: row.update(component="acquisition"))
        self.rejects("WRONG_COMPONENT", self.case.open)

    def test_duplicate_schedule_key_cannot_replace_an_unconsumed_row(self):
        key = self.case.plan["steps"][0]["row_keys"][0]
        self.case.plan["steps"] = []
        self.case.add_step([key] * 4)
        self.rejects("PLAN_COVERAGE", self.case.open)

    def test_row_identity_reuse_across_components_is_rejected(self):
        self.mutate_array("deterministic_replay", lambda row: row.update(row_id="acq-1"))
        self.rejects("DUPLICATE_ROW", self.case.open)

    def test_prompt_labels_and_dtype_violations_are_rejected(self):
        for mutation in (lambda row: row["labels"].__setitem__(0, 10),
                         lambda row: row["labels"].__setitem__(-1, -100),
                         lambda row: row["attention_mask"].__setitem__(0, True),
                         lambda row: row["input_ids"].__setitem__(0, True)):
            with self.subTest(mutation=mutation):
                self.case = Case(self.temp.name)
                self.mutate_array("acquisition", mutation)
                self.rejects("ARRAY_CONTRACT", self.case.open)

    def test_eos_count_and_all_one_mask_contract(self):
        for mutation in (lambda row: row.update(nonpadding_tokens=99),
                         lambda row: row["input_ids"].__setitem__(-1, 151643),
                         lambda row: row["input_ids"].__setitem__(-2, 151645),
                         lambda row: row["attention_mask"].__setitem__(2, 0),
                         lambda row: row.update(padding_tokens=1)):
            with self.subTest(mutation=mutation):
                self.case = Case(self.temp.name); self.mutate_array("acquisition", mutation)
                self.rejects("ARRAY_CONTRACT", self.case.open)

    def test_ce512_and_no_packing_cache_custom_positions_or_eligibility(self):
        for field in reader.FORBIDDEN_ARRAY_FIELDS:
            with self.subTest(field=field):
                self.case = Case(self.temp.name); self.mutate_array("acquisition", lambda row: row.update({field: False}))
                self.rejects("UNSUPPORTED_FORWARD_FIELDS", self.case.open)
        self.case = Case(self.temp.name); self.case.arrays["acquisition"][0] = array("acq-1", "acquisition", prompt=511)
        self.case.write(); self.case.plan["ordered_rows"] = deepcopy(self.case.binding["rows"])
        self.rejects("FORWARD_SCOPE", self.case.open)
        self.case = Case(self.temp.name); self.mutate_array("acquisition", lambda row: row.update(training_eligibility=True))
        self.rejects("ARRAY_NO_GRANT", self.case.open)

    def test_prior_language_reply_count_and_full_prompt_mask_are_rechecked(self):
        self.mutate_array("language_replay", lambda row: row.update(prior_assistant_turns=0))
        self.rejects("LANGUAGE_SOURCE_JOIN", self.case.open)
        self.case = Case(self.temp.name); self.case.profile[0]["all_prior_turns_masked"] = False
        self.rejects("LANGUAGE_SOURCE_JOIN", self.case.open)
        self.case = Case(self.temp.name); self.mutate_array("language_replay", lambda row: row["labels"].__setitem__(1, 10))
        self.rejects("ARRAY_CONTRACT", self.case.open)

    def test_language_original_body_and_source_eligibility_cannot_drift(self):
        self.case.lineage[0]["original_v2_canonical_sha256"] = "0" * 64
        self.rejects("LANGUAGE_SOURCE_JOIN", self.case.open)
        self.case = Case(self.temp.name); self.case.source["metadata"]["split"] = "validation"
        self.case.arrays["language_replay"][0]["source_row_sha256"] = reader.digest(self.case.source)
        self.case.write(); self.case.plan["ordered_rows"] = deepcopy(self.case.binding["rows"])
        self.rejects("LANGUAGE_SOURCE_JOIN", self.case.open)

    def test_location_offset_line_and_hash_binding_are_exact(self):
        key = self.case.binding["rows"][0]["key"]
        for field, value in (("byte_offset", 1), ("byte_offset", False), ("line", 2), ("line", True), ("raw_line_sha256", "0" * 64)):
            with self.subTest(field=field):
                original = self.case.binding["locations"][key][field]
                self.case.binding["locations"][key][field] = value
                self.rejects("LOCATION_BINDING", self.case.open)
                self.case.binding["locations"][key][field] = original

    def test_boolean_plan_dose_cannot_masquerade_as_integer_count(self):
        self.case.plan["steps"][0]["component_rows"]["acquisition"] = True
        self.rejects("PLAN_STEP_DOSE", self.case.open)

    def test_corrupt_private_offset_cannot_bypass_line_hash_check(self):
        bound = self.case.open(); key = self.case.binding["rows"][0]["key"]
        entries = dict(bound._index); entries[key] = replace(entries[key], offset=1)
        bound._index = MappingProxyType(entries)
        self.rejects("LOCATION_BOUNDARY", lambda: bound.get(key))

    def test_file_hash_drift_during_index_and_appended_oversize_line_fail(self):
        relative = self.case.components["acquisition"]["arrays"]
        path = self.case.root / relative; path.write_bytes(path.read_bytes() + b"\n")
        self.rejects("BLANK_JSONL_LINE", self.case.open)
        self.case = Case(self.temp.name); path = self.case.root / relative
        path.write_bytes(b" " * (reader.LINE_BYTES + 1) + b"\n")
        self.rejects("LINE_SIZE_LIMIT", self.case.open)

    def test_same_size_write_after_index_rejected_even_with_restored_mtime(self):
        bound = self.case.open(); key = self.case.binding["rows"][0]["key"]
        path = self.case.root / self.case.components["acquisition"]["arrays"]
        original = path.stat(); raw = path.read_bytes().replace(b'"family-acquisition"', b'"family-acquisitioX"', 1)
        path.write_bytes(raw); os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
        self.rejects("FILE_CHANGED", lambda: bound.get(key))

    def test_line_hash_and_final_full_hash_detect_drift_when_stat_is_spoofed(self):
        bound = self.case.open(); key = self.case.binding["rows"][0]["key"]
        path = self.case.root / self.case.components["acquisition"]["arrays"]
        snapshots = {str((bound._root / rel).resolve()): value[0] for rel, value in bound._snapshots.items()}
        path.write_bytes(path.read_bytes().replace(b'"family-acquisition"', b'"family-acquisitioX"', 1))
        with patch.object(reader, "_stat", side_effect=lambda p: snapshots[str(p.resolve())]):
            self.rejects("LINE_HASH_DRIFT", lambda: bound.get(key))
            self.rejects("FILE_PIN_DRIFT", lambda: bound.verify_unchanged(full_hash=True))

    def test_source_file_drift_invalidates_every_lookup_and_file_replacement(self):
        bound = self.case.open(); key = self.case.binding["rows"][0]["key"]
        path = self.case.root / self.case.components["language_replay"]["source"]
        path.write_bytes(path.read_bytes().replace(b"Earlier reply", b"Changed reply"))
        self.rejects("FILE_CHANGED", lambda: bound.get(key))
        self.case = Case(self.temp.name); bound = self.case.open()
        path = self.case.root / self.case.components["acquisition"]["arrays"]
        replacement = path.with_suffix(".replacement"); replacement.write_bytes(path.read_bytes()); os.replace(replacement, path)
        self.rejects("FILE_CHANGED", lambda: bound.get(key))

    def test_closed_path_absolute_windows_drive_and_parent_escape_refused(self):
        for path in ("../TRAIN.tokenized.jsonl", "C:/TRAIN.tokenized.jsonl", "C:TRAIN.tokenized.jsonl",
                     "CHECK/TRAIN.tokenized.jsonl", "fresh_sealed/TRAIN.tokenized.jsonl", "deterministic_final/TRAIN.tokenized.jsonl"):
            with self.subTest(path=path):
                self.rejects("PATH_ESCAPE" if ".." in path or path.startswith("C:") else "CLOSED_PATH",
                             lambda: reader.safe_path(self.case.root, path))
        self.mutate_array("acquisition", lambda row: row.update(role="CHECK"))
        self.rejects("ARRAY_CONTRACT", self.case.open)

    def test_fixture_cannot_bind_production_paths_or_factory(self):
        self.case.files[reader.PLAN_RELATIVE] = {"sha256": "0" * 64, "bytes": 1}
        self.rejects("FIXTURE_PATH", self.case.open)
        self.rejects("FACTORY_REQUIRED", lambda: reader.BoundTrainReader(None))
        self.rejects("PRODUCTION_METADATA_PATH", lambda: reader.BoundTrainReader.open_production(self.case.root,
            plan_relative="toy/PLAN.json", plan_pin={}, binding_relative="toy/DATA_BINDING.json", binding_pin={},
            scope_receipt_relative="toy/receipt.json", scope_receipt_pin={}))

    def test_metadata_limits_and_duplicate_json_keys_before_unbounded_parse(self):
        self.rejects("METADATA_SIZE_LIMIT", lambda: reader._read_metadata(self.case.root, "PLAN.json",
            {"sha256": "0" * 64, "bytes": reader.METADATA_FILE_BYTES + 1}, {}, [0]))
        self.rejects("METADATA_SIZE_LIMIT", lambda: reader._read_metadata(self.case.root, "PLAN.json",
            {"sha256": "0" * 64, "bytes": 1}, {}, [reader.COMBINED_METADATA_BYTES]))
        self.rejects("DUPLICATE_JSON_KEY", lambda: reader.parse(b'{"input_ids":[],"input_ids":[]}'))
        self.rejects("NONFINITE_JSON", lambda: reader.parse(b'{"loss":NaN}'))

    def test_closed_reader_never_returns_rows_or_claims_evidence(self):
        bound = self.case.open(); bound.close()
        self.rejects("READER_CLOSED", lambda: bound.get(self.case.binding["rows"][0]["key"]))
        self.rejects("READER_CLOSED", bound.receipt)


if __name__ == "__main__":
    unittest.main()

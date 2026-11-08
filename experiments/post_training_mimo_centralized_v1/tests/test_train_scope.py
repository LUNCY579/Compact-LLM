"""Stdlib source-join controls; synthetic token IDs are not tokenization evidence."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.post_training_mimo_centralized_v1.scripts import train_scope as scope


PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]


class NoBudget:
    def check(self):
        pass


def toy_array(source, component, *, row_id=None, parent_id=None, family=None, prompt=3, prior=0):
    ids = [10] * prompt + [11, 151645]
    record = {"row_id": row_id or source["row_id"], "parent_id": parent_id or source["parent_id"],
              "family": family or source["family"], "role": "TRAIN", "training_role_eligible": True,
              "training_eligibility": False, "prompt_tokens": prompt, "completion_tokens": 2,
              "nonpadding_tokens": prompt + 2, "padding_tokens": 0, "completion_eos_id": 151645,
              "input_ids": ids, "labels": [-100] * prompt + ids[prompt:], "attention_mask": [1] * len(ids),
              "completion_mask": [0] * prompt + [1, 1], "input_token_ids_sha256": scope.digest(ids),
              "tokenized_prompt_sha256": scope.digest(ids[:prompt]), "completion_token_ids_sha256": scope.digest(ids[prompt:]),
              "source_row_sha256": scope.digest(source), "target_json_tokens": 1}
    if component != "language_replay":
        record.update(variant_id=source["variant_id"], source_prompt_sha256=source["prompt_sha256"],
                      source_target_sha256=source["target_sha256"], template_kwargs={"add_generation_prompt": True, "enable_thinking": False},
                      token_hash_encoding="UTF8 canonical compact JSON integer list; SHA256")
        if component == "deterministic_replay":
            record.update({field: source[field] for field in ("stratum", "domain", "canonical_parent_sha256", "split_sha256", "source_revision")})
    else:
        record.update(prior_assistant_turns=prior, prior_turns_supervised=False)
    return record


class TrainScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((PACKAGE / scope.MANIFEST_RELATIVE).read_bytes())
        cls.modules = {name: scope.load_verified_module(ROOT, relative, cls.manifest["files"][relative])
                       for name, relative in scope.CODE_PATHS.items()}

    def deterministic_case(self, component="acquisition"):
        name = "bridge_data" if component == "acquisition" else "retention_data"
        verifier = self.modules[name]
        spec = self.manifest["components"][component]
        with (ROOT / spec["source"]).open("rb") as stream:
            source = json.loads(stream.readline())
        return source, toy_array(source, component), spec, verifier

    def join(self, source, array, component, spec, verifier=None, **kwargs):
        return scope.validate_source_join(source, array, component, spec, verifier, self.modules["e1_data"], **kwargs)

    def test_frozen_manifest_and_interface_are_exact_data_only(self):
        raw = (PACKAGE / scope.MANIFEST_RELATIVE).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), scope.MANIFEST_SHA256)
        scope.validate_manifest(self.manifest)
        self.assertEqual(len(scope.FILE_INVENTORY), 28)
        for field in ("native_qualified", "training_authorized", "e1_authorized", "CHECK_FINAL_access", "native_executor"):
            altered = deepcopy(self.manifest); altered[field] = True
            with self.subTest(field=field), self.assertRaises(scope.ScopeError):
                scope.validate_manifest(altered)
        for field, value in (("maximum_sequence_tokens", 1024), ("microbatch", 2), ("packing", True),
                             ("use_cache", True), ("target_truncation", True)):
            altered = deepcopy(self.manifest); altered["consumer"][field] = value
            with self.subTest(field=field), self.assertRaises(scope.ScopeError):
                scope.validate_manifest(altered)

    def test_manifest_identity_fails_before_any_helper_import(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "repo/experiments/post_training_mimo_centralized_v1"
            path = package / scope.MANIFEST_RELATIVE; path.parent.mkdir(parents=True)
            path.write_bytes((PACKAGE / scope.MANIFEST_RELATIVE).read_bytes() + b" ")
            with patch.object(scope, "load_verified_module", side_effect=AssertionError("no helper import")):
                report = scope.validate_train_scope(package)
            self.assertEqual(report["status"], "BLOCKED")
            self.assertEqual(report["reasons"][0]["code"], "MANIFEST_IDENTITY")
            self.assertFalse(report["e1_authorized"])

    def test_closed_and_escaping_paths_are_rejected_before_payload_open(self):
        for relative in ("../outside.json", "C:/outside.json", "artifacts/CHECK/rows.jsonl", "artifacts/final/rows.jsonl",
                         "artifacts/test2.jsonl", "artifacts/fresh_sealed/payload.jsonl"):
            with self.subTest(relative=relative), self.assertRaises(scope.ScopeError):
                scope.safe_path(ROOT, relative)
        with patch.object(scope.Path, "read_bytes", side_effect=AssertionError("closed payload not opened")):
            report = scope.validate_train_scope(PACKAGE, manifest_path=ROOT / "CHECK.json")
        self.assertEqual(report["status"], "BLOCKED")
        self.assertEqual(report["reasons"][0]["code"], "MANIFEST_PATH")

    def test_helper_bytes_verified_before_execution_and_preparation_modules_denied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); relative = scope.CODE_PATHS["bridge_data"]
            path = root / relative; path.parent.mkdir(parents=True)
            raw = b"raise AssertionError('untrusted helper executed')\n"; path.write_bytes(raw)
            with self.assertRaises(scope.ScopeError) as caught:
                scope.load_verified_module(root, relative, {"sha256": "0" * 64, "bytes": len(raw)})
            self.assertEqual(caught.exception.code, "HELPER_DRIFT")
            with self.assertRaises(scope.ScopeError):
                scope.load_verified_module(root, scope.PREFIX + "scripts/tokenize_bridge.py", {"sha256": "0" * 64, "bytes": 1})

    def test_jsonl_limits_duplicates_and_nonfinite_values_are_rejected(self):
        for raw in (b"\n", b'{"id":1,"id":2}\n', b'{"id":NaN}\n', b"x" * 65537):
            with self.subTest(size=len(raw)), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "TRAIN.jsonl"; path.write_bytes(raw)
                with self.assertRaises((scope.ScopeError, ValueError)):
                    list(scope.jsonl_rows(path, NoBudget()))

    def test_compact_source_index_joins_by_identity_not_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "TRAIN.jsonl"
            path.write_bytes(b'{"row_id":"second","target":"b"}\n{"row_id":"first","target":"a"}\n')
            index = scope.source_index(path, "acquisition", 2, NoBudget())
            with path.open("rb") as stream:
                stream.seek(index["first"][1]); row = scope.parse(stream.read(index["first"][2]))
            self.assertEqual(row["target"], "a")
            self.assertEqual(index["first"][0], 2)
            path.write_bytes(b'{"row_id":"same"}\n{"row_id":"same"}\n')
            with self.assertRaises(scope.ScopeError):
                scope.source_index(path, "acquisition", 2, NoBudget())

    def test_deterministic_source_body_and_reference_join_is_distinct_from_shape_only(self):
        for component in ("acquisition", "deterministic_replay"):
            source, array, spec, verifier = self.deterministic_case(component)
            row = self.join(source, array, component, spec, verifier)
            self.assertEqual(row["source_canonical_sha256"], scope.digest(source))
            self.assertEqual(row["target_text_sha256"], hashlib.sha256(source["target"].encode()).hexdigest())
            altered = deepcopy(source); altered["prompt"] += " semantic drift"
            with self.subTest(component=component), self.assertRaises(scope.ScopeError) as caught:
                self.join(altered, array, component, spec, verifier)
            self.assertEqual(caught.exception.code, "SOURCE_BODY_JOIN")

    def test_rehashed_semantically_wrong_target_or_prompt_is_still_rejected(self):
        for field, value in (("target", '{"answer":999}'), ("prompt", "A different task")):
            source, array, spec, verifier = self.deterministic_case()
            source[field] = value
            source[field + "_sha256"] = hashlib.sha256(value.encode()).hexdigest()
            array["source_row_sha256"] = scope.digest(source)
            array["source_" + field + "_sha256"] = source[field + "_sha256"]
            with self.subTest(field=field), self.assertRaises(scope.ScopeError):
                self.join(source, array, "acquisition", spec, verifier)

    def test_rehashed_acquisition_family_and_structure_drift_are_rejected(self):
        for field, value in (("family", "finite_set_queries"), ("structure_id", "different_structure")):
            source, array, spec, verifier = self.deterministic_case()
            source[field] = value
            if field == "family": array[field] = value
            array["source_row_sha256"] = scope.digest(source)
            with self.subTest(field=field), self.assertRaises(scope.ScopeError) as caught:
                self.join(source, array, "acquisition", spec, verifier)
            self.assertEqual(caught.exception.code, "SOURCE_CANONICAL_PARENT")

    def test_target_masks_eos_and_512_boundary_are_checked_without_tokenization(self):
        source, array, spec, verifier = self.deterministic_case()
        for mutation in ("prior_label", "target_label", "mask_zero", "missing_eos", "packing", "too_long"):
            record = deepcopy(array)
            if mutation == "prior_label": record["labels"][0] = record["input_ids"][0]
            elif mutation == "target_label": record["labels"][-2] = -100
            elif mutation == "mask_zero": record["attention_mask"][0] = 0
            elif mutation == "missing_eos": record["input_ids"][-1] = 11
            elif mutation == "packing": record["packed"] = True
            else: record = toy_array(source, "acquisition", prompt=511)
            with self.subTest(mutation=mutation), self.assertRaises((scope.ScopeError, ValueError)):
                self.join(source, record, "acquisition", spec, verifier)
        record = toy_array(source, "acquisition", prompt=510)
        self.join(source, record, "acquisition", spec, verifier)
        scope.validate_consumer(record, scope.CONSUMER, self.modules["laptop_profile"])

    def language_case(self):
        spec = self.manifest["components"]["language_replay"]
        # Use an actual selected multi-turn source body, but synthetic IDs only.
        with (ROOT / spec["source"]).open("rb") as stream:
            source = next(scope.parse(raw) for raw in stream if sum(message["role"] == "assistant" for message in scope.parse(raw)["messages"][:-1]) > 0)
        prior = sum(message["role"] == "assistant" for message in source["messages"][:-1])
        cluster = "v2.0:" + source["metadata"]["base_template_family"]
        array = toy_array(source, "language_replay", row_id=source["id"], parent_id=cluster,
                          family=source["metadata"]["reply_class"], prior=prior)
        original = deepcopy(source)
        for field in ("preservation_replay_for", "preservation_source_version"): del original["metadata"][field]
        lineage = {"row_id": source["id"], "cluster": cluster, "reply_class": array["family"],
                   "source_record_modified": False, "training_eligibility": False, "original_v2_canonical_sha256": scope.digest(original)}
        profile = {"row_id": source["id"], "parent_cluster": cluster, "reply_class": array["family"],
                   "prior_assistant_turns": prior, "all_prior_turns_masked": True,
                   "prompt_tokens": array["prompt_tokens"], "completion_tokens": array["completion_tokens"],
                   "nonpadding_tokens": array["nonpadding_tokens"], "prompt_token_ids_sha256": array["tokenized_prompt_sha256"],
                   "completion_token_ids_sha256": array["completion_token_ids_sha256"], "supervised_positions": array["completion_tokens"]}
        return source, array, spec, lineage, profile

    def test_multiturn_language_preserves_context_but_supervises_only_final_reply(self):
        source, array, spec, lineage, profile = self.language_case()
        row = self.join(source, array, "language_replay", spec, lineage=lineage, profile=profile)
        self.assertGreater(row["prior_assistant_turns"], 0)
        self.assertEqual(row["target_text_sha256"], hashlib.sha256(source["messages"][-1]["content"].encode()).hexdigest())
        altered = deepcopy(array); altered["prior_turns_supervised"] = True
        with self.assertRaises(scope.ScopeError):
            self.join(source, altered, "language_replay", spec, lineage=lineage, profile=profile)
        altered = deepcopy(array); altered["labels"][0] = altered["input_ids"][0]
        with self.assertRaises(ValueError):
            self.join(source, altered, "language_replay", spec, lineage=lineage, profile=profile)

    def test_lineage_strips_only_two_fields_and_detects_altered_original_body(self):
        source, array, spec, lineage, profile = self.language_case()
        for mutation in ("changed_content", "wrong_cluster", "wrong_class", "wrong_hash", "removed_metadata", "wrong_added_version", "wrong_replay_for"):
            current, record, link = deepcopy(source), deepcopy(array), deepcopy(lineage)
            if mutation == "changed_content":
                current["messages"][-1]["content"] += " changed"; record["source_row_sha256"] = scope.digest(current)
            elif mutation == "wrong_cluster": link["cluster"] = "other"
            elif mutation == "wrong_class": link["reply_class"] = "other"
            elif mutation == "wrong_hash": link["original_v2_canonical_sha256"] = "0" * 64
            elif mutation == "wrong_added_version":
                current["metadata"]["preservation_source_version"] = "v1.0"; record["source_row_sha256"] = scope.digest(current)
            elif mutation == "wrong_replay_for":
                current["metadata"]["preservation_replay_for"] = "other"; record["source_row_sha256"] = scope.digest(current)
            else:
                del current["metadata"]["preservation_source_version"]; record["source_row_sha256"] = scope.digest(current)
            with self.subTest(mutation=mutation), self.assertRaises(scope.ScopeError):
                self.join(current, record, "language_replay", spec, lineage=link, profile=profile)

    def test_language_measured_profile_token_boundaries_and_ids_are_bound(self):
        source, array, spec, lineage, profile = self.language_case()
        for field, value in (("prompt_tokens", 99), ("completion_tokens", 3), ("prior_assistant_turns", 0),
                             ("prompt_token_ids_sha256", "0" * 64), ("completion_token_ids_sha256", "0" * 64)):
            changed = deepcopy(profile); changed[field] = value
            with self.subTest(field=field), self.assertRaises(scope.ScopeError):
                self.join(source, array, "language_replay", spec, lineage=lineage, profile=changed)

    def test_receipt_graph_links_source_and_array_file_pins(self):
        metadata = {relative: scope.read_metadata(ROOT / relative) for relative in self.manifest["files"] if relative.endswith(".json")}
        scope.receipt_links(self.manifest, metadata)
        spec = self.manifest["components"]["acquisition"]
        changed = deepcopy(metadata); changed[spec["token_receipt"]]["source_open_files_verified"]["TRAIN.jsonl"]["sha256"] = "0" * 64
        with self.assertRaises(scope.ScopeError):
            scope.receipt_links(self.manifest, changed)

    def test_resource_bounds_stop_and_no_output_grant_is_created(self):
        with patch.object(scope, "rss_bytes", return_value=scope.LIMITS["RSS_bytes"] + 1):
            report = scope.validate_train_scope(PACKAGE)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertEqual(report["reasons"][0]["code"], "CPU_RSS_LIMIT")
        self.assertFalse(report["native_qualified"])
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["e1_authorized"])
        with patch.object(scope, "rss_bytes", return_value=1), patch.object(scope.time, "monotonic", side_effect=[0, 601]):
            with self.assertRaises(scope.ScopeError): scope.Budget().check()

    def test_production_producer_and_manifest_must_be_tracked_and_clean(self):
        for mutation in ("untracked", "dirty_script", "dirty_manifest"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); script = root / (scope.PREFIX + "scripts/train_scope.py")
                manifest = root / (scope.PREFIX + scope.MANIFEST_RELATIVE)
                script.parent.mkdir(parents=True); manifest.parent.mkdir(parents=True)
                script.write_bytes(b"pass\n"); manifest.write_bytes(b"{}\n")
                def history(command, **kwargs):
                    relative = command[-1].split(":", 1)[1]
                    if mutation == "untracked": return subprocess.CompletedProcess(command, 1, stdout=b"", stderr=b"missing")
                    raw = (root / relative).read_bytes()
                    if (mutation == "dirty_script" and relative.endswith("train_scope.py")) or (mutation == "dirty_manifest" and relative.endswith(".json")):
                        raw += b"committed difference"
                    return subprocess.CompletedProcess(command, 0, stdout=raw, stderr=b"")
                with patch.object(scope, "__file__", str(script)), patch.object(scope.subprocess, "check_output", return_value="a" * 40), \
                     patch.object(scope.subprocess, "run", side_effect=history), self.assertRaises(scope.ScopeError) as caught:
                    scope.production_snapshot(root, manifest)
                self.assertEqual(caught.exception.code, "PRODUCER_SOURCE_UNTRACKED" if mutation == "untracked" else "PRODUCER_SOURCE_DIRTY")

    def test_production_head_self_and_manifest_rechecked_before_publication(self):
        for mutation in ("head", "script", "manifest"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); script = root / (scope.PREFIX + "scripts/train_scope.py")
                manifest = root / (scope.PREFIX + scope.MANIFEST_RELATIVE)
                script.parent.mkdir(parents=True); manifest.parent.mkdir(parents=True)
                script.write_bytes(b"pass\n"); manifest.write_bytes(b"{}\n")
                def history(command, **kwargs):
                    return subprocess.CompletedProcess(command, 0, stdout=(root / command[-1].split(":", 1)[1]).read_bytes(), stderr=b"")
                with patch.object(scope, "__file__", str(script)), patch.object(scope.subprocess, "check_output", return_value="a" * 40), \
                     patch.object(scope.subprocess, "run", side_effect=history):
                    snapshot = scope.production_snapshot(root, manifest)
                if mutation == "script": script.write_bytes(b"changed\n")
                if mutation == "manifest": manifest.write_bytes(b"changed\n")
                with patch.object(scope.subprocess, "check_output", return_value=("b" if mutation == "head" else "a") * 40), \
                     self.assertRaises(scope.ScopeError) as caught:
                    scope.recheck_production(root, snapshot)
                self.assertEqual(caught.exception.code, "PRODUCER_HEAD_CHANGED" if mutation == "head" else "PRODUCER_SOURCE_CHANGED")

    def test_import_and_helper_calls_preserve_environment_and_never_load_frameworks(self):
        code = "import os,sys; before=dict(os.environ); import experiments.post_training_mimo_centralized_v1.scripts.train_scope; assert dict(os.environ)==before; assert not any(n == 'torch' or n.startswith('torch.') or n == 'transformers' or n.startswith('transformers.') or '.tokenize_' in n for n in sys.modules)"
        run = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertFalse(any(name == "torch" or name.startswith("torch.") or ".tokenize_" in name for name in sys.modules))


if __name__ == "__main__":
    unittest.main()

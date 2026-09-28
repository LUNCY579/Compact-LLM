"""Synthetic export integration; actual validation/byte hashes and mocked read-only Git."""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.post_training_mimo_centralized_v1.scripts import prepare_language_eval as publisher
from experiments.post_training_mimo_centralized_v1.tests.test_language_eval_inputs import spec_documents, payload_documents


def encoded(value):
    return publisher.language.canonical(value) + b"\n"


def synthetic_inputs():
    language = publisher.language
    specs_docs = spec_documents()
    specs = language.validate_specs(specs_docs)
    payload_docs = payload_documents(specs)
    rows = language.validate_payloads(specs, payload_docs)
    raw = {doc["source_path"]: doc["raw_bytes"] for doc in specs_docs + payload_docs}
    raw[publisher.AMENDMENT] = b"Synthetic Amendment 006 source fixture; no native authority.\n"
    raw[publisher.PROPOSAL] = encoded({"synthetic_original_proposal": True})
    registration = {
        "schema_version": 1, "status": "ACCEPTED_DATA_ONLY_AUTHORING_NOT_GRADING_OR_NATIVE_EXECUTION",
        "amendment": publisher.AMENDMENT, "proposal_sha256": publisher._identity(raw[publisher.PROPOSAL])["sha256"],
        "spec_freeze_required_before_rendering": True, "DEV_parent_count": 64, "CAL_parent_count": 16,
        "CAL_unique_candidate_controls": 96, "CAL_repeats_per_template": 16,
        "packet_templates": [{"template_id": "TEMPLATE_A", "seed": 20260928}, {"template_id": "TEMPLATE_B", "seed": 20260929}],
        "prompt_cap_including_template": 896, "reference_control_cap_including_EOS": 128,
        "truncation": False, "system_message": None, "qualified_reference_score_vectors": None,
        "qualified_graders": [], "human_approval": False, "native_launch_authorized": False, "training_eligibility": False,
        "CHECK": {"reserved_parent_slots": 128, "payloads": 0, "status": "CLOSED_COUNT_ONLY"},
        "FINAL": {"reserved_parent_slots": None, "payloads": 0, "status": "CLOSED_UNSPECIFIED"},
        "slots": [{**{key: spec[key] for key in ("parent_id", "role", "stratum", "conversation_form", "author_id")},
                   "spec_source": spec["spec_source_path"],
                   "payload_source": spec["spec_source_path"].replace("/SPECS_", "/PAYLOADS_")} for spec in specs],
    }
    raw[publisher.REGISTRATION] = encoded(registration)
    spec_review = {
        "status": "ASSISTANT_SPEC_REVIEW_NOT_QUALIFIED_GRADING", "reviewer_id": "/root",
        "reviewed_parents": [spec["parent_id"] for spec in specs],
        "source_files": {Path(label).name: publisher._identity(raw[label]) for label in publisher.SPEC_FILES},
        "remaining_blocking_spec_findings": [], "native_calls": 0, "qualified_graders": 0, "reference_score_vectors": None,
    }
    raw[publisher.SPEC_REVIEW] = encoded(spec_review)
    freeze = {
        "status": "FROZEN_LANGUAGE_WORLD_SPECS_BEFORE_RENDERING", "payloads_authored_at_freeze": 0,
        "native_calls": 0, "qualified_labels": False, "training_eligibility": False, "CHECK_FINAL_payloads_opened": False,
        "registration": publisher._identity(raw[publisher.REGISTRATION]),
        "assistant_review": publisher._identity(raw[publisher.SPEC_REVIEW]),
        "source_files": {label: publisher._identity(raw[label]) for label in publisher.SPEC_FILES},
        "worlds": [{field: spec[field] for field in ("parent_id", "role", "stratum", "conversation_form", "author_id",
                                                    "world_spec_sha256", "spec_file_sha256")} for spec in specs],
    }
    raw[publisher.FREEZE] = encoded(freeze)
    for reviewer_part, payload_part, label in (("A", "B", publisher.REVIEW_FILES[0]), ("B", "A", publisher.REVIEW_FILES[1])):
        payload_label = language.SOURCE_DIRECTORY + "PAYLOADS_" + payload_part + ".json"
        raw[label] = encoded({
            "status": "ASSISTANT_SOURCE_REVIEW_NOT_QUALIFIED_GRADING", "reviewer_id": language.AUTHORS[reviewer_part],
            "reviewed_payload_file": payload_label, "reviewed_payload_file_sha256": publisher._identity(raw[payload_label])["sha256"],
            "reviewed_parent_ids": [row["parent_id"] for row in rows if row["payload_source_path"] == payload_label],
            "blocking_findings": [], "native_grading_qualification": False, "human_approval": False,
            "descriptive_review_note": "Synthetic source-review fixture, not qualified grading.",
        })
    return raw, rows


class LanguagePublisherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs, cls.rows = synthetic_inputs()

    @contextmanager
    def sandbox(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory) / "repo"
            package = root / "experiments/synthetic_language"
            scripts = package / "scripts"
            scripts.mkdir(parents=True)
            for module, name in ((publisher, "prepare_language_eval.py"), (publisher.language, "language_eval_inputs.py")):
                path = scripts / name
                path.write_bytes(("# Synthetic source identity: " + name + "\n").encode())
                stack.enter_context(patch.object(module, "__file__", str(path)))
            for label, raw in self.inputs.items():
                path = package / label
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            state = {"head": "c" * 40, "dirty": {}}

            def git_read(command, **kwargs):
                self.assertEqual(command[:3], ["git", "-C", str(root)])
                arguments = command[3:]
                if arguments == ["rev-parse", "HEAD"]:
                    return state["head"] + "\n"
                if arguments == ["status", "--porcelain"]:
                    return "?? unrelated-source-draft.txt\n"
                if arguments[:3] == ["status", "--porcelain", "--"]:
                    return state["dirty"].get(arguments[3], "")
                raise AssertionError("Unexpected Git operation: " + repr(command))

            def git_run(command, **kwargs):
                self.assertEqual(command[:6], ["git", "-C", str(root), "ls-files", "--error-unmatch", "--"])
                return subprocess.CompletedProcess(command, 0, b"", b"")

            read_mock = stack.enter_context(patch.object(publisher.subprocess, "check_output", side_effect=git_read))
            run_mock = stack.enter_context(patch.object(publisher.subprocess, "run", side_effect=git_run))
            yield {"root": root, "package": package, "output": root / "artifacts/language_export", "state": state,
                   "git_read": read_mock, "git_run": run_mock}

    def alter_json(self, fixture, label, edit):
        path = fixture["package"] / label
        value = json.loads(path.read_bytes())
        edit(value)
        path.write_bytes(encoded(value))

    def test_export_exact_rows_separate_packet_views_and_bound_receipt(self):
        with self.sandbox() as fixture:
            receipt = publisher.prepare(fixture["output"])
            files = {path.relative_to(fixture["output"]).as_posix(): path for path in fixture["output"].rglob("*") if path.is_file()}
            expected = {"DEV.jsonl", "CAL.jsonl", "DATA_RECEIPT.json", "packets/TEMPLATE_A.rating_view.json",
                        "packets/TEMPLATE_B.rating_view.json", "coordinator/TEMPLATE_A.mapping.json", "coordinator/TEMPLATE_B.mapping.json"}
            self.assertEqual(set(files), expected)
            self.assertEqual(set(receipt["files"]), expected - {"DATA_RECEIPT.json"})
            dev = [json.loads(line) for line in files["DEV.jsonl"].read_bytes().splitlines()]
            cal = [json.loads(line) for line in files["CAL.jsonl"].read_bytes().splitlines()]
            self.assertEqual((len(dev), len(cal)), (64, 16))
            self.assertEqual(sum(len(row["cal_controls"]) for row in cal), 96)
            self.assertEqual(receipt["status"], "AUTHORED_LANGUAGE_CANDIDATES_UNQUALIFIED")
            self.assertEqual(receipt["rows_sha256"], publisher.language.digest(self.rows))
            self.assertEqual(len(receipt["source_files"]), 13)
            self.assertEqual(set(receipt["input_files"]), set(publisher.INPUT_FILES))
            self.assertEqual(receipt["code_commit"], "c" * 40)
            self.assertEqual(receipt["overall_dirty_status"], "?? unrelated-source-draft.txt\n")
            self.assertEqual(receipt["counts"]["actual_rating_records"], 0)
            for name, identity in receipt["files"].items():
                self.assertEqual(identity, publisher._identity(files[name].read_bytes()))
            for name in ("native_launch_authorized", "training_eligibility", "grading_qualified", "reference_labels_qualified",
                         "human_approval", "source_correct_membership", "tokenizer_fit_measured"):
                self.assertIs(receipt[name], False)
            for name in ("model_calls", "native_calls", "native_model_calls", "programmatic_model_calls", "training_updates", "teacher_calls"):
                self.assertEqual(receipt[name], 0)
            self.assertEqual(receipt["assistant_authoring_review_cost"],
                "UNMETERED_BY_THIS_SCRIPT; assistant authored source/control texts, not native Titan responses")
            self.assertEqual(json.loads(files["DATA_RECEIPT.json"].read_bytes()), receipt)
            for template in ("TEMPLATE_A", "TEMPLATE_B"):
                view = json.loads(files[f"packets/{template}.rating_view.json"].read_bytes())
                mapping = json.loads(files[f"coordinator/{template}.mapping.json"].read_bytes())
                self.assertEqual(len(view), 112)
                self.assertTrue(all(set(item) == {"id", "messages", "response", "instructions"} for item in view))
                self.assertEqual(sum(entry["is_repeat"] for entry in mapping["entries"].values()), 16)
                publisher.language._packet({"rating_view": view, "coordinator_mapping": mapping})

    def test_repeat_export_is_deterministic_but_existing_output_is_never_overwritten(self):
        with self.sandbox() as fixture:
            first = publisher.prepare(fixture["output"])
            second_path = fixture["output"].with_name("language_export_second")
            second = publisher.prepare(second_path)
            self.assertEqual(first["files"], second["files"])
            before = (fixture["output"] / "DATA_RECEIPT.json").read_bytes()
            fixture["git_read"].reset_mock()
            with self.assertRaisesRegex(ValueError, "write-once"):
                publisher.prepare(fixture["output"])
            self.assertEqual((fixture["output"] / "DATA_RECEIPT.json").read_bytes(), before)
            fixture["git_read"].assert_not_called()

    def test_registration_slots_and_native_or_token_limit_drift_blocks_export(self):
        edits = [lambda doc: doc["slots"][0].update(role="CHECK"), lambda doc: doc.update(prompt_cap_including_template=897),
                 lambda doc: doc.update(native_launch_authorized=True), lambda doc: doc.update(DEV_parent_count=True)]
        for edit in edits:
            with self.sandbox() as fixture:
                self.alter_json(fixture, publisher.REGISTRATION, edit)
                with self.assertRaises(ValueError):
                    publisher.prepare(fixture["output"])
                self.assertFalse(fixture["output"].exists())

    def test_freeze_world_hash_raw_source_and_review_binding_drift_blocks_export(self):
        edits = [lambda doc: doc["worlds"][0].update(world_spec_sha256="a" * 64),
                 lambda doc: doc["source_files"][publisher.SPEC_FILES[0]].update(bytes=1),
                 lambda doc: doc["assistant_review"].update(sha256="b" * 64),
                 lambda doc: doc.update(qualified_labels=True)]
        for edit in edits:
            with self.sandbox() as fixture:
                self.alter_json(fixture, publisher.FREEZE, edit)
                with self.assertRaises(ValueError):
                    publisher.prepare(fixture["output"])
                self.assertFalse(fixture["output"].exists())

    def test_cross_review_requires_exact_author_payload_hash_parents_and_no_qualification(self):
        edits = [lambda doc: doc.update(reviewer_id=publisher.language.AUTHORS["B"]),
                 lambda doc: doc.update(reviewed_payload_file=publisher.PAYLOAD_FILES[0]),
                 lambda doc: doc.update(reviewed_payload_file_sha256="a" * 64),
                 lambda doc: doc["reviewed_parent_ids"].__setitem__(1, doc["reviewed_parent_ids"][0]),
                 lambda doc: doc["reviewed_parent_ids"].pop(), lambda doc: doc.update(blocking_findings=["unresolved"]),
                 lambda doc: doc.update(native_grading_qualification=True), lambda doc: doc.update(human_approval=0)]
        for edit in edits:
            with self.sandbox() as fixture:
                self.alter_json(fixture, publisher.REVIEW_FILES[0], edit)
                with self.assertRaises(ValueError):
                    publisher.prepare(fixture["output"])
                self.assertFalse(fixture["output"].exists())

    def test_missing_review_untracked_source_or_dirty_source_prevents_publication(self):
        for kind in ("missing", "untracked", "dirty"):
            with self.subTest(kind=kind), self.sandbox() as fixture:
                if kind == "missing":
                    (fixture["package"] / publisher.REVIEW_FILES[0]).unlink()
                    error = FileNotFoundError
                elif kind == "untracked":
                    fixture["git_run"].side_effect = subprocess.CalledProcessError(1, ["git", "ls-files"])
                    error = subprocess.CalledProcessError
                else:
                    relative = (fixture["package"] / publisher.FREEZE).relative_to(fixture["root"]).as_posix()
                    fixture["state"]["dirty"][relative] = " M " + relative + "\n"
                    error = ValueError
                with self.assertRaises(error):
                    publisher.prepare(fixture["output"])
                self.assertFalse(fixture["output"].exists())

    def test_source_or_review_edit_during_packet_build_blocks_publication(self):
        for label in (publisher.SPEC_FILES[0], publisher.REVIEW_FILES[0], "scripts/language_eval_inputs.py", publisher.REGISTRATION):
            with self.subTest(label=label), self.sandbox() as fixture:
                original = publisher.language.build_cal_packet
                mutated = {"done": False}

                def build_then_edit(*args, **kwargs):
                    result = original(*args, **kwargs)
                    if not mutated["done"]:
                        path = fixture["package"] / label
                        path.write_bytes(path.read_bytes() + b"\n")
                        mutated["done"] = True
                    return result

                with patch.object(publisher.language, "build_cal_packet", side_effect=build_then_edit):
                    with self.assertRaisesRegex(ValueError, "source changed before publication"):
                        publisher.prepare(fixture["output"])
                self.assertFalse(fixture["output"].exists())

    def test_head_or_cleanliness_change_during_work_blocks_publication(self):
        for change in ("head", "dirty"):
            with self.subTest(change=change), self.sandbox() as fixture:
                original = publisher.language.build_cal_packet

                def build_then_change(*args, **kwargs):
                    result = original(*args, **kwargs)
                    if change == "head":
                        fixture["state"]["head"] = "d" * 40
                    else:
                        relative = (fixture["package"] / publisher.FREEZE).relative_to(fixture["root"]).as_posix()
                        fixture["state"]["dirty"][relative] = "M  " + relative + "\n"
                    return result

                with patch.object(publisher.language, "build_cal_packet", side_effect=build_then_change):
                    with self.assertRaises(ValueError):
                        publisher.prepare(fixture["output"])
                self.assertFalse(fixture["output"].exists())

    def test_partial_write_is_preserved_and_retry_cannot_overwrite_it(self):
        with self.sandbox() as fixture:
            original = publisher._write

            def fail_on_second(path, raw):
                if path.name == "CAL.jsonl":
                    raise OSError("synthetic disk failure")
                original(path, raw)

            with patch.object(publisher, "_write", side_effect=fail_on_second):
                with self.assertRaisesRegex(OSError, "synthetic disk failure"):
                    publisher.prepare(fixture["output"])
            before = (fixture["output"] / "DEV.jsonl").read_bytes()
            self.assertFalse((fixture["output"] / "DATA_RECEIPT.json").exists())
            with self.assertRaisesRegex(ValueError, "write-once"):
                publisher.prepare(fixture["output"])
            self.assertEqual((fixture["output"] / "DEV.jsonl").read_bytes(), before)

    def test_fresh_import_and_cli_help_have_no_native_or_environment_side_effect(self):
        root = Path(__file__).resolve().parents[3]
        code = ("import os,sys; before=dict(os.environ); "
                "from experiments.post_training_mimo_centralized_v1.scripts import prepare_language_eval; "
                "assert dict(os.environ)==before; "
                "assert not any(name in sys.modules for name in ('torch','transformers','tokenizers','safetensors')); "
                "assert not any('tokenize_bridge' in name for name in sys.modules)")
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, "-B", "-m",
            "experiments.post_training_mimo_centralized_v1.scripts.prepare_language_eval", "--help"],
            cwd=root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output", result.stdout)


if __name__ == "__main__":
    unittest.main()

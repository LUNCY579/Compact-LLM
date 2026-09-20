"""Session preparation refusal and provenance tests; entirely synthetic, no models."""
from __future__ import annotations

import copy
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack

from . import prepare_session as p
from . import receipts as r


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.canonical_bytes(value) + b"\n")


def token_fixture():
    originals, encoded = [], []
    for index in range(32):
        messages = [{"role": "user", "content": "open prompt " + str(index)}]
        original = {"id": "row" + str(index), "parent_id": "parent" + str(index),
                    "split": "fit" if index < 24 else "map_check", "open": True,
                    "prompt_sha256": r.digest(messages)}
        originals.append(original)
        rendered = "rendered open prompt " + str(index)
        ids, mask = [10, index, 20], [1, 1, 1]
        encoded.append({**original, "prompt": messages, "rendered_prompt": rendered,
            "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "input_ids": ids, "input_ids_sha256": r.digest(ids), "attention_mask": mask,
            "attention_mask_sha256": r.digest(mask), "position_ids": [0, 1, 2], "tokens": 3})
    calibration = {"role": "open_calibration", "dataset_sha256": "a" * 64,
                   "rows": originals, "calibration_sha256": r.digest(originals)}
    student = {"status": "TOKEN_ELIGIBILITY_PASSED", "model_loads": 0, "model_forwards": 0,
        "optimizer_steps": 0, "cuda_initialized": False, "calibration_file_sha256": "c" * 64,
        "calibration_sha256": calibration["calibration_sha256"], "dataset_sha256": "a" * 64,
        "rendering": dict(p.RENDERING), "registered_cap": 256, "overflow_ids": [],
        "rows": encoded, "encoded_rows_sha256": r.digest(encoded), "row_count": 32,
        "runtime": {"packages": dict(p.PINS["student_base"])}, "tokenizer_assets": {},
        "template_sha256": "d" * 64}
    donor = copy.deepcopy(student)
    donor["runtime"]["packages"] = dict(p.PINS["donor"])
    return student, donor, calibration


class TokenBindingTests(unittest.TestCase):
    def test_exact_rows_convert_to_worker_schema(self):
        student, donor, calibration = token_fixture()
        rows = p.validate_token_pair(student, donor, calibration, "c" * 64)
        self.assertEqual(len(rows), 32)
        self.assertEqual(rows[0]["messages_sha256"], calibration["rows"][0]["prompt_sha256"])
        self.assertEqual(rows[0]["prompt_sha256"], r.digest(student["rows"][0]["rendered_prompt"]))
        self.assertEqual(sum(row["split"] == "fit" for row in rows), 24)

    def test_internal_self_hash_cannot_hide_original_membership_change(self):
        student, donor, calibration = token_fixture()
        for receipt in (student, donor):
            receipt["rows"][0]["parent_id"] = "wrong-parent"
            receipt["encoded_rows_sha256"] = r.digest(receipt["rows"])
        with self.assertRaises(r.Blocked):
            p.validate_token_pair(student, donor, calibration, "c" * 64)

    def test_cross_runtime_changed_tokens_rejected(self):
        student, donor, calibration = token_fixture()
        donor["rows"][0]["input_ids"][0] = 99
        donor["encoded_rows_sha256"] = r.digest(donor["rows"])
        with self.assertRaises(r.Blocked):
            p.validate_token_pair(student, donor, calibration, "c" * 64)

    def test_padding_positions_overflow_and_rendered_drift_rejected(self):
        mutations = [lambda row: row.update(attention_mask=[0, 1, 1]),
                     lambda row: row.update(position_ids=[1, 2, 3]),
                     lambda row: row.update(input_ids=[1] * 257),
                     lambda row: row.update(rendered_prompt="changed")]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                student, donor, calibration = token_fixture()
                for receipt in (student, donor):
                    mutation(receipt["rows"][0])
                    receipt["encoded_rows_sha256"] = r.digest(receipt["rows"])
                with self.assertRaises(r.Blocked):
                    p.validate_token_pair(student, donor, calibration, "c" * 64)

    def test_runtime_rendering_calibration_and_framework_state_rejected(self):
        mutations = [lambda value: value["runtime"]["packages"].update(torch="changed"),
                     lambda value: value["rendering"].update(truncation=True),
                     lambda value: value.update(calibration_file_sha256="0" * 64),
                     lambda value: value.update(cuda_initialized=True),
                     lambda value: value.update(model_forwards=1)]
        for mutation in mutations:
            student, donor, calibration = token_fixture()
            mutation(student)
            with self.assertRaises(r.Blocked):
                p.validate_token_pair(student, donor, calibration, "c" * 64)


class RegistrationGuardTests(unittest.TestCase):
    def test_exact_resource_boundaries_and_occupied_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            lease = Path(directory) / "rineya-gpu-0.lock"
            memory = {"available_host_bytes": 12 * 2**30, "free_gpu_bytes": 6 * 2**30}
            p.resource_gate(memory, lease)
            for field in memory:
                low = dict(memory)
                low[field] -= 1
                with self.assertRaises(r.Blocked):
                    p.resource_gate(low, lease)
            lease.write_text("owned")
            with self.assertRaises(r.Blocked):
                p.resource_gate(memory, lease)
            self.assertEqual(lease.read_text(), "owned")

    def test_low_ram_refuses_before_source_checks_and_creates_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            output = base / "session002"
            with patch.object(p, "resource_snapshot", return_value={"available_host_bytes": int(4.748 * 2**30),
                    "free_gpu_bytes": 7 * 2**30}), patch.object(p, "source_inventory") as inventory:
                with self.assertRaises(r.Blocked):
                    p.prepare(root=source, source_commit="a" * 40, student_path=base / "s.json",
                        student_sha256="a" * 64, donor_path=base / "d.json", donor_sha256="b" * 64,
                        output=output, session_id="CSC-001-STAGE-C-002", nvidia_smi=base / "nvidia-smi.exe")
                inventory.assert_not_called()
            self.assertFalse(output.exists())

    def test_invalid_deadline_refuses_without_resource_probe(self):
        with patch.object(p, "resource_snapshot") as probe:
            for minutes in (0, 61, 1.5, True):
                with self.subTest(minutes=minutes), self.assertRaises(r.Blocked):
                    p.prepare(root=Path.cwd(), source_commit="a" * 40, student_path=Path("s"),
                        student_sha256="a" * 64, donor_path=Path("d"), donor_sha256="b" * 64,
                        output=Path(tempfile.gettempdir()) / "not-created-csc-test", session_id="CSC-001-STAGE-C-X",
                        nvidia_smi=Path("unused"), minutes=minutes)
            probe.assert_not_called()

    def test_fresh_id_existing_output_open_session_and_closure_binding(self):
        now = dt.datetime(2026, 9, 20, tzinfo=dt.timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            old = base / "old"
            old.mkdir()
            spec = {"session_id": "CSC-001-STAGE-C-001", "deadline_utc": (now + dt.timedelta(hours=1)).isoformat()}
            save(old / "SESSION_SPEC.json", spec)
            with self.assertRaises(r.Blocked):
                p.ensure_fresh(base / "new", "CSC-001-STAGE-C-002", now)
            save(old / "SESSION_CLOSED.json", {"spec_sha256": "0" * 64})
            with self.assertRaises(r.Blocked):
                p.ensure_fresh(base / "new", "CSC-001-STAGE-C-002", now)
            save(old / "SESSION_CLOSED.json", {"spec_sha256": r.file_digest(old / "SESSION_SPEC.json")})
            p.ensure_fresh(base / "new", "CSC-001-STAGE-C-002", now)
            with self.assertRaises(r.Blocked):
                p.ensure_fresh(base / "new", "CSC-001-STAGE-C-001", now)
            with self.assertRaises(r.Blocked):
                p.ensure_fresh(old, "CSC-001-STAGE-C-003", now)

    def test_dirty_token_preflight_rejected_without_git(self):
        receipt = {"source": {"commit": {"returncode": 0, "stdout": "a" * 40},
                              "worktree": {"returncode": 0, "stdout": "?? untracked"}}}
        with patch.object(p, "git") as git:
            with self.assertRaises(r.Blocked):
                p.verify_preflight_provenance(Path.cwd(), receipt, "a" * 40)
            git.assert_not_called()

    def test_ancestor_receipt_requires_same_committed_and_current_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / p.PACKAGE
            package.mkdir(parents=True)
            for name in ("native_preflight.py", "build_calibration.py"):
                (package / name).write_bytes(b"qualified helper\n")
            sha = hashlib.sha256(b"qualified helper\n").hexdigest()
            receipt = {"source": {"commit": {"returncode": 0, "stdout": "a" * 40},
                "worktree": {"returncode": 0, "stdout": ""}, "helper_sha256": sha, "selection_helper_sha256": sha}}
            with patch.object(p, "git", side_effect=lambda root, *args: b"" if args[0] == "merge-base" else b"qualified helper\n"):
                p.verify_preflight_provenance(root, receipt, "b" * 40)
                (package / "native_preflight.py").write_bytes(b"new helper\n")
                with self.assertRaises(r.Blocked):
                    p.verify_preflight_provenance(root, receipt, "b" * 40)

    def test_source_closure_includes_imported_native_dependencies_and_amendment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / p.PACKAGE
            package.mkdir(parents=True)
            (package / "native_capture.py").write_text("# fixture")
            with patch.object(r, "verify_source") as verify, patch.object(r, "implementation_digest", side_effect=lambda root, paths: paths):
                paths = p.source_inventory(root, "a" * 40)
            verify.assert_called_once_with(root, "a" * 40, clean=True)
            for name in [*p.DEPENDENCIES, p.PACKAGE + "STAGE_C_MECHANISM_AMENDMENT.md",
                         p.PACKAGE + "evidence/CALIBRATION_BINDING.v2.json"]:
                self.assertIn(name, paths)


class PreparationIntegrationTests(unittest.TestCase):
    def fixture(self, base):
        root = base / "source"
        package = root / p.PACKAGE
        (package / "evidence").mkdir(parents=True)
        namespace = base / "records"
        namespace.mkdir()
        output = namespace / "session002"
        student, donor, calibration = token_fixture()
        source = base / "open.jsonl.gz"
        with gzip.open(source, "wt", encoding="utf-8") as stream:
            for index, row in enumerate(calibration["rows"]):
                stream.write(json.dumps({"id": row["id"], "valid": True, "source_split": "train",
                                         "family": "ledger_replay" if index % 2 else "register_machine"}) + "\n")
        calibration["selection"] = {"source_path": str(source)}
        calibration["dataset_sha256"] = r.file_digest(source)
        cal_path = package / "evidence/CALIBRATION_BINDING.v2.json"
        save(cal_path, calibration)
        token_files = {name: "f" * 64 for name in
                       ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")}
        for receipt in (student, donor):
            receipt["calibration_file_sha256"] = r.file_digest(cal_path)
            receipt["dataset_sha256"] = calibration["dataset_sha256"]
            receipt["tokenizer_assets"] = token_files
        s_path, d_path = base / "student.json", base / "donor.json"
        save(s_path, student)
        save(d_path, donor)
        save(package / "evidence/ARCHITECTURE_MAP.json", {"layer_map": {
            "capture_pairs_student_donor": [[0, 0], [13, 17], [27, 35]],
            "donor_to_student": [min(range(28), key=lambda j: (abs(i * 27 - j * 35), j)) for i in range(36)]}})
        save(package / "evidence/BINDING_ASSETS.json", {})
        (package / "STAGE_C_MECHANISM_AMENDMENT.md").write_text("prospective mechanism gate\n")
        models = {"student": {"directory": str(base / "student"), "files": {
                    **token_files, "model.safetensors": "a" * 64}},
                  "donor": {"directory": str(base / "donor"), "files": token_files},
                  "adapter": {"directory": str(base / "adapter"), "files": {}}}
        args = {"root": root, "source_commit": "a" * 40, "student_path": s_path,
                "student_sha256": r.file_digest(s_path), "donor_path": d_path,
                "donor_sha256": r.file_digest(d_path), "output": output,
                "session_id": "CSC-001-STAGE-C-002", "nvidia_smi": Path("unused"), "minutes": 30}
        return args, models

    def patches(self, resources, models):
        stack = ExitStack()
        stack.enter_context(patch.object(p.tempfile, "gettempdir", return_value=str(Path(models["student"]["directory"]).parent)))
        stack.enter_context(patch.object(p, "resource_snapshot", side_effect=resources))
        stack.enter_context(patch.object(p, "source_inventory", return_value={"files": {"fixture": "f" * 64}, "sha256": "d" * 64}))
        stack.enter_context(patch.object(p, "verify_preflight_provenance"))
        stack.enter_context(patch.object(p, "rebuild_calibration"))
        stack.enter_context(patch.object(p, "assets_binding", return_value=models))
        stack.enter_context(patch.object(p, "runtime_binding", side_effect=lambda receipt, name: {
            "packages": p.PINS[name], "python_executable": "fixture-python", "device": "cuda:0"}))
        stack.enter_context(patch.object(r, "verify_source"))
        return stack

    def test_successful_fixture_registration_preserves_rows_protocol_and_deadline(self):
        high = {"available_host_bytes": 12 * 2**30, "free_gpu_bytes": 6 * 2**30}
        with tempfile.TemporaryDirectory() as directory:
            args, models = self.fixture(Path(directory))
            with self.patches([high, high], models):
                result = p.prepare(**args)
            spec = json.loads(Path(result["path"]).read_bytes())
            self.assertEqual(result["sha256"], r.file_digest(Path(result["path"])))
            self.assertEqual(spec["analysis_protocol"], p.ANALYSIS_PROTOCOL)
            self.assertEqual(spec["max_total_forwards"], 96)
            self.assertEqual(spec["max_role_forwards"], 32)
            self.assertEqual(spec["max_optimizer_steps"], 0)
            self.assertEqual(spec["max_generated_tokens"], 0)
            self.assertEqual(spec["authorization_evidence"], p.AUTHORIZATION)
            self.assertEqual(set(row["family"] for row in spec["calibration"]["rows"]), {"ledger_replay", "register_machine"})
            self.assertEqual((dt.datetime.fromisoformat(spec["deadline_utc"]) -
                              dt.datetime.fromisoformat(spec["registered_utc"])).total_seconds(), 1800)
            self.assertTrue((args["output"] / "reservations").is_dir())
            original = Path(result["path"]).read_bytes()
            with self.patches([high], models), self.assertRaises(r.Blocked):
                p.prepare(**args)
            self.assertEqual(original, Path(result["path"]).read_bytes())

    def test_resource_falls_during_preparation_no_registration_is_written(self):
        high = {"available_host_bytes": 12 * 2**30, "free_gpu_bytes": 6 * 2**30}
        low = {"available_host_bytes": 12 * 2**30 - 1, "free_gpu_bytes": 6 * 2**30}
        with tempfile.TemporaryDirectory() as directory:
            args, models = self.fixture(Path(directory))
            with self.patches([high, low], models), self.assertRaises(r.Blocked):
                p.prepare(**args)
            self.assertFalse(args["output"].exists())


if __name__ == "__main__":
    unittest.main()

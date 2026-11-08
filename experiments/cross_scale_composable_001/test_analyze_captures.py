"""SYNTHETIC TEST ONLY: fabricated custody graphs/arrays; zero native execution."""
from contextlib import contextmanager, ExitStack
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import analyze_captures as a
from . import mechanics as m
from . import receipts as r
from .test_native_capture import synthetic_spec


def save_npy(path, value):
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
    return r.file_digest(path)


def synthetic_custody_graph(root):
    """Fabricate only temporary test evidence. Never a native evidence producer."""
    spec = synthetic_spec(root)
    spec.pop("frozen_registration_sha256")
    spec.pop("lease")
    spec.update(max_optimizer_steps=0, max_generated_tokens=0, attempt_root=str(root / "attempts"),
                lease_path=str(root / "rineya-gpu-0.lock"))
    repository = root / "repo"
    repository.mkdir()
    package = repository / a.PACKAGE
    (package / "evidence").mkdir(parents=True)
    (package / "evidence/CALIBRATION_BINDING.v2.json").write_bytes((root / "CALIBRATION.json").read_bytes())
    original = json.loads((package / "evidence/CALIBRATION_BINDING.v2.json").read_text())
    original["role"] = "open_calibration"
    (package / "evidence/CALIBRATION_BINDING.v2.json").write_bytes(r.canonical_bytes(original))
    spec["calibration"]["source_binding_file_sha256"] = r.file_digest(package / "evidence/CALIBRATION_BINDING.v2.json")
    (package / "evidence/ARCHITECTURE_MAP.json").write_bytes((root / "ARCHITECTURE.json").read_bytes())
    source_names = ("native_capture.py", "native_session.py", "receipts.py", "STAGE_C_MECHANISM_AMENDMENT.md")
    for name in source_names:
        (package / name).write_text("# SYNTHETIC TEST ONLY source stub\n", encoding="utf-8")
    spec["source_files"] = {a.PACKAGE + name: r.file_digest(package / name) for name in source_names}
    spec["implementation_digest"] = r.digest(spec["source_files"])
    amendment_path = a.PACKAGE + "STAGE_C_MECHANISM_AMENDMENT.md"
    spec["stage_c_mechanism_amendment"] = {"path": amendment_path, "sha256": spec["source_files"][amendment_path]}
    spec["analysis_protocol"] = copy.deepcopy(a.ANALYSIS_PROTOCOL)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repository), *args], stderr=subprocess.DEVNULL).decode().strip()
    git("init", "-q")
    git("config", "user.name", "Synthetic test")
    git("config", "user.email", "synthetic@example.invalid")
    git("config", "core.autocrlf", "false")
    git("add", ".")
    git("commit", "-qm", "synthetic capture source")
    spec["source_commit"] = git("rev-parse", "HEAD")
    # Later HEAD deliberately differs: historical capture source must still work.
    (repository / "later.txt").write_text("synthetic later analysis revision\n")
    git("add", ".")
    git("commit", "-qm", "synthetic later revision")
    spec_path = root / "SPEC.json"
    spec_sha = r.write_once(spec_path, spec)
    journal = Path(spec["reservation_dir"])
    binding = {"session_id": spec["session_id"], "frozen_registration_sha256": spec_sha}
    r.write_once(journal / "SESSION_BINDING.json", binding)
    registry = {"schema": 1, "evidence_kind": "NATIVE_STAGE_C_CAPTURE", "frozen_registration_sha256": spec_sha, "roles": {}}
    latent = np.random.default_rng(55).normal(size=(32, 3))
    for role_index, role in enumerate(a.ROLES):
        attempt = Path(spec["attempt_root"]) / role
        supervision, capture_dir = attempt / "supervision", attempt / "supervision/capture"
        capture_dir.mkdir(parents=True)
        worker = {**spec, "lease": {"path": spec["lease_path"], "token": "synthetic-role-" + role, "owner_pid": 1234},
                  "frozen_registration_sha256": spec_sha}
        worker_path = attempt / "WORKER_SPEC.json"
        worker_sha = r.write_once(worker_path, worker)
        pid = 2000 + role_index
        active = {**binding, "worker_spec_sha256": worker_sha, "role": role, "pid": pid}
        role_journal = journal / role
        role_journal.mkdir()
        r.write_once(role_journal / "CLAIM.json", active)
        runtime = spec["runtime"]["student_base" if role == "student_base" else "donor"]
        argv = [runtime["python_executable"], "-B", "-m", "experiments.cross_scale_composable_001.native_capture",
                "--spec", str(worker_path), "--spec-sha256", worker_sha, "--role", role,
                "--out", str(capture_dir), "--supervision", str(supervision)]
        launch = {"schema": 1, "argv": argv, "timeout_seconds": 300, "cwd": str(repository),
                  "out": str(supervision), "supervisor_pid": 1234, "context_sha256": r.digest("test-context")}
        r.write_once(supervision / "launch.json", launch)
        r.write_once(supervision / "spawn.json", {"child_pid": pid, "supervisor_pid": 1234, "launch_sha256": r.digest(launch)})
        r.write_once(supervision / "worker_claim.json", {"pid": pid, "parent_pid": 1234, "supervisor_pid": 1234,
                                                        "launcher_redirected": False, "launch_sha256": r.digest(launch)})
        r.write_once(supervision / "supervisor.json", {"status": "EXITED", "returncode": 0, "child_pid": pid, "cleanup": None})
        result = {"schema": 1, "status": "COMPLETE", "completed": True, "role": role, "session_id": spec["session_id"],
                  "source_commit": spec["source_commit"], "spec_sha256": worker_sha, "native_model_forwards": 32,
                  "native_tokenizer_encodings": 32, "generation_calls": 0, "optimizer_steps": 0, "rows": [], "captures": {}}
        matrices = {}
        for layer in a.LAYERS[role]:
            matrix = np.zeros((32, a.WIDTHS[role]), dtype="float32")
            if role == "student_base":
                matrix[:, :3] = latent * np.array([2., 3., -1.]) * (1 + layer / 100)
                matrix += 1.
            else:
                matrix[:, :3] = latent * (1 + layer / 100)
                if role == "donor_skilled":
                    matrix[:, 0] += .1
            matrices[layer] = matrix
            filename = f"layer-{layer:02d}.npy"
            file_hash = save_npy(capture_dir / filename, matrix)
            metadata = a.expected_metadata(spec, role, layer, m.tensor_digest({"residual": matrix}))
            result["captures"][str(layer)] = {"file": filename, "file_sha256": file_hash, "metadata": metadata}
        for ordinal, registered in enumerate(spec["calibration"]["rows"], 1):
            artifacts = {f"row-{ordinal-1:03d}-layer-{layer:02d}.npy": save_npy(
                capture_dir / f"row-{ordinal-1:03d}-layer-{layer:02d}.npy", matrices[layer][ordinal-1]) for layer in a.LAYERS[role]}
            row = {key: registered[key] for key in ("id", "parent_id", "split", "prompt_sha256", "input_ids_sha256")}
            row.update(position=len(registered["input_ids"]) - 1, token_id=registered["input_ids"][-1],
                       forward_ordinal=ordinal, artifacts=artifacts)
            result["rows"].append(row)
            r.write_once(capture_dir / f"row-{ordinal-1:03d}.json", row)
            r.write_once(role_journal / f"{ordinal:03d}-RESERVED.json", {**active, "ordinal": ordinal, "row_id": registered["id"],
                         "input_ids_sha256": registered["input_ids_sha256"], "forwards_charged": 1, "reserved_unix": ordinal})
            r.write_once(role_journal / f"{ordinal:03d}-COMPLETED.json", row)
        r.write_once(role_journal / "TERMINAL.json", {**active, "status": "COMPLETE", "reserved_forwards": 32,
                                                      "error": None, "retry_authorized": False})
        r.write_once(attempt / "SESSION_TERMINAL.json", {"status": "COMPLETE", "completed": True, "role": role,
                                                        "child_returncode": 0, "native_forward_count": 32, "optimizer_steps": 0})
        result_path = capture_dir / "RESULT.json"
        result_sha = r.write_once(result_path, result)
        registry["roles"][role] = {"result_path": str(result_path), "result_sha256": result_sha,
                                   "receipt_files": {str(path): r.file_digest(path) for path in a.required_receipt_paths(spec, role)}}
    custody_path = root / "CUSTODY.json"
    r.write_once(custody_path, registry)
    (root / "SYNTHETIC_TEST_ONLY.txt").write_text("All capture receipts and arrays in this temporary directory are fabricated test fixtures. Zero native calls.\n")
    return spec, spec_path, custody_path, repository


class SyntheticReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="csc-synthetic-analysis-tests-")
        cls.root = Path(cls.temp.name)
        cls.spec, cls.spec_path, cls.custody_path, cls.repository = synthetic_custody_graph(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch.object(a, "ROOT", self.repository))
        self.patches.enter_context(patch.object(a, "analysis_identity", return_value={"synthetic_test_only": True}))

    def run_analysis(self, name):
        return a.run(self.spec_path, r.file_digest(self.spec_path), self.custody_path,
                     r.file_digest(self.custody_path), self.root / name)

    @contextmanager
    def rewrite(self, path, body, *, refresh_custody=False):
        saved, custody_saved = path.read_bytes(), self.custody_path.read_bytes()
        try:
            path.write_bytes(r.canonical_bytes(body))
            if refresh_custody:
                registry = json.loads(custody_saved)
                for entry in registry["roles"].values():
                    if str(path) == entry["result_path"]:
                        entry["result_sha256"] = r.file_digest(path)
                    if str(path) in entry["receipt_files"]:
                        entry["receipt_files"][str(path)] = r.file_digest(path)
                self.custody_path.write_bytes(r.canonical_bytes(registry))
            yield
        finally:
            path.write_bytes(saved)
            self.custody_path.write_bytes(custody_saved)

    def test_three_pairs_fit_from_verified_synthetic_custody(self):
        result = self.run_analysis("analysis-success")
        self.assertTrue(result["final_pair_continuation_pass"])
        self.assertFalse(result["scientific_improvement_claim"])
        self.assertEqual(result["native_model_calls_this_analysis"], 0)
        self.assertEqual(len(result["pairs"]), 3)
        for report in result["pairs"].values():
            self.assertEqual((report["fit_rows"], report["check_rows"]), (24, 8))
            self.assertLess(report["heldout"]["mse"], 1e-10)
            self.assertTrue(report["shuffle_baseline_pass"])
            self.assertGreater(report["residual_support"]["partitions"]["heldout"]["actual"]["identified_support_energy_fraction"], .999)
            self.assertFalse(report["residual_support"]["family_breakdown"]["available"])
            for artifact in report["artifacts"].values():
                self.assertEqual(r.file_digest(self.root / "analysis-success" / artifact["file"]), artifact["sha256"])
        with self.assertRaises(r.Blocked):
            self.run_analysis("analysis-success")

    def test_historical_commit_does_not_need_to_be_head(self):
        head = a.git_bytes("rev-parse", "HEAD").decode().strip()
        self.assertNotEqual(head, self.spec["source_commit"])
        a.verify_historical_source(self.spec)
        wrong = copy.deepcopy(self.spec)
        wrong["source_commit"] = "f" * 40
        with self.assertRaises(r.Blocked):
            a.verify_historical_source(wrong)

    def test_metadata_and_row_budget_negatives_even_with_new_result_custody(self):
        path = Path(self.spec["attempt_root"]) / "donor_skilled/supervision/capture/RESULT.json"
        for mode in ("weight", "calibration", "runtime", "session", "rowcount", "fixture"):
            result = a.read_json(path)
            meta = result["captures"]["35"]["metadata"]
            if mode == "weight": meta["model_sha256"] = "a" * 64
            if mode == "calibration": meta["calibration_sha256"] = "b" * 64
            if mode == "runtime": meta["runtime_binding_sha256"] = "c" * 64
            if mode == "session": result["session_id"] = "other-session"
            if mode == "rowcount": result["rows"].pop()
            if mode == "fixture": result["evidence_kind"] = "SYNTHETIC_CPU_FIXTURE_ONLY"
            with self.subTest(mode=mode), self.rewrite(path, result, refresh_custody=True):
                with self.assertRaises(r.Blocked):
                    self.run_analysis("blocked-" + mode)

    def test_failed_supervisor_or_controller_refused(self):
        attempt = Path(self.spec["attempt_root"]) / "student_base"
        for suffix in ("supervision/supervisor.json", "SESSION_TERMINAL.json"):
            path = attempt / suffix
            body = a.read_json(path)
            body["status"] = "FAILED"
            with self.subTest(suffix=suffix), self.rewrite(path, body, refresh_custody=True):
                with self.assertRaises(r.Blocked):
                    self.run_analysis("bad-child")

    def test_tampered_result_and_array_bytes_refused(self):
        path = Path(self.spec["attempt_root"]) / "student_base/supervision/capture/RESULT.json"
        body = a.read_json(path)
        body["status"] = "CHANGED"
        with self.rewrite(path, body):
            with self.assertRaisesRegex(r.Blocked, "bytes differ"):
                self.run_analysis("tampered-result")
        array = path.parent / "layer-00.npy"
        saved = array.read_bytes()
        try:
            array.write_bytes(saved + b"tampered")
            with self.assertRaisesRegex(r.Blocked, "bytes differ"):
                self.run_analysis("tampered-array")
        finally:
            array.write_bytes(saved)

    def test_missing_or_wrong_reserved_call_refused(self):
        path = Path(self.spec["reservation_dir"]) / "student_base/032-RESERVED.json"
        body = a.read_json(path)
        body["row_id"] = "wrong-row"
        with self.rewrite(path, body, refresh_custody=True):
            with self.assertRaisesRegex(r.Blocked, "reservation identity"):
                self.run_analysis("wrong-reservation")
        saved = path.read_bytes()
        try:
            path.unlink()
            with self.assertRaisesRegex(r.Blocked, "missing artifact"):
                self.run_analysis("missing-reservation")
        finally:
            path.write_bytes(saved)

    def test_registry_requires_exact_three_roles_and_full_receipt_custody(self):
        registry = a.read_json(self.custody_path)
        registry["roles"].pop("donor_skilled")
        with self.rewrite(self.custody_path, registry):
            with self.assertRaises(r.Blocked):
                self.run_analysis("missing-role")
        registry = a.read_json(self.custody_path)
        registry["roles"]["student_base"]["receipt_files"].pop(next(iter(registry["roles"]["student_base"]["receipt_files"])))
        with self.rewrite(self.custody_path, registry):
            with self.assertRaisesRegex(r.Blocked, "custody receipt inventory"):
                self.run_analysis("missing-custody")

    def test_prospective_amendment_is_required(self):
        spec = copy.deepcopy(self.spec)
        spec.pop("stage_c_mechanism_amendment")
        with self.assertRaisesRegex(r.Blocked, "amendment"):
            a.verify_historical_source(spec)
        spec = copy.deepcopy(self.spec)
        spec["analysis_protocol"]["require_beats_shuffle"] = False
        with self.assertRaisesRegex(r.Blocked, "protocol"):
            a.verify_historical_source(spec)

    def test_bound_array_wrong_dtype_or_shape_refused(self):
        result_path = Path(self.spec["attempt_root"]) / "student_base/supervision/capture/RESULT.json"
        array_path = result_path.parent / "layer-00.npy"
        saved = array_path.read_bytes()
        original = np.load(array_path, allow_pickle=False)
        try:
            for mode, array in (("dtype", original.astype("float64")), ("shape", original[:31])):
                with array_path.open("wb") as stream:
                    np.save(stream, array, allow_pickle=False)
                result = a.read_json(result_path)
                result["captures"]["0"]["file_sha256"] = r.file_digest(array_path)
                with self.subTest(mode=mode), self.rewrite(result_path, result, refresh_custody=True):
                    with self.assertRaises(r.Blocked):
                        self.run_analysis("bad-array-" + mode)
        finally:
            array_path.write_bytes(saved)

    def test_family_support_and_zero_support_have_numeric_meaning(self):
        x = np.array([[1., 0.], [-1., 0.], [2., 0.], [-2., 0.]])
        actual = np.tile([0., 1.], (8, 1))
        random = np.tile([1., 0.], (8, 1))
        mask = np.array([True] * 4 + [False] * 4)
        rows = [{"family": "A" if i % 2 else "B"} for i in range(8)]
        report = a.residual_support(x, actual, random, np.zeros((8, 1)), np.ones((8, 1)), mask, rows)
        self.assertEqual(report["partitions"]["pooled"]["actual"]["identified_support_energy_fraction"], 0)
        self.assertTrue(report["partitions"]["pooled"]["actual"]["zero_or_numerically_tiny_support"])
        self.assertEqual(report["partitions"]["pooled"]["random"]["output_input_l2_gain"], 1)
        self.assertTrue(report["family_breakdown"]["available"])

    def test_import_has_no_native_model_stack(self):
        from experiments.rab001_exploration.torchfree import assert_import_is_clean
        assert_import_is_clean(self, "experiments.cross_scale_composable_001.analyze_captures")


if __name__ == "__main__":
    unittest.main()

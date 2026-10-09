"""Three pure JSON controls; no tokenizers, model, corpus or actual-run reads."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arithmetic_token_projection as P


class ProjectionControls(unittest.TestCase):
    def setUp(self):
        self.policy = (P.canonical({"rows": [{"row_id": "AUTHORED", "input_ids": [1, 2]}], "native_scoring_authorized": False}) + "\n").encode()
        self.oracle = (P.canonical({"rows": [{"row_id": "AUTHORED", "completion_ids": [3, 151645]}], "oracle_exposed_CPU_only": True}) + "\n").encode()
        def ref(name, raw): return {"path": str(HERE / name), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        self.policy_ref, self.oracle_ref = ref("inert_policy.json", self.policy), ref("inert_oracle.json", self.oracle)
        self.full = {"kind": "AUTHORED_CPU_TOKEN_RESULT", "status": "CPU_ONLY", "expected_rows": 1,
            "policy_rows": P.parse(self.policy)["rows"], "oracle_rows_CPU_only": P.parse(self.oracle)["rows"],
            "native_scoring_authorized": False}
        self.full["sha256"] = P.digest(self.full)
        self.projected = P.project(self.full, self.policy_ref, self.oracle_ref)
    def reseal(self, value):
        value["sha256"] = P.digest({k: v for k, v in value.items() if k != "sha256"}); return value
    def test_projection_has_own_seal_and_exact_full_rejoin(self):
        self.assertNotEqual(self.full["sha256"], self.projected["sha256"])
        self.assertEqual(self.full["sha256"], self.projected["full_result_sha256"])
        self.assertEqual(self.full, P.join(self.projected, self.policy, self.oracle))
        self.assertNotIn("policy_rows", self.projected); self.assertNotIn("oracle_rows_CPU_only", self.projected)
    def test_stale_array_file_ref_and_full_hash_changes_rejected_after_resealing(self):
        for change in (lambda p: p["policy_file"].update(sha256="0" * 64),
                       lambda p: p["oracle_file"].update(bytes=p["oracle_file"]["bytes"] + 1),
                       lambda p: p.update(full_result_sha256="0" * 64),
                       lambda p: p.update(expected_rows=2)):
            value = deepcopy(self.projected); change(value); self.reseal(value)
            with self.subTest(change=change), self.assertRaises(ValueError): P.join(value, self.policy, self.oracle)
        with self.assertRaises(ValueError): P.join(self.projected, self.policy + b" ", self.oracle)
        with self.assertRaises(ValueError): P.project({k: v for k, v in self.full.items() if k != "policy_rows"}, self.policy_ref, self.oracle_ref)
    def test_missing_refs_schema_duplicate_nonfinite_and_mutable_payloads_rejected(self):
        for version in (True, 1.0, "1", None, 2):
            value = deepcopy(self.projected); value["projection_schema_version"] = version; self.reseal(value)
            with self.subTest(version=version), self.assertRaises(ValueError): P.join(value, self.policy, self.oracle)
        value = deepcopy(self.projected); value.pop("policy_file"); self.reseal(value)
        with self.assertRaises(ValueError): P.join(value, self.policy, self.oracle)
        for raw in (b'{"rows":[],"rows":[]}', b'{"value":NaN}', b'{"value":1e999}'):
            with self.assertRaises(ValueError): P.parse(raw)
        with self.assertRaises(ValueError): P.join(self.projected, bytearray(self.policy), self.oracle)
        corrupted = deepcopy(self.projected); corrupted["sha256"] = "0" * 64
        with patch.object(Path, "open", side_effect=AssertionError("I/O before header validation")):
            with self.assertRaises(ValueError): P.read_join(corrupted)
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            policy_file = Path(directory) / "policy.json"; oracle_file = Path(directory) / "oracle.json"
            policy_file.write_bytes(self.policy); oracle_file.write_bytes(self.oracle)
            policy_ref = {**self.policy_ref, "path": str(policy_file)}
            oracle_ref = {**self.oracle_ref, "path": str(oracle_file)}
            projected = P.project(self.full, policy_ref, oracle_ref)
            self.assertEqual(self.full, P.read_join(projected))
            policy_file.write_bytes(self.policy + b"x" * 1024)
            with self.assertRaisesRegex(ValueError, "physical array size changed before read"): P.read_join(projected)


if __name__ == "__main__":
    files = ("arithmetic_token_projection.py", "measure_arithmetic_cpu.py", "test_arithmetic_token_projection.py")
    before = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in files}
    start = time.monotonic(); result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProjectionControls))
    value = {"kind": "ARITHMETIC_TOKEN_PROJECTION_CPU_CONTROLS001", "tests": result.testsRun,
        "success": result.wasSuccessful(), "failures": len(result.failures), "errors": len(result.errors),
        "seconds": time.monotonic() - start, "source_before": before,
        "source_after": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in files},
        "actual_run_array_or_corpus_read": False, "tokenizer_model_GPU_or_resource_probe_performed": False}
    value["sha256"] = P.digest(value); (HERE / "CPU_CONTROLS001.json").write_bytes((P.canonical(value) + "\n").encode())
    raise SystemExit(0 if result.wasSuccessful() else 1)

"""Inert tokenizer boundaries; no Rust tokenizer/model/runtime diagnostic import."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import ast
import hashlib
import json
import os
import re
import sys
import time
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
PACKAGE = next(p for p in HERE.parents if p.name == "rineya_synth_centralized_reasoning_v1")
READS = []; IMPORTS = []
def audit(event, args):
    if event == "import" and args and args[0].split(".", 1)[0] in ("tokenizers", "jinja2", "torch", "transformers", "ctypes", "psutil"):
        IMPORTS.append(args[0]); raise AssertionError("actual engine/framework/native imports forbidden in fixtures")
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        value = str(args[0]).replace("\\", "/").lower()
        if any(part in value for part in ("/data/", "token_pool", "/weights", "/model", "/check/", "/final/")):
            raise AssertionError("actual corpus/model read forbidden")
        READS.append(str(args[0]))
sys.addaudithook(audit); sys.path.insert(0, str(HERE))
import arithmetic_tokenmeasurement as M
import measure_arithmetic_cpu as R
G, T = M.modules()
tree = ast.parse((PACKAGE / "tests/test_cpu_tokenizer.py").read_bytes())
definitions = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ("FakeEngine", "FakeTemplate")]
NS = {"tokens": T, "SimpleNamespace": SimpleNamespace, "re": re, "TARGET": ""}
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(PACKAGE / "tests/test_cpu_tokenizer.py"), "exec"), NS)

class CheckedFixture:
    def __init__(self):
        self._engine = NS["FakeEngine"](); self._template = NS["FakeTemplate"]()
        self._engine.padding = self._engine.truncation = None
        self.calls = 0; self.drift = None
    def verify_binding(self):
        self.calls += 1
        return {"fixture_only": True, "identity": "CHANGED" if self.drift and self.calls >= self.drift else "INERT"}
    def encode_example(self, *args): raise AssertionError("R/S-only encoder called")

class CharacterEngine:
    padding = truncation = None
    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        values = re.findall(r"<\|im_end\|>|<\|im_start\|>|.", text, flags=re.S)
        return SimpleNamespace(ids=[151645 if x == "<|im_end|>" else 151644 if x == "<|im_start|>" else 1000 + ord(x) for x in values])
    def decode(self, values, skip_special_tokens):
        assert skip_special_tokens is False
        return "".join("<|im_end|>" if x == 151645 else "<|im_start|>" if x == 151644 else chr(x - 1000) for x in values)

class TokenControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.policy = G.policy(); cls.population = G.compile_population(cls.policy)
    def test_prompt_only_and_oracle_masks_EOS_joint_roundtrip(self):
        row = self.population["rows"][0]; fixture = CheckedFixture()
        result = M.encode_arithmetic(row, fixture, self.policy)
        p, o = result["policy_row"], result["oracle_CPU_only"]
        self.assertNotIn("target", p); self.assertNotIn("labels", p)
        self.assertEqual(o["input_ids"], p["input_ids"] + o["completion_ids"])
        self.assertEqual(o["labels"], [-100] * p["prompt_tokens"] + o["completion_ids"])
        self.assertEqual([1] * p["prompt_tokens"], p["attention_mask"])
        self.assertEqual(T.EOS_ID, o["completion_ids"][-1]); self.assertTrue(result["joint_boundaries_verified"])
    def test_all512_overflows_retained_no_resampling_or_truncation(self):
        fixture = CheckedFixture(); fixture._engine = CharacterEngine()
        value = M.measure_population(self.population, fixture, self.policy, check=lambda stage: None)
        self.assertEqual(512, value["measured_rows"]); self.assertEqual(0, value["unmeasured_rows"])
        self.assertEqual("BLOCKED_CONTEXT_FIT", value["status"]); self.assertTrue(value["overflows"])
        self.assertEqual([r["row_id"] for r in self.population["rows"]], [r["row_id"] for r in value["policy_rows"]])
        self.assertTrue(any(row["prompt_tokens"] > 480 for row in value["policy_rows"]))
        self.assertFalse(value["native_scoring_authorized"])
    def test_prefix_completion_tail_decode_targetEOS_and_typed_ID_refusals(self):
        row = self.population["rows"][0]; NS["TARGET"] = G.canonical({"answer": row["answer"]})
        for fault in ("prefix", "completion", "tail", "decode", "target_eos", "boolean_id", "empty"):
            fixture = CheckedFixture(); fixture._engine.fault = fault
            with self.subTest(fault=fault), self.assertRaises(M.EncodingFailure) as caught:
                M.encode_arithmetic(row, fixture, self.policy)
            self.assertEqual(row["row_id"], caught.exception.partial["row_id"])
    def test_padding_template_and_host_mutations_refuse_with_partial_evidence(self):
        for fault in ("padding", "template", "answer", "hash"):
            fixture = CheckedFixture(); row = deepcopy(self.population["rows"][0])
            if fault == "padding": fixture._engine.padding = {}
            elif fault == "template": fixture._template.fault = "full"
            elif fault == "answer": row["answer"] += 1
            else: row["prompt_sha256"] = "0" * 64
            error = ValueError if fault in ("answer", "hash") else M.EncodingFailure
            with self.subTest(fault=fault), self.assertRaises(error): M.encode_arithmetic(row, fixture, self.policy)
    def test_binding_drift_and_resource_failure_retains_completed_rows_and_pending_count(self):
        fixture = CheckedFixture(); fixture.drift = 5
        value = M.measure_population(self.population, fixture, self.policy, check=lambda stage: None)
        self.assertEqual("FAILED_TOKEN_BOUNDARY", value["status"]); self.assertIsNotNone(value["failure"])
        self.assertEqual(512, value["measured_rows"] + value["unmeasured_rows"])
        def fail(stage):
            if stage == "before_arithmetic_row_2": raise ValueError("inert resource cap")
        value = M.measure_population(self.population, CheckedFixture(), self.policy, check=fail)
        self.assertEqual(2, value["measured_rows"]); self.assertEqual(510, value["unmeasured_rows"])
    def test_runner_has_no_default_inventory_permit_and_blocks_weights_banks(self):
        with self.assertRaisesRegex(ValueError, "live inventory helper"): R.run(population_path=None, model_dir=None, inventory_check=None)
        for path in ("C:/model.safetensors", "C:/config.json", "C:/generation_config.json", "C:/CHECK/bank.json", "C:/FINAL/bank.json"):
            with self.subTest(path=path), self.assertRaises(ValueError): R._protect("open", (path,))
    def test_full_inert_runner_counts_output_writes_and_retains_files_on_late_budget_or_write_failure(self):
        class Watch:
            def __init__(self, output, fail): self.output = output; self.fail = fail; self.stages = []; self.bad = False
            def check(self, stage):
                self.stages.append(stage)
                if stage == "after_result_files_written":
                    assert (self.output / "policy_prompt_tokens.json").is_file()
                    assert (self.output / "oracle_CPU_only.json").is_file()
                    assert (self.output / "measurement.json").is_file()
                    if self.fail: self.bad = True; raise ValueError("inert late output budget")
            def record(self): return {"cpu_budget_satisfied": not self.bad, "last_stage": self.stages[-1]}
        for mode in ("budget", "write"):
            with tempfile.TemporaryDirectory(dir=HERE) as directory:
                root = Path(directory); output = root / "owned_output"; population = root / "population.json"
                population.write_bytes((G.canonical(self.population) + "\n").encode())
                watch = Watch(output, mode == "budget"); original_write = R._write
                def write(path, value):
                    if mode == "write" and path.name == "oracle_CPU_only.json": raise OSError("inert second write failure")
                    return original_write(path, value)
                with patch.object(R, "OUTPUT", output), patch.object(R, "_watch", lambda: watch), \
                     patch.object(R, "_write", write), patch.object(T, "load_local_tokenizer", lambda **kwargs: CheckedFixture()):
                    value = R.run(population_path=population, model_dir=None,
                        inventory_check=lambda stage: {"verified": True, "identity_sha256": "a" * 64})
                self.assertEqual("FAILED_CPU_MEASUREMENT", value["status"])
                self.assertTrue((output / "policy_prompt_tokens.json").is_file())
                self.assertTrue((output / "RUN_RECEIPT.json").is_file())
                if mode == "budget": self.assertEqual("after_result_files_written", value["resources"]["last_stage"])
                else: self.assertIn("policy_prompt_tokens.json", value["files"])

if __name__ == "__main__":
    files = ("arithmetic_tokenmeasurement.py", "measure_arithmetic_cpu.py", "test_arithmetic_tokenmeasurement.py")
    before = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in files}
    start = time.monotonic(); result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TokenControls))
    value = {"kind": "ARITHMETIC_TOKENMEASUREMENT_INERT_CONTROLS001", "tests": result.testsRun,
        "success": result.wasSuccessful(), "failures": len(result.failures), "errors": len(result.errors), "seconds": time.monotonic() - start,
        "source_before": before, "source_after": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in files},
        "read_paths": sorted(set(READS)), "forbidden_import_attempts": IMPORTS,
        "actual_tokenizer_Rust_Jinja_import_or_model_GPU_work": False, "actual_ResourceWatch_run": False}
    value["sha256"] = M.digest(value); (HERE / "CPU_CONTROLS001.json").write_bytes((M.canonical(value) + "\n").encode())
    raise SystemExit(0 if result.wasSuccessful() else 1)

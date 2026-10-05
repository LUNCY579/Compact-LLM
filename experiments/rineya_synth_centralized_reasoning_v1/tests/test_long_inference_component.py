"""Sparse, full-length CPU controls; no actual model, decoder or framework."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
STAGED = HERE.name == "long_context_qualification_work" and HERE.parent.name == "artifacts"
PACKAGE = HERE.parents[1] if STAGED else HERE.parent
CODE = HERE if STAGED else PACKAGE
ORIGINAL = PACKAGE / "evidence/long_context_component_staged_cpu_20261005/original_inference_component.py"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PACKAGE))
import long_inference_component as component

spec = importlib.util.spec_from_file_location("long_sparse_fixtures", PACKAGE / "tests/test_inference_component.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
fixtures.component = component


class LongControls(unittest.TestCase):
    def fixture(self, cap, *, forwards=65, positions=None, tokens=None):
        length = cap - 32
        torch, journal, decoder = fixtures.FakeTorch(), fixtures.Journal(), fixtures.Decoder()
        model = fixtures.Model(torch, tokens=([9] * 32 if tokens is None else tokens), prompt_length=length)
        guards = []
        engine = component.InferenceComponent(torch, model, decoder, journal.call,
            guards.append, journal.preserve, max_attempted_forwards=forwards,
            max_input_positions=(65 * length + 1023 if positions is None else positions),
            context_cap=cap, generation_cap=32, device="cpu")
        row = fixtures.row(ids=[5] * length)
        return engine, row, model, journal

    def test_full32_repeat_and_bulk_each_long_envelope(self):
        for cap in (1024, 2048, 4096):
            with self.subTest(cap=cap):
                engine, row, model, journal = self.fixture(cap)
                result = engine.qualify(row, source_packet_sha256=component.digest("synthetic_only"))
                self.assertEqual(result["status"], "COMPONENT_NUMERICAL_CONTROL_PASSED")
                self.assertTrue(result["repeat_exact"])
                self.assertEqual(result["selected_logprob_max_error"], 0)
                self.assertEqual(result["selected_logprob_mean_error"], 0)
                self.assertEqual(result["generated_tokens_per_repeat"], 32)
                self.assertEqual(len(model.calls), 65)
                self.assertEqual(max(map(len, model.calls)), cap - 1)
                self.assertEqual(model.calls[-1], row["prompt_ids"] + [9] * 31)
                costs = engine.costs()
                self.assertEqual(costs["input_positions"]["returned"], 65 * (cap - 32) + 1023)
                self.assertFalse(result["native_qualification_established"])
                self.assertFalse(result["native_launch_authorized"])

    def test_original_position_ceiling_stops_before_overspend(self):
        engine, row, model, journal = self.fixture(4096, positions=134144)
        with self.assertRaisesRegex(ValueError, "budget exhausted"):
            engine.qualify(row, source_packet_sha256=component.digest("synthetic_only"))
        self.assertEqual(len(model.calls), 32)
        self.assertEqual(engine.costs()["input_positions"]["returned"], 130544)
        self.assertTrue(engine.halted)
        self.assertEqual(journal.saved[-1][1]["status"], "COMPONENT_NUMERICAL_CONTROL_FAILED")

    def test_reservation_failure_has_no_forward_or_decode(self):
        engine, row, model, journal = self.fixture(4096)
        row["prompt_ids"].append(5); row["attention_mask"].append(1); row["prompt_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "reservation"):
            engine.collect([row], source_packet_sha256=component.digest("synthetic_only"))
        self.assertEqual(model.calls, [])
        self.assertEqual(engine.tokenizer.calls, [])

    def test_eos_preserves_actual_cost_at_long_length(self):
        engine, row, model, journal = self.fixture(4096, tokens=[9, 151645])
        output = engine.collect([row], source_packet_sha256=component.digest("synthetic_only"))
        self.assertEqual(output["rows"][0]["finish_reason"], "eos")
        self.assertFalse(output["rows"][0]["truncated"])
        self.assertEqual(output["rows"][0]["raw_generated_ids"], [9, 151645])
        self.assertEqual(engine.costs()["input_positions"]["returned"], 8129)

    def test_declared_source_delta_only(self):
        old = ORIGINAL.read_text()
        new = (CODE / "long_inference_component.py").read_text()
        expected = old.replace("Injected prompt-only greedy collection;", "Staged long prompt-only greedy collection;", 1)
        expected = expected.replace('VERSION = "synth-prompt-only-inference-component-1"',
            'VERSION = "synth-long-prompt-inference-component-staged-1"', 1)
        expected = expected.replace('require(type(context_cap) is int and 1 <= context_cap <= 512, "context cap must stay within 512")',
            'require(type(context_cap) is int and (1 <= context_cap <= 512 or context_cap in (1024, 2048, 4096)), "context cap must be short or a registered long envelope")', 1)
        self.assertEqual(new, expected)
        self.assertEqual((PACKAGE / "inference_component.py").read_bytes(), ORIGINAL.read_bytes())

    def test_no_framework_solver_or_launch_import(self):
        code = "import sys;sys.path.insert(0,sys.argv[1]);import long_inference_component;assert not any(x in sys.modules for x in ['torch','transformers','cpu_tokenizer','rs_verifier','native_worker']);print('CPU_ONLY')"
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(CODE)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "CPU_ONLY")


# Existing semantic controls explicitly exercise this derived component.
# Its independent import control above checks the new module; the old test's
# subprocess intentionally imports the original helper and is excluded here.
def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite([tests])
    for name in loader.getTestCaseNames(fixtures.ComponentTests):
        if name != "test_framework_free_import_and_no_scorer_or_launch_imports":
            suite.addTest(fixtures.ComponentTests(name))
    return suite


if __name__ == "__main__":
    unittest.main(verbosity=2)

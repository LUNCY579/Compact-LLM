"""Synthetic geometry/ledger controls; no model framework or checkpoint access."""
from copy import deepcopy
import ast
import json
from pathlib import Path
import subprocess
import sys
import unittest
from experiments.post_training_mimo_centralized_v1.scripts import diagnose_capture_geometry as diagnostic


class GeometryControls(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((diagnostic.PACKAGE / diagnostic.CONFIG_RELATIVE).read_bytes())

    def test_config_is_exact_and_grants_no_qualification(self):
        self.assertEqual(diagnostic.sha(diagnostic.PACKAGE / diagnostic.CONFIG_RELATIVE), diagnostic.CONFIG_SHA256)
        diagnostic.validate_config(self.cfg)
        for field in ("qualification_authorized", "DEV_collection_authorized", "E1_authorized", "CHECK_FINAL_access"):
            self.assertFalse(self.cfg[field])
            changed = deepcopy(self.cfg); changed[field] = True
            with self.subTest(field=field), self.assertRaises(RuntimeError): diagnostic.validate_config(changed)
        changed = deepcopy(self.cfg); changed["limits"]["gpu_worker_free_gib"] = .1
        with self.assertRaises(RuntimeError): diagnostic.validate_config(changed)

    def test_exact_failed_tokens_and_causal_positions(self):
        plan = diagnostic.causal_plan(self.cfg["prompt_ids"], self.cfg["generated_ids"])
        self.assertEqual([len(prefix) for prefix in plan["prefixes"]], [128, 129, 130])
        self.assertEqual(len(plan["full"]), 131)
        self.assertEqual(plan["target_positions"], [127, 128, 129])
        self.assertEqual(plan["aligned_full_positions"], [127, 128, 129, 130])
        self.assertEqual(plan["generated_target_count"], 3)
        self.assertEqual(plan["common_layout_position"], 130)
        for prompt, generated in ((self.cfg["prompt_ids"][:-1], self.cfg["generated_ids"]),
                                  (self.cfg["prompt_ids"], [16, 22, 151643]),
                                  (self.cfg["prompt_ids"], [16, 22, 151645, 7])):
            with self.assertRaises(RuntimeError): diagnostic.causal_plan(prompt, generated)

    def test_projection_grid_has_three_targets_and_one_fourrow_layout(self):
        plan = diagnostic.projection_plan()
        self.assertEqual(len(plan), 16)
        for dtype in ("bfloat16", "float32"):
            for origin in ("prefix", "full"):
                cells = [cell for cell in plan if cell["head_dtype"] == dtype and cell["hidden_origin"] == origin]
                self.assertEqual([(cell["head_rows"], cell["row_index"]) for cell in cells], [(1, 0), (1, 1), (1, 2), (4, None)])
                self.assertFalse(any(cell["row_index"] == 3 for cell in cells))

    def test_actual_schedule_is_eight_not_ten_and_excludes_implicit_head_calls(self):
        count = diagnostic.expected_counts()
        self.assertEqual(count["model_or_base_forwards"], 8)
        self.assertEqual(count["forward_input_positions"], 2 * sum((128, 129, 130, 131)))
        self.assertEqual(count["logical_model_calls"], 1 + 1 + 310 + 8)
        self.assertEqual(count["direct_head_projection_calls"], 16)
        self.assertEqual(count["implicit_original_LM_head_calls"], 4)
        self.assertEqual(count["new_generated_tokens"], 0)

    def test_endpoint_parity_measures_all_three_targets_and_mismatch(self):
        unchanged = diagnostic.endpoint_measurement([-1.0, -2.0, -.01], [-1.0, -2.0, -.01], [16, 22, 151645], [16, 22, 151645])
        self.assertTrue(unchanged["exact_logprob_match"])
        self.assertEqual(unchanged["argmax_changed_target_rows"], [])
        changed = diagnostic.endpoint_measurement([-1.0, -2.0, -.01], [-.95, -2.0, -.01], [16, 22, 151645], [17, 22, 151645])
        self.assertFalse(changed["exact_logprob_match"])
        self.assertAlmostEqual(changed["max_abs_model_logprob_delta"], .05)
        self.assertAlmostEqual(changed["mean_abs_model_logprob_delta"], .05 / 3)
        self.assertEqual(changed["argmax_changed_target_rows"], [0])
        with self.assertRaises(RuntimeError): diagnostic.endpoint_measurement([-1.0] * 4, [-1.0] * 4, [1] * 4, [1] * 4)
        with self.assertRaises(RuntimeError): diagnostic.endpoint_measurement([-1.0, float("nan"), -1.0], [-1.0] * 3, [1] * 3, [1] * 3)

    def test_projection_intent_precedes_call_and_return_records_actual_layout(self):
        events = []; ledger = diagnostic.ProjectionLedger(lambda kind, **fields: events.append({"event": kind, **fields}))
        metadata = {**diagnostic.projection_plan()[0], "held_hidden_sha256": "a" * 64}
        def operation():
            self.assertEqual(events[-1]["event"], "projection_attempt")
            self.assertEqual(ledger.snapshot()["pending"][0]["projection_call_id"], 1)
            return object()
        value = ledger.call(operation, metadata)
        self.assertIsNotNone(value)
        self.assertEqual([event["event"] for event in events], ["projection_attempt", "projection_returned"])
        self.assertEqual(ledger.snapshot(), {"ceiling": 16, "attempted": 1, "returned": 1, "failed": 0, "pending": []})

    def test_projection_failure_keeps_attempt_and_never_invents_a_cell(self):
        events = []; ledger = diagnostic.ProjectionLedger(lambda kind, **fields: events.append({"event": kind, **fields}))
        metadata = {**diagnostic.projection_plan()[0], "held_hidden_sha256": "a" * 64}
        def failed(): raise RuntimeError("synthetic projection failure")
        with self.assertRaises(RuntimeError): ledger.call(failed, metadata)
        self.assertEqual(ledger.snapshot(), {"ceiling": 16, "attempted": 1, "returned": 0, "failed": 1, "pending": []})
        self.assertEqual(events[-1]["event"], "projection_failed")
        self.assertEqual(events[-1]["projection_call_id"], 1)

    def test_journal_failure_before_dispatch_keeps_pending_uncertainty(self):
        invoked = []
        def failed_emit(*args, **kwargs): raise OSError("synthetic journal failure")
        ledger = diagnostic.ProjectionLedger(failed_emit)
        with self.assertRaises(OSError): ledger.call(lambda: invoked.append(True), {**diagnostic.projection_plan()[0], "held_hidden_sha256": "a" * 64})
        self.assertEqual(invoked, [])
        self.assertEqual(ledger.snapshot()["attempted"], 1)
        self.assertEqual(len(ledger.snapshot()["pending"]), 1)

    def test_projection_ceiling_and_bad_layout_stop_before_invocation(self):
        ledger = diagnostic.ProjectionLedger(lambda *args, **kwargs: None); called = []
        for cell in diagnostic.projection_plan(): ledger.call(lambda: 1, {**cell, "held_hidden_sha256": "a" * 64})
        with self.assertRaises(RuntimeError): ledger.call(lambda: called.append(True), {**diagnostic.projection_plan()[0], "held_hidden_sha256": "a" * 64})
        self.assertEqual(called, [])
        for field, value in (("head_rows", True), ("head_rows", 4.0), ("row_index", 3), ("head_dtype", "float16"),
                             ("hidden_origin", "changed"), ("held_hidden_sha256", "bad")):
            fresh = diagnostic.ProjectionLedger(lambda *args, **kwargs: None)
            cell = {**diagnostic.projection_plan()[0], "held_hidden_sha256": "a" * 64, field: value}
            with self.subTest(field=field), self.assertRaises(RuntimeError): fresh.call(lambda: called.append(True), cell)
        self.assertEqual(called, [])

    def test_reproduction_precedes_hidden_grid_and_no_synthetic_repeated_logits(self):
        source = Path(diagnostic.__file__).read_text()
        self.assertLess(source.index('event("reproduction_complete"'), source.index('prefix_rows ='))
        self.assertIn('matrix_measure(logits, [cfg["generated_ids"][index]])', source)
        self.assertNotIn('logits.repeat', source)
        self.assertIn('full[3].clone()', source)
        self.assertIn('prefix[:3].float() - full[:3].float()', source)

    def test_held_hidden_values_are_checked_after_each_head_layout(self):
        source = Path(diagnostic.__file__).read_text()
        self.assertIn('tensor_digest(torch, held[origin]) == held_hash[origin]', source)
        self.assertIn('projected_input_sha256', source)
        self.assertIn('model.lm_head.weight.float()', source)
        self.assertNotIn('model.float()', source)
        self.assertIn('parameter.dtype == torch.bfloat16', source)
        self.assertIn('model.lm_head.weight is model.model.embed_tokens.weight', source)
        self.assertIn('.unsqueeze(0)', source)
        self.assertIn('tuple(logits.shape) == (1, rows, 151936)', source)
        self.assertIn('selected_model_logprob_absolute_deltas', source)
        self.assertIn('grid attribution is limited to the controlled projection experiment', source)

    def test_failure_snapshots_current_both_ledgers_before_failure_publication(self):
        tree = ast.parse(Path(diagnostic.__file__).read_text())
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        handler = next(node for node in run.body if isinstance(node, ast.Try)).handlers[0]
        source = ast.unparse(handler)
        self.assertLess(source.index("result['call_accounting'] = accounting.snapshot()"), source.index("'FAILURE_RECEIPT.json'"))
        self.assertLess(source.index("result['projection_accounting'] = projections.snapshot()"), source.index("'FAILURE_RECEIPT.json'"))

    def test_no_model_mutation_or_decode_in_diagnostic_source(self):
        tree = ast.parse(Path(diagnostic.__file__).read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(any(isinstance(node.func, ast.Attribute) and node.func.attr in ("generate", "multinomial", "backward", "step", "train", "gradient_checkpointing_enable") for node in calls))
        self.assertFalse(any(isinstance(node.func, ast.Attribute) and node.func.attr == "decode"
                             and isinstance(node.func.value, ast.Name) and node.func.value.id == "tokenizer" for node in calls))
        self.assertTrue(any(isinstance(node.func, ast.Attribute) and node.func.attr == "no_grad" for node in calls))

    def test_imports_no_framework_or_prior_capture_producer(self):
        code = "import sys; import experiments.post_training_mimo_centralized_v1.scripts.diagnose_capture_geometry; assert not any(n == 'torch' or n.startswith('torch.') or n == 'transformers' or n.startswith('transformers.') or n.endswith('.greedy_capture') or n.endswith('.qualify_capture') for n in sys.modules)"
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

if __name__ == "__main__": unittest.main()

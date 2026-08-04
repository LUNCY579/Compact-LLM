from __future__ import annotations

import tempfile
import unittest
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from scripts.common.experiment_utils import read_json

if "transformers" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoModelForCausalLM = object
    transformers_stub.AutoTokenizer = object
    transformers_stub.StoppingCriteria = object
    transformers_stub.StoppingCriteriaList = list
    sys.modules["transformers"] = transformers_stub

from scripts.phase_1_dense_sft.evaluate_model import update_incremental_progress


class EvaluateModelIncrementalProgressTests(unittest.TestCase):
    def test_reference_evaluation_updates_live_progress_without_candidate_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "reference" / "generation"
            output.mkdir(parents=True)
            context = root / "progress_context.json"
            context.write_text(
                '{"current_phase":"phase_2b_reference_evaluation","current_candidate":"full_reference"}',
                encoding="utf-8",
            )
            args = SimpleNamespace(
                output_dir=output,
                candidate_status=None,
                run_root=root,
                progress_context=context,
                progress_update_interval=5,
                disable_live_markdown=False,
            )
            rows = [
                {
                    "id": "a",
                    "reply": "Okay.",
                    "quality_score": 0.9,
                    "quality_dimensions": {"comprehensibility": 0.9},
                    "critical_failures": [],
                    "major_issues": [],
                    "semantic_warnings": [],
                    "review_diagnostics": [],
                    "reference_token_f1": None,
                    "reference_alignment_score": 0.9,
                    "quality_pass": True,
                    "intelligibility_pass": True,
                    "quality_hard_fail": False,
                    "mechanical_integrity_score": 1.0,
                    "contract_fulfillment_score": 0.9,
                    "case_issue_cost": 0.1,
                    "checks": {"complete_utterance": True},
                }
            ]
            update_incremental_progress(args, rows, total=2, status="running", force_markdown=True)
            live = read_json(root / "LIVE_PROGRESS.json")
            self.assertEqual(live["current_candidate"], "full_reference")
            self.assertEqual(live["candidate_status"], "evaluating")
            self.assertEqual(live["examples_completed"], 1)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.phase_4_model_soups.build_greedy_soup import accepted_trial
from scripts.phase_5_context_system.context_system import Budget, assemble, retrieve_with_trace
from scripts.phase_8_release_candidate.build_release_audit import gate_status


class PostPhase3PipelineTests(unittest.TestCase):
    def test_greedy_soup_rejects_category_regression(self) -> None:
        current = {
            "mean_heuristic_score": 0.90,
            "semantic_pass_rate": 1.0,
            "critical_failure_count": 0,
            "category_scores": {"safety": 1.0, "facts": 0.8},
        }
        trial = {
            "mean_heuristic_score": 0.91,
            "semantic_pass_rate": 1.0,
            "critical_failure_count": 0,
            "category_scores": {"safety": 0.95, "facts": 0.9},
        }
        accepted, reasons = accepted_trial(current, trial, 0.0, 0.01)
        self.assertFalse(accepted)
        self.assertTrue(any("safety" in reason for reason in reasons))

    def test_retrieval_excludes_stale_contradicted_and_duplicate(self) -> None:
        memories = [
            {"id": "right", "text": "Steam tax verification is pending.", "salience": 1.0, "recency": 1.0},
            {"id": "stale", "text": "Steam tax verification is approved.", "stale": True},
            {"id": "contradicted", "text": "Publishing is already enabled.", "contradicted": True},
            {"id": "duplicate", "text": "Steam tax verification is pending.", "salience": 0.5},
        ]
        selected, trace = retrieve_with_trace("Why can I not publish while tax verification is pending?", memories, 4)
        self.assertEqual([row["id"] for row in selected], ["right"])
        reasons = {row["id"]: row["excluded_reason"] for row in trace}
        self.assertEqual(reasons["stale"], "stale")
        self.assertEqual(reasons["contradicted"], "contradicted")
        self.assertEqual(reasons["duplicate"], "duplicate")


    def test_retrieval_keywords_bridge_semantic_wording(self) -> None:
        memories = [
            {
                "id": "tax",
                "text": "Steam tax identity verification is still pending.",
                "keywords": ["publish app", "publishing blocked"],
                "salience": 1.0,
            },
            {"id": "icon", "text": "The tray icon was updated.", "salience": 0.5},
        ]
        selected, _ = retrieve_with_trace("Why can I not publish the app yet?", memories, 2)
        self.assertEqual(selected[0]["id"], "tax")

    def test_phase2_deviation_is_recognized_by_release_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            phase_gate = root / "PHASE_2_STOP_GATE.md"
            deviation = root / "PHASE_2_DEVIATION_APPROVAL.md"
            phase_gate.write_text("## Automated Gate Status\n\n`FAIL`\n", encoding="utf-8")
            deviation.write_text("- Human decision: `PASS`\n", encoding="utf-8")
            self.assertEqual(
                gate_status(phase_gate, deviation_path=deviation),
                "PASS_BY_DEVIATION",
            )

    def test_context_assembly_preserves_current_user_and_budget(self) -> None:
        prompt = assemble(
            "Reply briefly.",
            [{"text": "Relevant fact one."}, {"text": "Less relevant fact two."}],
            ["User: old turn", "Assistant: old answer"],
            "What is current?",
            Budget(total=25, memory=8, recent=6),
        )
        self.assertIn("User: What is current?", prompt)
        self.assertIn("Assistant:", prompt)

    def test_all_post_phase3_runners_execute_python_steps(self) -> None:
        root = Path(__file__).resolve().parents[1]
        runners = [
            root / "scripts/phase_4_model_soups/run_phase_4.ps1",
            root / "scripts/phase_4b_structural_reduction/run_phase_4b.ps1",
            root / "scripts/phase_5_context_system/run_phase_5.ps1",
            root / "scripts/phase_6_sparse_simulation/run_phase_6.ps1",
            root / "scripts/phase_7_sparse_runtime/run_phase_7.ps1",
            root / "scripts/phase_8_release_candidate/run_phase_8.ps1",
        ]
        for runner in runners:
            text = runner.read_text(encoding="utf-8")
            self.assertIn("Invoke-PythonStep", text, runner)
            self.assertGreaterEqual(text.count(".py"), 2, runner)


if __name__ == "__main__":
    unittest.main()

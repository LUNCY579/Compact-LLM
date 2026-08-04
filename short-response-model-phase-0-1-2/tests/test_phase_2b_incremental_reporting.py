from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.common.experiment_utils import (
    atomic_write_text,
    read_json,
    read_jsonl_recover,
    recover_atomic_write,
    write_json,
    write_jsonl,
)
from scripts.common.incremental_results import DurableJsonlStore
from scripts.phase_2_layer_selection.incremental_reporting import (
    candidate_gate_fields,
    collect_candidate_statuses,
    initialize_candidate_status,
    rebuild_reports_from_artifacts,
    transition_candidate_status,
    write_live_progress,
    write_partial_run_report,
    write_run_state,
    write_stage_report,
)
from scripts.phase_2_layer_selection.progressive_reduction import (
    _mark_run_failure,
    persist_gated_candidate_row,
    require_candidate_promotion_allowed,
)


def quality_row(case_id: str, *, quality: float = 0.9, critical: bool = False, major: bool = False) -> dict:
    critical_items = ["generation_loop"] if critical else []
    major_items = ["broken_meaning"] if major else []
    issue_free = not critical_items and not major_items
    return {
        "id": case_id,
        "category": "test",
        "reply": "Okay.",
        "quality_score": quality,
        "legacy_heuristic_score": quality,
        "quality_dimensions": {"comprehensibility": quality, "task_adequacy": quality},
        "critical_failures": critical_items,
        "major_issues": major_items,
        "minor_issues": [],
        "semantic_warnings": [],
        "review_diagnostics": [],
        "reference_token_f1": None,
        "reference_alignment_score": quality,
        "quality_pass": issue_free,
        "intelligibility_pass": not critical,
        "quality_hard_fail": critical,
        "mechanical_integrity_score": 0.0 if critical else 1.0,
        "contract_fulfillment_score": quality,
        "case_issue_cost": 1.0 - quality,
        "checks": {"complete_utterance": True},
    }


def candidate_row(root: Path, *, accepted: bool) -> dict:
    evidence = root / "evidence" / "stage_01" / ("accepted" if accepted else "rejected")
    evidence.mkdir(parents=True, exist_ok=True)
    generation = {
        "case_count": 2,
        "mean_quality_score": 0.9 if accepted else 0.4,
        "worst_decile_mean_quality_score": 0.8 if accepted else 0.1,
        "complete_utterance_rate": 1.0,
        "critical_failure_count": 0 if accepted else 1,
        "major_issue_count": 0,
        "semantic_warning_count": 0,
    }
    assessment = {
        "eligible_for_progression": accepted,
        "rejection_reasons": [] if accepted else ["quality opportunity loss exceeds memory gain"],
        "weight_memory_reduction_fraction": 0.05,
        "quality_opportunity_loss_fraction": 0.02 if accepted else 0.20,
        "opportunity_margin": 0.03 if accepted else -0.15,
        "incremental_opportunity_margin": 0.02 if accepted else -0.16,
    }
    return {
        "variant_id": evidence.name,
        "variant_kind": "raw deletion",
        "stage": 1,
        "parent_model": str(root / "reference"),
        "dropped_original_layer_id": 3,
        "retained_original_layer_ids": [0, 1, 2],
        "generation_summary": generation,
        "assessment": assessment,
        "evidence_dir": str(evidence),
    }


class Phase2BIncrementalReportingTests(unittest.TestCase):
    def test_per_example_result_is_immediately_persistent(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "results.jsonl"
            store = DurableJsonlStore.open(path, resume=False)
            store.append({"id": "a", "value": 1})
            self.assertEqual(json.loads(path.read_text().strip())["id"], "a")

    def test_resume_skips_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "results.jsonl"
            store = DurableJsonlStore.open(path, resume=False)
            self.assertTrue(store.append({"id": "a", "value": 1}))
            resumed = DurableJsonlStore.open(path, resume=True)
            self.assertFalse(resumed.append({"id": "a", "value": 2}))
            rows, _ = read_jsonl_recover(path, id_field="id")
            self.assertEqual(rows, [{"id": "a", "value": 1}])

    def test_truncated_final_jsonl_line_is_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "results.jsonl"
            path.write_text('{"id":"a","value":1}\n{"id":"b"', encoding="utf-8")
            rows, warnings = read_jsonl_recover(path, id_field="id", repair_truncated_tail=True)
            self.assertEqual([row["id"] for row in rows], ["a"])
            self.assertTrue(warnings)
            self.assertEqual(path.read_text(encoding="utf-8"), '{"id": "a", "value": 1}\n')

    def test_atomic_json_and_markdown_writes_leave_no_temp_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "state.json", {"ok": True})
            atomic_write_text(root / "report.md", "# complete\n")
            self.assertTrue(read_json(root / "state.json")["ok"])
            self.assertEqual((root / "report.md").read_text(), "# complete\n")
            self.assertFalse(any(".tmp-" in path.name for path in root.iterdir()))

    def test_orphaned_atomic_json_temp_is_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            destination = root / "state.json"
            orphan = root / ".state.json.tmp-99999999-deadbeef"
            orphan.write_text('{"status":"accepted"}\n', encoding="utf-8")
            self.assertTrue(recover_atomic_write(destination, validator="json"))
            self.assertEqual(read_json(destination)["status"], "accepted")
            self.assertFalse(orphan.exists())

    def test_live_progress_json_and_markdown_update(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = write_live_progress(
                root,
                run_status="IN PROGRESS",
                current_stage=2,
                current_candidate="candidate-x",
                examples_completed=3,
                examples_total=10,
                metrics_are_provisional=True,
                current_critical_count=1,
            )
            self.assertEqual(payload["examples_completed"], 3)
            self.assertEqual(read_json(root / "LIVE_PROGRESS.json")["current_stage"], 2)
            self.assertIn("PROVISIONAL — 3 of 10", (root / "LIVE_PROGRESS.md").read_text())

    def test_candidate_state_transitions_are_durable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "candidate_status.json"
            initialize_candidate_status(
                path,
                stage=1,
                candidate_name="c",
                parent_stage=0,
                parent_candidate="reference",
                parent_checkpoint="reference/model",
                candidate_type="raw deletion",
                removed_layers=[3],
                examples_total=10,
            )
            transition_candidate_status(path, "evaluating", examples_completed=4)
            transition_candidate_status(path, "evaluation_complete", examples_completed=10)
            status = read_json(path)
            self.assertEqual(status["status"], "evaluation_complete")
            self.assertEqual([item["status"] for item in status["state_history"]], ["pending", "evaluating", "evaluation_complete"])

    def test_completed_candidate_is_gated_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = candidate_row(root, accepted=True)
            status_path = Path(row["evidence_dir"]) / "candidate_status.json"
            initialize_candidate_status(
                status_path,
                stage=1,
                candidate_name=row["variant_id"],
                parent_stage=0,
                parent_candidate="reference",
                parent_checkpoint=row["parent_model"],
                candidate_type="raw deletion",
                removed_layers=[3],
                examples_total=2,
            )
            status = persist_gated_candidate_row(Path(row["evidence_dir"]), row)
            self.assertEqual(status["gate_decision"], "ACCEPTED")
            self.assertTrue(status["promotion_allowed"])
            self.assertTrue((Path(row["evidence_dir"]) / "CANDIDATE_REPORT.md").exists())

    def test_rejected_candidate_cannot_be_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = candidate_row(root, accepted=False)
            status_path = Path(row["evidence_dir"]) / "candidate_status.json"
            initialize_candidate_status(
                status_path,
                stage=1,
                candidate_name=row["variant_id"],
                parent_stage=0,
                parent_candidate="reference",
                parent_checkpoint=row["parent_model"],
                candidate_type="raw deletion",
                removed_layers=[3],
                examples_total=2,
            )
            persist_gated_candidate_row(Path(row["evidence_dir"]), row)
            with self.assertRaises(RuntimeError):
                require_candidate_promotion_allowed(row)

    def test_accepted_candidate_can_be_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = candidate_row(root, accepted=True)
            status_path = Path(row["evidence_dir"]) / "candidate_status.json"
            initialize_candidate_status(
                status_path,
                stage=1,
                candidate_name=row["variant_id"],
                parent_stage=0,
                parent_candidate="reference",
                parent_checkpoint=row["parent_model"],
                candidate_type="raw deletion",
                removed_layers=[3],
                examples_total=2,
            )
            persist_gated_candidate_row(Path(row["evidence_dir"]), row)
            self.assertTrue(require_candidate_promotion_allowed(row)["promotion_allowed"])

    def test_partial_report_is_available_before_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_run_state(root, run_status="IN PROGRESS", current_stage=1, current_candidate="c")
            path = write_partial_run_report(root)
            text = path.read_text()
            self.assertIn("Run status: **IN PROGRESS**", text)
            self.assertIn("Final report: **not yet available**", text)

    def test_offline_report_rebuilding_uses_saved_artifacts_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root / "evidence" / "stage_01" / "legacy_candidate"
            generation = candidate / "generation"
            generation.mkdir(parents=True)
            write_jsonl(generation / "results.jsonl", [quality_row("a"), quality_row("b")])
            write_json(candidate / "combined_metrics.json", {"assessment": {"eligible_for_progression": False}})
            result = rebuild_reports_from_artifacts(root, allow_partial=True)
            self.assertTrue(result["ok"])
            self.assertTrue((candidate / "candidate_status.json").exists())
            self.assertTrue((root / "evidence" / "stage_01" / "STAGE_01_REPORT.md").exists())
            self.assertTrue((root / "PARTIAL_RUN_REPORT.md").exists())

    def test_partial_rebuild_does_not_create_completion_summary_that_poison_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root / "evidence" / "stage_01" / "partial_candidate"
            generation = candidate / "generation"
            generation.mkdir(parents=True)
            write_jsonl(generation / "results.jsonl", [quality_row("a"), quality_row("b")])
            initialize_candidate_status(
                candidate / "candidate_status.json",
                stage=1,
                candidate_name="partial_candidate",
                parent_stage=0,
                parent_candidate="reference",
                parent_checkpoint="reference/model",
                candidate_type="raw deletion",
                removed_layers=[3],
                examples_total=3,
            )
            transition_candidate_status(
                candidate / "candidate_status.json", "evaluating", examples_completed=2, examples_total=3
            )
            result = rebuild_reports_from_artifacts(root, allow_partial=True)
            self.assertTrue(result["ok"])
            self.assertFalse((generation / "summary.json").exists())
            provisional = read_json(generation / "provisional_summary.json")
            self.assertTrue(provisional["metrics_are_provisional"])
            self.assertEqual(provisional["examples_completed"], 2)
            self.assertEqual(provisional["examples_total"], 3)

    def test_incomplete_stage_is_labeled_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root / "evidence" / "stage_02" / "candidate"
            candidate.mkdir(parents=True)
            status_path = candidate / "candidate_status.json"
            initialize_candidate_status(
                status_path,
                stage=2,
                candidate_name="candidate",
                parent_stage=1,
                parent_candidate="stage1",
                parent_checkpoint="frontier/stage1/model",
                candidate_type="raw deletion",
                removed_layers=[4],
                examples_total=10,
            )
            transition_candidate_status(status_path, "evaluating", examples_completed=3)
            write_stage_report(
                root,
                stage=2,
                parent_checkpoint="frontier/stage1/model",
                stage_status="INCOMPLETE",
                accepted_candidate=None,
                later_stages_pending=True,
            )
            self.assertIn("Stage status: **INCOMPLETE**", (root / "evidence/stage_02/STAGE_02_REPORT.md").read_text())

    def test_rejected_terminal_stage_counts_as_completed_in_partial_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_stage_report(
                root,
                stage=6,
                parent_checkpoint="frontier/stage5/model",
                stage_status="COMPLETE_REJECTED",
                accepted_candidate=None,
                later_stages_pending=False,
            )
            write_run_state(root, run_status="IN PROGRESS", current_stage=6)
            report = write_partial_run_report(root).read_text()
            self.assertIn("Completed stages: **[6]**", report)
            self.assertIn("Rejected terminal stages: **[6]**", report)
            self.assertTrue((root / "evidence/stage_06/stage_status.json").exists())

    def test_simulated_interrupt_preserves_completed_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "results.jsonl"
            store = DurableJsonlStore.open(path, resume=False)
            try:
                store.append({"id": "a", "value": 1})
                raise KeyboardInterrupt()
            except KeyboardInterrupt:
                pass
            resumed = DurableJsonlStore.open(path, resume=True)
            self.assertTrue(resumed.contains("a"))

    def test_exception_handler_updates_candidate_and_run_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root / "evidence" / "stage_01" / "candidate"
            candidate.mkdir(parents=True)
            initialize_candidate_status(
                candidate / "candidate_status.json",
                stage=1,
                candidate_name="candidate",
                parent_stage=0,
                parent_candidate="reference",
                parent_checkpoint="reference",
                candidate_type="raw deletion",
                removed_layers=[3],
                examples_total=10,
            )
            write_live_progress(root, current_stage=1, current_candidate="candidate")
            args = SimpleNamespace(
                output_root=root,
                disable_live_markdown=False,
                model_id="model",
                sft_adapter=Path("adapter"),
                train_file=Path("train.jsonl"),
                dev_file=Path("dev.jsonl"),
                probe_file=Path("probe.jsonl"),
                phase_benchmark=Path("benchmark.jsonl"),
                report=Path("report.md"),
                gate_report=Path("gate.md"),
                probe_limit=10,
                probe_seed=1,
                max_drops=2,
                recovery_top_k=1,
                recovery_steps=1,
                recovery_seed=1,
                max_new_tokens=8,
                quality_profile="balanced",
                progress_update_interval=5,
                qlora=False,
                gradient_checkpointing=False,
                skip_recovery=False,
                keep_candidate_checkpoints=False,
            )
            _mark_run_failure(args, interrupted=False, error=RuntimeError("boom"))
            self.assertEqual(read_json(candidate / "candidate_status.json")["status"], "failed")
            self.assertEqual(read_json(root / "LIVE_PROGRESS.json")["run_status"], "FAILED")
            self.assertTrue((root / "PARTIAL_RUN_REPORT.md").exists())

    def test_report_failure_does_not_erase_completed_candidate_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = candidate_row(root, accepted=True)
            candidate = Path(row["evidence_dir"])
            initialize_candidate_status(
                candidate / "candidate_status.json",
                stage=1,
                candidate_name=row["variant_id"],
                parent_stage=0,
                parent_candidate="reference",
                parent_checkpoint=row["parent_model"],
                candidate_type="raw deletion",
                removed_layers=[3],
                examples_total=2,
            )
            persist_gated_candidate_row(candidate, row)
            write_live_progress(root, current_stage=1, current_candidate=row["variant_id"])
            args = SimpleNamespace(
                output_root=root,
                disable_live_markdown=False,
                model_id="model",
                sft_adapter=Path("adapter"),
                train_file=Path("train.jsonl"),
                dev_file=Path("dev.jsonl"),
                probe_file=Path("probe.jsonl"),
                phase_benchmark=Path("benchmark.jsonl"),
                report=Path("report.md"),
                gate_report=Path("gate.md"),
                probe_limit=2,
                probe_seed=1,
                max_drops=2,
                recovery_top_k=1,
                recovery_steps=1,
                recovery_seed=1,
                max_new_tokens=8,
                quality_profile="balanced",
                progress_update_interval=5,
                qlora=False,
                gradient_checkpointing=False,
                skip_recovery=False,
                keep_candidate_checkpoints=False,
            )
            _mark_run_failure(args, interrupted=False, error=RuntimeError("report failed"))
            status = read_json(candidate / "candidate_status.json")
            self.assertEqual(status["status"], "accepted")
            self.assertTrue(status["promotion_allowed"])

    def test_rebuilding_reports_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root / "evidence" / "stage_01" / "legacy_candidate"
            generation = candidate / "generation"
            generation.mkdir(parents=True)
            write_jsonl(generation / "results.jsonl", [quality_row("a")])
            first = rebuild_reports_from_artifacts(root, allow_partial=True)
            second = rebuild_reports_from_artifacts(root, allow_partial=True)
            self.assertEqual(first["candidate_count"], second["candidate_count"])
            statuses = collect_candidate_statuses(root)
            self.assertEqual(len(statuses), 1)

    def test_old_artifact_directory_compatibility_does_not_invent_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate = root / "evidence" / "stage_03" / "old"
            generation = candidate / "generation"
            generation.mkdir(parents=True)
            write_jsonl(generation / "results.jsonl", [quality_row("a")])
            rebuild_reports_from_artifacts(root, allow_partial=True)
            status = read_json(candidate / "candidate_status.json")
            self.assertTrue(status["inferred_from_legacy_artifacts"])
            self.assertIsNone(status["gate_decision"])
            self.assertFalse(status["promotion_allowed"])

    def test_windows_compatible_paths_with_spaces(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "Phase 2B Run With Spaces"
            root.mkdir()
            write_live_progress(root, current_candidate=r"candidate\with\windows\style")
            self.assertTrue((root / "LIVE_PROGRESS.json").exists())
            self.assertIn("windows", (root / "LIVE_PROGRESS.md").read_text())

    def test_small_end_to_end_offline_smoke_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "smoke run"
            candidate = root / "evidence" / "stage_01" / "candidate"
            generation = candidate / "generation"
            generation.mkdir(parents=True)
            write_jsonl(generation / "results.jsonl", [quality_row("a"), quality_row("b", major=True)])
            command = [
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "scripts/phase_2_layer_selection/build_phase_2b_report.py"),
                "--run-root",
                str(root),
                "--allow-partial",
                "--rebuild",
            ]
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((root / "PARTIAL_RUN_REPORT.md").exists())
            self.assertIn("Final report: not generated", completed.stdout)

    def test_complete_run_offline_builder_regenerates_canonical_final_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "complete run"
            (root / "evidence").mkdir(parents=True)
            final_report = root / "PHASE_2B_FINAL.md"
            baseline_profile = {
                "profile": "full_quality",
                "remaining_layer_count": 28,
                "inference_model": "reference/full_model",
            }
            write_json(
                root / "model_catalog.json",
                {
                    "status": "PENDING_HUMAN_REVIEW",
                    "probe_case_count": 2,
                    "baseline": {
                        "generation_summary": {"mean_quality_score": 0.9},
                        "nll_summary": {"mean_target_nll": 1.0},
                    },
                    "frontier": [],
                    "profiles": {
                        "full_quality": baseline_profile,
                        "balanced": baseline_profile,
                        "min_memory": baseline_profile,
                    },
                    "termination": {
                        "reason": "no_marginally_worthwhile_deletion_after_exhaustive_recovery",
                        "terminal_stage_attempted": 1,
                        "accepted_frontier_stages": 0,
                    },
                    "run_configuration": {"probe_limit": 2, "recovery_top_k": 1},
                    "opportunity_policy": {"quality_profile": "balanced"},
                    "canonical_report": str(final_report),
                },
            )
            write_json(root / "candidate_registry.json", {"candidates": []})
            write_run_state(root, run_status="COMPLETE", final_report_available=False)
            command = [
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "scripts/phase_2_layer_selection/build_phase_2b_report.py"),
                "--run-root",
                str(root),
                "--rebuild",
            ]
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(final_report.exists())
            self.assertIn("PROGRESSIVE STRUCTURAL REDUCTION REPORT", final_report.read_text())
            self.assertTrue(read_json(root / "RUN_STATE.json")["final_report_available"])


if __name__ == "__main__":
    unittest.main()

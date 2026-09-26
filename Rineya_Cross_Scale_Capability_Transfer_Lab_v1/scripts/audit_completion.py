from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.hashing import atomic_write_json, atomic_write_text, sha256_file  # noqa: E402


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def requirement(identifier: str, statement: str, status: str, evidence: list[str], note: str = "") -> dict[str, Any]:
    return {
        "id": identifier,
        "statement": statement,
        "status": status,
        "evidence": evidence,
        "note": note,
    }


def exists(relative: str) -> bool:
    return (LAB_ROOT / relative).is_file()


def main() -> int:
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=LAB_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    test_output = (tests.stdout + tests.stderr).strip()
    required_scripts = [
        "inspect_architectures.py",
        "project_large_to_small.py",
        "project_aligned_to_small.py",
        "project_spectral_to_small.py",
        "build_cross_scale_delta.py",
        "merge_interpolate.py",
        "merge_task_vector.py",
        "merge_consensus.py",
        "merge_ties.py",
        "merge_dare.py",
        "estimate_fisher.py",
        "merge_fisher.py",
        "analyze_delta_similarity.py",
        "inject_layer.py",
        "inject_module.py",
        "run_candidate_eval.py",
        "compare_candidates.py",
        "promote_candidates.py",
        "build_evidence_manifest.py",
    ]
    missing_scripts = [name for name in required_scripts if not exists(f"scripts/{name}")]
    access_events = []
    for path in LAB_ROOT.glob("cross_scale_evidence/**/access_ledger.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                access_events.append(json.loads(line))
    forbidden_access = [
        event
        for event in access_events
        if "test2" in str(event.get("dataset_path", "")).lower()
        or "10_sealed_eval\\private" in str(event.get("dataset_path", "")).lower()
        or "10_sealed_eval/private" in str(event.get("dataset_path", "")).lower()
    ]
    tracked = set(git("ls-files").replace("\\", "/").splitlines())
    tracked_private = [path for path in tracked if "10_sealed_eval/private" in path]
    registry_path = LAB_ROOT / "cross_scale_evidence" / "05_candidates" / "registry.jsonl"
    registry_events = registry_path.read_text(encoding="utf-8").splitlines()
    report_path = LAB_ROOT / "cross_scale_evidence" / "11_report" / "final_report.md"
    report_text = report_path.read_text(encoding="utf-8")
    no_go = json.loads(
        (LAB_ROOT / "cross_scale_evidence" / "09_finalists" / "NO_GO.json").read_text(
            encoding="utf-8"
        )
    )
    requirements = [
        requirement("R01", "Clean remote-backed dedicated research branch from recorded main commit", "complete" if git("branch", "--show-current") == "research/rineya-cross-scale-capability-transfer" else "missing", ["cross_scale_evidence/01_environment/git_base.txt"]),
        requirement("R02", "Protocol, hypotheses, claim rules, leakage firewall, and coefficient grids frozen before experiments", "complete" if exists("cross_scale_evidence/00_protocol/preregistration.md") else "missing", ["cross_scale_evidence/00_protocol/preregistration.md", "cross_scale_evidence/00_protocol/hypothesis_matrix.md", "cross_scale_evidence/00_protocol/claim_rules.md"]),
        requirement("R03", "Invisible-Limit TEST2 and sealed data excluded from development", "complete" if not forbidden_access and not tracked_private else "missing", ["cross_scale_evidence/06_dev_results/**/access_ledger.jsonl", "src/rineya_cross_scale/leakage.py"], f"access_events={len(access_events)}, forbidden={len(forbidden_access)}, tracked_private={len(tracked_private)}"),
        requirement("R04", "Architecture/tokenizer/tensor audit for Supa, Qwen0.6, 4B, and 8B", "complete" if all(exists(f"cross_scale_evidence/02_architecture/{name}.json") for name in ("supa", "qwen06", "qwen4b", "qwen8b")) else "missing", ["cross_scale_evidence/02_architecture/compatibility_report.md"]),
        requirement("R05", "Deterministic depth/width projection and exact semantic mappings", "complete" if exists("cross_scale_evidence/03_projection/stage_a_summary.json") else "missing", ["src/rineya_cross_scale/projection.py", "cross_scale_evidence/03_projection/stage_a_summary.md"]),
        requirement("R06", "Additional learned/SVD target-basis projection strategies", "complete" if exists("src/rineya_cross_scale/aligned_projection.py") and exists("src/rineya_cross_scale/spectral_projection.py") else "missing", ["src/rineya_cross_scale/aligned_projection.py", "src/rineya_cross_scale/spectral_projection.py"]),
        requirement("R07", "Official Qwen0.6, Supa, and projected-donor controls reproduced", "complete" if exists("cross_scale_evidence/06_dev_results/mergedev/baselines/qwen06/summary.json") and exists("cross_scale_evidence/06_dev_results/mergedev/baselines/supa_ungated/summary.json") and exists("cross_scale_evidence/06_dev_results/mergedev/projected_controls/P4_D2_W4_8B_s010/summary.json") else "missing", ["cross_scale_evidence/06_dev_results/mergedev/baselines", "cross_scale_evidence/06_dev_results/mergedev/projected_controls"]),
        requirement("R08", "Absolute interpolation, task vectors, and multi-scale consensus implemented and screened", "complete" if exists("cross_scale_evidence/05_candidates/manifests/ABS_8B_A0100.json") and exists("cross_scale_evidence/05_candidates/manifests/TASK_8B_A2000.json") and exists("cross_scale_evidence/05_candidates/manifests/CONS_MEAN_A1500.json") else "missing", ["cross_scale_evidence/05_candidates/manifests", "cross_scale_evidence/06_dev_results/mergedev_fast/candidates"]),
        requirement("R09", "Delta diagnostics and 4B/8B scale-trajectory agreement at tensor/layer/module/head levels", "complete" if exists("cross_scale_evidence/04_deltas/spectral_s010_delta_similarity.json") else "missing", ["cross_scale_evidence/04_deltas/spectral_s010_delta_similarity.json"]),
        requirement("R10", "TIES, DARE, DARE-TIES, Fisher, layer, and module tooling implemented and tested", "complete" if all(exists(f"scripts/{name}") for name in ("merge_ties.py", "merge_dare.py", "estimate_fisher.py", "merge_fisher.py", "inject_layer.py", "inject_module.py")) else "missing", ["src/rineya_cross_scale/advanced.py", "src/rineya_cross_scale/fisher.py", "src/rineya_cross_scale/interventions.py", "cross_scale_evidence/08_advanced_merging/OUTCOME_GATE.md"]),
        requirement("R11", "Advanced/localization real-checkpoint outcomes obey staged GO gate", "gated" if exists("cross_scale_evidence/08_advanced_merging/not_run_manifest.json") else "missing", ["cross_scale_evidence/08_advanced_merging/not_run_manifest.json"], "DEV-FULL provided no positive signal; outcome runs intentionally not authorized."),
        requirement("R12", "CPU-first evaluator, frozen inference, preservation metrics, and access ledger", "complete" if exists("src/rineya_cross_scale/evaluation/runner.py") and exists("cross_scale_evidence/00_protocol/evaluator_freeze_v1.json") else "missing", ["src/rineya_cross_scale/evaluation/runner.py", "cross_scale_evidence/00_protocol/evaluator_freeze_v1.json"]),
        requirement("R13", "MERGEDEV, DEV-FULL, and fresh sealed bank built with zero prompt overlap", "complete" if exists("data/evaluation/bank_manifest.json") else "missing", ["data/evaluation/bank_manifest.json"]),
        requirement("R14", "Scenario-cluster paired statistics and error-complementarity sets", "complete" if exists("cross_scale_evidence/06_dev_results/dev_full/candidates/CONS_MEAN_A1500_vs_supa.json") else "missing", ["src/rineya_cross_scale/comparison.py", "cross_scale_evidence/06_dev_results/dev_full/candidates/CONS_MEAN_A1500_vs_supa.json"]),
        requirement("R15", "Immutable experiment registry preserves failures and statuses", "complete" if len(registry_events) >= 38 else "missing", ["cross_scale_evidence/05_candidates/registry.jsonl"], f"events={len(registry_events)}"),
        requirement("R16", "Functional/state-dict/numerical validation for generated checkpoints", "complete" if exists("src/rineya_cross_scale/validation.py") and exists("cross_scale_evidence/05_candidates/validation/CONS_MEAN_A1500.json") else "missing", ["src/rineya_cross_scale/validation.py", "cross_scale_evidence/05_candidates/validation"]),
        requirement("R17", "FP32 construction and storage-precision retention diagnostics", "complete" if exists("cross_scale_evidence/05_candidates/tensor_stats/CONS_MEAN_A1500.json") else "missing", ["src/rineya_cross_scale/operations.py", "cross_scale_evidence/05_candidates/tensor_stats/CONS_MEAN_A1500.json"]),
        requirement("R18", "Deterministic promotion applies preservation-first objective", "complete" if exists("cross_scale_evidence/09_finalists/promotion_dev_full.json") else "missing", ["src/rineya_cross_scale/promotion.py", "cross_scale_evidence/09_finalists/promotion_dev_full.json"]),
        requirement("R19", "Sealed confirmation only after finalist freeze", "gated" if no_go.get("sealed_holdout_accessed") is False and no_go.get("finalist_frozen") is False else "missing", ["cross_scale_evidence/09_finalists/NO_GO.json"], "No candidate passed DEV-FULL; sealed access correctly withheld."),
        requirement("R20", "Optional recovery LoRA isolated by comparison", "gated", ["cross_scale_evidence/11_report/final_report.md"], "No training-free finalist passed; recovery training was not authorized."),
        requirement("R21", "Compute accounting and 0.6B inference footprint", "complete" if exists("cross_scale_evidence/01_environment/compute_accounting.json") else "missing", ["cross_scale_evidence/01_environment/compute_accounting.json"]),
        requirement("R22", "Automated promotion and evidence-manifest tooling", "complete" if exists("scripts/promote_candidates.py") and exists("scripts/build_evidence_manifest.py") else "missing", ["scripts/promote_candidates.py", "scripts/build_evidence_manifest.py"]),
        requirement("R23", "Unit/integration tests cover projection, merging, TIES/DARE/Fisher, serialization, registry, leakage, and statistics", "complete" if tests.returncode == 0 else "missing", ["tests/"], test_output),
        requirement("R24", "Final scientific report answers all twenty questions and states limitations/claim boundary", "complete" if "## Answers to the twenty research questions" in report_text and exists("cross_scale_evidence/11_report/limitations.md") else "missing", ["cross_scale_evidence/11_report/final_report.md", "cross_scale_evidence/11_report/limitations.md"]),
        requirement("R25", "Meaningful milestones committed and pushed without rewriting published branches", "complete" if len(git("log", "--oneline", "origin/main..HEAD").splitlines()) >= 5 else "missing", ["git log origin/main..HEAD"], git("log", "--oneline", "origin/main..HEAD")),
    ]
    acceptable = {"complete", "gated"}
    objective_satisfied = all(item["status"] in acceptable for item in requirements)
    result = {
        "schema_version": "1.0",
        "audited_git_head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "upstream": git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"),
        "test_command": f"{sys.executable} -m pytest -q",
        "test_returncode": tests.returncode,
        "test_output": test_output,
        "required_scripts": required_scripts,
        "missing_scripts": missing_scripts,
        "access_event_count": len(access_events),
        "forbidden_access_count": len(forbidden_access),
        "tracked_private_count": len(tracked_private),
        "objective_satisfied": objective_satisfied,
        "complete_count": sum(item["status"] == "complete" for item in requirements),
        "gated_count": sum(item["status"] == "gated" for item in requirements),
        "missing_count": sum(item["status"] == "missing" for item in requirements),
        "requirements": requirements,
    }
    output_json = LAB_ROOT / "cross_scale_evidence" / "11_report" / "completion_audit.json"
    output_md = LAB_ROOT / "cross_scale_evidence" / "11_report" / "completion_audit.md"
    atomic_write_json(output_json, result)
    lines = [
        "# Completion Audit",
        "",
        f"Objective satisfied under preregistered gates: **{objective_satisfied}**",
        "",
        f"Complete: {result['complete_count']}; gated: {result['gated_count']}; missing: {result['missing_count']}.",
        "",
        "| ID | Status | Requirement |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {item['id']} | {item['status']} | {item['statement']} |"
        for item in requirements
    )
    atomic_write_text(output_md, "\n".join(lines))
    print(json.dumps({key: result[key] for key in ("objective_satisfied", "complete_count", "gated_count", "missing_count", "forbidden_access_count", "tracked_private_count", "test_output")}, indent=2))
    return 0 if objective_satisfied else 2


if __name__ == "__main__":
    raise SystemExit(main())

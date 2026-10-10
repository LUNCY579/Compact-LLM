"""Index actual source, immutable runs, reviews and safe continuation commands."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from .run_apparatus import PACKAGE, ROOT, git, sha


def main(name):
    if Path(name).name != name:
        raise ValueError("filename only")
    ledger = [json.loads(line) for line in (PACKAGE / "BUDGET_LEDGER.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    native_events = [entry for entry in ledger if entry.get("event") == "native_completed"]
    if any(not isinstance(entry.get("native_worker_seconds"), (int, float)) for entry in native_events):
        raise RuntimeError("native budget contains missing measurements; do not invent remaining time")
    native_spent = sum(entry["native_worker_seconds"] for entry in native_events)
    groups = {
        "authority_and_state": ["AUTHORITY.txt", "CAMPAIGN.md", "PRIOR_FINDINGS.json", "ENVIRONMENT.json",
            "MODEL_BINDING.json", "OWNERSHIP.json", "STATE.json", "CONTINUE.md", "RESEARCH_REPORT.md",
            "REGISTRY.jsonl", "BUDGET_LEDGER.jsonl", "CLAIM_LEDGER.jsonl", "RESEARCH_POLICY.jsonl", "STATE_HISTORY.jsonl",
            "audits/completion_audit.json", "audits/completion_audit.md", "reviews/research_policy_v5_receipt.json"],
        "original_reference_provenance": ["PREREGISTRATION.md", "GPU_HOST_REGISTRATION.md", "RESOURCE_AMENDMENT_001.md",
            "gpu_admission_v2.json", "gpu_admission_v4.json", "reviews/gpu_dispatch_receipt.json",
            "reviews/gpu_retry_dispatch_receipt.json", "analyses/gpu-reference-002/MANIFEST.json",
            "analyses/gpu-reference-002/REPORT.json", "analyses/gpu-reference-002/receipt.json",
            "reviews/gpu_result_review_002.md", "reviews/gpu_analysis_review_attempt002_receipt.json"],
        "reference_raw_evidence": ["runs/native-reference-001/result/preflight.json", "runs/native-reference-001/terminal_receipt.json",
            "runs/native-gpu-reference-001/terminal_receipt.json", "runs/native-gpu-reference-001/result/qualification.json",
            "runs/native-gpu-reference-001/result/matrix_forecast.json", "runs/native-gpu-reference-002/terminal_receipt.json",
            "runs/native-gpu-reference-002/result/receipt.json", "runs/native-gpu-reference-002/result/registration.json",
            "runs/native-gpu-reference-002/result/qualification.json", "runs/native-gpu-reference-002/result/A5_qualification.json",
            "runs/native-gpu-reference-002/result/frozen_dev_selection.json", "runs/native-gpu-reference-002/result/outcomes.jsonl",
            "runs/native-gpu-reference-002/result/trajectories.jsonl", "runs/native-gpu-reference-002/result/forwards.jsonl",
            "runs/native-gpu-reference-002/resources.jsonl", "runs/native-gpu-reference-002/admission.json",
            "runs/native-gpu-reference-002/review.json", "runs/native-gpu-reference-002/source_manifest.json"],
        "OPEN_ready_not_measured": ["OPEN_BANK_DESIGN.md", "OPEN_NATIVE_REGISTRATION.md", "open_admission_v2.json",
            "reviews/open_dispatch_receipt.json", "reviews/open_bank_review_receipt.json", "reviews/open_native_review_receipt.json",
            "runs/open-bank-apparatus-002/manifest.json", "runs/open-bank-apparatus-002/bank.json",
            "runs/open-bank-apparatus-002/audit.json", "runs/open-bank-apparatus-002/positive_controls.json",
            "runs/open-tokenizer-binding-001/OPEN_TOKENIZER_BINDING.json", "reviews/open_analysis_review_receipt.json"],
        "H3_ready_not_qualified": ["DIAGNOSTIC_DESIGN.md", "DIAGNOSTIC_NATIVE_REGISTRATION.md",
            "diagnostic_admission_v3.json", "reviews/diagnostic_dispatch_receipt.json"],
        "source_archives": ["versions/gpu_ref_v1_final/ARCHIVE.json", "versions/gpu_ref_retry1_final/ARCHIVE.json"],
        "reference_tables": ["versions/reference_tables_v1/export_reference_tables.py", "export_reference_tables.py",
            "analyses/reference-tables-001/MANIFEST.json", "analyses/reference-tables-002/MANIFEST.json",
            "analyses/reference-tables-002/policy_accuracy.csv", "analyses/reference-tables-002/paired_contrasts.csv",
            "analyses/reference-tables-002/paired_cases.csv", "analyses/reference-tables-002/alpha_features.csv",
            "analyses/reference-tables-002/instrumented_costs.csv", "reviews/reference_tables_review_001.json",
            "reviews/reference_tables_review_002.json"],
        "blocked_handoff": ["BLOCKED_RESOURCE_RECEIPT_001.json", "GOAL_BLOCKED_RECEIPT_001.json", "audits/blocked_consistency_001.json",
            "audits/blocked_consistency_001.md"],
        "larger_screen_source_and_admission": ["AUTHORITY_EXTENSION_001.md", "LARGE_BANK_DESIGN_001.md",
            "BULK_ANALYSIS_PLAN_001.md", "BULK_NATIVE_REGISTRATION_001.md", "BULK_NATIVE_DRAFT_001.md",
            "LARGER_SCREEN_STATUS_001.md", "bulk_admission_v1.json", "reviews/bulk_design_review_001.json",
            "reviews/bulk_binding_cpu_receipt_001.json", "reviews/bulk_dispatch_receipt.json",
            "reviews/research_policy_v6_receipt.json", "reviews/bulk_analysis_review_receipt.json",
            "runs/bulk-bank-apparatus-001/manifest.json", "runs/bulk-bank-apparatus-001/bank.json",
            "runs/bulk-bank-apparatus-001/audit.json", "runs/bulk-bank-apparatus-001/positive_controls.json",
            "runs/bulk-tokenizer-binding-001/BULK_TOKENIZER_BINDING.json",
            "runs/bulk-tokenizer-binding-002/BULK_TOKENIZER_BINDING.json", "versions/bulk_v1_final/ARCHIVE.json",
            "versions/bulk_binding_v1/bulk_binding.py", "versions/bulk_analysis_v1/bulk_analysis.py",
            "versions/bulk_analysis_v2/bulk_analysis.py", "BULK_READINESS_001.json"],
        "larger_native_failed_partial_evidence": ["runs/native-bulk-001/terminal_receipt.json",
            "runs/native-bulk-001/result/receipt.json", "runs/native-bulk-001/result/registration.json",
            "runs/native-bulk-001/result/qualification.json", "runs/native-bulk-001/result/matrix_forecast.json",
            "runs/native-bulk-001/result/frozen_development.json", "runs/native-bulk-001/result/A5_qualification.json",
            "runs/native-bulk-001/result/outcomes.jsonl", "runs/native-bulk-001/result/trajectories.jsonl",
            "runs/native-bulk-001/result/forwards.jsonl", "runs/native-bulk-001/result/readout_pairs.jsonl",
            "runs/native-bulk-001/resources.jsonl", "analyses/bulk-001/REPORT.json", "analyses/bulk-001/MANIFEST.json",
            "analyses/bulk-001/DEV_DIAGNOSTIC.json", "analyses/bulk-001/DEV_DIAGNOSTIC.md",
            "reviews/bulk_native_qualification_review_001.json", "reviews/bulk_result_failure_review_001.json",
            "reviews/runtime_debug_DEV_diagnostic_review_001.json", "audits/bulk_runtime_failure_001.json",
            "audits/bulk_runtime_failure_001.md"],
        "runtime_debug_pending_human_additional_attempt": ["NONNATIVEdebugproposal_001.md",
            "RUNTIME_DEBUG_REGISTRATION_001.md", "RUNTIME_DEBUG_IMPLEMENTATION_001.md", "RUNTIME_DEBUG_COMMAND_002.md",
            "runtime_debug.py", "runtime_debug_entry.py", "runtime_debug_supervisor.py",
            "runtime_debug_admission_v1.json", "runtime_debug_admission_v2.json",
            "RUNTIME_DEBUG_GRANT_TEMPLATE_001.json", "RUNTIME_DEBUG_GRANT_TEMPLATE_002.json",
            "reviews/runtime_debug_dispatch_receipt.json", "reviews/runtime_debug_packet_review_001.md",
            "versions/runtime_debug_v1/runtime_debug.py", "versions/runtime_debug_v2/runtime_debug.py",
            "versions/runtime_debug_v2_final/ARCHIVE.json"],
        "runtime_debug_authorized_completed": ["HUMAN_RUNTIME_DEBUG_ALLOWANCE_001.json",
            "HUMAN_RUNTIME_DEBUG_APPROVAL_RECORD_001.json", "RUNTIME_DEBUG_RESULT_001.md",
            "runs/native-runtime-debug-001/terminal_receipt.json", "runs/native-runtime-debug-001/result/receipt.json",
            "runs/native-runtime-debug-001/result/registration.json", "runs/native-runtime-debug-001/result/qualification.json",
            "runs/native-runtime-debug-001/result/matrix_forecast.json", "runs/native-runtime-debug-001/result/operations.jsonl",
            "runs/native-runtime-debug-001/result/forwards.jsonl", "runs/native-runtime-debug-001/result/outcomes.jsonl",
            "runs/native-runtime-debug-001/result/trajectories.jsonl", "runs/native-runtime-debug-001/resources.jsonl",
            "reviews/runtime_debug_execution_review_001.json", "audits/cuda_system_event_window_001.json",
            "audits/cuda_system_event_xml_001.json", "audits/runtime_debug_execution_math_001.json",
            "audits/runtime_debug_execution_math_001.md"],
        "post_debug_campaign_reconciliation": ["audits/post_debug_authority_audit_001.json",
            "audits/post_debug_authority_audit_001.md", "audits/completion_audit_v2.json",
            "audits/completion_audit_v2.md", "RESEARCH_POLICY_V7.md", "reviews/research_policy_v7_receipt.json",
            "H3_READINESS_POST_DEBUG_001.json", "versions/report_before_reconcile_001/RESEARCH_REPORT.md"],
    }
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "workspace": str(ROOT),
        "head": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
        "git_status": git("status", "--porcelain=v1"), "groups": {},
        "current_source": {str(p.relative_to(PACKAGE)): sha(p) for p in sorted(PACKAGE.glob("*.py"))},
        "current_tests": {str(p.relative_to(PACKAGE)): sha(p) for p in sorted((PACKAGE / "tests").glob("*.py"))},
        "native_worker_seconds": round(native_spent, 3), "native_seconds_remaining": round(21600-native_spent, 3),
        "retry_available": False, "training_steps": 0, "cloud_spending": 0,
        "completion": "UNMET;bulk192DEV complete/validationCUDAfault;authorizedshortdebugclean;originalTDRtriggerunresolved;primaryNULL",
        "live_worker": False, "safe_resume_requirements": ["No unknown/live native_worker.lock",
            "Actual parent hostRAM>=7GiB/GPUfree>=6GiB", "Current source matches exact reviewed admission",
            "Complete job deadline plus1800reserve fits remaining native allowance", "No further retry"],
        "next_OPEN_command": [r"C:\Users\clclc\BusinessProject\Chat-Bubble\.venv\Scripts\python.exe", "-B", "-m",
            "experiments.rineya_taps_adaptive_recurrence_campaign_v2.open_supervisor", "--source",
            r"C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores\Rineya-Supa-A1.1-1.1.0-devfreeze",
            "--out", str(PACKAGE / "runs/native-open-001"), "--admission", str(PACKAGE / "open_admission_v2.json")],
        "next_H3_command": [r"C:\Users\clclc\BusinessProject\Chat-Bubble\.venv\Scripts\python.exe", "-B", "-m",
            "experiments.rineya_taps_adaptive_recurrence_campaign_v2.diagnostic_supervisor", "--source",
            r"C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores\Rineya-Supa-A1.1-1.1.0-devfreeze",
            "--out", str(PACKAGE / "runs/native-diagnostic-001"), "--admission", str(PACKAGE / "diagnostic_admission_v3.json")],
        "next_bulk_command": [r"C:\Users\clclc\BusinessProject\Chat-Bubble\.venv\Scripts\python.exe", "-B", "-m",
            "experiments.rineya_taps_adaptive_recurrence_campaign_v2.bulk_supervisor", "--source",
            r"C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores\Rineya-Supa-A1.1-1.1.0-devfreeze",
            "--out", str(PACKAGE / "runs/native-bulk-001"), "--admission", str(PACKAGE / "bulk_admission_v1.json")],
        "current_priority": "Distinctoriginal-scopeH3authorized/freshgradientandresourcegatesrequired;GPUfree6GiBblocked;debugpermitCONSUMED/nobulkrestart",
        "next_runtime_debug_command_pending_approval": [r"C:\Users\clclc\BusinessProject\Chat-Bubble\.venv\Scripts\python.exe", "-B", "-m",
            "experiments.rineya_taps_adaptive_recurrence_campaign_v2.runtime_debug_supervisor", "--source",
            r"C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores\Rineya-Supa-A1.1-1.1.0-devfreeze",
            "--out", str(PACKAGE / "runs/native-runtime-debug-001"), "--admission", str(PACKAGE / "runtime_debug_admission_v2.json")],
        "actual_runtime_debug_allowance_exists": (PACKAGE / "HUMAN_RUNTIME_DEBUG_ALLOWANCE_001.json").exists(),
        "scope": "All index entries are campaign-owned or copied allowed DEV/source evidence; no model weights copied"}
    for group, names in groups.items():
        result["groups"][group] = [{"path": value, "sha256": sha(PACKAGE / value) if (PACKAGE / value).is_file() else None,
            "status": "PRESENT" if (PACKAGE / value).is_file() else "MISSING"} for value in names]
    with (PACKAGE / name).open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"index": name, "missing": [row["path"] for rows in result["groups"].values() for row in rows if row["status"] == "MISSING"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="REPRODUCIBILITY_INDEX.json")
    main(parser.parse_args().name)

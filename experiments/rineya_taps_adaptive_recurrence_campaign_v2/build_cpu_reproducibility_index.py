"""Refresh campaign provenance, including terminal and request-only causal epochs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from .run_apparatus import PACKAGE, sha


def main(name):
    if Path(name).name != name:
        raise ValueError("filename only")
    previous = "REPRODUCIBILITY_INDEX_v11.json"
    result = json.loads((PACKAGE / previous).read_text(encoding="utf-8"))
    result["previous_index_path"] = previous
    result["previous_index_sha256"] = sha(PACKAGE / previous)
    result["current_source"] = {path.name: sha(path) for path in sorted(PACKAGE.glob("*.py"))}
    result["current_tests"] = {path.relative_to(PACKAGE).as_posix(): sha(path) for path in sorted((PACKAGE / "tests").glob("*.py"))}
    for entries in result["groups"].values():
        for entry in entries:
            path = PACKAGE / entry["path"]
            entry.update(sha256=sha(path) if path.is_file() else None, status="PRESENT" if path.is_file() else "MISSING")
    names = ["cpu_h3_native.py", "cpu_h3_entry.py", "cpu_h3_supervisor.py", "cpu_h3_analysis.py",
             "cpu_h3_readiness.py", "prepare_cpu_h3_admission.py", "CPU_H3_DESIGN_001.md",
             "tests/test_cpu_h3_native.py", "tests/test_cpu_h3_supervision.py", "tests/test_cpu_h3_analysis.py",
             "tests/test_sensitivity.py", "tests/test_supervision.py",
             "CPU_H3_NATIVE_REGISTRATION_001.md", "CPU_H3_FEASIBILITY_001.json", "cpu_h3_admission_v1.json",
             "audits/cpu_H3_authority_feasibility_review_001.json", "audits/cpu_H3_math_review_001.json",
             "audits/cpu_H3_math_review_addendum_002.json", "audits/cpu_H3_native_execution_audit_001.json",
             "RESEARCH_POLICY_V8.md", "reviews/research_policy_v8_receipt.json",
             "runs/cpu-h3-analysis-cpu-003/junit.xml", "versions/cpu_h3_v1_final/ARCHIVE.json",
             "reviews/cpu_h3_parent_review_001.json", "reviews/cpu_h3_worker_review_001.json",
             "reviews/cpu_h3_dispatch_receipt.json", "runs/preflight-cpu-002/preflight.json"]
    names.extend(str(path.relative_to(PACKAGE)).replace("\\", "/") for directory in
                 (PACKAGE / "runs/native-cpu-h3-001", PACKAGE / "analyses/cpu-h3-001")
                 if directory.exists() for path in directory.rglob("*") if path.is_file())
    names.extend(str(path.relative_to(PACKAGE)).replace("\\", "/") for pattern in
                 ("CPU_H3_READINESS_*.json", "CPU_H3_RESULT_*.md", "audits/cpu_H3_execution_math_*.json",
                  "reviews/cpu_h3_execution_review_*.json") for path in PACKAGE.glob(pattern) if path.is_file())
    result["groups"]["CPU_H3_successor"] = [{"path": value, "sha256": sha(PACKAGE / value) if (PACKAGE / value).is_file() else None,
                                                   "status": "PRESENT" if (PACKAGE / value).is_file() else "MISSING"} for value in sorted(set(names))]
    proposals = [str(path.relative_to(PACKAGE)).replace("\\", "/") for path in (PACKAGE / "runs/causal-replay-DEV-export-001").glob("*.json")]
    proposals.extend(str(path.relative_to(PACKAGE)).replace("\\", "/") for path in PACKAGE.glob("CPU_CAUSAL_REPLAY_*DRAFT_001.*") if path.is_file())
    result["groups"]["causal_replay_prospective_no_native_admission"] = [{"path": value, "sha256": sha(PACKAGE / value), "status": "PRESENT_DRAFT_NOT_NATIVE_READY"} for value in sorted(set(proposals))]
    causal = [path for pattern in ("*causal_replay*.py", "CPU_CAUSAL_REPLAY_*", "*CAUSAL_REPLAY*ALLOWANCE*.json", "HUMAN_CAUSAL_REPLAY*RECORD*.json", "cpu_causal_replay_admission_v*.json")
              for path in PACKAGE.glob(pattern) if path.is_file()]
    causal.extend(path for folder in ("tests", "reviews", "audits") for path in (PACKAGE / folder).glob("*causal_replay*") if path.is_file())
    for pattern in ("runs/native-cpu-causal-replay-*", "runs/causal-replay-DEV-export-*", "analyses/cpu-causal-replay-*", "versions/cpu_causal_replay*", "versions/causal_parent*"):
        causal.extend(path for directory in PACKAGE.glob(pattern) if directory.is_dir() for path in directory.rglob("*") if path.is_file())
    result["groups"]["CPU_causal_replay_terminal_and_request_only"] = [
        {"path": path.relative_to(PACKAGE).as_posix(), "sha256": sha(path), "status": "PRESENT"}
        for path in sorted(set(causal))]
    ledger = [json.loads(line) for line in (PACKAGE / "BUDGET_LEDGER.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    spent = sum(e["native_worker_seconds"] for e in ledger if e.get("event") == "native_completed")
    result.update(created_utc=datetime.now(timezone.utc).isoformat(), timestamp_utc=datetime.now(timezone.utc).isoformat(), native_worker_seconds=round(spent, 3),
                  native_seconds_remaining=round(21600-spent, 3), live_worker=(PACKAGE / "native_worker.lock").exists(),
                  completion="UNMET; larger validation incomplete; causal task primary null after forecast refusal",
                  current_priority="Exact reviewed4500 request-only full192 study; new direct human attempt approval required",
                  safe_resume_requirements=["No live or unknown native lock", "Do not rerun consumed H3/bulk/debug attempts",
                                            "Consult current STATE/CONTINUE and exact reviewed prospective contract",
                                            "Preserve host/device floors, remaining native budget and disk/time reserves"])
    result["historical_commands_do_not_dispatch"] = {key: result.pop(key) for key in list(result) if key.startswith("next_")}
    result["CPU_H3_native_attempt_exists"] = (PACKAGE / "runs/native-cpu-h3-001").exists()
    result["causal_replay_current_admission"] = "cpu_causal_replay_admission_v4.json"
    result["causal_replay_extended_grant_exists"] = (PACKAGE / "HUMAN_CAUSAL_REPLAY_EXTENDED_ALLOWANCE_001.json").exists()
    result["causal_replay_model_dispatch_authorized"] = False
    result["retry_grants"] = {"original_infrastructure": "CONSUMED", "runtime_debug": "CONSUMED", "causal_corrected3600": "CONSUMED", "causal_extended4500": "ABSENT_NOT_AUTHORIZED"}
    result["latest_causal_attempt"] = {"run": "runs/native-cpu-causal-replay-002", "qualification": "PASS", "task_cases": 0, "primary": None, "forecast_seconds": 3778.2977913044815, "cap_seconds": 3600}
    result["historical_group_labels"] = "Legacy readiness/pending group names retain historical artifacts; consult current STATE and explicit grant statuses for live authority."
    result["groups"]["research_policy9"] = [{"path": value, "sha256": sha(PACKAGE / value), "status": "PRESENT"}
        for value in ("RESEARCH_POLICY_V9.md", "reviews/research_policy_v9_receipt.json", "reviews/research_policy_v9_receipt.md") if (PACKAGE / value).is_file()]
    result["next_CPU_H3_command"] = None if result["CPU_H3_native_attempt_exists"] else [
        r"C:\Users\clclc\BusinessProject\Chat-Bubble\.venv\Scripts\python.exe", "-B", "-m",
        "experiments.rineya_taps_adaptive_recurrence_campaign_v2.cpu_h3_supervisor", "--source",
        r"C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores\Rineya-Supa-A1.1-1.1.0-devfreeze",
        "--out", str(PACKAGE / "runs/native-cpu-h3-001"), "--admission", str(PACKAGE / "cpu_h3_admission_v1.json")]
    with (PACKAGE / name).open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"index": name, "missing": [row["path"] for entries in result["groups"].values() for row in entries if row["status"] == "MISSING"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    main(parser.parse_args().name)

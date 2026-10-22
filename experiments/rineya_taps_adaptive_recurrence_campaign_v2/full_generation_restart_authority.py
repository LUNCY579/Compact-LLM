"""Exact additional-attempt scope and anti-replay; a template grants no authority."""
import json
from pathlib import Path
from .run_apparatus import PACKAGE, sha


def validate_human_scope(admission):
    required = (PACKAGE / "HUMAN_FULL_GENERATION_RESTART_ALLOWANCE_001.json").resolve()
    path = Path(admission.get("human_extension_path", "")).resolve()
    if admission.get("requires_additional_human_authorization") is not True or path != required or not path.is_file():
        raise PermissionError("EXPLICIT_ADDITIONAL_ATTEMPT_REQUIRED: actual human allowance absent; no worker/model launch")
    expected_sha = admission.get("human_extension_sha256")
    if sha(path) != expected_sha:
        raise PermissionError("actual allowance bytes differ from independently reviewed scope")
    grant = json.loads(path.read_text(encoding="utf-8"))
    expected = {"status": "EXPLICIT_ADDITIONAL_FULL_GENERATION_RESTART_AUTHORIZED",
                "id": "FULL-GENERATION-RESTART-001", "lane": "GPU_FP32_FULL_GENERATION_RESTART",
                "attempts": 1, "max_native_worker_seconds": 16000, "model_API_attempts": 28436,
                "state_adjoints": 0, "cases": 288, "physical_evaluations": 3296, "maximum_prefix_tokens": 357,
                "scientific_generation_cap": 8, "SMOKE_generation_cap": 24, "no_further_retry": True,
                "parent_run": "native-bulk-001", "training": False,
                "full_bulk_matrix_retry": True, "worker_sha256": sha(PACKAGE / "bulk_restart_native.py")}
    if any(grant.get(key) != value for key, value in expected.items()):
        raise PermissionError("allowance does not authorize this exact original-config-preserving full generation restart")
    worker_relative = "experiments/rineya_taps_adaptive_recurrence_campaign_v2/bulk_restart_native.py"
    if (admission.get("hard_deadline_seconds") != 16000
            or admission.get("max_new_tokens") != 8
            or admission.get("qualified_maximum_prefix_tokens") != 357
            or admission.get("native_finalization_reserve_seconds") != 1800
            or admission.get("forecast", {}).get("physical_case_evaluations") != 3296
            or admission.get("forecast", {}).get("maximum_model_API_attempts") != 28436
            or admission.get("dispatch_hashes", {}).get(worker_relative) != expected["worker_sha256"]):
        raise PermissionError("admission differs from the exact full generation allowance caps/source")
    return grant


def validate_human_extension(admission):
    """Parent-only prelaunch guard; child repeats exact scope, not consumption."""
    grant = validate_human_scope(admission)
    expected_sha = admission["human_extension_sha256"]
    ledger = PACKAGE / "BUDGET_LEDGER.jsonl"
    events = [json.loads(line) for line in ledger.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    consumed = [event for event in events if event.get("human_extension_sha256") == expected_sha
                and (event.get("event") == "native_started"
                     or event.get("event") == "native_completed" and event.get("worker_pid") is not None)]
    if consumed:
        raise PermissionError("single additional full generation restart already consumed; no retry")
    return grant

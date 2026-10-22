"""Exact additional-attempt scope and anti-replay; a template grants no authority."""
import json
from pathlib import Path
from .run_apparatus import PACKAGE, sha


def validate_human_scope(admission):
    required = (PACKAGE / "HUMAN_CAUSAL_REPLAY_ALLOWANCE_001.json").resolve()
    path = Path(admission.get("human_extension_path", "")).resolve()
    if admission.get("requires_additional_human_authorization") is not True or path != required or not path.is_file():
        raise PermissionError("EXPLICIT_ADDITIONAL_ATTEMPT_REQUIRED: actual human allowance absent; no worker/model launch")
    expected_sha = admission.get("human_extension_sha256")
    if sha(path) != expected_sha:
        raise PermissionError("actual allowance bytes differ from independently reviewed scope")
    grant = json.loads(path.read_text(encoding="utf-8"))
    expected = {"status": "EXPLICIT_ADDITIONAL_CAUSAL_REPLAY_ATTEMPT_AUTHORIZED",
                "id": "CAUSAL-REPLAY-ADDITIONAL-001", "lane": "CPU_CAUSAL_REPLAY_FP32",
                "attempts": 1, "max_native_worker_seconds": 3600, "model_API_attempts": 1925,
                "state_adjoints": 635, "cases": 192, "maximum_prefix_tokens": 256,
                "parent_run": "native-cpu-causal-replay-001", "training": False,
                "full_bulk_matrix_retry": False, "worker_sha256": sha(PACKAGE / "cpu_causal_replay_native.py")}
    if any(grant.get(key) != value for key, value in expected.items()):
        raise PermissionError("allowance does not authorize this exact corrected bounded experiment")
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
        raise PermissionError("single additional causal replay attempt already consumed; no retry")
    return grant

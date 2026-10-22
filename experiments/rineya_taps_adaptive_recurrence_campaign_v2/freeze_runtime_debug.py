"""Freeze a missing-human-allowance diagnostic; never create the actual grant."""
import ast
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import hashlib
from .run_apparatus import PACKAGE, ROOT, sha


def main(version=1, expected="e927f099663844693e5963076ada2a478913d479abfbe1547f9de2f7157b467d"):
    if type(version) is not int or version < 1:
        raise ValueError("positive freeze version required")
    worker = PACKAGE / "runtime_debug.py"
    if sha(worker) != expected:
        raise RuntimeError("review-candidate worker changed; freeze a new version")
    actual_grant = PACKAGE / "HUMAN_RUNTIME_DEBUG_ALLOWANCE_001.json"
    if actual_grant.exists():
        raise RuntimeError("actual grant must be absent at proposal freeze")
    payload = {"status": "EXPLICIT_ADDITIONAL_ATTEMPT_AUTHORIZED", "lane": "GPU_FP32_RUNTIME_DEBUG",
        "attempts": 1, "max_native_worker_seconds": 300, "parent_run": "native-bulk-001",
        "worker_sha256": expected, "training": False, "full_matrix_retry": False}
    payload_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    proposed = {"status": "PROPOSED_TEMPLATE_NOT_AUTHORIZED", "actual_grant_path": str(actual_grant),
        "actual_path_exists_at_freeze": False, "proposed_payload_utf8": payload_text,
        "proposed_payload_sha256": hashlib.sha256(payload_text.encode("utf-8")).hexdigest(),
        "condition": "Only root may write these bytes after a direct human approval of this exact additional diagnostic; template is not a grant"}
    template = PACKAGE / f"RUNTIME_DEBUG_GRANT_TEMPLATE_{version:03d}.json"
    with template.open("x", encoding="utf-8") as handle:
        json.dump(proposed, handle, indent=2); handle.write("\n")
    old = json.loads((PACKAGE / "bulk_admission_v1.json").read_text())
    tree = ast.parse(worker.read_text(encoding="utf-8"))
    forecast_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "count_forecast")
    forecast = ast.literal_eval(next(node.value for node in forecast_node.body if isinstance(node, ast.Return)))
    dispatch = dict(old["dispatch_hashes"])
    names = ["runtime_debug.py", "runtime_debug_entry.py", "runtime_debug_supervisor.py",
        "tests/test_runtime_debug.py", "tests/test_runtime_debug_supervision.py",
        "RUNTIME_DEBUG_IMPLEMENTATION_001.md", "RUNTIME_DEBUG_REGISTRATION_001.md", "NONNATIVEdebugproposal_001.md",
        template.name, "audits/bulk_runtime_failure_001.json",
        "reviews/bulk_result_failure_review_001.json"]
    for name in names:
        path = PACKAGE / name
        dispatch[path.relative_to(ROOT).as_posix()] = sha(path)
    for relative, value in dispatch.items():
        if sha(ROOT / relative) != value:
            raise RuntimeError("immutable source/input changed: " + relative)
    result = dict(old)
    result.update({"id": "RUNTIME-DEBUG-GPU-001", "status": "READY_FOR_RUNTIME_DEBUG_FP32_DISPATCH",
        "lane": "GPU_FP32_RUNTIME_DEBUG", "review_receipt": "reviews/runtime_debug_dispatch_receipt.json",
        "registered_utc": datetime.now(timezone.utc).isoformat(), "hard_deadline_seconds": 300,
        "idle_forward_deadline_seconds": 300, "evidence_reserve_seconds": 25,
        "CUDA_LAUNCH_BLOCKING": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TF32": False,
        "debug_case_ids": ["BULK-TAPS-001:9463250d48f1:dialogue_memory:" + value for value in ("21:v0", "20:v1", "13:v1")],
        "smoke_case_ids": ["calibration_mathematics_429ca310e753b947", "calibration_unanswerable_abstention_3006ee0f901c7173"],
        "max_debug_new_tokens": 8, "max_smoke_new_tokens": 24, "maximum_prefix_tokens": 357,
        "fixed_alpha": 1.1473425965417514, "seed": 17, "count_forecast": forecast,
        "human_extension_path": str(actual_grant), "human_extension_sha256": proposed["proposed_payload_sha256"],
        "human_authorization_status": "MISSING_REQUIRES_DIRECT_USER_APPROVAL", "full_matrix_retry": False,
        "infrastructure_retry_available": False, "dispatch_hashes": dispatch,
        "new_native_efficacy": "NOT_RUN;diagnostic-only;no primary endpoint", "native_seconds_used_at_freeze": 1287.921,
        "native_seconds_remaining_at_freeze": 20312.079})
    result.pop("forecast", None)
    target = PACKAGE / f"runtime_debug_admission_v{version}.json"
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True); handle.write("\n")
    print(json.dumps({"admission": str(target), "sha256": sha(target), "files": len(dispatch),
        "actual_human_grant_exists": actual_grant.exists(), "native_model_loads": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--worker-sha256", default="e927f099663844693e5963076ada2a478913d479abfbe1547f9de2f7157b467d")
    args = parser.parse_args()
    main(args.version, args.worker_sha256)

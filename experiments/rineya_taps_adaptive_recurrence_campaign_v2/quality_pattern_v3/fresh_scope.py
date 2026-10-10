"""Prospective fresh-study contract; standard library only, no worker launch."""
import hashlib
import json
import math
from pathlib import Path

V3 = Path(__file__).resolve().parent
CAMPAIGN = V3.parent
ROOT = CAMPAIGN.parents[1]
GIB = 1024**3
STUDY_ID = "V3-FRESH-K-REPLICATION-001"
LANE = "GPU_FP32_FROZEN_K01_FRESH_OPEN_REPLICATION"
AUTHORITY_SHA = "18f9b5079c05138a28d783404b7c9f919d62e829c17fe0c7a7334b789dd2fa3d"
SOURCE = Path(r"C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores\Rineya-Supa-A1.1-1.1.0-devfreeze")
INTERPRETER = r"C:\Users\clclc\BusinessProject\Chat-Bubble\.venv\Scripts\python.exe"
FIXED_CONTRACT = {
    "study_id": STUDY_ID, "lane": LANE, "scientific_cases": 432, "parent_units": 216,
    "preservation_cases": 24, "task_physical_rows": 4104, "model_API_cap": 34596,
    "hard_deadline_seconds": 15600, "internal_reserve_seconds": 1800,
    "global_finalization_reserve_seconds": 1800, "maximum_prefix_tokens": 357,
    "scientific_output_cap": 8, "qualification_output_cap": 24, "extra_budgets": [0, 1],
    "training": False, "protected_data": False, "native_test": False,
    "cloud_spending": False, "readout_mutation": False, "operator_mutation": False,
    "use_cache": False, "retry_available": False, "device": "cuda:0",
    "dtype": "float32", "attn_implementation": "eager", "TF32": False,
    "parent_host_available_floor_bytes": 7 * GIB, "worker_host_available_floor_bytes": 6 * GIB,
    "worker_rss_ceiling_bytes": 6 * GIB, "running_host_reserve_bytes": GIB,
    "gpu_initial_free_floor_bytes": 6 * GIB, "gpu_allocated_ceiling_bytes": 5 * GIB,
    "gpu_global_free_reserve_bytes": GIB, "campaign_disk_ceiling_bytes": 2 * GIB,
    "campaign_disk_reserve_bytes": GIB // 2, "host_disk_free_reserve_bytes": 15 * GIB,
    "candidate_revisions_used": 3, "candidate_revisions_maximum": 6,
}
INHERITED_METADATA = {
    "experiments/anchor_routed_middle_recurrence_v1/evidence/02c_fixed_cpu_budget_v2/20260905T132743_802018Z/run_manifest.json",
    "experiments/anchor_routed_middle_recurrence_v1/evidence/00_baseline_requalification/20260905T040646_352570Z/run_manifest.json",
    "experiments/anchor_routed_middle_recurrence_v1/configs/reapplication_screen.json",
    "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/data/evaluation/supa_layer_ablation_smoke.jsonl",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_contract(admission):
    if admission.get("status") != "READY_FOR_FRESH_K01_FP32_DISPATCH":
        raise PermissionError("new reviewed fresh admission required")
    if any(admission.get(k) != value or type(admission.get(k)) is not type(value) for k, value in FIXED_CONTRACT.items()):
        raise PermissionError("fresh scientific/resource contract differs")
    if Path(admission.get("source", "")).resolve() != SOURCE.resolve() or admission.get("interpreter") != INTERPRETER:
        raise PermissionError("canonical source/interpreter required")
    if admission.get("primary_policy") != "EVIDENCE_CONSISTENCY_GUARD" or admission.get("DEV_K1_rate") != 160 / 192:
        raise PermissionError("frozen mechanical nominee/marginal differs")
    return admission


def budget_and_unused(events):
    if any(e.get("study_id") == STUDY_ID and (e.get("event") == "native_started" or
           e.get("event") == "native_completed" and e.get("worker_pid") is not None) for e in events):
        raise PermissionError("fresh study consumed; no automatic retry")
    costs = [e.get("native_worker_seconds", 0.) for e in events if e.get("event") == "native_completed"]
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in costs):
        raise ValueError("invalid charged native ledger")
    spent = sum(costs)
    if spent + FIXED_CONTRACT["hard_deadline_seconds"] + 1800 > 21600:
        raise RuntimeError("complete new parent reservation and global reserve do not fit")
    return spent


def validate_scope_files(admission):
    validate_contract(admission)
    if sha(CAMPAIGN / "AUTHORITY.txt") != AUTHORITY_SHA:
        raise PermissionError("initial authority altered")
    for name, field in (("FRESH_SCOPE_001.json", "scope_sha256"),
                        ("PRIMARY_POLICY_FREEZE_001.json", "primary_freeze_sha256"),
                        ("FRESH_POLICY_SOURCE_FREEZE_001.json", "fresh_policy_source_freeze_sha256"),
                        ("ACTIVE_GOAL_SNAPSHOT_002.json", "goal_snapshot_sha256")):
        if sha(V3 / name) != admission.get(field):
            raise PermissionError("fresh authority/primary/goal binding differs:" + name)
    scope = json.loads((V3 / "FRESH_SCOPE_001.json").read_text(encoding="utf-8"))
    if scope.get("contract") != FIXED_CONTRACT or scope.get("status") != "DISTINCT_FROZEN_OPEN_K01_RESEARCH":
        raise PermissionError("new fresh scope absent")
    goal = json.loads((V3 / "ACTIVE_GOAL_SNAPSHOT_002.json").read_text(encoding="utf-8"))["goal"]
    if goal.get("status") != "active" or scope.get("objective") != goal.get("objective"):
        raise PermissionError("active quality-research objective not bound")
    for name, status in (("reviews/opposed_replay_execution_review_001.json", "PASS_EXPOSED_DEV_REPLAY_DESCRIPTIVE_ONLY"),
                         ("reviews/fresh_bank_source_review_001.json", "PASS_FRESH_OPEN_CPU_SOURCE")):
        if sha(V3 / name) != admission.get(name + "_sha256") or json.loads((V3 / name).read_text(encoding="utf-8")).get("status") != status:
            raise PermissionError("independent prerequisite review absent:" + name)
    primary = json.loads((V3 / "PRIMARY_POLICY_FREEZE_001.json").read_text(encoding="utf-8"))
    selected = json.loads((V3 / "FRESH_POLICY_SOURCE_FREEZE_001.json").read_text(encoding="utf-8"))
    report_path = V3 / "analyses/opposed-policy-replay-001/REPORT.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (primary.get("status") != "FROZEN_MECHANICAL_PRIMARY_PENDING_INDEPENDENT_RAW_REVIEW"
            or selected.get("status") != "FROZEN_SELECTED_POLICY_FOR_FRESH_OPEN"
            or selected.get("selected_policy") != primary.get("primary_policy")
            or primary.get("primary_policy") != admission["primary_policy"]
            or report.get("status") != "COMPLETE_EXPOSED_DEV_POLICY_REPLAY"
            or report.get("mechanical_nominee") != admission["primary_policy"]
            or selected.get("fresh_nonce") != "f0216a9d60b74c8e" or selected.get("fresh_seed") != 17
            or selected.get("primary_policy_freeze_sha256") != sha(V3 / "PRIMARY_POLICY_FREEZE_001.json")
            or selected.get("replay_report_sha256") != sha(report_path)
            or selected.get("replay_execution_review_sha256") != sha(V3 / "reviews/opposed_replay_execution_review_001.json")
            or selected.get("replay_source_review_sha256") != sha(V3 / "reviews/opposed_replay_source_review_001.json")
            or selected.get("selected_policy_source_sha256") != sha(V3 / "opposed_gate_policy.py")
            or primary.get("policy_source_sha256") != sha(V3 / "opposed_gate_policy.py")):
        raise PermissionError("actual accepted mechanical nominee/replay/source pedigree differs")
    source_review = json.loads((V3 / "reviews/fresh_bank_source_review_001.json").read_text(encoding="utf-8"))
    current = {name: sha(V3 / name) for name in ("fresh_open_bank.py", "fresh_open_verifier.py", "tests/test_fresh_open_bank.py")}
    if (selected.get("fresh_source_hashes") != current or source_review.get("source_hashes") != current
            or selected.get("source_review_sha256") != sha(V3 / "reviews/fresh_bank_source_review_001.json")):
        raise PermissionError("selected data-source/producer review pedigree differs")
    return scope


def verify_dispatch(admission_path):
    if admission_path.resolve() != (V3 / "fresh_admission_v2.json").resolve():
        raise PermissionError("exact fresh admission path required")
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    validate_scope_files(admission)
    required = {"fresh_scope.py", "fresh_entry.py", "fresh_supervisor.py", "fresh_native.py",
                "opposed_gate_generation.py", "opposed_gate_policy.py", "fresh_open_bank.py", "fresh_open_verifier.py"}
    hashes = admission.get("dispatch_hashes", {})
    if not hashes or not {str((V3 / name).relative_to(ROOT)).replace("\\", "/") for name in required} <= set(hashes):
        raise PermissionError("actual new native source membership missing")
    for relative, expected in hashes.items():
        path = (ROOT / relative).resolve(strict=True)
        allowed_source = relative.endswith(".py") and relative.startswith((
            "experiments/rineya_taps_adaptive_recurrence_v1/",
            "experiments/anchor_routed_middle_recurrence_v1/src/",
            "experiments/anchor_routed_middle_recurrence_v1/scripts/",
            "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src/"))
        if (ROOT not in path.parents or not (CAMPAIGN in path.parents or allowed_source or relative in INHERITED_METADATA)
                or sha(path) != expected):
            raise PermissionError("actual native source binding differs:" + relative)
    review = (V3 / admission["review_receipt"]).resolve(strict=True)
    if V3 not in review.parents:
        raise PermissionError("fresh independent review must remain in V3")
    receipt = json.loads(review.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS_FRESH_K01_FP32_DISPATCH" or receipt.get("admission_sha256") != sha(admission_path):
        raise PermissionError("independent exact fresh admission review required")
    return admission

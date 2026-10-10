"""Record the completed no-model setup cycle from actual local receipts."""
from datetime import datetime, timezone
import json
from pathlib import Path

from experiments.rineya_taps_adaptive_recurrence_campaign_v2.run_apparatus import PACKAGE, sha


def append(name, row):
    with (PACKAGE / name).open("a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp_utc": datetime.now(timezone.utc).isoformat(), **row}, sort_keys=True) + "\n")


def main():
    marker = PACKAGE / "SETUP_RECORDED.json"
    with marker.open("x", encoding="utf-8") as f:
        json.dump({"status": "STARTED"}, f)
    for run in ("cpu-v1-001", "cpu-armr-001", "fixture-reference-001", "preflight-cpu-001"):
        receipt = PACKAGE / "runs" / run / "receipt.json"
        actual = json.loads(receipt.read_text())
        append("BUDGET_LEDGER.jsonl", {"event": "apparatus_completed", "run": run,
                "receipt": str(receipt.relative_to(PACKAGE)), "receipt_sha256": sha(receipt),
                "apparatus_wall_seconds": actual["wall_seconds"], "native_worker_seconds": 0,
                "native_model_loads": 0, "optimizer_steps": 0, "exit_code": actual["exit_code"]})
    append("BUDGET_LEDGER.jsonl", {"event": "resource_contract", "native_limit_seconds": 21600,
            "native_used_seconds": 0, "native_finalization_reserve_seconds": 1800,
            "disk_ceiling_bytes": 2 * 1024**3, "disk_finalization_reserve_bytes": 512 * 1024**2,
            "host_disk_floor_bytes": 15 * 1024**3, "max_controller_candidate_revisions": 6,
            "model_workers_max": 1, "cloud_spend": 0, "optimizer_steps": 0})
    append("REGISTRY.jsonl", {"id": "REF-V1", "parent": None, "type": "frozen_historical_reference",
            "hypotheses": ["H1", "H2", "H3", "H4-restricted"], "controller_revision_counted": False,
            "status": "REGISTERED_NOT_DISPATCHED", "registration": "PREREGISTRATION.md",
            "registration_sha256": sha(PACKAGE / "PREREGISTRATION.md")})
    append("REGISTRY.jsonl", {"id": "CONTROL-RNG-V2", "parent": "REF-V1", "type": "control_apparatus_correction",
            "hypothesis": "A5 prefix-random schedule semantics", "controller_revision_counted": False,
            "status": "CPU_TESTED_NATIVE_NOT_QUALIFIED", "code_sha256": sha(PACKAGE / "control.py"),
            "independent_review": "reviews/control_review_receipt.json"})
    append("CLAIM_LEDGER.jsonl", {"id": "C001", "claim": "62 v1 and 65 complete inherited ARMR CPU tests pass here",
            "tier": "reproduced_CPU_apparatus", "evidence": ["runs/cpu-v1-001/junit.xml", "runs/cpu-armr-001/junit.xml"],
            "native_efficacy": False, "remaining": "native model mechanics/quality not measured"})
    append("CLAIM_LEDGER.jsonl", {"id": "C002", "claim": "Reference fixture DEV K4 constant-grid headroom .2 with capture0; fixture TEST headroom0",
            "tier": "reproduced_analytic_fixture", "evidence": ["runs/fixture-reference-001/result/outcomes.jsonl", "runs/fixture-reference-001/result/headroom.csv"],
            "remaining": "No transfer to native Rineya demonstrated"})
    append("CLAIM_LEDGER.jsonl", {"id": "C003", "claim": "Exact historical metadata preflight succeeds on this host",
            "tier": "no_model_metadata", "evidence": ["runs/preflight-cpu-001/result/preflight.json"],
            "status": "METADATA_READY_NOT_QUALIFIED", "native_efficacy": False})
    append("CLAIM_LEDGER.jsonl", {"id": "C004", "claim": "Legacy A5 is per-example constant, identical under nominal repeats17/29",
            "tier": "source_audit_plus_independent_RNG_reproduction", "evidence": ["reviews/initial_issue_receipt.json"],
            "correction": "control.py CPU tested; native caller seed binding/qualification pending"})
    append("RESEARCH_POLICY.jsonl", {"version": 1, "status": "AUTHORITY_PRESERVED",
            "policy": "propose/register/implement/independent-review/qualify/measure/analyze/revise; no outcome-driven subset",
            "authority_sha256": sha(PACKAGE / "AUTHORITY.txt")})
    append("RESEARCH_POLICY.jsonl", {"version": 2, "status": "PROPOSED_REVIEW_PENDING",
            "parent": 1, "failure": "nominal RNG seeds/reset semantics were not verified end-to-end",
            "change": "Require seed-to-wrapper binding and multi-prefix stream audit before admitting stochastic controls",
            "acceptance": "Independent draws across prefix calls; moments reset; same seed replicates; different seed changes schedule; zero/unit warmup no draws; full raw schedule retained",
            "scientific_gates_weakened": False, "empirical_research_policy_improvement": "NOT_MEASURED"})
    environment = {"initial_host_snapshot": json.loads((PACKAGE / "ENVIRONMENT.initial.json").read_text(encoding="utf-8-sig")),
                   "exact_preflight_snapshot": json.loads((PACKAGE / "runs/preflight-cpu-001/result/preflight.json").read_text()),
                   "native_qualified": False}
    with (PACKAGE / "ENVIRONMENT.json").open("x", encoding="utf-8") as f:
        json.dump(environment, f, indent=2)
        f.write("\n")
    owner = json.loads((PACKAGE / "OWNERSHIP.json").read_text())
    owner["owners"].update({"binding.py": "/root/native_engineer", "tests/test_supervision.py": "/root/evidence_math", "audits/supervision_tests.md": "/root/evidence_math"})
    (PACKAGE / "OWNERSHIP.json").write_text(json.dumps(owner, indent=2) + "\n", encoding="utf-8")
    marker.write_text(json.dumps({"status": "RECORDED", "timestamp_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

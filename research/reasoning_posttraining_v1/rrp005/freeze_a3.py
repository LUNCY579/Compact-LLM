"""Freeze identity-based reuse and changed-input acquisition; preserve interrupted A2 runs."""
import datetime
import json
from pathlib import Path
import shutil

from experiments.reasoning_posttraining_v1 import repair_preservation as spec
from experiments.reasoning_posttraining_v1.controller import Journal
from experiments.model_quality_gpu_v1.integrity import digest, write_new
from experiments.reasoning_posttraining_v1.sources import stable_hash

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RECORDS = ROOT.parent / "rineya-dataset-retention-records"
old_grant = json.loads((OUT / "GRANT_A2.json").read_text())
journal = RECORDS / "preservation-baseline-a2-journal"
state = Journal(journal, old_grant).read()
if state["active_reservations"] or state["budget_breached"]:
    raise ValueError("A2 has not settled")
limits = {**state["remaining"], "model_worker_launches": 2, "retries": 1}
if limits != spec.DELTA_LIMITS:
    raise ValueError("declared A3 ceiling differs from actual A2 remainder")
reuse = RECORDS / "baseline-reuse-audit-001"
audit = json.loads((reuse / "AUDIT.json").read_text())
if (audit["native_token_and_mask_identity_checked"] != 545 or
    audit["fresh_overlap_checked"] != 301 or audit["fresh_overlap_differences"] != 0):
    raise ValueError("reuse proof does not match the proposed acquisition")
if stable_hash(Path(audit["historical_baseline"])) != audit["historical_baseline_sha256"]:
    raise ValueError("completed historical measurement changed")
new_bank = reuse / "CHANGED_OR_NEW_INPUTS.jsonl"
if stable_hash(new_bank) != audit["new_acquisition_sha256"]:
    raise ValueError("acquisition bank changed")
plan = json.loads((OUT / "PLAN_A2.json").read_text())
plan["experiment"] = spec.DELTA_EXPERIMENT
plan["baseline_protocol"].update(bank_path=str(new_bank), bank_sha256=stable_hash(new_bank),
    analysis_partitions="308 new/changed inputs only; union with 545 identity-verified prior measurements outside native-run claims",
    resource_wait=spec.WAIT_POLICY, reused_source_audit_sha256=stable_hash(reuse / "AUDIT.json"))
write_new(OUT / "PLAN_A3.json", plan)
archive = {}
for run_id in ("baseline-001", "baseline-002"):
    original = journal / "runs" / run_id
    target = OUT / "attempts" / ("a2-" + run_id)
    target.mkdir(parents=True, exist_ok=False)
    archive[run_id] = {}
    for path in sorted(original.iterdir()):
        if path.is_file():
            shutil.copyfile(path, target / path.name)
            if stable_hash(path) != stable_hash(target / path.name):
                raise ValueError("attempt copy changed bytes")
            archive[run_id][path.name] = stable_hash(path)
    archive[run_id]["durable_prediction_stream"] = str(original / "prediction-stream")
    archive[run_id]["durable_prediction_count"] = len(list((original / "prediction-stream/records").glob("*.json")))
amendment = {"recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "old_experiment": spec.AMENDED_EXPERIMENT, "new_experiment": spec.DELTA_EXPERIMENT,
    "measured_A2_remainder": state, "archive": archive,
    "new_model_calls_or_tokens_added": 0, "new_wall_gpu_seconds_added": 0,
    "added_launches": 2, "added_retries": 1,
    "reuse": {"completed_historical_predictions": 545, "native_token_identity_checks": 545,
              "fresh_token_for_token_output_checks": 301, "disagreements": 0,
              "new_or_changed_inputs": 308, "selection_by_correctness": False},
    "scope": "Source-only changed-input measurement; old completed predictions remain separately attributed, not relabeled as new native execution.",
    "resource_policy": "Pause before model work while the unchanged guard refuses; maximum 90 seconds per episode / 240 seconds total. No model-operation retry or guard override. All waiting counts in the same watchdog and wall budget.",
    "authorization": "Current user authorization for local experiments; agent-instantiated numerical envelope, no paid service, training, model selection or release.",
    "quality_claim": False}
write_new(OUT / "AMENDMENT_A3.json", amendment)
write_new(OUT / "GRANT_A3.json", {**old_grant, "experiment": spec.DELTA_EXPERIMENT,
    "plan_sha256": digest(plan), "limits": limits, "authorization_reference": str(OUT / "AMENDMENT_A3.json")})
for source, name in ((reuse / "AUDIT.json", "REUSE_AUDIT.json"),
                     (RECORDS / "audit_baseline_reuse.py", "audit_baseline_reuse.py"),
                     (RECORDS / "process-transition-probe-001/AUDIT.json", "PROCESS_TRANSITION_AUDIT.json"),
                     (RECORDS / "process_transition_probe.py", "process_transition_probe.py")):
    if (OUT / name).exists():
        raise FileExistsError(name)
    shutil.copyfile(source, OUT / name)
print("A3 frozen: 545 verified reused measurements + 308 new inputs; no added model-call/token/time budget.")

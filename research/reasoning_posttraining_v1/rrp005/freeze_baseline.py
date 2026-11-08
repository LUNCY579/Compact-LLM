"""Freeze this source-only measurement; no model execution and no claimed human review."""
import datetime
import json
from pathlib import Path
import shutil

from experiments.reasoning_posttraining_v1 import repair_preservation as spec
from experiments.model_quality_gpu_v1.integrity import digest, write_new
from experiments.reasoning_posttraining_v1.sources import stable_hash

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PREPARED = ROOT.parent / "rineya-dataset-retention-records/a1-prepared-003"
CORRECTED = ROOT.parent / "rineya-dataset-retention-records/preservation-correction-002"

for name in ("CORRECTION.json", "REPLAY_ADMISSION_AUDIT.json"):
    if (OUT / name).exists():
        raise FileExistsError(name)
    shutil.copyfile(CORRECTED / name, OUT / name)
plan = json.loads((PREPARED / "PLAN.json").read_text())
plan["experiment"] = spec.EXPERIMENT
for key in ("heldout_banks", "evaluation_protocols", "required_native_qualification"):
    plan.pop(key, None)
plan["role"] = "SOURCE_ONLY_BASELINE_NO_TRAINING"
plan["training_fields_note"] = "Inherited fields identify the immutable prepared artifact for the existing controller; no training presentations are consumed in this baseline phase. Training budget is exactly zero."
bank = CORRECTED / "RRP5_BASELINE_INPUTS.jsonl"
plan["baseline_protocol"] = {"bank_path": str(bank), "bank_sha256": stable_hash(bank),
                             "max_sequence": 2048, "max_new_tokens": 64,
                             "serialization": "dialogue_final_response_v1",
                             "inference_qualification": "in-run-2048-zero-update",
                             "analysis_partitions": "757 main / 96 negative probes; reuse 96 main positives by exact alias"}
write_new(OUT / "PLAN.json", plan)
authorization = {"recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "authority": "current user message and goal continuation",
                 "user_verbatim": "Fully arrange the experiments and wisely utilize the GPU and CPU to start enhancing the models. I will give you the full permissions to improve and enhance the codebase as long as the models' reasoning abilities can be proven to improve",
                 "numeric_provenance": "Agent-instantiated limits; not user-quoted numbers or an inherited permit.",
                 "scope": "Corrected source-only retention measurement needed to avoid an invalid training gate.",
                 "human_data_review": "NOT_PERFORMED", "quality_claim": False}
write_new(OUT / "SESSION_AUTHORIZATION.json", authorization)
grant = {"experiment": spec.EXPERIMENT, "decision": "APPROVED", "training_permission": True,
         "what_training_permission_means": "Legacy controller name for engineering-budget permission. This lane forbids training and gives it zero tokens.",
         "approved_by": "current-session user broad local-compute authority; agent-instantiated numeric ceiling",
         "authorization_reference": str(OUT / "SESSION_AUTHORIZATION.json"),
         "data_contract_sha256": stable_hash(PREPARED / "DATA_CONTRACT.json"),
         "plan_sha256": digest(plan), "limits": spec.LIMITS, "excludes": list(spec.EXCLUDES),
         "human_data_review": "NOT_PERFORMED", "quality_claim_permitted": False,
         "release_permitted": False}
write_new(OUT / "GRANT.json", grant)
print("Frozen source-only RRP-005 plan and scoped grant; model calls 0.")

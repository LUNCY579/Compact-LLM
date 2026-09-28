"""Preserve the real failed attempt and prospectively bind the durable evaluation successor."""
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
old_grant = json.loads((OUT / "GRANT.json").read_text())
journal = RECORDS / "preservation-baseline-journal"
state = Journal(journal, old_grant).read()
if state["active_reservations"] or state["budget_breached"]:
    raise ValueError("old model worker has not settled cleanly")
expected = {"evaluation_and_qualification_model_calls": 554,
            "generated_tokens_all_candidates": 35072,
            "model_worker_wall_seconds_and_conservative_gpu_seconds": 423,
            "processed_training_tokens_all_attempts": 0, "model_worker_launches": 1}
if any(state["used_or_reserved"][k] != v for k, v in expected.items()):
    raise ValueError("old measured spend differs; recompute the amendment before authorizing it")
limits = dict(state["remaining"])
limits["evaluation_and_qualification_model_calls"] += 1000
limits["generated_tokens_all_candidates"] += 65536
limits["model_worker_launches"] += 1
if limits != spec.AMENDED_LIMITS:
    raise ValueError("amended ceiling does not equal the declared remainder plus additions")
attempt = journal / "runs/baseline-001"
copied = OUT / "attempts/baseline-001"
copied.mkdir(parents=True, exist_ok=False)
hashes = {}
for path in sorted(attempt.iterdir()):
    if path.is_file():
        target = copied / path.name
        shutil.copyfile(path, target)
        if stable_hash(path) != stable_hash(target):
            raise ValueError("attempt archive copy differs")
        hashes[path.name] = stable_hash(target)
plan = json.loads((OUT / "PLAN.json").read_text())
plan["experiment"] = spec.AMENDED_EXPERIMENT
plan["baseline_protocol"].update(prediction_storage="create_only_hash_chained_rows_v1",
    cache_policy="release_unused_between_cases")
write_new(OUT / "PLAN_A2.json", plan)
amendment = {"recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "old_experiment": spec.EXPERIMENT, "new_experiment": spec.AMENDED_EXPERIMENT,
    "old_journal_measured": state, "old_attempt_files": hashes,
    "reason": "baseline-001 reached its 548th input, then the 1.5 GiB host-memory floor refused at 1.49 GiB. No predictions survived because the evaluator wrote only on completion. It is a failed attempt, not a partial accuracy result.",
    "changes": ["fsynced hash-chained record after every completed case", "same-binding source-only prediction continuation", "persist qualification before evaluation", "release only this process's unused CUDA cache between independent cases", "per-row resource telemetry"],
    "added_budget": {"evaluation_and_qualification_model_calls": 1000,
                     "generated_tokens_all_candidates": 65536, "model_worker_launches": 1},
    "unchanged_total_budget": "No additional wall/GPU seconds, disk, training, adaptive selection, RL or paid spend; A2 receives the exact old remainder on these axes.",
    "authorization": "The current user's explicit local CPU/GPU experimentation authorization remains in force; these limits are agent-instantiated and never attributed as user-quoted numbers.",
    "cache_policy_reference": "https://docs.pytorch.org/docs/2.10/generated/torch.cuda.memory.empty_cache.html",
    "cache_claim_limit": "Releases unoccupied allocator blocks, never live parameters. Host-memory benefit is a hypothesis to measure, not established by documentation.",
    "resource_guards_lowered": False, "lost_predictions_recovered": False,
    "quality_claim": False}
write_new(OUT / "AMENDMENT_A2.json", amendment)
grant = {**old_grant, "experiment": spec.AMENDED_EXPERIMENT, "plan_sha256": digest(plan),
         "limits": limits, "authorization_reference": str(OUT / "AMENDMENT_A2.json")}
write_new(OUT / "GRANT_A2.json", grant)
survey = RECORDS / "scope-efficiency-001/TOKEN_SURVEY.json"
shutil.copyfile(survey, OUT / "SCOPE_TOKEN_SURVEY.json")
shutil.copyfile(RECORDS / "scope_efficiency_probe.py", OUT / "scope_efficiency_probe.py")
print("Archived failed attempt, froze A2 from measured remaining budget, preserved tokenizer-only survey.")

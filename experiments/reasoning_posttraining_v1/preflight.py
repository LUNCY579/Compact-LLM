"""Build a concrete bounded microfit request. This command cannot execute a model."""
from __future__ import annotations
import argparse
from pathlib import Path
import importlib.metadata

from experiments.model_quality_gpu_v1.data import group_updates
from experiments.model_quality_gpu_v1.integrity import digest, new_dir, read_json, write_new
from .prepare import read_rows
from .sources import safetensor_inventory, stable_hash

LIMITS = {
    "total_owned_work_wall_seconds": 3600,
    "model_worker_wall_seconds_and_conservative_gpu_seconds": 2400,
    "processed_training_tokens_all_attempts": 400000,
    "evaluation_and_qualification_model_calls": 160,
    "generated_tokens_all_candidates": 20480,
    "on_policy_rl_rollout_tokens": 0,
    "new_disk_bytes": 16 * 2**30,
    "adaptive_model_selection_rounds": 1,
    "model_worker_launches": 4,
    "retries": 1,
    "cloud_or_teacher_spend": 0,
}


def build_plan(contract: dict, rows: list[dict]) -> dict:
    if contract["microfit_source"] != "forge_v1_primitives" or contract["selected_worlds"] != 32:
        raise ValueError("this request covers only the specified 32-world primitive microfit")
    if [r["id"] for r in rows] != contract["selected_ids"] or len({r["parent_id"] for r in rows}) != 32:
        raise ValueError("selected world/order mismatch")
    if any(r["source_split"] != "train" or len(r["ids"]) > 2048 or len(r["prompt_ids"]) + 128 > 2048 for r in rows):
        raise ValueError("source split or native length differs")
    repeated = [r for _ in range(8) for r in rows]
    groups = group_updates(repeated, 2048)
    processed = sum(len(r["ids"]) for r in repeated)
    targets = sum(sum(t != -100 for t in r["labels"][1:]) for r in repeated)
    if processed * 2 + 4096 > LIMITS["processed_training_tokens_all_attempts"]:
        raise ValueError("candidate plus complete resume verification exceeds proposed allowance")
    return {"experiment": "RRP-001-MICROFIT", "kind": "PROPOSED_ENGINEERING_EXPERIMENT_NOT_EXECUTED",
            "source": "exact ungated Supa 1.7", "trainable": "blocks8-27", "unique_worlds": 32,
            "exposures_per_world": 8, "presentations": len(repeated), "processed_tokens": processed,
            "supervised_tokens": targets, "update_groups": groups,
            "row_order_per_exposure": contract["selected_ids"], "selection_seed": 17, "order_seed": 17,
            "objective": "assistant_response_plus_eos", "precision": "fp32",
            "optimizer": {"name": "MQ CpuAdamW", "betas": [.9, .95], "epsilon": 1e-8, "weight_decay": 0,
                          "gradient_clip": 1, "peak_lr": 1e-5, "schedule": "WSD20/60/20_to_0.1_peak"},
            "max_sequence": 2048, "max_new_tokens": 128,
            "training_only_success": {"minimum_exact_canonical_correct": 31, "denominator": 32,
                                      "finite_gradients": True, "frozen_weights_unchanged": True},
            "checkpoint_policy": "Preserve update2 and final; resume in a new attempt and preserve its final checkpoint",
            "resume_verification": "fresh source load from complete update2 checkpoint; final parameters and predictions must match",
            "evaluation_schedule": "source baseline, fixed midpoint, candidate final, resumed final; at most128 generations plus32 qualification calls",
            "quality_claim": False, "preservation_or_generalization_claim": False,
            "native_runner_status": "IMPLEMENTATION_AND_QUALIFICATION_REQUIRED"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "source", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args(argv)
    out = new_dir(a.out)
    try:
        contract = read_json(a.prepared / "DATA_CONTRACT.json")
        for name, sha in contract["files"].items():
            if stable_hash(a.prepared / name) != sha:
                raise ValueError("prepared artifact changed")
        rows = list(read_rows(a.prepared / "microfit-train.jsonl"))
        plan = build_plan(contract, rows)
        inventory = safetensor_inventory(a.source / "model.safetensors",
            "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b")
        request = {"experiment": "RRP-001-MICROFIT", "decision": "PENDING", "approved_by": None,
                   "authorization_reference": None, "data_contract_sha256": stable_hash(a.prepared / "DATA_CONTRACT.json"),
                   "plan_sha256": digest(plan), "limits": LIMITS,
                   "scope": "local zero-update qualification, one32-world engineering microfit and fresh-process resume verification",
                   "human_data_review": "NOT_PERFORMED; explicit engineering-only authorization requested, no human approval inferred",
                   "excludes": ["broad_SFT", "RL", "larger_teacher", "cloud", "release", "architectural_change"],
                   "training_permission": False}
        write_new(out / "MICROFIT_PLAN.json", plan)
        write_new(out / "BUDGET_REQUEST.json", request)
        write_new(out / "SOURCE_INVENTORY.json", inventory)
        write_new(out / "PREFLIGHT.json", {"status": "BUDGET_AND_NATIVE_RUNNER_PENDING", "plan_sha256": digest(plan),
            "stored_inventory_verified": True, "native_instantiated_inventory": "NOT_RUN", "model_calls": 0,
            "runtime_packages": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "safetensors", "tokenizers"]},
            "resource_guards": "existing MQ guards retained; fresh actual fit and model-job ownership required before model load",
            "approved_budget_remaining": {name: 0 for name in LIMITS}, "proposed_limits_are_not_authorization": True})
        print({"status": "BUDGET_AND_NATIVE_RUNNER_PENDING", "processed_tokens": plan["processed_tokens"],
               "supervised_tokens": plan["supervised_tokens"], "optimizer_groups": len(plan["update_groups"]),
               "model_calls": 0})
    except Exception as exc:
        write_new(out / "failure.json", {"type": type(exc).__name__, "message": str(exc), "model_calls": 0})
        raise


if __name__ == "__main__": main()

"""Create-only preparation for the prospectively registered RRP-015 local pilot."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd().resolve()))
from experiments.model_quality_gpu_v1.integrity import digest, write_new
from experiments.model_quality_gpu_v1.native import load_tokenizer
from experiments.reasoning_posttraining_v1 import process_arm, process_dose, process_mix
from experiments.reasoning_posttraining_v1.sources import stable_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prereg", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError("pilot preparation is create-only")
    args.out.mkdir(parents=True)
    original, records = args.original.resolve(), args.records.resolve()
    binding = json.loads((records / "a1-prepared-003/SOURCE_BINDING.json").read_text())
    views = records / "process-transition-probe-001/CANDIDATE_VIEWS.jsonl"
    selection = records / "select-views-001/SELECT_VIEWS.jsonl"
    if stable_hash(selection) != process_arm.SELECT_VIEWS_SHA256:
        raise ValueError("selection instrument differs from registration")
    source = original / "Supa"
    if stable_hash(source / "model.safetensors") != "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b":
        raise ValueError("Supa source differs from registration")
    tokenizer = load_tokenizer(source)
    rehearsal = (original / "Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0"
                 / "data/sft/rineya_v1_v2_preservation_slice_14000.jsonl")
    if not rehearsal.is_file():
        raise ValueError(f"registered rehearsal source absent: {rehearsal}")
    crows, cplan, creport = process_arm.build(
        binding, views, tokenizer, per_family=128, exposures=1, selection_seed=17,
        experiment=process_dose.EXPERIMENT, also_exclude_views=[selection])
    rrows, rplan, rreport = process_mix.build(
        binding, views, rehearsal, tokenizer, per_family=128, exposures=1,
        selection_seed=17, select="abstention", also_exclude_views=[selection])
    ctrans = {r["id"]: r for r in crows}
    rtrans = {r["id"]: r for r in rrows if r["source_release"] == "forge_v2"}
    if ctrans != rtrans or len(ctrans) != 384:
        raise ValueError("arms must contain the identical 384 complete transition examples")
    summaries = {}
    for tag, rows, plan, report, lane in (
        ("control", crows, cplan, creport, process_dose),
        ("rehearsal", rrows, rplan, rreport, process_mix),
    ):
        prepared = args.out / f"{tag}-prepared"
        process_arm.write_prepared(prepared, rows, report)
        write_new(args.out / f"{tag}-PLAN.json", plan)
        limits = {
            "total_owned_work_wall_seconds": 1800,
            "model_worker_wall_seconds_and_conservative_gpu_seconds": 1800,
            "processed_training_tokens_all_attempts": 600000,
            "evaluation_and_qualification_model_calls": 240,
            "generated_tokens_all_candidates": 30720,
            "on_policy_rl_rollout_tokens": 0,
            "new_disk_bytes": 8 * 2**30,
            "adaptive_model_selection_rounds": 1,
            "model_worker_launches": 2,
            "retries": 1,
            "cloud_or_teacher_spend": 0,
        }
        grant = {
            "approved_by": "current user broad local-compute authority; agent-chosen numeric bounds",
            "authorization_reference": str(args.prereg.resolve()),
            "authorization_prereg_sha256": stable_hash(args.prereg),
            "experiment": lane.EXPERIMENT, "comparison": "RRP-015", "arm": tag,
            "decision": "APPROVED", "training_permission": True,
            "human_data_review": "NOT_PERFORMED", "quality_claim_permitted": False,
            "release_permitted": False, "excludes": list(lane.EXCLUDES),
            "limits": limits, "plan_sha256": digest(plan),
            "data_contract_sha256": stable_hash(prepared / "DATA_CONTRACT.json"),
        }
        from experiments.reasoning_posttraining_v1.budget import validate_grant
        validate_grant(grant, grant["data_contract_sha256"], grant["plan_sha256"], lane.EXPERIMENT)
        write_new(args.out / f"{tag}-GRANT.json", grant)
        summaries[tag] = {
            "rows": len(rows), "transition_rows": len(ctrans),
            "processed_tokens": plan["processed_tokens"],
            "supervised_tokens": plan["supervised_tokens"],
            "updates": len(plan["update_groups"]),
            "max_rendered_tokens": max(len(r["ids"]) for r in rows),
            "plan_sha256": digest(plan),
            "data_contract_sha256": grant["data_contract_sha256"],
            "new_source_qualification_required": True,
        }
    write_new(args.out / "PREPARATION.json", {
        "comparison": "RRP-015", "arms": summaries,
        "same_transition_rows_verified": True,
        "prereg_sha256": stable_hash(args.prereg),
        "source_weight_sha256": stable_hash(source / "model.safetensors"),
        "selection_sha256": stable_hash(selection),
        "rehearsal_sha256": stable_hash(rehearsal),
        "optimizer_steps": 0, "model_calls": 0, "quality_claim": False,
    })
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

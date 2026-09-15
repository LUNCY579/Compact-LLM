"""RRP-004: corrected, shuffled Forge control before a retention replay contrast.

Imports stay torch-free. Registration is a ceiling, never execution permission.
The historical A1 plan, grant and prepared bytes remain unchanged.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path

from experiments.model_quality_gpu_v1.integrity import digest, write_new
from . import budget

EXPERIMENT = "RRP-004-A1-CORRECTED"
LIMITS = {
    "total_owned_work_wall_seconds": 16000,
    "model_worker_wall_seconds_and_conservative_gpu_seconds": 12000,
    "processed_training_tokens_all_attempts": 4000000,
    "evaluation_and_qualification_model_calls": 3000,
    "generated_tokens_all_candidates": 262144,
    "on_policy_rl_rollout_tokens": 0,
    "new_disk_bytes": 10 * 2**30,
    "adaptive_model_selection_rounds": 1,
    "model_worker_launches": 2,
    "retries": 1,
    "cloud_or_teacher_spend": 0,
}
EXCLUDES = ("RL", "larger_teacher", "cloud", "release", "architectural_change")
budget.register(EXPERIMENT, ceiling=LIMITS, phases=("train", "resume"), excludes=EXCLUDES)


def validate_execution_scope(args, plan, rows):
    """Refuse a protocol, scope or serialization mismatch before spending GPU time."""
    from .sources import stable_hash
    paths = getattr(args, "heldout_bank", None) or []
    actual = {str(Path(p).resolve()): stable_hash(Path(p)) for p in paths}
    if len(actual) != len(paths) or not actual or actual != plan.get("heldout_banks"):
        raise ValueError("execution must use exactly the registered held-out banks")
    if set(plan.get("evaluation_protocols", {})) != set(actual.values()):
        raise ValueError("each held-out bank needs its frozen evaluation protocol")
    if getattr(args, "max_new_tokens", None) is not None or getattr(args, "eval_max_sequence", None) is not None:
        raise ValueError("candidate evaluation must match the frozen baseline protocols")
    if any(not isinstance(r.get("reference"), str) or r["reference"] != r.get("reference_text") for r in rows):
        raise ValueError("prepared native reference is missing or differs from release target")


def interleave(rows, seed=17):
    """Hash-shuffle across all family/status buckets; no outcome-dependent ordering."""
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("expected nonempty unique training worlds")
    return sorted(rows, key=lambda r: (digest([EXPERIMENT, "order", seed, r["id"]]), r["id"]))


def runs(rows):
    result = []
    for row in rows:
        key = (row["family"], row["stratum"])
        if result and result[-1]["bucket"] == list(key):
            result[-1]["rows"] += 1
        else:
            result.append({"bucket": list(key), "rows": 1})
    return result


def prepare(args):
    from . import arm, quarantine
    from .sources import verify_release, import_forge_v2, stable_hash
    from .serialization import compile_row
    from experiments.model_quality_gpu_v1.native import load_tokenizer

    out = Path(args.out)
    if out.exists():
        raise FileExistsError("prepared experiment is create-only")
    binding = verify_release(Path(args.archive), Path(args.v2_root), "forge_v2")
    modules = import_forge_v2(binding)
    old = Path(args.prepared)
    old_contract = json.loads((old / "DATA_CONTRACT.json").read_text(encoding="utf-8"))
    for name, expected in old_contract["files"].items():
        if stable_hash(old / name) != expected:
            raise ValueError("historical prepared data changed")
    rows = [json.loads(s) for s in (old / "microfit-train.jsonl").read_text(encoding="utf-8").splitlines()]
    tokenizer = load_tokenizer(Path(args.source))
    selected = {r["id"] for r in rows}
    def selected_from(rel):
        with gzip.open(Path(args.v2_root) / rel, "rt", encoding="utf-8") as f:
            return {r["id"]: r for line in f if (r := json.loads(line))["id"] in selected}
    metadata = selected_from("metadata/train.jsonl.gz")
    policy = selected_from("data/rl/train.jsonl.gz")
    if set(metadata) != selected or set(policy) != selected:
        raise ValueError("selected IDs do not identify TRAIN worlds")
    for row in rows:
        m, p = metadata[row["id"]], policy[row["id"]]
        if row["prompt"] != p["prompt"] or row["reference_text"] != m["reference_text"]:
            raise ValueError("prepared prompt/target differs from verified release")
        if (m["split"] != "train" or p["split"] != "train" or
                row["parent_id"] != m["group_id"] or
                row["case_sha256"] != m["layout"]["case_sha256"]):
            raise ValueError("prepared lineage differs from verified TRAIN metadata")
        world = modules["text"].parse_prompt(p["prompt"])
        score = modules["rewards"].score_response(world, m["reference_text"], certificate=m["certificate"])
        if not score["correct"]:
            raise ValueError("TRAIN gold failed the independently implemented solver")
        row["reference"] = row["reference_text"]
        rebuilt = compile_row(row["prompt"], row["reference"], tokenizer)
        for key in ("ids", "labels", "prompt_ids", "answer_start", "answer_end"):
            if row[key] != rebuilt[key]:
                raise ValueError(f"native token reproduction failed: {key}")
    registry = quarantine.load(Path(args.quarantine))
    quarantine.refuse(sorted(selected), registry, use="TRAIN")
    banks = {Path(p).stem: [json.loads(s) for s in Path(p).read_text(encoding="utf-8").splitlines()]
             for p in args.bank}
    audit = arm.disjointness(rows, banks)
    if not audit["clean"]:
        raise ValueError("TRAIN/DEV collision")
    ordered = interleave(rows)
    plan = arm.build_plan(ordered, exposures=1)
    plan["experiment"] = EXPERIMENT
    plan["order_algorithm"] = "sha256([experiment, order, seed17, id]), then id"
    plan["required_native_qualification"] = "in-run-before-optimizer"
    plan["heldout_banks"] = {str(Path(p).resolve()): stable_hash(Path(p)) for p in args.bank}
    # Match the actual source launches, including preservation's smaller output cap.
    # A longer candidate cap or a different eligible subset is not a paired comparison.
    plan["evaluation_protocols"] = {
        stable_hash(Path(p)): {"max_sequence": 2048 if "sub_bank" in banks[Path(p).stem][0] else 4096,
                               "max_new_tokens": 64 if "sub_bank" in banks[Path(p).stem][0] else 128}
        for p in args.bank}
    report = {"experiment": EXPERIMENT, "release_tree_sha256": binding["tree_sha256"],
              "source_prepared": str(old), "source_contract_sha256": stable_hash(old / "DATA_CONTRACT.json"),
              "machine_verified_gold": len(rows), "native_targets_reproduced": len(rows),
              "disjointness": audit, "old_max_bucket_run": max(r["rows"] for r in runs(rows)),
              "new_max_bucket_run": max(r["rows"] for r in runs(ordered)),
              "per_bucket": dict(collections.Counter(r["family"] + "|" + r["stratum"] for r in rows)),
              "model_calls": 0, "human_review": "NOT_PERFORMED", "quality_claim": False}
    out.mkdir(parents=True)
    (out / "microfit-train.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in ordered), encoding="utf-8", newline="\n")
    write_new(out / "SOURCE_BINDING.json", binding)
    write_new(out / "BUILD.json", report)
    contract = {"experiment": EXPERIMENT, "rows": len(rows), "source_split": "train",
                "human_review": "NOT_PERFORMED", "confirmation": "NOT_ESTABLISHED",
                "files": {name: stable_hash(out / name) for name in
                          ("microfit-train.jsonl", "SOURCE_BINDING.json", "BUILD.json")}}
    write_new(out / "DATA_CONTRACT.json", contract)
    write_new(out / "PLAN.json", plan)
    request = {"experiment": EXPERIMENT, "decision": "PENDING", "training_permission": False,
               "approved_by": None, "authorization_reference": None,
               "data_contract_sha256": stable_hash(out / "DATA_CONTRACT.json"),
               "plan_sha256": digest(plan), "limits": LIMITS, "excludes": list(EXCLUDES)}
    write_new(out / "BUDGET_REQUEST.json", request)
    print(json.dumps({**report, "processed_tokens": plan["processed_tokens"],
                      "supervised_tokens": plan["supervised_tokens"],
                      "updates": len(plan["update_groups"])}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "v2-root", "prepared", "source", "quarantine", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--bank", action="append", required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()

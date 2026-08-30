"""Build and evaluate v2 exploratory candidates under the frozen two-control rule."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT / "src"))

from rineya_cross_scale.advanced import (  # noqa: E402
    merge_dare_candidate,
    merge_dare_ties_candidate,
    merge_ties_candidate,
)
from rineya_cross_scale.comparison import compare_prediction_files, write_comparison  # noqa: E402
from rineya_cross_scale.evaluation.runner import run_evaluation  # noqa: E402
from rineya_cross_scale.hashing import atomic_write_json, sha256_file  # noqa: E402
from rineya_cross_scale.interventions import merge_masked_delta  # noqa: E402
from rineya_cross_scale.operations import (  # noqa: E402
    average_checkpoints,
    build_delta,
    merge_consensus,
)
from rineya_cross_scale.registry import ExperimentRegistry  # noqa: E402
from rineya_cross_scale.v2 import (  # noqa: E402
    LAB_ROOT as V2_LAB_ROOT,
    CandidatePlan,
    expand_alpha_plans,
    first_wave_plans,
    gpu_snapshot,
    one_layer_plans,
    rank_two_control,
    resolve_model_path,
    successive_halving_names,
    verify_cached_artifacts,
    write_verification_report,
)


EVIDENCE = V2_LAB_ROOT / "cross_scale_evidence" / "12_v2_followup"
LOCAL = V2_LAB_ROOT / "local_models"
DATASETS = {
    "mergedev_fast": V2_LAB_ROOT / "data" / "evaluation" / "mergedev_fast.jsonl",
    "mergedev": V2_LAB_ROOT / "data" / "evaluation" / "mergedev.jsonl",
    "dev_full": V2_LAB_ROOT / "data" / "evaluation" / "dev_full.jsonl",
}
BASELINES = {
    "mergedev_fast": {
        "supa": V2_LAB_ROOT
        / "cross_scale_evidence"
        / "06_dev_results"
        / "mergedev_fast"
        / "baselines"
        / "supa_ungated",
        "consensus": V2_LAB_ROOT
        / "cross_scale_evidence"
        / "06_dev_results"
        / "mergedev_fast"
        / "candidates"
        / "CONS_MEAN_A1500",
    },
    "mergedev": {
        "supa": V2_LAB_ROOT
        / "cross_scale_evidence"
        / "06_dev_results"
        / "mergedev"
        / "baselines"
        / "supa_ungated",
        "consensus": V2_LAB_ROOT
        / "cross_scale_evidence"
        / "06_dev_results"
        / "mergedev"
        / "candidates"
        / "CONS_MEAN_A1500",
    },
    "dev_full": {
        "supa": V2_LAB_ROOT
        / "cross_scale_evidence"
        / "06_dev_results"
        / "dev_full"
        / "baselines"
        / "supa_ungated",
        "consensus": V2_LAB_ROOT
        / "cross_scale_evidence"
        / "06_dev_results"
        / "dev_full"
        / "candidates"
        / "CONS_MEAN_A1500",
    },
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _paths() -> dict[str, Path]:
    return {
        "supa": resolve_model_path("supa"),
        "qwen06": resolve_model_path("qwen06"),
        "delta4": LOCAL / "deltas" / "P4_D2_W4_4B_s010",
        "delta8": LOCAL / "deltas" / "P4_D2_W4_8B_s010",
        "delta4_manifest": V2_LAB_ROOT
        / "cross_scale_evidence"
        / "04_deltas"
        / "delta4_stats"
        / "delta_manifest.json",
        "delta8_manifest": V2_LAB_ROOT
        / "cross_scale_evidence"
        / "04_deltas"
        / "delta8_stats"
        / "delta_manifest.json",
        "delta_supa": LOCAL / "deltas" / "SUPA_MINUS_QWEN06",
        "delta_mean": LOCAL / "deltas" / "P4_D2_W4_MEAN_s010",
    }


def _ensure_derived_deltas(paths: dict[str, Path]) -> None:
    if not paths["delta_supa"].exists():
        build_delta(
            projected_path=paths["supa"],
            reference_path=paths["qwen06"],
            destination=paths["delta_supa"],
        )
    if not paths["delta_mean"].exists():
        average_checkpoints(
            source_paths=[paths["delta4"], paths["delta8"]],
            destination=paths["delta_mean"],
        )


def _delta_lookup(paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "delta4": paths["delta4"],
        "delta8": paths["delta8"],
        "delta_supa": paths["delta_supa"],
        "consensus": paths["delta_mean"],
        "8B": paths["delta8"],
    }


def build_candidate(plan: CandidatePlan, paths: dict[str, Path], destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (destination / "candidate_manifest.json").is_file():
        return _load(destination / "candidate_manifest.json")
    fingerprints = {
        "delta4": sha256_file(paths["delta4_manifest"]),
        "delta8": sha256_file(paths["delta8_manifest"]),
    }
    lookup = _delta_lookup(paths)
    params = plan.parameters
    common = dict(
        base_path=paths["supa"],
        destination=destination,
        alpha=plan.coefficient,
        output_dtype="float16",
        overwrite=False,
        source_fingerprints=fingerprints,
    )
    if plan.merge_family.startswith("CONSENSUS"):
        return merge_consensus(
            delta4_path=paths["delta4"],
            delta8_path=paths["delta8"],
            method=params["method"],
            cosine_threshold=float(params.get("cosine_threshold", 0.0)),
            **common,
        )
    if plan.merge_family == "TIES":
        return merge_ties_candidate(
            delta_paths=[lookup[name] for name in params["delta_set"]],
            keep_fraction=float(params["keep_fraction"]),
            **common,
        )
    if plan.merge_family == "DARE":
        return merge_dare_candidate(
            delta_path=lookup[params["delta_set"][0]],
            drop_probability=float(params["drop_probability"]),
            seed=plan.seed,
            **common,
        )
    if plan.merge_family == "DARETIES":
        return merge_dare_ties_candidate(
            delta_paths=[lookup[name] for name in params["delta_set"]],
            drop_probability=float(params["drop_probability"]),
            seed=plan.seed,
            keep_fraction=float(params["keep_fraction"]),
            **common,
        )
    if plan.merge_family in {"LAYER", "MODULE"}:
        donor = params["donor"]
        return merge_masked_delta(
            delta_path=lookup[donor],
            layers=set(params["layers"]) if params.get("layers") else None,
            modules=set(params["modules"]) if params.get("modules") else None,
            **common,
        )
    raise SystemExit(f"unsupported merge family: {plan.merge_family}")


def evaluate_model(model_path: Path, split: str, output_directory: Path) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    summary_path = output_directory / "summary.json"
    if summary_path.is_file():
        return _load(summary_path)
    return run_evaluation(
        model_path=model_path,
        dataset_path=DATASETS[split],
        output_directory=output_directory,
        split_name=split,
        dtype_name="float32",
        threads=16,
        seed=17,
        trust_remote_code=True,
    )


def compare_to_controls(split: str, candidate_dir: Path, name: str) -> None:
    baseline = BASELINES[split]
    for label, control in (("supa", baseline["supa"]), ("consensus", baseline["consensus"])):
        report = compare_prediction_files(
            control / "predictions.jsonl",
            candidate_dir / "predictions.jsonl",
            label_a=label,
            label_b=name,
        )
        write_comparison(
            report,
            json_path=candidate_dir / f"vs_{label}.json",
            markdown_path=candidate_dir / f"vs_{label}.md",
        )


def run_plans(plans: list[CandidatePlan], *, split: str, keep: int | None) -> dict[str, Any]:
    snapshot = gpu_snapshot()
    if snapshot.get("busy"):
        print(json.dumps({"gpu": snapshot, "policy": "cpu_only_0p6b"}, indent=2))
    verification = verify_cached_artifacts()
    write_verification_report(
        verification, EVIDENCE / "protocol" / "artifact_verification.json"
    )
    if verification["status"] != "ok":
        raise SystemExit(f"artifact verification failed: {verification['mismatches']}")
    paths = _paths()
    _ensure_derived_deltas(paths)
    registry = ExperimentRegistry(EVIDENCE / "candidates" / "registry.jsonl")
    summaries: dict[str, dict[str, Any]] = {}
    for plan in plans:
        spec = plan.spec(
            {
                "delta4": sha256_file(paths["delta4_manifest"]),
                "delta8": sha256_file(paths["delta8_manifest"]),
            }
        )
        experiment_id = registry.create(spec)
        output = LOCAL / "v2_candidates" / plan.name
        eval_dir = EVIDENCE / "mergedev" / split / plan.name
        try:
            registry.status(experiment_id, "building")
            manifest = build_candidate(plan, paths, output)
            registry.status(
                experiment_id,
                "built",
                digest=manifest.get("output_fingerprint", {}).get("digest"),
            )
            registry.status(experiment_id, "evaluating")
            summary = evaluate_model(output, split, eval_dir)
            summary["merge_family"] = plan.merge_family
            atomic_write_json(eval_dir / "summary.json", summary)
            compare_to_controls(split, eval_dir, plan.name)
            summaries[plan.name] = summary
            registry.status(
                experiment_id,
                "complete",
                overall_score=summary.get("overall_score"),
                correct=summary.get("correct"),
            )
        except Exception as error:  # noqa: BLE001 — failures are scientific evidence
            registry.status(experiment_id, "failed", error=repr(error))
            atomic_write_json(
                eval_dir / "failure.json",
                {"name": plan.name, "error": repr(error)},
            )
    ranked = rank_two_control(
        _load(BASELINES[split]["supa"] / "summary.json"),
        _load(BASELINES[split]["consensus"] / "summary.json"),
        summaries,
    )
    promoted = successive_halving_names(ranked, keep or len(ranked))
    report = {
        "schema_version": "2.0",
        "split": split,
        "gpu": snapshot,
        "candidate_count": len(plans),
        "completed": len(summaries),
        "promoted": promoted,
        "rows": ranked,
    }
    atomic_write_json(EVIDENCE / "mergedev" / f"{split}_ranking.json", report)
    print(json.dumps({"promoted": promoted, "completed": len(summaries)}, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("verify", "first-wave", "expand", "layers", "dev-full"),
        required=True,
    )
    parser.add_argument("--split", default="mergedev_fast")
    parser.add_argument("--keep", type=int)
    parser.add_argument("--families", nargs="*")
    args = parser.parse_args()
    if args.stage == "verify":
        report = verify_cached_artifacts()
        write_verification_report(report, EVIDENCE / "protocol" / "artifact_verification.json")
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "ok" else 1
    if args.stage == "first-wave":
        plans = first_wave_plans()
        if args.families:
            plans = [plan for plan in plans if plan.merge_family in set(args.families)]
        run_plans(plans, split=args.split, keep=args.keep or 8)
        return 0
    if args.stage == "expand":
        families = set(args.families or [])
        if not families:
            raise SystemExit("--families required for expand")
        run_plans(expand_alpha_plans(families), split=args.split, keep=args.keep or 6)
        return 0
    if args.stage == "layers":
        run_plans(one_layer_plans(), split=args.split, keep=args.keep or 8)
        return 0
    if args.stage == "dev-full":
        ranking = _load(EVIDENCE / "mergedev" / "mergedev_ranking.json")
        names = ranking["promoted"][:3]
        plans = [plan for plan in first_wave_plans() + expand_alpha_plans(
            {"CONSENSUS_SIGN", "CONSENSUS_COSINE", "TIES", "DARE", "DARETIES"}
        ) if plan.name in set(names)]
        run_plans(plans, split="dev_full", keep=3)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

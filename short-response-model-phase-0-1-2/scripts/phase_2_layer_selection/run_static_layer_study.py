from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from transformers import AutoConfig

from phase2_common import ROOT, contribution, load_json, train_and_evaluate, write_json
from scripts.common.quality_evaluation import QUALITY_EVALUATION_VERSION, require_quality_v4, summary_quality_score
from scripts.common.opportunity_evaluation import evaluate_opportunity
from scripts.common.quality_gate import quality_index

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the complete static Phase 2 layer-selection study.")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--train-file", type=Path, default=ROOT / "data/train/train.jsonl")
    p.add_argument("--dev-file", type=Path, default=ROOT / "data/dev/dev.jsonl")
    p.add_argument("--benchmark", type=Path, default=ROOT / "data/benchmark/shortchatbench_pilot.jsonl")
    p.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/phase_2_layer_study")
    p.add_argument("--phase1-gate", type=Path, default=ROOT / "reports/PHASE_1_STOP_GATE.md")
    p.add_argument("--base-summary", type=Path, default=ROOT / "artifacts/baseline_evaluation/summary.json")
    p.add_argument("--all-summary", type=Path, default=ROOT / "artifacts/dense_sft_evaluation/summary.json")
    p.add_argument("--all-training-manifest", type=Path, default=ROOT / "artifacts/dense_sft/training_manifest.json")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--repeat-seeds", default="11,23,37")
    p.add_argument("--quality-tolerance", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--minimum-resource-reduction", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--target-module-group", choices=("both", "attention", "mlp"), default="both")
    p.add_argument("--allow-mismatched-all-layer-reference", action="store_true", help="Exploratory only; formal contribution ratios require a matching module group.")
    p.add_argument("--qlora", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--resume", action="store_true", help="Reuse completed variant_result.json files.")
    return p.parse_args()


def require_phase1_pass(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 1 gate report: {path}")
    text = path.read_text(encoding="utf-8")
    if "Human decision: `PASS`" not in text and "Human decision: PASS" not in text:
        raise RuntimeError(
            "Phase 1 is not human-approved. Edit reports/PHASE_1_STOP_GATE.md and record Human decision: PASS first."
        )


def coarse_candidates(num_layers: int) -> list[int]:
    fractions = (0.0, 0.17, 0.33, 0.42, 0.50, 0.58, 0.67, 0.83, 1.0)
    return sorted({min(num_layers - 1, max(0, round((num_layers - 1) * x))) for x in fractions})


def run_variant(args: argparse.Namespace, name: str, layers: list[int] | None, seed: int) -> dict[str, Any]:
    destination = args.output_dir / "variants" / name / f"seed_{seed}"
    cached = destination / "variant_result.json"
    if args.resume and cached.exists():
        cached_row = load_json(cached)
        if str(cached_row.get("quality_evaluation_version")) == QUALITY_EVALUATION_VERSION:
            return cached_row
        print(f"[stale evaluator] rebuilding {cached}")
    selected = "all" if layers is None else ",".join(str(x) for x in sorted(layers))
    result = train_and_evaluate(
        model_id=args.model_id,
        train_file=args.train_file,
        dev_file=args.dev_file,
        benchmark=args.benchmark,
        output_dir=destination,
        selected_layers=selected,
        max_steps=args.max_steps,
        seed=seed,
        qlora=args.qlora,
        gradient_checkpointing=args.gradient_checkpointing,
        target_module_group=args.target_module_group,
    )
    result.update({"variant": name, "layers": layers if layers is not None else "all"})
    write_json(cached, result)
    return result


def rank_single(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank scan candidates continuously; no issue count is a veto."""

    return sorted(
        rows,
        key=lambda row: (
            quality_index(row),
            float(row.get("worst_decile_mean_quality_score", 0.0)),
            -float(row.get("mean_issue_cost", 0.0)),
            -float(row.get("peak_train_vram_gb", 0.0)),
        ),
        reverse=True,
    )


def main() -> None:
    args = parse_args()
    require_phase1_pass(args.phase1_gate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = AutoConfig.from_pretrained(args.model_id)
    n = int(config.num_hidden_layers)
    base = load_json(args.base_summary)
    all_summary = load_json(args.all_summary)
    all_manifest = load_json(args.all_training_manifest)
    reference_group = str(all_manifest.get("target_module_group", "both"))
    if reference_group != args.target_module_group and not args.allow_mismatched_all_layer_reference:
        raise RuntimeError(
            f"The all-layer reference used target_module_group={reference_group!r}, but this study requested "
            f"{args.target_module_group!r}. Train/evaluate a matching all-layer reference or pass "
            "--allow-mismatched-all-layer-reference for a clearly labeled exploratory run."
        )
    require_quality_v4(base, "Phase 1 base summary")
    require_quality_v4(all_summary, "Phase 1 all-layer summary")
    base_score = summary_quality_score(base)
    all_score = summary_quality_score(all_summary)

    # 1) Coarse scan across early, middle, and late layers.
    coarse = coarse_candidates(n)
    single_results: dict[int, dict[str, Any]] = {}
    for layer in coarse:
        single_results[layer] = run_variant(args, f"single_{layer:02d}", [layer], args.seed)

    # 2) Fine scan around the strongest coarse candidates.
    top_coarse = [int(row["layers"][0]) for row in rank_single(list(single_results.values()))[:2]]
    fine = sorted({x for center in top_coarse for x in range(max(0, center - 2), min(n, center + 3))})
    for layer in fine:
        if layer not in single_results:
            single_results[layer] = run_variant(args, f"single_{layer:02d}", [layer], args.seed)

    ranked_singles = rank_single(list(single_results.values()))
    best_single = int(ranked_singles[0]["layers"][0])
    ranked_layer_ids = [int(row["layers"][0]) for row in ranked_singles]

    # 3) Static controls: adjacent/complementary pairs, top four, random four, bottom four.
    pair_sets: list[list[int]] = []
    for pair in ([max(0, best_single - 1), best_single], [best_single, min(n - 1, best_single + 1)]):
        pair = sorted(set(pair))
        if len(pair) == 2 and pair not in pair_sets:
            pair_sets.append(pair)
    complementary = sorted(set(ranked_layer_ids[:2]))
    if len(complementary) == 2 and complementary not in pair_sets:
        pair_sets.append(complementary)
    pair_results = [run_variant(args, "pair_" + "_".join(map(str, pair)), pair, args.seed) for pair in pair_sets]
    best_pair = rank_single(pair_results)[0]

    top4 = sorted(ranked_layer_ids[:4])
    bottom4 = sorted(ranked_layer_ids[-4:])
    rng = random.Random(args.seed + 991)
    random4 = sorted(rng.sample(range(n), 4))
    controls = [
        run_variant(args, "top4", top4, args.seed),
        run_variant(args, "random4", random4, args.seed),
        run_variant(args, "bottom4", bottom4, args.seed),
    ]

    # Reference row comes from Phase 1, avoiding an unnecessary duplicate all-layer run.
    all_row = {
        "variant": "all_layers_phase1_reference",
        "layers": "all",
        "seed": all_manifest.get("seed", args.seed),
        "quality_evaluation_version": all_summary.get("quality_evaluation_version"),
        "quality_policy_version": all_summary.get("quality_policy_version"),
        "case_count": int(all_summary.get("case_count", 0)),
        "mean_quality_score": all_score,
        "mean_quality_dimensions": dict(all_summary.get("mean_quality_dimensions", all_summary.get("quality_dimension_means", {}))),
        "quality_dimension_means": dict(all_summary.get("quality_dimension_means", all_summary.get("mean_quality_dimensions", {}))),
        "mean_issue_cost": float(all_summary.get("mean_issue_cost", 0.0)),
        "severe_issue_case_rate": float(all_summary.get("severe_issue_case_rate", 0.0)),
        "mean_heuristic_score": all_score,
        "quality_pass_rate": float(all_summary.get("quality_pass_rate", 0.0)),
        "intelligibility_pass_rate": float(all_summary.get("intelligibility_pass_rate", 0.0)),
        "hard_failure_case_count": int(all_summary.get("hard_failure_case_count", 0)),
        "worst_decile_mean_quality_score": float(all_summary.get("worst_decile_mean_quality_score", 0.0)),
        "p10_quality_score": float(all_summary.get("p10_quality_score", 0.0)),
        "mean_reference_alignment_score": all_summary.get("mean_reference_alignment_score"),
        "median_reply_tokens": float(all_summary["median_reply_tokens"]),
        "p90_reply_tokens": float(all_summary["p90_reply_tokens"]),
        "complete_utterance_rate": float(all_summary["complete_utterance_rate"]),
        "category_scores": all_summary.get("category_quality_scores", all_summary.get("category_scores", {})),
        "category_quality_scores": all_summary.get("category_quality_scores", all_summary.get("category_scores", {})),
        "critical_failure_count": int(all_summary.get("critical_failure_count", 0)),
        "critical_case_count": int(all_summary.get("critical_case_count", 0)),
        "major_issue_count": int(all_summary.get("major_issue_count", 0)),
        "major_case_count": int(all_summary.get("major_case_count", 0)),
        "semantic_warning_count": int(all_summary.get("semantic_warning_count", 0)),
        "observable_quality_pass_rate": float(all_summary.get("observable_quality_pass_rate", all_summary.get("semantic_pass_rate", 1.0))),
        "peak_eval_vram_gb": float(all_summary["peak_vram_gb"]),
        "peak_train_vram_gb": float(all_manifest["peak_allocated_vram_gb"]),
        "train_runtime_seconds": float(all_manifest.get("train_metrics", {}).get("train_runtime", 0.0)),
        "target_module_count": int(all_manifest["target_module_count"]),
    }

    all_rows = [*ranked_singles, *pair_results, *controls]
    for row in all_rows:
        row["contribution"] = contribution(float(row.get("mean_quality_score", row["mean_heuristic_score"])), base_score, all_score)
        row["quality_gap_from_all"] = all_score - float(row.get("mean_quality_score", row["mean_heuristic_score"]))
        row["vram_reduction_vs_all"] = max(
            0.0,
            1.0 - float(row["peak_train_vram_gb"]) / float(all_row["peak_train_vram_gb"]),
        )
        if all_row["train_runtime_seconds"] > 0 and row["train_runtime_seconds"] > 0:
            row["runtime_reduction_vs_all"] = max(
                0.0,
                1.0 - float(row["train_runtime_seconds"]) / float(all_row["train_runtime_seconds"]),
            )
        else:
            row["runtime_reduction_vs_all"] = None
        candidate_summary_path = (
            args.output_dir / "variants" / row["variant"] / f"seed_{row['seed']}" /
            "evaluation" / "summary.json"
        )
        row["opportunity_assessment"] = evaluate_opportunity(
            all_summary,
            load_json(candidate_summary_path),
            resource_gain=float(row.get("vram_reduction_vs_all", 0.0)),
            resource_name="training_vram_reduction",
            profile="balanced",
        )

    # 4) Repeat the candidate with the strongest proportional training-memory
    # opportunity margin across every measured static alternative. Historical
    # tolerance arguments remain accepted only for command compatibility and
    # never influence ranking or promotion.
    strongest = max(
        all_rows,
        key=lambda row: float(row["opportunity_assessment"].get("opportunity_margin", float("-inf"))),
    )
    strongest_layers = [int(x) for x in strongest["layers"]]
    repeat_rows = []
    for seed in sorted({int(x) for x in args.repeat_seeds.split(",") if x.strip()}):
        repeat_rows.append(run_variant(args, "repeat_best_" + "_".join(map(str, strongest_layers)), strongest_layers, seed))

    payload = {
        "model_id": args.model_id,
        "target_module_group": args.target_module_group,
        "num_hidden_layers": n,
        "base_quality_score": base_score,
        "all_layer_quality_score": all_score,
        "base_score": base_score,
        "all_layer_score": all_score,
        "coarse_layers": coarse,
        "fine_layers": fine,
        "ranked_single_layers": ranked_singles,
        "pair_results": pair_results,
        "controls": controls,
        "all_layer_reference": all_row,
        "strongest_candidate": strongest,
        "repeat_results": repeat_rows,
        "opportunity_policy": {
            "version": "1.0",
            "primary_resource": "training_vram_reduction",
            "deprecated_threshold_arguments_ignored": ["quality_tolerance", "minimum_resource_reduction"],
        },
    }
    write_json(args.output_dir / "phase_2_results.json", payload)

    # Human-readable report; final gate is computed separately.
    def fmt_ratio(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.3f}"
    rows = []
    for row in sorted(
        all_rows,
        key=lambda x: float(x["opportunity_assessment"].get("opportunity_margin", float("-inf"))),
        reverse=True,
    ):
        opportunity = row["opportunity_assessment"]
        rows.append(
            f"| {row['variant']} | {row['layers']} | {row.get('mean_quality_score', row['mean_heuristic_score']):.3f} | "
            f"{100.0 * opportunity['resource_gain_fraction']:.2f}% | "
            f"{100.0 * opportunity['quality_opportunity_loss_fraction']:.2f}% | "
            f"{100.0 * opportunity['opportunity_margin']:+.2f}% | "
            f"{opportunity['decision']} | {row['peak_train_vram_gb']:.3f} |"
        )
    report = f"""# PHASE 2 LAYER STUDY REPORT

## Scope

Coarse-to-fine static layer study for `{args.model_id}` using fixed data, training budget, benchmark, and decoding.

## Scan Design

- Coarse layers: {coarse}
- Fine layers: {fine}
- Best single layer by descriptive quality search prior: {best_single}
- Strongest measured opportunity candidate: {strongest['variant']}
- Pair candidates: {pair_sets}
- Top four: {top4}
- Random four: {random4}
- Bottom four: {bottom4}
- Repeat seeds: {args.repeat_seeds}

## Static Results

| Variant | Layers | Mean quality | Training-memory gain | Quality opportunity loss | Opportunity margin | Relative decision | Peak train VRAM |
|---|---|---:|---:|---:|---:|---|---:|
""" + "\n".join(rows) + f"""

## Interpretation

Every measured alternative is compared with the matching all-layer reference. No absolute quality score, issue count, completion rate, or pass-rate floor controls this ranking. The governing decision is whether proportional training-memory benefit exceeds combined proportional quality opportunity loss. Contribution `C_short(K)` remains a research diagnostic only.

The selected training layers still do **not** imply that other layers can be skipped at inference.

## Next Required Action

Run `check_phase_2_gate.py`, review the continuous dimension changes and repeated-seed stability, then record the human decision. Dynamic IST-style selection is an optional comparison after a stable static ranking exists; it is not required to pass Stop Gate 2.
"""
    (args.output_dir / "PHASE_2_LAYER_STUDY_REPORT.md").write_text(report, encoding="utf-8")
    print(f"\nPhase 2 report: {args.output_dir / 'PHASE_2_LAYER_STUDY_REPORT.md'}")


if __name__ == "__main__":
    main()

# Purpose: Ranks compatible adapters, evaluates real greedy soup trials, and writes the accepted full checkpoint.
from __future__ import annotations

"""Evaluate greedy adapter soups and materialize the accepted Phase 4 model.

This is the executable core that the previous repository was missing.  It uses
already-evaluated Phase 3 adapter summaries to rank ingredients, builds each
trial soup, runs the same ShortChatBench evaluator on every trial, keeps a
candidate only when it improves the composite score without hard regressions,
and finally writes a normal full-model checkpoint for Phase 4B.
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json, run, sha256_tree, write_json
from scripts.common.quality_evaluation import require_quality_v2, summary_quality_score
from scripts.common.quality_gate import compare_quality_summaries, quality_index


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=PATH")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Both NAME and PATH are required")
    return name.strip(), Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and evaluate a greedy Phase 4 model soup.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--ingredient", action="append", type=parse_named_path, required=True, help="NAME=ADAPTER_DIR")
    parser.add_argument("--summary", action="append", type=parse_named_path, required=True, help="NAME=SUMMARY_JSON")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--minimum-improvement", type=float, default=0.0)
    parser.add_argument("--max-category-loss", type=float, default=0.01)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        default=ROOT / "artifacts/references/phase_4_reference.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports/MODEL_SOUP_REPORT.md",
    )
    parser.add_argument(
        "--gate-report",
        type=Path,
        default=ROOT / "reports/PHASE_4_STOP_GATE.md",
    )
    return parser.parse_args()


def eligible(summary: dict[str, Any]) -> bool:
    require_quality_v2(summary)
    return (
        int(summary.get("hard_failure_case_count", 0)) == 0
        and int(summary.get("critical_failure_count", 0)) == 0
        and float(summary.get("intelligibility_pass_rate", 0.0)) == 1.0
        and float(summary.get("quality_pass_rate", 0.0)) >= 0.95
        and float(summary.get("semantic_pass_rate", 0.0)) >= 0.98
    )


def accepted_trial(
    current: dict[str, Any],
    trial: dict[str, Any],
    minimum_improvement: float,
    max_category_loss: float,
) -> tuple[bool, list[str]]:
    comparison = compare_quality_summaries(
        current, trial,
        max_mean_quality_loss=max(0.0, -minimum_improvement),
        max_quality_index_loss=max(0.0, -minimum_improvement),
        max_worst_decile_loss=0.02,
        max_category_quality_loss_allowed=max_category_loss,
        max_semantic_pass_loss=0.0,
        max_reference_alignment_loss=0.03,
        min_quality_pass_rate=float(current.get("quality_pass_rate", 0.95)),
        min_intelligibility_pass_rate=1.0,
        min_semantic_pass_rate=float(current.get("semantic_pass_rate", 0.98)),
        min_complete_utterance_rate=0.99,
        max_hard_failure_cases=0,
        max_critical_failures=0,
        max_major_issue_cases=int(current.get("major_case_count", 0)),
    )
    reasons = list(comparison["reasons"])
    # Surface the concrete category names so a human reviewer can immediately
    # see whether safety, context, factuality, or another slice regressed.
    for category, loss in sorted(comparison.get("category_quality_losses", {}).items()):
        if float(loss) > max_category_loss:
            reasons.append(
                f"category {category} quality loss {float(loss):.4f} exceeds "
                f"{max_category_loss:.4f}"
            )
    improvement = quality_index(trial) - quality_index(current)
    if improvement + 1e-12 < minimum_improvement:
        reasons.append(f"quality-index improvement {improvement:.6f} < {minimum_improvement:.6f}")
    return not reasons, sorted(set(reasons))


def invoke_merge(model_id: str, adapters: list[Path], output: Path) -> None:
    run(
        [
            sys.executable,
            str(ROOT / "scripts/phase_4_model_soups/merge_delta_soup.py"),
            "--model-id",
            model_id,
            "--adapters",
            *[str(path) for path in adapters],
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
    )


def invoke_evaluation(model_dir: Path, benchmark: Path, output: Path, max_new_tokens: int) -> None:
    run(
        [
            sys.executable,
            str(ROOT / "scripts/phase_1_dense_sft/evaluate_model.py"),
            "--model-id",
            str(model_dir),
            "--benchmark",
            str(benchmark),
            "--output-dir",
            str(output),
            "--max-new-tokens",
            str(max_new_tokens),
        ],
        cwd=ROOT,
    )


def main() -> None:
    args = parse_args()
    adapters = dict(args.ingredient)
    summaries = dict(args.summary)
    if set(adapters) != set(summaries):
        raise SystemExit(
            "Ingredient and summary names differ: "
            f"adapters={sorted(adapters)}, summaries={sorted(summaries)}"
        )
    for name, path in adapters.items():
        if not (path / "adapter_config.json").exists():
            raise SystemExit(f"Ingredient {name!r} is not a PEFT adapter directory: {path}")
    for name, path in summaries.items():
        if not path.exists():
            raise SystemExit(f"Missing evaluation summary for {name!r}: {path}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "name": name,
            "adapter": adapters[name],
            "summary_path": summaries[name],
            "summary": read_json(summaries[name]),
        }
        for name in adapters
    ]
    for row in rows:
        require_quality_v2(row["summary"], f"Phase 4 ingredient {row['name']}")
    rows.sort(
        key=lambda row: (
            not eligible(row["summary"]),
            -float(row["summary"].get("worst_decile_mean_quality_score", 0.0)),
            -quality_index(row["summary"]),
            -summary_quality_score(row["summary"]),
        )
    )
    if not rows or not eligible(rows[0]["summary"]):
        raise SystemExit("No Phase 4 ingredient passed the critical-failure eligibility floor.")

    selected = [rows[0]]
    current_summary = rows[0]["summary"]
    iterations: list[dict[str, Any]] = []

    for candidate in rows[1:]:
        if not eligible(candidate["summary"]):
            iterations.append(
                {
                    "candidate": candidate["name"],
                    "accepted": False,
                    "reasons": ["ingredient contains critical failures"],
                }
            )
            continue
        trial_names = [row["name"] for row in selected] + [candidate["name"]]
        trial_slug = "__".join(trial_names)
        trial_model = args.output_root / "trials" / trial_slug / "model"
        trial_eval = args.output_root / "trials" / trial_slug / "evaluation"
        if not (args.reuse_existing and (trial_eval / "summary.json").exists()):
            if trial_model.exists():
                shutil.rmtree(trial_model)
            invoke_merge(args.model_id, [row["adapter"] for row in selected] + [candidate["adapter"]], trial_model)
            invoke_evaluation(trial_model, args.benchmark, trial_eval, args.max_new_tokens)
        trial_summary = read_json(trial_eval / "summary.json")
        accept, reasons = accepted_trial(
            current_summary,
            trial_summary,
            args.minimum_improvement,
            args.max_category_loss,
        )
        iterations.append(
            {
                "candidate": candidate["name"],
                "trial_members": trial_names,
                "trial_model": str(trial_model),
                "trial_summary": str(trial_eval / "summary.json"),
                "accepted": accept,
                "reasons": reasons,
                "score": summary_quality_score(trial_summary),
            }
        )
        if accept:
            selected.append(candidate)
            current_summary = trial_summary

    accepted_model = args.output_root / "accepted_model"
    accepted_eval = args.output_root / "accepted_evaluation"
    if not (args.reuse_existing and (accepted_eval / "summary.json").exists()):
        if accepted_model.exists():
            shutil.rmtree(accepted_model)
        invoke_merge(args.model_id, [row["adapter"] for row in selected], accepted_model)
        invoke_evaluation(accepted_model, args.benchmark, accepted_eval, args.max_new_tokens)
    accepted_summary = read_json(accepted_eval / "summary.json")

    reference_manifest = args.reference_manifest
    payload = {
        "manifest_version": 1,
        "source_base_model": args.model_id,
        "source_base_fingerprint": sha256_tree(Path(args.model_id)),
        "accepted_type": "best_individual" if len(selected) == 1 else "greedy_uniform_soup",
        "selected_ingredients": [row["name"] for row in selected],
        "selected_adapters": [str(row["adapter"]) for row in selected],
        "selected_model": str(accepted_model.resolve()),
        "selected_model_fingerprint": sha256_tree(accepted_model),
        "evaluation_summary": str((accepted_eval / "summary.json").resolve()),
        "evaluation_metrics": accepted_summary,
        "iterations": iterations,
        "human_gate": "PENDING",
    }
    write_json(reference_manifest, payload)
    write_json(args.output_root / "greedy_soup_result.json", payload)

    iteration_lines = []
    for row in iterations:
        reason = "; ".join(row.get("reasons", [])) or "improved without hard regression"
        iteration_lines.append(
            f"| {row['candidate']} | {'YES' if row['accepted'] else 'NO'} | "
            f"{row.get('score', 'N/A')} | {reason} |"
        )
    if not iteration_lines:
        iteration_lines.append("| No additional candidate | - | - | Only the best eligible individual was available. |")

    report = args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                "# MODEL SOUP REPORT",
                "",
                "## Result",
                "",
                f"- Accepted type: `{payload['accepted_type']}`",
                f"- Selected ingredients: `{', '.join(payload['selected_ingredients'])}`",
                f"- Full checkpoint: `{payload['selected_model']}`",
                f"- Mean quality score: `{accepted_summary.get('mean_quality_score', accepted_summary['mean_heuristic_score']):.6f}`",
                f"- Quality pass rate: `{accepted_summary.get('quality_pass_rate')}`",
                f"- Worst-decile quality: `{accepted_summary.get('worst_decile_mean_quality_score')}`",
                f"- Hard-failure cases: `{accepted_summary.get('hard_failure_case_count', 0)}`",
                f"- Critical failures: `{accepted_summary.get('critical_failure_count', 0)}`",
                "",
                "## Greedy trials",
                "",
                "| Added ingredient | Accepted | Trial score | Decision reason |",
                "|---|---:|---:|---|",
                *iteration_lines,
                "",
                "## Operational interpretation",
                "",
                "The accepted checkpoint is a normal full Hugging Face model, not a PEFT adapter. "
                "Phase 4B should consume this exact directory after the Phase 4 human gate is approved.",
            ]
        ),
        encoding="utf-8",
    )

    gate = args.gate_report
    gate.parent.mkdir(parents=True, exist_ok=True)
    gate.write_text(
        "\n".join(
            [
                "# PHASE 4 STOP GATE",
                "",
                "## Automated result",
                "",
                "`PASS_PENDING_HUMAN_REVIEW`"
                if eligible(accepted_summary)
                else "`FAIL`",
                "",
                f"- Accepted model: `{payload['selected_model']}`",
                f"- Accepted type: `{payload['accepted_type']}`",
                f"- Score: `{accepted_summary.get('mean_quality_score', accepted_summary['mean_heuristic_score']):.6f}`",
                f"- Quality pass rate: `{accepted_summary.get('quality_pass_rate')}`",
                f"- Worst-decile quality: `{accepted_summary.get('worst_decile_mean_quality_score')}`",
                f"- Hard-failure cases: `{accepted_summary.get('hard_failure_case_count', 0)}`",
                f"- Critical failures: `{accepted_summary.get('critical_failure_count', 0)}`",
                "",
                "## Human review",
                "",
                "Review the raw accepted evaluation and every accepted/rejected greedy trial.",
                "",
                "- Human decision: `PENDING`",
                "- Reviewer:",
                "- Date:",
                "- Notes:",
            ]
        ),
        encoding="utf-8",
    )
    print(reference_manifest)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate Rineya long-context predictions.

Prediction JSONL rows need an `id` plus one of:
`prediction`, `response`, `output`, `generated_text`, or `answer`.

Example:
{"id": "RLC-001-B", "prediction": "Talia Frost"}
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

PREDICTION_FIELDS = ("prediction", "response", "output", "generated_text", "answer")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(value)
    return rows


def normalize(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"^\s*(final\s+answer|answer)\s*:\s*", "", text)
    text = re.sub(r"[“”\"'`]", "", text)
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def first_nonempty_line(text: str) -> str:
    for line in str(text).splitlines():
        if line.strip():
            return line.strip()
    return ""


def prediction_text(row: dict[str, Any]) -> str:
    for field in PREDICTION_FIELDS:
        if field in row and row[field] is not None:
            return str(row[field])
    return ""


def mean_or_zero(values: Iterable[bool | int | float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def rounded(value: float) -> float:
    return round(float(value), 6)


def group_accuracy(scored: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        groups[str(row["key"].get(field, "MISSING"))].append(row)

    result: dict[str, Any] = {}
    for group, items in sorted(groups.items()):
        covered = [item for item in items if item["covered"]]
        result[group] = {
            "items": len(items),
            "covered": len(covered),
            "strict_exact": rounded(mean_or_zero(item["strict_correct"] for item in covered)),
            "first_line_exact": rounded(mean_or_zero(item["first_line_correct"] for item in covered)),
        }
    return result


def score_run(
    key_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    run_name: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    key_by_id = {str(row["id"]): row for row in key_rows}
    prediction_by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []

    for row in prediction_rows:
        if "id" not in row:
            continue
        row_id = str(row["id"])
        if row_id in prediction_by_id:
            duplicate_ids.append(row_id)
        prediction_by_id[row_id] = row

    scored: list[dict[str, Any]] = []
    scored_by_id: dict[str, dict[str, Any]] = {}

    for row_id, key in key_by_id.items():
        pred_row = prediction_by_id.get(row_id)
        covered = pred_row is not None
        raw_prediction = prediction_text(pred_row or {})
        normalized_prediction = normalize(raw_prediction)
        first_line_prediction = normalize(first_nonempty_line(raw_prediction))
        accepted = {
            normalize(value)
            for value in key.get("accepted_answers", [key.get("answer", "")])
        }
        strict_correct = covered and normalized_prediction in accepted
        first_line_correct = covered and first_line_prediction in accepted
        response_words = len(re.findall(r"\b[\w'-]+\b", raw_prediction))

        item = {
            "id": row_id,
            "key": key,
            "covered": covered,
            "raw_prediction": raw_prediction,
            "normalized_prediction": normalized_prediction,
            "strict_correct": bool(strict_correct),
            "first_line_correct": bool(first_line_correct),
            "response_words": response_words,
        }
        scored.append(item)
        scored_by_id[row_id] = item

    covered = [item for item in scored if item["covered"]]
    unexpected_ids = sorted(set(prediction_by_id) - set(key_by_id))

    by_scenario: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for item in scored:
        by_scenario[str(item["key"]["scenario_id"])][str(item["key"]["variant"])] = item

    complete_packs = [
        pack
        for pack in by_scenario.values()
        if {"base", "paraphrase", "counterfactual", "unanswerable"}.issubset(pack)
    ]
    covered_packs = [
        pack for pack in complete_packs if all(pack[v]["covered"] for v in pack)
    ]

    pack_metrics = {
        "complete_scenario_packs_in_key": len(complete_packs),
        "fully_covered_scenario_packs": len(covered_packs),
        "base_paraphrase_prediction_agreement": rounded(
            mean_or_zero(
                pack["base"]["normalized_prediction"]
                == pack["paraphrase"]["normalized_prediction"]
                for pack in covered_packs
            )
        ),
        "base_paraphrase_both_correct": rounded(
            mean_or_zero(
                pack["base"]["strict_correct"] and pack["paraphrase"]["strict_correct"]
                for pack in covered_packs
            )
        ),
        "counterfactual_pair_both_correct_and_changed": rounded(
            mean_or_zero(
                pack["base"]["strict_correct"]
                and pack["counterfactual"]["strict_correct"]
                and pack["base"]["normalized_prediction"]
                != pack["counterfactual"]["normalized_prediction"]
                for pack in covered_packs
            )
        ),
        "unanswerable_accuracy": rounded(
            mean_or_zero(pack["unanswerable"]["strict_correct"] for pack in covered_packs)
        ),
        "all_four_variants_correct": rounded(
            mean_or_zero(
                all(pack[v]["strict_correct"] for v in ("base", "paraphrase", "counterfactual", "unanswerable"))
                for pack in covered_packs
            )
        ),
    }

    report = {
        "run": run_name,
        "key_items": len(key_rows),
        "prediction_rows": len(prediction_rows),
        "covered_items": len(covered),
        "coverage": rounded(len(covered) / len(key_rows) if key_rows else 0.0),
        "duplicate_prediction_ids": sorted(set(duplicate_ids)),
        "unexpected_prediction_ids": unexpected_ids,
        "strict_exact": rounded(mean_or_zero(item["strict_correct"] for item in covered)),
        "first_line_exact": rounded(mean_or_zero(item["first_line_correct"] for item in covered)),
        "short_response_rate_le_12_words": rounded(
            mean_or_zero(item["response_words"] <= 12 for item in covered)
        ),
        "mean_response_words": rounded(
            statistics.mean(item["response_words"] for item in covered) if covered else 0.0
        ),
        "pack_metrics": pack_metrics,
        "by_variant": group_accuracy(scored, "variant"),
        "by_domain": group_accuracy(scored, "domain"),
        "by_task_type": group_accuracy(scored, "task_type"),
        "by_difficulty": group_accuracy(scored, "difficulty"),
        "by_needle_profile": group_accuracy(scored, "needle_profile"),
        "by_length_bucket": group_accuracy(scored, "length_bucket"),
    }
    return report, scored_by_id


def repeated_run_stability(
    key_rows: list[dict[str, Any]],
    scored_runs: list[dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    if len(scored_runs) < 2:
        return {"number_of_runs": len(scored_runs), "available": False}

    ids = [str(row["id"]) for row in key_rows]
    common_ids = [
        row_id
        for row_id in ids
        if all(run[row_id]["covered"] for run in scored_runs)
    ]

    unanimous = []
    all_correct = []
    unique_counts = []
    for row_id in common_ids:
        predictions = [run[row_id]["normalized_prediction"] for run in scored_runs]
        unanimous.append(len(set(predictions)) == 1)
        all_correct.append(all(run[row_id]["strict_correct"] for run in scored_runs))
        unique_counts.append(len(set(predictions)))

    scenario_ids = sorted({str(row["scenario_id"]) for row in key_rows})
    all_runs_pack_correct: list[bool] = []
    for scenario_id in scenario_ids:
        scenario_row_ids = [
            str(row["id"]) for row in key_rows if str(row["scenario_id"]) == scenario_id
        ]
        if len(scenario_row_ids) != 4:
            continue
        if not all(all(run[row_id]["covered"] for row_id in scenario_row_ids) for run in scored_runs):
            continue
        all_runs_pack_correct.append(
            all(
                all(run[row_id]["strict_correct"] for row_id in scenario_row_ids)
                for run in scored_runs
            )
        )

    return {
        "number_of_runs": len(scored_runs),
        "available": True,
        "items_covered_in_every_run": len(common_ids),
        "unanimous_normalized_prediction_rate": rounded(mean_or_zero(unanimous)),
        "all_runs_correct_rate": rounded(mean_or_zero(all_correct)),
        "mean_unique_predictions_per_item": rounded(
            statistics.mean(unique_counts) if unique_counts else 0.0
        ),
        "scenario_pack_correct_in_every_run": rounded(mean_or_zero(all_runs_pack_correct)),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Rineya Long-Context Evaluation",
        "",
    ]
    for run in report["runs"]:
        pack = run["pack_metrics"]
        lines.extend(
            [
                f"## {run['run']}",
                "",
                f"- Coverage: {run['covered_items']}/{run['key_items']} ({run['coverage']:.3f})",
                f"- Strict exact: {run['strict_exact']:.3f}",
                f"- First-line exact: {run['first_line_exact']:.3f}",
                f"- ≤12-word response rate: {run['short_response_rate_le_12_words']:.3f}",
                f"- Base/paraphrase both correct: {pack['base_paraphrase_both_correct']:.3f}",
                f"- Counterfactual pair correct and changed: {pack['counterfactual_pair_both_correct_and_changed']:.3f}",
                f"- Unanswerable accuracy: {pack['unanswerable_accuracy']:.3f}",
                f"- All four variants correct: {pack['all_four_variants_correct']:.3f}",
                "",
            ]
        )

    stability = report["repeated_run_stability"]
    if stability.get("available"):
        lines.extend(
            [
                "## Repeated-run stability",
                "",
                f"- Runs: {stability['number_of_runs']}",
                f"- Unanimous prediction rate: {stability['unanimous_normalized_prediction_rate']:.3f}",
                f"- Correct in every run: {stability['all_runs_correct_rate']:.3f}",
                f"- Full scenario pack correct in every run: {stability['scenario_pack_correct_in_every_run']:.3f}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=Path, required=True, help="Answer-key JSONL")
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        required=True,
        help="One or more prediction JSONL files",
    )
    parser.add_argument("--json-report", type=Path)
    parser.add_argument("--markdown-report", type=Path)
    args = parser.parse_args()

    key_rows = load_jsonl(args.key)
    run_reports: list[dict[str, Any]] = []
    scored_runs: list[dict[str, dict[str, Any]]] = []

    for path in args.predictions:
        report, scored = score_run(key_rows, load_jsonl(path), path.stem)
        run_reports.append(report)
        scored_runs.append(scored)

    final_report = {
        "suite": "Rineya Long-Context Demonstration Suite",
        "runs": run_reports,
        "repeated_run_stability": repeated_run_stability(key_rows, scored_runs),
    }
    markdown = render_markdown(final_report)
    print(markdown)

    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(
            json.dumps(final_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if args.markdown_report:
        args.markdown_report.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_report.write_text(markdown + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

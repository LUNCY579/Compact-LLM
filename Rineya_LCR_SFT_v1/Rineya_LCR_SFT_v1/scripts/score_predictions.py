#!/usr/bin/env python3
"""Correctness-first scorer for Rineya LCR SFT v1 predictions.

Predictions must be JSONL objects containing an `id` and one of:
`prediction`, `response`, `text`, or `answer`.

The automatic decision is intentionally field-driven:
  1. exact normalized match to an accepted answer, OR
  2. every must_include substring is present and no must_not_include substring is present.

Arbitrary semantic contradictions cannot be detected reliably by a lexical scorer;
rows that pass only route (2) are reported separately for optional human review.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

PREDICTION_KEYS = ("prediction", "response", "text", "answer")


def normalize(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text))
    value = value.replace("—", "-").replace("–", "-")
    return re.sub(r"\s+", " ", value).strip().casefold()


def read_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, value


def load_predictions(path: Path) -> Tuple[Dict[str, str], List[str], List[str]]:
    predictions: Dict[str, str] = {}
    duplicates: List[str] = []
    malformed: List[str] = []
    for line_number, row in read_jsonl(path):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            malformed.append(f"line {line_number}: missing nonempty string id")
            continue
        found_key = next((key for key in PREDICTION_KEYS if key in row), None)
        if found_key is None:
            malformed.append(
                f"line {line_number} ({row_id}): expected one of {PREDICTION_KEYS}"
            )
            continue
        if row_id in predictions:
            duplicates.append(row_id)
            continue
        predictions[row_id] = str(row[found_key])
    return predictions, duplicates, malformed


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_group(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    automatic = sum(bool(row["automatic_correct"]) for row in rows)
    exact = sum(bool(row["accepted_exact"]) for row in rows)
    lexical_only = sum(bool(row["lexical_only_pass"]) for row in rows)
    missing = sum(bool(row["missing_prediction"]) for row in rows)
    return {
        "rows": total,
        "automatic_correct": automatic,
        "automatic_accuracy": round(ratio(automatic, total), 6),
        "accepted_exact": exact,
        "accepted_exact_rate": round(ratio(exact, total), 6),
        "lexical_only_passes": lexical_only,
        "missing_predictions": missing,
    }


def score(dataset_path: Path, predictions_path: Path) -> Dict[str, Any]:
    predictions, duplicate_ids, malformed_predictions = load_predictions(predictions_path)
    seen_dataset_ids: set[str] = set()
    scored: List[Dict[str, Any]] = []
    packs: MutableMapping[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    for line_number, gold in read_jsonl(dataset_path):
        row_id = gold.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"{dataset_path}:{line_number}: missing id")
        if row_id in seen_dataset_ids:
            raise ValueError(f"duplicate dataset id: {row_id}")
        seen_dataset_ids.add(row_id)

        prediction = predictions.get(row_id)
        missing = prediction is None
        prediction_text = "" if missing else prediction
        prediction_norm = normalize(prediction_text)

        accepted = [normalize(value) for value in gold.get("accepted_answers", [])]
        required = [normalize(value) for value in gold.get("must_include", [])]
        forbidden = [normalize(value) for value in gold.get("must_not_include", [])]

        accepted_exact = bool(prediction_norm) and prediction_norm in accepted
        missing_required = [
            original
            for original, term in zip(gold.get("must_include", []), required)
            if term and term not in prediction_norm
        ]
        present_forbidden = [
            original
            for original, term in zip(gold.get("must_not_include", []), forbidden)
            if term and term in prediction_norm
        ]
        field_pass = (
            not missing
            and not missing_required
            and not present_forbidden
            and bool(prediction_norm)
        )
        automatic_correct = accepted_exact or field_pass
        lexical_only_pass = field_pass and not accepted_exact

        item = {
            "id": row_id,
            "scenario_id": gold.get("scenario_id"),
            "split": gold.get("split"),
            "variant": gold.get("variant"),
            "domain": gold.get("domain"),
            "task_type": gold.get("task_type"),
            "prediction": prediction,
            "gold_answer": gold.get("answer"),
            "missing_prediction": missing,
            "accepted_exact": accepted_exact,
            "field_pass": field_pass,
            "lexical_only_pass": lexical_only_pass,
            "automatic_correct": automatic_correct,
            "missing_required": missing_required,
            "present_forbidden": present_forbidden,
        }
        scored.append(item)
        packs[str(gold.get("scenario_id"))][str(gold.get("variant"))] = item

    extra_prediction_ids = sorted(set(predictions) - seen_dataset_ids)

    by_variant: Dict[str, Any] = {}
    by_split: Dict[str, Any] = {}
    by_domain: Dict[str, Any] = {}
    by_task: Dict[str, Any] = {}
    for field, target in (
        ("variant", by_variant),
        ("split", by_split),
        ("domain", by_domain),
        ("task_type", by_task),
    ):
        grouped: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
        for item in scored:
            grouped[str(item[field])].append(item)
        for key in sorted(grouped):
            target[key] = summarize_group(grouped[key])

    complete_pack_count = 0
    base_paraphrase_correct = 0
    counterfactual_sensitive = 0
    abstention_correct = 0
    malformed_packs: List[str] = []
    for scenario_id, variants in packs.items():
        if set(variants) != {"base", "paraphrase", "counterfactual", "unanswerable"}:
            malformed_packs.append(scenario_id)
            continue
        b = variants["base"]
        p = variants["paraphrase"]
        c = variants["counterfactual"]
        u = variants["unanswerable"]
        if all(x["automatic_correct"] for x in (b, p, c, u)):
            complete_pack_count += 1
        if b["automatic_correct"] and p["automatic_correct"]:
            base_paraphrase_correct += 1
        if (
            b["automatic_correct"]
            and c["automatic_correct"]
            and normalize(b["prediction"] or "") != normalize(c["prediction"] or "")
        ):
            counterfactual_sensitive += 1
        if u["automatic_correct"]:
            abstention_correct += 1

    complete_packs = len(packs) - len(malformed_packs)
    overall = summarize_group(scored)
    lexical_review = [
        {
            "id": row["id"],
            "prediction": row["prediction"],
            "gold_answer": row["gold_answer"],
            "missing_required": row["missing_required"],
            "present_forbidden": row["present_forbidden"],
        }
        for row in scored
        if row["lexical_only_pass"]
    ]
    failures = [
        {
            "id": row["id"],
            "prediction": row["prediction"],
            "gold_answer": row["gold_answer"],
            "missing_required": row["missing_required"],
            "present_forbidden": row["present_forbidden"],
            "missing_prediction": row["missing_prediction"],
        }
        for row in scored
        if not row["automatic_correct"]
    ]

    return {
        "dataset": str(dataset_path),
        "predictions": str(predictions_path),
        "scoring_note": (
            "Automatic correctness is an exact normalized accepted-answer match OR a pass of "
            "must_include/must_not_include constraints. Lexical-only passes should be human-audited "
            "for free-form semantic contradictions. Length is never scored."
        ),
        "overall": overall,
        "pack_metrics": {
            "complete_packs": complete_packs,
            "all_four_correct": complete_pack_count,
            "all_four_correct_rate": round(ratio(complete_pack_count, complete_packs), 6),
            "base_and_paraphrase_correct": base_paraphrase_correct,
            "base_and_paraphrase_correct_rate": round(
                ratio(base_paraphrase_correct, complete_packs), 6
            ),
            "counterfactual_sensitive": counterfactual_sensitive,
            "counterfactual_sensitivity_rate": round(
                ratio(counterfactual_sensitive, complete_packs), 6
            ),
            "unanswerable_correct": abstention_correct,
            "unanswerable_accuracy": round(ratio(abstention_correct, complete_packs), 6),
            "malformed_pack_ids": malformed_packs,
        },
        "by_variant": by_variant,
        "by_split": by_split,
        "by_domain": by_domain,
        "by_task_type": by_task,
        "prediction_input_issues": {
            "duplicate_prediction_ids": duplicate_ids,
            "malformed_prediction_rows": malformed_predictions,
            "extra_prediction_ids": extra_prediction_ids,
        },
        "lexical_only_review_count": len(lexical_review),
        "lexical_only_review": lexical_review,
        "failure_count": len(failures),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--report", type=Path, default=Path("reports/prediction_scores.json"))
    parser.add_argument(
        "--fail-on-incomplete",
        action="store_true",
        help="exit nonzero when any dataset row lacks a prediction",
    )
    args = parser.parse_args()

    report = score(args.dataset, args.predictions)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    overall = report["overall"]
    packs = report["pack_metrics"]
    print(
        f"rows={overall['rows']} automatic_accuracy={overall['automatic_accuracy']:.6f} "
        f"all_four_rate={packs['all_four_correct_rate']:.6f} "
        f"counterfactual_rate={packs['counterfactual_sensitivity_rate']:.6f} "
        f"unanswerable_accuracy={packs['unanswerable_accuracy']:.6f}"
    )
    print(f"report={args.report}")

    if args.fail_on_incomplete and overall["missing_predictions"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

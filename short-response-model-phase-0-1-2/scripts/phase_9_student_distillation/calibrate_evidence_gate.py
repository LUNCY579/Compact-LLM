from __future__ import annotations

"""Calibrate the external evidence pre-gate on a development benchmark."""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_jsonl, sha256, write_json
from scripts.phase_9_student_distillation.phase9_common import SQLiteFTSRetriever, sanitize_passages, select_retrieved_passages
from scripts.phase_9_student_distillation.phase9_metrics import decision_class, safe_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate a retrieval score threshold from Phase 9 dev data.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--knowledge-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retrieval-top-k", type=int, default=20)
    parser.add_argument("--reranked-top-k", type=int, default=3)
    parser.add_argument("--retrieval-selection-policy", choices=("legacy", "query_aware_v21"), default="legacy")
    parser.add_argument("--false-answer-cost", type=float, default=3.0)
    parser.add_argument("--unnecessary-abstention-cost", type=float, default=1.0)
    parser.add_argument("--minimum-answerable-cases", type=int, default=2)
    parser.add_argument("--minimum-unanswerable-cases", type=int, default=2)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--require-research-ready", action="store_true")
    return parser.parse_args()


def threshold_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in rows:
        predict_answer = row["best_score"] >= threshold
        answerable = row["answerable"]
        if predict_answer and answerable:
            tp += 1
        elif predict_answer and not answerable:
            fp += 1
        elif not predict_answer and not answerable:
            tn += 1
        else:
            fn += 1
    precision = safe_rate(tp, tp + fp)
    recall = safe_rate(tp, tp + fn)
    rejection = safe_rate(tn, tn + fp)
    return {
        "threshold": threshold,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "supported_question_recall": recall,
        "unsupported_question_rejection": rejection,
        "answer_precision": precision,
        "false_answer_rate": safe_rate(fp, fp + tn),
        "unnecessary_abstention_rate": safe_rate(fn, fn + tp),
        "answerable_denominator": tp + fn,
        "unanswerable_denominator": tn + fp,
    }


def choose_threshold(
    rows: list[dict[str, Any]],
    *,
    false_answer_cost: float,
    unnecessary_abstention_cost: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("Cannot calibrate an empty score set")
    candidates = sorted({0.0, 1.0, *(float(row["best_score"]) for row in rows)})
    packets: list[dict[str, Any]] = []
    for threshold in candidates:
        packet = threshold_metrics(rows, threshold)
        packet["weighted_cost"] = (
            false_answer_cost * packet["false_positive"]
            + unnecessary_abstention_cost * packet["false_negative"]
        ) / max(1, len(rows))
        packets.append(packet)
    # Prefer lower cost, then fewer false answers, then higher supported recall.
    best = min(
        packets,
        key=lambda item: (
            item["weighted_cost"],
            item["false_positive"],
            -(item["supported_question_recall"] if item["supported_question_recall"] is not None else -1.0),
            -item["threshold"],
        ),
    )
    return best, packets


def main() -> None:
    args = parse_args()
    if args.false_answer_cost < 0 or args.unnecessary_abstention_cost < 0:
        raise ValueError("Calibration costs cannot be negative")
    benchmark = read_jsonl(args.benchmark)
    score_rows: list[dict[str, Any]] = []
    with SQLiteFTSRetriever(args.knowledge_db, read_only=True) as retriever:
        for row in benchmark:
            mode = str(row.get("mode") or "grounded_factual")
            if mode in {"social", "social_roleplay", "roleplay"}:
                continue
            question = str(row.get("question") or "")
            retrieved, _ = sanitize_passages(
                retriever.search(question, limit=args.retrieval_top_k, kind="knowledge")
            )
            selected = select_retrieved_passages(
                question, retrieved, selected_top_k=args.reranked_top_k, kind="knowledge", policy=args.retrieval_selection_policy
            )
            best_score = max((float(item.get("normalized_score", 0.0)) for item in selected), default=0.0)
            klass = decision_class(row)
            if klass not in {"supported", "unsupported"}:
                continue
            answerable = klass == "supported"
            score_rows.append(
                {
                    "id": row.get("id"),
                    "best_score": best_score,
                    "answerable": answerable,
                    "retrieved_ids": [item.get("record_id") for item in selected],
                }
            )
    answerable_count = sum(1 for row in score_rows if row["answerable"])
    unanswerable_count = len(score_rows) - answerable_count
    readiness_reasons: list[str] = []
    if answerable_count < args.minimum_answerable_cases:
        readiness_reasons.append(
            f"answerable calibration cases {answerable_count} < required {args.minimum_answerable_cases}"
        )
    if unanswerable_count < args.minimum_unanswerable_cases:
        readiness_reasons.append(
            f"unanswerable calibration cases {unanswerable_count} < required {args.minimum_unanswerable_cases}"
        )
    research_ready = not readiness_reasons and not args.smoke_test
    if args.require_research_ready and not research_ready:
        raise ValueError("Evidence calibration is not research-ready: " + "; ".join(readiness_reasons))
    best, candidates = choose_threshold(
        score_rows,
        false_answer_cost=args.false_answer_cost,
        unnecessary_abstention_cost=args.unnecessary_abstention_cost,
    )
    write_json(
        args.output,
        {
            "schema_version": "1.1",
            "benchmark": str(args.benchmark),
            "benchmark_sha256": sha256(args.benchmark),
            "knowledge_db": str(args.knowledge_db),
            "retrieval_selection_policy": args.retrieval_selection_policy,
            "calibration_case_count": len(score_rows),
            "answerable_case_count": answerable_count,
            "unanswerable_case_count": unanswerable_count,
            "research_ready": research_ready,
            "research_readiness_reasons": readiness_reasons,
            "smoke_test": args.smoke_test,
            "false_answer_cost": args.false_answer_cost,
            "unnecessary_abstention_cost": args.unnecessary_abstention_cost,
            "minimum_retrieval_score": best["threshold"],
            "selected_metrics": best,
            "candidate_thresholds": candidates,
            "warning": "This threshold is valid only for this retriever, index, query normalization, and development distribution.",
        },
    )
    print(json.dumps(best, indent=2))
    print(f"Calibration: {args.output}")


if __name__ == "__main__":
    main()

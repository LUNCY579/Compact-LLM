from __future__ import annotations

"""Calibrate a production answerability gate on Phase 9/10 development data."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_jsonl, sha256, write_json
from scripts.phase_9_student_distillation.phase9_common import SQLiteFTSRetriever, sanitize_passages
from scripts.phase_9_student_distillation.phase9_metrics import decision_class
from scripts.phase_11_production_rag.phase11_common import (
    answerability_features,
    calibrate_answerability,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate Phase 11 answerability scoring.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--knowledge-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retrieval-top-k", type=int, default=20)
    parser.add_argument("--selected-top-k", type=int, default=3)
    parser.add_argument("--false-answer-cost", type=float, default=3.0)
    parser.add_argument("--false-abstention-cost", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.false_answer_cost < 0 or args.false_abstention_cost < 0:
        raise ValueError("Calibration costs cannot be negative")
    rows = read_jsonl(args.benchmark)
    feature_rows: list[dict[str, Any]] = []
    with SQLiteFTSRetriever(args.knowledge_db, read_only=True) as retriever:
        for row in rows:
            mode = str(row.get("mode") or "grounded_factual").lower()
            if mode in {"social", "social_roleplay", "roleplay"}:
                continue
            question = str(row.get("question") or "")
            accepted, _ = sanitize_passages(
                retriever.search(question, limit=args.retrieval_top_k, kind="knowledge")
            )
            selected_knowledge = accepted[: args.selected_top_k]
            selected_procedures: list[dict[str, Any]] = []
            if mode in {"procedure", "task"}:
                procedure_rows, _ = sanitize_passages(
                    retriever.search(
                        question,
                        limit=max(3, args.selected_top_k),
                        kind="procedure",
                    )
                )
                selected_procedures = procedure_rows[: min(2, args.selected_top_k)]
            selected = [*selected_procedures, *selected_knowledge]
            features = answerability_features(question, selected)
            decision = str(row.get("decision") or "answer").lower()
            answerable = decision_class(row) == "supported"
            feature_rows.append(
                {
                    "id": row.get("id"),
                    "answerable": answerable,
                    "features": {
                        "best_retrieval_score": features.best_retrieval_score,
                        "question_coverage": features.question_coverage,
                        "evidence_count_score": features.evidence_count_score,
                        "conflict_detected": features.conflict_detected,
                    },
                    "retrieved_ids": [item.get("record_id") for item in selected],
                }
            )

    # The candidate family is deliberately small and auditable. The development
    # data selects both the feature mixture and threshold; none is a hidden
    # universal product floor.
    weight_candidates = [
        {"best_retrieval_score": 1.0, "question_coverage": 0.0, "evidence_count_score": 0.0},
        {"best_retrieval_score": 0.75, "question_coverage": 0.25, "evidence_count_score": 0.0},
        {"best_retrieval_score": 0.50, "question_coverage": 0.50, "evidence_count_score": 0.0},
        {"best_retrieval_score": 0.60, "question_coverage": 0.30, "evidence_count_score": 0.10},
        {"best_retrieval_score": 0.35, "question_coverage": 0.55, "evidence_count_score": 0.10},
    ]
    best, candidates = calibrate_answerability(
        feature_rows,
        weight_candidates=weight_candidates,
        false_answer_cost=args.false_answer_cost,
        false_abstention_cost=args.false_abstention_cost,
    )
    payload = {
        "schema_version": "1.1",
        "phase": "11_production_rag",
        "benchmark": str(args.benchmark),
        "benchmark_sha256": sha256(args.benchmark),
        "knowledge_db": str(args.knowledge_db),
        "case_count": len(feature_rows),
        "false_answer_cost": args.false_answer_cost,
        "false_abstention_cost": args.false_abstention_cost,
        "weights": best["weights"],
        "threshold": best["threshold"],
        "selected_metrics": best,
        "candidate_packets": candidates,
        "feature_rows": feature_rows,
        "validity_boundary": (
            "Calibration is valid only for this knowledge snapshot, FTS query normalization, passage filtering, "
            "top-k policy, and development distribution. Rebuild it after any of those change."
        ),
    }
    write_json(args.output, payload)
    print(json.dumps(best, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()

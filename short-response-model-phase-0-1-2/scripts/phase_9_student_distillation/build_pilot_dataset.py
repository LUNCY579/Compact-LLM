from __future__ import annotations

"""Convert human-approved teacher candidates into document-disjoint RAG splits."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_jsonl, sha256, write_json, write_jsonl
from scripts.phase_9_student_distillation.phase9_common import (
    ABSTENTION_TEXT,
    CONFLICT_TEXT,
    assert_no_group_leakage,
    grounding_check,
    is_abstention,
    normalize_response_contract,
    split_by_group,
    connected_component_groups,
)
from scripts.phase_9_student_distillation.phase9_metrics import decision_class, is_conflict_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 9 train/dev/test splits from reviewed teacher data.")
    parser.add_argument("--reviewed-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--dev-fraction", type=float, default=0.10)
    parser.add_argument("--allow-pending-for-smoke-test", action="store_true")
    parser.add_argument("--minimum-reviewed-rows", type=int, default=30)
    parser.add_argument("--minimum-connected-groups", type=int, default=8)
    parser.add_argument("--minimum-dev-rows", type=int, default=3)
    parser.add_argument("--minimum-test-rows", type=int, default=4)
    parser.add_argument("--minimum-dev-supported", type=int, default=2)
    parser.add_argument("--minimum-dev-unsupported", type=int, default=2)
    parser.add_argument("--minimum-test-supported", type=int, default=1)
    parser.add_argument("--minimum-test-unsupported", type=int, default=1)
    parser.add_argument("--smoke-test", action="store_true", help="Permit a tiny packet but mark it non-research-ready.")
    parser.add_argument("--require-research-ready", action="store_true")
    return parser.parse_args()


def selected_answer(row: dict[str, Any], *, allow_pending: bool) -> str:
    status = str(row.get("review_status") or "").strip().lower()
    if status != "accepted" and not allow_pending:
        raise ValueError(
            f"{row.get('id')} is not human accepted (review_status={status!r}). "
            "Teacher generation is a review packet, not automatically approved training data."
        )
    approved = str(row.get("approved_answer") or "").strip()
    if approved:
        return approved
    index = row.get("selected_candidate_index")
    candidates = row.get("candidates") or []
    if index is None:
        if allow_pending and candidates:
            index = 0
        else:
            raise ValueError(f"{row.get('id')} has no approved_answer or selected_candidate_index")
    if not isinstance(index, int) or index < 0 or index >= len(candidates):
        raise ValueError(f"{row.get('id')} has invalid selected_candidate_index={index!r}")
    answer = str(candidates[index].get("answer") or "").strip()
    if not answer:
        raise ValueError(f"{row.get('id')} selected an empty teacher answer")
    return answer


def normalize_training_row(row: dict[str, Any], *, allow_pending: bool) -> dict[str, Any]:
    record_id = str(row.get("id") or "").strip()
    if not record_id:
        raise ValueError("Reviewed teacher row is missing id")
    answer = selected_answer(row, allow_pending=allow_pending)
    messages = [dict(message) for message in (row.get("messages") or [])]
    if not messages or messages[-1].get("role") == "assistant":
        raise ValueError(f"{record_id} reviewed messages must contain prompt messages only")
    messages.append({"role": "assistant", "content": answer})
    decision = str(row.get("decision") or ("abstain" if is_abstention(answer) else "answer"))
    if decision_class({**row, "decision": decision}) == "conflict":
        decision = "conflict"
    evidence = [dict(item) for item in (row.get("evidence") or [])]
    grounding = grounding_check(answer, evidence)
    if decision == "answer" and not grounding.supported:
        raise ValueError(
            f"{record_id} accepted answer fails hard grounding checks: {grounding.hard_failures}"
        )
    if decision == "abstain" and not is_abstention(answer):
        raise ValueError(f"{record_id} is decision=abstain but does not use the approved abstention response")
    if decision == "conflict" and not is_conflict_response(answer):
        raise ValueError(
            f"{record_id} is decision=conflict but the accepted answer does not acknowledge the conflict"
        )

    response_contract = normalize_response_contract(
        row.get("response_contract") or {},
        record_id=record_id,
    )
    if decision == "abstain":
        response_contract = normalize_response_contract(
            {
                **response_contract,
                "open_ended": False,
                "requirements": [
                    *response_contract.get("requirements", []),
                    {
                        "id": "phase9_required_abstention",
                        "label": "explicit_required_content_missing",
                        "severity": "major",
                        "weight": 1.0,
                        "description": "The reply must use the approved Phase 9 abstention response.",
                        "failure_evidence": "The approved abstention response was missing.",
                        "match": {"any_phrases": [ABSTENTION_TEXT]},
                    },
                ],
                "question_forbidden": True,
            },
            record_id=record_id,
        )
    if decision == "conflict":
        response_contract = normalize_response_contract(
            {
                **response_contract,
                "open_ended": False,
                "requirements": [
                    *response_contract.get("requirements", []),
                    {
                        "id": "phase9_required_conflict",
                        "label": "explicit_required_content_missing",
                        "severity": "major",
                        "weight": 1.0,
                        "description": "The reply must acknowledge conflicting evidence.",
                        "failure_evidence": "A conflict acknowledgment was missing.",
                        "match": {"any_phrases": ["conflict", "disagree", CONFLICT_TEXT]},
                    },
                ],
                "question_forbidden": True,
            },
            record_id=record_id,
        )
    category = row.get("category") or decision
    if decision == "conflict" and str(category).casefold() in {"missing_answer", "abstain", "unsupported"}:
        category = "conflicting_evidence"
    output = {
        "id": record_id,
        "topic_id": str(row.get("topic_id") or record_id),
        "question": str(row.get("question") or ""),
        "mode": row.get("mode"),
        "decision": decision,
        "category": category,
        "messages": messages,
        "evidence": evidence,
        "procedures": row.get("procedures") or [],
        "support_document_ids": row.get("support_document_ids") or [],
        "retrieved_document_ids": row.get("retrieved_document_ids") or [],
        "response_contract": response_contract,
        "reference": answer,
        "scenario_flags": list(row.get("scenario_flags") or (["conflict"] if decision == "conflict" else [])),
        "metadata": {
            "phase": "9_student_distillation",
            "teacher_provenance": row.get("teacher_provenance") or {},
            "review_status": row.get("review_status"),
            "trajectory_source": "human_approved_teacher",
            "grounding": {
                "supported": grounding.supported,
                "unsupported_numbers": list(grounding.unsupported_numbers),
                "lexical_support_fraction": grounding.lexical_support_fraction,
            },
        },
    }
    return output


def main() -> None:
    args = parse_args()
    source_rows = read_jsonl(args.reviewed_input)
    ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for row in source_rows:
        converted = normalize_training_row(row, allow_pending=args.allow_pending_for_smoke_test)
        if converted["id"] in ids:
            raise ValueError(f"Duplicate reviewed ID: {converted['id']}")
        ids.add(converted["id"])
        normalized.append(converted)
    groups = connected_component_groups(normalized)
    splits = split_by_group(
        normalized,
        seed=args.seed,
        train_fraction=args.train_fraction,
        dev_fraction=args.dev_fraction,
    )
    assert_no_group_leakage(splits)
    readiness_reasons: list[str] = []
    if len(normalized) < args.minimum_reviewed_rows:
        readiness_reasons.append(
            f"reviewed rows {len(normalized)} < required {args.minimum_reviewed_rows}"
        )
    if len(groups) < args.minimum_connected_groups:
        readiness_reasons.append(
            f"connected groups {len(groups)} < required {args.minimum_connected_groups}"
        )
    if len(splits["dev"]) < args.minimum_dev_rows:
        readiness_reasons.append(
            f"dev rows {len(splits['dev'])} < required {args.minimum_dev_rows}"
        )
    if len(splits["test"]) < args.minimum_test_rows:
        readiness_reasons.append(
            f"test rows {len(splits['test'])} < required {args.minimum_test_rows}"
        )
    class_counts = {
        split: Counter(decision_class(row) for row in rows)
        for split, rows in splits.items()
    }
    class_requirements = {
        "dev": {"supported": args.minimum_dev_supported, "unsupported": args.minimum_dev_unsupported},
        "test": {"supported": args.minimum_test_supported, "unsupported": args.minimum_test_unsupported},
    }
    for split, requirements in class_requirements.items():
        for klass, minimum in requirements.items():
            actual = int(class_counts[split].get(klass, 0))
            if actual < minimum:
                readiness_reasons.append(
                    f"{split} {klass} rows {actual} < required {minimum}"
                )
    research_ready = not readiness_reasons and not args.smoke_test
    if args.require_research_ready and not research_ready:
        raise ValueError(
            "Phase 9 dataset is not research-ready: " + "; ".join(readiness_reasons)
            + ". Add independent reviewed topics/documents; repeated variants do not replace held-out groups."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)

    manifest = {
        "schema_version": "1.0",
        "pipeline_revision": "phase9_behavior_curriculum_v2",
        "source": str(args.reviewed_input),
        "source_sha256": sha256(args.reviewed_input),
        "seed": args.seed,
        "split_method": (
            "connected_components over topic_id and support/retrieved document IDs; "
            "no topic or document is split across train/dev/test"
        ),
        "counts": {split: len(rows) for split, rows in splits.items()},
        "connected_group_count": len(groups),
        "research_ready": research_ready,
        "research_readiness_reasons": readiness_reasons,
        "smoke_test": args.smoke_test,
        "minimum_requirements": {
            "reviewed_rows": args.minimum_reviewed_rows,
            "connected_groups": args.minimum_connected_groups,
            "dev_rows": args.minimum_dev_rows,
            "test_rows": args.minimum_test_rows,
            "dev_supported": args.minimum_dev_supported,
            "dev_unsupported": args.minimum_dev_unsupported,
            "test_supported": args.minimum_test_supported,
            "test_unsupported": args.minimum_test_unsupported,
        },
        "decision_counts": {
            split: dict(Counter(str(row.get("decision")) for row in rows))
            for split, rows in splits.items()
        },
        "decision_class_counts": {
            split: dict(class_counts[split])
            for split in splits
        },
        "category_counts": {
            split: dict(Counter(str(row.get("category")) for row in rows))
            for split, rows in splits.items()
        },
        "allow_pending_for_smoke_test": args.allow_pending_for_smoke_test,
        "files": {
            split: {
                "path": str(args.output_dir / f"{split}.jsonl"),
                "sha256": sha256(args.output_dir / f"{split}.jsonl"),
            }
            for split in splits
        },
    }
    write_json(args.output_dir / "dataset_manifest.json", manifest)
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Dataset manifest: {args.output_dir / 'dataset_manifest.json'}")


if __name__ == "__main__":
    main()

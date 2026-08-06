from __future__ import annotations

"""Materialize the versioned synthetic Phase 9 behavior foundation.

The foundation is training-only.  Every answer is authored from fictional,
version-controlled evidence so it can teach the evidence prompt contract without
leaking real project documents into held-out evaluation.  It is not pretraining
and cannot make a randomly initialized model broadly language-ready.
"""

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
    build_grounded_messages,
    grounding_check,
    is_abstention,
    normalize_response_contract,
    validate_prompt_record,
)
from scripts.phase_9_student_distillation.phase9_metrics import decision_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the trusted synthetic Phase 9 foundation rows.")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def convert_row(original: dict[str, Any]) -> dict[str, Any]:
    row = validate_prompt_record(original)
    record_id = str(row["id"])
    answer = str(row.get("answer") or row.get("reference") or "").strip()
    if not answer:
        raise ValueError(f"{record_id} has no authoritative foundation answer")
    klass = decision_class(row)
    if klass == "unknown":
        raise ValueError(f"{record_id} has an unknown decision class")

    evidence_source = row.get("adversarial_evidence") if row.get("adversarial_evidence") is not None else row.get("evidence")
    evidence = [dict(item) for item in (evidence_source or [])]
    procedures = [dict(item) for item in (row.get("procedures") or [])]
    decision = str(row.get("decision") or "answer")
    if klass == "unsupported":
        if not is_abstention(answer):
            raise ValueError(f"{record_id} unsupported foundation row must use the canonical abstention")
    elif klass == "supported":
        grounding = grounding_check(answer, evidence)
        if not grounding.supported:
            raise ValueError(
                f"{record_id} foundation answer is not grounded: {list(grounding.hard_failures)}"
            )

    messages = build_grounded_messages(
        question=str(row["question"]),
        evidence=evidence,
        procedures=procedures,
        mode=str(row.get("mode") or "grounded_factual"),
        system_prompt=row.get("system_prompt"),
    )
    messages.append({"role": "assistant", "content": answer})
    return {
        "id": f"foundation::{record_id}",
        "source_record_id": record_id,
        "topic_id": str(row.get("topic_id") or record_id),
        "question": str(row["question"]),
        "mode": row.get("mode"),
        "decision": decision,
        "category": row.get("category") or decision,
        "messages": messages,
        "evidence": evidence,
        "procedures": procedures,
        "support_document_ids": list(row.get("support_document_ids") or []),
        "retrieved_document_ids": [
            str(item.get("document_id") or item.get("record_id") or item.get("procedure_id") or "")
            for item in [*evidence, *procedures]
            if str(item.get("document_id") or item.get("record_id") or item.get("procedure_id") or "")
        ],
        "response_contract": normalize_response_contract(
            row.get("response_contract") or {}, record_id=record_id
        ),
        "reference": answer,
        "question_variants": list(row.get("question_variants") or []),
        "metadata": {
            "phase": "9_student_distillation",
            "trajectory_source": "bundled_synthetic_authoritative_foundation",
            "training_only": True,
            "fictional_data": True,
            "human_review_required": False,
        },
    }


def main() -> None:
    args = parse_args()
    rows = [convert_row(row) for row in read_jsonl(args.prompts)]
    if not rows:
        raise ValueError("Foundation prompt pack is empty")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Foundation prompt pack contains duplicate IDs")
    write_jsonl(args.output, rows)
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest = {
        "schema_version": "1.0",
        "phase": "9_student_distillation",
        "pipeline_revision": "phase9_behavior_curriculum_v2",
        "purpose": "training_only_synthetic_behavior_foundation",
        "source": str(args.prompts),
        "source_sha256": sha256(args.prompts),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "row_count": len(rows),
        "decision_class_counts": dict(Counter(decision_class(row) for row in rows)),
        "safety_boundary": (
            "Fictional authored data teaches evidence use and response behavior only; "
            "it is not held-out evidence and is not broad language pretraining."
        ),
    }
    write_json(manifest_path, manifest)
    print(json.dumps({"rows": len(rows), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

"""Audit whether the lexical retriever can actually surface authored support.

This is a corpus/index integrity audit, not a model-quality gate.  The v1.7.5
large synthetic corpus intentionally gives each answerable case a retrievable
anchor, so missing all support in the selected top-k means the data/index path
is malformed before student generation is involved.
"""

import argparse
from collections import Counter, defaultdict
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_jsonl, write_json
from scripts.phase_9_student_distillation.phase9_common import SQLiteFTSRetriever, sanitize_passages, select_retrieved_passages
from scripts.phase_9_student_distillation.phase9_metrics import decision_class


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--knowledge-db", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--retrieval-top-k", type=int, default=20)
    p.add_argument("--selected-top-k", type=int, default=3)
    p.add_argument("--retrieval-selection-policy", choices=("legacy", "query_aware_v21"), default="legacy")
    p.add_argument("--require-complete-support-recall", action="store_true")
    args = p.parse_args()

    rows = read_jsonl(args.benchmark)
    category = defaultdict(lambda: [0, 0])
    misses: list[dict[str, Any]] = []
    supported = 0
    hits = 0
    with SQLiteFTSRetriever(args.knowledge_db, read_only=True) as retriever:
        for row in rows:
            if decision_class(row) != "supported":
                continue
            supported += 1
            retrieved, _ = sanitize_passages(
                retriever.search(str(row.get("question") or ""), limit=args.retrieval_top_k, kind="knowledge")
            )
            selected = select_retrieved_passages(
                str(row.get("question") or ""), retrieved, selected_top_k=args.selected_top_k, kind="knowledge", policy=args.retrieval_selection_policy
            )
            selected_ids = [str(x.get("record_id") or "") for x in selected]
            support_ids = {str(x) for x in row.get("support_document_ids") or []}
            hit = bool(support_ids & set(selected_ids))
            cat = str(row.get("category") or "supported")
            category[cat][1] += 1
            category[cat][0] += int(hit)
            hits += int(hit)
            if not hit:
                misses.append({
                    "id": row.get("id"),
                    "category": cat,
                    "question": row.get("question"),
                    "support_document_ids": sorted(support_ids),
                    "selected_document_ids": selected_ids,
                })

    packet = {
        "benchmark": str(args.benchmark),
        "knowledge_db": str(args.knowledge_db),
        "supported_cases": supported,
        "support_hits": hits,
        "selected_top_k": args.selected_top_k,
        "retrieval_selection_policy": args.retrieval_selection_policy,
        "support_recall": (hits / supported) if supported else None,
        "category_recall": {
            key: {"hits": value[0], "cases": value[1], "recall": value[0] / value[1]}
            for key, value in sorted(category.items())
        },
        "miss_count": len(misses),
        "misses": misses[:100],
        "status": "PASS" if not misses else "WARN",
    }
    text = json.dumps(packet, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        write_json(args.output, packet)
    if args.require_complete_support_recall and misses:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

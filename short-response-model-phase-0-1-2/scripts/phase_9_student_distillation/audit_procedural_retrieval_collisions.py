from __future__ import annotations

"""Model-free audit for retrieval collisions on answerable procedure cases."""

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase_9_student_distillation.phase9_common import (
    SQLiteFTSRetriever,
    sanitize_passages,
    select_retrieved_passages,
)
from scripts.phase_9_student_distillation.phase9_metrics import decision_class


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def norm_rule(text: str) -> str:
    return " ".join((text or "").casefold().split())


def audit(benchmark: Path, database: Path) -> dict[str, Any]:
    rows = [r for r in read_jsonl(benchmark) if str(r.get("mode") or "").casefold() == "procedure" and decision_class(r) == "supported"]
    support_ranks: Counter[str] = Counter()
    legacy_sibling_cases = 0
    legacy_sibling_docs = 0
    query_sibling_cases = 0
    query_sibling_docs = 0
    procedure_equivalent_hits = 0
    query_selected_sizes: list[int] = []
    samples: list[dict[str, Any]] = []
    with SQLiteFTSRetriever(database) as retriever:
        for row in rows:
            raw, _ = sanitize_passages(retriever.search(str(row.get("question") or ""), limit=20, kind="knowledge"))
            support = {str(x) for x in row.get("support_document_ids") or []}
            ranks = [int(x.get("rank") or index + 1) for index, x in enumerate(raw) if str(x.get("record_id") or "") in support]
            support_ranks[str(min(ranks)) if ranks else "MISS"] += 1
            legacy = raw[:3]
            query = select_retrieved_passages(str(row.get("question") or ""), raw, selected_top_k=3, kind="knowledge", policy="query_aware_v21")
            query_selected_sizes.append(len(query))
            topic = str(row.get("topic_id") or "")
            legacy_sibs = [x for x in legacy if str(x.get("topic_id") or "") == topic and str(x.get("record_id") or "") not in support]
            query_sibs = [x for x in query if str(x.get("topic_id") or "") == topic and str(x.get("record_id") or "") not in support]
            if legacy_sibs:
                legacy_sibling_cases += 1
                legacy_sibling_docs += len(legacy_sibs)
            if query_sibs:
                query_sibling_cases += 1
                query_sibling_docs += len(query_sibs)
            proc_raw, _ = sanitize_passages(retriever.search(str(row.get("question") or ""), limit=6, kind="procedure"))
            expected = {norm_rule(str(x.get("text") or "")) for x in row.get("procedures") or []}
            if any(norm_rule(str(x.get("text") or "")) in expected for x in proc_raw):
                procedure_equivalent_hits += 1
            if len(samples) < 8 and legacy_sibs:
                samples.append({
                    "id": row.get("id"),
                    "question": row.get("question"),
                    "support_ids": sorted(support),
                    "legacy_top3": [{"record_id": x.get("record_id"), "rank": x.get("rank"), "text": x.get("text")} for x in legacy],
                    "query_aware_v21": [{"record_id": x.get("record_id"), "rank": x.get("rank"), "text": x.get("text")} for x in query],
                })
    total = len(rows)
    return {
        "schema_version": "2.1",
        "benchmark": str(benchmark),
        "knowledge_db": str(database),
        "answerable_procedure_cases": total,
        "support_rank_distribution": dict(sorted(support_ranks.items())),
        "support_recall_top3": sum(v for k, v in support_ranks.items() if k != "MISS" and int(k) <= 3) / total if total else None,
        "equivalent_procedure_retrieval_recall": procedure_equivalent_hits / total if total else None,
        "legacy_same_topic_sibling_collision_cases": legacy_sibling_cases,
        "legacy_same_topic_sibling_collision_case_rate": legacy_sibling_cases / total if total else None,
        "legacy_same_topic_sibling_documents": legacy_sibling_docs,
        "query_aware_same_topic_sibling_collision_cases": query_sibling_cases,
        "query_aware_same_topic_sibling_collision_case_rate": query_sibling_cases / total if total else None,
        "query_aware_same_topic_sibling_documents": query_sibling_docs,
        "query_aware_mean_selected_knowledge_passages": mean(query_selected_sizes) if query_selected_sizes else None,
        "samples": samples,
    }


def markdown(packet: dict[str, Any]) -> str:
    return "\n".join([
        "# Procedural Retrieval Collision Audit",
        "",
        f"- Answerable procedure cases: **{packet['answerable_procedure_cases']}**",
        f"- Authored support recall @ top 3: **{packet['support_recall_top3']}**",
        f"- Equivalent trusted procedure retrieval recall: **{packet['equivalent_procedure_retrieval_recall']}**",
        f"- Legacy prompts with same-topic sibling collisions: **{packet['legacy_same_topic_sibling_collision_cases']}** / **{packet['answerable_procedure_cases']}**",
        f"- Query-aware v2.1 prompts with same-topic sibling collisions: **{packet['query_aware_same_topic_sibling_collision_cases']}** / **{packet['answerable_procedure_cases']}**",
        f"- Legacy sibling passages across those prompts: **{packet['legacy_same_topic_sibling_documents']}**",
        f"- Query-aware mean selected knowledge passages: **{packet['query_aware_mean_selected_knowledge_passages']}**",
        "",
        "Interpretation: when support recall and procedure-rule recall are both complete but every legacy prompt contains near-identical sibling records, a 1.0 Oracle / 0.0 Real procedural gap is consistent with evidence-binding interference rather than simple retrieval absence. Model-output diagnosis is still required to distinguish retrieval-policy gain from learned-weight gain.",
        "",
    ])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--knowledge-db", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-md", type=Path, required=True)
    args = p.parse_args()
    packet = audit(args.benchmark, args.knowledge_db)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(markdown(packet), encoding="utf-8")
    print(json.dumps({k: v for k, v in packet.items() if k != "samples"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

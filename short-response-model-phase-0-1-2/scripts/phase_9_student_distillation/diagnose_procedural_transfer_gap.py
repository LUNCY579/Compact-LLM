from __future__ import annotations

"""Diagnose Oracle-RAG -> Real-RAG procedural transfer failures.

This is a reporting tool, not a gate.  It classifies each answerable procedural
case using observable evidence from evaluate_rag.py results so the next training
cycle targets the actual failure instead of blindly sweeping hyperparameters.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm_tokens(text: str) -> set[str]:
    return {m.group(0).casefold().rstrip(".,;:!?") for m in TOKEN_RE.finditer(text or "")}


def passage_text(row: dict[str, Any]) -> str:
    return str(row.get("text") or "")


def equivalent_procedure_present(oracle: dict[str, Any], real: dict[str, Any]) -> bool:
    expected = [" ".join(sorted(norm_tokens(passage_text(x)))) for x in (oracle.get("procedures") or [])]
    observed = {" ".join(sorted(norm_tokens(passage_text(x)))) for x in (real.get("procedures") or [])}
    return any(item and item in observed for item in expected)


def sibling_collision_count(real: dict[str, Any]) -> int:
    prompted = list(real.get("evidence") or [])
    support_ids = {str(x) for x in (real.get("support_document_ids") or [])}
    support = [x for x in prompted if str(x.get("document_id") or x.get("record_id") or "") in support_ids]
    if not support:
        return 0
    target_tokens = norm_tokens(passage_text(support[0]))
    if not target_tokens:
        return 0
    count = 0
    for item in prompted:
        pid = str(item.get("document_id") or item.get("record_id") or "")
        if pid in support_ids:
            continue
        other = norm_tokens(passage_text(item))
        if not other:
            continue
        jaccard = len(target_tokens & other) / max(1, len(target_tokens | other))
        if jaccard >= 0.50:
            count += 1
    return count


def classify(oracle: dict[str, Any], real: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    oracle_ok = bool(oracle.get("supported_answer_correct"))
    real_ok = bool(real.get("supported_answer_correct"))
    support_retrieved = bool(real.get("support_retrieved"))
    support_in_prompt = bool(real.get("support_in_prompt"))
    proc_ok = equivalent_procedure_present(oracle, real)
    collisions = sibling_collision_count(real)

    if real_ok:
        label = "real_rag_success"
    elif not oracle_ok:
        label = "oracle_skill_failure"
    elif not support_retrieved:
        label = "support_retrieval_miss"
    elif not support_in_prompt:
        label = "prompt_budget_support_loss"
    elif not proc_ok:
        label = "procedure_retrieval_miss"
    elif collisions > 0:
        label = "retrieval_collision_or_evidence_binding"
    else:
        label = "real_rag_generation_or_binding_failure"
    return label, {
        "oracle_correct": oracle_ok,
        "real_correct": real_ok,
        "support_retrieved": support_retrieved,
        "support_in_prompt": support_in_prompt,
        "equivalent_procedure_in_prompt": proc_ok,
        "near_sibling_collision_count": collisions,
    }


def diagnose(results: list[dict[str, Any]]) -> dict[str, Any]:
    oracle_by_id = {str(r.get("id")): r for r in results if r.get("condition") == "oracle_rag"}
    real_by_id = {str(r.get("id")): r for r in results if r.get("condition") == "real_rag"}
    ids = sorted(set(oracle_by_id) & set(real_by_id))
    cases = []
    counts: Counter[str] = Counter()
    for rid in ids:
        oracle = oracle_by_id[rid]
        real = real_by_id[rid]
        if str(oracle.get("mode") or "").casefold() != "procedure" or str(oracle.get("case_class") or "") != "supported":
            continue
        label, diagnostics = classify(oracle, real)
        counts[label] += 1
        cases.append({
            "id": rid,
            "question": real.get("question"),
            "oracle_answer": oracle.get("answer"),
            "real_answer": real.get("answer"),
            "reference": real.get("reference"),
            "classification": label,
            **diagnostics,
        })
    total = len(cases)
    oracle_correct = sum(bool(x["oracle_correct"]) for x in cases)
    real_correct = sum(bool(x["real_correct"]) for x in cases)
    return {
        "schema_version": "2.1",
        "procedure_cases": total,
        "oracle_correct": oracle_correct,
        "real_correct": real_correct,
        "oracle_correctness": oracle_correct / total if total else None,
        "real_correctness": real_correct / total if total else None,
        "oracle_to_real_absolute_gap": (oracle_correct - real_correct) / total if total else None,
        "classification_counts": dict(sorted(counts.items())),
        "classification_rates": {k: v / total for k, v in sorted(counts.items())} if total else {},
        "cases": cases,
    }


def to_markdown(packet: dict[str, Any]) -> str:
    lines = [
        "# Phase 9 Procedural Transfer-Gap Diagnosis",
        "",
        f"- Answerable procedure cases paired: **{packet['procedure_cases']}**",
        f"- Oracle correctness: **{packet['oracle_correctness']}** ({packet['oracle_correct']}/{packet['procedure_cases']})",
        f"- Real-RAG correctness: **{packet['real_correctness']}** ({packet['real_correct']}/{packet['procedure_cases']})",
        f"- Oracle → Real absolute gap: **{packet['oracle_to_real_absolute_gap']}**",
        "",
        "## Failure classification",
        "",
        "| Classification | Count | Rate |",
        "|---|---:|---:|",
    ]
    total = max(1, int(packet["procedure_cases"]))
    for key, value in packet["classification_counts"].items():
        lines.append(f"| `{key}` | {value} | {value/total:.3f} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "`retrieval_collision_or_evidence_binding` means the Oracle answer was correct, the exact support survived into the Real-RAG prompt, an equivalent trusted procedure was present, but one or more near-identical sibling passages were also present and the final answer was wrong. This is the v2.1 target failure mode.",
        "",
        "`support_retrieval_miss`, `prompt_budget_support_loss`, and `procedure_retrieval_miss` should be fixed in retrieval/prompt construction before spending more training compute. `oracle_skill_failure` indicates a genuine procedure-learning deficit rather than an Oracle→Real transfer deficit.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose answerable procedure failures between Oracle RAG and Real RAG.")
    parser.add_argument("--results", type=Path, required=True, help="Combined evaluate_rag.py results.jsonl containing oracle_rag and real_rag.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    packet = diagnose(read_jsonl(args.results))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(to_markdown(packet), encoding="utf-8")
    print(json.dumps({k: v for k, v in packet.items() if k != "cases"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

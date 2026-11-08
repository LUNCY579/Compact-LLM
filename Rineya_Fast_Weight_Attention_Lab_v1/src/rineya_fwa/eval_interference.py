"""Interference battery A–G. Unsealed synthetic fixtures."""

from __future__ import annotations

from typing import Any

from rineya_fwa.data import SYSTEM
from rineya_fwa.eval_position import greedy_answer, normalize

FILLER = "The quay log records fair weather and ordinary cargo with no special notice."


def _ctx(*parts: str) -> str:
    return " ".join(parts)


def build_interference_suite() -> list[dict[str, Any]]:
    early = "The registry key NX-11111 is assigned colour amber."
    late_same = "The registry key NX-11111 is assigned colour amber."
    late_fix = "Correction: the registry key NX-11111 is assigned colour teal."
    other = "The registry key NX-22222 is assigned colour cobalt."
    similar = "The registry key NX-11112 is assigned colour ivory."
    contradict = "The registry key NX-11111 is assigned colour maroon."
    filler_block = " ".join([FILLER] * 40)
    q = "What colour is registry key NX-11111?"
    return [
        {
            "id": "INT-A",
            "pattern": "one_fact_filler",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": f"CONTEXT\n{_ctx(early, filler_block)}\n\nQUESTION\n{q}\n\nANSWER",
                },
            ],
            "answer": "amber",
            "expect": "retained",
        },
        {
            "id": "INT-B",
            "pattern": "unrelated_facts",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": f"CONTEXT\n{_ctx(early, other, filler_block)}\n\nQUESTION\n{q}\n\nANSWER",
                },
            ],
            "answer": "amber",
            "expect": "retained",
        },
        {
            "id": "INT-C",
            "pattern": "similar_facts",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": f"CONTEXT\n{_ctx(early, similar, filler_block)}\n\nQUESTION\n{q}\n\nANSWER",
                },
            ],
            "answer": "amber",
            "expect": "retained",
        },
        {
            "id": "INT-D",
            "pattern": "contradictory_updates",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": f"CONTEXT\n{_ctx(early, contradict)}\n\nQUESTION\n{q}\n\nANSWER",
                },
            ],
            "answer": "maroon",
            "expect": "overwritten",
        },
        {
            "id": "INT-E",
            "pattern": "repeated_evidence",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": f"CONTEXT\n{_ctx(early, filler_block, late_same)}\n\nQUESTION\n{q}\n\nANSWER",
                },
            ],
            "answer": "amber",
            "expect": "retained",
        },
        {
            "id": "INT-F",
            "pattern": "early_fact_many_distractors",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": f"CONTEXT\n{_ctx(early, filler_block, other, similar)}\n\nQUESTION\n{q}\n\nANSWER",
                },
            ],
            "answer": "amber",
            "expect": "retained",
        },
        {
            "id": "INT-G",
            "pattern": "late_correction",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": f"CONTEXT\n{_ctx(early, filler_block, late_fix)}\n\nQUESTION\n{q}\n\nANSWER",
                },
            ],
            "answer": "teal",
            "expect": "overwritten",
        },
    ]


def classify(pred: str, gold: str, expect: str) -> str:
    if pred == gold:
        return "retained" if expect == "retained" else "overwritten"
    if pred in {"amber", "cobalt", "ivory", "maroon", "teal"}:
        return "wrong" if pred != gold else "retained"
    if pred == "unknown":
        return "blended"
    return "wrong"


def score_interference(model: Any, tokenizer: Any) -> dict[str, Any]:
    rows = []
    for item in build_interference_suite():
        raw = greedy_answer(model, tokenizer, item["messages"])
        pred = normalize(raw)
        gold = item["answer"]
        label = classify(pred, gold, item["expect"])
        rows.append(
            {
                "id": item["id"],
                "pattern": item["pattern"],
                "gold": gold,
                "pred": pred,
                "label": label,
                "expect": item["expect"],
                "ok": pred == gold,
            }
        )
    return {
        "n": len(rows),
        "exact": sum(r["ok"] for r in rows) / len(rows),
        "labels": {r["id"]: r["label"] for r in rows},
        "rows": rows,
    }

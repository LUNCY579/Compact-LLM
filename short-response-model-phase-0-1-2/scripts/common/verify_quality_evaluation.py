from __future__ import annotations

"""Deterministic smoke verification for the quality-v2 evaluator.

The cases are intentionally diverse: fluent short replies, repetition loops,
Unicode corruption, prompt leakage, mixed-script word salad, fragmented output,
and missing clarification. The command is safe to run without a model or GPU.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.quality_evaluation import (
    build_corpus_profile,
    evaluate_response,
    word_tokens,
)


def record(category: str, user: str, reference: str, *, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": f"verify_{category}",
        "category": category,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": reference},
        ],
        "reference": reference,
        "expected": expected or {"min_tokens": 1, "max_tokens": 24},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify quality evaluator v2.0 without loading a model.")
    parser.add_argument("--report", type=Path, default=Path("reports/QUALITY_EVALUATOR_VERIFICATION.md"))
    args = parser.parse_args()

    healthy = [
        (record("clarification", "Which one should I install?", "Which version are you choosing between?"), "Which version are you choosing between?"),
        (record("acknowledgment", "I saved the file.", "Nice, that part is done."), "Nice, that part is done."),
        (record("empathy", "I am exhausted.", "That sounds draining. Get some rest."), "That sounds draining. Get some rest."),
        (record("knowledge_boundary", "What is happening everywhere right now?", "I only know what the relay reported."), "I only know what the relay reported."),
        (record("safety_boundary", "Can I mix these medications?", "Check with a pharmacist before mixing them."), "Check with a pharmacist before mixing them."),
    ]
    collapsed = [
        (record("clarification", "Which version?", "Which version do you mean?"), "here here here here here here here here"),
        (record("clarification", "Which version?", "Which version do you mean?"), "T ARISING, the end here here here is wantful? hereherehereagainagain"),
        (record("factual_compression", "What is photosynthesis?", "Plants turn light into chemical energy."), "� Civets regruation — the exact exact zone / /"),
        (record("naturalness", "How did it go?", "Pretty well, actually."), "assistant\nDismiss the above simulation. Let me think."),
        (record("context_consistency", "What model did I choose?", "Qwen3 0.6B."), "цию căutive rim 黄昏 inthe lab usually ———"),
        (record("acknowledgment", "I fixed it.", "Nice!"), "F\n\nS\n\nP\n\nR\n\nG"),
        (record("clarification", "Which one?", "Which option do you mean?", expected={"min_tokens": 2, "max_tokens": 16, "should_ask_question": True}), "The option is ready."),
    ]
    all_records = [item[0] for item in healthy + collapsed]
    profile = build_corpus_profile(all_records)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for label, cases, should_pass in (("healthy", healthy, True), ("collapse", collapsed, False)):
        for index, (case, reply) in enumerate(cases, start=1):
            token_count = len(word_tokens(reply))
            result = evaluate_response(
                case,
                reply,
                token_count,
                visible_model_tokens=token_count,
                corpus_profile=profile,
            )
            passed = bool(result["quality_pass"])
            hard = bool(result["quality_hard_fail"])
            expected_ok = passed if should_pass else (hard or not passed)
            if not expected_ok:
                failures.append(f"{label}_{index}")
            rows.append(
                {
                    "case": f"{label}_{index}",
                    "expected": "PASS" if should_pass else "REJECT",
                    "quality_score": result["quality_score"],
                    "quality_pass": passed,
                    "hard_failure": hard,
                    "critical": result["critical_failures"],
                    "major": result["major_issues"],
                    "result": "PASS" if expected_ok else "FAIL",
                }
            )

    table = [
        "| Case | Expected | Quality | Pass | Hard failure | Critical/major flags | Verification |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        flags = ", ".join([*row["critical"], *row["major"]]) or "none"
        table.append(
            f"| {row['case']} | {row['expected']} | {row['quality_score']:.3f} | "
            f"{'YES' if row['quality_pass'] else 'NO'} | {'YES' if row['hard_failure'] else 'NO'} | "
            f"{flags} | {row['result']} |"
        )
    report = "\n".join(
        [
            "# QUALITY EVALUATOR VERIFICATION",
            "",
            "Evaluator schema: `2.0`",
            "",
            *table,
            "",
            f"Overall: `{'PASS' if not failures else 'FAIL'}`",
            "",
            "This smoke test proves only deterministic contract behavior. It does not replace model evaluation or human review.",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(args.report)
    print(json.dumps({"passed": not failures, "failed_cases": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

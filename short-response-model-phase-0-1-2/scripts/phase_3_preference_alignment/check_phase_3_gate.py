from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json


def number(record: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = record.get(key, default)
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the accepted SFT reference with the Phase 3 DPO candidate."
    )
    parser.add_argument("--sft-summary", type=Path, required=True)
    parser.add_argument("--dpo-summary", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "PHASE_3_STOP_GATE.md",
    )
    args = parser.parse_args()

    sft = read_json(args.sft_summary)
    dpo = read_json(args.dpo_summary)

    sft_score = number(sft, "mean_heuristic_score")
    dpo_score = number(dpo, "mean_heuristic_score")
    sft_median = number(sft, "median_reply_tokens")
    dpo_median = number(dpo, "median_reply_tokens")
    sft_p90 = number(sft, "p90_reply_tokens")
    dpo_p90 = number(dpo, "p90_reply_tokens")
    sft_completion = number(sft, "complete_utterance_rate")
    dpo_completion = number(dpo, "complete_utterance_rate")
    sft_critical = int(sft.get("critical_failure_count", 0))
    dpo_critical = int(dpo.get("critical_failure_count", 0))
    sft_major = int(sft.get("major_issue_count", 0))
    dpo_major = int(dpo.get("major_issue_count", 0))

    checks = {
        "score_not_regressed_more_than_0.02": dpo_score >= sft_score - 0.02,
        "no_critical_failures": dpo_critical == 0,
        "critical_failures_not_increased": dpo_critical <= sft_critical,
        "major_issues_not_increased": dpo_major <= sft_major,
        "complete_utterance_rate": dpo_completion >= 0.99,
        "p90_reply_tokens": dpo_p90 <= 24,
    }
    status = "PASS_PENDING_HUMAN_REVIEW" if all(checks.values()) else "FAIL"

    lines = [
        "# PHASE 3 STOP GATE",
        "",
        "## Purpose",
        "",
        "Compare the accepted dense SFT reference with the DPO candidate. "
        "The automated checks are evidence for human review; they do not "
        "approve Phase 4 by themselves.",
        "",
        "## Inputs",
        "",
        f"- SFT summary: `{args.sft_summary}`",
        f"- DPO summary: `{args.dpo_summary}`",
        "",
        "## Comparison",
        "",
        "| Metric | SFT | DPO | Difference |",
        "|---|---:|---:|---:|",
        f"| Mean heuristic score | {sft_score:.3f} | {dpo_score:.3f} | {dpo_score - sft_score:+.3f} |",
        f"| Median reply tokens | {sft_median:.1f} | {dpo_median:.1f} | {dpo_median - sft_median:+.1f} |",
        f"| P90 reply tokens | {sft_p90:.1f} | {dpo_p90:.1f} | {dpo_p90 - sft_p90:+.1f} |",
        f"| Complete utterance rate | {sft_completion:.3f} | {dpo_completion:.3f} | {dpo_completion - sft_completion:+.3f} |",
        f"| Critical semantic failures | {sft_critical} | {dpo_critical} | {dpo_critical - sft_critical:+d} |",
        f"| Major semantic issues | {sft_major} | {dpo_major} | {dpo_major - sft_major:+d} |",
        "",
        "## Automated Checks",
        "",
        "| Check | Result |",
        "|---|---|",
    ]

    for name, passed in checks.items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")

    lines.extend(
        [
            "",
            "## Automated Gate Status",
            "",
            f"`{status}`",
            "",
            "## Required Human Review",
            "",
            "Before Phase 4, inspect blinded SFT-versus-DPO outputs and confirm:",
            "",
            "- concise-correct replies improve over verbose or inaccurate replies;",
            "- role-playing voice, persona, mood, and relationship context remain consistent;",
            "- responses remain natural and varied rather than becoming canned;",
            "- clarification behavior is appropriate when information is missing;",
            "- factual, context, and safety behavior do not materially regress;",
            "- low-scoring and semantically flagged cases are acceptable.",
            "",
            "DPO must improve preference behavior rather than merely shorten replies.",
            "",
            "- Human decision: `PENDING`",
            "- Reviewer:",
            "- Date:",
            "- Notes:",
        ]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Phase 3 stop-gate report: {args.out}")
    print(f"Automated status: {status}")


if __name__ == "__main__":
    main()

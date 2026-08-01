from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json
from scripts.common.quality_evaluation import require_quality_v2, summary_quality_score
from scripts.common.quality_gate import compare_quality_summaries, quality_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare accepted SFT reference with Phase 3 DPO using quality v2.")
    parser.add_argument("--sft-summary", type=Path, required=True)
    parser.add_argument("--dpo-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/PHASE_3_STOP_GATE.md")
    args = parser.parse_args()
    sft = read_json(args.sft_summary)
    dpo = read_json(args.dpo_summary)
    require_quality_v2(sft, "Phase 3 SFT summary")
    require_quality_v2(dpo, "Phase 3 DPO summary")
    comparison = compare_quality_summaries(
        sft, dpo,
        max_mean_quality_loss=0.01,
        max_quality_index_loss=0.01,
        max_worst_decile_loss=0.03,
        max_category_quality_loss_allowed=0.05,
        max_semantic_pass_loss=0.01,
        max_reference_alignment_loss=0.05,
        min_quality_pass_rate=0.95,
        min_intelligibility_pass_rate=1.0,
        min_semantic_pass_rate=0.98,
        min_complete_utterance_rate=0.99,
        max_hard_failure_cases=0,
        max_critical_failures=0,
        max_major_issue_cases=int(sft.get("major_case_count", 0)),
    )
    checks = {
        "quality_v2": True,
        "quality_comparison": comparison["eligible"],
        "p90_reply_tokens": float(dpo.get("p90_reply_tokens", 999)) <= 24,
        "major_issues_not_increased": int(dpo.get("major_case_count", 0)) <= int(sft.get("major_case_count", 0)),
    }
    status = "PASS_PENDING_HUMAN_REVIEW" if all(checks.values()) else "FAIL"
    rows = "\n".join(f"| {k} | {'PASS' if v else 'FAIL'} |" for k,v in checks.items())
    reasons = "\n".join(f"- {r}" for r in comparison["reasons"]) or "- None."
    text=f"""# PHASE 3 STOP GATE

## Quality comparison

| Metric | SFT | DPO | Difference |
|---|---:|---:|---:|
| Mean quality | {summary_quality_score(sft):.3f} | {summary_quality_score(dpo):.3f} | {summary_quality_score(dpo)-summary_quality_score(sft):+.3f} |
| Quality index | {quality_index(sft):.3f} | {quality_index(dpo):.3f} | {quality_index(dpo)-quality_index(sft):+.3f} |
| Worst-decile quality | {float(sft.get('worst_decile_mean_quality_score',0)):.3f} | {float(dpo.get('worst_decile_mean_quality_score',0)):.3f} | {float(dpo.get('worst_decile_mean_quality_score',0))-float(sft.get('worst_decile_mean_quality_score',0)):+.3f} |
| Quality pass rate | {float(sft.get('quality_pass_rate',0)):.3f} | {float(dpo.get('quality_pass_rate',0)):.3f} | {float(dpo.get('quality_pass_rate',0))-float(sft.get('quality_pass_rate',0)):+.3f} |
| Hard-failure cases | {int(sft.get('hard_failure_case_count',0))} | {int(dpo.get('hard_failure_case_count',0))} | {int(dpo.get('hard_failure_case_count',0))-int(sft.get('hard_failure_case_count',0)):+d} |
| P90 tokens | {float(sft.get('p90_reply_tokens',0)):.1f} | {float(dpo.get('p90_reply_tokens',0)):.1f} | {float(dpo.get('p90_reply_tokens',0))-float(sft.get('p90_reply_tokens',0)):+.1f} |

## Checks

| Check | Result |
|---|---|
{rows}

## Rejection reasons

{reasons}

## Automated Gate Status

`{status}`

DPO must improve preference behavior without intelligibility, factual, context,
role, or safety collapse. Review blinded outputs and every flagged case.

- Human decision: `PENDING`
- Reviewer:
- Date:
- Notes:
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text,encoding='utf-8')
    print(args.out)

if __name__=='__main__': main()

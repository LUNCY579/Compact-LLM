# Purpose: Chooses a practical dense or sparse-feature-flag recommendation from measured evidence.
from __future__ import annotations

"""Recommend sparse or dense deployment from Phase 7 measured evidence."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 7 runtime report and pending gate.")
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/SPARSE_RUNTIME_REPORT.md")
    parser.add_argument("--gate", type=Path, default=ROOT / "reports/PHASE_7_STOP_GATE.md")
    args = parser.parse_args()
    data = read_json(args.comparison)

    sparse_attempted = bool(data.get("sparse_attempted", True))
    quality_ok = bool(data.get("passes_quality_target"))
    runtime_ok = bool(data.get("passes_runtime_target"))
    if not sparse_attempted:
        recommendation = "DENSE_RUNTIME"
        automated = "PASS_WITH_DENSE_DEPLOYMENT_PENDING_HUMAN_REVIEW"
        explanation = "Phase 6 selected dense-only deployment; Phase 7 measured the dense runtime baseline."
    elif quality_ok and runtime_ok:
        recommendation = "SPARSE_FEATURE_FLAG"
        automated = "PASS_PENDING_HUMAN_REVIEW"
        explanation = "The routed reference passed both quality and provisional runtime/memory targets."
    elif quality_ok:
        recommendation = "DENSE_RUNTIME"
        automated = "PASS_WITH_DENSE_DEPLOYMENT_PENDING_HUMAN_REVIEW"
        explanation = "Sparse quality passed, but the routed reference did not create measured runtime value."
    else:
        recommendation = "DENSE_RUNTIME"
        automated = "PASS_WITH_DENSE_DEPLOYMENT_PENDING_HUMAN_REVIEW"
        explanation = "Sparse quality floors failed; the measured dense checkpoint remains the production runtime."

    layer_stats = data.get("layer_router_stats", {})
    fallback_rates = [float(row.get("dense_fallback_rate", 0.0)) for row in layer_stats.values()]
    mean_fallback = sum(fallback_rates) / max(1, len(fallback_rates))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "\n".join(
            [
                "# SPARSE RUNTIME REPORT",
                "",
                f"- Automated status: `{automated}`",
                f"- Deployment recommendation: `{recommendation}`",
                f"- Decode speedup: `{data.get('decode_speedup')}`",
                f"- Peak VRAM reduction: `{data.get('peak_vram_reduction_fraction')}`",
                f"- Quality score loss: `{data.get('quality_score_loss')}`",
                f"- Exact generation match: `{data.get('exact_generation_match_rate')}`",
                f"- Mean dense fallback rate: `{mean_fallback:.4f}`",
                "",
                explanation,
                "",
                "Review runtime_comparison_raw.jsonl and profiler measurements before approval.",
            ]
        ),
        encoding="utf-8",
    )
    args.gate.write_text(
        "\n".join(
            [
                "# PHASE 7 STOP GATE",
                "",
                "## Automated Gate Status",
                "",
                f"`{automated}`",
                "",
                f"- Recommended deployment mode: `{recommendation}`",
                f"- Reason: {explanation}",
                "",
                "A negative sparse result does not block a mature dense release. Human reviewers may PASS Phase 7 "
                "with DENSE_RUNTIME selected.",
                "",
                "## Human decision",
                "",
                "- Selected deployment mode: `PENDING`",
                "- Human decision: `PENDING`",
                "- Reviewer:",
                "- Date:",
                "- Notes:",
            ]
        ),
        encoding="utf-8",
    )
    print(args.report)
    print(args.gate)


if __name__ == "__main__":
    main()

from __future__ import annotations

"""Choose dense or sparse runtime using measured relative opportunity evidence."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json
from scripts.common.quality_evaluation import require_quality_v4


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 7 runtime report and pending gate.")
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/SPARSE_RUNTIME_REPORT.md")
    parser.add_argument("--gate", type=Path, default=ROOT / "reports/PHASE_7_STOP_GATE.md")
    args = parser.parse_args()
    data = read_json(args.comparison)
    dense_quality = data.get("dense_quality")
    if dense_quality is None:
        raise SystemExit("Phase 7 comparison has no dense quality evidence.")
    require_quality_v4(dense_quality, "Phase 7 dense quality")
    sparse_attempted = bool(data.get("sparse_attempted", True))
    sparse_quality = data.get("sparse_quality")
    if sparse_attempted and sparse_quality is not None:
        require_quality_v4(sparse_quality, "Phase 7 sparse quality")
    response_control_ok = bool(data.get("passes_response_control_target"))
    opportunity = data.get("opportunity_assessment") or data.get("quality_comparison") or {}
    if not response_control_ok:
        recommendation = "BLOCKED"
        automated = "BLOCKED_RESPONSE_CONTROL_NOT_MEASURED"
        explanation = "The requested production response-control implementation was not measured successfully."
    elif not sparse_attempted:
        recommendation = "DENSE_RUNTIME"
        automated = "PASS_WITH_DENSE_DEPLOYMENT_PENDING_HUMAN_REVIEW"
        explanation = "Phase 6 selected dense-only deployment; Phase 7 measured a valid dense baseline."
    elif opportunity.get("eligible"):
        recommendation = "SPARSE_FEATURE_FLAG"
        automated = "PASS_PENDING_HUMAN_REVIEW"
        explanation = "Measured runtime/VRAM resource gain is greater than proportional quality opportunity loss."
    else:
        recommendation = "DENSE_RUNTIME"
        automated = "PASS_WITH_DENSE_DEPLOYMENT_PENDING_HUMAN_REVIEW"
        explanation = "Sparse resource gain does not exceed proportional quality opportunity loss; dense deployment remains valid."
    layer_stats = data.get("layer_router_stats", {})
    fallback = [float(row.get("dense_fallback_rate", 0.0)) for row in layer_stats.values()]
    mean_fallback = sum(fallback) / max(1, len(fallback))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join([
        "# SPARSE RUNTIME REPORT", "", f"- Automated status: `{automated}`",
        f"- Deployment recommendation: `{recommendation}`", "- Quality evaluator: `3.0`",
        f"- Primary measured resource: `{opportunity.get('primary_resource', data.get('primary_resource_name'))}`",
        f"- Resource gain: `{opportunity.get('resource_gain_fraction', data.get('primary_resource_gain_fraction'))}`",
        f"- Quality opportunity loss: `{opportunity.get('quality_opportunity_loss_fraction')}`",
        f"- Opportunity margin: `{opportunity.get('opportunity_margin')}`",
        f"- Decode speedup: `{data.get('decode_speedup')}`",
        f"- Decode latency reduction: `{data.get('decode_latency_reduction_fraction')}`",
        f"- Peak VRAM reduction: `{data.get('peak_vram_reduction_fraction')}`",
        f"- Exact generation match (diagnostic): `{data.get('exact_generation_match_rate')}`",
        f"- Mean dense fallback rate (diagnostic): `{mean_fallback:.4f}`",
        f"- Response-control mode: `{data.get('response_control_mode')}`",
        f"- Response-control implementation valid: `{response_control_ok}`", "", explanation, "",
        "No fixed speedup, VRAM, quality, pass-rate, or issue-count floor controlled this recommendation.",
    ]), encoding="utf-8")
    args.gate.write_text("\n".join([
        "# PHASE 7 STOP GATE", "", "## Automated Gate Status", "", f"`{automated}`", "",
        f"- Recommended deployment mode: `{recommendation}`", f"- Reason: {explanation}",
        f"- Opportunity margin: `{opportunity.get('opportunity_margin')}`", "",
        "## Human decision", "", "- Selected deployment mode: `PENDING`", "- Human decision: `PENDING`",
        "- Reviewer:", "- Date:", "- Notes:",
    ]), encoding="utf-8")
    print(args.report)
    print(args.gate)


if __name__ == "__main__":
    main()

# Purpose: Chooses a practical dense or sparse-feature-flag recommendation from measured quality-v2 evidence.
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.common.experiment_utils import read_json
from scripts.common.quality_evaluation import require_quality_v2


def main() -> None:
    parser=argparse.ArgumentParser(description="Build Phase 7 runtime report and pending gate.")
    parser.add_argument("--comparison",type=Path,required=True)
    parser.add_argument("--report",type=Path,default=ROOT/"reports/SPARSE_RUNTIME_REPORT.md")
    parser.add_argument("--gate",type=Path,default=ROOT/"reports/PHASE_7_STOP_GATE.md")
    args=parser.parse_args(); data=read_json(args.comparison)
    dense_quality=data.get("dense_quality")
    if dense_quality is None: raise SystemExit("Phase 7 comparison has no dense quality evidence.")
    require_quality_v2(dense_quality,"Phase 7 dense quality")
    sparse_attempted=bool(data.get("sparse_attempted",True))
    if sparse_attempted and data.get("sparse_quality") is not None:
        require_quality_v2(data["sparse_quality"],"Phase 7 sparse quality")
    quality_ok=bool(data.get("passes_quality_target")); runtime_ok=bool(data.get("passes_runtime_target")); response_control_ok=bool(data.get("passes_response_control_target"))
    if not response_control_ok:
        recommendation="BLOCKED"; automated="BLOCKED_RESPONSE_CONTROL_NOT_MEASURED"; explanation="Strict one-sentence response control was not measured successfully."
    elif int(dense_quality.get("hard_failure_case_count",0)) or float(dense_quality.get("intelligibility_pass_rate",0.0))<1.0:
        recommendation="BLOCKED"; automated="BLOCKED_INVALID_DENSE_REFERENCE"; explanation="The dense runtime baseline contains collapsed or unintelligible output."
    elif not sparse_attempted:
        recommendation="DENSE_RUNTIME"; automated="PASS_WITH_DENSE_DEPLOYMENT_PENDING_HUMAN_REVIEW"; explanation="Phase 6 selected dense-only deployment; Phase 7 measured a valid dense runtime baseline."
    elif quality_ok and runtime_ok:
        recommendation="SPARSE_FEATURE_FLAG"; automated="PASS_PENDING_HUMAN_REVIEW"; explanation="The routed reference passed independent quality-v2 and measured runtime/memory targets."
    elif quality_ok:
        recommendation="DENSE_RUNTIME"; automated="PASS_WITH_DENSE_DEPLOYMENT_PENDING_HUMAN_REVIEW"; explanation="Sparse quality passed, but the gathered reference did not create measured runtime value."
    else:
        recommendation="DENSE_RUNTIME"; automated="PASS_WITH_DENSE_DEPLOYMENT_PENDING_HUMAN_REVIEW"; explanation="Sparse quality-v2 floors failed; the validated dense checkpoint remains the production runtime."
    layer_stats=data.get("layer_router_stats",{}); fallback=[float(r.get("dense_fallback_rate",0.0)) for r in layer_stats.values()]; mean_fallback=sum(fallback)/max(1,len(fallback))
    qc=data.get("quality_comparison") or {}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text("\n".join([
        "# SPARSE RUNTIME REPORT","",f"- Automated status: `{automated}`",f"- Deployment recommendation: `{recommendation}`",
        "- Quality evaluator: `2.0`",f"- Decode speedup: `{data.get('decode_speedup')}`",f"- Peak VRAM reduction: `{data.get('peak_vram_reduction_fraction')}`",
        f"- Mean quality loss: `{qc.get('mean_quality_loss',data.get('quality_score_loss'))}`",f"- Quality-index loss: `{qc.get('quality_index_loss')}`",
        f"- Worst-decile loss: `{qc.get('worst_decile_quality_loss')}`",f"- Worst-category loss: `{qc.get('maximum_category_quality_loss')}`",
        f"- Sparse hard-failure cases: `{(data.get('sparse_quality') or {}).get('hard_failure_case_count')}`",
        f"- Sparse intelligibility pass: `{(data.get('sparse_quality') or {}).get('intelligibility_pass_rate')}`",
        f"- Exact generation match: `{data.get('exact_generation_match_rate')}`",f"- Mean dense fallback rate: `{mean_fallback:.4f}`",
        f"- Response-control mode: `{data.get('response_control_mode')}`",f"- Response-control target passed: `{response_control_ok}`","",explanation,"",
        "Review runtime_comparison_raw.jsonl, every hard/major/low-tail case, and profiler measurements before approval.",
    ]),encoding="utf-8")
    args.gate.write_text("\n".join([
        "# PHASE 7 STOP GATE","","## Automated Gate Status","",f"`{automated}`","",f"- Recommended deployment mode: `{recommendation}`",f"- Reason: {explanation}","",
        "A negative sparse result does not block a mature dense release. The dense reference itself must pass quality-v2 and response control.","","## Human decision","",
        "- Selected deployment mode: `PENDING`","- Human decision: `PENDING`","- Reviewer:","- Date:","- Notes:",
    ]),encoding="utf-8")
    print(args.report); print(args.gate)


if __name__=="__main__": main()

# Purpose: Combines gate chain, sealed quality-v2 metrics, hashes, runtime mode, and blockers.
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.common.experiment_utils import read_json, sha256, sha256_tree, write_json
from scripts.common.reference_selection import DEFAULT_REFERENCE_MANIFEST
from scripts.common.quality_evaluation import require_quality_v2, summary_quality_score

GATE_LABELS=("1","2","3","4","4B","5","6","7")


def gate_status(path: Path, *, deviation_path: Path | None=None) -> str:
    if path.exists():
        text=path.read_text(encoding="utf-8")
        if re.search(r"(?im)^\s*-?\s*Human decision:\s*`?PASS`?\s*$",text): return "PASS"
    if deviation_path is not None and deviation_path.exists():
        text=deviation_path.read_text(encoding="utf-8")
        if re.search(r"(?im)^\s*-?\s*Human decision:\s*`?PASS`?\s*$",text): return "PASS_BY_DEVIATION"
    return "MISSING" if not path.exists() else "NOT_APPROVED"


def deployment_mode(path: Path) -> str:
    if not path.exists(): return "UNKNOWN"
    match=re.search(r"(?im)^\s*-\s*Selected deployment mode:\s*`?([A-Z_]+)`?\s*$",path.read_text(encoding="utf-8"))
    return match.group(1) if match else "UNKNOWN"


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="Build the Phase 8 release-candidate audit.")
    p.add_argument("--model-dir",type=Path,required=True); p.add_argument("--sealed-summary",type=Path,required=True)
    p.add_argument("--ablation-matrix",type=Path,required=True); p.add_argument("--reference-manifest",type=Path,default=DEFAULT_REFERENCE_MANIFEST)
    p.add_argument("--out",type=Path,default=ROOT/"reports/RELEASE_CANDIDATE_AUDIT.md")
    p.add_argument("--min-sealed-quality", "--min-sealed-score", dest="min_sealed_quality",type=float,default=.85)
    p.add_argument("--min-quality-pass-rate",type=float,default=.98); p.add_argument("--min-intelligibility-pass-rate",type=float,default=1.0)
    p.add_argument("--min-semantic-pass-rate",type=float,default=.98); p.add_argument("--min-worst-decile-quality",type=float,default=.70)
    p.add_argument("--min-complete-rate",type=float,default=.99); p.add_argument("--max-major-issue-cases",type=int,default=0)
    p.add_argument("--required-response-control",default="one_sentence"); p.add_argument("--max-one-sentence-violation-rate",type=float,default=0.0)
    return p.parse_args()


def main() -> None:
    args=parse_args()
    if not args.model_dir.is_dir(): raise SystemExit(f"Release model must be a local checkpoint directory: {args.model_dir}")
    for required in (args.sealed_summary,args.ablation_matrix,args.reference_manifest):
        if not required.exists(): raise SystemExit(f"Missing Phase 8 input: {required}")
    gates={label:gate_status(ROOT/"reports"/f"PHASE_{label}_STOP_GATE.md",deviation_path=(ROOT/"reports/PHASE_2_DEVIATION_APPROVAL.md") if label=="2" else None) for label in GATE_LABELS}
    sealed=read_json(args.sealed_summary); ablations=read_json(args.ablation_matrix); mode=deployment_mode(ROOT/"reports/PHASE_7_STOP_GATE.md")
    reasons:list[str]=[]
    try: require_quality_v2(sealed,"sealed release evaluation")
    except ValueError as exc: reasons.append(str(exc))
    if any(v not in {"PASS","PASS_BY_DEVIATION"} for v in gates.values()): reasons.append("one or more prior human gates are not PASS")
    if int(sealed.get("hard_failure_case_count",0))!=0: reasons.append("sealed evaluation contains hard-failure outputs")
    if int(sealed.get("critical_failure_count",0))!=0: reasons.append("sealed evaluation contains critical failures")
    if int(sealed.get("major_case_count",sealed.get("major_issue_count",0)))>args.max_major_issue_cases: reasons.append("sealed evaluation contains too many major-issue cases")
    if summary_quality_score(sealed)<args.min_sealed_quality: reasons.append("sealed mean quality is below the release floor")
    if float(sealed.get("quality_pass_rate",0.0))<args.min_quality_pass_rate: reasons.append("sealed quality-pass rate is below the release floor")
    if float(sealed.get("intelligibility_pass_rate",0.0))<args.min_intelligibility_pass_rate: reasons.append("sealed intelligibility pass rate is below the zero-tolerance floor")
    if float(sealed.get("semantic_pass_rate",0.0))<args.min_semantic_pass_rate: reasons.append("sealed semantic pass rate is below the release floor")
    if float(sealed.get("worst_decile_mean_quality_score",0.0))<args.min_worst_decile_quality: reasons.append("sealed worst-decile quality is below the release floor")
    if float(sealed.get("complete_utterance_rate",0.0))<args.min_complete_rate: reasons.append("sealed complete-utterance rate is below the release floor")
    if any(int(v)>0 for v in (sealed.get("category_hard_failure_counts") or {}).values()): reasons.append("one or more sealed categories contain hard failures")
    if sealed.get("response_control_mode")!=args.required_response_control: reasons.append(f"sealed evaluation did not use required response control {args.required_response_control!r}")
    if float(sealed.get("one_sentence_violation_rate",1.0))>args.max_one_sentence_violation_rate: reasons.append("sealed one-sentence violation rate exceeds the release floor")
    if mode not in {"DENSE_RUNTIME","SPARSE_FEATURE_FLAG"}: reasons.append("Phase 7 deployment mode is not finalized")

    automated="PASS_PENDING_HUMAN_RELEASE_DECISION" if not reasons else "BLOCKED"
    reference=read_json(args.reference_manifest)
    hashes={"model_tree_sha256":sha256_tree(args.model_dir),"reference_manifest_sha256":sha256(args.reference_manifest),"sealed_summary_sha256":sha256(args.sealed_summary),"ablation_matrix_sha256":sha256(args.ablation_matrix)}
    payload:dict[str,Any]={
        "status":automated,"quality_evaluation_version":sealed.get("quality_evaluation_version"),"model_dir":str(args.model_dir.resolve()),"deployment_mode":mode,
        "response_control":{"required_mode":args.required_response_control,"observed_mode":sealed.get("response_control_mode"),"one_sentence_violation_rate":sealed.get("one_sentence_violation_rate"),"max_allowed_violation_rate":args.max_one_sentence_violation_rate,"max_new_tokens":sealed.get("max_new_tokens"),"stop_reason_counts":sealed.get("stop_reason_counts")},
        "quality_thresholds":{"min_mean_quality":args.min_sealed_quality,"min_quality_pass_rate":args.min_quality_pass_rate,"min_intelligibility_pass_rate":args.min_intelligibility_pass_rate,"min_semantic_pass_rate":args.min_semantic_pass_rate,"min_worst_decile_quality":args.min_worst_decile_quality,"min_complete_rate":args.min_complete_rate,"max_major_issue_cases":args.max_major_issue_cases},
        "reference_manifest":str(args.reference_manifest),"selected_candidate":reference.get("selected_candidate"),"gates":gates,"sealed_summary":sealed,
        "ablation_coverage":{"available":ablations.get("available"),"missing":ablations.get("missing"),"stale_quality_evidence":ablations.get("stale_quality_evidence",[])},
        "blocking_reasons":reasons,"hashes":hashes,
    }
    lines=["# RELEASE CANDIDATE AUDIT","","## Automated status","",f"`{automated}`","",f"- Selected model: `{args.model_dir.resolve()}`",f"- Selected candidate: `{reference.get('selected_candidate')}`",f"- Deployment mode: `{mode}`",f"- Model tree SHA-256: `{hashes['model_tree_sha256']}`","","## Gate chain","",*[f"- Phase {k}: `{v}`" for k,v in gates.items()],"","## Sealed quality-v2 evaluation","",f"- Cases: `{sealed.get('case_count')}`",f"- Mean quality: `{sealed.get('mean_quality_score')}`",f"- Quality-pass rate: `{sealed.get('quality_pass_rate')}`",f"- Intelligibility-pass rate: `{sealed.get('intelligibility_pass_rate')}`",f"- Worst-decile quality: `{sealed.get('worst_decile_mean_quality_score')}`",f"- Semantic-pass rate: `{sealed.get('semantic_pass_rate')}`",f"- Complete-utterance rate: `{sealed.get('complete_utterance_rate')}`",f"- Hard-failure cases: `{sealed.get('hard_failure_case_count')}`",f"- Critical failures: `{sealed.get('critical_failure_count')}`",f"- Major-issue cases: `{sealed.get('major_case_count')}`",f"- Dimension means: `{sealed.get('quality_dimension_means')}`",f"- Category quality: `{sealed.get('category_quality_scores')}`",f"- Response-control mode: `{sealed.get('response_control_mode')}`",f"- One-sentence violation rate: `{sealed.get('one_sentence_violation_rate')}`",f"- Stop reasons: `{sealed.get('stop_reason_counts')}`",f"- Reference alignment: `{sealed.get('mean_reference_alignment_score')}`","","## Ablation evidence coverage","",f"- Available entries: `{ablations.get('available')}`",f"- Missing entries: `{', '.join(ablations.get('missing',[])) or 'None'}`",f"- Stale legacy-quality entries: `{', '.join(ablations.get('stale_quality_evidence',[])) or 'None'}`","","## Blocking reasons","",*([f"- {r}" for r in reasons] or ["- None."]),"","## Human release decision","","- Decision: `PENDING`","- Reviewer:","- Date:","- Version:","- Rollback checkpoint verified: `PENDING`","- Notes:"]
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text("\n".join(lines),encoding="utf-8"); write_json(args.out.with_suffix(".json"),payload); print(args.out)


if __name__=="__main__": main()

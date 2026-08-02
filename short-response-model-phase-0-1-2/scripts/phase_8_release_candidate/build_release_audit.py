from __future__ import annotations

"""Build the final technical and human-release audit without quality floors."""

import argparse
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json, sha256, sha256_tree, write_json
from scripts.common.quality_evaluation import require_quality_v4
from scripts.common.opportunity_evaluation import OPPORTUNITY_POLICY_VERSION
from scripts.common.reference_selection import DEFAULT_REFERENCE_MANIFEST

GATE_LABELS = ("1", "2", "3", "4", "4B", "5", "6", "7")


def gate_status(path: Path, *, deviation_path: Path | None = None) -> str:
    if path.exists() and re.search(r"(?im)^\s*-?\s*Human decision:\s*`?PASS`?\s*$", path.read_text(encoding="utf-8")):
        return "PASS"
    if deviation_path is not None and deviation_path.exists() and re.search(
        r"(?im)^\s*-?\s*Human decision:\s*`?PASS`?\s*$", deviation_path.read_text(encoding="utf-8")
    ):
        return "PASS_BY_DEVIATION"
    return "MISSING" if not path.exists() else "NOT_APPROVED"


def deployment_mode(path: Path) -> str:
    if not path.exists():
        return "UNKNOWN"
    match = re.search(r"(?im)^\s*-\s*Selected deployment mode:\s*`?([A-Z_]+)`?\s*$", path.read_text(encoding="utf-8"))
    return match.group(1) if match else "UNKNOWN"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Phase 8 release-candidate audit.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--sealed-summary", type=Path, required=True)
    parser.add_argument("--ablation-matrix", type=Path, required=True)
    parser.add_argument(
        "--runtime-comparison",
        type=Path,
        default=ROOT / "artifacts/phase_7_router/runtime_comparison.json",
        help="Phase 7 dense-versus-sparse relative opportunity evidence.",
    )
    parser.add_argument("--reference-manifest", type=Path, default=DEFAULT_REFERENCE_MANIFEST)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/RELEASE_CANDIDATE_AUDIT.md")
    parser.add_argument("--required-response-control", default="one_sentence")
    # Historical fixed release floors are accepted and ignored.
    for flag in (
        "--min-sealed-quality", "--min-sealed-score", "--min-quality-pass-rate",
        "--min-intelligibility-pass-rate", "--min-semantic-pass-rate",
        "--min-worst-decile-quality", "--min-complete-rate", "--max-major-issue-cases",
        "--max-one-sentence-violation-rate",
    ):
        parser.add_argument(flag, type=float, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_dir.is_dir():
        raise SystemExit(f"Release model must be a local checkpoint directory: {args.model_dir}")
    for required in (args.sealed_summary, args.ablation_matrix, args.runtime_comparison, args.reference_manifest):
        if not required.exists():
            raise SystemExit(f"Missing Phase 8 input: {required}")
    gates = {
        label: gate_status(
            ROOT / "reports" / f"PHASE_{label}_STOP_GATE.md",
            deviation_path=(ROOT / "reports/PHASE_2_DEVIATION_APPROVAL.md") if label == "2" else None,
        )
        for label in GATE_LABELS
    }
    sealed = read_json(args.sealed_summary)
    ablations = read_json(args.ablation_matrix)
    runtime = read_json(args.runtime_comparison)
    reference = read_json(args.reference_manifest)
    mode = deployment_mode(ROOT / "reports/PHASE_7_STOP_GATE.md")
    reasons: list[str] = []
    try:
        require_quality_v4(sealed, "sealed release evaluation")
    except (ValueError, TypeError) as exc:
        reasons.append(str(exc))
    if any(value not in {"PASS", "PASS_BY_DEVIATION"} for value in gates.values()):
        reasons.append("one or more prior human gates are not PASS")
    if sealed.get("response_control_mode") != args.required_response_control:
        reasons.append(f"sealed evaluation did not use required response-control mode {args.required_response_control!r}")
    if mode not in {"DENSE_RUNTIME", "SPARSE_FEATURE_FLAG"}:
        reasons.append("Phase 7 deployment mode is not finalized")

    # Phase 8 does not create an absolute sealed-quality floor. Its automated
    # comparative evidence is the already measured Phase 7 dense-versus-sparse
    # opportunity result, produced on identical cases and hardware. The sealed
    # candidate evaluation remains a one-time, continuous human-review packet.
    runtime_opportunity = runtime.get("quality_comparison") or runtime.get("opportunity_assessment") or {}
    dense_runtime_quality = runtime.get("dense_quality") or {}
    sparse_runtime_quality = runtime.get("sparse_quality") or {}
    try:
        require_quality_v4(dense_runtime_quality, "Phase 7 dense runtime quality")
        require_quality_v4(sparse_runtime_quality, "Phase 7 sparse runtime quality")
    except (ValueError, TypeError) as exc:
        reasons.append(str(exc))
    if not isinstance(runtime_opportunity, dict) or not runtime_opportunity.get("technical_valid", False):
        reasons.append("Phase 7 relative opportunity evidence is missing or technically invalid")
    if mode == "SPARSE_FEATURE_FLAG" and not runtime_opportunity.get("eligible", False):
        reasons.append("Phase 7 sparse mode was selected without a positive relative opportunity margin")
    automated = "PASS_PENDING_HUMAN_RELEASE_DECISION" if not reasons else "BLOCKED_TECHNICAL_EVIDENCE"
    hashes = {
        "model_tree_sha256": sha256_tree(args.model_dir),
        "reference_manifest_sha256": sha256(args.reference_manifest),
        "sealed_summary_sha256": sha256(args.sealed_summary),
        "ablation_matrix_sha256": sha256(args.ablation_matrix),
        "runtime_comparison_sha256": sha256(args.runtime_comparison),
    }
    deprecated = [
        name for name, value in vars(args).items()
        if name not in {"model_dir", "sealed_summary", "ablation_matrix", "runtime_comparison", "reference_manifest", "out", "required_response_control"}
        and value is not None
    ]
    payload: dict[str, Any] = {
        "status": automated,
        "quality_evaluation_version": sealed.get("quality_evaluation_version"),
        "opportunity_policy_version": OPPORTUNITY_POLICY_VERSION,
        "model_dir": str(args.model_dir.resolve()),
        "deployment_mode": mode,
        "response_control": {
            "required_mode": args.required_response_control,
            "observed_mode": sealed.get("response_control_mode"),
            "one_sentence_violation_rate_diagnostic": sealed.get("one_sentence_violation_rate"),
            "max_new_tokens": sealed.get("max_new_tokens"),
            "stop_reason_counts": sealed.get("stop_reason_counts"),
        },
        "quality_policy": "No absolute sealed-quality floor controls release promotion; sealed evidence is reviewed continuously and comparatively by humans.",
        "deprecated_threshold_arguments_ignored": deprecated,
        "reference_manifest": str(args.reference_manifest),
        "selected_candidate": reference.get("selected_candidate"),
        "gates": gates,
        "sealed_summary": sealed,
        "relative_release_comparison": {
            "source": str(args.runtime_comparison),
            "deployment_mode": mode,
            "primary_resource": runtime_opportunity.get("primary_resource"),
            "resource_gain_fraction": runtime_opportunity.get("resource_gain_fraction"),
            "quality_opportunity_loss_fraction": runtime_opportunity.get("quality_opportunity_loss_fraction"),
            "opportunity_margin": runtime_opportunity.get("opportunity_margin"),
            "decision": runtime_opportunity.get("decision"),
            "technical_valid": runtime_opportunity.get("technical_valid"),
        },
        "ablation_coverage": {
            "available": ablations.get("available"),
            "missing": ablations.get("missing"),
            "stale_quality_evidence": ablations.get("stale_quality_evidence", []),
        },
        "blocking_reasons": reasons,
        "hashes": hashes,
    }
    lines = [
        "# RELEASE CANDIDATE AUDIT", "", "## Automated status", "", f"`{automated}`", "",
        f"- Selected model: `{args.model_dir.resolve()}`", f"- Selected candidate: `{reference.get('selected_candidate')}`",
        f"- Deployment mode: `{mode}`", f"- Model tree SHA-256: `{hashes['model_tree_sha256']}`", "",
        "## Gate chain", "", *[f"- Phase {key}: `{value}`" for key, value in gates.items()], "",
        "## Sealed continuous quality evidence", "", f"- Cases: `{sealed.get('case_count')}`",
        f"- Mean quality: `{sealed.get('mean_quality_score')}`", f"- Worst-decile quality: `{sealed.get('worst_decile_mean_quality_score')}`",
        f"- Dimension means: `{sealed.get('quality_dimension_means')}`", f"- Category quality: `{sealed.get('category_quality_scores')}`",
        f"- Mean issue cost: `{sealed.get('mean_issue_cost')}`", f"- Severe issue case rate: `{sealed.get('severe_issue_case_rate')}`",
        f"- Diagnostic issue-free rate: `{sealed.get('quality_pass_rate')}`", f"- Response-control mode: `{sealed.get('response_control_mode')}`",
        f"- One-sentence violation rate (diagnostic): `{sealed.get('one_sentence_violation_rate')}`", "",
        "No absolute numerical quality value in this section independently blocks or approves release.", "",
        "## Relative release-candidate comparison", "",
        f"- Source: `{args.runtime_comparison}`",
        f"- Deployment mode: `{mode}`",
        f"- Primary resource: `{runtime_opportunity.get('primary_resource')}`",
        f"- Resource gain: `{runtime_opportunity.get('resource_gain_fraction')}`",
        f"- Quality opportunity loss: `{runtime_opportunity.get('quality_opportunity_loss_fraction')}`",
        f"- Opportunity margin: `{runtime_opportunity.get('opportunity_margin')}`",
        f"- Relative decision: `{runtime_opportunity.get('decision')}`", "",
        "Sparse release is technically blocked when this same-case, same-hardware relative evidence is invalid or non-positive. Dense release remains the comparison reference and still requires human approval.", "",
        "## Ablation evidence coverage", "", f"- Available entries: `{ablations.get('available')}`",
        f"- Missing entries: `{', '.join(ablations.get('missing', [])) or 'None'}`",
        f"- Stale legacy-quality entries: `{', '.join(ablations.get('stale_quality_evidence', [])) or 'None'}`", "",
        "## Technical blocking reasons", "", *([f"- {reason}" for reason in reasons] or ["- None."]), "",
        "## Human release decision", "", "- Decision: `PENDING`", "- Reviewer:", "- Date:", "- Version:",
        "- Rollback checkpoint verified: `PENDING`", "- Notes:",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    write_json(args.out.with_suffix(".json"), payload)
    print(args.out)


if __name__ == "__main__":
    main()

# Purpose: Combines gate chain, sealed metrics, hashes, runtime mode, and blockers into the release audit.
from __future__ import annotations

"""Build a measured release audit for the exact selected checkpoint.

The audit checks every human gate, reads the sealed evaluation rather than only
printing a checklist, records the Phase 7 deployment choice, hashes the model
and key manifests, and produces a practical release recommendation.  Final
commercial release still requires an explicit human decision.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_json, sha256, sha256_tree, write_json
from scripts.common.reference_selection import DEFAULT_REFERENCE_MANIFEST

GATE_LABELS = ("1", "2", "3", "4", "4B", "5", "6", "7")


def gate_status(path: Path, *, deviation_path: Path | None = None) -> str:
    """Return PASS, PASS_BY_DEVIATION, MISSING, or NOT_APPROVED.

    Phase 2 has an established negative-result workflow: the automated gate may
    remain FAIL while a separate human deviation report explicitly approves
    continuing with the all-layer reference.  The release audit must preserve
    that decision instead of incorrectly blocking the mature pipeline.
    """

    if path.exists():
        text = path.read_text(encoding="utf-8")
        if re.search(r"(?im)^\s*-?\s*Human decision:\s*`?PASS`?\s*$", text):
            return "PASS"
    if deviation_path is not None and deviation_path.exists():
        deviation_text = deviation_path.read_text(encoding="utf-8")
        if re.search(r"(?im)^\s*-?\s*Human decision:\s*`?PASS`?\s*$", deviation_text):
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
    parser.add_argument("--reference-manifest", type=Path, default=DEFAULT_REFERENCE_MANIFEST)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/RELEASE_CANDIDATE_AUDIT.md")
    parser.add_argument("--min-sealed-score", type=float, default=0.80)
    parser.add_argument("--min-complete-rate", type=float, default=0.99)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_dir.is_dir():
        raise SystemExit(f"Release model must be a local checkpoint directory: {args.model_dir}")
    for required in (args.sealed_summary, args.ablation_matrix, args.reference_manifest):
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
    mode = deployment_mode(ROOT / "reports/PHASE_7_STOP_GATE.md")
    reasons: list[str] = []
    if any(value not in {"PASS", "PASS_BY_DEVIATION"} for value in gates.values()):
        reasons.append("one or more prior human gates are not PASS")
    if int(sealed.get("critical_failure_count", 0)) != 0:
        reasons.append("sealed evaluation contains critical failures")
    if float(sealed.get("mean_heuristic_score", 0.0)) < args.min_sealed_score:
        reasons.append("sealed mean score is below the release floor")
    if float(sealed.get("complete_utterance_rate", 0.0)) < args.min_complete_rate:
        reasons.append("sealed complete-utterance rate is below the release floor")
    if mode not in {"DENSE_RUNTIME", "SPARSE_FEATURE_FLAG"}:
        reasons.append("Phase 7 deployment mode is not finalized")

    automated = "PASS_PENDING_HUMAN_RELEASE_DECISION" if not reasons else "BLOCKED"
    reference = read_json(args.reference_manifest)
    hashes = {
        "model_tree_sha256": sha256_tree(args.model_dir),
        "reference_manifest_sha256": sha256(args.reference_manifest),
        "sealed_summary_sha256": sha256(args.sealed_summary),
        "ablation_matrix_sha256": sha256(args.ablation_matrix),
    }
    payload: dict[str, Any] = {
        "status": automated,
        "model_dir": str(args.model_dir.resolve()),
        "deployment_mode": mode,
        "reference_manifest": str(args.reference_manifest),
        "selected_candidate": reference.get("selected_candidate"),
        "gates": gates,
        "sealed_summary": sealed,
        "ablation_coverage": {
            "available": ablations.get("available"),
            "missing": ablations.get("missing"),
        },
        "blocking_reasons": reasons,
        "hashes": hashes,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RELEASE CANDIDATE AUDIT",
        "",
        "## Automated status",
        "",
        f"`{automated}`",
        "",
        f"- Selected model: `{args.model_dir.resolve()}`",
        f"- Selected candidate: `{reference.get('selected_candidate')}`",
        f"- Deployment mode: `{mode}`",
        f"- Model tree SHA-256: `{hashes['model_tree_sha256']}`",
        "",
        "## Gate chain",
        "",
        *[f"- Phase {label}: `{value}`" for label, value in gates.items()],
        "",
        "## Sealed evaluation",
        "",
        f"- Cases: `{sealed.get('case_count')}`",
        f"- Mean heuristic score: `{sealed.get('mean_heuristic_score')}`",
        f"- Complete-utterance rate: `{sealed.get('complete_utterance_rate')}`",
        f"- Semantic pass rate: `{sealed.get('semantic_pass_rate')}`",
        f"- Critical failures: `{sealed.get('critical_failure_count')}`",
        f"- Major issues: `{sealed.get('major_issue_count')}`",
        f"- Lexical reference token-F1: `{sealed.get('mean_reference_token_f1')}`",
        "",
        "## Ablation evidence coverage",
        "",
        f"- Available entries: `{ablations.get('available')}`",
        f"- Missing entries: `{', '.join(ablations.get('missing', [])) or 'None'}`",
        "",
        "Missing controls limit research attribution but do not automatically block a product release when final "
        "quality, safety, rollback, and runtime gates pass.",
        "",
        "## Blocking reasons",
        "",
        *([f"- {reason}" for reason in reasons] or ["- None."]),
        "",
        "## Human release decision",
        "",
        "- Decision: `PENDING`",
        "- Reviewer:",
        "- Date:",
        "- Version:",
        "- Rollback checkpoint verified: `PENDING`",
        "- Notes:",
    ]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    write_json(args.out.with_suffix(".json"), payload)
    print(args.out)


if __name__ == "__main__":
    main()

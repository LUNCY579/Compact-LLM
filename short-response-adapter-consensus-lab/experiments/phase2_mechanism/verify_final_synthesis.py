from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from consensus_lab.io import write_json_atomic  # noqa: E402


OUTPUT = ROOT / "reports/phase2_final_synthesis_verification.json"

ANALYSES = {
    "results/phase2_mechanism/selected_dev_free_run_analysis.json": (
        "DEV_SELECTED_FREE_RUNNING_ANALYSIS_COMPLETE"
    ),
    "results/phase2_mechanism/adaptive_free_run_dev_analysis.json": (
        "DEV_ADAPTIVE_FREE_RUNNING_ANALYSIS_COMPLETE"
    ),
    "results/phase2_mechanism/selected_committee_dev_analysis.json": (
        "DEV_SELECTED_COMMITTEE_PANEL_ANALYSIS_COMPLETE"
    ),
    "results/phase2_mechanism/v22_calibrated_test_analysis.json": (
        "V22_CALIBRATED_TEST_CONFIRMATION_COMPLETE"
    ),
    "results/phase2_mechanism/cross_v212_analysis.json": (
        "V212_FIXED_METHOD_CROSS_CORPUS_ANALYSIS_COMPLETE"
    ),
    "results/phase2_mechanism/final_synthesis_facts.json": (
        "FINAL_SYNTHESIS_FACTS_COMPLETE"
    ),
}

SYNTHESIS_DOCUMENTS = (
    "reports/PHASE2_MECHANISM_REPORT.md",
    "paper/PAPER_ANALYSIS.md",
    "paper/RESULTS_ANALYSIS.md",
    "paper/EXPERIMENTAL_PROTOCOL.md",
    "paper/CLAIMS_AND_SUPPORT.md",
    "paper/EVIDENCE_LEDGER.md",
    "paper/REQUIREMENT_TRACEABILITY.md",
    "paper/LIMITATIONS_AND_ETHICS.md",
    "paper/QUALITATIVE_ANALYSIS.md",
    "reports/PHASE2_LIVE_RESULTS.md",
    "reports/PHASE2_FINAL_SYNTHESIS_FACTS.md",
    "reports/PHASE2_ENVIRONMENT.md",
    "artifacts/phase2_mechanism/environment.json",
    "paper/TABLES.md",
    "paper/FIGURE_PLAN.md",
    "paper/TABLE_PLAN.md",
    "paper/figures/figure_metadata.json",
    "paper/tables/table_metadata.json",
)

# These are workflow placeholders, not legitimate scientific statements of
# residual uncertainty. Final unresolved causal claims may remain unresolved,
# but completed frozen runs may no longer be described as pending.
INTERIM_MARKERS = (
    "[FROZEN SAME-SPLIT DEV COMPARISON REQUIRED]",
    "[FROZEN 48,600-ROW DEV PANEL REQUIRED]",
    "[FROZEN ONE-CONDITION CONFIRMATION REQUIRED]",
    "[v2.1.2 PROVENANCE AND HOLDOUT RESULT REQUIRED]",
    "TABLES_PARTIAL_WITH_EXPLICIT_PENDING_BLOCKS",
    "PENDING FROZEN RUN",
    "FROZEN DEV PANEL / PENDING RUN",
    "FROZEN BEFORE CALIBRATED OUTPUT / PENDING RUN",
    "FROZEN / PENDING RUN",
    "UNRESOLVED pending frozen run",
    "COMPLETE except Figure 9 pending",
    "Tables 10/12 and Table 11 free run pending",
    "INTERIM COMPLETE; answers 11, 12, 15-18 update after frozen runs",
    "Paper figures: **13 rendered; Figure 9 pending free-running DEV**",
    "free-running analysis **PENDING**",
    "cross-corpus analysis **PENDING**",
    "Selected committee/leave-one-out panel: **PENDING**",
    "Selected calibrated v2.2 TEST confirmation: **PENDING**",
    "Final synthesis fact packet: **PENDING**",
    "answerable/abstention sequence deltas remain pending free-running analysis",
    "free-running exact accuracy and measured latency remain pending",
    "not rendered | PENDING same-split free-running DEV",
    "PENDING the frozen same-split free-running DEV run",
    "PENDING fixed-method v2.1.2 transfer",
    "PENDING one-condition calibrated v2.2 TEST confirmation",
    "PENDING E-FREERUN-DEV",
    "PENDING E-COMMITTEE-FREERUN",
    "PENDING E-V22-CALTEST",
    "AWAITING COMPLETE E-FREERUN-DEV",
    "Counterfactual trajectory effects remain pending until the frozen same-split",
    "## Pending result blocks",
    "same-split free-running DEV, adaptive consultation, and fixed-method v2.1.2 transfer remain open",
)

REQUIRED_FINAL_REFERENCES = {
    "paper/PAPER_ANALYSIS.md": (
        "selected_dev_free_run_analysis.json",
        "adaptive_free_run_dev_analysis.json",
        "selected_committee_dev_analysis.json",
        "v22_calibrated_test_analysis.json",
        "cross_v212_analysis.json",
        "final_synthesis_facts.json",
    ),
    "reports/PHASE2_MECHANISM_REPORT.md": (
        "selected_dev_free_run_analysis.json",
        "adaptive_free_run_dev_analysis.json",
        "selected_committee_dev_analysis.json",
        "v22_calibrated_test_analysis.json",
        "cross_v212_analysis.json",
        "final_synthesis_facts.json",
    ),
    "paper/RESULTS_ANALYSIS.md": (
        "selected_dev_free_run_analysis.json",
        "adaptive_free_run_dev_analysis.json",
        "selected_committee_dev_analysis.json",
        "v22_calibrated_test_analysis.json",
        "cross_v212_analysis.json",
        "final_synthesis_facts.json",
    ),
    "paper/QUALITATIVE_ANALYSIS.md": (
        "selected_dev_free_run_analysis.json",
        "final_synthesis_facts.json",
    ),
    "paper/EVIDENCE_LEDGER.md": (
        "selected_dev_free_run_analysis.json",
        "adaptive_free_run_dev_analysis.json",
        "selected_committee_dev_analysis.json",
        "v22_calibrated_test_analysis.json",
        "cross_v212_analysis.json",
        "final_synthesis_facts.json",
        "rank_topk_parity_verification.json",
        "selected_dev_resume_history.json",
        "selected_dev_runtime_boundaries.json",
    ),
    "paper/CLAIMS_AND_SUPPORT.md": (
        "selected_dev_free_run_analysis.json",
        "adaptive_free_run_dev_analysis.json",
        "selected_committee_dev_analysis.json",
        "v22_calibrated_test_analysis.json",
        "cross_v212_analysis.json",
        "final_synthesis_facts.json",
    ),
    "paper/REQUIREMENT_TRACEABILITY.md": (
        "selected_dev_free_run_analysis.json",
        "adaptive_free_run_dev_analysis.json",
        "selected_committee_dev_analysis.json",
        "v22_calibrated_test_analysis.json",
        "cross_v212_analysis.json",
        "final_synthesis_facts.json",
    ),
    "paper/EXPERIMENTAL_PROTOCOL.md": (
        "final_synthesis_facts.json",
        "rank_topk_parity_verification.json",
        "selected_dev_resume_history.json",
        "selected_dev_runtime_boundaries.json",
    ),
    "paper/LIMITATIONS_AND_ETHICS.md": (
        "final_synthesis_facts.json",
        "rank_topk_parity_verification.json",
        "selected_dev_runtime_boundaries.json",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def interim_marker_hits(documents: dict[str, str]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for relative, text in documents.items():
        folded = text.casefold()
        for marker in INTERIM_MARKERS:
            if marker.casefold() in folded:
                hits.append({"path": relative, "marker": marker})
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bind the final written Phase 2 synthesis to complete long-run evidence."
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write an INCOMPLETE packet without exiting nonzero.",
    )
    args = parser.parse_args()

    analysis_packets: dict[str, dict[str, Any]] = {}
    analysis_checks: dict[str, bool] = {}
    for relative, expected_status in ANALYSES.items():
        path = ROOT / relative
        packet = load_json(path) if path.is_file() else {}
        analysis_packets[relative] = packet
        analysis_checks[relative] = packet.get("status") == expected_status

    documents: dict[str, str] = {}
    for relative in SYNTHESIS_DOCUMENTS:
        path = ROOT / relative
        documents[relative] = path.read_text(encoding="utf-8") if path.is_file() else ""

    marker_hits = interim_marker_hits(documents)
    required_reference_checks = {
        relative: all(fragment in documents.get(relative, "") for fragment in fragments)
        for relative, fragments in REQUIRED_FINAL_REFERENCES.items()
    }
    report_text = documents.get("reports/PHASE2_MECHANISM_REPORT.md", "")
    question_numbers = [
        int(value) for value in re.findall(r"^###\s+(\d+)\.", report_text, flags=re.MULTILINE)
    ]

    figures_path = ROOT / "paper/figures/figure_metadata.json"
    tables_path = ROOT / "paper/tables/table_metadata.json"
    figures = load_json(figures_path) if figures_path.is_file() else {}
    tables = load_json(tables_path) if tables_path.is_file() else {}
    checks = {
        "all_long_run_analyses_complete": all(analysis_checks.values()),
        "all_synthesis_documents_present": all(documents.values()),
        "no_interim_workflow_markers": not marker_hits,
        "final_analysis_references_present": all(required_reference_checks.values()),
        "all_twenty_questions_present": question_numbers == list(range(1, 21)),
        "final_tables_complete": tables.get("status") == "ALL_PLANNED_TABLES_COMPLETE",
        "figure09_present": "figure09_teacher_vs_free" in figures.get("figures", {}),
        "selected_committee_integrated_in_figure03": (
            "results/phase2_mechanism/selected_committee_dev_analysis.json"
            in figures.get("sources", {})
        ),
    }
    status = "FINAL_SYNTHESIS_VERIFIED" if all(checks.values()) else "INCOMPLETE"
    source_paths = [ROOT / relative for relative in (*ANALYSES, *SYNTHESIS_DOCUMENTS)]
    packet = {
        "schema_version": "1.0",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "analysis_status_checks": analysis_checks,
        "required_reference_checks": required_reference_checks,
        "question_numbers": question_numbers,
        "interim_marker_hits": marker_hits,
        "sources": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in source_paths
            if path.is_file()
        },
        "verification_script": str(Path(__file__).resolve().relative_to(ROOT)).replace(
            "\\", "/"
        ),
        "verification_script_sha256": sha256(Path(__file__).resolve()),
    }
    write_json_atomic(OUTPUT, packet)
    print(json.dumps({"status": status, "output": str(OUTPUT), "checks": checks}, indent=2))
    if status != "FINAL_SYNTHESIS_VERIFIED" and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

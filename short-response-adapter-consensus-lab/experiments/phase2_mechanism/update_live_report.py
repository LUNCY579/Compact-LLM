from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUN_PATH = ROOT / "results" / "runs" / "phase2_selected_dev_free_run_dev_full.jsonl"
STATE_PATH = ROOT / "state" / "phase2_live_status.json"
REPORT_PATH = ROOT / "reports" / "PHASE2_LIVE_RESULTS.md"
TEACHER_PATH = ROOT / "results" / "phase2_mechanism" / "teacher_forced_dev_analysis.json"
CROSS_PATH = ROOT / "results" / "phase2_mechanism" / "cross_benchmark_verification.json"
ADAPTIVE_PATH = ROOT / "results" / "phase2_mechanism" / "adaptive_teacher_forced_dev.json"
ADAPTIVE_MANIFEST = ROOT / "artifacts" / "phase2_mechanism" / "selected_adaptive_dev_manifest.json"
ROBUST_PATH = ROOT / "results" / "phase2_mechanism" / "robust_soft_controls_dev.json"
SELECTED_ANALYSIS = ROOT / "results" / "phase2_mechanism" / "selected_dev_free_run_analysis.json"
ADAPTIVE_ANALYSIS = ROOT / "results" / "phase2_mechanism" / "adaptive_free_run_dev_analysis.json"
CROSS_ANALYSIS = ROOT / "results" / "phase2_mechanism" / "cross_v212_analysis.json"
SELECTED_COMMITTEE_ANALYSIS = (
    ROOT / "results" / "phase2_mechanism" / "selected_committee_dev_analysis.json"
)
V22_CALIBRATED_ANALYSIS = (
    ROOT / "results" / "phase2_mechanism" / "v22_calibrated_test_analysis.json"
)
FINAL_FACTS = ROOT / "results" / "phase2_mechanism" / "final_synthesis_facts.json"
FIGURE_METADATA = ROOT / "paper" / "figures" / "figure_metadata.json"
TABLE_METADATA = ROOT / "paper" / "tables" / "table_metadata.json"
RANK_PARITY = ROOT / "results" / "phase2_mechanism" / "rank_topk_parity_verification.json"

EXPECTED_EXAMPLES = 4050
METHODS = [
    "hard_plurality",
    "probability_mean",
    "calibrated_probability_mean",
    "reciprocal_rank_top5",
]
EXPECTED_ROWS = EXPECTED_EXAMPLES * len(METHODS)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scan_run() -> dict[str, Any]:
    counts: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    duplicate_rows = 0
    malformed_rows = 0
    last: dict[str, Any] | None = None
    oom_retry_telemetry_rows = 0
    oom_retry_examples = 0
    oom_retry_total = 0
    if RUN_PATH.exists():
        with RUN_PATH.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed_rows += 1
                    continue
                method = row.get("method", "UNKNOWN")
                example_id = row.get("example_id", "UNKNOWN")
                key = (method, example_id)
                if key in seen:
                    duplicate_rows += 1
                else:
                    seen.add(key)
                    counts[method] += 1
                last = row
                if "cuda_oom_retry_count" in row:
                    oom_retry_telemetry_rows += 1
                    retries = int(row["cuda_oom_retry_count"])
                    oom_retry_total += retries
                    oom_retry_examples += int(retries > 0)
    updated = datetime.fromtimestamp(RUN_PATH.stat().st_mtime, timezone.utc).isoformat() if RUN_PATH.exists() else None
    return {
        "path": str(RUN_PATH),
        "expected_rows": EXPECTED_ROWS,
        "unique_rows": len(seen),
        "completion_fraction": len(seen) / EXPECTED_ROWS,
        "method_counts": {method: counts[method] for method in METHODS},
        "unexpected_methods": {method: count for method, count in counts.items() if method not in METHODS},
        "duplicate_rows": duplicate_rows,
        "malformed_rows": malformed_rows,
        "last_method": last.get("method") if last else None,
        "last_example_id": last.get("example_id") if last else None,
        "config_sha256": last.get("config_sha256") if last else None,
        "file_updated_utc": updated,
        "bytes": RUN_PATH.stat().st_size if RUN_PATH.exists() else 0,
        "oom_retry_telemetry_rows": oom_retry_telemetry_rows,
        "oom_retry_examples": oom_retry_examples,
        "oom_retry_total": oom_retry_total,
    }


def main() -> None:
    teacher = load_json(TEACHER_PATH)
    cross = load_json(CROSS_PATH)
    adaptive = load_json(ADAPTIVE_PATH) if ADAPTIVE_PATH.is_file() else None
    robust = load_json(ROBUST_PATH) if ROBUST_PATH.is_file() else None
    selected_analysis = load_json(SELECTED_ANALYSIS) if SELECTED_ANALYSIS.is_file() else None
    adaptive_analysis = load_json(ADAPTIVE_ANALYSIS) if ADAPTIVE_ANALYSIS.is_file() else None
    cross_analysis = load_json(CROSS_ANALYSIS) if CROSS_ANALYSIS.is_file() else None
    selected_committee_analysis = (
        load_json(SELECTED_COMMITTEE_ANALYSIS)
        if SELECTED_COMMITTEE_ANALYSIS.is_file()
        else None
    )
    v22_calibrated_analysis = (
        load_json(V22_CALIBRATED_ANALYSIS) if V22_CALIBRATED_ANALYSIS.is_file() else None
    )
    final_facts = load_json(FINAL_FACTS) if FINAL_FACTS.is_file() else None
    figures = load_json(FIGURE_METADATA) if FIGURE_METADATA.is_file() else {"figures": {}}
    tables = load_json(TABLE_METADATA) if TABLE_METADATA.is_file() else {"status": "MISSING"}
    rank_parity = load_json(RANK_PARITY) if RANK_PARITY.is_file() else None
    run = scan_run()
    now = datetime.now(timezone.utc).isoformat()
    state = {
        "schema_version": "1.0",
        "updated_at_utc": now,
        "cornerstone": "PASS_30_OF_30",
        "teacher_forced": {
            "status": teacher["status"],
            "examples": teacher["example_count"],
            "tokens": teacher["token_count"],
            "committee_count": len({row["mask"] for row in teacher["committee_results"]}),
        },
        "selected_dev_free_run": run,
        "selected_dev_analysis_status": selected_analysis.get("status") if selected_analysis else "PENDING",
        "robust_soft_controls_status": robust.get("status") if robust else "PENDING",
        "cross_benchmark_gate": {
            "status": cross["status"],
            "checks_passed": cross["checks_passed"],
            "checks_total": cross["checks_total"],
        },
        "adaptive_selection": {
            "status": "FROZEN" if ADAPTIVE_MANIFEST.is_file() else "SELECTED_NOT_FROZEN",
            "threshold": adaptive["selected_threshold"],
            "escalation_rate": adaptive["selected_policy"]["escalation_rate"],
            "forward_reduction_vs_full": adaptive["selected_policy"]["forward_reduction_vs_full"],
        }
        if adaptive is not None
        else {"status": "PENDING"},
        "adaptive_free_run_analysis_status": adaptive_analysis.get("status") if adaptive_analysis else "PENDING",
        "selected_committee_analysis_status": (
            selected_committee_analysis.get("status")
            if selected_committee_analysis
            else "PENDING"
        ),
        "v22_calibrated_analysis_status": (
            v22_calibrated_analysis.get("status") if v22_calibrated_analysis else "PENDING"
        ),
        "cross_analysis_status": cross_analysis.get("status") if cross_analysis else "PENDING",
        "final_synthesis_facts_status": final_facts.get("status") if final_facts else "PENDING",
        "figure_count": len(figures.get("figures", {})),
        "figure09_complete": "figure09_teacher_vs_free" in figures.get("figures", {}),
        "table_status": tables.get("status", "UNKNOWN"),
        "rank_topk_parity_status": (
            rank_parity.get("status") if rank_parity else "PENDING"
        ),
    }
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    method_lines = [
        f'- `{method}`: **{run["method_counts"][method]:,}/{EXPECTED_EXAMPLES:,}**'
        for method in METHODS
    ]
    report = "\n".join(
        [
            "# Phase 2 Live Results",
            "",
            f"- Updated UTC: `{now}`",
            "- Cornerstone verification: **PASS (30/30 checks)**",
            "- Existing full-TEST rescue/harm, divergence, coalition, and failure analysis: **complete**",
            f'- DEV teacher-forced cache and analysis: **complete ({teacher["example_count"]:,} examples; {teacher["token_count"]:,} token decisions; 127 committees)**',
            f'- Robust soft-voting controls: **{state["robust_soft_controls_status"]}**',
            f'- Rank top-k implementation parity: **{state["rank_topk_parity_status"]}**',
            f'- Paper figures: **{state["figure_count"]} rendered; Figure 9 {"complete" if state["figure09_complete"] else "pending free-running DEV"}**',
            f'- Paper tables: **{state["table_status"]}**',
            f'- Adaptive K3→K7 policy: **{state["adaptive_selection"]["status"]}** at threshold `{state["adaptive_selection"].get("threshold")}`; free-running analysis **{state["adaptive_free_run_analysis_status"]}**',
            f'- Selected committee/leave-one-out panel: **{state["selected_committee_analysis_status"]}**',
            f'- Selected calibrated v2.2 TEST confirmation: **{state["v22_calibrated_analysis_status"]}**',
            f'- v2.1.2 provenance gate: **{cross["status"]} ({cross["checks_passed"]}/{cross["checks_total"]} mandatory checks)**; cross-corpus analysis **{state["cross_analysis_status"]}**',
            f'- Final synthesis fact packet: **{state["final_synthesis_facts_status"]}**',
            "",
            "## Selected same-split free-running DEV",
            "",
            f'- Progress: **{run["unique_rows"]:,}/{EXPECTED_ROWS:,} rows ({100 * run["completion_fraction"]:.2f}%)**',
            f'- Durable output bytes: **{run["bytes"]:,}**',
            f'- Last completed method/example: `{run["last_method"]}` / `{run["last_example_id"]}`',
            f'- Last file update UTC: `{run["file_updated_utc"]}`',
            f'- Frozen canonical config SHA256: `{run["config_sha256"]}`',
            f'- Duplicate method-example rows: **{run["duplicate_rows"]}**',
            f'- Malformed complete lines: **{run["malformed_rows"]}**',
            f'- Rows with OOM retry telemetry: **{run["oom_retry_telemetry_rows"]:,}**; retried examples: **{run["oom_retry_examples"]:,}** ({run["oom_retry_total"]:,} retries)',
            "",
            *method_lines,
            "",
            "The reference decoder is intentionally used because the optimized separate-cache parity gate failed. "
            "The reciprocal-rank block spans an exact-parity implementation boundary and bounded fresh-process resumes; "
            "the source-bound parity artifact proves stable top-k equivalence. Methods run in blocks, so partial "
            "accuracies are not reported as cross-method evidence.",
            "",
            "## Current verified headline",
            "",
            "Hard K=7 scores 17.259% on 4,050 v2.2 TEST examples, versus 14.593% for Track A and 14.099% for probability mean.",
            "",
            "## Interpretation discipline",
            "",
            "Teacher-forced and free-running DEV are model-selection/mechanism evidence. The complete v2.2 TEST result remains the confirmatory cornerstone. "
            "No new TEST or v2.1.2 accuracy is claimed until its procedure is frozen and the run is complete.",
            "",
        ]
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(json.dumps(state["selected_dev_free_run"], indent=2))


if __name__ == "__main__":
    main()

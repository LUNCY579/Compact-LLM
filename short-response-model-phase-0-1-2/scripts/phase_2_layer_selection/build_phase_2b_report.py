from __future__ import annotations

"""Rebuild Phase 2B candidate, stage, partial, and final reports from disk.

This command never imports or invokes Transformers model inference. It operates
only on saved JSON, JSONL, Markdown, and checkpoint metadata artifacts.
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import atomic_write_text, read_json
from scripts.phase_2_layer_selection.incremental_reporting import (
    collect_stage_statuses,
    load_json_if_present,
    rebuild_reports_from_artifacts,
    write_live_progress,
    write_partial_run_report,
    write_run_state,
)
from scripts.phase_2_layer_selection.progressive_reduction import render_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild Phase 2B reports without model inference.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Generate useful reports for incomplete runs and tolerate candidate-local gaps.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Explicitly refresh all reconstructable reports. The operation is idempotent.",
    )
    return parser.parse_args()


def _load_candidate_registry(run_root: Path) -> list[dict[str, Any]]:
    path = run_root / "candidate_registry.json"
    if not path.exists():
        return []
    payload = read_json(path)
    return list(payload.get("candidates") or []) if isinstance(payload, dict) else []


def rebuild_final_report(run_root: Path) -> Path | None:
    catalog = load_json_if_present(run_root / "model_catalog.json") or {}
    state = load_json_if_present(run_root / "RUN_STATE.json") or {}
    run_complete = str(state.get("run_status", "")).upper() == "COMPLETE" or catalog.get("status") == "PENDING_HUMAN_REVIEW"
    if not run_complete:
        return None
    required = ("baseline", "frontier", "profiles", "termination")
    missing = [name for name in required if not catalog.get(name) and name not in {"frontier"}]
    if missing:
        raise RuntimeError(f"Run is marked complete but final-report inputs are missing: {missing}")
    config = catalog.get("run_configuration") or {}
    args = SimpleNamespace(
        probe_limit=int(catalog.get("probe_case_count", config.get("probe_limit", 0))),
        recovery_top_k=int(config.get("recovery_top_k", 0)),
        quality_profile=str((catalog.get("opportunity_policy") or {}).get("quality_profile", "balanced")),
    )
    candidates = _load_candidate_registry(run_root)
    destination = Path(str(catalog.get("canonical_report") or (run_root / "PHASE_2B_PROGRESSIVE_REDUCTION_REPORT.md")))
    if not destination.is_absolute():
        destination = ROOT / destination
    atomic_write_text(
        destination,
        render_report(
            baseline=catalog["baseline"],
            stages=list(catalog.get("frontier") or []),
            candidates=candidates,
            profiles=dict(catalog.get("profiles") or {}),
            termination=dict(catalog.get("termination") or {}),
            args=args,
        ),
    )
    return destination


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    result = rebuild_reports_from_artifacts(run_root, allow_partial=args.allow_partial)
    final_report: Path | None = None
    final_error: str | None = None
    try:
        final_report = rebuild_final_report(run_root)
    except Exception as exc:
        final_error = f"{type(exc).__name__}: {exc}"
        result.setdefault("structural_errors", []).append(final_error)

    state = load_json_if_present(run_root / "RUN_STATE.json") or {}
    if final_report is not None:
        merged_state = dict(state)
        catalog = load_json_if_present(run_root / "model_catalog.json") or {}
        frontier = list(catalog.get("frontier") or merged_state.get("frontier") or [])
        termination = dict(catalog.get("termination") or merged_state.get("termination") or {})
        last_frontier = frontier[-1] if frontier else None
        completed_stages = sorted(
            {
                int(item.get("stage", 0))
                for item in collect_stage_statuses(run_root)
                if str(item.get("status", "")).startswith("COMPLETE")
            }
            | {int(row.get("stage", 0)) for row in frontier}
        )
        merged_state.update(
            run_status="COMPLETE",
            current_phase="phase_2b_report_rebuild_complete",
            current_stage=termination.get("terminal_stage_attempted"),
            current_candidate=None,
            last_accepted_stage=(last_frontier or {}).get("stage"),
            last_accepted_checkpoint=(last_frontier or {}).get("inference_model"),
            frontier=frontier,
            termination=termination,
            final_report_available=True,
            final_report=str(final_report),
        )
        state = write_run_state(
            run_root,
            **{key: value for key, value in merged_state.items() if key != "run_root"},
        )
        write_live_progress(
            run_root,
            run_status="COMPLETE",
            current_phase="phase_2b_report_rebuild_complete",
            current_stage=termination.get("terminal_stage_attempted"),
            current_candidate=None,
            last_accepted_stage=(last_frontier or {}).get("stage"),
            last_accepted_checkpoint=(last_frontier or {}).get("inference_model"),
            completed_stages=completed_stages,
            candidate_status="complete",
            stage_status="COMPLETE",
            promotion_occurred=bool(frontier),
            metrics_are_provisional=False,
            most_recent_error=None,
        )
    write_partial_run_report(run_root, state)

    print(f"Partial report: {result.get('partial_report')}")
    if final_report is None:
        print("Final report: not generated because completion requirements are not met.")
    else:
        print(f"Final report: {final_report}")
    for warning in result.get("warnings", []):
        print(f"WARNING: {warning}", file=sys.stderr)
    for error in result.get("structural_errors", []):
        print(f"ERROR: {error}", file=sys.stderr)

    errors = result.get("structural_errors") or []
    missing_evidence = any("Missing evidence directory" in str(error) for error in errors)
    if missing_evidence or final_error:
        return 2
    if errors and not args.allow_partial:
        return 2
    if not result.get("run_complete") and not args.allow_partial:
        print("ERROR: run is incomplete; rerun with --allow-partial for a partial report.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

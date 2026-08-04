from __future__ import annotations

"""Incremental status and report generation for Phase 2B.

This module is deliberately model-free. It reads and writes only durable
artifacts, which allows progress reporting and offline report rebuilding without
loading a tokenizer, model, adapter, or CUDA runtime.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from scripts.common.experiment_utils import (
    atomic_write_text,
    read_json,
    read_jsonl_recover,
    recover_atomic_write,
    write_json,
)
from scripts.common.quality_evaluation import aggregate_quality_rows

LIVE_SCHEMA_VERSION = 1
CANDIDATE_STATUS_SCHEMA_VERSION = 1
RUN_STATE_SCHEMA_VERSION = 1
STAGE_STATUS_SCHEMA_VERSION = 1
CANDIDATE_STATES = {
    "pending",
    "preparing",
    "evaluating",
    "evaluation_complete",
    "recovering",
    "recovery_complete",
    "gating",
    "accepted",
    "rejected",
    "interrupted",
    "failed",
}
TERMINAL_CANDIDATE_STATES = {"accepted", "rejected", "failed"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.exists() and not recover_atomic_write(path, validator="json"):
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def initialize_candidate_status(
    path: Path,
    *,
    stage: int,
    candidate_name: str,
    parent_stage: int | None,
    parent_candidate: str | None,
    parent_checkpoint: str,
    candidate_type: str,
    removed_layers: list[int],
    examples_total: int,
    artifact_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    existing = load_json_if_present(path)
    if existing:
        return existing
    now = utc_now()
    payload: dict[str, Any] = {
        "schema_version": CANDIDATE_STATUS_SCHEMA_VERSION,
        "stage": int(stage),
        "candidate_name": str(candidate_name),
        "parent_stage": parent_stage,
        "parent_candidate": parent_candidate,
        "parent_checkpoint": str(parent_checkpoint),
        "candidate_type": str(candidate_type),
        "removed_layer": removed_layers[0] if len(removed_layers) == 1 else None,
        "removed_layers": [int(value) for value in removed_layers],
        "status": "pending",
        "examples_completed": 0,
        "examples_total": int(examples_total),
        "gate_decision": None,
        "promotion_allowed": False,
        "promoted": False,
        "critical_defects": 0,
        "major_defect_count": 0,
        "warning_count": 0,
        "quality_metrics": {},
        "resource_metrics": {},
        "failure": None,
        "interruption": None,
        "artifacts": artifact_paths or {},
        "created_at": now,
        "started_at": None,
        "completed_at": None,
        "last_update_time": now,
        "state_history": [{"status": "pending", "time": now}],
        "inferred_from_legacy_artifacts": False,
    }
    write_json(path, payload)
    return payload


def transition_candidate_status(
    path: Path,
    status: str,
    **updates: Any,
) -> dict[str, Any]:
    if status not in CANDIDATE_STATES:
        raise ValueError(f"Unsupported candidate status: {status}")
    payload = load_json_if_present(path) or {
        "schema_version": CANDIDATE_STATUS_SCHEMA_VERSION,
        "candidate_name": path.parent.name,
        "created_at": utc_now(),
        "state_history": [],
    }
    previous = payload.get("status")
    now = utc_now()
    payload.update(updates)
    payload["status"] = status
    payload["last_update_time"] = now
    if status in {"preparing", "evaluating", "recovering", "gating"} and not payload.get("started_at"):
        payload["started_at"] = now
    if status in TERMINAL_CANDIDATE_STATES:
        payload["completed_at"] = now
    history = list(payload.get("state_history") or [])
    if previous != status:
        history.append({"status": status, "time": now})
    payload["state_history"] = history
    write_json(path, payload)
    return payload


def candidate_gate_fields(row: dict[str, Any]) -> dict[str, Any]:
    assessment = row.get("assessment") or {}
    generation = row.get("generation_summary") or {}
    eligible = bool(assessment.get("eligible_for_progression"))
    return {
        "gate_decision": "ACCEPTED" if eligible else "REJECTED",
        "promotion_allowed": eligible,
        "critical_defects": int(generation.get("critical_failure_count", 0)),
        "major_defect_count": int(generation.get("major_issue_count", 0)),
        "warning_count": int(generation.get("semantic_warning_count", generation.get("review_diagnostic_count", 0))),
        "quality_metrics": {
            "mean_quality_score": generation.get("mean_quality_score"),
            "worst_decile_mean_quality_score": generation.get("worst_decile_mean_quality_score"),
            "complete_utterance_rate": generation.get("complete_utterance_rate"),
            "quality_opportunity_loss_fraction": assessment.get("quality_opportunity_loss_fraction"),
            "opportunity_margin": assessment.get("opportunity_margin"),
            "incremental_opportunity_margin": assessment.get("incremental_opportunity_margin"),
        },
        "resource_metrics": {
            "weight_memory_reduction_fraction": assessment.get("weight_memory_reduction_fraction"),
            "incremental_weight_memory_reduction_fraction": assessment.get(
                "incremental_weight_memory_reduction_fraction"
            ),
            "parameter_count_reduction_fraction": assessment.get("parameter_count_reduction_fraction"),
            "peak_evaluation_vram_reduction_fraction": assessment.get(
                "peak_evaluation_vram_reduction_fraction"
            ),
            "mean_generation_latency_reduction_fraction": assessment.get(
                "mean_generation_latency_reduction_fraction"
            ),
        },
    }


def render_candidate_report(status: dict[str, Any], row: dict[str, Any] | None = None) -> str:
    row = row or {}
    assessment = row.get("assessment") or {}
    generation = row.get("generation_summary") or {}
    reasons = assessment.get("rejection_reasons") or []
    reason_text = "\n".join(f"- {reason}" for reason in reasons) or "- None"
    artifact_lines = "\n".join(
        f"- {name}: `{value}`" for name, value in sorted((status.get("artifacts") or {}).items())
    ) or "- None recorded"
    provisional = status.get("status") not in {"accepted", "rejected"}
    return f"""# Phase 2B Candidate Report — {status.get('candidate_name')}

- Stage: **{status.get('stage')}**
- Candidate type: **{status.get('candidate_type')}**
- Parent stage: **{status.get('parent_stage')}**
- Parent candidate: **{status.get('parent_candidate')}**
- Parent checkpoint: `{status.get('parent_checkpoint')}`
- Removed layers: `{status.get('removed_layers')}`
- Status: **{status.get('status')}**
- Gate decision: **{status.get('gate_decision') or 'PENDING'}**
- Promotion allowed: **{bool(status.get('promotion_allowed'))}**
- Promoted as next parent: **{bool(status.get('promoted'))}**
- Evaluation progress: **{status.get('examples_completed', 0)}/{status.get('examples_total', 0)}**
- Evidence state: **{'PROVISIONAL' if provisional else 'FINAL FOR THIS CANDIDATE'}**

## Quality evidence

| Metric | Value |
|---|---:|
| Mean quality | {generation.get('mean_quality_score', (status.get('quality_metrics') or {}).get('mean_quality_score', 'n/a'))} |
| Worst-decile quality | {generation.get('worst_decile_mean_quality_score', (status.get('quality_metrics') or {}).get('worst_decile_mean_quality_score', 'n/a'))} |
| Critical defects | {status.get('critical_defects', 0)} |
| Major defects | {status.get('major_defect_count', 0)} |
| Warnings | {status.get('warning_count', 0)} |
| Quality opportunity loss | {assessment.get('quality_opportunity_loss_fraction', (status.get('quality_metrics') or {}).get('quality_opportunity_loss_fraction', 'n/a'))} |

## Resource and opportunity evidence

| Metric | Value |
|---|---:|
| Weight-memory reduction | {assessment.get('weight_memory_reduction_fraction', (status.get('resource_metrics') or {}).get('weight_memory_reduction_fraction', 'n/a'))} |
| Cumulative opportunity margin | {assessment.get('opportunity_margin', (status.get('quality_metrics') or {}).get('opportunity_margin', 'n/a'))} |
| Incremental opportunity margin | {assessment.get('incremental_opportunity_margin', (status.get('quality_metrics') or {}).get('incremental_opportunity_margin', 'n/a'))} |

## Gate reasons

{reason_text}

## Artifacts

{artifact_lines}

Last updated: `{status.get('last_update_time')}`
"""


def write_candidate_report(candidate_dir: Path, status: dict[str, Any], row: dict[str, Any] | None = None) -> Path:
    destination = candidate_dir / "CANDIDATE_REPORT.md"
    atomic_write_text(destination, render_candidate_report(status, row))
    return destination


def stage_directory(run_root: Path, stage: int) -> Path:
    return run_root / "evidence" / f"stage_{stage:02d}"


def collect_candidate_statuses(run_root: Path, stage: int | None = None) -> list[dict[str, Any]]:
    root = run_root / "evidence"
    stage_dirs = [stage_directory(run_root, stage)] if stage is not None else sorted(root.glob("stage_*"))
    rows: list[dict[str, Any]] = []
    for current_stage in stage_dirs:
        if not current_stage.exists():
            continue
        for candidate_dir in sorted(path for path in current_stage.iterdir() if path.is_dir()):
            status_path = candidate_dir / "candidate_status.json"
            status = load_json_if_present(status_path)
            if status is None:
                status = infer_legacy_candidate_status(candidate_dir)
            if status:
                rows.append(status)
    return rows


def infer_legacy_candidate_status(candidate_dir: Path) -> dict[str, Any] | None:
    """Best-effort status for older artifact folders; never invent a gate decision."""

    match = re.search(r"stage_(\d+)", candidate_dir.as_posix())
    if match is None:
        return None
    stage = int(match.group(1))
    generation_dir = candidate_dir / "generation"
    summary = load_json_if_present(generation_dir / "summary.json")
    combined = load_json_if_present(candidate_dir / "combined_metrics.json")
    results_path = generation_dir / "results.jsonl"
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    if results_path.exists():
        rows, warnings = read_jsonl_recover(results_path, id_field="id", repair_truncated_tail=True)
    assessment = (combined or {}).get("assessment") or {}
    prov = load_json_if_present(candidate_dir / "evidence_provenance.json") or {}
    gate_known = "eligible_for_progression" in assessment
    eligible = bool(assessment.get("eligible_for_progression")) if gate_known else False
    status = "accepted" if gate_known and eligible else "rejected" if gate_known else (
        "evaluation_complete" if summary else "interrupted" if rows else "pending"
    )
    now = utc_now()
    return {
        "schema_version": CANDIDATE_STATUS_SCHEMA_VERSION,
        "stage": stage,
        "candidate_name": candidate_dir.name,
        "parent_stage": stage - 1 if stage > 1 else 0,
        "parent_candidate": None,
        "parent_checkpoint": prov.get("parent_model"),
        "candidate_type": prov.get("variant_kind", "legacy candidate"),
        "removed_layer": prov.get("dropped_original_layer_id"),
        "removed_layers": [prov.get("dropped_original_layer_id")] if prov.get("dropped_original_layer_id") is not None else [],
        "status": status,
        "examples_completed": len(rows) if rows else int((summary or {}).get("case_count", 0)),
        "examples_total": int((summary or {}).get("case_count", len(rows))),
        "gate_decision": ("ACCEPTED" if eligible else "REJECTED") if gate_known else None,
        "promotion_allowed": eligible if gate_known else False,
        "promoted": False,
        "critical_defects": int((summary or {}).get("critical_failure_count", 0)),
        "major_defect_count": int((summary or {}).get("major_issue_count", 0)),
        "warning_count": int((summary or {}).get("semantic_warning_count", 0)),
        "quality_metrics": {
            "mean_quality_score": (summary or {}).get("mean_quality_score"),
            "opportunity_margin": assessment.get("opportunity_margin"),
        },
        "resource_metrics": {
            "weight_memory_reduction_fraction": assessment.get("weight_memory_reduction_fraction"),
        },
        "failure": None,
        "interruption": "; ".join(warnings) if warnings else None,
        "artifacts": {
            "candidate_dir": str(candidate_dir),
            "results": str(results_path),
            "summary": str(generation_dir / "summary.json"),
            "combined_metrics": str(candidate_dir / "combined_metrics.json"),
        },
        "created_at": now,
        "last_update_time": now,
        "state_history": [],
        "inferred_from_legacy_artifacts": True,
        "inference_limits": (
            "Gate decision inferred from combined_metrics.json." if gate_known else
            "No saved gate assessment proves acceptance; promotion remains disabled."
        ),
    }


def _candidate_table(statuses: Iterable[dict[str, Any]]) -> str:
    lines = []
    for status in sorted(statuses, key=lambda item: str(item.get("candidate_name", ""))):
        quality = (status.get("quality_metrics") or {}).get("mean_quality_score")
        margin = (status.get("quality_metrics") or {}).get("opportunity_margin")
        resource = (status.get("resource_metrics") or {}).get("weight_memory_reduction_fraction")
        lines.append(
            f"| {status.get('candidate_name')} | {status.get('candidate_type')} | "
            f"{status.get('status')} | {status.get('gate_decision') or 'PENDING'} | "
            f"{bool(status.get('promotion_allowed'))} | {bool(status.get('promoted'))} | "
            f"{quality if quality is not None else 'n/a'} | {resource if resource is not None else 'n/a'} | "
            f"{margin if margin is not None else 'n/a'} | {status.get('critical_defects', 0)} | "
            f"{status.get('major_defect_count', 0)} |"
        )
    return "\n".join(lines) or "| None | - | pending | PENDING | False | False | n/a | n/a | n/a | 0 | 0 |"


def render_stage_report(
    *,
    stage: int,
    parent_checkpoint: str | None,
    statuses: list[dict[str, Any]],
    stage_status: str,
    accepted_candidate: str | None,
    later_stages_pending: bool,
    stage_evidence_path: str | None = None,
) -> str:
    rejected = [status for status in statuses if status.get("gate_decision") == "REJECTED"]
    reasons: list[str] = []
    for status in rejected:
        candidate_dir = Path(str((status.get("artifacts") or {}).get("candidate_dir", "")))
        row = load_json_if_present(candidate_dir / "candidate_row.json") if candidate_dir else None
        rejection_reasons = ((row or {}).get("assessment") or {}).get("rejection_reasons") or []
        reasons.append(
            f"- `{status.get('candidate_name')}`: " + ("; ".join(map(str, rejection_reasons)) or "gate rejected")
        )
    reason_text = "\n".join(reasons) or "- None"
    artifact_stage_path = stage_evidence_path or (
        statuses[0].get("artifacts", {}).get("stage_dir") if statuses else "not available"
    )
    return f"""# Phase 2B Stage {stage:02d} Report

- Stage identifier: **{stage}**
- Parent checkpoint: `{parent_checkpoint or 'unknown'}`
- Stage status: **{stage_status}**
- Accepted/promoted candidate: **{accepted_candidate or 'None'}**
- Later stages pending: **{later_stages_pending}**
- Candidates tested or discovered: **{len(statuses)}**

## Candidate outcomes

| Candidate | Type | Status | Gate | Promotion allowed | Promoted | Mean quality | Memory gain | Opportunity margin | Critical | Major |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
{_candidate_table(statuses)}

## Rejected candidates and concrete gate reasons

{reason_text}

## Artifact paths

- Stage evidence: `{artifact_stage_path}`
- Each candidate directory contains `candidate_status.json`, `candidate_row.json` when gated, and `CANDIDATE_REPORT.md`.

Generated incrementally at `{utc_now()}`. This report does not depend on later stages finishing.
"""


def write_stage_report(
    run_root: Path,
    *,
    stage: int,
    parent_checkpoint: str | None,
    stage_status: str,
    accepted_candidate: str | None,
    later_stages_pending: bool,
) -> Path:
    statuses = collect_candidate_statuses(run_root, stage)
    stage_dir = stage_directory(run_root, stage)
    destination = stage_dir / f"STAGE_{stage:02d}_REPORT.md"
    atomic_write_text(
        destination,
        render_stage_report(
            stage=stage,
            parent_checkpoint=parent_checkpoint,
            statuses=statuses,
            stage_status=stage_status,
            accepted_candidate=accepted_candidate,
            later_stages_pending=later_stages_pending,
            stage_evidence_path=str(stage_dir.resolve()),
        ),
    )
    status_path = stage_dir / "stage_status.json"
    existing = load_json_if_present(status_path) or {
        "schema_version": STAGE_STATUS_SCHEMA_VERSION,
        "stage": int(stage),
        "created_at": utc_now(),
        "state_history": [],
    }
    previous = existing.get("status")
    now = utc_now()
    existing.update(
        {
            "schema_version": STAGE_STATUS_SCHEMA_VERSION,
            "stage": int(stage),
            "status": stage_status,
            "parent_checkpoint": parent_checkpoint,
            "accepted_candidate": accepted_candidate,
            "later_stages_pending": bool(later_stages_pending),
            "candidate_count": len(statuses),
            "candidates": [item.get("candidate_name") for item in statuses],
            "last_update_time": now,
            "report": str(destination),
        }
    )
    history = list(existing.get("state_history") or [])
    if previous != stage_status:
        history.append({"status": stage_status, "time": now})
    existing["state_history"] = history
    if str(stage_status).startswith("COMPLETE"):
        existing["completed_at"] = now
    write_json(status_path, existing)
    return destination


def collect_stage_statuses(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    evidence = run_root / "evidence"
    if not evidence.exists():
        return rows
    for stage_dir in sorted(evidence.glob("stage_*")):
        payload = load_json_if_present(stage_dir / "stage_status.json")
        if payload:
            rows.append(payload)
    return rows


def write_run_state(run_root: Path, **updates: Any) -> dict[str, Any]:
    path = run_root / "RUN_STATE.json"
    payload = load_json_if_present(path) or {
        "schema_version": RUN_STATE_SCHEMA_VERSION,
        "run_root": str(run_root),
        "started_at": utc_now(),
        "frontier": [],
        "termination": None,
    }
    payload.update(updates)
    payload["last_update_time"] = utc_now()
    write_json(path, payload)
    return payload


def _recent_defects(rows: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    flagged = [row for row in rows if row.get("critical_failures") or row.get("major_issues")]
    return [
        {
            "id": row.get("id"),
            "reply": row.get("reply"),
            "critical": row.get("critical_failures") or [],
            "major": row.get("major_issues") or [],
        }
        for row in flagged[-limit:]
    ]


def provisional_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "critical_count": 0,
            "major_count": 0,
            "warning_count": 0,
            "mean_quality_score": None,
            "recent_defects": [],
        }
    summary = aggregate_quality_rows(rows)
    return {
        "critical_count": int(summary.get("critical_failure_count", 0)),
        "major_count": int(summary.get("major_issue_count", 0)),
        "warning_count": int(summary.get("semantic_warning_count", 0)),
        "mean_quality_score": summary.get("mean_quality_score"),
        "recent_defects": _recent_defects(rows),
    }


def render_live_progress_markdown(payload: dict[str, Any]) -> str:
    completed = payload.get("completed_stages") or []
    defects = payload.get("recent_defects") or []
    defect_lines = "\n".join(
        f"- `{item.get('id')}`: critical={item.get('critical') or '-'}; major={item.get('major') or '-'}; reply={item.get('reply')!r}"
        for item in defects
    ) or "- None among completed examples."
    completed_text = ", ".join(map(str, completed)) if completed else "None"
    provisional = bool(payload.get("metrics_are_provisional", True))
    warning = (
        f"**PROVISIONAL — {payload.get('examples_completed', 0)} of {payload.get('examples_total', 0)} examples evaluated.**\n"
        "Do not make a promotion decision from this partial result."
        if provisional
        else "Candidate evaluation is complete; use the persisted gate decision."
    )
    return f"""# Phase 2B Live Progress

- Run status: **{payload.get('run_status')}**
- Current phase: **{payload.get('current_phase')}**
- Completed stages: **{completed_text}**
- Current stage: **{payload.get('current_stage')}**
- Current candidate: **{payload.get('current_candidate')}**
- Parent candidate/stage: **{payload.get('parent_candidate') or payload.get('parent_stage')}**
- Last accepted stage: **{payload.get('last_accepted_stage')}**
- Last accepted checkpoint: `{payload.get('last_accepted_checkpoint')}`
- Candidate status: **{payload.get('candidate_status')}**
- Stage status: **{payload.get('stage_status')}**
- Promotion occurred: **{bool(payload.get('promotion_occurred'))}**

{warning}

## Provisional metrics

| Metric | Value |
|---|---:|
| Evaluation progress | {payload.get('examples_completed', 0)}/{payload.get('examples_total', 0)} |
| Critical defects | {payload.get('current_critical_count', 0)} |
| Major defects | {payload.get('current_major_count', 0)} |
| Warnings | {payload.get('current_warning_count', 0)} |
| Mean quality | {payload.get('provisional_mean_quality')} |

## Recent critical or major defects

{defect_lines}

## Error or interruption

`{payload.get('most_recent_error') or 'None'}`

Last updated: `{payload.get('last_update_time')}`
"""


def write_live_progress(
    run_root: Path,
    *,
    markdown: bool = True,
    **updates: Any,
) -> dict[str, Any]:
    path = run_root / "LIVE_PROGRESS.json"
    existing = load_json_if_present(path) or {
        "schema_version": LIVE_SCHEMA_VERSION,
        "run_root": str(run_root),
        "run_status": "IN PROGRESS",
        "current_phase": "phase_2b",
        "current_stage": None,
        "current_candidate": None,
        "parent_candidate": None,
        "parent_stage": None,
        "last_accepted_stage": None,
        "last_accepted_checkpoint": None,
        "completed_stages": [],
        "examples_completed": 0,
        "examples_total": 0,
        "current_critical_count": 0,
        "current_major_count": 0,
        "current_warning_count": 0,
        "provisional_mean_quality": None,
        "candidate_status": None,
        "stage_status": None,
        "promotion_occurred": False,
        "metrics_are_provisional": True,
        "start_time": utc_now(),
        "most_recent_error": None,
        "recent_defects": [],
    }
    existing.update(updates)
    existing["last_update_time"] = utc_now()
    write_json(path, existing)
    if markdown:
        atomic_write_text(run_root / "LIVE_PROGRESS.md", render_live_progress_markdown(existing))
    return existing


def render_partial_run_report(run_root: Path, state: dict[str, Any], statuses: list[dict[str, Any]]) -> str:
    stage_states = collect_stage_statuses(run_root)
    completed_stages = sorted(
        int(item.get("stage"))
        for item in stage_states
        if str(item.get("status", "")).startswith("COMPLETE")
    )
    accepted_stages = sorted(
        int(item.get("stage"))
        for item in stage_states
        if item.get("status") == "COMPLETE_ACCEPTED"
    )
    rejected_stages = sorted(
        int(item.get("stage"))
        for item in stage_states
        if item.get("status") == "COMPLETE_REJECTED"
    )
    discovered_stages = {
        int(item.get("stage")) for item in statuses if item.get("stage") is not None
    } | {
        int(item.get("stage")) for item in stage_states if item.get("stage") is not None
    }
    all_stages = sorted(discovered_stages)
    current_stage = state.get("current_stage")
    incomplete_stages = [stage for stage in all_stages if stage not in completed_stages]
    accepted = [item for item in statuses if item.get("gate_decision") == "ACCEPTED"]
    rejected = [item for item in statuses if item.get("gate_decision") == "REJECTED"]
    live = load_json_if_present(run_root / "LIVE_PROGRESS.json") or {}
    missing: list[str] = []
    for status in statuses:
        artifacts = status.get("artifacts") or {}
        for key in ("results", "summary"):
            value = artifacts.get(key)
            if value and not Path(str(value)).exists():
                missing.append(f"{status.get('candidate_name')}: {key} -> {value}")
    candidate_table = _candidate_table(statuses)
    status_name = str(state.get("run_status") or live.get("run_status") or "IN PROGRESS")
    final_available = bool(state.get("final_report_available"))
    rejection_details: list[str] = []
    for status in rejected:
        candidate_dir = Path(str((status.get("artifacts") or {}).get("candidate_dir", "")))
        row = load_json_if_present(candidate_dir / "candidate_row.json") if candidate_dir else None
        reasons = ((row or {}).get("assessment") or {}).get("rejection_reasons") or []
        rejection_details.append(
            f"- `{status.get('candidate_name')}`: " + ("; ".join(map(str, reasons)) or "candidate gate rejected")
        )
    pending_text = (
        "- None. The run is complete and the canonical final report is available."
        if status_name.upper() == "COMPLETE" and final_available
        else "\n".join(
            [
                "- Complete the current candidate and persist its gate decision.",
                "- Complete or terminate the current stage using the existing Phase 2B search strategy.",
                "- Generate the canonical final report only after the run reaches a terminal completion state.",
            ]
        )
    )
    return f"""# Phase 2B Partial Run Report

Run status: **{status_name}**  
Completed stages: **{completed_stages or 'None'}**  
Accepted stages: **{accepted_stages or 'None'}**  
Rejected terminal stages: **{rejected_stages or 'None'}**  
Incomplete stages: **{incomplete_stages or 'None'}**  
Current stage: **{current_stage}**  
Current candidate: **{state.get('current_candidate') or live.get('current_candidate')}**  
Last accepted stage: **{state.get('last_accepted_stage')}**  
Current winning checkpoint: `{state.get('last_accepted_checkpoint')}`  
Final report: **{'available' if final_available else 'not yet available'}**

## Accepted and rejected candidates

| Candidate | Type | Status | Gate | Promotion allowed | Promoted | Mean quality | Memory gain | Opportunity margin | Critical | Major |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
{candidate_table}

- Accepted by candidate gate: **{len(accepted)}**
- Rejected by candidate gate: **{len(rejected)}**
- A gate-accepted candidate is only the next parent when its status also records `promoted=true`.

## Important gate reasons

{chr(10).join(rejection_details) if rejection_details else '- No rejected candidate gate reasons are currently recorded.'}

## Resource and quality progression

The candidate table reports durable per-candidate mean quality, loaded-weight memory gain, and opportunity margin when available. Values for an incomplete candidate remain provisional in `LIVE_PROGRESS.json` and are not promotion evidence.

## Missing or inconsistent artifacts

{chr(10).join(f'- {item}' for item in missing) if missing else '- None detected by the report scanner.'}

## Pending work

{pending_text}

## Live files

- `{run_root / 'LIVE_PROGRESS.json'}`
- `{run_root / 'LIVE_PROGRESS.md'}`
- `{run_root / 'RUN_STATE.json'}`

Generated from saved artifacts at `{utc_now()}`. No model inference was performed.
"""


def write_partial_run_report(run_root: Path, state: dict[str, Any] | None = None) -> Path:
    state = state or load_json_if_present(run_root / "RUN_STATE.json") or {}
    statuses = collect_candidate_statuses(run_root)
    destination = run_root / "PARTIAL_RUN_REPORT.md"
    atomic_write_text(destination, render_partial_run_report(run_root, state, statuses))
    return destination


def rebuild_generation_summary(candidate_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    generation = candidate_dir / "generation"
    summary_path = generation / "summary.json"
    existing = load_json_if_present(summary_path)
    results_path = generation / "results.jsonl"
    warnings: list[str] = []
    if not results_path.exists():
        return existing, warnings
    rows, recovered = read_jsonl_recover(results_path, id_field="id", repair_truncated_tail=True)
    warnings.extend(recovered)
    if existing is not None:
        return existing, warnings
    if not rows:
        return None, warnings
    summary = aggregate_quality_rows(rows)
    summary["summary_rebuilt_offline"] = True
    status = load_json_if_present(candidate_dir / "candidate_status.json") or {}
    progress = load_json_if_present(generation / "evaluation_progress.json") or {}
    combined = load_json_if_present(candidate_dir / "combined_metrics.json") or {}
    expected_total = int(status.get("examples_total") or progress.get("examples_total") or 0)
    complete_status = status.get("status") in {
        "evaluation_complete", "recovery_complete", "gating", "accepted", "rejected"
    }
    complete_proven = (
        bool(combined.get("assessment"))
        or (complete_status and expected_total > 0 and len(rows) >= expected_total)
    )
    summary["summary_rebuild_limitations"] = (
        "Quality aggregates were rebuilt from result rows. Runtime/token distribution fields that were not "
        "recoverable from rows may be absent."
    )
    if complete_proven:
        write_json(summary_path, summary)
    else:
        summary["metrics_are_provisional"] = True
        summary["examples_completed"] = len(rows)
        summary["examples_total"] = expected_total or None
        write_json(generation / "provisional_summary.json", summary)
        warnings.append(
            f"{candidate_dir}: results are incomplete or completion is unproven; wrote provisional_summary.json "
            "instead of canonical summary.json"
        )
    return summary, warnings


def rebuild_reports_from_artifacts(run_root: Path, *, allow_partial: bool = False) -> dict[str, Any]:
    """Reconstruct candidate/stage/partial reports without model inference."""

    structural_errors: list[str] = []
    warnings: list[str] = []
    statuses: list[dict[str, Any]] = []
    evidence_root = run_root / "evidence"
    if not evidence_root.exists():
        structural_errors.append(f"Missing evidence directory: {evidence_root}")
    else:
        for stage_dir in sorted(evidence_root.glob("stage_*")):
            for candidate_dir in sorted(path for path in stage_dir.iterdir() if path.is_dir()):
                try:
                    summary, recovered = rebuild_generation_summary(candidate_dir)
                    warnings.extend(recovered)
                    status_path = candidate_dir / "candidate_status.json"
                    status = load_json_if_present(status_path) or infer_legacy_candidate_status(candidate_dir)
                    if status is None:
                        warnings.append(f"Could not infer status for {candidate_dir}")
                        continue
                    if summary:
                        status["examples_completed"] = max(
                            int(status.get("examples_completed", 0)), int(summary.get("case_count", 0))
                        )
                        status["examples_total"] = max(
                            int(status.get("examples_total", 0)), int(summary.get("case_count", 0))
                        )
                    if not status_path.exists():
                        write_json(status_path, status)
                    row = load_json_if_present(candidate_dir / "candidate_row.json")
                    write_candidate_report(candidate_dir, status, row)
                    statuses.append(status)
                except Exception as exc:  # keep scanning other candidates
                    structural_errors.append(f"{candidate_dir}: {exc}")

    state = load_json_if_present(run_root / "RUN_STATE.json") or {}
    catalog = load_json_if_present(run_root / "model_catalog.json") or {}
    frontier = state.get("frontier") or catalog.get("frontier") or []
    frontier_by_stage = {int(row.get("stage", 0)): row for row in frontier}
    stages = sorted({int(item.get("stage")) for item in statuses if item.get("stage") is not None})
    for stage in stages:
        stage_statuses = [item for item in statuses if int(item.get("stage", -1)) == stage]
        promoted = next((item for item in stage_statuses if item.get("promoted")), None)
        frontier_row = frontier_by_stage.get(stage)
        accepted_name = (promoted or {}).get("candidate_name") or (frontier_row or {}).get("variant_id")
        if accepted_name:
            stage_status = "COMPLETE_ACCEPTED"
        elif any(item.get("status") in {"pending", "preparing", "evaluating", "recovering", "gating", "interrupted"} for item in stage_statuses):
            stage_status = "INCOMPLETE"
        elif stage_statuses and all(item.get("gate_decision") == "REJECTED" for item in stage_statuses):
            stage_status = "COMPLETE_REJECTED"
        else:
            stage_status = "INCOMPLETE"
        parent_checkpoint = next((item.get("parent_checkpoint") for item in stage_statuses if item.get("parent_checkpoint")), None)
        write_stage_report(
            run_root,
            stage=stage,
            parent_checkpoint=parent_checkpoint,
            stage_status=stage_status,
            accepted_candidate=accepted_name,
            later_stages_pending=stage_status == "INCOMPLETE" or stage != max(stages),
        )
    partial = write_partial_run_report(run_root, state)
    run_complete = str(state.get("run_status") or "").upper() == "COMPLETE" or bool(catalog.get("termination")) and bool(catalog.get("profiles"))
    if structural_errors and not allow_partial:
        return {
            "ok": False,
            "run_complete": run_complete,
            "structural_errors": structural_errors,
            "warnings": warnings,
            "partial_report": str(partial),
        }
    return {
        "ok": True,
        "run_complete": run_complete,
        "structural_errors": structural_errors,
        "warnings": warnings,
        "partial_report": str(partial),
        "candidate_count": len(statuses),
        "stage_count": len(stages),
    }

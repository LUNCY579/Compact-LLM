from __future__ import annotations

"""Durable state + incremental Markdown reporting for Student Optimization v2.0.

The master workflow updates this file before and after every expensive stage.  A
run is never labelled COMPLETE merely because a child process exited: COMPLETE
is reserved for a packaged Phase 11 release, while human-review boundaries are
represented explicitly as WAITING_* checkpoints.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


STAGE_ORDER = [
    ("S0", "Accepted-student intake and provenance"),
    ("S1", "Student layer-sensitivity scan"),
    ("S2", "Phase 10 structural reduction + recovery"),
    ("S3", "Phase 10 human checkpoint + materialization"),
    ("S4", "Context / KV-budget optimization on DEV"),
    ("S5", "Contextual sparsity simulation + measured sparse runtime"),
    ("S6", "Quantization/runtime sweep on DEV"),
    ("S7", "Phase 11 frozen production-runtime evaluation"),
    ("S8", "Phase 11 human checkpoint + release package"),
]


def default_state() -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "workflow": "student_optimization_v2",
        "status": "NOT_STARTED",
        "terminal": False,
        "checkpoint": None,
        "started_at": None,
        "updated_at": utc_now(),
        "current_stage": None,
        "inputs": {},
        "stages": {
            key: {
                "id": key,
                "name": name,
                "status": "PENDING",
                "started_at": None,
                "completed_at": None,
                "command": None,
                "artifacts": [],
                "notes": [],
            }
            for key, name in STAGE_ORDER
        },
        "events": [],
    }


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return default_state()
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = utc_now()
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def event(state: dict[str, Any], kind: str, message: str, stage: str | None = None) -> None:
    state.setdefault("events", []).append(
        {"time": utc_now(), "kind": kind, "stage": stage, "message": message}
    )
    # Prevent pathological growth while preserving a useful audit trail.
    state["events"] = state["events"][-500:]


def status_icon(status: str) -> str:
    return {
        "PENDING": "⬜",
        "RUNNING": "🟨",
        "COMPLETE": "✅",
        "SKIPPED": "⏭️",
        "FAILED": "❌",
        "WAITING": "⏸️",
    }.get(status, "•")


def render_report(state: dict[str, Any], path: Path) -> None:
    lines = [
        "# Student Optimization v2.0 — Live Master Report",
        "",
        f"- Workflow status: `{state.get('status')}`",
        f"- Terminal: `{bool(state.get('terminal'))}`",
        f"- Current stage: `{state.get('current_stage') or 'none'}`",
        f"- Checkpoint: `{state.get('checkpoint') or 'none'}`",
        f"- Started: `{state.get('started_at') or 'not started'}`",
        f"- Updated: `{state.get('updated_at')}`",
        "",
        "## Stage ledger",
        "",
        "| Stage | Responsibility | Status | Started | Completed |",
        "|---|---|---|---|---|",
    ]
    for key, name in STAGE_ORDER:
        packet = (state.get("stages") or {}).get(key, {})
        st = str(packet.get("status") or "PENDING")
        lines.append(
            f"| {key} | {name} | {status_icon(st)} `{st}` | "
            f"{packet.get('started_at') or ''} | {packet.get('completed_at') or ''} |"
        )
        artifacts = packet.get("artifacts") or []
        notes = packet.get("notes") or []
        if packet.get("command") or artifacts or notes:
            lines.extend(["", f"### {key} — {name}", ""])
            if packet.get("command"):
                lines.extend(["Command:", "", "```text", str(packet["command"]), "```", ""])
            if artifacts:
                lines.append("Artifacts:")
                lines.extend([f"- `{item}`" for item in artifacts])
                lines.append("")
            if notes:
                lines.append("Notes:")
                lines.extend([f"- {item}" for item in notes])
                lines.append("")

    lines.extend(
        [
            "## Completion semantics",
            "",
            "`COMPLETE` means the automated workflow has no remaining executable work and the approved Phase 11 package exists. "
            "A human gate is never silently converted into PASS.  When human review is required, the workflow exits cleanly at a "
            "`WAITING_*` checkpoint and resumes from that exact boundary when the same command is rerun after approval.",
            "",
            "## Recent events",
            "",
        ]
    )
    for item in (state.get("events") or [])[-30:]:
        lines.append(
            f"- `{item.get('time')}` **{item.get('kind')}**"
            f"{f' [{item.get('stage')}]' if item.get('stage') else ''}: {item.get('message')}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def add_artifacts(packet: dict[str, Any], artifacts: list[str]) -> None:
    existing = list(packet.get("artifacts") or [])
    for item in artifacts:
        if item and item not in existing:
            existing.append(item)
    packet["artifacts"] = existing


def add_notes(packet: dict[str, Any], notes: list[str]) -> None:
    existing = list(packet.get("notes") or [])
    for item in notes:
        if item and item not in existing:
            existing.append(item)
    packet["notes"] = existing


def mutate(args: argparse.Namespace) -> None:
    state_path = args.state
    report_path = args.report
    state = load_state(state_path)
    now = utc_now()

    if args.action == "init":
        if state.get("started_at") is None:
            state["started_at"] = now
        state["status"] = "RUNNING"
        state["terminal"] = False
        state["checkpoint"] = None
        if args.inputs_json:
            state["inputs"] = json.loads(args.inputs_json)
        event(state, "INIT", args.message or "Workflow initialized or resumed.")

    elif args.action in {"start", "finish", "skip", "fail"}:
        if args.stage not in state["stages"]:
            raise ValueError(f"Unknown stage: {args.stage}")
        packet = state["stages"][args.stage]
        if args.command:
            packet["command"] = args.command
        add_artifacts(packet, args.artifact or [])
        add_notes(packet, args.note or [])

        if args.action == "start":
            packet["status"] = "RUNNING"
            packet["started_at"] = packet.get("started_at") or now
            packet["completed_at"] = None
            state["status"] = "RUNNING"
            state["terminal"] = False
            state["checkpoint"] = None
            state["current_stage"] = args.stage
            event(state, "START", args.message or "Stage started.", args.stage)
        elif args.action == "finish":
            packet["status"] = "COMPLETE"
            packet["started_at"] = packet.get("started_at") or now
            packet["completed_at"] = now
            state["status"] = "RUNNING"
            state["current_stage"] = None
            event(state, "FINISH", args.message or "Stage completed.", args.stage)
        elif args.action == "skip":
            packet["status"] = "SKIPPED"
            packet["started_at"] = packet.get("started_at") or now
            packet["completed_at"] = now
            state["status"] = "RUNNING"
            state["current_stage"] = None
            event(state, "SKIP", args.message or "Stage skipped by declared workflow configuration.", args.stage)
        else:
            packet["status"] = "FAILED"
            packet["started_at"] = packet.get("started_at") or now
            packet["completed_at"] = now
            state["status"] = "FAILED"
            state["terminal"] = True
            state["checkpoint"] = "FAILED"
            state["current_stage"] = args.stage
            event(state, "FAIL", args.message or "Stage failed.", args.stage)

    elif args.action == "checkpoint":
        if args.stage:
            if args.stage not in state["stages"]:
                raise ValueError(f"Unknown stage: {args.stage}")
            packet = state["stages"][args.stage]
            if packet.get("status") == "RUNNING":
                packet["status"] = "WAITING"
            add_artifacts(packet, args.artifact or [])
            add_notes(packet, args.note or [])
        state["status"] = args.checkpoint or "WAITING_FOR_HUMAN"
        state["checkpoint"] = args.checkpoint or "WAITING_FOR_HUMAN"
        state["terminal"] = False
        state["current_stage"] = args.stage
        event(state, "CHECKPOINT", args.message or state["status"], args.stage)

    elif args.action == "complete":
        state["status"] = "COMPLETE"
        state["terminal"] = True
        state["checkpoint"] = "NO_FURTHER_AUTOMATED_WORK"
        state["current_stage"] = None
        event(state, "COMPLETE", args.message or "All automated work and approved release packaging completed.")
    elif args.action == "render":
        pass
    else:
        raise ValueError(args.action)

    save_state(state_path, state)
    render_report(state, report_path)
    print(json.dumps({
        "state": str(state_path),
        "report": str(report_path),
        "status": state.get("status"),
        "checkpoint": state.get("checkpoint"),
    }, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update Student Optimization v2.0 workflow state and live report.")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--action",
        required=True,
        choices=("init", "start", "finish", "skip", "fail", "checkpoint", "complete", "render"),
    )
    parser.add_argument("--stage", choices=[item[0] for item in STAGE_ORDER], default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--message", default=None)
    parser.add_argument("--command", default=None)
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--inputs-json", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    mutate(parse_args())

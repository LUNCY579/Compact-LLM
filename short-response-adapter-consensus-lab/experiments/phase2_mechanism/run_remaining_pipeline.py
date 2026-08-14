from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from consensus_lab.io import write_json_atomic  # noqa: E402


STATE = ROOT / "state" / "phase2_remaining_pipeline.json"
SELECTED_RUN = ROOT / "results" / "runs" / "phase2_selected_dev_free_run_dev_full.jsonl"
SELECTED_LOCK = SELECTED_RUN.with_suffix(SELECTED_RUN.suffix + ".lock")
GPU_STAGES: dict[str, tuple[Path, int]] = {
    "run_adaptive_dev_free_run": (
        ROOT / "results/runs/phase2_selected_adaptive_dev_free_run_dev_full.jsonl",
        4_050,
    ),
    "run_selected_committee_dev_panel": (
        ROOT / "results/runs/phase2_selected_committee_dev_free_run_dev_full.jsonl",
        48_600,
    ),
    "run_v22_calibrated_test_confirmation": (
        ROOT / "results/runs/phase2_v22_calibrated_test_confirmation_dev_full.jsonl",
        4_050,
    ),
    "run_cross_v212_controls": (
        ROOT / "results/runs/phase2_cross_v212_controls_dev_full.jsonl",
        3_960,
    ),
    "run_cross_v212_full_k7": (
        ROOT / "results/runs/phase2_cross_v212_full_k7_dev_full.jsonl",
        3_960,
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonl_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def live_lock_owner(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        owner = int(json.loads(path.read_text(encoding="utf-8"))["pid"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    return owner if owner > 0 and psutil.pid_exists(owner) else None


def selected_run_command() -> list[str]:
    return [
        sys.executable,
        "scripts/run_dev.py",
        "--config",
        "configs/phase2_selected_dev_free_run.yaml",
        "--dataset",
        "v22_dev",
        "--device",
        "auto",
        "--precision",
        "auto",
        "--summary-every",
        "25",
        "--max-new-rows",
        "100",
    ]


def run_gpu_stage_batched(
    *,
    name: str,
    command: list[str],
    output_path: Path,
    expected_rows: int,
    batch_rows: int,
    cooldown_seconds: int,
    max_consecutive_failures: int,
    state: dict[str, Any],
) -> None:
    """Run a resumable GPU step in fresh bounded workers.

    Progress made before a CUDA/process failure remains durable. Only repeated
    failures with no new row stop the pipeline; successful batches release the
    model and GPU allocator before an optional cooldown.
    """
    attempts = 0
    consecutive_no_progress_failures = 0
    while True:
        before = jsonl_count(output_path)
        if before == expected_rows:
            state["gpu_batch"] = None
            write_state(state)
            return
        if before > expected_rows:
            raise RuntimeError(f"{name} exceeds its frozen row count: {before}/{expected_rows}")
        attempts += 1
        batched_command = [*command, "--max-new-rows", str(batch_rows)]
        state["gpu_batch"] = {
            "stage": name,
            "attempt": attempts,
            "rows_before": before,
            "expected_rows": expected_rows,
            "batch_rows": batch_rows,
            "cooldown_seconds": cooldown_seconds,
            "command": batched_command,
        }
        write_state(state)
        print(
            json.dumps(
                {
                    "status": "GPU_BATCH_STARTING",
                    "stage": name,
                    "attempt": attempts,
                    "rows": before,
                    "expected_rows": expected_rows,
                    "command": batched_command,
                }
            ),
            flush=True,
        )
        returncode = 0
        try:
            subprocess.run(batched_command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            returncode = exc.returncode
        after = jsonl_count(output_path)
        progress = after - before
        state["gpu_batch"].update(
            {
                "finished_at_utc": utc_now(),
                "rows_after": after,
                "rows_added": progress,
                "returncode": returncode,
            }
        )
        write_state(state)
        if after > expected_rows:
            raise RuntimeError(f"{name} exceeds its frozen row count: {after}/{expected_rows}")
        if returncode and progress <= 0:
            consecutive_no_progress_failures += 1
        elif progress > 0:
            consecutive_no_progress_failures = 0
        elif after < expected_rows:
            consecutive_no_progress_failures += 1
        if consecutive_no_progress_failures >= max_consecutive_failures:
            raise RuntimeError(
                f"{name} made no progress in {consecutive_no_progress_failures} consecutive workers"
            )
        print(
            json.dumps(
                {
                    "status": "GPU_BATCH_RETURNED",
                    "stage": name,
                    "rows_before": before,
                    "rows_after": after,
                    "expected_rows": expected_rows,
                    "returncode": returncode,
                    "cooldown_seconds": cooldown_seconds if after < expected_rows else 0,
                }
            ),
            flush=True,
        )
        if after < expected_rows and cooldown_seconds:
            time.sleep(cooldown_seconds)


def wait_or_resume_selected(*, poll_seconds: int) -> int:
    consecutive_failures = 0
    previous_state: dict[str, Any] = {}
    if STATE.is_file():
        try:
            previous_state = json.loads(STATE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_state = {}
    resume_attempts = int(previous_state.get("selected_dev_resume_attempts", 0))
    restart_events: list[dict[str, Any]] = list(
        previous_state.get("selected_dev_restart_events", [])
    )[-20:]
    last_reported: tuple[int, int | None] | None = None
    while True:
        rows = jsonl_count(SELECTED_RUN)
        if rows == 16_200:
            write_state(
                {
                    "schema_version": "1.0",
                    "status": "SELECTED_DEV_COMPLETE",
                    "selected_dev_rows": rows,
                    "selected_dev_expected_rows": 16_200,
                    "selected_dev_resume_attempts": resume_attempts,
                    "selected_dev_restart_events": restart_events,
                }
            )
            return rows
        if rows > 16_200:
            raise RuntimeError(f"Selected DEV output exceeds its frozen row count: {rows}/16200")
        owner = live_lock_owner(SELECTED_LOCK)
        snapshot = (rows, owner)
        if snapshot != last_reported:
            status_packet = {
                "schema_version": "1.0",
                "status": "WAITING_FOR_SELECTED_DEV" if owner else "RESUMING_SELECTED_DEV",
                "selected_dev_rows": rows,
                "selected_dev_expected_rows": 16_200,
                "selected_dev_live_lock_owner": owner,
                "selected_dev_resume_attempts": resume_attempts,
                "selected_dev_consecutive_failures": consecutive_failures,
                "selected_dev_restart_events": restart_events,
            }
            write_state(status_packet)
            print(json.dumps(status_packet), flush=True)
            last_reported = snapshot
        if owner is not None:
            time.sleep(poll_seconds)
            continue
        try:
            resume_attempts += 1
            restart_events.append(
                {
                    "attempt": resume_attempts,
                    "started_at_utc": utc_now(),
                    "rows_before_resume": rows,
                    "command": selected_run_command(),
                }
            )
            restart_events[:] = restart_events[-20:]
            write_state(
                {
                    "schema_version": "1.0",
                    "status": "SELECTED_DEV_RESUME_LAUNCHED",
                    "selected_dev_rows": rows,
                    "selected_dev_expected_rows": 16_200,
                    "selected_dev_resume_attempts": resume_attempts,
                    "selected_dev_restart_events": restart_events,
                }
            )
            subprocess.run(selected_run_command(), cwd=ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            consecutive_failures += 1
            restart_events[-1]["finished_at_utc"] = utc_now()
            restart_events[-1]["outcome"] = "FAILED"
            restart_events[-1]["returncode"] = exc.returncode
            print(
                json.dumps(
                    {
                        "status": "SELECTED_DEV_RESUME_FAILED",
                        "returncode": exc.returncode,
                        "consecutive_failures": consecutive_failures,
                    }
                ),
                flush=True,
            )
            if consecutive_failures >= 3:
                raise RuntimeError("Selected DEV resume failed three consecutive times") from exc
            time.sleep(poll_seconds)
        else:
            restart_events[-1]["finished_at_utc"] = utc_now()
            restart_events[-1]["outcome"] = "COMPLETED_OR_RETURNED_CLEANLY"
            consecutive_failures = 0


def commands() -> list[tuple[str, list[str]]]:
    python = sys.executable
    return [
        (
            "analyze_selected_dev_free_run",
            [python, "experiments/phase2_mechanism/analyze_selected_free_run.py"],
        ),
        (
            "run_adaptive_dev_free_run",
            [
                python,
                "scripts/run_dev.py",
                "--config",
                "configs/phase2_selected_adaptive_dev_free_run.yaml",
                "--dataset",
                "v22_dev",
                "--device",
                "auto",
                "--precision",
                "auto",
                "--summary-every",
                "25",
            ],
        ),
        (
            "analyze_adaptive_dev_free_run",
            [python, "experiments/phase2_mechanism/analyze_adaptive_free_run.py"],
        ),
        (
            "run_selected_committee_dev_panel",
            [
                python,
                "scripts/run_dev.py",
                "--config",
                "configs/phase2_selected_committee_dev_free_run.yaml",
                "--dataset",
                "v22_dev",
                "--device",
                "auto",
                "--precision",
                "auto",
                "--summary-every",
                "25",
            ],
        ),
        (
            "analyze_selected_committee_dev_panel",
            [python, "experiments/phase2_mechanism/analyze_selected_committee_dev.py"],
        ),
        (
            "run_v22_calibrated_test_confirmation",
            [
                python,
                "scripts/run_dev.py",
                "--config",
                "configs/phase2_v22_calibrated_test_confirmation.yaml",
                "--dataset",
                "v22_sealed",
                "--device",
                "auto",
                "--precision",
                "auto",
                "--summary-every",
                "25",
            ],
        ),
        (
            "analyze_v22_calibrated_test_confirmation",
            [python, "experiments/phase2_mechanism/analyze_v22_calibrated_test.py"],
        ),
        (
            "run_cross_v212_controls",
            [
                python,
                "scripts/run_dev.py",
                "--config",
                "configs/phase2_cross_v212_controls.yaml",
                "--dataset",
                "v212_test",
                "--device",
                "auto",
                "--precision",
                "auto",
                "--summary-every",
                "25",
            ],
        ),
        (
            "run_cross_v212_full_k7",
            [
                python,
                "scripts/run_dev.py",
                "--config",
                "configs/phase2_cross_v212_full_k7.yaml",
                "--dataset",
                "v212_test",
                "--device",
                "auto",
                "--precision",
                "auto",
                "--summary-every",
                "25",
            ],
        ),
        (
            "analyze_cross_v212",
            [python, "experiments/phase2_mechanism/analyze_cross_v212.py"],
        ),
        (
            "generate_final_synthesis_fact_packet",
            [python, "experiments/phase2_mechanism/make_final_synthesis_facts.py"],
        ),
        (
            "regenerate_paper_figures",
            [python, "experiments/phase2_mechanism/make_paper_figures.py"],
        ),
        (
            "regenerate_paper_tables",
            [python, "experiments/phase2_mechanism/make_paper_tables.py"],
        ),
        (
            "capture_final_environment",
            [python, "experiments/phase2_mechanism/capture_environment.py"],
        ),
        (
            "refresh_live_report",
            [python, "experiments/phase2_mechanism/update_live_report.py"],
        ),
        (
            "run_full_tests",
            [python, "-m", "pytest", "-q"],
        ),
    ]


def write_state(packet: dict[str, Any]) -> None:
    packet["updated_at_utc"] = utc_now()
    write_json_atomic(STATE, packet)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume the frozen post-selection Phase 2 pipeline after the selected DEV run is complete."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the exact ordered commands without running them.",
    )
    parser.add_argument(
        "--wait-for-selected",
        action="store_true",
        help=(
            "Wait on a live selected-DEV PID lock; if it exits before 16,200 rows, "
            "resume the exact frozen run before advancing."
        ),
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument(
        "--gpu-batch-rows",
        type=int,
        default=100,
        help="Maximum newly durable rows per fresh GPU worker.",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=int,
        default=15,
        help="GPU-free rest interval between bounded workers.",
    )
    parser.add_argument("--max-consecutive-gpu-failures", type=int, default=3)
    args = parser.parse_args()
    if args.poll_seconds < 5 or args.poll_seconds > 60:
        raise ValueError("--poll-seconds must be between 5 and 60")
    if args.gpu_batch_rows <= 0:
        raise ValueError("--gpu-batch-rows must be positive")
    if args.cooldown_seconds < 0 or args.cooldown_seconds > 600:
        raise ValueError("--cooldown-seconds must be between 0 and 600")
    if args.max_consecutive_gpu_failures <= 0:
        raise ValueError("--max-consecutive-gpu-failures must be positive")
    ordered = commands()
    if args.list:
        print(json.dumps([{"step": name, "command": command} for name, command in ordered], indent=2))
        return

    selected_rows = (
        wait_or_resume_selected(poll_seconds=args.poll_seconds)
        if args.wait_for_selected
        else jsonl_count(SELECTED_RUN)
    )
    if selected_rows != 16_200:
        raise RuntimeError(
            f"Selected DEV run is not complete: {selected_rows}/16200 rows. "
            "Resume configs/phase2_selected_dev_free_run.yaml first."
        )
    state: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "RUNNING",
        "started_at_utc": utc_now(),
        "python": sys.executable,
        "selected_dev_rows": selected_rows,
        "completed_steps": [],
        "current_step": None,
        "failure": None,
        "final_boundary": (
            "Analyses, figures, tables, environment, live report, and tests only. "
            "Update the synthesis/claims from actual outputs, run "
            "verify_final_synthesis.py, then run audit_phase2_completion.py manually."
        ),
    }
    write_state(state)
    for name, command in ordered:
        state["current_step"] = name
        state["command"] = command
        write_state(state)
        print(json.dumps({"status": "STARTING", "step": name, "command": command}), flush=True)
        try:
            if name in GPU_STAGES:
                output_path, expected_rows = GPU_STAGES[name]
                run_gpu_stage_batched(
                    name=name,
                    command=command,
                    output_path=output_path,
                    expected_rows=expected_rows,
                    batch_rows=args.gpu_batch_rows,
                    cooldown_seconds=args.cooldown_seconds,
                    max_consecutive_failures=args.max_consecutive_gpu_failures,
                    state=state,
                )
            else:
                subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            state["status"] = "FAILED_RESUMABLE"
            state["failure"] = {"step": name, "returncode": exc.returncode}
            write_state(state)
            raise
        state["completed_steps"].append(name)
        state["current_step"] = None
        state["command"] = None
        write_state(state)
    state["status"] = "ANALYSES_COMPLETE_PENDING_FINAL_SYNTHESIS_AND_AUDIT"
    write_state(state)
    print(json.dumps({"status": state["status"], "state": str(STATE)}, indent=2))


if __name__ == "__main__":
    main()

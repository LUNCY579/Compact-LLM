from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PIPELINE_STATE = ROOT / "state/phase2_remaining_pipeline.json"
RUN = ROOT / "results/runs/phase2_selected_dev_free_run_dev_full.jsonl"
OUTPUT = ROOT / "artifacts/phase2_mechanism/selected_dev_resume_history.json"
EXPECTED_ROWS = 16_200


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_atomic(path: Path, packet: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def merge_once() -> dict[str, Any]:
    pipeline = load_json(PIPELINE_STATE)
    previous = load_json(OUTPUT)
    merged: dict[int, dict[str, Any]] = {
        int(event["attempt"]): event
        for event in previous.get("restart_events", [])
        if isinstance(event, dict) and event.get("attempt") is not None
    }
    for event in pipeline.get("selected_dev_restart_events", []):
        if not isinstance(event, dict) or event.get("attempt") is None:
            continue
        attempt = int(event["attempt"])
        merged[attempt] = {**merged.get(attempt, {}), **event}
    rows = line_count(RUN)
    if rows == EXPECTED_ROWS and merged:
        final_attempt = merged[max(merged)]
        if "finished_at_utc" not in final_attempt:
            final_attempt["finished_at_utc"] = utc_now()
            final_attempt["outcome"] = "EXPECTED_ROWS_REACHED_INFERRED_FROM_COMPLETE_RAW_RUN"
            final_attempt["rows_after"] = rows
    packet = {
        "schema_version": "1.0",
        "status": (
            "SELECTED_DEV_RESUME_HISTORY_COMPLETE"
            if rows == EXPECTED_ROWS
            else "SELECTED_DEV_RESUME_HISTORY_LIVE"
        ),
        "updated_at_utc": utc_now(),
        "expected_rows": EXPECTED_ROWS,
        "observed_rows": rows,
        "pipeline_state_status_at_snapshot": pipeline.get("status"),
        "pipeline_state_updated_at_utc": pipeline.get("updated_at_utc"),
        "restart_events": [merged[key] for key in sorted(merged)],
        "restart_event_count": len(merged),
        "failed_attempt_count": sum(
            event.get("outcome") == "FAILED" for event in merged.values()
        ),
        "clean_batch_count": sum(
            event.get("outcome") == "COMPLETED_OR_RETURNED_CLEANLY"
            for event in merged.values()
        ),
        "incomplete_event_count": sum(
            "finished_at_utc" not in event for event in merged.values()
        ),
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
    }
    write_atomic(OUTPUT, packet)
    return packet


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Permanently merge the selected DEV supervisor's bounded restart history."
    )
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=5)
    args = parser.parse_args()
    if args.poll_seconds < 2 or args.poll_seconds > 60:
        raise ValueError("--poll-seconds must be between 2 and 60")
    while True:
        packet = merge_once()
        if not args.wait or packet["status"] == "SELECTED_DEV_RESUME_HISTORY_COMPLETE":
            print(
                json.dumps(
                    {
                        "status": packet["status"],
                        "observed_rows": packet["observed_rows"],
                        "restart_event_count": packet["restart_event_count"],
                        "output": str(OUTPUT),
                    },
                    indent=2,
                )
            )
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()

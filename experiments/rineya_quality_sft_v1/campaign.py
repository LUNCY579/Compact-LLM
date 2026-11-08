"""Durable one-attempt-per-stage screen budget. No automatic retry or refill."""
from pathlib import Path
import os
import time
import re
from . import EXPERIMENT
from .contract import external_output
from .integrity import file_hash, read_json, seal, write_new


def claim(plan_sha256, review_sha256, phase, arm, output, root=None):
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
           for value in (plan_sha256, review_sha256)):
        raise ValueError("Exact plan/review digests required; no namespace traversal")
    if phase not in {"qualify", "baseline", "train"} or phase == "train" and arm not in {"whole_1e-5", "response_1e-5"}:
        raise ValueError("Only one qualification/source and two fixed cells are registered")
    directory = external_output(root or "C:/Users/clclc/BusinessProject/rineya-quality-sft-records/campaigns") / plan_sha256
    directory.mkdir(parents=True, exist_ok=True)
    label = arm if phase == "train" else phase
    event = {"experiment": EXPERIMENT, "plan_sha256": plan_sha256, "review_sha256": review_sha256,
        "phase": phase, "arm": arm, "output": str(Path(output).absolute()),
        "reserved_unix": time.time(), "owner_pid": os.getpid(), "attempts_reserved": 1,
        "automatic_retry": False, "stage_outputs": "Must retain failures/partial cases; reservation is consumed even if startup fails"}
    path = directory / (label + "-reserved.json")
    write_new(path, seal(event))
    return path


def settle(reservation, returncode, output):
    reservation = Path(reservation)
    original = read_json(reservation)
    write_new(reservation.with_name(reservation.stem + "-terminal.json"), seal({
        "experiment": EXPERIMENT, "reservation_file_sha256": file_hash(reservation),
        "output": str(Path(output).absolute()), "returncode": returncode,
        "status": "COMPLETED" if returncode == 0 else "FAILED_OR_INCOMPLETE",
        "settled_unix": time.time(), "reserved_event_sha256": original["sha256"],
        "reservation_refunded": False, "automatic_retry": False}))

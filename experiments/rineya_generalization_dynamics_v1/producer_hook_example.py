"""Prospective producer handoff example; importing this file performs no work.

The producer owner must approve the plan and wire this function only after its
existing immutable export returns successfully. This module does not save,
train, evaluate, convert, delete or prune checkpoints.
"""
from __future__ import annotations

from pathlib import Path

from .common import file_hash, read, require
from .integration import record_snapshot
from .protected_paths import reject_protected_path


def after_completed_save(*, approval_plan: dict, receipt_directory: str,
                         checkpoint_directory: str, step: int, kind: str) -> dict:
    """Record an already completed, owner-approved producer save.

    Example owner wiring (prospective approval required):

        existing_export_to_unique_step_directory(checkpoint_path)
        entry = after_completed_save(
            approval_plan={"path": approved_plan_path, "sha256": approved_hash},
            receipt_directory=gd_receipt_directory,
            checkpoint_directory=checkpoint_path, step=global_step, kind="peft",
        )
        # Retain entry, original weights, source/filter/consume event payloads,
        # exact tokenizer, approved plan and eventual completion receipt.

    Call only after the producer has finished all shards and save metadata.
    The callback cannot prove upstream authorization or save ordering itself.
    """
    plan_path = Path(approval_plan["path"]).expanduser()
    require(plan_path.is_absolute()
            and not plan_path.is_symlink(), "Use the retained absolute approval plan")
    plan_path = reject_protected_path(plan_path, "Producer approval plan")
    require(plan_path.is_file(), "Use the retained absolute approval plan")
    require(file_hash(plan_path) == approval_plan["sha256"], "Approved plan changed")
    plan = read(plan_path)
    require(plan["status"] == "APPROVED_PROSPECTIVE" and plan["purpose"] == "research_training",
            "Owner prospective research approval required")
    producer = plan["producer"]
    require(producer["independently_approved"] is True
            and producer["artifact_role"] == "original_training_trajectory",
            "Qualification updates and derived exports are not trajectories")
    require(plan["retention"] == "all_original_snapshots_and_provenance",
            "Original payload retention must be approved before saves")
    require(type(step) is int and step > 0 and kind in {"hf", "peft"},
            "Use the producer's actual positive save step and original format")
    checkpoint = Path(checkpoint_directory).expanduser()
    require(checkpoint.is_absolute() and not checkpoint.is_symlink(),
            "Stable absolute checkpoint directory required")
    checkpoint = reject_protected_path(checkpoint, "Producer checkpoint")
    receipt_root = Path(__file__).resolve().parents[3] / "gd001-records"
    receipt_path = Path(receipt_directory).expanduser()
    require(receipt_path.is_absolute(), "Snapshot receipts need an absolute external path")
    receipt_path = reject_protected_path(receipt_path, "Producer snapshot receipts")
    require(receipt_path.is_absolute() and receipt_path.resolve().is_relative_to(receipt_root)
            and all(part.lower() != ".codex" for part in receipt_path.resolve().parts),
            "Snapshot receipts must stay in external BusinessProject/gd001-records")
    schedule = plan["checkpoint_schedule"]
    require(all(Path(row["path"]).is_absolute() for row in schedule),
            "All prospectively approved checkpoint paths must be absolute")
    steps = [row["step"] for row in schedule]
    require(all(type(value) is int and value > 0 for value in steps)
            and steps == sorted(set(steps)), "Prospective save steps must strictly increase")
    matching = [row for row in schedule if row["step"] == step]
    require(len(matching) == 1 and matching[0]["kind"] == kind
            and Path(matching[0]["path"]).resolve() == checkpoint,
            "Save path/step/format differ from prospective schedule")
    planned_paths = [str(Path(row["path"]).resolve()) for row in schedule]
    require(len(set(planned_paths)) == len(planned_paths), "A rolling export path is not stable")
    require(str(checkpoint) != str(Path(plan["base"]["path"]).resolve()),
            "Do not save into the exact base export")
    # This is deliberately AFTER the producer's existing save; it never calls it.
    receipt = record_snapshot(receipt_directory, str(checkpoint), step, kind)
    receipt_path = Path(receipt_directory).resolve() / (receipt["entry"]["id"] + ".json")
    return {"step": step, "kind": kind, "path": receipt["asset"]["path"],
            "content_sha256": receipt["asset"]["content_sha256"],
            "snapshot_receipt": {"path": str(receipt_path), "sha256": file_hash(receipt_path)}}

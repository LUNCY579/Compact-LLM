from __future__ import annotations

"""Materialize reviewed Phase 9B records into leakage-safe training/evaluation data."""

import argparse
from collections import Counter
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.experiment_utils import read_jsonl, sha256, write_json
from scripts.phase_9_student_distillation.phase9_common import assert_no_group_leakage, connected_component_groups, split_by_group
from scripts.phase_9b_memory_reconstruction.phase9b_common import (
    PIPELINE_REVISION,
    reconstruction_preference_row,
    reconstruction_training_rows,
    validate_reconstruction_record,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 9B train/dev/test data from reviewed rows.")
    parser.add_argument("--reviewed-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--minimum-reviewed-rows", type=int, default=30)
    parser.add_argument("--minimum-connected-groups", type=int, default=8)
    parser.add_argument("--minimum-decision-classes", type=int, default=4)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--require-research-ready", action="store_true")
    return parser.parse_args()


def _selected_target(row: dict[str, Any]) -> dict[str, Any]:
    explicit = row.get("approved_reconstruction")
    if isinstance(explicit, dict):
        return explicit
    selected = str(row.get("selected_reconstruction_source") or "").strip().lower()
    if selected == "teacher" and isinstance(row.get("teacher_reconstruction"), dict):
        return dict(row["teacher_reconstruction"])
    if selected == "provisional" and isinstance(row.get("provisional_reconstruction"), dict):
        return dict(row["provisional_reconstruction"])
    raise ValueError(
        f"{row.get('id')} is accepted but has no approved_reconstruction or valid selected_reconstruction_source"
    )


def normalize_reviewed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        status = str(raw.get("review_status") or "").strip().lower()
        if status not in {"accepted", "approved"}:
            continue
        row = {**raw, "approved_reconstruction": _selected_target(raw)}
        normalized = validate_reconstruction_record(row, require_reviewed=True)
        if normalized["id"] in seen:
            raise ValueError(f"Duplicate reviewed Phase 9B id: {normalized['id']}")
        seen.add(normalized["id"])
        accepted.append(normalized)
    if not accepted:
        raise ValueError("No accepted Phase 9B rows were found")
    return accepted


def _write_split_files(output_dir: Path, splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    for split_name, canonical_rows in splits.items():
        canonical_path = output_dir / f"{split_name}.jsonl"
        write_jsonl(canonical_path, canonical_rows)
        sft_rows = [training for row in canonical_rows for training in reconstruction_training_rows(row)]
        sft_path = output_dir / f"{split_name}_combined.jsonl"
        write_jsonl(sft_path, sft_rows)
        reconstruct_path = output_dir / f"{split_name}_reconstruction.jsonl"
        answer_path = output_dir / f"{split_name}_answer.jsonl"
        direct_path = output_dir / f"{split_name}_direct.jsonl"
        write_jsonl(reconstruct_path, [row for row in sft_rows if row["task_type"] == "memory_reconstruction"])
        write_jsonl(answer_path, [row for row in sft_rows if row["task_type"] == "answer_from_reconstruction"])
        write_jsonl(direct_path, [row for row in sft_rows if row["task_type"] == "direct_memory_distillation"])
        preference_path = output_dir / f"preferences_{split_name}.jsonl"
        write_jsonl(preference_path, [reconstruction_preference_row(row) for row in canonical_rows])
        manifest[split_name] = {
            "canonical": str(canonical_path),
            "canonical_rows": len(canonical_rows),
            "combined": str(sft_path),
            "combined_rows": len(sft_rows),
            "reconstruction": str(reconstruct_path),
            "answer": str(answer_path),
            "direct": str(direct_path),
            "preferences": str(preference_path),
        }
    return manifest


def main() -> None:
    args = parse_args()
    if args.smoke_test and args.require_research_ready:
        raise ValueError("Choose either --smoke-test or --require-research-ready")
    reviewed = normalize_reviewed(read_jsonl(args.reviewed_input))
    groups = connected_component_groups(reviewed)
    decisions = Counter(row["approved_reconstruction"]["decision"] for row in reviewed)
    research_ready = (
        len(reviewed) >= args.minimum_reviewed_rows
        and len(groups) >= args.minimum_connected_groups
        and len(decisions) >= args.minimum_decision_classes
    )
    if args.require_research_ready and not research_ready:
        raise ValueError(
            "Phase 9B packet is not research-ready: "
            f"rows={len(reviewed)}/{args.minimum_reviewed_rows}, "
            f"groups={len(groups)}/{args.minimum_connected_groups}, "
            f"decision_classes={len(decisions)}/{args.minimum_decision_classes}"
        )
    if not research_ready and not args.smoke_test:
        raise ValueError(
            "Phase 9B packet does not meet research minima. Use --smoke-test only for code-path validation."
        )

    splits = split_by_group(
        reviewed,
        seed=args.seed,
        train_fraction=args.train_fraction,
        dev_fraction=args.dev_fraction,
    )
    assert_no_group_leakage(splits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = _write_split_files(args.output_dir, splits)
    payload = {
        "schema_version": "1.0",
        "pipeline_revision": PIPELINE_REVISION,
        "reviewed_input": str(args.reviewed_input),
        "reviewed_input_sha256": sha256(args.reviewed_input),
        "accepted_rows": len(reviewed),
        "connected_groups": len(groups),
        "decision_counts": dict(sorted(decisions.items())),
        "research_ready": research_ready,
        "smoke_test": bool(args.smoke_test),
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "dev_fraction": args.dev_fraction,
        "splits": split_manifest,
    }
    write_json(args.output_dir / "dataset_manifest.json", payload)
    (args.output_dir / "README.md").write_text(
        "# Phase 9B materialized dataset\n\n"
        f"Accepted rows: {len(reviewed)}  \n"
        f"Connected groups: {len(groups)}  \n"
        f"Decision counts: `{json.dumps(dict(decisions), sort_keys=True)}`  \n"
        f"Research ready: `{research_ready}`  \n\n"
        "Each canonical row yields three SFT records: explicit reconstruction, answer-from-guidance, "
        "and direct single-pass memory distillation. Preference files contrast approved reconstruction "
        "with a naive replay baseline.\n",
        encoding="utf-8",
    )
    print(args.output_dir / "dataset_manifest.json")


if __name__ == "__main__":
    main()

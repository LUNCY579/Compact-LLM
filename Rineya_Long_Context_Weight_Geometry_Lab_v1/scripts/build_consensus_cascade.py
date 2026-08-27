#!/usr/bin/env python3
"""Build a deterministic answer-blind UNKNOWN/consensus prediction cascade."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

PredictionRow = dict[str, Any]
PredictionRows = list[PredictionRow]
OverrideRows = tuple[str, PredictionRows]
ConsensusRows = tuple[str, PredictionRows, str, PredictionRows]

METHOD = "unknown_override_then_ordered_non_unknown_consensus"


def normalize(text: str) -> str:
    """Apply the Demo suite's canonical answer normalization."""

    value = str(text).strip().lower()
    value = re.sub(r"^\s*(final\s+answer|answer)\s*:\s*", "", value)
    value = re.sub(r"[“”\"'`]", "", value)
    value = re.sub(r"[^a-z0-9\s-]", " ", value)
    value = re.sub(r"\b(the|a|an)\b", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _validate_rows(rows: PredictionRows, source: str) -> None:
    if not rows:
        raise ValueError(f"{source}: no prediction rows")

    seen: set[str] = set()
    for row_index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or "id" not in row or "prediction" not in row:
            raise ValueError(
                f"{source}: row {row_index}: expected id and prediction fields"
            )
        row_id = str(row["id"])
        if row_id in seen:
            raise ValueError(f"{source}: row {row_index}: duplicate id {row_id}")
        status = str(row.get("status", "ok"))
        if status != "ok":
            raise ValueError(
                f"{source}: row {row_index}: non-ok prediction for {row_id}: "
                f"{status}"
            )
        seen.add(row_id)


def load_jsonl(path: Path) -> PredictionRows:
    """Load and validate one complete prediction JSONL file."""

    rows: PredictionRows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise TypeError(f"{path}:{line_no}: expected a JSON object")
        rows.append(row)
    _validate_rows(rows, str(path))
    return rows


def _align_rows(
    primary_rows: PredictionRows,
    candidate_rows: PredictionRows,
    label: str,
) -> PredictionRows:
    primary_ids = [str(row["id"]) for row in primary_rows]
    candidate_by_id = {str(row["id"]): row for row in candidate_rows}
    if set(candidate_by_id) != set(primary_ids) or len(candidate_rows) != len(
        primary_rows
    ):
        raise ValueError(f"{label}: prediction ids do not match primary ids")

    aligned = [candidate_by_id[row_id] for row_id in primary_ids]
    primary_statuses = [str(row.get("status", "ok")) for row in primary_rows]
    candidate_statuses = [str(row.get("status", "ok")) for row in aligned]
    if candidate_statuses != primary_statuses:
        raise ValueError(f"{label}: prediction statuses do not match primary statuses")
    return aligned


def _validate_labels(
    unknown_overrides: list[OverrideRows], consensus_pairs: list[ConsensusRows]
) -> None:
    labels = ["primary"]
    labels.extend(label for label, _ in unknown_overrides)
    for left_label, _, right_label, _ in consensus_pairs:
        labels.extend((left_label, right_label))
    duplicate_labels = sorted(
        label for label, count in Counter(labels).items() if count > 1
    )
    if duplicate_labels:
        raise ValueError(f"Duplicate arm labels: {duplicate_labels}")


def build_cascade(
    primary_rows: PredictionRows,
    unknown_override_rows: list[OverrideRows],
    consensus_pairs: list[ConsensusRows],
) -> tuple[PredictionRows, dict[str, Any]]:
    """Apply the fixed answer-blind cascade in input and row order.

    UNKNOWN overrides run first. Ordered consensus pairs then replace a current
    non-UNKNOWN answer only when both arms normalize to the same non-empty,
    non-UNKNOWN answer. A selected consensus copies the left arm's raw text.
    """

    _validate_rows(primary_rows, "primary")
    _validate_labels(unknown_override_rows, consensus_pairs)

    aligned_overrides: list[OverrideRows] = []
    for label, rows in unknown_override_rows:
        _validate_rows(rows, label)
        aligned_overrides.append((label, _align_rows(primary_rows, rows, label)))
    aligned_pairs: list[ConsensusRows] = []
    for left_label, left_rows, right_label, right_rows in consensus_pairs:
        _validate_rows(left_rows, left_label)
        _validate_rows(right_rows, right_label)
        aligned_pairs.append(
            (
                left_label,
                _align_rows(primary_rows, left_rows, left_label),
                right_label,
                _align_rows(primary_rows, right_rows, right_label),
            )
        )

    override_counts = {label: 0 for label, _ in aligned_overrides}
    pair_counts = {
        f"consensus_{index}": 0
        for index, _ in enumerate(aligned_pairs, 1)
    }
    result: PredictionRows = []

    for row_index, primary in enumerate(primary_rows):
        current_prediction = str(primary["prediction"])
        current_arm = "primary"
        trace: list[dict[str, Any]] = [
            {
                "stage": "primary",
                "arm": "primary",
                "selected": True,
                "normalized_prediction": normalize(current_prediction),
            }
        ]

        for label, rows in aligned_overrides:
            prediction = str(rows[row_index]["prediction"])
            normalized_prediction = normalize(prediction)
            selected = normalized_prediction == "unknown"
            if selected:
                current_prediction = prediction
                current_arm = label
                override_counts[label] += 1
            trace.append(
                {
                    "stage": "unknown_override",
                    "arm": label,
                    "selected": selected,
                    "reason": "normalized_unknown" if selected else "not_unknown",
                    "normalized_prediction": normalized_prediction,
                }
            )

        for pair_index, (
            left_label,
            left_rows,
            right_label,
            right_rows,
        ) in enumerate(aligned_pairs, 1):
            current_normalized = normalize(current_prediction)
            left_prediction = str(left_rows[row_index]["prediction"])
            right_prediction = str(right_rows[row_index]["prediction"])
            left_normalized = normalize(left_prediction)
            right_normalized = normalize(right_prediction)
            selected = False
            reason: str

            if current_normalized == "unknown":
                reason = "current_unknown"
            elif left_normalized != right_normalized:
                reason = "disagreement"
            elif left_normalized == "unknown":
                reason = "unknown_agreement_rejected"
            elif not left_normalized:
                reason = "empty_agreement_rejected"
            else:
                selected = True
                reason = "non_unknown_agreement"
                current_prediction = left_prediction
                current_arm = left_label
                pair_counts[f"consensus_{pair_index}"] += 1

            trace.append(
                {
                    "stage": "consensus",
                    "index": pair_index,
                    "left_arm": left_label,
                    "right_arm": right_label,
                    "selected": selected,
                    "selected_arm": left_label if selected else None,
                    "reason": reason,
                    "left_normalized_prediction": left_normalized,
                    "right_normalized_prediction": right_normalized,
                }
            )

        result.append(
            {
                "id": str(primary["id"]),
                "status": "ok",
                "prediction": current_prediction,
                "cascade_method": METHOD,
                "selected_arm": current_arm,
                "selection_trace": trace,
            }
        )

    final_counts = dict(sorted(Counter(row["selected_arm"] for row in result).items()))
    summary = {
        "rows": len(result),
        "unknown_override_selections": override_counts,
        "consensus_pair_selections": pair_counts,
        "final_selected_arm_counts": final_counts,
    }
    return result, summary


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _input_manifest(label: str, path: Path, rows: PredictionRows) -> dict[str, Any]:
    return {
        "label": label,
        "path": str(path),
        "sha256": sha256(path),
        "rows": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument(
        "--unknown-override",
        type=Path,
        action="append",
        default=[],
        help="Prediction file whose normalized UNKNOWN outputs override the primary.",
    )
    parser.add_argument(
        "--consensus-pair",
        type=Path,
        nargs=2,
        action="append",
        default=[],
        metavar=("LEFT", "RIGHT"),
        help="Ordered pair whose agreeing non-UNKNOWN answer may replace current.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    primary_path = args.primary.resolve(strict=True)
    override_paths = [path.resolve(strict=True) for path in args.unknown_override]
    pair_paths = [
        (left.resolve(strict=True), right.resolve(strict=True))
        for left, right in args.consensus_pair
    ]
    output_path = args.output.resolve()

    input_paths = [primary_path, *override_paths]
    input_paths.extend(path for pair in pair_paths for path in pair)
    if output_path in input_paths:
        raise ValueError("Output path must not overwrite an input prediction file")

    primary_rows = load_jsonl(primary_path)
    overrides: list[OverrideRows] = []
    override_manifests: list[dict[str, Any]] = []
    for index, path in enumerate(override_paths, 1):
        label = f"unknown_override_{index}"
        rows = load_jsonl(path)
        overrides.append((label, rows))
        override_manifests.append(_input_manifest(label, path, rows))

    pairs: list[ConsensusRows] = []
    pair_manifests: list[dict[str, Any]] = []
    for index, (left_path, right_path) in enumerate(pair_paths, 1):
        left_label = f"consensus_{index}_left"
        right_label = f"consensus_{index}_right"
        left_rows = load_jsonl(left_path)
        right_rows = load_jsonl(right_path)
        pairs.append((left_label, left_rows, right_label, right_rows))
        pair_manifests.append(
            {
                "index": index,
                "left": _input_manifest(left_label, left_path, left_rows),
                "right": _input_manifest(right_label, right_path, right_rows),
            }
        )

    result, summary = build_cascade(primary_rows, overrides, pairs)
    atomic_write(
        output_path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result),
    )
    manifest = {
        "schema_version": "1.0",
        "method": METHOD,
        "answer_key_accessed": False,
        "primary": _input_manifest("primary", primary_path, primary_rows),
        "unknown_overrides": override_manifests,
        "consensus_pairs": pair_manifests,
        "output": {
            "path": str(output_path),
            "sha256": sha256(output_path),
            "rows": len(result),
        },
        "summary": summary,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    atomic_write(manifest_path, json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "manifest": str(manifest_path),
                **summary,
            }
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Replay or route prediction arms for a published Rineya family system."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def normalize(text: str) -> str:
    value = str(text).strip().lower()
    value = re.sub(r"^\s*(final\s+answer|answer)\s*:\s*", "", value)
    value = re.sub(r"[“”\"'`]", "", value)
    value = re.sub(r"[^a-z0-9\s-]", " ", value)
    value = re.sub(r"\b(the|a|an)\b", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if (
            not isinstance(value, dict)
            or "id" not in value
            or "prediction" not in value
        ):
            raise ValueError(f"{path}:{line_number}: expected id and prediction")
        row_id = str(value["id"])
        if row_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate id {row_id}")
        if str(value.get("status", "ok")) != "ok":
            raise ValueError(f"{path}:{line_number}: non-ok row {row_id}")
        seen.add(row_id)
        rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no prediction rows")
    return rows


def align_rows(
    primary: list[dict[str, Any]], candidate: list[dict[str, Any]], label: str
) -> list[dict[str, Any]]:
    ids = [str(row["id"]) for row in primary]
    by_id = {str(row["id"]): row for row in candidate}
    if set(ids) != set(by_id) or len(ids) != len(candidate):
        raise ValueError(f"{label}: prediction ids differ from the initial arm")
    return [by_id[row_id] for row_id in ids]


def route_predictions(
    system: dict[str, Any], arm_rows: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    routing = system["routing"]
    initial = str(routing["initial_arm"])
    if initial not in arm_rows:
        raise KeyError(f"Missing initial arm: {initial}")
    primary = arm_rows[initial]
    aligned = {name: align_rows(primary, rows, name) for name, rows in arm_rows.items()}
    output: list[dict[str, Any]] = []
    for index, base_row in enumerate(primary):
        current = str(base_row["prediction"])
        selected_arm = initial
        trace: list[dict[str, Any]] = [
            {"op": "initial", "arm": initial, "selected": True}
        ]
        for step in routing["steps"]:
            operation = str(step["op"])
            if operation == "select_if_arm_unknown":
                arm = str(step["arm"])
                candidate = str(aligned[arm][index]["prediction"])
                selected = normalize(candidate) == "unknown"
                if selected:
                    current = candidate
                    selected_arm = arm
                trace.append({"op": operation, "arm": arm, "selected": selected})
            elif operation == "consensus_replace":
                left = str(step["left"])
                right = str(step["right"])
                left_prediction = str(aligned[left][index]["prediction"])
                right_prediction = str(aligned[right][index]["prediction"])
                left_normalized = normalize(left_prediction)
                selected = (
                    normalize(current) != "unknown"
                    and bool(left_normalized)
                    and left_normalized != "unknown"
                    and left_normalized == normalize(right_prediction)
                )
                if selected:
                    current = left_prediction
                    selected_arm = left
                trace.append(
                    {
                        "op": operation,
                        "left": left,
                        "right": right,
                        "selected": selected,
                    }
                )
            else:
                raise ValueError(f"Unsupported routing operation: {operation}")
        output.append(
            {
                "id": str(base_row["id"]),
                "status": "ok",
                "prediction": current,
                "family_system": system["system_id"],
                "selected_arm": selected_arm,
                "routing_trace": trace,
            }
        )
    return output


def canonical_prediction_hash(rows: list[dict[str, Any]]) -> str:
    payload = [
        {"id": str(row["id"]), "prediction": str(row["prediction"])} for row in rows
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def replay(system_directory: Path, split: str, output: Path | None) -> dict[str, Any]:
    system = json.loads((system_directory / "SYSTEM.json").read_text(encoding="utf-8"))
    evidence = system["evidence"][split]
    if not evidence.get("replayable", True):
        raise RuntimeError(f"{split}: bundled arm inputs are not available for replay")
    arm_rows = {
        arm: load_jsonl(system_directory / record["bundled_path"])
        for arm, record in evidence["inputs"].items()
    }
    rows = route_predictions(system, arm_rows)
    prediction_hash = canonical_prediction_hash(rows)
    if prediction_hash != evidence["canonical_prediction_sha256"]:
        raise RuntimeError(
            f"{split}: canonical prediction mismatch: {prediction_hash} != "
            f"{evidence['canonical_prediction_sha256']}"
        )
    if output is not None:
        write_jsonl(output, rows)
    return {
        "system_id": system["system_id"],
        "split": split,
        "rows": len(rows),
        "canonical_prediction_sha256": prediction_hash,
        "expected_hits": evidence["hits"],
        "items": evidence["items"],
        "verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--replay", choices=("showcase64", "stability192", "all256"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    system_directory = args.system_dir.resolve()
    if args.replay is None:
        system = json.loads((system_directory / "SYSTEM.json").read_text("utf-8"))
        print(
            json.dumps(
                {"system_id": system["system_id"], "splits": list(system["evidence"])}
            )
        )
        return 0
    print(json.dumps(replay(system_directory, args.replay, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

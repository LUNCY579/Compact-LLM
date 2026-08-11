from __future__ import annotations

"""Summarize one-factor v2.3.1 screens into a phase-diagram table."""

import argparse
import json
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase_9_student_distillation.detect_grokking_transitions_v231 import temporal_summary
from scripts.phase_9_student_distillation.grokking_utils_v231 import read_json, write_csv, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--factor-manifest", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    args = ap.parse_args()
    cfg = read_json(args.config)
    transition_cfg = cfg["transition_detection"]
    manifest = read_json(args.factor_manifest)
    rows: list[dict[str, Any]] = []
    for record in manifest.get("records") or []:
        curve = Path(str(record.get("time_curve") or ""))
        if not curve.exists():
            continue
        summary = temporal_summary(read_json(curve).get("rows") or [], transition_cfg)
        rows.append({"axis": record.get("axis"), "value": record.get("value"), **summary, "time_curve": str(curve)})
    payload = {"schema_version": "2.3.1", "sealed_data_used": False, "rows": rows}
    write_json(args.output_json, payload)
    write_csv(args.output_csv, rows)
    md = [
        "# v2.3.1 Factor Phase Diagram",
        "",
        "| Axis | Value | Train gain | OOD gain | Train step | OOD step | Delay | Grokking-like | Final OOD |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['axis']} | {row['value']} | {row.get('train_observed_gain')} | {row.get('ood_observed_gain')} | "
            f"{row.get('train_progress_step')} | {row.get('ood_progress_step')} | {row.get('ood_to_train_delay_ratio')} | "
            f"{row.get('grokking_like_under_primary_thresholds')} | {row.get('final_ood_answerable')} |"
        )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

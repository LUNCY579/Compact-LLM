#!/usr/bin/env python3
"""Run the controlled PCSM synthetic mathematical feasibility experiment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rineya_pcsm.synthetic import (  # noqa: E402
    REQUIRED_SECTION16_ARMS,
    SYNTHETIC_DISCLAIMER,
    SYNTHETIC_LABEL,
    SyntheticConfig,
    run_experiment,
    summarize_experiment,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run controlled structured-matrix PCSM tests. This is synthetic "
            "feasibility evidence only."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "synthetic.yaml",
        help="Synthetic experiment YAML configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "synthetic",
        help="Directory for raw per-seed JSON/CSV and the statistical summary.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10_000,
        help="Paired percentile bootstrap resamples (minimum: 100).",
    )
    parser.add_argument(
        "--statistics-seed",
        type=int,
        default=20_260_826,
        help="Deterministic seed for bootstrap and Monte Carlo statistics.",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _csv_value(value: object) -> object:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return value


def write_outputs(
    output_dir: Path,
    raw_records: list[dict[str, object]],
    summary: dict[str, object],
) -> tuple[Path, Path, Path]:
    """Persist raw per-seed evidence in JSON and CSV plus a summary JSON."""

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_json_path = output_dir / "raw_per_seed.json"
    raw_csv_path = output_dir / "raw_per_seed.csv"
    summary_path = output_dir / "summary.json"
    _write_json(
        raw_json_path,
        {
            "evidence_label": SYNTHETIC_LABEL,
            "disclaimer": SYNTHETIC_DISCLAIMER,
            "real_model_improvement_claim": False,
            "section_16_required_arms": list(REQUIRED_SECTION16_ARMS),
            "all_required_arms_present": set(REQUIRED_SECTION16_ARMS).issubset(
                {str(record["arm"]) for record in raw_records}
            ),
            "records": raw_records,
        },
    )
    if not raw_records:
        raise ValueError("raw_records must be non-empty")
    fieldnames = list(raw_records[0])
    with raw_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for record in raw_records:
            writer.writerow({key: _csv_value(value) for key, value in record.items()})
    _write_json(summary_path, summary)
    return raw_json_path, raw_csv_path, summary_path


def main() -> int:
    args = _parse_args()
    loaded: Any = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise TypeError("synthetic configuration must be a mapping")
    config = SyntheticConfig.from_mapping(cast(Mapping[str, object], loaded))
    records = run_experiment(config)
    raw_records = [record.to_dict() for record in records]
    summary = summarize_experiment(
        records,
        config,
        bootstrap_resamples=args.bootstrap_resamples,
        statistics_seed=args.statistics_seed,
    )
    paths = write_outputs(args.output_dir, raw_records, summary)
    print(SYNTHETIC_DISCLAIMER)
    print(
        f"Completed {len(records)} one-update arm runs across "
        f"{len(config.seeds)} independent seeds."
    )
    decisive = summary["decisive_section16_checks"]
    if not isinstance(decisive, Mapping):
        raise TypeError("decisive Section 16 summary must be a mapping")
    print(f"Section 16 decisive checks passed: {decisive['all_passed']}")
    for path in paths:
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

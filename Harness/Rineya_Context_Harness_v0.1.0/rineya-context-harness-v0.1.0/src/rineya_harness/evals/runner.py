from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rineya_harness.evals.rsf import RSFEvaluator, RSFMetrics, RSFTrial


def load_trials(path: str | Path) -> list[RSFTrial]:
    source = Path(path)
    trials: list[RSFTrial] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            trials.append(RSFTrial.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"Invalid RSF trial at {source}:{line_number}: {exc}") from exc
    return trials


def run_eval(path: str | Path) -> RSFMetrics:
    return RSFEvaluator().evaluate(load_trials(path))


def write_report(metrics: RSFMetrics, output: str | Path) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(metrics.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target

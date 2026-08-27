from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_showcase_results.py"
SPEC = importlib.util.spec_from_file_location("aggregate_showcase_results", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def report(run: str, *, covered: int, exact: float, all_four: float) -> dict:
    return {
        "runs": [
            {
                "run": run,
                "key_items": 64,
                "covered_items": covered,
                "strict_exact": exact,
                "short_response_rate_le_12_words": 1.0,
                "pack_metrics": {
                    "base_paraphrase_both_correct": 0.5,
                    "counterfactual_pair_both_correct_and_changed": 0.5,
                    "unanswerable_accuracy": 0.25,
                    "all_four_variants_correct": all_four,
                },
            }
        ]
    }


def test_complete_runs_rank_before_misleading_partial_accuracy(tmp_path: Path) -> None:
    (tmp_path / "complete_report.json").write_text(
        json.dumps(report("complete", covered=64, exact=0.625, all_four=0.125)),
        encoding="utf-8",
    )
    (tmp_path / "partial_report.json").write_text(
        json.dumps(report("partial", covered=4, exact=1.0, all_four=1.0)),
        encoding="utf-8",
    )

    rows = MODULE.collect_reports(tmp_path)

    assert [row["run"] for row in rows] == ["complete", "partial"]
    assert rows[0]["hits"] == 40
    assert rows[0]["rank_eligible"] is True
    assert rows[1]["hits"] is None
    assert rows[1]["rank_eligible"] is False


def test_duplicate_run_names_are_rejected(tmp_path: Path) -> None:
    for index in range(2):
        (tmp_path / f"{index}_report.json").write_text(
            json.dumps(report("same", covered=64, exact=0.5, all_four=0.0)),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="Duplicate run name"):
        MODULE.collect_reports(tmp_path)


def test_write_outputs_renders_complete_and_partial_hits(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "complete_report.json").write_text(
        json.dumps(report("complete", covered=64, exact=0.625, all_four=0.125)),
        encoding="utf-8",
    )
    (source / "partial_report.json").write_text(
        json.dumps(report("partial", covered=4, exact=1.0, all_four=1.0)),
        encoding="utf-8",
    )
    output = tmp_path / "output"

    MODULE.write_outputs(MODULE.collect_reports(source), output)

    markdown = (output / "SCOREBOARD.md").read_text(encoding="utf-8")
    assert "| complete | 40 |" in markdown
    assert "| partial | — |" in markdown
    assert (output / "scoreboard.csv").is_file()
    assert (output / "scoreboard.json").is_file()

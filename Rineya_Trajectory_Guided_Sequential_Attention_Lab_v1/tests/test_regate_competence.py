"""Re-gating recomputes a verdict from raw runs; it must never manufacture one."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
if str(LAB / "scripts") not in sys.path:
    sys.path.insert(0, str(LAB / "scripts"))

from regate_competence_report import FROZEN, arms_from_runs  # noqa: E402


def _run(arm: str, seed: int, holdout: float, seen: float, nll: float = 1.6) -> dict:
    return {
        "arm": arm, "seed": seed,
        "seen": {"accuracy": seen},
        "holdout": {
            "accuracy": holdout, "answer_nll": nll, "distinct_predictions": 30,
            "per_family_accuracy": {
                "direct_lookup": holdout + 0.05,
                "pointer_chase": holdout, "state_update": holdout - 0.01,
            },
        },
    }


def test_arms_are_rebuilt_from_raw_runs() -> None:
    runs = [_run("cascade", s, h, h + 0.02)
            for s, h in zip((1, 2, 3), (0.40, 0.44, 0.42))]
    arm = arms_from_runs(runs)[0]
    assert arm.arm == "cascade"
    assert arm.holdout_accuracy == pytest.approx(0.42)
    assert arm.per_seed_holdout_accuracy == [0.40, 0.44, 0.42]
    assert arm.per_family_accuracy["pointer_chase"] == pytest.approx(0.42)


def test_distinct_predictions_takes_the_worst_run_not_the_mean() -> None:
    """A single constant-predictor seed must not be averaged away."""

    runs = [_run("cascade", s, 0.40, 0.42) for s in (1, 2, 3)]
    runs[1]["holdout"]["distinct_predictions"] = 1
    assert arms_from_runs(runs)[0].distinct_predictions == 1


def test_the_frozen_thresholds_match_the_pilot_report() -> None:
    """These are registered in PHASE1_V4_PILOT_REPORT.md section 7."""

    assert FROZEN.minimum_margin_over_trivial == 0.030
    assert FROZEN.minimum_absolute_accuracy == 0.070
    assert FROZEN.maximum_answer_nll_fraction_of_uniform == 0.97
    assert FROZEN.maximum_generalization_gap == 0.10
    assert FROZEN.maximum_seed_coefficient_of_variation == 0.35
    assert FROZEN.minimum_seed_runs == 3


def _write(tmp_path: Path, runs: list[dict], verdict: str = "GO") -> Path:
    path = tmp_path / "stored.json"
    path.write_text(json.dumps({
        "schema_version": "tgsa.phase1-competence-gate.v2",
        "verdict": verdict,
        "trivial_baselines": {"uniform": 0.03125, "eval_majority": 0.039},
        "regime": {"band": "easy_nontrivial", "steps": 20000},
        # A properly measured report records the per-family null it was judged
        # against. Re-gating reads it rather than inventing one.
        "detail": {"family_null": {
            "pointer_chase": {"null": 0.0312},
            "state_update": {"null": 0.0312},
        }},
        "runs": runs,
    }), encoding="utf-8")
    return path


def _regate(tmp_path: Path, source: Path) -> tuple[int, dict | None]:
    import subprocess

    output = tmp_path / "regated.json"
    completed = subprocess.run(
        [sys.executable, str(LAB / "scripts" / "regate_competence_report.py"),
         "--input", str(source), "--output", str(output)],
        capture_output=True, text=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else None
    return completed.returncode, payload


def test_a_competent_report_regates_to_go(tmp_path: Path) -> None:
    runs = [_run(a, s, h, h + 0.02)
            for a in ("cascade", "history_control", "learned_delta")
            for s, h in zip((1, 2, 3), (0.40, 0.44, 0.42))]
    code, payload = _regate(tmp_path, _write(tmp_path, runs))
    assert code == 0
    assert payload["verdict"] == "GO"
    assert payload["schema_version"] == "tgsa.phase1-competence-gate.v3"


def test_a_stored_go_cannot_survive_raw_runs_that_fail(tmp_path: Path) -> None:
    """The stored verdict is not evidence. This is the whole safety property."""

    runs = [_run(a, s, h, h + 0.02)
            for a in ("cascade", "history_control", "learned_delta")
            for s, h in zip((1, 2, 3), (0.40, 0.44, 0.42))]
    runs[0]["holdout"]["accuracy"] = 0.04  # one run collapses to the floor
    source = _write(tmp_path, runs, verdict="GO")
    code, payload = _regate(tmp_path, source)
    assert code == 1
    assert payload["verdict"] == "NO_GO"
    assert payload["derived_from"]["original_verdict"] == "GO"


def test_a_report_without_raw_runs_is_refused(tmp_path: Path) -> None:
    import subprocess

    path = tmp_path / "stored.json"
    path.write_text(json.dumps({
        "schema_version": "tgsa.phase1-competence-gate.v2", "verdict": "GO",
        "trivial_baselines": {"uniform": 0.03125}, "regime": {},
    }), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(LAB / "scripts" / "regate_competence_report.py"),
         "--input", str(path), "--output", str(tmp_path / "out.json")],
        capture_output=True, text=True,
    )
    assert completed.returncode != 0
    assert "not evidence" in completed.stderr


def test_the_input_is_not_modified(tmp_path: Path) -> None:
    runs = [_run(a, s, h, h + 0.02)
            for a in ("cascade", "history_control", "learned_delta")
            for s, h in zip((1, 2, 3), (0.40, 0.44, 0.42))]
    source = _write(tmp_path, runs)
    before = source.read_bytes()
    _regate(tmp_path, source)
    assert source.read_bytes() == before


def test_the_source_is_hash_bound_into_the_output(tmp_path: Path) -> None:
    import hashlib

    runs = [_run(a, s, h, h + 0.02)
            for a in ("cascade", "history_control", "learned_delta")
            for s, h in zip((1, 2, 3), (0.40, 0.44, 0.42))]
    source = _write(tmp_path, runs)
    _, payload = _regate(tmp_path, source)
    assert payload["derived_from"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_a_stored_report_without_a_recorded_null_fails_closed(tmp_path: Path) -> None:
    """v4-era reports were judged against a global baseline and recorded no
    per-family null. Re-gating one must not invent the comparison it lacked."""

    runs = [_run(a, s, h, h + 0.02)
            for a in ("cascade", "history_control", "learned_delta")
            for s, h in zip((1, 2, 3), (0.40, 0.44, 0.42))]
    source = _write(tmp_path, runs)
    payload = json.loads(source.read_text(encoding="utf-8"))
    del payload["detail"]
    source.write_text(json.dumps(payload), encoding="utf-8")

    code, regated = _regate(tmp_path, source)
    assert code == 1
    assert regated["verdict"] == "NO_GO"
    assert regated["checks"]["every_primary_family_beats_its_own_null"] is False

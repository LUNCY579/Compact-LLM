"""Reading a seed count off a simulated surface, at the realised variance.

The property that matters is the ORDERING: the whole surface is simulated first,
and the realised sigma_a selects a cell from it afterwards. Re-running the audit
after seeing the variance would invite choosing a grid that gives a convenient
answer, so the reader must refuse to interpolate onto a cell that was never
simulated.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
READER = LAB / "scripts" / "phase1_v4_read_power_surface.py"

SEED_COUNTS = (10, 20, 30, 40)
SEED_SDS = (0.04, 0.08)
EFFECTS = (0.0, 0.02)


def _condition(seeds: int, effect: float, sd: float, rate: float) -> dict:
    return {
        "design": {"seed_count": seeds, "true_effect": effect, "seed_sd": sd},
        "gate_lower_bound_above_zero": {
            "rate": rate,
            "ci_low": max(0.0, rate - 0.01),
            "ci_high": min(1.0, rate + 0.01),
        },
    }


def _surface(tmp_path: Path, powers: dict[tuple[int, float], float],
             null_rate: float = 0.024) -> Path:
    conditions = []
    for seeds in SEED_COUNTS:
        for sd in SEED_SDS:
            conditions.append(_condition(seeds, 0.02, sd, powers[(seeds, sd)]))
            conditions.append(_condition(seeds, 0.0, sd, null_rate))
    path = tmp_path / "surface.json"
    path.write_text(json.dumps({"conditions": conditions}), encoding="utf-8")
    return path


def _read(surface: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(READER), "--surface", str(surface), *extra],
        capture_output=True, text=True,
    )


def test_the_smallest_adequate_seed_count_is_selected(tmp_path: Path) -> None:
    powers = {
        (10, 0.04): 0.55, (10, 0.08): 0.35,
        (20, 0.04): 0.78, (20, 0.08): 0.52,
        (30, 0.04): 0.88, (30, 0.08): 0.66,   # first row clearing 0.80 / 0.60
        (40, 0.04): 0.94, (40, 0.08): 0.75,
    }
    result = _read(_surface(tmp_path, powers), "--central-seed-sd", "0.04")
    assert result.returncode == 0, result.stderr
    assert "RECOMMENDED SEED COUNT: 30" in result.stdout


def test_an_unpowered_surface_reports_no_adequate_count(tmp_path: Path) -> None:
    powers = {(s, sd): 0.30 for s in SEED_COUNTS for sd in SEED_SDS}
    result = _read(_surface(tmp_path, powers), "--central-seed-sd", "0.04")
    assert result.returncode == 1
    assert "NO ADEQUATE SEED COUNT" in result.stdout
    assert "must not be registered" in result.stdout


def test_an_inflated_null_rate_blocks_every_seed_count(tmp_path: Path) -> None:
    """High power is not adequacy if the gate also fires under the null."""

    powers = {(s, sd): 0.99 for s in SEED_COUNTS for sd in SEED_SDS}
    result = _read(_surface(tmp_path, powers, null_rate=0.20), "--central-seed-sd", "0.04")
    assert result.returncode == 1
    assert "NO ADEQUATE SEED COUNT" in result.stdout


def test_an_unsimulated_variance_is_refused_not_interpolated(tmp_path: Path) -> None:
    powers = {(s, sd): 0.99 for s in SEED_COUNTS for sd in SEED_SDS}
    result = _read(_surface(tmp_path, powers), "--central-seed-sd", "0.055")
    assert result.returncode != 0
    assert "not on the simulated grid" in result.stderr
    assert "a nearby cell is a different" in result.stderr


def test_the_realised_variance_is_read_from_the_competence_report(tmp_path: Path) -> None:
    powers = {
        (10, 0.04): 0.55, (10, 0.08): 0.35,
        (20, 0.04): 0.85, (20, 0.08): 0.65,
        (30, 0.04): 0.90, (30, 0.08): 0.70,
        (40, 0.04): 0.94, (40, 0.08): 0.75,
    }
    competence = tmp_path / "competence.json"
    competence.write_text(json.dumps(
        {"detail": {"realised_seed_sd_for_power_audit": 0.04}}), encoding="utf-8")
    result = _read(_surface(tmp_path, powers), "--competence", str(competence))
    assert result.returncode == 0, result.stderr
    assert "RECOMMENDED SEED COUNT: 20" in result.stdout
    assert "realised in competence.json" in result.stdout


def test_an_incomplete_competence_report_is_refused(tmp_path: Path) -> None:
    powers = {(s, sd): 0.99 for s in SEED_COUNTS for sd in SEED_SDS}
    competence = tmp_path / "competence.json"
    competence.write_text(json.dumps(
        {"detail": {"realised_seed_sd_for_power_audit": None}}), encoding="utf-8")
    result = _read(_surface(tmp_path, powers), "--competence", str(competence))
    assert result.returncode != 0
    assert "must complete before the seed count can be fixed" in result.stderr


def test_the_elevated_variance_defaults_to_twice_the_central(tmp_path: Path) -> None:
    powers = {(s, sd): 0.99 for s in SEED_COUNTS for sd in SEED_SDS}
    result = _read(_surface(tmp_path, powers), "--central-seed-sd", "0.04")
    assert "elevated seed SD  0.08" in result.stdout


def test_the_recommendation_can_be_written_out(tmp_path: Path) -> None:
    powers = {
        (10, 0.04): 0.55, (10, 0.08): 0.35,
        (20, 0.04): 0.85, (20, 0.08): 0.65,
        (30, 0.04): 0.90, (30, 0.08): 0.70,
        (40, 0.04): 0.94, (40, 0.08): 0.75,
    }
    output = tmp_path / "recommendation.json"
    result = _read(_surface(tmp_path, powers), "--central-seed-sd", "0.04",
                   "--output", str(output))
    assert result.returncode == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["recommended_seed_count"] == 20
    assert payload["central_seed_sd"] == 0.04
    assert payload["elevated_seed_sd"] == 0.08

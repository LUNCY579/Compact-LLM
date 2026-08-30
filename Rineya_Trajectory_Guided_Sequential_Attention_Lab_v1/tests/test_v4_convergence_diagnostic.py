"""The diagnostic must separate a truncated transition from a stochastic one."""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
if str(LAB / "scripts") not in sys.path:
    sys.path.insert(0, str(LAB / "scripts"))

from phase1_v4_convergence_diagnostic import (  # noqa: E402
    describe,
    split_point,
    still_descending,
)


def _curve(start: float, end: float, points: int = 11, flat_tail: bool = False):
    if flat_tail:
        body = [start + (end - start) * i / (points - 3) for i in range(points - 2)]
        return [[i * 1000, v] for i, v in enumerate(body + [end, end])]
    return [[i * 1000, start + (end - start) * i / (points - 1)] for i in range(points)]


def test_a_clear_bimodal_split_is_found() -> None:
    assert split_point([1.3, 1.5, 1.6, 2.2, 2.3]) == pytest_approx(1.9)


def pytest_approx(value: float, tol: float = 1e-9):
    class _A:
        def __eq__(self, other): return abs(other - value) < tol
        def __repr__(self): return f"~{value}"
    return _A()


def test_a_unimodal_population_has_no_split() -> None:
    """Returning None is the answer, not a failure to answer."""

    assert split_point([1.30, 1.35, 1.42, 1.48, 1.51]) is None


def test_too_few_runs_has_no_split() -> None:
    assert split_point([1.3, 2.3]) is None


def test_a_truncated_run_is_still_descending() -> None:
    assert still_descending(_curve(4.6, 2.2)) is True


def test_a_plateaued_run_is_not() -> None:
    assert still_descending(_curve(4.6, 2.2, flat_tail=True)) is False


def _report(tmp_path: Path, spec: list[tuple[str, int, float, float, bool]]) -> Path:
    runs = []
    for arm, seed, loss, holdout, descending in spec:
        runs.append({
            "arm": arm, "seed": seed, "final_train_loss": loss,
            "holdout": {"accuracy": holdout},
            "loss_curve": _curve(4.6, loss, flat_tail=not descending),
        })
    path = tmp_path / "competence.json"
    path.write_text(json.dumps({
        "runs": runs, "regime": {"train_examples": 50000, "steps": 20000},
    }), encoding="utf-8")
    return path


def test_truncation_signature_is_reported(tmp_path: Path) -> None:
    """Stalled runs still descending => the budget is the problem."""

    path = _report(tmp_path, [
        ("cascade", 1, 1.40, 0.44, False), ("cascade", 2, 2.30, 0.12, True),
        ("history_control", 1, 1.45, 0.43, False), ("history_control", 2, 2.25, 0.13, True),
        ("learned_delta", 1, 1.38, 0.45, False), ("learned_delta", 2, 1.42, 0.44, False),
    ])
    summary = describe(path)
    assert summary["transitioned_fraction"] == 4 / 6
    assert summary["stalled_still_descending"] == 2
    assert summary["stalled_plateaued"] == 0


def test_stochastic_signature_is_reported(tmp_path: Path) -> None:
    """Stalled runs plateaued => more updates will not fix it."""

    path = _report(tmp_path, [
        ("cascade", 1, 1.40, 0.44, False), ("cascade", 2, 2.60, 0.12, False),
        ("history_control", 1, 1.45, 0.43, False), ("history_control", 2, 2.55, 0.13, False),
        ("learned_delta", 1, 1.38, 0.45, False), ("learned_delta", 2, 2.62, 0.12, False),
    ])
    summary = describe(path)
    assert summary["stalled_plateaued"] == 3
    assert summary["stalled_still_descending"] == 0


def test_a_report_without_loss_is_refused(tmp_path: Path) -> None:
    import pytest

    path = tmp_path / "competence.json"
    path.write_text(json.dumps({
        "runs": [{"arm": "cascade", "seed": 1, "holdout": {"accuracy": 0.4}}],
        "regime": {},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="final_train_loss"):
        describe(path)

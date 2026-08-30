"""Resume must never fabricate a measurement it did not take.

A nine-cell competence run is four hours and has twice been lost whole — once to
GPU contention, once to a machine reboot — so per-cell checkpointing is worth
having. But a resume bug is strictly worse than losing the run: a report
assembled from cells measured under two different regimes certifies neither, and
nothing downstream would notice, because the gate reads the report and not the
cells.

So every test here is a way a wrong cell could be accepted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

LAB = Path(__file__).resolve().parents[1]
if str(LAB / "scripts") not in sys.path:
    sys.path.insert(0, str(LAB / "scripts"))

from phase1_v4_competence_run import (  # noqa: E402
    CELL_SCHEMA_VERSION,
    load_cell,
    regime_fingerprint,
    write_cell_atomically,
)

BASE = dict(
    band="easy_nontrivial", train_examples=50_000, holdout=3000, steps=20_000,
    embed_dim=96, layers=2, heads=6, batch_size=64, lr=1e-3,
    holdout_split_seed=987_654_321,
)


def _args(**overrides) -> SimpleNamespace:
    return SimpleNamespace(**{**BASE, **overrides})


def _cell(path: Path, fingerprint: str, arm: str = "cascade", seed: int = 1) -> Path:
    write_cell_atomically(path, {
        "schema_version": CELL_SCHEMA_VERSION,
        "regime_fingerprint": fingerprint,
        "run": {"arm": arm, "seed": seed, "holdout": {"accuracy": 0.42},
                "seen": {"accuracy": 0.45}},
    })
    return path


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


def test_a_matching_cell_is_reused(tmp_path: Path) -> None:
    fingerprint = regime_fingerprint(_args(), train_seed=1234)
    path = _cell(tmp_path / "cascade__seed1.json", fingerprint)
    cell = load_cell(path, fingerprint, "cascade", 1)
    assert cell is not None and cell["holdout"]["accuracy"] == 0.42


def test_an_absent_cell_is_simply_missing(tmp_path: Path) -> None:
    """Absence must be None, not an error: that is the normal first run."""

    fingerprint = regime_fingerprint(_args(), train_seed=1234)
    assert load_cell(tmp_path / "absent.json", fingerprint, "cascade", 1) is None


# --------------------------------------------------------------------------
# Every knob that changes the measurement must invalidate the cell
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field,value", [
    ("steps", 30_000),
    ("train_examples", 200_000),
    ("holdout", 6000),
    ("band", "medium"),
    ("embed_dim", 192),
    ("layers", 4),
    ("heads", 8),
    ("batch_size", 128),
    ("lr", 3e-4),
    ("holdout_split_seed", 111),
])
def test_changing_a_regime_knob_refuses_the_cell(tmp_path: Path, field, value) -> None:
    original = regime_fingerprint(_args(), train_seed=1234)
    path = _cell(tmp_path / "cascade__seed1.json", original)
    changed = regime_fingerprint(_args(**{field: value}), train_seed=1234)
    assert changed != original, f"{field} does not affect the fingerprint"
    with pytest.raises(ValueError, match="different regime"):
        load_cell(path, changed, "cascade", 1)


def test_changing_the_train_split_seed_refuses_the_cell(tmp_path: Path) -> None:
    original = regime_fingerprint(_args(), train_seed=1234)
    path = _cell(tmp_path / "cascade__seed1.json", original)
    with pytest.raises(ValueError, match="different regime"):
        load_cell(path, regime_fingerprint(_args(), train_seed=9999), "cascade", 1)


def test_adding_a_seed_does_not_invalidate_existing_cells(tmp_path: Path) -> None:
    """Seeds are excluded from the fingerprint deliberately.

    Measuring a fourth seed later must not discard the first three, and each
    cell already records which seed it is.
    """

    fingerprint = regime_fingerprint(_args(), train_seed=1234)
    path = _cell(tmp_path / "cascade__seed1.json", fingerprint)
    assert load_cell(path, fingerprint, "cascade", 1) is not None


# --------------------------------------------------------------------------
# Corruption and mislabelling
# --------------------------------------------------------------------------


def test_a_truncated_cell_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "cascade__seed1.json"
    path.write_text('{"schema_version": "tgsa.phase1-competence-cell.v1", "regime_fin',
                    encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_cell(path, regime_fingerprint(_args(), train_seed=1234), "cascade", 1)


def test_an_old_schema_is_refused(tmp_path: Path) -> None:
    fingerprint = regime_fingerprint(_args(), train_seed=1234)
    path = tmp_path / "cascade__seed1.json"
    path.write_text(json.dumps({
        "schema_version": "tgsa.phase1-competence-cell.v0",
        "regime_fingerprint": fingerprint,
        "run": {"arm": "cascade", "seed": 1},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_cell(path, fingerprint, "cascade", 1)


def test_a_cell_holding_the_wrong_arm_is_refused(tmp_path: Path) -> None:
    """Guards against a directory assembled by hand or copied between runs."""

    fingerprint = regime_fingerprint(_args(), train_seed=1234)
    path = _cell(tmp_path / "cascade__seed1.json", fingerprint, arm="learned_delta")
    with pytest.raises(ValueError, match="inconsistent"):
        load_cell(path, fingerprint, "cascade", 1)


def test_a_cell_holding_the_wrong_seed_is_refused(tmp_path: Path) -> None:
    fingerprint = regime_fingerprint(_args(), train_seed=1234)
    path = _cell(tmp_path / "cascade__seed1.json", fingerprint, seed=2)
    with pytest.raises(ValueError, match="inconsistent"):
        load_cell(path, fingerprint, "cascade", 1)


def test_a_cell_with_no_run_payload_is_refused(tmp_path: Path) -> None:
    fingerprint = regime_fingerprint(_args(), train_seed=1234)
    path = tmp_path / "cascade__seed1.json"
    path.write_text(json.dumps({
        "schema_version": CELL_SCHEMA_VERSION, "regime_fingerprint": fingerprint,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="no run payload"):
        load_cell(path, fingerprint, "cascade", 1)


# --------------------------------------------------------------------------
# Atomicity
# --------------------------------------------------------------------------


def test_the_write_leaves_no_partial_file_behind(tmp_path: Path) -> None:
    path = tmp_path / "cascade__seed1.json"
    _cell(path, regime_fingerprint(_args(), train_seed=1234))
    assert path.is_file()
    assert not list(tmp_path.glob("*.partial")), "a .partial file survived the write"


def test_an_interrupted_write_cannot_be_mistaken_for_a_cell(tmp_path: Path) -> None:
    """The partial name must not match what the runner looks for."""

    fingerprint = regime_fingerprint(_args(), train_seed=1234)
    (tmp_path / "cascade__seed1.json.partial").write_text("{}", encoding="utf-8")
    assert load_cell(tmp_path / "cascade__seed1.json", fingerprint, "cascade", 1) is None


def test_rewriting_a_cell_replaces_it_cleanly(tmp_path: Path) -> None:
    fingerprint = regime_fingerprint(_args(), train_seed=1234)
    path = tmp_path / "cascade__seed1.json"
    _cell(path, fingerprint)
    write_cell_atomically(path, {
        "schema_version": CELL_SCHEMA_VERSION, "regime_fingerprint": fingerprint,
        "run": {"arm": "cascade", "seed": 1, "holdout": {"accuracy": 0.99},
                "seen": {"accuracy": 0.99}},
    })
    assert load_cell(path, fingerprint, "cascade", 1)["holdout"]["accuracy"] == 0.99


# --------------------------------------------------------------------------
# The fingerprint itself
# --------------------------------------------------------------------------


def test_the_fingerprint_is_stable_across_processes() -> None:
    """It is a SHA-256 of canonical JSON, not builtin hash()."""

    import os
    import subprocess

    script = (
        "import sys; sys.path.insert(0, %r); "
        "from types import SimpleNamespace; "
        "from phase1_v4_competence_run import regime_fingerprint; "
        "print(regime_fingerprint(SimpleNamespace(**%r), train_seed=1234))"
        % (str(LAB / "scripts"), BASE)
    )
    values = set()
    for salt in ("0", "1", "12345"):
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True,
            check=True, env={**os.environ, "PYTHONHASHSEED": salt},
        )
        values.add(completed.stdout.strip())
    assert len(values) == 1, f"fingerprint varied with PYTHONHASHSEED: {values}"
    assert values.pop() == regime_fingerprint(_args(), train_seed=1234)


# --------------------------------------------------------------------------
# The compute backend is part of what a cell measures
# --------------------------------------------------------------------------


def test_the_backend_is_in_the_fingerprint(monkeypatch) -> None:
    """The one measurement input that comes from the environment, not the CLI.

    Twenty thousand optimizer steps through a different BLAS backend is a
    different measurement. A CPU fallback -- a wedged driver, an empty
    CUDA_VISIBLE_DEVICES, a venv without a CUDA wheel -- is silent, and the loop
    order splits an interrupted run exactly along an arm boundary, which is the
    worst possible split for a gate whose headline check is an arm-blind minimum.
    """

    import phase1_v4_competence_run as runner

    monkeypatch.setattr(runner, "compute_backend", lambda: {"device_type": "cuda", "torch": "x"})
    on_cuda = runner.regime_fingerprint(_args(), train_seed=1234)
    monkeypatch.setattr(runner, "compute_backend", lambda: {"device_type": "cpu", "torch": "x"})
    on_cpu = runner.regime_fingerprint(_args(), train_seed=1234)
    assert on_cuda != on_cpu, "a CPU cell would be reused by a CUDA run"


def test_a_cell_from_another_backend_is_refused(tmp_path: Path, monkeypatch) -> None:
    import phase1_v4_competence_run as runner

    monkeypatch.setattr(runner, "compute_backend", lambda: {"device_type": "cpu", "torch": "x"})
    cpu_print = runner.regime_fingerprint(_args(), train_seed=1234)
    path = tmp_path / "cascade__seed1.json"
    runner.write_cell_atomically(path, {
        "schema_version": runner.CELL_SCHEMA_VERSION,
        "regime_fingerprint": cpu_print,
        "backend": {"device_type": "cpu", "torch": "x"},
        "run": {"arm": "cascade", "seed": 1, "holdout": {"accuracy": 0.42},
                "seen": {"accuracy": 0.45}},
    })
    monkeypatch.setattr(runner, "compute_backend", lambda: {"device_type": "cuda", "torch": "x"})
    cuda_print = runner.regime_fingerprint(_args(), train_seed=1234)
    with pytest.raises(ValueError, match="different regime"):
        runner.load_cell(path, cuda_print, "cascade", 1)


def test_the_refusal_names_the_backend_difference(tmp_path: Path, monkeypatch) -> None:
    """A digest mismatch alone tells an operator nothing actionable."""

    import phase1_v4_competence_run as runner

    monkeypatch.setattr(runner, "compute_backend", lambda: {"device_type": "cpu", "torch": "x"})
    path = tmp_path / "cascade__seed1.json"
    runner.write_cell_atomically(path, {
        "schema_version": runner.CELL_SCHEMA_VERSION,
        "regime_fingerprint": runner.regime_fingerprint(_args(), train_seed=1234),
        "backend": {"device_type": "cpu", "torch": "x"},
        "run": {"arm": "cascade", "seed": 1, "holdout": {"accuracy": 0.42},
                "seen": {"accuracy": 0.45}},
    })
    monkeypatch.setattr(runner, "compute_backend", lambda: {"device_type": "cuda", "torch": "x"})
    with pytest.raises(ValueError, match="backend"):
        runner.load_cell(path, runner.regime_fingerprint(_args(), train_seed=1234), "cascade", 1)


def test_v1_cells_are_refused_because_they_recorded_no_backend(tmp_path: Path) -> None:
    """There is nothing to compare a v1 cell against, and guessing defeats the point."""

    import phase1_v4_competence_run as runner

    path = tmp_path / "cascade__seed1.json"
    path.write_text(json.dumps({
        "schema_version": "tgsa.phase1-competence-cell.v1",
        "regime_fingerprint": runner.regime_fingerprint(_args(), train_seed=1234),
        "run": {"arm": "cascade", "seed": 1},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        runner.load_cell(path, runner.regime_fingerprint(_args(), train_seed=1234), "cascade", 1)

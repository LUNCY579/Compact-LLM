"""Every seed in registered code must be reproducible from its own record.

Python salts `hash()` of str/bytes per process (PYTHONHASHSEED), so any seed
derived from it silently changes between runs. The power simulator did exactly
that: the same audit, run twice, drew different streams, and the recorded
numbers could not be regenerated from the recorded configuration.

That is a reproducibility defect rather than a statistical one — the estimates
stayed unbiased — but it breaks the property the evidence chain depends on, so
these tests pin it shut.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from tgsa.phase1_data import derive_seed

LAB = Path(__file__).resolve().parents[1]
REGISTERED = sorted([*(LAB / "tgsa").rglob("*.py"), *(LAB / "scripts").rglob("*.py")])

# `hash` is only unsafe when its result influences a value we must reproduce.
# Membership tests and dict/set construction are fine; seeding is not.
_ALLOWED_HASH_CALLERS: frozenset[str] = frozenset()


def _builtin_hash_calls(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "hash"
        and node.func.id not in _ALLOWED_HASH_CALLERS
    ]


@pytest.mark.parametrize("path", REGISTERED, ids=lambda p: str(p.relative_to(LAB)).replace("\\", "/"))
def test_registered_code_never_uses_builtin_hash(path: Path) -> None:
    offenders = _builtin_hash_calls(path)
    assert not offenders, (
        f"{path.relative_to(LAB)} calls builtin hash() at line(s) {offenders}; "
        "str/bytes hashing is salted per process. Use derive_seed() instead."
    )


def test_derive_seed_is_stable_across_processes() -> None:
    """The replacement must survive the salt that broke the original."""

    script = (
        "import sys; sys.path.insert(0, %r);"
        "from tgsa.phase1_data import derive_seed;"
        "print(derive_seed(4242, 'power-scenario', 'null_effect'))" % str(LAB)
    )
    values = set()
    for salt in ("0", "1", "12345"):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, check=True,
            env={**__import__("os").environ, "PYTHONHASHSEED": salt},
        )
        values.add(completed.stdout.strip())
    assert len(values) == 1, f"derive_seed varied with PYTHONHASHSEED: {values}"
    assert values.pop() == str(derive_seed(4242, "power-scenario", "null_effect"))


def test_builtin_hash_really_is_unstable() -> None:
    """Guard the guard: if this ever stops varying, the test above is vacuous."""

    import os

    values = set()
    for salt in ("0", "1", "12345"):
        completed = subprocess.run(
            [sys.executable, "-c", "print(abs(hash('null_effect')) % 10000)"],
            capture_output=True, text=True, check=True,
            env={**os.environ, "PYTHONHASHSEED": salt},
        )
        values.add(completed.stdout.strip())
    assert len(values) > 1, "builtin hash() no longer varies; revisit this module"


def test_power_simulation_is_reproducible() -> None:
    """Two identical audits must agree exactly, not merely within Monte Carlo error."""

    from tgsa.phase1_power import PowerScenario, simulate_power

    scenarios = [
        PowerScenario(name="null_effect", true_effect=0.0, seed_sd=0.02,
                      stratum_sd=0.0, disagreement_rate=0.30),
        PowerScenario(name="target_effect", true_effect=0.020, seed_sd=0.02,
                      stratum_sd=0.0, disagreement_rate=0.30),
    ]
    kwargs = dict(
        scenarios=scenarios, seeds=list(range(1, 6)), strata_count=3,
        examples_per_stratum=40, threshold=0.0, bootstrap_samples=200,
        confidence=0.95, bootstrap_scheme="crossed_paired", trials=25,
        rng_seed=4242,
    )
    first = simulate_power(**kwargs)
    second = simulate_power(**kwargs)
    assert first == second


# --------------------------------------------------------------------------
# Provenance probes must never be able to abort a run
# --------------------------------------------------------------------------


def test_environment_snapshot_survives_a_failing_probe(monkeypatch) -> None:
    """torch reports cuda_available on a CUDA build with every device masked,
    and cudnn.version() then raises. Uncaught, that killed the first cell of a
    matrix over an environment note."""

    import torch

    from tgsa.phase1_runner import environment_snapshot

    def explode() -> int:
        raise ValueError("min() arg is an empty sequence")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.cudnn, "version", explode)

    snapshot = environment_snapshot(torch.device("cpu"))
    assert isinstance(snapshot["cudnn_version"], str)
    assert "min() arg is an empty sequence" in snapshot["cudnn_version"]
    assert snapshot["device"] == "cpu"
    assert snapshot["torch"] == torch.__version__


def test_environment_snapshot_reports_a_working_probe_normally(monkeypatch) -> None:
    """The rescue must not swallow a real value."""

    import torch

    from tgsa.phase1_runner import environment_snapshot

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 91002)
    assert environment_snapshot(torch.device("cpu"))["cudnn_version"] == 91002

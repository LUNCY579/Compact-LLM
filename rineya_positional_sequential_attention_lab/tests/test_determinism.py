"""Tests for ``rineya_psa.determinism`` -- SPEC section 10, implemented as written.

SPEC 10 is a *reproducible-seeding* contract, not a bit-exactness one:

    ``torch.manual_seed``, ``numpy.default_rng(seed)``,
    ``torch.use_deterministic_algorithms(False)`` (SDPA kernels are not bit-deterministic
    on this card; we record that rather than pay the speed cost). Generation is
    **greedy, do_sample=False** everywhere.

These tests therefore assert two things that are easy to confuse:

1. Seeding *is* honoured -- same seed, same draws, from every RNG the lab uses.
2. ``use_deterministic_algorithms`` is left **False**, and that decision is *recorded*
   rather than silently inherited. A test that demanded ``True`` here would be
   enforcing a stricter contract than the one the lab signed, and would make the
   SPEC 11 compute budget unpayable.

Nothing here touches the GPU: a job owns the card while this suite runs, and
``test_seeding_does_not_initialise_cuda`` is the assertion that keeps it that way.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rineya_psa import determinism as D

SRC = str(Path(__file__).resolve().parents[1] / "src")


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter that can import the lab."""
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )


@pytest.fixture(autouse=True)
def _restore_determinism_flag():
    """Keep a test that opts into deterministic algorithms from leaking into the suite."""
    torch = pytest.importorskip("torch")
    before = torch.are_deterministic_algorithms_enabled()
    yield
    torch.use_deterministic_algorithms(before)


# --------------------------------------------------------------------------------------
# seed_everything: the seeding half of the contract
# --------------------------------------------------------------------------------------


def test_seed_everything_makes_python_numpy_and_torch_repeat():
    import random

    import torch

    def draws() -> tuple:
        rng = D.seed_everything(1234)
        return (
            random.random(),
            float(np.random.rand()),
            rng.random(),
            float(torch.rand(1)),
        )

    assert draws() == draws()


def test_a_different_seed_gives_different_draws():
    first = D.seed_everything(1).random()
    second = D.seed_everything(2).random()
    assert first != second


def test_seed_everything_returns_the_generator_spec_10_names():
    """SPEC 10 names ``numpy.default_rng(seed)``; the returned object must be exactly it."""
    rng = D.seed_everything(777)
    assert isinstance(rng, np.random.Generator)
    assert rng.random(5).tolist() == np.random.default_rng(777).random(5).tolist()


def test_seed_everything_rejects_a_seed_numpy_cannot_take():
    for bad in (-1, 2**32, 2**64):
        with pytest.raises(ValueError, match="uint32"):
            D.seed_everything(bad)


def test_the_default_seed_is_the_one_the_lab_uses():
    assert D.seed_everything().random() == np.random.default_rng(D.DEFAULT_SEED).random()


# --------------------------------------------------------------------------------------
# The recorded decision: deterministic algorithms stay OFF, on purpose
# --------------------------------------------------------------------------------------


def test_seeding_leaves_deterministic_algorithms_off_and_says_so():
    """SPEC 10's actual contract. Do not "strengthen" this to ``True``."""
    import torch

    torch.use_deterministic_algorithms(True)  # a previous caller turned it on
    D.seed_everything(5)

    assert torch.are_deterministic_algorithms_enabled() is False
    state = D.state()
    assert state.seed == 5
    assert state.deterministic_algorithms is False
    assert "torch.use_deterministic_algorithms" in state.skipped
    assert "SPEC 10" in state.skipped["torch.use_deterministic_algorithms"]
    assert "CUBLAS_WORKSPACE_CONFIG" in state.skipped


def test_a_caller_that_needs_bit_exactness_can_opt_in():
    import torch

    D.seed_everything(5, deterministic_algorithms=True)
    assert torch.are_deterministic_algorithms_enabled() is True
    assert D.state().deterministic_algorithms is True
    # The opt-in path must not claim the SPEC's exemptions apply to it.
    assert "torch.use_deterministic_algorithms" not in D.state().skipped


def test_seed_everything_records_what_it_actually_seeded():
    D.seed_everything(11)
    seeded = D.state().seeded
    assert {"python_random", "numpy_legacy_global", "numpy_default_rng"} <= set(seeded)
    assert "torch_cpu" in seeded
    # PYTHONHASHSEED cannot be set from inside a running interpreter; record, do not lie.
    assert "PYTHONHASHSEED" in D.state().skipped


# --------------------------------------------------------------------------------------
# describe(): the record that ships inside every sidecar
# --------------------------------------------------------------------------------------


def test_describe_is_json_serialisable_and_names_the_spec_section():
    payload = D.describe()
    assert payload["spec_section"] == "10"
    assert json.loads(json.dumps(payload)) == payload


def test_describe_records_the_sources_it_deliberately_does_not_fix():
    payload = D.describe()
    sources = {item["source"] for item in payload["not_deterministic"]}
    assert any("scaled_dot_product_attention" in s for s in sources)
    assert all(item["reason"] for item in payload["not_deterministic"])
    assert payload["deterministic_algorithms"]["requested_by_spec"] is False
    assert "SPEC 10" in payload["deterministic_algorithms"]["reason"]


def test_describe_records_greedy_generation():
    """SPEC 10: generation is greedy, do_sample=False everywhere."""
    assert D.describe()["generation"] == {"do_sample": False, "greedy": True, "num_beams": 1}


def test_describe_records_cublas_workspace_config_as_read_not_set():
    """DEVIATION: the env var is deliberately left alone; the record proves it."""
    env = D.describe()["env"]
    assert "CUBLAS_WORKSPACE_CONFIG" in env
    assert env["CUBLAS_WORKSPACE_CONFIG"] == os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    assert "PYTHONHASHSEED" in env


def test_describe_reflects_the_current_seed():
    D.seed_everything(4321)
    payload = D.describe()
    assert payload["seed"] == 4321
    assert payload["seed_everything_called"] is True
    assert payload["default_seed"] == D.DEFAULT_SEED


def test_describe_before_any_seeding_admits_nothing_was_seeded():
    """An unseeded run is a fact a sidecar must be able to state."""
    proc = _run_python(
        "import json; from rineya_psa import determinism as D;"
        "p = D.describe();"
        "print(json.dumps({'seed': p['seed'], 'called': p['seed_everything_called'],"
        " 'seeded': p['seeded']}))"
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["seed"] is None
    assert payload["called"] is False
    assert payload["seeded"] == []


# --------------------------------------------------------------------------------------
# SPEC 9 x SPEC 10: the record has to reach the artifact
# --------------------------------------------------------------------------------------


def test_every_provenance_stamp_carries_the_determinism_record():
    from rineya_psa.statistics import provenance as PV

    D.seed_everything(999)
    block = PV.stamp({"seed": 999})["determinism"]
    assert block["spec_section"] == "10"
    assert block["seed"] == 999
    assert block["deterministic_algorithms"]["enabled"] is False
    assert block["not_deterministic"]


def test_write_result_sidecar_carries_the_determinism_record(tmp_path: Path):
    from rineya_psa.statistics import provenance as PV

    D.seed_everything(31337)
    target = PV.write_result({"x": 1}, tmp_path / "r.json", config={"seed": 31337})
    payload = json.loads(PV.sidecar_path(target).read_text(encoding="utf-8"))
    assert payload["determinism"]["seed"] == 31337
    assert payload["determinism"]["contract"]


# --------------------------------------------------------------------------------------
# Environment discipline
# --------------------------------------------------------------------------------------


def test_seeding_does_not_initialise_cuda():
    """A job owns the card. ``torch.manual_seed`` defers its CUDA half via _lazy_call."""
    proc = _run_python(
        "import torch;"
        "from rineya_psa import determinism as D;"
        "D.seed_everything(3);"
        "D.describe();"
        "assert not torch.cuda.is_initialized(), 'seeding created a CUDA context';"
        "print('ok')"
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_determinism_and_provenance_import_without_torch():
    """SPEC section 2: neither module may import torch at module scope."""
    proc = _run_python(
        "import sys;"
        "import rineya_psa.determinism, rineya_psa.statistics.provenance;"
        "assert 'torch' not in sys.modules,"
        " sorted(m for m in sys.modules if 'torch' in m)"
    )
    assert proc.returncode == 0, proc.stderr


def test_describe_survives_an_interpreter_without_torch():
    """Analysis boxes without torch must still be able to stamp an artifact."""
    proc = _run_python(
        "import sys, json;"
        "sys.modules['torch'] = None;"  # importlib.import_module returns None -> falsy path
        "import builtins;"
        "_real = builtins.__import__\n"
        "def _blocked(name, *a, **k):\n"
        "    if name == 'torch' or name.startswith('torch.'):\n"
        "        raise ImportError('no torch')\n"
        "    return _real(name, *a, **k)\n"
        "builtins.__import__ = _blocked\n"
        "sys.modules.pop('torch', None)\n"
        "from rineya_psa import determinism as D\n"
        "p = D.describe()\n"
        "assert p['torch'] is None, p['torch']\n"
        "assert p['deterministic_algorithms']['enabled'] is None\n"
        "json.dumps(p)\n"
        "print('ok')\n"
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout

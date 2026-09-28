"""The Phase 4 chain's run scope (SCOPE.json max_tier): which steps a cap keeps, and that a bad file fails closed.
Stdlib only -- no torch -- so it is cheap to run next to a GPU job."""
from __future__ import annotations

import json

import pytest

import run_phase4


def _names(st):
    return [s["name"] for s in st]


def test_absent_scope_runs_every_tier(tmp_path):
    assert run_phase4.max_tier(tmp_path / "SCOPE.json") is None
    all_steps = run_phase4.steps()
    keep, dropped = run_phase4.in_scope(all_steps, None)
    assert keep == all_steps and dropped == []
    assert {s["tier"] for s in keep} == set(run_phase4.TIERS)


def test_max_tier_1_keeps_the_registered_first_run_only():
    all_steps = run_phase4.steps()
    keep, dropped = run_phase4.in_scope(all_steps, 1)
    assert {s["tier"] for s in keep} == {0, 1}
    assert _names(keep)[:2] == ["native-qualify", "lr-pilot"]
    trained = [n.split(" ", 1)[1] for n in _names(keep) if n.startswith("train ")]
    assert trained == ["A_dense_sequential__s0__canonical", "D_random_disjoint__s0__canonical",
                       "E_learned_disjoint__s0__canonical"]
    assert len(keep) + len(dropped) == len(all_steps)
    assert not any("__s1__" in n or "__secondary" in n or "B_independent" in n or "C_magnitude" in n for n in _names(keep))
    assert all(("__s1__" in n or "__secondary" in n or "B_independent" in n or "C_magnitude" in n) for n in dropped)


def test_cap_zero_keeps_only_qualification_and_lr_gate():
    keep, _ = run_phase4.in_scope(run_phase4.steps(), 0)
    assert _names(keep) == ["native-qualify", "lr-pilot"]


@pytest.mark.parametrize("value", [5, -1, "1", 1.0, None, True])
def test_malformed_scope_fails_closed(tmp_path, value):
    p = tmp_path / "SCOPE.json"
    p.write_bytes(json.dumps({"max_tier": value}).encode("ascii"))
    with pytest.raises(SystemExit):
        run_phase4.max_tier(p)


def test_valid_scope_file_is_read(tmp_path):
    p = tmp_path / "SCOPE.json"
    p.write_bytes(json.dumps({"max_tier": 1, "granted_by": "operator"}).encode("ascii"))
    assert run_phase4.max_tier(p) == 1

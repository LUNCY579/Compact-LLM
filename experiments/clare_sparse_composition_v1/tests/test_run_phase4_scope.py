"""Scope admission and tier restrictions. No torch, model loads or native subprocesses."""
from __future__ import annotations

import json

import pytest

import run_phase4


def _names(st):
    return [s["name"] for s in st]


def test_absent_scope_fails_closed(tmp_path):
    with pytest.raises(SystemExit, match="explicit valid SCOPE.json required"):
        run_phase4.max_tier(tmp_path / "SCOPE.json")


@pytest.mark.parametrize("raw", [b"", b"{", b"\xff", b"null", b"[]", b"1", b'"1"', b"{}"])
def test_invalid_scope_document_fails_closed(tmp_path, raw):
    p = tmp_path / "SCOPE.json"
    p.write_bytes(raw)
    with pytest.raises(SystemExit, match="SCOPE.json"):
        run_phase4.max_tier(p)


def test_unreadable_scope_fails_closed(tmp_path):
    p = tmp_path / "SCOPE.json"
    p.mkdir()
    with pytest.raises(SystemExit, match="explicit valid SCOPE.json required"):
        run_phase4.max_tier(p)


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


@pytest.mark.parametrize("value", [5, -1, "1", 1.0, None, True, False, [], {}])
def test_malformed_scope_fails_closed(tmp_path, value):
    p = tmp_path / "SCOPE.json"
    p.write_bytes(json.dumps({"max_tier": value}).encode("ascii"))
    with pytest.raises(SystemExit):
        run_phase4.max_tier(p)


@pytest.mark.parametrize("cap", run_phase4.TIERS)
def test_valid_scope_file_restricts_tiers(tmp_path, cap):
    p = tmp_path / "SCOPE.json"
    p.write_bytes(json.dumps({"max_tier": cap, "granted_by": "operator"}).encode("ascii"))
    value = run_phase4.max_tier(p)
    assert value == cap
    all_steps = run_phase4.steps()
    keep, dropped = run_phase4.in_scope(all_steps, value)
    assert {st["tier"] for st in keep} == set(range(cap + 1))
    assert _names(keep) == [st["name"] for st in all_steps if st["tier"] <= cap]
    assert dropped == [st["name"] for st in all_steps if st["tier"] > cap]


@pytest.mark.parametrize("cap", [None, True, False, -1, 5, "1", 1.0])
def test_invalid_cap_cannot_bypass_scope(cap):
    with pytest.raises(SystemExit):
        run_phase4.in_scope(run_phase4.steps(), cap)


@pytest.mark.parametrize("raw", [None, b"{", b"[]", b"{}", b'{"max_tier": 5}'])
def test_invalid_admission_has_no_side_effects(tmp_path, monkeypatch, raw):
    root = tmp_path / "runs" / "clare-rineya-001"
    scope = tmp_path / "SCOPE.json"
    if raw is not None:
        scope.write_bytes(raw)
    monkeypatch.setattr(run_phase4, "ROOT", root)
    monkeypatch.setattr(run_phase4, "SCOPE", scope)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid scope must stop before building steps, writing status or launching a process")

    monkeypatch.setattr(run_phase4, "steps", forbidden)
    monkeypatch.setattr(run_phase4, "_write_status", forbidden)
    monkeypatch.setattr(run_phase4.subprocess, "call", forbidden)
    with pytest.raises(SystemExit):
        run_phase4.main()
    assert not root.exists()

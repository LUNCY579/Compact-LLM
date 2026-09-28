"""clare.metrics against hand-built records with known answers, including every refusal path."""
from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from clare import metrics as M


def rec(config, stage, row, sig, correct, arm="E", seed=0, order="RUSP", parent=None):
    return {"arm": arm, "seed": seed, "order": order, "config": config, "stage": stage, "row_id": row,
            "parent_id": parent or f"p-{row}", "signature": sig, "correct": correct, "terminated": True}


def seq_records(pattern):
    """pattern[skill][stage] = list of 0/1 over that skill's rows (2 rows per skill)."""
    out = []
    for sig, stages in pattern.items():
        for k, vals in stages.items():
            for j, v in enumerate(vals):
                out.append(rec(f"prefix:{k}", k, f"{sig}{j}", sig, v))
    return out


PATTERN = {  # R learned at 1, forgotten by 4; P learned last
    "R": {1: [1, 1], 2: [1, 1], 3: [1, 0], 4: [0, 0]},
    "U": {1: [0, 0], 2: [1, 1], 3: [1, 1], 4: [1, 1]},
    "S": {1: [0, 0], 2: [0, 0], 3: [1, 0], 4: [1, 1]},
    "P": {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [1, 0]},
}


def test_retention_matrix_known_values():
    m = M.retention_matrix(seq_records(PATTERN), "E", 0, "RUSP")
    assert m["acquisition"] == {"R": 1.0, "U": 1.0, "S": 0.5, "P": 0.5}
    assert m["final"] == {"R": 0.0, "U": 1.0, "S": 1.0, "P": 0.5}
    assert m["forgetting_final"] == {"R": 1.0, "U": 0.0, "S": -0.5, "P": 0.0}
    assert m["forgetting_max"] == {"R": 1.0, "U": 0.0, "S": 0.0, "P": 0.0}
    assert m["backward_transfer"] == pytest.approx(((0 - 1) + (1 - 1) + (1 - 0.5)) / 3)


def test_retention_refuses_missing_stage_and_row_mismatch():
    recs = [r for r in seq_records(PATTERN) if not (r["signature"] == "U" and r["stage"] == 3)]
    with pytest.raises(M.MetricsError):
        M.retention_matrix(recs, "E", 0, "RUSP")
    recs = seq_records(PATTERN)
    recs[0] = {**recs[0], "row_id": "R-other"}
    with pytest.raises(M.MetricsError):
        M.retention_matrix(recs, "E", 0, "RUSP")


def test_validation_refusals():
    good = rec("prefix:1", 1, "a", "R", 1)
    for bad in ({**good, "correct": "1"}, {**good, "stage": 2}, {**good, "config": "prefix:9"},
                {**good, "signature": "RX"}, {**good, "terminated": 1}, {k: v for k, v in good.items() if k != "arm"}):
        with pytest.raises(M.MetricsError):
            M.validate([bad])
    with pytest.raises(M.MetricsError):
        M.validate([good, good])
    with pytest.raises(M.MetricsError):
        M.validate([good, {**good, "config": "base", "stage": 0, "parent_id": "other"}])


def test_composition_matrix_covers_and_final():
    recs = []
    for k, vals in {3: [1, 0], 4: [1, 1]}.items():
        for j, v in enumerate(vals):
            recs.append(rec(f"prefix:{k}", k, f"RS{j}", "RS", v))
    m = M.composition_matrix(recs, "E", 0, "RUSP", ["RS"])
    assert m["stages"]["RS"] == [3, 4] and m["first_covered_stage"]["RS"] == 3
    assert m["acc"]["RS"] == {3: 0.5, 4: 1.0} and m["final"]["RS"] == 1.0


def test_paired_counts_mcnemar_and_bootstrap_determinism():
    x = [rec("prefix:4", 4, f"r{i}", "RS", v, arm="D", parent=f"w{i // 2}") for i, v in enumerate([1, 1, 0, 0, 1, 0, 1, 0])]
    y = [rec("prefix:4", 4, f"r{i}", "RS", v, arm="E", parent=f"w{i // 2}") for i, v in enumerate([1, 0, 1, 1, 1, 1, 1, 0])]
    p = M.paired(x, y, n_boot=2000, seed=7)
    assert (p["gains"], p["losses"], p["net"], p["n"], p["clusters"]) == (3, 1, 2, 8, 4)
    # independent binomial computation
    n, k = 4, 1
    want = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n)
    assert p["mcnemar_p"] == pytest.approx(want, rel=1e-12)
    assert M.paired(x, y, n_boot=2000, seed=7)["ci95"] == p["ci95"]
    with pytest.raises(M.MetricsError):
        M.paired(x[:-1], y)


def test_damage_does_not_net_offsetting_changes():
    ref = [rec("base", 0, f"r{i}", "R", 1 if i < 5 else 0) for i in range(10)]
    cand = [rec("prefix:4", 4, f"r{i}", "R", 0 if i < 5 else 1) for i in range(10)]
    d = M.damage(ref, cand)
    assert d["losses"] == 5 and d["gains"] == 5 and d["retained"] == 0 and "net" not in d
    assert M.paired(ref, cand, n_boot=100)["net"] == 0  # the net hides all of it


def test_floor_adjustment_and_csv_determinism(tmp_path):
    assert M.acc_above_floor(0.5, 0.0) == 0.5 and M.acc_above_floor(0.1, 0.2) == pytest.approx(-0.125)
    assert M.acc_above_floor(0.5, 1.0) is None
    m = M.retention_matrix(seq_records(PATTERN), "E", 0, "RUSP")
    h1 = M.write_retention_csv(tmp_path / "a.csv", m)
    h2 = M.write_retention_csv(tmp_path / "b.csv", m)
    assert h1 == h2 and b"\r" not in (tmp_path / "a.csv").read_bytes()
    with pytest.raises(FileExistsError):
        M.write_retention_csv(tmp_path / "a.csv", m)
    M.write_table_csv(tmp_path / "t.csv", [{"b": 1, "a": 2.5}, {"a": None, "b": 3}])
    assert (tmp_path / "t.csv").read_bytes() == b"a,b\n2.5,1\n,3\n"


def test_vendored_mcnemar_is_verbatim():
    src = Path(__file__).resolve().parents[2] / "titan_model_soup_mini_v1" / "score.py"
    theirs = next(n for n in ast.parse(src.read_text(encoding="utf-8")).body
                  if isinstance(n, ast.FunctionDef) and n.name == "exact_mcnemar")
    mine = next(n for n in ast.parse(Path(M.__file__).read_text(encoding="utf-8")).body
                if isinstance(n, ast.FunctionDef) and n.name == "exact_mcnemar")
    assert ast.dump(theirs) == ast.dump(mine)

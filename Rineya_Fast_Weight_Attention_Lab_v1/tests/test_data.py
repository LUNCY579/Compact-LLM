from __future__ import annotations

from rineya_fwa.data import build_lab_mix
from rineya_fwa.eval_interference import build_interference_suite
from rineya_fwa.eval_position import build_position_suite


def test_mix_is_deterministic(tmp_path):
    a = build_lab_mix(train_rows=8, dev_rows=4, strata=["2k"], seed=11, dest=tmp_path / "a")
    b = build_lab_mix(train_rows=8, dev_rows=4, strata=["2k"], seed=11, dest=tmp_path / "b")
    assert a["train_sha256"] == b["train_sha256"]
    assert a["source"] in {"lab_unsealed_synthetic", "official_cc0_unsealed"}


def test_position_and_interference_counts():
    items = build_position_suite(lengths=[64], n_per_cell=1, seed=0)
    assert len(items) == 7
    suite = build_interference_suite()
    assert [row["id"] for row in suite] == [f"INT-{x}" for x in "ABCDEFG"]

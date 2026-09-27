from __future__ import annotations

import pytest

from experiments.maxrl_comp002_pretraining_v1.rs_repairs import analyse


def row(identifier, parent, b, r, s, c, *, c_content=None):
    values = {"BASE": b, "R": r, "S": s, "RS": c}
    return {"id": identifier, "parent_id": parent,
            "scores": {arm: {"strict": value, "content": value if arm != "RS" or c_content is None else c_content}
                       for arm, value in values.items()}}


def test_repairs_and_damage_are_rowwise_but_uncertainty_is_parentwise():
    rows = [row("a1", "a", False, False, False, True),
            row("a2", "a", False, True, False, False),
            row("b1", "b", True, True, True, True),
            row("b2", "b", False, False, False, False)]
    result = analyse(rows, "strict", draws=200, seed=17)
    assert result["counts"]["composition_only_repair"] == 1
    assert result["counts"]["joint_damage"] == 1
    assert result["counts"]["all_four_fail"] == 1
    assert result["parents"] == 2
    assert result["parent_cluster_bootstrap"]["composition_only_repair"]["lower95_count"] == 0
    assert result["parent_cluster_bootstrap"]["composition_only_repair"]["upper95_count"] == 2


def test_format_only_rs_repair_appears_only_in_content_column():
    rows = [row("a", "a", False, False, False, False, c_content=True)]
    assert analyse(rows, "strict", draws=4)["counts"]["composition_only_repair"] == 0
    assert analyse(rows, "content", draws=4)["counts"]["composition_only_repair"] == 1


def test_duplicate_ids_and_inconsistent_sequential_panel_fail():
    a = row("a", "p", False, False, False, False)
    with pytest.raises(ValueError, match="duplicate"):
        analyse([a, a], "strict", draws=4)
    b = row("b", "p", False, False, False, False)
    b["scores"]["R_to_S"] = {"strict": False, "content": False}
    with pytest.raises(ValueError, match="partial"):
        analyse([a, b], "strict", draws=4)

"""Regression tests for the Phase-3 position axis (``scripts/phase3_analyze.py``).

The bug these guard against: ``multi_hop`` is the only family whose grid sweeps *both*
evidence sentences. ``multi_hop_pairs`` emits the six start/middle/end combinations, so
grouping on the decisive hop alone produces a triangular design --

    p=0.02 : {(0.02, 0.02)}
    p=0.50 : {(0.02, 0.50), (0.50, 0.50)}
    p=0.98 : {(0.02, 0.98), (0.50, 0.98), (0.98, 0.98)}

-- in which the cells differ in experimental *content*, not only in where the sentence
under study sits. The far cell is the only one that can contain "both hops adjacent at
the end", which is far easier, so the p=0.98 mean is pulled down and the
middle-minus-edge gap is inflated. Pure CPU: no model, no checkpoint, no GPU.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from rineya_psa.paths import LAB_ROOT

SCRIPT = LAB_ROOT / "scripts" / "phase3_analyze.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("phase3_analyze_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


phase3 = _load_script()


# --------------------------------------------------------------------------------------
# The non-decisive key
# --------------------------------------------------------------------------------------


def test_nondecisive_key_drops_exactly_one_occurrence_of_the_decisive_position():
    # multi_hop's (start, start) cell: both hops at 0.02, decisive is the second. The
    # key must be the remaining 0.02, not the empty string -- otherwise (0.02, 0.02)
    # looks like a single-evidence item and is silently pooled with everything else.
    assert phase3.nondecisive_key([0.02, 0.02], 0.02) == "0.02"
    assert phase3.nondecisive_key([0.02, 0.98], 0.98) == "0.02"
    assert phase3.nondecisive_key([0.50, 0.98], 0.98) == "0.50"
    # Single-evidence families have no non-decisive evidence at all.
    assert phase3.nondecisive_key([0.33], 0.33) == ""
    # state_constraint: three pinned anchors plus the swept correction. The key is the
    # same whichever occurrence is removed when the correction lands on an anchor.
    assert phase3.nondecisive_key([0.25, 0.50, 0.75, 0.25], 0.25) == "0.25,0.50,0.75"
    assert phase3.nondecisive_key([0.25, 0.50, 0.75, 0.10], 0.10) == "0.25,0.50,0.75"


# --------------------------------------------------------------------------------------
# The restriction itself, on a synthetic frame with a known ground truth
# --------------------------------------------------------------------------------------


def _synthetic_multi_hop(n_scenarios: int = 24) -> pd.DataFrame:
    """The real multi_hop design, with a NLL model whose true position effect is zero.

    Cost is built from two additive terms and nothing else:

    * a per-scenario intercept, and
    * a far-edge adjacency bonus: both hops at 0.98 is worth -10 nats.

    That config exists only in the p=0.98 cell of the triangular grid, so averaging
    unrestricted cells makes the far edge look artificially easy. The position of the
    decisive hop itself contributes **nothing**. Restricting to the shared
    non-decisive key (secondary at 0.02) drops every far-only adjacency row, so the
    restricted middle-minus-edge gap must be zero.
    """
    pairs = [(0.02, 0.02), (0.02, 0.50), (0.02, 0.98),
             (0.50, 0.50), (0.50, 0.98), (0.98, 0.98)]
    rng = np.random.default_rng(20260828)
    rows = []
    for s in range(n_scenarios):
        intercept = 20.0 + rng.normal(0.0, 0.5)
        for secondary, primary in pairs:
            far_adjacent = secondary == 0.98 and primary == 0.98
            rows.append({
                "item_id": f"i{s}_{secondary}_{primary}",
                "scenario_id": f"s{s}",
                "target_positions": [secondary, primary],
                "primary_target": primary,
                "gold_nll": intercept - (10.0 if far_adjacent else 0.0),
            })
    return pd.DataFrame(rows)


def test_restriction_balances_the_multi_hop_cells():
    d = _synthetic_multi_hop()
    kept, info = phase3.restrict_position_axis(d)

    assert info["axis_confounded"] is True
    assert info["axis_restricted"] is True
    # The only non-decisive configuration present at all three levels is "hop A at the
    # start"; the other two span two levels and one level respectively.
    assert info["nondecisive_pinned_at"] == "0.02"
    assert sorted(info["nondecisive_keys"]) == ["0.02", "0.50", "0.98"]

    before = d.groupby("primary_target").size().to_dict()
    after = kept.groupby("primary_target").size().to_dict()
    assert before == {0.02: 24, 0.50: 48, 0.98: 72}, "the published design is triangular"
    assert after == {0.02: 24, 0.50: 24, 0.98: 24}, "the restricted design is balanced"


def test_restriction_removes_a_gap_that_is_entirely_confound():
    """The load-bearing assertion: without the restriction the gap is a fiction.

    In this frame the decisive hop's position is worth exactly zero nats. The
    unrestricted middle-minus-edge statistic nevertheless reports a positive gap,
    because only the far cell contains the both-at-end adjacency bonus. The
    restricted statistic reports zero, which is the truth.
    """
    d = _synthetic_multi_hop()
    kept, info = phase3.restrict_position_axis(d)

    unrestricted = phase3.curve_stats(d, "gold_nll")
    restricted = phase3.curve_stats(kept, "gold_nll")

    # mid and near are clean; far is pulled down by the (0.98, 0.98) third → gap≈1.67
    assert unrestricted["middle_minus_edge"] == pytest.approx(1.667, abs=0.02)
    assert unrestricted["middle_minus_edge_significant"] is True
    assert restricted["middle_minus_edge"] == pytest.approx(0.0, abs=1e-9)
    assert restricted["middle_minus_edge_significant"] is False
    assert info["n_dropped"] == len(d) - len(kept) == 72


def test_restriction_is_a_no_op_for_pinned_and_single_evidence_families():
    """It must not quietly thin families whose non-decisive evidence really is pinned."""
    # exact_retrieval / semantic_single_hop: one evidence sentence, seven positions.
    single = pd.DataFrame({
        "item_id": [f"e{i}" for i in range(7)],
        "scenario_id": ["s0"] * 7,
        "target_positions": [[p] for p in (0.02, 0.17, 0.33, 0.50, 0.67, 0.83, 0.98)],
        "primary_target": [0.02, 0.17, 0.33, 0.50, 0.67, 0.83, 0.98],
        "gold_nll": [1.0] * 7,
    })
    kept, info = phase3.restrict_position_axis(single)
    assert len(kept) == len(single)
    assert info["axis_restricted"] is False and info["axis_confounded"] is False

    # state_constraint: three anchors pinned by construction, correction swept.
    positions = (0.02, 0.17, 0.33, 0.50, 0.67, 0.83, 0.98)
    state = pd.DataFrame({
        "item_id": [f"c{i}" for i in range(7)],
        "scenario_id": ["s0"] * 7,
        "target_positions": [[0.25, 0.50, 0.75, p] for p in positions],
        "primary_target": list(positions),
        "gold_nll": [1.0] * 7,
    })
    kept, info = phase3.restrict_position_axis(state)
    assert len(kept) == len(state)
    assert info["axis_restricted"] is False and info["axis_confounded"] is False


# --------------------------------------------------------------------------------------
# The real calibration artifact
# --------------------------------------------------------------------------------------

PARQUET = LAB_ROOT / "results" / "baseline_position_results_calibration_tf.parquet"


@pytest.mark.skipif(not PARQUET.exists(), reason="Phase-3 calibration parquet not built")
def test_real_multi_hop_axis_is_confounded_and_the_restriction_changes_the_number():
    d = pd.read_parquet(PARQUET)
    d = d[(d.local_position == "middle") & (d.fmt == "natural")].copy()
    d["primary_target"] = d.primary_target.round(2)
    sub = d[(d.family == "multi_hop") & (~d.guided) & (d.arm == "halo")]
    assert not sub.empty

    kept, info = phase3.restrict_position_axis(sub)
    assert info["axis_confounded"] is True
    assert info["axis_restricted"] is True
    assert info["nondecisive_pinned_at"] == "0.02"

    assert sub.groupby("primary_target").size().to_dict() == {0.02: 39, 0.50: 78, 0.98: 117}
    assert kept.groupby("primary_target").size().to_dict() == {0.02: 39, 0.50: 39, 0.98: 39}

    published = phase3.curve_stats(sub, "gold_nll")
    honest = phase3.curve_stats(kept, "gold_nll")
    # As published: +5.379. On a clean axis: +3.964. The effect survives, at ~74 % of
    # the tabulated size; the difference is the confound, not noise.
    assert published["middle_minus_edge"] == pytest.approx(5.379, abs=0.01)
    assert honest["middle_minus_edge"] == pytest.approx(3.964, abs=0.01)
    assert honest["middle_minus_edge_significant"] is True

    # The "recovery at the far edge" is what the confound mostly manufactured: 4.73
    # nats of apparent primacy-recency U becomes 1.08 once the cells match.
    pub = published["by_pos"]
    hon = honest["by_pos"]
    assert pub["0.5"] - pub["0.98"] == pytest.approx(4.731, abs=0.01)
    assert hon["0.5"] - hon["0.98"] == pytest.approx(1.078, abs=0.01)


@pytest.mark.skipif(not PARQUET.exists(), reason="Phase-3 calibration parquet not built")
def test_moving_only_the_non_decisive_hop_is_worth_more_than_the_whole_position_effect():
    """Quantifies the nuisance factor the triangular design lets leak into the axis."""
    d = pd.read_parquet(PARQUET)
    d = d[(d.local_position == "middle") & (d.fmt == "natural")].copy()
    d["primary_target"] = d.primary_target.round(2)
    d["secondary"] = d.target_positions.map(lambda t: round(float(t[0]), 2))
    sub = d[(d.family == "multi_hop") & (~d.guided) & (d.arm == "halo")
            & (d.primary_target == 0.98)]
    by_secondary = sub.groupby("secondary").gold_nll.mean()
    # Decisive hop pinned at the very end; only the OTHER hop moves. 24.6 -> 11.1 nats.
    swing = by_secondary[0.02] - by_secondary[0.98]
    assert swing == pytest.approx(13.53, abs=0.05)
    assert swing > 5.379, "the nuisance factor dwarfs the published position effect"


def test_analysis_docstring_names_the_non_decisive_confound():
    """The script's header is the first thing a reader of the artifact consults."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert "NON-decisive" in text
    assert "multi_hop_pairs" in text

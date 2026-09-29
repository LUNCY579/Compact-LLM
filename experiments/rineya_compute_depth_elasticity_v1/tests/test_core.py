from dataclasses import replace
import math
import pytest

from experiments.rineya_compute_depth_elasticity_v1.core import (
    Observation, Policy, Trace, calibrate, cell_metrics, choose_depth,
    grid, nested_seed, replay, total_variation,
)
from experiments.rineya_compute_depth_elasticity_v1.fixture import bank


def trace(labels=(False, False, False, True), *, split="dev", parent="p", rollout=1):
    obs = tuple(Observation(i, "same-prefix", str(int(ok)),
                            (0.1, 0.9) if ok else (0.9, 0.1), float(i))
                for i, ok in enumerate(labels, 1))
    return Trace(parent, split, rollout, "fixed-model", obs, tuple(labels))


def test_grid_marks_proxy_not_flops():
    cells = grid((1, 2, 4, 8), (1, 2, 4, 8))
    assert len(cells) == 16
    assert {(c["depth"], c["width"]) for c in cells if c["nominal_RK"] == 8} == {
        (1, 8), (2, 4), (4, 2), (8, 1)}
    assert all(c["compute_match"] == "UNQUALIFIED_PROXY_ONLY" for c in cells)


@pytest.mark.parametrize("axis", [(), (0,), (-1,), (True,), (1, 1), (1.5,)])
def test_bad_grid(axis):
    with pytest.raises(ValueError):
        grid(axis, (1,))


def test_seeds_are_nested_not_depth_or_width_dependent():
    a = [nested_seed(17, "parent", k) for k in range(1, 5)]
    b = [nested_seed(17, "parent", k) for k in range(1, 9)]
    assert a == b[:4] and len(set(b)) == 8
    assert nested_seed(17, "other", 1) != a[0]


@pytest.mark.parametrize("p", [(), (0.4, 0.4), (-0.1, 1.1), (math.nan, 1), (math.inf, 0)])
def test_invalid_probabilities(p):
    with pytest.raises(ValueError):
        Observation(1, "prefix", "x", p, 1)


def test_tv_known_result_and_vocab_guard():
    assert total_variation((0.9, 0.1), (0.1, 0.9)) == pytest.approx(0.8)
    with pytest.raises(ValueError):
        total_variation((1.0,), (0.5, 0.5))


def test_policy_cannot_see_gold_and_never_exits_at_one():
    t = trace((True,) * 4)
    assert "correct" not in Observation.__dataclass_fields__
    assert choose_depth(t.observations, Policy(patience=1)) == 2
    assert choose_depth(t.observations, Policy(patience=2)) == 3
    assert choose_depth(t.observations[:1], Policy(patience=1)) == 1  # exhausted cap


def test_prefix_decision_invariance():
    t = trace((True, True, True, False))
    assert choose_depth(t.observations[:3], Policy()) == 3
    assert choose_depth(t.observations, Policy()) == 3


def test_wrong_agreement_is_reported_as_missed_rescue():
    m = replay([trace()], Policy(patience=2))
    assert m["selected_accuracy"] == 0 and m["final_accuracy"] == 1
    assert m["missed_rescue"] == 1 and m["late_rescue"] == 1
    assert m["selected_work_counterfactual"] == 3


def test_calibration_falls_back_instead_of_missing_rescue():
    policy = calibrate([trace()], [0.0, 0.1], patience=2)
    assert policy.full_depth_only
    assert policy.calibration_parents == ("p",)


def test_calibration_finds_safe_work_reduction():
    policy = calibrate([trace((True,) * 8)], [0.0, 0.1], patience=2)
    assert not policy.full_depth_only
    assert replay([trace((True,) * 8)], policy)["chosen_depth"] == 3


def test_reversion_and_oracle_are_not_final_quality():
    t = trace((False, True, False, False))
    m = replay([t], Policy(full_depth_only=True))
    assert m["oracle_any_depth"] == 1 and m["final_accuracy"] == 0
    assert m["reversions"] == 1


def test_parent_balancing_not_rollout_pseudoreplication():
    traces = [trace((True,), parent="a", rollout=k) for k in range(1, 5)]
    traces += [trace((False,), parent="b")]
    assert replay(traces, Policy())["final_accuracy"] == 0.5


@pytest.mark.parametrize("split", ["train", "test"])
def test_calibrate_dev_only(split):
    with pytest.raises(ValueError):
        calibrate([trace(split=split)], [0])


def test_holdout_rejects_calibration_parent_and_wrong_split():
    policy = calibrate([trace()], [0])
    with pytest.raises(ValueError, match="leakage"):
        replay([trace(split="test")], policy, heldout=True)
    with pytest.raises(ValueError, match="test parents"):
        replay([trace()], policy, heldout=True)
    replay([trace(split="test", parent="new")], policy, heldout=True)


def test_missing_depth_changing_prefix_decreasing_work_rejected():
    t = trace()
    for changed in (replace(t.observations[1], depth=3),
                    replace(t.observations[1], prefix_id="future-token-prefix"),
                    replace(t.observations[1], cumulative_work=0)):
        with pytest.raises(ValueError):
            replace(t, observations=(t.observations[0], changed) + t.observations[2:])


def test_duplicate_or_mixed_model_rejected():
    t = trace()
    with pytest.raises(ValueError):
        replay([t, t], Policy())
    with pytest.raises(ValueError):
        replay([t, replace(t, parent_id="other", checkpoint_id="different")], Policy())


def test_width_majority_never_selects_with_oracle():
    ts = [trace((False,), parent="p", rollout=1), trace((False,), parent="p", rollout=2),
          trace((True,), parent="p", rollout=3)]
    m = cell_metrics(ts, 1, 3)
    assert m["selected_accuracy"] == 0 and m["oracle_final_depth_width"] == 1
    assert m["summed_work"] == 3
    with pytest.raises(ValueError):
        cell_metrics(ts, 1, 3, selector="oracle")


def test_width_nested_subset_and_missing_rollout():
    ts = [trace((False,), rollout=1), trace((True,), rollout=2)]
    assert cell_metrics(ts, 1, 1)["selected_accuracy"] == 0
    assert cell_metrics(ts, 1, 2)["selected_accuracy"] == 0  # tie -> first
    with pytest.raises(ValueError):
        cell_metrics(ts[1:], 1, 1)
    with pytest.raises(ValueError):
        cell_metrics(ts, 2, 1)


def test_inconsistent_identical_answer_verdict_rejected():
    a = trace((True,), rollout=1)
    b = replace(a, rollout=2, correct=(False,))
    with pytest.raises(ValueError, match="inconsistent verifier"):
        cell_metrics([a, b], 1, 2)


def test_fixture_reproducible_and_separated():
    assert bank("dev") == bank("dev")
    assert {t.parent_id for t in bank("dev")}.isdisjoint(t.parent_id for t in bank("test"))
    policy = calibrate(bank("dev"), [0, 0.1])
    result = replay(bank("test"), policy, heldout=True)
    assert result["n_parents"] == 12
    assert "NOT_NATIVE_QUALIFICATION" in result["evidence"]


def test_grid_does_not_pool_dev_and_test():
    with pytest.raises(ValueError, match="one split"):
        cell_metrics([trace(parent="a"), trace(parent="b", split="test")], 1, 1)


def test_calibrated_policy_binds_checkpoint_and_depth():
    policy = calibrate([trace()], [0])
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        replay([replace(trace(parent="new", split="test"), checkpoint_id="other")],
               policy, heldout=True)
    with pytest.raises(ValueError, match="maximum-depth mismatch"):
        replay([trace((True,) * 8, parent="new", split="test")], policy, heldout=True)

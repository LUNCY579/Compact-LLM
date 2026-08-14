from __future__ import annotations

import torch

from consensus_lab.voting.rules import _stable_topk_indices, aggregate_logits


def logits(top: int, *, vocab: int = 6, strength: float = 5.0) -> torch.Tensor:
    values = torch.zeros(vocab)
    values[top] = strength
    return values


def test_plain_plurality_selects_mode() -> None:
    decision = aggregate_logits(
        {"a": logits(2), "b": logits(2), "c": logits(4)},
        method="hard_plurality",
        eos_token_id=5,
    )
    assert decision.token_id == 2
    assert decision.diagnostics["agreement"] == 2 / 3
    assert decision.diagnostics["strict_majority"] is True
    assert decision.diagnostics["majority_margin"] == 1


def test_deterministic_tie_uses_sum_log_probability_then_token_id() -> None:
    symmetric = {"a": logits(1), "b": logits(2)}
    decision = aggregate_logits(symmetric, method="hard_plurality", eos_token_id=5)
    assert decision.token_id == 1
    assert decision.diagnostics["tie"] is True
    assert decision.diagnostics["tied_tokens"] == [1, 2]


def test_tie_sum_log_probability_can_override_smallest_id() -> None:
    a = torch.tensor([0.0, 5.0, 4.9, 0.0])
    b = torch.tensor([0.0, -5.0, 5.0, 0.0])
    decision = aggregate_logits({"a": a, "b": b}, method="hard_plurality", eos_token_id=3)
    assert decision.token_id == 2
    assert decision.diagnostics["tie"] is True


def test_positive_logit_scaling_preserves_unique_plurality_winner() -> None:
    values = {"a": logits(1), "b": logits(1), "c": logits(2)}
    scaled = {"a": values["a"] * 0.1, "b": values["b"] * 10.0, "c": values["c"] * 3.0}
    plain = aggregate_logits(values, method="hard_plurality", eos_token_id=None)
    intervention = aggregate_logits(scaled, method="hard_plurality", eos_token_id=None)
    assert plain.token_id == intervention.token_id == 1
    assert plain.diagnostics["top_tokens"] == intervention.diagnostics["top_tokens"]


def test_unequal_positive_scaling_can_change_confidence_tie_break() -> None:
    values = {
        "a": torch.tensor([0.0, 5.0, 4.9, 0.0]),
        "b": torch.tensor([0.0, -5.0, 5.0, 0.0]),
    }
    scaled = {"a": values["a"], "b": values["b"] * 0.001}
    plain = aggregate_logits(values, method="hard_plurality", eos_token_id=None)
    intervention = aggregate_logits(scaled, method="hard_plurality", eos_token_id=None)
    assert plain.diagnostics["top_tokens"] == intervention.diagnostics["top_tokens"]
    assert plain.diagnostics["tie"] is intervention.diagnostics["tie"] is True
    assert plain.token_id == 2
    assert intervention.token_id == 1


def test_k1_invariant() -> None:
    decision = aggregate_logits({"only": logits(3)}, method="hard_plurality", eos_token_id=5)
    assert decision.token_id == 3
    assert decision.diagnostics["agreement"] == 1.0
    assert decision.diagnostics["strict_majority"] is True


def test_expert_argmax_uses_stable_smallest_token_id_on_exact_tie() -> None:
    values = torch.zeros(7)
    values[4] = values[2] = 5.0
    decision = aggregate_logits({"only": values}, method="hard_plurality", eos_token_id=6)
    assert decision.token_id == 2


def test_duplicate_expert_invariant() -> None:
    single = aggregate_logits({"a": logits(4)}, method="hard_plurality", eos_token_id=5)
    duplicate = aggregate_logits(
        {"a": logits(4), "b": logits(4), "c": logits(4)}, method="hard_plurality", eos_token_id=5
    )
    assert duplicate.token_id == single.token_id == 4
    assert duplicate.diagnostics["agreement"] == 1.0


def test_soft_methods_and_weighted_vote_are_defined() -> None:
    values = {"a": logits(1), "b": logits(2), "c": logits(2)}
    for method in (
        "weighted_hard",
        "probability_mean",
        "calibrated_probability_mean",
        "logit_mean",
        "calibrated_logit_mean",
        "normalized_logit_mean",
        "top2_approval",
        "borda_top5",
        "reciprocal_rank_top5",
        "confidence_weighted_probability",
        "confidence_weighted_logit",
    ):
        decision = aggregate_logits(
            values,
            method=method,
            eos_token_id=5,
            validation_weights={"a": 0.1, "b": 0.5, "c": 0.5},
            temperatures={"a": 1.2, "b": 0.9, "c": 1.0},
        )
        assert 0 <= decision.token_id < 6


def test_raw_probability_mean_is_not_contaminated_by_temperatures() -> None:
    values = {"a": torch.tensor([0.0, 4.0, 0.0]), "b": torch.tensor([0.0, 0.0, 3.0])}
    plain = aggregate_logits(values, method="probability_mean", eos_token_id=None)
    temperatures = aggregate_logits(
        values,
        method="probability_mean",
        eos_token_id=None,
        temperatures={"a": 0.1, "b": 10.0},
    )
    assert temperatures.token_id == plain.token_id


def test_calibrated_probability_mean_uses_temperatures() -> None:
    values = {"a": torch.tensor([0.0, 4.0, 0.0]), "b": torch.tensor([0.0, 0.0, 3.0])}
    raw = aggregate_logits(values, method="probability_mean", eos_token_id=None)
    calibrated = aggregate_logits(
        values,
        method="calibrated_probability_mean",
        eos_token_id=None,
        temperatures={"a": 10.0, "b": 0.1},
    )
    assert raw.token_id == 1
    assert calibrated.token_id == 2


def test_rank_rules_reward_repeated_high_rank_support() -> None:
    values = {
        "a": torch.tensor([0.0, 5.0, 4.0, 0.0]),
        "b": torch.tensor([0.0, 0.0, 4.0, 5.0]),
        "c": torch.tensor([0.0, 0.0, 4.0, 5.0]),
    }
    approval = aggregate_logits(values, method="top2_approval", eos_token_id=None)
    borda = aggregate_logits(values, method="borda_top5", eos_token_id=None)
    reciprocal = aggregate_logits(values, method="reciprocal_rank_top5", eos_token_id=None)
    assert approval.token_id == 2
    assert borda.token_id in range(4)
    assert reciprocal.token_id in range(4)
    assert approval.diagnostics["rank_depth"] == 2
    assert approval.diagnostics["rank_score_totals"]["2"] == 3.0


def test_stable_partial_topk_matches_full_stable_sort_with_ties() -> None:
    scores = torch.tensor(
        [
            [0.0, 4.0, 4.0, 3.0, 3.0, 3.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [5.0, 1.0, 4.0, 2.0, 3.0, 3.0, 3.0],
        ]
    )
    expected = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :5]
    actual = _stable_topk_indices(scores, 5)
    assert torch.equal(actual, expected)


def test_masked_zero_probability_diagnostics_remain_finite() -> None:
    values = {"a": torch.tensor([0.0, float("-inf"), 2.0]), "b": torch.tensor([1.0, float("-inf"), 0.0])}
    decision = aggregate_logits(values, method="hard_plurality", eos_token_id=1)
    assert torch.isfinite(torch.tensor(decision.diagnostics["mean_pairwise_js"]))
    assert all(torch.isfinite(torch.tensor(value)) for value in decision.diagnostics["expert_entropies"].values())

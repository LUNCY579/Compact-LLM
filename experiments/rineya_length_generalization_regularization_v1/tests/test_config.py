from dataclasses import replace
import pytest
from experiments.rineya_length_generalization_regularization_v1.config import Plan, screening_plans, factorial_plans


@pytest.mark.parametrize("changes", [
    {"mode": "invalid"}, {"probability": 1}, {"probability": -0.1}, {"probability": float("nan")},
    {"probability": 0.1}, {"seed": -1}, {"seed": True}, {"rank": 0}, {"expected_layers": 0},
    {"max_steps": 1.2}, {"accumulation": False}, {"max_tokens": 0}, {"max_new_tokens": 0},
    {"learning_rate": 0}, {"weight_decay": -1}, {"learning_rate": float("inf")},
    {"beta1": 1}, {"beta2": -1}, {"warmup_fraction": 1}, {"max_grad_norm": 0},
    {"eval_lengths": []}, {"eval_lengths": [2, 1]}, {"eval_lengths": [1, 1]},
    {"eval_lengths": [1, False]}, {"dtype": "float16"}, {"attention": "flash_attention_2"},
    {"detach_beta": 1}, {"gradient_checkpointing": "false"}, {"lora_alpha": 0}])
def test_invalid_plans(changes):
    with pytest.raises((ValueError, TypeError)):
        Plan(**changes)


def test_controlled_arms():
    plans = screening_plans()
    normalized = []
    for plan in plans.values():
        row = plan.to_dict()
        row.pop("mode"); row.pop("probability")
        normalized.append(row)
    assert all(r == normalized[0] for r in normalized)
    assert {p.mode for p in plans.values()} == {"none", "pre_linear", "vpad"}
    assert Plan(**Plan().to_dict()) == Plan()


def test_factorial_is_explicit():
    plan = replace(Plan(), mode="vpad", probability=0.1)
    arms = factorial_plans(plan, 0.1)
    assert len(arms) == 4
    assert {p.weight_decay for p in arms.values()} == {0, 0.1}
    with pytest.raises(ValueError):
        factorial_plans(plan, 0)
    with pytest.raises(ValueError):
        factorial_plans(Plan(), 0.1)

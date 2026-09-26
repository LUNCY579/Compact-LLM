from experiments.gflowrl_reasoning_v1.evaluation import paired_repairs_damages, summarize_rollouts


def test_evaluation_schema_preserves_successful_mode_diversity():
    rows = [
        {"prompt_id": "p1", "rollout_index": 0, "reward": 1.0, "trajectory": "a", "strategy": "algebra", "response_length": 4},
        {"prompt_id": "p1", "rollout_index": 1, "reward": 1.0, "trajectory": "b", "strategy": "table", "response_length": 5},
        {"prompt_id": "p1", "rollout_index": 2, "reward": 0.0, "trajectory": "c", "strategy": "guess", "response_length": 3},
        {"prompt_id": "p1", "rollout_index": 3, "reward": 0.0, "trajectory": "d", "strategy": "guess", "response_length": 2},
    ]
    result = summarize_rollouts(rows)
    assert result["pass_at_1"] == 1.0
    assert result["pass_at_k"]["1"] == 1.0
    assert result["pass_at_k"]["4"] == 1.0
    assert result["pass_at_k_denominators"]["4"]["eligible_prompts"] == 1
    assert result["correct_solution_diversity"]["unique_successful_strategies_global"] == 2
    assert result["correct_solution_diversity"]["mean_unique_successful_strategies_per_prompt"] == 2.0
    assert result["correct_solution_diversity"]["fraction_prompts_with_multiple_successful_strategies"] == 1.0
    assert result["exact_reward_distribution"] == {"0.0": 2, "1.0": 2}


def test_pass_at_k_never_silently_uses_short_groups():
    rows = [
        {"prompt_id": "p1", "rollout_index": 0, "reward": 1.0, "trajectory": "a", "response_length": 2},
        {"prompt_id": "p1", "rollout_index": 1, "reward": 0.0, "trajectory": "b", "response_length": 2},
        {"prompt_id": "p1", "rollout_index": 2, "reward": 0.0, "trajectory": "c", "response_length": 2},
        {"prompt_id": "p1", "rollout_index": 3, "reward": 0.0, "trajectory": "d", "response_length": 2},
        {"prompt_id": "p2", "rollout_index": 0, "reward": 0.0, "trajectory": "e", "response_length": 2},
        {"prompt_id": "p2", "rollout_index": 1, "reward": 1.0, "trajectory": "f", "response_length": 2},
        {"prompt_id": "p2", "rollout_index": 2, "reward": 0.0, "trajectory": "g", "response_length": 2, "valid": False},
        {"prompt_id": "p2", "rollout_index": 3, "reward": 0.0, "trajectory": "h", "response_length": 2, "valid": False},
    ]
    result = summarize_rollouts(rows)
    assert result["pass_at_k"]["1"] == 0.5
    assert result["pass_at_k"]["4"] == 1.0
    assert result["pass_at_k_denominators"]["4"] == {
        "eligible_prompts": 1,
        "passed_prompts": 1,
        "insufficient_valid_rollouts": 1,
    }
    assert result["invalid_rollout_count"] == 2


def test_paired_repairs_and_damages_use_exact_rollout_zero():
    r0 = [
        {"prompt_id": "repair", "rollout_index": 0, "reward": 0.0},
        {"prompt_id": "damage", "rollout_index": 0, "reward": 1.0},
        {"prompt_id": "same-good", "rollout_index": 0, "reward": 1.0},
        {"prompt_id": "same-bad", "rollout_index": 0, "reward": 0.0},
        {"prompt_id": "missing-r1", "rollout_index": 0, "reward": 0.0},
    ]
    r1 = [
        {"prompt_id": "repair", "rollout_index": 0, "reward": 1.0},
        {"prompt_id": "damage", "rollout_index": 0, "reward": 0.0},
        {"prompt_id": "same-good", "rollout_index": 0, "reward": 1.0},
        {"prompt_id": "same-bad", "rollout_index": 0, "reward": 0.0},
        {"prompt_id": "missing-r0", "rollout_index": 0, "reward": 1.0},
        {"prompt_id": "repair", "rollout_index": 1, "reward": 0.0},
    ]
    result = paired_repairs_damages(r0, r1)
    assert result["paired_prompts"] == 4
    assert result["repairs"] == 1
    assert result["damages"] == 1
    assert result["net_repairs_minus_damages"] == 0
    assert result["repair_prompt_ids"] == ["repair"]
    assert result["damage_prompt_ids"] == ["damage"]
    assert result["missing_r0_prompt_ids"] == ["missing-r0"]
    assert result["missing_r1_prompt_ids"] == ["missing-r1"]

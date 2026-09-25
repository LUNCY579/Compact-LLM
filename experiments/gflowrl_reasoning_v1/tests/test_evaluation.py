from experiments.gflowrl_reasoning_v1.evaluation import summarize_rollouts


def test_evaluation_schema_preserves_successful_mode_diversity():
    rows = [
        {"prompt_id": "p1", "rollout_index": 0, "reward": 1.0, "trajectory": "a", "strategy": "algebra", "response_length": 4},
        {"prompt_id": "p1", "rollout_index": 1, "reward": 1.0, "trajectory": "b", "strategy": "table", "response_length": 5},
        {"prompt_id": "p1", "rollout_index": 2, "reward": 0.0, "trajectory": "c", "strategy": "guess", "response_length": 3},
        {"prompt_id": "p1", "rollout_index": 3, "reward": 0.0, "trajectory": "d", "strategy": "guess", "response_length": 2},
    ]
    result = summarize_rollouts(rows)
    assert result["pass_at_k"]["1"] == 1.0
    assert result["pass_at_k"]["4"] == 1.0
    assert result["correct_solution_diversity"]["unique_successful_strategies"] == 2
    assert result["exact_reward_distribution"] == {"0.0": 2, "1.0": 2}

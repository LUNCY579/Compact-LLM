from math import isclose, sqrt

import pytest

from experiments.skill_acquisition_method_lab_v1 import exact_math as m


@pytest.mark.parametrize("p", [0.0005, 0.01, 0.1, 0.5, 0.9, 0.999])
@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_exact_maxrl_orders_and_unbiased_rloo(p, n):
    success = m.gradient_statistics("maxrl_success", p, n)
    full_cv = m.gradient_statistics("maxrl_full_cv", p, n)
    practical = m.gradient_statistics("maxrl_practical", p, n)
    assert success["expected_logit_gradient"] == pytest.approx(m.maxrl_closed_gradient(p, n, practical=False), abs=1e-12)
    assert full_cv["expected_logit_gradient"] == pytest.approx(success["expected_logit_gradient"], abs=1e-12)
    assert practical["expected_logit_gradient"] == pytest.approx(m.maxrl_closed_gradient(p, n, practical=True), abs=1e-12)
    assert m.gradient_statistics("rloo", p, n)["expected_logit_gradient"] == pytest.approx(p*(1-p), abs=1e-12)


def test_all_failure_and_mastered_groups_have_no_centered_signal():
    for method in ("maxrl_practical", "grpo_sample", "grpo_population", "rloo"):
        assert m.group_gradient(method, 0.01, 8, 0) == 0
        assert m.group_gradient(method, 0.99, 8, 8) == 0
    assert m.group_gradient("maxrl_full_cv", 0.01, 8, 0) == 0.01


def test_group_two_degeneracy_and_epsilon_bias():
    assert m.group_gradient("maxrl_practical", .2, 2, 1) / m.group_gradient("grpo_sample", .2, 2, 1) == pytest.approx(sqrt(2))
    exact = m.gradient_statistics("maxrl_practical", .05, 8)
    perturbed = m.gradient_statistics("maxrl_practical", .05, 8, 1e-6)
    assert perturbed["expected_logit_gradient"] < exact["expected_logit_gradient"]


@pytest.mark.parametrize("p", [0, .005, .1, .7, 1])
@pytest.mark.parametrize("gamma", [0, .5, 1])
def test_stopped_sampling_cost_and_probability(p, gamma):
    result = m.bounded_failure_continuation(p, 4, 3, gamma)
    assert result["probability_sum"] == pytest.approx(1, abs=1e-12)
    assert result["expected_draws"] == pytest.approx(result["expected_draws_closed"], abs=1e-12)
    if gamma == 0:
        assert result["expected_pooled_practical_gradient"] == pytest.approx(result["fixed_initial_group_gradient"], abs=1e-12)


def test_optional_stopping_does_not_inherit_fixed_n_objective():
    result = m.bounded_failure_continuation(.05, 4, 3, 1)
    assert not isclose(result["expected_pooled_practical_gradient"], result["fixed_max_group_gradient"], rel_tol=1e-3)
    mixture = m.rejection_mixture([.02, .5], [.5, .5], 8)
    assert mixture["global_rejection_expected_gradient"] * mixture["acceptance"] == pytest.approx(mixture["fixed_proposal_expected_gradient"])
    assert mixture["per_prompt_quota_expected_gradient"] != pytest.approx(mixture["global_rejection_expected_gradient"])


@pytest.mark.parametrize("uniform", [0,1])
def test_uniform_prompt_cannot_fill_a_mixed_group_quota(uniform):
    result = m.rejection_mixture([uniform,.5],[.5,.5],8)
    assert result["acceptance"] > 0  # Global proposals can accept the other prompt.
    assert result["per_prompt_quota_status"] == "IMPOSSIBLE_UNIFORM_PROMPT"
    assert result["per_prompt_quota_expected_gradient"] is None
    assert result["impossible_prompt_indices"] == [0]


def test_orthogonal_gradient_order_commutator_and_curvature():
    result = m.sequential_counterexample()
    assert result["gradient_dot"] == 0
    assert result["order_difference"] == pytest.approx(result["hessian_commutator_prediction"])
    assert any(abs(v) > 0 for v in result["order_difference"])
    half = m.sequential_counterexample(.05, .05)
    assert half["order_difference"] == pytest.approx(tuple(v/4 for v in result["order_difference"]))
    interaction = m.composition_counterexample()
    assert interaction["delta_dot"] == interaction["matrix_frobenius_dot"] == 0
    assert interaction["functional_interaction"] == pytest.approx(interaction["hessian_prediction"])
    assert interaction["activation_weighted_cross"] == 99


def test_dual_fisher_proxy_matches_exhaustive_masks():
    result = m.dual_fisher_allocation([1]*4, [9,8,4,1], [10,1,0,0], 2, 1)
    assert result["keep"] == [1,2]
    assert result["proxy_objective"] == result["exhaustive_minimum"]
    assert m.row_ranking_invariance([[4,3,2,1],[1,3,2,4]],[100,.01],2)["unchanged"]


def test_lora_factor_average_is_gauge_dependent():
    result = m.lora_gauge_counterexample()
    assert result["donor1_effective_unchanged"]
    assert result["factor_mean"] != result["gauge_changed_factor_mean"]
    assert result["effective_delta_mean"] != result["factor_mean"]


def test_replay_kl_derivative_matches_finite_difference():
    from math import exp, log
    p0, p = .99, .9
    theta = log(p/(1-p))
    sigmoid = lambda x: 1/(1+exp(-x))
    eps = 1e-5
    finite_diff = (m.bernoulli_kl(p0,sigmoid(theta+eps))-m.bernoulli_kl(p0,sigmoid(theta-eps)))/(2*eps)
    assert finite_diff == pytest.approx(p-p0, abs=1e-9)
    assert m.replay_signal(p0,p)["uniform_success_rl_gradient"] == 0

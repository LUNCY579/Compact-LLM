from __future__ import annotations

import pytest

from rineya_significance_lab.evaluation import (
    cluster_differences,
    clustered_paired_analysis,
    exact_or_sampled_sign_flip,
    normal_approximation_mde,
    validate_cluster_splits,
)


def test_repeated_rows_collapse_to_independent_clusters() -> None:
    clusters = ["a", "a", "b", "b"]
    got = cluster_differences(clusters, [0, 1, 0, 0], [1, 1, 0, 1])
    assert got == {"a": 0.5, "b": 0.5}


def test_exact_sign_flip_uses_clusters_not_rows() -> None:
    p_value, method, samples = exact_or_sampled_sign_flip([1.0] * 6)
    assert method == "exact_cluster_sign_flip"
    assert samples == 64
    assert p_value == pytest.approx(2 / 64)


def test_sampled_sign_flip_is_deterministic_and_never_zero() -> None:
    values = [1.0] * 21
    first = exact_or_sampled_sign_flip(values, monte_carlo_samples=500, seed=17)
    second = exact_or_sampled_sign_flip(values, monte_carlo_samples=500, seed=17)
    assert first == second
    assert 0 < first[0] <= 1


def test_clustered_analysis_enforces_minimum_information_rule() -> None:
    result = clustered_paired_analysis(
        ["a", "a", "b", "b"],
        [0, 0, 0, 0],
        [1, 1, 0, 0],
        minimum_clusters=3,
        bootstrap_samples=200,
        seed=3,
    )
    assert result.cluster_count == 2
    assert not result.eligible
    assert "at least 3" in result.eligibility_reason


def test_cluster_split_leakage_fails() -> None:
    with pytest.raises(ValueError, match="leaks"):
        validate_cluster_splits([("world-1", "train"), ("world-1", "test")])


def test_mde_decreases_with_independent_cluster_count() -> None:
    small = normal_approximation_mde(independent_clusters=8, cluster_difference_sd=0.2)
    large = normal_approximation_mde(independent_clusters=32, cluster_difference_sd=0.2)
    assert large < small

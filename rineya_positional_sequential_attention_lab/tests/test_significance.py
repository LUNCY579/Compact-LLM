"""Phase 4-5/7 tests: significance, bias, distance, low-rank, and the section-37 controls.

Everything model-facing runs on a hand-made 2-layer Qwen3 with random weights, CPU,
float32. No checkpoint is opened and nothing touches the GPU (another process owns it).

The four properties the task brief names are pinned by, in order:

``test_empty_bin_cell_has_no_effect``
    a cell the model provably cannot use leaves ``gold_nll`` unchanged;
``test_group_of_empty_cells_matches_single_cell_reference`` /
``test_group_additivity_error_is_first_order_in_eps``
    multi-cell batching against the one-cell-at-a-time reference, exactly where the
    algebra says it must be exact and approximately where it says it is first order;
``test_reconstruction_error_decreases_in_rank``
    SVD reconstruction error is monotone in rank;
``test_shuffle_bins_preserves_row_multiset`` and friends
    the control matrices preserve exactly the marginals they claim to.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from rineya_psa.hooks.binning import AbsoluteBinning, DistanceBinning
from rineya_psa.loading import arch_from_config
from rineya_psa.significance.bias import (
    BiasContext,
    edge_bias_score,
    measure_positional_bias,
    middle_mass,
    tail_query_mask,
)
from rineya_psa.significance.causal import (
    cost_estimate,
    estimate_causal_significance,
    estimate_causal_significance_ablation,
    estimate_causal_significance_attribution,
    representative_layers,
    select_validation_cells,
    validate_attribution_against_ablation,
    validate_group_additivity,
)
from rineya_psa.significance.distance import (
    DistanceProfile,
    fit_additive_decomposition,
    measure_distance_profile,
    variance_explained_summary,
)
from rineya_psa.significance.lowrank import (
    analyse_low_rank,
    fit_profile_model,
    low_rank_spectrum,
)
from rineya_psa.significance.matrices import SignificanceMatrix, delta_matrices
from rineya_psa.types import BenchmarkItem, TokenizedItem

TINY = dict(
    vocab_size=256,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    rope_theta=1e6,
    max_position_embeddings=512,
    tie_word_embeddings=True,
)

#: 5 prompt tokens + 3 answer tokens. With ``n_bins = 2 * N_TOTAL`` every odd bin is
#: empty, which is what makes "a cell the model provably cannot use" constructible.
N_PROMPT = 5
N_ANSWER = 3
N_TOTAL = N_PROMPT + N_ANSWER
N_BINS_EMPTY = 2 * N_TOTAL


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(20260828)
    model = Qwen3ForCausalLM(Qwen3Config(**TINY)).to(torch.float32).eval()
    return model, arch_from_config(model.config)


class _FakeTokenizer:
    """``prepare_items`` only ever decodes with it, and only for ``objective='correct'``."""

    pad_token_id = 0
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens: bool = True) -> str:  # pragma: no cover
        return " ".join(str(int(i)) for i in ids)


def _pairs(n_items: int = 3, seed: int = 7) -> list[tuple[BenchmarkItem, TokenizedItem]]:
    """``(BenchmarkItem, TokenizedItem)`` pairs with pre-made ids, so no tokenizer runs."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_items):
        ids = tuple(int(x) for x in rng.integers(1, TINY["vocab_size"], N_PROMPT))
        answer = tuple(int(x) for x in rng.integers(1, TINY["vocab_size"], N_ANSWER))
        item = BenchmarkItem(
            item_id=f"item{i}",
            scenario_id=f"scenario{i // 2}",  # two items per scenario, for the clustering
            family="exact_retrieval",
            fmt="natural",
            length_bucket="2k",
            target_positions=(0.5,),
            local_position="middle",
            split="calibration",
            system="",
            user="",
            answer="x",
            answer_aliases=(),
            distractors=(),
            evidence_sentences=(),
            seed=seed,
        )
        tok_item = TokenizedItem(
            item_id=f"item{i}",
            input_ids=ids,
            answer_ids=answer,
            evidence_spans=(),
            actual_positions=(0.5,),
        )
        out.append((item, tok_item))
    return out


def _bias_contexts(n: int = 2, n_tokens: int = 96, seed: int = 3) -> list[BiasContext]:
    rng = np.random.default_rng(seed)
    return [
        BiasContext(
            context_id=f"random_ids-{seed}-{k}",
            kind="random_ids",
            input_ids=tuple(int(x) for x in rng.integers(1, TINY["vocab_size"], n_tokens)),
        )
        for k in range(n)
    ]


def _fake_matrix(seed: int = 0, shape: tuple[int, int, int] = (4, 3, 6)) -> SignificanceMatrix:
    rng = np.random.default_rng(seed)
    return SignificanceMatrix(
        values=rng.normal(size=shape),
        layers=tuple(range(shape[0])),
        kv_heads=tuple(range(shape[1])),
        bins=tuple(range(shape[2])),
        binning_desc=AbsoluteBinning(shape[2]).describe(),
        objective="gold_nll",
        model_key="tiny",
        provenance={},
    )


# ======================================================================================
# (a) a cell the model provably cannot use
# ======================================================================================


def _empty_and_used_bins() -> tuple[list[int], list[int]]:
    ids = AbsoluteBinning(N_BINS_EMPTY).bin_ids_numpy(N_TOTAL)
    used = sorted(set(int(b) for b in ids))
    empty = [b for b in range(N_BINS_EMPTY) if b not in used]
    return empty, used


def test_binning_leaves_odd_bins_empty():
    empty, used = _empty_and_used_bins()
    assert used == [2 * j for j in range(N_TOTAL)]
    assert empty == [2 * j + 1 for j in range(N_TOTAL)]


def test_empty_bin_cell_has_no_effect(tiny_model):
    """Attenuating a bin no source token occupies must not move ``gold_nll`` at all.

    Bitwise, not merely closely: the Mode-2 gate expands to a vector of exact 1.0s, so
    ``value * s`` is the identity in float32 and the perturbed forward is the baseline
    forward. Any drift here means the bin lookup is off by one somewhere.
    """
    model, _ = tiny_model
    result = estimate_causal_significance_ablation(
        model,
        _FakeTokenizer(),
        _pairs(2),
        layers=[0],
        kv_heads=[0],
        n_bins=N_BINS_EMPTY,
        mode="full",
        eps=0.25,
        objective="gold_nll",
        model_key="tiny",
    )
    empty, used = _empty_and_used_bins()
    values = result.matrix.values[0, 0]
    assert np.all(values[empty] == 0.0), values[empty]
    # And the used bins must actually move, or the test above is vacuous.
    assert np.max(np.abs(values[used])) > 1e-4


def test_empty_bin_cell_has_no_effect_per_item(tiny_model):
    model, _ = tiny_model
    result = estimate_causal_significance_ablation(
        model, _FakeTokenizer(), _pairs(2), layers=[0], kv_heads=[0],
        n_bins=N_BINS_EMPTY, mode="full", model_key="tiny",
    )
    empty, _ = _empty_and_used_bins()
    assert np.all(result.per_item[:, 0, 0, empty] == 0.0)
    assert result.evaluated.all()
    assert result.coverage() == 1.0


# ======================================================================================
# (b) multi-cell batching against the sequential reference
# ======================================================================================


def test_group_of_empty_cells_matches_single_cell_reference(tiny_model):
    """One live cell plus any number of provably dead ones == that cell alone, exactly.

    This is the part of "grouping equals the sequential reference" that is *exact*: the
    dead cells contribute an identity factor to the Mode-2 table, so the grouped forward
    is bit-for-bit the single-cell forward. It pins the batching plumbing without
    depending on the first-order argument that the general case needs.
    """
    model, _ = tiny_model
    empty, used = _empty_and_used_bins()
    live, dead = used[1], empty[:3]
    pairs = _pairs(2)
    report = validate_group_additivity(
        model,
        _FakeTokenizer(),
        pairs,
        cells=[(0, 0, live), *[(0, 0, b) for b in dead]],
        n_bins=N_BINS_EMPTY,
        eps=0.25,
    )
    group = np.asarray(report["per_item_group"])
    summed = np.asarray(report["per_item_sum_of_singles"])
    assert np.allclose(group, summed, rtol=0, atol=0)
    assert report["max_abs_error"] == 0.0
    assert abs(report["group_delta_mean"]) > 1e-6  # the live cell really does something


def test_group_additivity_error_is_first_order_in_eps(tiny_model):
    """Two live cells in one layer: the group delta tracks the sum of singles, and the
    error shrinks with ``eps`` the way a second-order remainder must.

    ``L`` is a non-linear function of ``V``, so this is an approximation and the test
    says so by *measuring* the approximation rather than asserting it away.
    """
    model, _ = tiny_model
    _, used = _empty_and_used_bins()
    pairs = _pairs(2)
    errors = {}
    for eps in (0.4, 0.05):
        report = validate_group_additivity(
            model, _FakeTokenizer(), pairs,
            cells=[(0, 0, used[1]), (0, 0, used[3])],
            n_bins=N_BINS_EMPTY, eps=eps,
        )
        errors[eps] = report["relative_error"]
    assert errors[0.05] < errors[0.4]
    assert errors[0.05] < 0.1


def test_grouped_cells_must_share_a_layer(tiny_model):
    model, _ = tiny_model
    _, used = _empty_and_used_bins()
    with pytest.raises(ValueError, match="within a layer"):
        validate_group_additivity(
            model, _FakeTokenizer(), _pairs(1),
            cells=[(0, 0, used[1]), (1, 0, used[1])],
            n_bins=N_BINS_EMPTY,
        )


def test_grouped_mode_screens_and_refines(tiny_model):
    model, _ = tiny_model
    result = estimate_causal_significance_ablation(
        model, _FakeTokenizer(), _pairs(2), layers=[0], kv_heads=[0, 1],
        n_bins=4, mode="grouped", group_size=2, screen_quantile=0.5, model_key="tiny",
    )
    assert result.evaluated.all()
    assert "screen" in result.notes
    assert result.notes["screen"]["n_screened_out"] > 0
    # Screened cells are recorded as exactly zero, and the screen says so.
    assert "caveat" in result.notes["screen"]


# ======================================================================================
# the ladder, the two tiers, and the cost gate
# ======================================================================================


def test_representative_layers_matches_the_brief():
    assert representative_layers(28) == (2, 7, 12, 17, 22, 27)


def test_ladder_leaves_unvisited_cells_nan_not_zero(tiny_model):
    model, _ = tiny_model
    result = estimate_causal_significance(
        model, _FakeTokenizer(), _pairs(2), None, [0], 4,
        tier="ablation", mode="ladder", model_key="tiny",
    )
    assert result.tier == "ablation"
    assert result.notes["ladder_caveat"]
    unvisited = ~result.evaluated
    if unvisited.any():
        assert np.all(np.isnan(result.matrix.values[unvisited]))


def test_attribution_tracks_ablation_on_a_tiny_model(tiny_model):
    """Tier 1 versus Tier 2 on the same grid. The point is the machinery, not the value.

    A randomly initialised model has no structure for the two tiers to agree *about*,
    so this asserts the comparison runs, is finite, and is bounded — not that the
    correlation clears the acceptance threshold, which is a claim about a trained model.
    """
    model, _ = tiny_model
    pairs = _pairs(3)
    attribution = estimate_causal_significance_attribution(
        model, _FakeTokenizer(), pairs, layers=[0, 1], kv_heads=[0, 1], n_bins=4,
        eps=0.25, model_key="tiny",
    )
    ablation = estimate_causal_significance_ablation(
        model, _FakeTokenizer(), pairs, layers=[0, 1], kv_heads=[0, 1], n_bins=4,
        mode="full", eps=0.25, model_key="tiny",
    )
    assert attribution.tier == "attribution"
    assert np.isfinite(attribution.matrix.values).all()
    assert attribution.n_forwards == len(pairs)

    report = validate_attribution_against_ablation(attribution, ablation, n_resamples=200)
    assert report["n_cells"] == 16
    assert -1.0 <= report["spearman"] <= 1.0
    assert report["verdict"] in ("attribution_usable", "fallback_to_coarse_to_fine_ablation")

    subsets = select_validation_cells(attribution, k=4)
    assert len(subsets["top"]) == 4
    assert not set(subsets["top"]) & set(subsets["control"])


def test_attribution_first_order_matches_a_small_eps_ablation(tiny_model):
    """``I_hat = eps * dL/ds`` is the linear response, so it must track a *small*-``eps``
    ablation far better than a large one. That is the only claim attribution makes."""
    model, _ = tiny_model
    pairs = _pairs(2)
    kwargs = dict(layers=[0], kv_heads=[0], n_bins=4, model_key="tiny")
    close = None
    for eps in (0.5, 0.01):
        attribution = estimate_causal_significance_attribution(
            model, _FakeTokenizer(), pairs, eps=eps, **kwargs
        )
        ablation = estimate_causal_significance_ablation(
            model, _FakeTokenizer(), pairs, mode="full", eps=eps, **kwargs
        )
        scale = float(np.max(np.abs(ablation.matrix.values))) or 1.0
        error = float(np.max(np.abs(attribution.matrix.values - ablation.matrix.values))) / scale
        if close is None:
            close = error
        else:
            assert error < close
            assert error < 0.05


def test_attribution_rejects_a_distance_binning(tiny_model):
    model, _ = tiny_model
    with pytest.raises(ValueError, match="query-dependent"):
        estimate_causal_significance_attribution(
            model, _FakeTokenizer(), _pairs(1), binning=DistanceBinning.uniform(4)
        )


def test_cost_estimate_flags_the_brute_force_budget():
    ladder = cost_estimate(n_items=256, n_layers=28, n_kv_heads=8, n_bins=8, tier="ablation")
    attribution = cost_estimate(
        n_items=256, n_layers=28, n_kv_heads=8, n_bins=8, tier="attribution"
    )
    assert ladder["brute_force_forwards"] == 256 * (1 + 28 * 8 * 8)
    assert ladder["speedup_vs_brute_force"] > 1.0
    assert attribution["estimated_gpu_hours"] < ladder["estimated_gpu_hours"]
    assert ladder["exceeds_budget"] is (ladder["estimated_gpu_hours"] > 2.0)


def test_result_round_trips_through_npz(tiny_model, tmp_path):
    from rineya_psa.significance.causal import SignificanceResult

    model, _ = tiny_model
    result = estimate_causal_significance_ablation(
        model, _FakeTokenizer(), _pairs(2), layers=[0], kv_heads=[0], n_bins=4,
        mode="full", model_key="tiny",
    )
    path = result.save(tmp_path / "sig.npz")
    assert path.with_name(path.name + ".provenance.json").exists()
    loaded = SignificanceResult.load(path)
    assert loaded.item_ids == result.item_ids
    assert loaded.scenario_ids == result.scenario_ids
    np.testing.assert_allclose(loaded.per_item, result.per_item)
    np.testing.assert_allclose(loaded.matrix.values, result.matrix.values)


def test_bootstrap_cell_clusters_on_scenarios(tiny_model):
    from rineya_psa.significance.causal import bootstrap_cell

    model, _ = tiny_model
    result = estimate_causal_significance_ablation(
        model, _FakeTokenizer(), _pairs(4), layers=[0], kv_heads=[0], n_bins=4,
        mode="full", model_key="tiny",
    )
    boot = bootstrap_cell(result, (0, 0, 1), n=200, seed=1)
    assert boot.n_items == 4
    assert boot.n_scenarios == 2  # two items share each scenario_id
    assert boot.ci_low <= boot.estimate <= boot.ci_high


# ======================================================================================
# bias (brief section 13)
# ======================================================================================


def test_positional_bias_rows_are_probability_distributions(tiny_model):
    model, arch = tiny_model
    bias = measure_positional_bias(
        model, contexts=_bias_contexts(), n_bins=4, chunk=32, model_key="tiny"
    )
    assert bias.shape == (arch.num_layers, arch.num_q_heads, 4)
    np.testing.assert_allclose(bias.attn_by_bin.sum(axis=-1), 1.0, atol=1e-5)
    assert np.isfinite(bias.qk_logit_mean).all()
    # Entropy of a distribution over at most T sources cannot exceed log T.
    assert (bias.entropy >= 0).all()
    assert (bias.entropy <= np.log(96) + 1e-6).all()


def test_positional_bias_kv_aggregate_is_a_group_mean(tiny_model):
    model, arch = tiny_model
    bias = measure_positional_bias(
        model, contexts=_bias_contexts(), n_bins=4, chunk=32, model_key="tiny"
    )
    kv = bias.to_kv_heads("attn_by_bin")
    assert kv.shape == (arch.num_layers, arch.num_kv_heads, 4)
    for h in range(arch.num_kv_heads):
        members = arch.q_heads_for_kv(h)
        np.testing.assert_allclose(kv[:, h], bias.attn_by_bin[:, list(members)].mean(axis=1))
    # The approximation's cost is reported, not hidden.
    assert bias.kv_head_dispersion().shape == (arch.num_layers, arch.num_kv_heads)


def test_tail_query_mask_keeps_only_the_tail():
    mask = tail_query_mask(100, 0.25).numpy()
    assert mask.sum() == 25
    assert mask[-25:].all() and not mask[:-25].any()
    assert tail_query_mask(100, 1.0).numpy().all()
    assert tail_query_mask(3, 0.01).numpy().sum() == 1  # never empty


def test_edge_bias_and_middle_mass_are_computed_on_the_right_bins():
    # Six bins, all mass on the two ends: maximally edge-biased, no middle mass.
    values = np.array([[[0.5, 0.0, 0.0, 0.0, 0.0, 0.5]]])
    assert edge_bias_score(values) == pytest.approx(0.25)
    assert middle_mass(values, width=0.5) == pytest.approx(0.0)
    flat = np.full((1, 1, 6), 1.0 / 6.0)
    assert edge_bias_score(flat) == pytest.approx(0.0)
    assert middle_mass(flat, width=0.5, reduce="mean") == pytest.approx(1.0 / 6.0)
    with pytest.raises(ValueError):
        edge_bias_score(values, edge_fraction=0.9)


def test_bias_round_trips_through_npz(tiny_model, tmp_path):
    from rineya_psa.significance.bias import PositionalBias

    model, _ = tiny_model
    bias = measure_positional_bias(
        model, contexts=_bias_contexts(), n_bins=4, chunk=32, model_key="tiny"
    )
    path = bias.save_npz(tmp_path / "bias.npz")
    loaded = PositionalBias.load(path)
    np.testing.assert_allclose(loaded.attn_by_bin, bias.attn_by_bin)
    np.testing.assert_allclose(loaded.entropy, bias.entropy)
    assert loaded.provenance["query_tail_fraction"] == bias.provenance["query_tail_fraction"]


# ======================================================================================
# distance (brief section 14)
# ======================================================================================


def test_joint_table_is_a_partition_of_the_attention_mass(tiny_model):
    model, arch = tiny_model
    profile = measure_distance_profile(
        model, contexts=_bias_contexts(), n_absolute_bins=4, n_distance_bins=4,
        chunk=32, model_key="tiny",
    )
    assert profile.joint_mass.shape == (arch.num_layers, arch.num_q_heads, 4, 4)
    np.testing.assert_allclose(profile.joint_mass.sum(axis=(2, 3)), 1.0, atol=1e-5)
    np.testing.assert_allclose(
        profile.distance_marginal(), profile.joint_mass.sum(axis=2), atol=0
    )
    np.testing.assert_allclose(
        profile.absolute_marginal(), profile.joint_mass.sum(axis=3), atol=0
    )


def test_structurally_impossible_pairs_have_zero_count(tiny_model):
    """A source in the last absolute quarter cannot be far away from any query row.

    Those pairs must carry zero allowed cells (so the fit excludes them) *and* zero mass
    (so nothing leaked into them). Both, or the joint binning is wrong.
    """
    model, _ = tiny_model
    profile = measure_distance_profile(
        model, contexts=_bias_contexts(), n_absolute_bins=4, n_distance_bins=4,
        chunk=32, model_key="tiny",
    )
    impossible = profile.cell_counts == 0.0
    assert impossible[3, 1:].all()
    assert np.all(profile.joint_mass[:, :, impossible] == 0.0)
    assert np.all(np.isnan(profile.weight_per_cell[:, :, impossible]))


def test_additive_decomposition_recovers_a_planted_additive_table():
    absolute = np.array([1.0, -2.0, 0.5, 0.5])
    distance = np.array([0.3, -0.1, -0.2])
    table = 5.0 + absolute[:, None] + distance[None, :]
    fit = fit_additive_decomposition(np.stack([table, 2.0 * table]))
    np.testing.assert_allclose(fit.mu, [5.0, 10.0], atol=1e-9)
    np.testing.assert_allclose(fit.absolute[0], absolute, atol=1e-9)
    np.testing.assert_allclose(fit.distance[0], distance, atol=1e-9)
    np.testing.assert_allclose(fit.r2_additive, 1.0, atol=1e-9)
    # The identifiability constraint the docstring promises.
    np.testing.assert_allclose(fit.absolute.sum(axis=-1), 0.0, atol=1e-9)
    np.testing.assert_allclose(fit.distance.sum(axis=-1), 0.0, atol=1e-9)


def test_additive_decomposition_reports_an_interaction_it_cannot_fit():
    rng = np.random.default_rng(0)
    interaction = rng.normal(size=(6, 5))
    fit = fit_additive_decomposition(interaction[None])
    summary = variance_explained_summary(fit)
    assert 0.0 < summary["additive_pooled_r2"] < 1.0
    assert summary["residual_after_additive"] > 0.0
    assert summary["additive_pooled_r2"] >= summary["absolute_only_pooled_r2"] - 1e-12
    assert summary["additive_pooled_r2"] >= summary["distance_only_pooled_r2"] - 1e-12


def test_additive_decomposition_ignores_zero_weight_cells():
    absolute = np.array([1.0, -1.0, 0.0])
    distance = np.array([0.5, -0.5])
    table = absolute[:, None] + distance[None, :]
    holed = table.copy()
    holed[0, 0] = 1e6  # a structurally empty cell filled with nonsense
    weights = np.ones_like(table)
    weights[0, 0] = 0.0
    fit = fit_additive_decomposition(holed[None], weights)
    np.testing.assert_allclose(fit.r2_additive, 1.0, atol=1e-8)


def test_distance_profile_round_trips_through_npz(tiny_model, tmp_path):
    model, _ = tiny_model
    profile = measure_distance_profile(
        model, contexts=_bias_contexts(), n_absolute_bins=4, n_distance_bins=4,
        chunk=32, model_key="tiny",
    )
    path = profile.save_npz(tmp_path / "dist.npz")
    loaded = DistanceProfile.load(path)
    np.testing.assert_allclose(loaded.joint_mass, profile.joint_mass)
    np.testing.assert_allclose(loaded.cell_counts, profile.cell_counts)
    assert loaded.to_distance_matrix().shape == profile.to_distance_matrix().shape


def test_chunking_does_not_change_the_joint_table(tiny_model):
    """The accumulator reconstructs each chunk's query offset by counting rows. If that
    bookkeeping were wrong, changing the chunk size would move the distance bins."""
    model, _ = tiny_model
    contexts = _bias_contexts(1, n_tokens=64)
    a = measure_distance_profile(
        model, contexts=contexts, n_absolute_bins=4, n_distance_bins=4, chunk=64
    )
    b = measure_distance_profile(
        model, contexts=contexts, n_absolute_bins=4, n_distance_bins=4, chunk=7
    )
    np.testing.assert_allclose(a.joint_mass, b.joint_mass, atol=1e-6)
    np.testing.assert_allclose(a.cell_counts, b.cell_counts, atol=1e-9)


# ======================================================================================
# (c) low rank (brief section 21)
# ======================================================================================


def test_reconstruction_error_decreases_in_rank():
    rng = np.random.default_rng(11)
    gains = rng.normal(size=(24, 1))
    shape = np.sin(np.linspace(0.0, 3.0, 8))[None, :]
    values = gains * shape + 0.05 * rng.normal(size=(24, 8))
    spectrum = low_rank_spectrum(values, name="I", ranks=(1, 2, 4, 8))
    errors = [spectrum.reconstruction_error(k) for k in range(spectrum.max_rank + 1)]
    assert all(b <= a + 1e-12 for a, b in zip(errors, errors[1:], strict=False))
    assert errors[-1] == pytest.approx(0.0, abs=1e-10)
    cumulative = [spectrum.cumulative_at(k) for k in (1, 2, 4, 8)]
    assert all(b >= a - 1e-12 for a, b in zip(cumulative, cumulative[1:], strict=False))
    assert cumulative[0] > 0.9  # one planted direction dominates


def test_spectrum_drops_incomplete_rows_rather_than_imputing():
    values = np.arange(24, dtype=np.float64).reshape(4, 6)
    values[2, 3] = np.nan
    spectrum = low_rank_spectrum(values, name="ladder")
    assert spectrum.n_rows_dropped == 1
    assert spectrum.shape == (3, 6)
    with pytest.raises(ValueError, match="incomplete rows"):
        spectrum.reconstruct_matrix(1, _fake_matrix(shape=(1, 4, 6)))


def test_profile_model_agrees_with_the_rank_one_svd_of_the_row_centred_matrix():
    """The cross-check the module docstring promises: with ``c_h`` free, ALS is solving
    exactly the best rank-1 approximation of the row-centred matrix."""
    rng = np.random.default_rng(5)
    values = rng.normal(size=(30, 9))
    model = fit_profile_model(values)
    centred = values - values.mean(axis=1, keepdims=True)
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    reference = values.mean(axis=1, keepdims=True) + np.outer(u[:, 0] * s[0], vt[0])
    np.testing.assert_allclose(model.fitted(), reference, atol=1e-8)
    assert model.converged
    np.testing.assert_allclose(model.profile.sum(), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(model.profile), 1.0, atol=1e-10)
    assert model.gain[int(np.argmax(np.abs(model.gain)))] > 0


def test_profile_model_is_exact_on_data_it_can_represent():
    gains = np.linspace(-2.0, 3.0, 12)
    offsets = np.linspace(0.5, 1.5, 12)
    profile = np.array([1.0, -0.5, 0.0, -0.5, 1.0, -1.0])
    values = gains[:, None] * profile[None, :] + offsets[:, None]
    model = fit_profile_model(values)
    assert model.r2_total == pytest.approx(1.0, abs=1e-10)
    assert model.r2_shape == pytest.approx(1.0, abs=1e-10)
    assert model.n_model_parameters < model.n_free_parameters


def test_profile_model_tolerates_unevaluated_cells():
    rng = np.random.default_rng(6)
    values = rng.normal(size=(2, 5, 7))
    values[0, 1, 2] = np.nan
    model = fit_profile_model(values)
    assert model.n_observed == values.size - 1
    assert np.isfinite(model.fitted()).all()


def test_analysis_writes_both_artifacts(tmp_path):
    analysis = analyse_low_rank(
        {"significance": _fake_matrix(0), "bias": _fake_matrix(1)}, ranks=(1, 2)
    )
    written = analysis.write(tmp_path)
    assert written["svd_json"].name == "significance_svd.json"
    assert written["low_rank_npz"].name == "significance_low_rank_models.npz"
    for path in written.values():
        assert path.with_name(path.name + ".provenance.json").exists()
    with np.load(written["low_rank_npz"]) as archive:
        assert "significance__reconstruction_rank1" in archive
        assert "bias__profile_f" in archive
    payload = analysis.to_json()
    assert set(payload["spectra"]) == {"significance", "bias"}
    assert payload["profile_models"]["bias"]["identifiability"]


def test_rank_reconstruction_folds_back_into_a_matrix():
    matrix = _fake_matrix(2, shape=(3, 2, 5))
    spectrum = low_rank_spectrum(matrix, name="I", ranks=(1,))
    recon = spectrum.reconstruct_matrix(spectrum.max_rank, matrix)
    np.testing.assert_allclose(recon.values, matrix.values, atol=1e-10)
    assert recon.provenance["derived_from"]["transform"].startswith("rank")


# ======================================================================================
# (d) the brief-section-37 control matrices
# ======================================================================================


def test_shuffle_bins_preserves_row_multiset():
    matrix = _fake_matrix(3)
    control = matrix.shuffle_bins(seed=1)
    for li in range(matrix.n_layers):
        for hi in range(matrix.n_heads):
            np.testing.assert_allclose(
                np.sort(control.values[li, hi]), np.sort(matrix.values[li, hi])
            )
    # Per-head amplitude survives; the position-to-value association does not.
    np.testing.assert_allclose(control.values.sum(axis=2), matrix.values.sum(axis=2))
    assert not np.allclose(control.values, matrix.values)
    assert control.provenance["derived_from"]["transform"] == "shuffle_bins"
    np.testing.assert_allclose(
        matrix.shuffle_bins(seed=1).values, control.values
    )  # seeded, hence replayable


def test_shuffle_heads_preserves_layer_and_bin_marginals():
    matrix = _fake_matrix(4)
    control = matrix.shuffle_heads(seed=2)
    np.testing.assert_allclose(control.values.sum(axis=1), matrix.values.sum(axis=1))
    for li in range(matrix.n_layers):
        np.testing.assert_allclose(
            np.sort(control.values[li], axis=0), np.sort(matrix.values[li], axis=0)
        )
    assert not np.allclose(control.values, matrix.values)


def test_sign_flip_preserves_magnitudes_and_inverts_order():
    matrix = _fake_matrix(5)
    control = matrix.sign_flip()
    np.testing.assert_allclose(np.abs(control.values), np.abs(matrix.values))
    np.testing.assert_allclose(control.values, -matrix.values)
    # Importance ordering is exactly reversed, which is the point of the control.
    assert control.top_cells(1)[0][0] == matrix.top_cells(matrix.values.size)[-1][0]


def test_uniform_middle_boost_is_structureless_and_scale_matched():
    matrix = _fake_matrix(6)
    control = matrix.uniform_middle_boost()
    rows = control.values.reshape(-1, control.n_bins)
    assert np.allclose(rows, rows[0])  # every (layer, head) row identical
    nonzero = np.flatnonzero(rows[0])
    assert nonzero.size and nonzero.size < control.n_bins
    centers = control.bin_centers
    assert np.all((centers[nonzero] >= 0.25) & (centers[nonzero] <= 0.75))
    expected = float(np.nanmean(matrix.importance()[:, :, nonzero]))
    assert rows[0][nonzero[0]] == pytest.approx(expected)
    assert control.objective == "importance"


def test_controls_bundle_is_complete_and_seeded():
    matrix = _fake_matrix(7)
    controls = matrix.controls(seed=11)
    assert set(controls) == {"shuffle_bins", "shuffle_heads", "sign_flip", "uniform_middle_boost"}
    again = matrix.controls(seed=11)
    for name, control in controls.items():
        np.testing.assert_allclose(control.values, again[name].values)
        assert control.shape == matrix.shape


def test_delta_matrices_refuses_mismatched_grids():
    a = _fake_matrix(8)
    b = _fake_matrix(9)
    delta = delta_matrices(a, b)
    np.testing.assert_allclose(delta.values, b.values - a.values)
    assert delta.model_key == "tiny-minus-tiny"
    other_binning = dict(AbsoluteBinning(6).describe())
    other_binning["kind"] = "distance"
    with pytest.raises(ValueError, match="binning mismatch"):
        delta_matrices(a, b.replace(binning_desc=other_binning))
    with pytest.raises(ValueError, match="objective mismatch"):
        delta_matrices(a, b.replace(objective="margin"))
    with pytest.raises(ValueError, match="shape mismatch"):
        delta_matrices(a, _fake_matrix(9, shape=(4, 3, 5)))


def test_matrix_csv_round_trip_is_bit_exact(tmp_path):
    """``save_csv`` writes ``repr(float(v))``, not ``repr(numpy.float64)``.

    numpy 2 renders the latter as ``np.float64(0.5)``, which no float parser accepts —
    a silent one-way trip for the only human-readable artifact format the matrix has.
    """
    matrix = _fake_matrix(12)
    values = matrix.values.copy()
    values[1, 1, 3] = np.nan  # an unevaluated ladder cell must survive too
    matrix = matrix.replace(values=values)
    path = matrix.save_csv(tmp_path / "m.csv")
    loaded = SignificanceMatrix.load(path)
    np.testing.assert_array_equal(np.isnan(loaded.values), np.isnan(values))
    finite = ~np.isnan(values)
    np.testing.assert_array_equal(loaded.values[finite], values[finite])
    assert dict(loaded.binning_desc) == dict(matrix.binning_desc)


def test_intervention_specs_cover_the_full_model_grid():
    """A matrix measured on a *subset* of layers must still produce a full-size table,
    with the unmeasured rows left neutral rather than filled with a guess."""
    from rineya_psa.types import Arch

    arch = Arch(
        num_layers=6, hidden_size=32, num_q_heads=4, num_kv_heads=2, head_dim=8, vocab_size=256
    )
    matrix = _fake_matrix(13, shape=(2, 2, 5))
    bias = matrix.to_logit_bias_spec(0.5, arch=arch)
    scale = matrix.to_value_scale_spec(0.25, arch=arch, clip=(0.5, 1.5))
    assert tuple(bias.table.shape) == (6, 2, 5)
    assert tuple(scale.table.shape) == (6, 2, 5)
    assert bias.lam == 0.5
    assert float(bias.table[2:].abs().max()) == 0.0  # unmeasured layers stay neutral
    assert float(scale.table[2:].min()) == 1.0 and float(scale.table[2:].max()) == 1.0
    assert float(scale.table.min()) >= 0.5 and float(scale.table.max()) <= 1.5


# ---------------------------------------------------------------------------
# Direction pinning for the intervention specs.
#
# Regression guard. `to_logit_bias_spec` used to scatter the RAW Deltas, while
# `to_value_scale_spec` used `importance()`. On a loss objective Delta is negative at an
# important cell, so the two modes pointed in opposite directions: Mode 1 suppressed
# attention to exactly the bins the matrix said mattered most, and `sign_flip()` -- a
# negative control -- was the correctly-signed intervention. Nothing in the old tests
# checked direction, only shape.
# ---------------------------------------------------------------------------

import numpy as _np  # noqa: E402
import pytest as _pytest  # noqa: E402

from rineya_psa.significance.matrices import SignificanceMatrix as _SigMat  # noqa: E402


def _direction_matrix(objective: str, important_is_negative: bool) -> _SigMat:
    """One layer, one KV head, four bins; bin 3 is the important cell."""
    peak = -1.0 if important_is_negative else 1.0
    values = _np.array([[[0.5 * -peak, 0.0, 0.0, peak]]], dtype=_np.float64)
    return _SigMat(
        values=values, layers=(0,), kv_heads=(0,), bins=(0, 1, 2, 3),
        binning_desc={"kind": "absolute", "n_bins": 4},
        objective=objective, model_key="direction-test", provenance={},
    )


@_pytest.mark.parametrize(
    ("objective", "important_is_negative"),
    [("gold_nll", True), ("margin", False), ("correct", False)],
)
def test_both_intervention_modes_boost_the_same_cell(objective, important_is_negative):
    m = _direction_matrix(objective, important_is_negative)
    best = int(_np.argmax(m.importance()[0, 0]))
    assert best == 3, "fixture is wrong: bin 3 should be the important cell"
    assert m.top_cells(1)[0][0] == (0, 0, 3)

    bias = _np.asarray(m.to_logit_bias_spec(1.0).table)[0, 0]
    scale = _np.asarray(m.to_value_scale_spec(0.5).table)[0, 0]

    assert bias[best] > 0, f"{objective}: Mode 1 must BOOST the most important bin"
    assert scale[best] > 1.0, f"{objective}: Mode 2 must BOOST the most important bin"
    # and both must penalise the actively-harmful bin
    worst = int(_np.argmin(m.importance()[0, 0]))
    assert bias[worst] < 0
    assert scale[worst] < 1.0
    # the two modes must agree in direction on EVERY bin
    agree = _np.sign(bias) == _np.sign(scale - 1.0)
    assert agree.all(), f"{objective}: modes disagree on bins {_np.where(~agree)[0].tolist()}"


def test_sign_flip_control_is_genuinely_the_opposite_of_the_real_thing():
    """The whole point of the control is that it opposes the real intervention."""
    m = _direction_matrix("gold_nll", True)
    real = _np.asarray(m.to_logit_bias_spec(1.0).table)[0, 0]
    ctrl = _np.asarray(m.sign_flip().to_logit_bias_spec(1.0).table)[0, 0]
    nz = real != 0.0
    assert (_np.sign(real[nz]) == -_np.sign(ctrl[nz])).all()

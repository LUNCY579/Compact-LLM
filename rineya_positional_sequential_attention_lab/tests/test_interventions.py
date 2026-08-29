"""Interventions: identity at neutral settings, controls that control, cost that adds up.

Three properties are load-bearing for everything ``interventions/`` produces, and each
gets its own section below.

1. **Every intervention is the identity at its neutral setting.** Each result in the
   study is a difference against an untouched run, so a correction that leaks a small
   offset at ``lam = 0`` puts that offset into every delta in the lab and no other test
   would notice. The RoPE flat schedule is held to *bit* equality, not a tolerance,
   because it swaps ``inv_freq`` for ``inv_freq * 1.0`` and there is no rounding to hide
   behind; the rest are held to SPEC 5.2's float32 tolerance of ``atol=1e-5``.

2. **A control controls.** A shuffled matrix that accidentally kept its structure, or a
   "matched" random gate whose sd is 3% off, is worse than no control at all — it is a
   defence a reviewer can take apart. Each control is checked against the specific
   statistic it claims to preserve.

3. **The compute model is arithmetic, not a vibe.** The FLOPs formula is checked against
   a case small enough to have been worked out by hand in the docstring of this file's
   ``test_flops_matches_hand_computed_small_case``.

The model is a 2-layer Qwen3 with random weights, float32, on CPU. No checkpoint is read
and nothing touches the GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import TINY_QWEN3_CONFIG
from transformers import Qwen3Config, Qwen3ForCausalLM

from rineya_psa.hooks.binning import AbsoluteBinning
from rineya_psa.hooks.registry import psa_session
from rineya_psa.interventions.cheap import (
    CHEAP_LAMBDAS,
    LOW_RANK_RANKS,
    ROPE_SCHEDULES,
    estimate_position_channel_means,
    low_rank_positional_projection,
    position_logit_correction,
    position_mean_centering,
    positional_subspace,
    rope_scale_values,
    rope_scaling_schedule,
)
from rineya_psa.interventions.compute import (
    COMPUTE_METRICS_COLUMNS,
    ComputeConfig,
    arm_projection_counts,
    compute_metrics_row,
    estimate_flops,
    intervention_flops,
    measure_compute,
    psutil_available,
    write_compute_metrics,
)
from rineya_psa.interventions.controls import (
    MANDATORY_CONTROLS,
    all_controls_for,
    assert_controls_complete,
    frozen_gates,
    random_gates,
    random_head_assignment,
    random_position_matrix,
    random_sequential,
    repeat_parallel_stage,
    reversed_functional_ordering,
    sign_flipped,
    uniform_middle_boost,
)
from rineya_psa.interventions.psa import (
    PSA_LAMBDA_GRID,
    PSA_MODES,
    build_psa_context,
    coerce_significance,
    psa_lambda_sweep,
    psa_sweep,
    standardize_significance,
)
from rineya_psa.interventions.sgsa import (
    ALPHA_SCHEDULES,
    MATRIX_CONDITIONS,
    STAGE_PROFILES,
    alpha_schedule,
    build_sgsa,
    build_sgsa_arm,
    sgsa_factorial,
    sgsa_session,
    significance_bin_order,
    significance_head_order,
    stage_gates,
)
from rineya_psa.loading import arch_from_config
from rineya_psa.sequential_attention.arms import arm_four_stage, arm_query_refinement
from rineya_psa.sequential_attention.wrapper import sequential_session
from rineya_psa.types import Arch

#: The canonical toy architecture, owned by ``tests/conftest.py`` so the nine suites
#: that need one cannot drift apart. This file keeps its own seed.
TINY = dict(TINY_QWEN3_CONFIG)
N_TOKENS = 512  # SPEC 5.2 asks the identity invariant to be shown on >= 512 tokens
N_BINS = 8
ATOL = 1e-5
SEED = 20260828


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture():
    """``(model, arch, input_ids, baseline_logits)`` — one random tiny model per module."""
    torch.manual_seed(SEED)
    model = Qwen3ForCausalLM(Qwen3Config(**TINY)).to(torch.float32).eval()
    model.config.use_cache = False  # the sequential wrapper refuses an incremental cache
    arch = arch_from_config(model.config)
    input_ids = torch.randint(0, TINY["vocab_size"], (1, N_TOKENS))
    with torch.no_grad():
        baseline = model(input_ids).logits.detach().clone()
    return model, arch, input_ids, baseline


@pytest.fixture(scope="module")
def matrix(fixture):
    """A significance matrix with real structure: shape ``[2, 2, 8]``, no zero rows."""
    _, arch, _, _ = fixture
    rng = np.random.default_rng(SEED)
    values = rng.normal(size=(arch.num_layers, arch.num_kv_heads, N_BINS))
    # A deliberate U-shape on top of the noise so "the middle matters" is detectable.
    bins = (np.arange(N_BINS) + 0.5) / N_BINS
    values += 2.0 * (bins - 0.5) ** 2
    return coerce_significance(values, binning=AbsoluteBinning(N_BINS), name="test_matrix")


@pytest.fixture(scope="module")
def contexts():
    """Control contexts for the channel-mean estimator: short, content-free, fixed."""
    rng = np.random.default_rng(SEED + 1)
    return [
        torch.as_tensor(rng.integers(0, TINY["vocab_size"], size=192), dtype=torch.long)
        for _ in range(3)
    ]


@pytest.fixture(scope="module")
def channel_means(fixture, contexts):
    model, _, _, _ = fixture
    return estimate_position_channel_means(model, contexts, n_bins=N_BINS)


def _logits(model, input_ids) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids).logits.detach().clone()


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


# --------------------------------------------------------------------------------------
# 1. identity at the neutral setting — PSA (brief section 32)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", PSA_MODES)
def test_psa_is_identity_at_zero_lambda(fixture, matrix, mode):
    model, arch, ids, baseline = fixture
    ctx = build_psa_context(matrix, mode, 0.0, arch=arch)
    assert ctx.is_identity(), f"{mode} at lam=0 does not report itself as an identity"
    with psa_session(model, ctx):
        got = _logits(model, ids)
    assert _max_abs(got, baseline) <= ATOL


@pytest.mark.parametrize("mode", PSA_MODES)
def test_psa_moves_logits_at_full_lambda(fixture, matrix, mode):
    """Guards the test above from being vacuously true."""
    model, arch, ids, baseline = fixture
    ctx = build_psa_context(matrix, mode, 1.0, arch=arch)
    assert not ctx.is_identity()
    with psa_session(model, ctx):
        got = _logits(model, ids)
    assert _max_abs(got, baseline) > 10 * ATOL


def test_psa_lambda_sweep_is_anchored_at_baseline(matrix, fixture):
    _, arch, _, _ = fixture
    variants = psa_lambda_sweep(matrix, "value", PSA_LAMBDA_GRID, arch=arch)
    assert [v.lam for v in variants] == list(PSA_LAMBDA_GRID)
    assert variants[0].lam == 0.0 and variants[0].ctx.is_identity()
    assert all(not v.ctx.is_identity() for v in variants[1:])
    assert len({v.label for v in variants}) == len(variants)


def test_psa_sweep_covers_the_mode_by_lambda_grid(matrix, fixture):
    _, arch, _, _ = fixture
    variants = psa_sweep(matrix, arch=arch)
    assert len(variants) == len(PSA_MODES) * len(PSA_LAMBDA_GRID)
    assert len({v.label for v in variants}) == len(variants)


# --------------------------------------------------------------------------------------
# 1. identity at the neutral setting — cheap corrections (brief section 33)
# --------------------------------------------------------------------------------------


def test_cheap_a_logit_correction_identity_and_effect(fixture, matrix):
    model, _, ids, baseline = fixture
    bias = matrix.values * 0.5
    with position_logit_correction(model, bias, 0.0):
        neutral = _logits(model, ids)
    assert _max_abs(neutral, baseline) <= ATOL

    moved = {}
    for lam in CHEAP_LAMBDAS:
        with position_logit_correction(model, bias, lam):
            moved[lam] = _max_abs(_logits(model, ids), baseline)
    assert moved[CHEAP_LAMBDAS[0]] > 10 * ATOL
    # Dose-response: a larger lambda must not move the model less.
    assert moved[max(CHEAP_LAMBDAS)] >= moved[min(CHEAP_LAMBDAS)]


def test_cheap_b_mean_centering_identity_and_effect(fixture, channel_means):
    model, _, ids, baseline = fixture
    assert channel_means.values.shape == (TINY["num_hidden_layers"], N_BINS, TINY["hidden_size"])
    assert int(channel_means.counts.min()) > 0

    with position_mean_centering(model, channel_means, 0.0) as record:
        neutral = _logits(model, ids)
    assert record["n_hooks"] == 0  # lam = 0 installs nothing, so it is exact by construction
    assert torch.equal(neutral, baseline)

    with position_mean_centering(model, channel_means, 1.0) as record:
        moved = _logits(model, ids)
    assert record["n_hooks"] == 2 * TINY["num_hidden_layers"]  # q_proj and k_proj per layer
    assert _max_abs(moved, baseline) > 10 * ATOL


def test_cheap_b_refuses_a_single_token_step(fixture, channel_means):
    """A decode step carries no absolute position, so the hook must raise, not guess."""
    model, _, _, _ = fixture
    one = torch.randint(0, TINY["vocab_size"], (1, 1))
    with position_mean_centering(model, channel_means, 1.0), pytest.raises(RuntimeError):
        _logits(model, one)


@pytest.mark.parametrize("rank", LOW_RANK_RANKS)
def test_cheap_c_low_rank_projection_identity_and_effect(fixture, channel_means, rank):
    model, _, ids, baseline = fixture
    basis = positional_subspace(channel_means, rank)
    assert basis.shape == (TINY["num_hidden_layers"], TINY["hidden_size"], rank)
    for layer in range(basis.shape[0]):
        gram = basis[layer].T @ basis[layer]
        assert np.allclose(gram, np.eye(rank), atol=1e-8), "columns are not orthonormal"

    with low_rank_positional_projection(model, basis, 0.0) as record:
        neutral = _logits(model, ids)
    assert record["n_hooks"] == 0
    assert torch.equal(neutral, baseline)

    with low_rank_positional_projection(model, basis, 1.0) as record:
        moved = _logits(model, ids)
    assert record["rank"] == rank
    assert _max_abs(moved, baseline) > 10 * ATOL


def test_rope_flat_schedule_at_one_is_bit_exact(fixture):
    """The one place a tolerance would be a cop-out: ``inv_freq * 1.0`` is ``inv_freq``."""
    model, _, ids, baseline = fixture
    with rope_scaling_schedule(model, "flat", value=1.0) as record:
        got = _logits(model, ids)
    assert record["scales"] == [1.0] * TINY["num_hidden_layers"]
    assert torch.equal(got, baseline)


@pytest.mark.parametrize("schedule", sorted(ROPE_SCHEDULES))
def test_rope_every_schedule_is_bit_exact_when_lo_equals_hi_equals_one(fixture, schedule):
    """Degenerate endpoints collapse all seven schedules onto the no-op."""
    model, _, ids, baseline = fixture
    with rope_scaling_schedule(model, schedule, lo=1.0, hi=1.0, value=1.0) as record:
        got = _logits(model, ids)
    assert record["scales"] == [1.0] * TINY["num_hidden_layers"]
    assert torch.equal(got, baseline)


def test_rope_schedule_shapes():
    n = 9
    assert set(ROPE_SCHEDULES) == {
        "flat",
        "increasing",
        "decreasing",
        "middle_hump",
        "late_hump",
        "middle_valley",
        "early_late",
    }
    inc = rope_scale_values("increasing", n, lo=0.75, hi=1.25)
    dec = rope_scale_values("decreasing", n, lo=0.75, hi=1.25)
    assert len(inc) == n and inc == tuple(sorted(inc))
    assert dec == tuple(reversed(inc))

    hump = rope_scale_values("middle_hump", n, lo=0.75, hi=1.25)
    valley = rope_scale_values("middle_valley", n, lo=0.75, hi=1.25)
    assert hump.index(max(hump)) == n // 2
    assert valley.index(min(valley)) == n // 2

    late = rope_scale_values("late_hump", n, lo=0.75, hi=1.25)
    assert late.index(max(late)) > n // 2

    early_late = rope_scale_values("early_late", n, lo=0.75, hi=1.25)
    assert early_late[0] == 1.25 and early_late[-1] == 1.25
    assert early_late[n // 2] == 0.75

    with pytest.raises(ValueError):
        rope_scale_values("nope", n)


def test_rope_schedules_actually_change_the_model(fixture):
    model, _, ids, baseline = fixture
    for schedule in sorted(set(ROPE_SCHEDULES) - {"flat"}):
        with rope_scaling_schedule(model, schedule, lo=0.5, hi=1.5):
            got = _logits(model, ids)
        assert _max_abs(got, baseline) > 10 * ATOL, schedule


# --------------------------------------------------------------------------------------
# 1. identity at the neutral setting — SGSA (brief section 31)
# --------------------------------------------------------------------------------------


def test_sgsa_single_stage_at_zero_lambda_reproduces_baseline(fixture, matrix):
    """One stage, alpha 1, neutral matrix: SGSA collapses onto stock attention."""
    model, arch, ids, baseline = fixture
    config = build_sgsa(arch, matrix, n_stages=1, condition="sequential", lam=0.0)
    assert config.is_identity_matrix()
    with sgsa_session(model, "all", config):
        got = _logits(model, ids)
    assert torch.equal(got, baseline)


@pytest.mark.parametrize("condition", MATRIX_CONDITIONS)
def test_sgsa_neutral_matrix_equals_the_plain_sequential_arm(fixture, matrix, condition):
    """The matrix is the only difference from ``sequential_attention``; at lam=0 it vanishes."""
    model, arch, ids, _ = fixture
    config = build_sgsa(arch, matrix, n_stages=4, condition=condition, lam=0.0)
    with sequential_session(model, "all", config.arm):
        plain = _logits(model, ids)
    with sgsa_session(model, "all", config):
        got = _logits(model, ids)
    assert torch.equal(got, plain)


@pytest.mark.parametrize("condition", ["global", "sequential"])
def test_sgsa_matrix_moves_the_arm(fixture, matrix, condition):
    model, arch, ids, _ = fixture
    config = build_sgsa(arch, matrix, n_stages=4, condition=condition, lam=1.0)
    assert not config.is_identity_matrix()
    with sequential_session(model, "all", config.arm):
        plain = _logits(model, ids)
    with sgsa_session(model, "all", config):
        got = _logits(model, ids)
    assert _max_abs(got, plain) > 10 * ATOL


def test_sgsa_global_and_sequential_conditions_differ(fixture, matrix):
    """If the two matrix conditions coincided, the section-28 factorial would be a lie."""
    model, arch, ids, _ = fixture
    outputs = {}
    for condition in ("global", "sequential"):
        config = build_sgsa(arch, matrix, n_stages=4, condition=condition, lam=1.0)
        with sgsa_session(model, "all", config):
            outputs[condition] = _logits(model, ids)
    assert _max_abs(outputs["global"], outputs["sequential"]) > 10 * ATOL


def test_sgsa_single_stage_global_matches_psa_value_mode(fixture, matrix):
    """Cross-check: a one-stage SGSA *is* PSA Mode 2, reached by a different code path.

    SGSA multiplies ``value`` inside the wrapper; PSA multiplies it inside the ``"psa"``
    kernel. They disagree only through the mask representation (``eager_mask``'s explicit
    additive float mask versus SDPA's causal shortcut), which is a float32 rounding
    difference, so the two are held to the SPEC tolerance rather than to bit equality.
    """
    model, arch, ids, _ = fixture
    lam = 0.75
    config = build_sgsa(arch, matrix, n_stages=1, condition="global", lam=lam)
    with sgsa_session(model, "all", config):
        via_sgsa = _logits(model, ids)
    ctx = build_psa_context(matrix, "value", lam, arch=arch)
    with psa_session(model, ctx):
        via_psa = _logits(model, ids)
    assert _max_abs(via_sgsa, via_psa) <= ATOL


def test_sgsa_install_is_fully_reversible(fixture, matrix):
    model, arch, ids, baseline = fixture
    originals = [block.self_attn for block in model.model.layers]
    config = build_sgsa(arch, matrix, n_stages=2, condition="sequential", lam=1.0)
    with sgsa_session(model, [0], config) as handle:
        assert handle.active
    assert [block.self_attn for block in model.model.layers] == originals
    assert torch.equal(_logits(model, ids), baseline)


def test_sgsa_session_restores_after_an_exception(fixture, matrix):
    model, arch, ids, baseline = fixture
    originals = [block.self_attn for block in model.model.layers]
    config = build_sgsa(arch, matrix, n_stages=2, condition="sequential", lam=1.0)
    with pytest.raises(RuntimeError, match="boom"), sgsa_session(model, "all", config):
        raise RuntimeError("boom")
    assert [block.self_attn for block in model.model.layers] == originals
    assert torch.equal(_logits(model, ids), baseline)


def test_sgsa_tables_are_indexed_by_absolute_layer(fixture, matrix):
    """``tables[idx]`` is keyed on the model's layer index, not on position in the install.

    Installed on layer 1 alone, a config whose layer-1 row is neutral must reproduce the
    matrix-free arm, and one whose layer-*0* row is neutral must not. Indexing the stack
    by enumeration order would swap those two outcomes, and on a full-stack install the
    two indexings coincide — so a partial install is the only way to see the difference.
    """
    model, arch, ids, _ = fixture
    real = build_sgsa(arch, matrix, n_stages=2, condition="sequential", lam=1.0)
    unmatrixed = build_sgsa(arch, matrix, n_stages=2, condition="none")
    assert unmatrixed.arm.groups == real.arm.groups
    assert unmatrixed.arm.alphas == real.arm.alphas

    def _with_neutral_layer(layer: int):
        tables = real.tables.copy()
        tables[layer] = 1.0
        return type(real)(
            label=f"{real.label}:neutral{layer}",
            arm=real.arm,
            condition=real.condition,
            lam=real.lam,
            tables=tables,
            binning=real.binning,
        )

    with sgsa_session(model, [1], unmatrixed):
        no_matrix = _logits(model, ids)
    with sgsa_session(model, [1], _with_neutral_layer(1)):
        neutral_on_the_installed_layer = _logits(model, ids)
    with sgsa_session(model, [1], _with_neutral_layer(0)):
        neutral_on_the_other_layer = _logits(model, ids)

    assert torch.equal(neutral_on_the_installed_layer, no_matrix)
    assert _max_abs(neutral_on_the_other_layer, no_matrix) > 10 * ATOL


# --------------------------------------------------------------------------------------
# section 27: stage-dependent residual strength
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(ALPHA_SCHEDULES))
def test_alpha_schedules_are_normalised_and_positive(name):
    for n in (1, 2, 4, 5, 16):
        alphas = alpha_schedule(name, n, total=1.0)
        assert len(alphas) == n
        assert all(a > 0.0 for a in alphas)
        assert alpha_schedule(name, n, total=1.0) == alphas  # deterministic
        assert sum(alphas) == pytest.approx(1.0)
        assert sum(alpha_schedule(name, n, total=float(n))) == pytest.approx(float(n))


def test_alpha_schedule_shapes():
    assert alpha_schedule("uniform", 4) == pytest.approx((0.25,) * 4)
    inc = alpha_schedule("increasing", 4)
    dec = alpha_schedule("decreasing", 4)
    assert list(inc) == sorted(inc)
    assert list(dec) == sorted(dec, reverse=True)
    assert inc == pytest.approx(tuple(reversed(dec)))
    mid = alpha_schedule("middle_heavy", 5)
    assert mid.index(max(mid)) == 2
    assert mid == pytest.approx(tuple(reversed(mid)))
    # n_stages = 1 collapses every schedule onto the same single coefficient.
    assert {alpha_schedule(n, 1) for n in ALPHA_SCHEDULES} == {(1.0,)}
    with pytest.raises(ValueError):
        alpha_schedule("nope", 4)


def test_alpha_schedule_reproduces_the_two_mandated_conventions():
    from rineya_psa.sequential_attention.arms import alphas_for

    assert alpha_schedule("uniform", 4, total=1.0) == pytest.approx(
        alphas_for(4, "compute_matched")
    )
    assert alpha_schedule("uniform", 4, total=4.0) == pytest.approx(alphas_for(4, "naive"))


def test_build_sgsa_arm_carries_the_schedule_into_the_arm(fixture):
    _, arch, _, _ = fixture
    arm = build_sgsa_arm(arch, 2, schedule="increasing")
    assert arm.n_stages == 2
    assert list(arm.alphas) == sorted(arm.alphas)
    assert sum(arm.alphas) == pytest.approx(1.0)
    arm.validate_against(arch)


# --------------------------------------------------------------------------------------
# stage gates and orderings
# --------------------------------------------------------------------------------------


def test_stage_gates_properties():
    n_stages, n_bins = 4, 8
    assert np.all(stage_gates(n_stages, n_bins, "replicate") == 1.0)

    ramp = stage_gates(n_stages, n_bins, "ramp")
    assert np.allclose(ramp[:, 0], [0.25, 0.5, 0.75, 1.0])
    assert np.all(ramp[0] == ramp[0, 0])  # no positional structure, only dose

    cumulative = stage_gates(n_stages, n_bins, "cumulative")
    counts = cumulative.sum(axis=1)
    assert list(counts) == [2.0, 4.0, 6.0, 8.0]
    # Nested: every bin a stage sees is still seen by every later stage.
    for s in range(n_stages - 1):
        assert np.all(cumulative[s] <= cumulative[s + 1])
    assert np.all(cumulative[-1] == 1.0)

    partition = stage_gates(n_stages, n_bins, "partition")
    assert np.all(partition.sum(axis=0) == 1.0)  # disjoint and covering
    assert np.all(partition.sum(axis=1) == 2.0)

    order = (7, 6, 5, 4, 3, 2, 1, 0)
    reordered = stage_gates(n_stages, n_bins, "cumulative", bin_order=order)
    assert list(np.flatnonzero(reordered[0])) == [6, 7]

    with pytest.raises(ValueError):
        stage_gates(4, 8, "nope")
    with pytest.raises(ValueError):
        stage_gates(9, 8, "partition")
    with pytest.raises(ValueError):
        stage_gates(4, 8, "cumulative", bin_order=(0, 0, 1, 2, 3, 4, 5, 6))
    assert set(STAGE_PROFILES) == {"replicate", "ramp", "cumulative", "partition"}


def test_significance_orderings_rank_by_strength(fixture):
    _, arch, _, _ = fixture
    values = np.zeros((arch.num_layers, arch.num_kv_heads, 4))
    values[:, :, 2] = 5.0  # bin 2 is the strongest everywhere
    values[:, :, 0] = 1.0
    values[:, 1, :] += 10.0  # kv head 1 dominates
    table = coerce_significance(values, binning=AbsoluteBinning(4))
    assert significance_bin_order(table)[0] == 2
    order = significance_head_order(table, arch)
    assert sorted(order) == list(range(arch.num_q_heads))
    assert set(order[: arch.kv_group_size]) == set(arch.q_heads_for_kv(1))


def test_sgsa_factorial_does_not_duplicate_the_none_condition(fixture, matrix):
    _, arch, _, _ = fixture
    cells = sgsa_factorial(arch, matrix, n_stages=2, lams=(0.5, 1.0), schedules=("uniform",))
    by_condition = {c: 0 for c in MATRIX_CONDITIONS}
    for cell in cells:
        by_condition[cell.condition] += 1
    assert by_condition == {"none": 1, "global": 2, "sequential": 2}
    assert len({c.label for c in cells}) == len(cells)
    for cell in cells:
        cell.validate_against(arch)


# --------------------------------------------------------------------------------------
# 2. controls that control (brief sections 36-37)
# --------------------------------------------------------------------------------------


def test_uniform_middle_boost_has_no_layer_or_head_structure(matrix):
    control = uniform_middle_boost(matrix)
    assert control.shape == matrix.shape
    flat = control.values.reshape(-1, control.n_bins)
    assert np.allclose(flat, flat[0]), "the boost must be identical in every layer and head"
    profile = flat[0]
    assert profile.mean() == pytest.approx(0.0, abs=1e-12)  # a reallocation, not a gain
    assert int(np.argmax(profile)) in (N_BINS // 2 - 1, N_BINS // 2)
    boosted = control.meta["boosted_bins"]
    assert boosted and all(profile[b] == profile[boosted[0]] for b in boosted)
    # Dose matched to the reference's RMS deviation, by default.
    dev = matrix.values - matrix.values.mean(axis=-1, keepdims=True)
    assert control.meta["amplitude"] == pytest.approx(float(np.sqrt(np.mean(dev**2))))


def test_random_position_matrix_preserves_every_per_head_value(matrix):
    control = random_position_matrix(matrix, SEED)
    assert np.allclose(np.sort(control.values, axis=-1), np.sort(matrix.values, axis=-1))
    assert np.allclose(control.values.mean(axis=-1), matrix.values.mean(axis=-1))
    assert np.allclose(control.values.std(axis=-1), matrix.values.std(axis=-1))
    assert not np.allclose(control.values, matrix.values), "the bin order did not move"
    # Independent per row: two rows must not have received the same permutation.
    assert not np.allclose(control.values[0, 0], control.values[0, 1])


def test_random_head_assignment_keeps_the_curves_and_moves_the_heads(matrix):
    control = random_head_assignment(matrix, SEED)
    for layer in range(matrix.n_layers):
        original = {tuple(row) for row in matrix.values[layer]}
        shuffled = {tuple(row) for row in control.values[layer]}
        assert original == shuffled, "a curve was altered, not just reassigned"
    assert not np.allclose(control.values, matrix.values)
    for perm in control.meta["permutations"]:
        assert perm != list(range(matrix.n_kv_heads))


def test_sign_flipped_is_an_involution_that_keeps_the_row_mean(matrix):
    control = sign_flipped(matrix)
    assert np.allclose(control.values.mean(axis=-1), matrix.values.mean(axis=-1))
    dev = matrix.values - matrix.values.mean(axis=-1, keepdims=True)
    control_dev = control.values - control.values.mean(axis=-1, keepdims=True)
    assert np.allclose(control_dev, -dev)
    assert np.allclose(sign_flipped(control).values, matrix.values)
    # Standardising commutes with the flip, so the PSA dose is matched too.
    assert np.allclose(
        standardize_significance(control).values, -standardize_significance(matrix).values
    )


def test_frozen_gates_are_constant_across_positions(matrix):
    control = frozen_gates(matrix)
    assert np.allclose(control.values.std(axis=-1), 0.0)
    assert np.allclose(control.values.mean(axis=-1), matrix.values.mean(axis=-1))
    # A frozen gate carries no positional information, so standardising erases it.
    assert np.allclose(standardize_significance(control).values, 0.0)


def test_random_gates_match_mean_and_sd_exactly(matrix):
    control = random_gates(matrix, SEED)
    assert np.allclose(control.values.mean(axis=-1), matrix.values.mean(axis=-1), atol=1e-12)
    assert np.allclose(control.values.std(axis=-1), matrix.values.std(axis=-1), atol=1e-12)
    assert not np.allclose(control.values, matrix.values)
    assert not np.allclose(
        np.sort(control.values, axis=-1), np.sort(matrix.values, axis=-1)
    ), "matched moments, not a permutation — otherwise this duplicates random_position"


def test_control_seeds_are_independent_streams(matrix):
    """Two controls drawn from one seed must not be the same draw wearing two names."""
    a = random_position_matrix(matrix, SEED).values
    b = random_gates(matrix, SEED).values
    assert not np.allclose(np.argsort(a, axis=-1), np.argsort(b, axis=-1))


def test_arm_controls_are_compute_matched_and_differently_ordered(fixture):
    _, arch, _, _ = fixture
    plain = repeat_parallel_stage(arch, 4)
    shuffled = random_sequential(arch, 4, SEED)
    reversed_arm = reversed_functional_ordering(plain, arch=arch)

    assert plain.order == tuple(range(arch.num_q_heads))
    assert reversed_arm.order == tuple(reversed(range(arch.num_q_heads)))
    for arm in (plain, shuffled, reversed_arm):
        arm.validate_against(arch)
        assert arm.n_stages == 4
        assert arm.alphas == plain.alphas
        assert arm.recompute_kv == plain.recompute_kv
        assert [len(g) for g in arm.groups] == [len(g) for g in plain.groups]
    assert shuffled.order != plain.order

    config = ComputeConfig(arch=arch, intermediate_size=TINY["intermediate_size"])
    totals = {estimate_flops(config, 256, arm=arm).total for arm in (plain, shuffled, reversed_arm)}
    assert len(totals) == 1, "the arm controls are supposed to be compute-matched"


def test_reversed_ordering_reverses_the_treatment_not_the_identity(fixture, matrix):
    _, arch, _, _ = fixture
    order = significance_head_order(matrix, arch)
    treatment = build_sgsa_arm(arch, 2, order=order, schedule="decreasing")
    control = reversed_functional_ordering(treatment, arch=arch)
    assert control.order == tuple(reversed(order))
    assert control.alphas == treatment.alphas
    assert [len(g) for g in control.groups] == [len(g) for g in treatment.groups]


def test_all_controls_for_returns_every_mandatory_control(fixture, matrix):
    _, arch, _, _ = fixture
    controls = all_controls_for(matrix, SEED, arch=arch)
    assert set(controls) == set(MANDATORY_CONTROLS)
    assert {c.kind for c in controls.values()} == {"matrix", "arm"}
    for name, control in controls.items():
        assert control.name == name
        assert control.rationale
        assert control.describe()["name"] == name
    assert_controls_complete(controls)
    with pytest.raises(ValueError, match="missing mandatory controls"):
        assert_controls_complete({k: v for k, v in controls.items() if k != "sign_flipped"})


def test_all_controls_for_matches_a_named_treatment_arm(fixture, matrix):
    _, arch, _, _ = fixture
    treatment = build_sgsa_arm(
        arch, 2, order=significance_head_order(matrix, arch), schedule="middle_heavy"
    )
    controls = all_controls_for(matrix, SEED, arch=arch, treatment_arm=treatment)
    for name in ("repeat_parallel_stage", "random_sequential", "reversed_functional_ordering"):
        arm = controls[name].payload
        assert arm.n_stages == treatment.n_stages
        assert arm.alphas == treatment.alphas
        assert arm.recompute_kv == treatment.recompute_kv
    assert controls["reversed_functional_ordering"].payload.order == tuple(
        reversed(treatment.order)
    )


def test_controls_are_reproducible_from_the_seed(matrix):
    assert np.array_equal(
        random_position_matrix(matrix, 7).values, random_position_matrix(matrix, 7).values
    )
    assert not np.array_equal(
        random_position_matrix(matrix, 7).values, random_position_matrix(matrix, 8).values
    )


def test_matrix_controls_run_end_to_end_through_psa(fixture, matrix):
    """A control has to be usable, not just well-shaped."""
    model, arch, ids, baseline = fixture
    controls = all_controls_for(matrix, SEED, arch=arch)
    for name, control in controls.items():
        if control.kind != "matrix":
            continue
        ctx = build_psa_context(control.payload, "value", 1.0, arch=arch)
        with psa_session(model, ctx):
            got = _logits(model, ids)
        if name == "frozen_gates":
            # A constant gate standardises to zero, so it is *exactly* the identity.
            assert torch.equal(got, baseline)
        else:
            assert _max_abs(got, baseline) > 10 * ATOL, name


# --------------------------------------------------------------------------------------
# 3. compute accounting (brief section 36)
# --------------------------------------------------------------------------------------


def test_flops_matches_hand_computed_small_case():
    """T=8 on the TINY arch, worked out term by term.

    H=64, I=128, Hq=4, Hkv=2, D=16 so Dq=64 and Dkv=32; V=512; L=2; k=1; c=1.

        q_proj  = 2*8*64*64                        =    65_536   per layer
        k+v     = 2 * (2*8*64*32)                  =    65_536   per layer
        attn    = 2 * (2*1*4*8*8*16)               =    16_384   per layer
        o_proj  = 2*8*64*64                        =    65_536   per layer
        mlp     = 6*8*64*128                       =   393_216   per layer
        ------------------------------------------------------------------
        per layer                                  =   606_208
        two layers                                 = 1_212_416
        lm_head = 2*1*64*512                       =    65_536
        total                                      = 1_277_952
    """
    arch = Arch(
        num_layers=2, hidden_size=64, num_q_heads=4, num_kv_heads=2, head_dim=16, vocab_size=512
    )
    config = ComputeConfig(arch=arch, intermediate_size=128)
    flops = estimate_flops(config, 8, logits_to_keep=1)
    assert flops.q_proj == 65_536 * 2
    assert flops.kv_proj == 65_536 * 2
    assert flops.attention == 16_384 * 2
    assert flops.o_proj == 65_536 * 2
    assert flops.mlp == 393_216 * 2
    assert flops.lm_head == 65_536
    assert flops.total == 1_277_952
    assert flops.per_token == pytest.approx(1_277_952 / 8)


def test_flops_causal_factor_only_touches_attention():
    arch = Arch(
        num_layers=2, hidden_size=64, num_q_heads=4, num_kv_heads=2, head_dim=16, vocab_size=512
    )
    config = ComputeConfig(arch=arch, intermediate_size=128)
    dense = estimate_flops(config, 8)
    causal = estimate_flops(config, 8, causal_factor=0.5)
    assert causal.attention == dense.attention // 2
    assert causal.total == dense.total - dense.attention // 2
    with pytest.raises(ValueError):
        estimate_flops(config, 8, causal_factor=0.0)


def test_flops_of_a_staged_arm_only_grows_the_projections(fixture):
    """The claim in the module docstring, checked: attention and o_proj are invariant."""
    _, arch, _, _ = fixture
    config = ComputeConfig(arch=arch, intermediate_size=TINY["intermediate_size"])
    parallel = estimate_flops(config, 256)
    arm = arm_four_stage(arch)

    as_implemented = estimate_flops(config, 256, arm=arm)
    assert as_implemented.n_q_passes == 4 and as_implemented.n_kv_passes == 4
    assert as_implemented.q_proj == parallel.q_proj * 4
    assert as_implemented.kv_proj == parallel.kv_proj * 4
    assert as_implemented.attention == parallel.attention
    assert as_implemented.o_proj == parallel.o_proj
    assert as_implemented.mlp == parallel.mlp
    assert as_implemented.total > parallel.total

    algorithmic = estimate_flops(config, 256, arm=arm, as_implemented=False)
    assert algorithmic.n_q_passes == 1 and algorithmic.n_kv_passes == 4
    assert algorithmic.total < as_implemented.total


def test_flops_partial_install_is_billed_pro_rata(fixture):
    _, arch, _, _ = fixture
    config = ComputeConfig(arch=arch, intermediate_size=TINY["intermediate_size"])
    arm = arm_four_stage(arch)
    full = estimate_flops(config, 128, arm=arm)
    half = estimate_flops(config, 128, arm=arm, arm_layers=[0])
    parallel = estimate_flops(config, 128)
    assert parallel.total < half.total < full.total
    assert half.arm_layers == 1
    with pytest.raises(ValueError):
        estimate_flops(config, 128, arm=arm, arm_layers=99)


def test_arm_projection_counts(fixture):
    _, arch, _, _ = fixture
    assert arm_projection_counts(None) == (1, 1)
    four = arm_four_stage(arch)
    assert arm_projection_counts(four) == (4, 4)
    assert arm_projection_counts(four, as_implemented=False) == (1, 4)
    cheap_kv = arm_four_stage(arch, recompute_kv=False)
    assert arm_projection_counts(cheap_kv, as_implemented=False) == (1, 1)
    refine = arm_query_refinement(arch, n_stages=3)
    assert arm_projection_counts(refine) == (3, 1)


def test_intervention_flops(fixture):
    _, arch, _, _ = fixture
    config = ComputeConfig(arch=arch, intermediate_size=TINY["intermediate_size"])
    T = 64
    assert intervention_flops(config, T) == 0
    assert intervention_flops(config, T, modes=["logit_bias"]) == (
        arch.num_q_heads * T * T * arch.num_layers
    )
    assert intervention_flops(config, T, modes=["value_scale"]) == (
        arch.num_kv_heads * T * arch.head_dim * arch.num_layers
    )
    assert intervention_flops(config, T, low_rank=2, qk_targets=2) == (
        2 * (2 * T * arch.hidden_size * 2) * arch.num_layers * 2
    )
    assert intervention_flops(config, T, mean_centering=True, qk_targets=1) == (
        T * arch.hidden_size * arch.num_layers
    )
    with pytest.raises(ValueError):
        intervention_flops(config, T, modes=["nope"])


def test_measure_compute_records_wall_clock_and_host_ram(fixture):
    model, _, ids, _ = fixture
    with measure_compute() as probe:
        _logits(model, ids)
    result = probe.result
    assert result.wall_clock_s > 0.0
    assert result.psutil_available == psutil_available()
    if psutil_available():
        assert result.host_rss_bytes and result.host_rss_bytes > 0
        assert result.host_rss_delta_bytes is not None
    if not torch.cuda.is_available():
        assert result.peak_vram_bytes is None
        assert result.device == "cpu"


def test_measure_compute_finishes_even_when_the_block_raises():
    with pytest.raises(RuntimeError, match="boom"), measure_compute() as probe:
        raise RuntimeError("boom")
    assert probe.result.wall_clock_s >= 0.0


def test_compute_metrics_row_and_csv(fixture, tmp_path):
    model, arch, ids, _ = fixture
    config = ComputeConfig(arch=arch, intermediate_size=TINY["intermediate_size"])
    arm = arm_four_stage(arch)

    with measure_compute() as probe, sequential_session(model, "all", arm):
        _logits(model, ids)
    row = compute_metrics_row(
        "four_stage@512",
        config,
        seq_len=N_TOKENS,
        n_items=1,
        measurement=probe,
        arm=arm,
        model_key="tiny",
        dtype="float32",
        condition="sequential",
        lam=1.0,
    )
    assert set(row) == set(COMPUTE_METRICS_COLUMNS)
    assert row["n_stages"] == 4
    assert row["n_tokens"] == N_TOKENS
    assert row["tokens_per_s"] > 0.0
    assert row["flops_ratio_vs_parallel"] > 1.0
    assert row["flops_total"] == estimate_flops(config, N_TOKENS, arm=arm).total

    parallel_row = compute_metrics_row("parallel@512", config, seq_len=N_TOKENS)
    assert parallel_row["flops_ratio_vs_parallel"] == pytest.approx(1.0)
    assert parallel_row["wall_clock_s"] is None
    assert parallel_row["tokens_per_s"] is None

    target = write_compute_metrics([row, parallel_row], tmp_path / "compute_metrics.csv")
    assert target.exists()
    header = target.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == list(COMPUTE_METRICS_COLUMNS)
    assert target.with_name(target.name + ".provenance.json").exists()

    with pytest.raises(ValueError, match="unknown columns"):
        write_compute_metrics([{**row, "surprise": 1}], tmp_path / "bad.csv")

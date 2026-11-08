"""Sequential attention: identity, partitions, scaling, safety, orderings.

Same setup as ``tests/test_baseline_equivalence.py`` — a hand-made small Qwen3 with
random weights, CPU, float32, no checkpoint read and nothing on the GPU. The identity
test is held to that file's ``atol=1e-5``, because Arm A is the baseline every other
arm is reported as a difference against; a drift there is a silent offset on every
number the phase produces.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from rineya_psa.hooks import HeadScaleSpec, PSAContext, psa_session
from rineya_psa.loading import arch_from_config
from rineya_psa.sequential_attention import (
    ARM_NAMES,
    HeadStats,
    SequentialArm,
    SequentialQwen3Attention,
    StabilityAbort,
    StabilityConfig,
    all_arms,
    all_orders,
    alphas_for,
    arm_four_stage,
    arm_fully_sequential,
    arm_grid,
    arm_parallel,
    arm_query_refinement,
    arm_two_stage,
    assign_roles,
    build_arm,
    compare_profiles,
    install_sequential,
    original_order,
    partition_heads,
    profile_arm,
    profile_forward,
    random_orders,
    role_guided_order,
    role_reverse_order,
    run_arm_with_stability,
    sequential_session,
    significance_descending_order,
)
from rineya_psa.sequential_attention.ordering import ROLE_SEQUENCE, is_permutation
from rineya_psa.significance.matrices import SignificanceMatrix

TINY = dict(
    vocab_size=512,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    rope_theta=1e6,
    max_position_embeddings=1024,
    tie_word_embeddings=True,
)
N_TOKENS = 128
ATOL = 1e-5


@pytest.fixture(scope="module")
def fixture():
    torch.manual_seed(4242)
    model = Qwen3ForCausalLM(Qwen3Config(**TINY)).to(torch.float32).eval()
    model.config.use_cache = False
    arch = arch_from_config(model.config)
    input_ids = torch.randint(0, TINY["vocab_size"], (1, N_TOKENS))
    return model, arch, input_ids


def _logits(model, input_ids) -> torch.Tensor:
    with torch.inference_mode():
        return model(input_ids=input_ids, use_cache=False).logits.clone()


# ---------------------------------------------------------------------------
# Arm A identity
# ---------------------------------------------------------------------------


def test_single_stage_all_heads_alpha_one_is_the_identity(fixture):
    """The mandatory baseline: Arm A on every layer must not move a single logit."""
    model, arch, input_ids = fixture
    reference = _logits(model, input_ids)
    with sequential_session(model, "all", arm_parallel(arch)):
        staged = _logits(model, input_ids)
    assert torch.allclose(reference, staged, atol=ATOL, rtol=0.0)


def test_identity_holds_for_a_hand_built_single_stage_arm(fixture):
    """Not just the factory: any 1-stage/alpha-1 arm is the identity."""
    model, arch, input_ids = fixture
    reference = _logits(model, input_ids)
    arm = SequentialArm(
        name="parallel",
        n_stages=1,
        groups=(tuple(range(arch.num_q_heads)),),
        alphas=(1.0,),
        recompute_kv=True,
    )
    with sequential_session(model, [0, 2], arm):
        staged = _logits(model, input_ids)
    assert torch.allclose(reference, staged, atol=ATOL, rtol=0.0)


def test_uninstall_restores_the_original_module_objects(fixture):
    model, arch, input_ids = fixture
    originals = [block.self_attn for block in model.model.layers]
    handle = install_sequential(model, [1, 3], arm_two_stage(arch))
    assert isinstance(model.model.layers[1].self_attn, SequentialQwen3Attention)
    assert model.model.layers[0].self_attn is originals[0]
    handle.uninstall()
    handle.uninstall()  # idempotent
    assert [block.self_attn for block in model.model.layers] == originals
    assert torch.allclose(_logits(model, input_ids), _logits(model, input_ids))


def test_wrapper_shares_parameters_by_reference(fixture):
    model, arch, _ = fixture
    original = model.model.layers[0].self_attn
    wrapper = SequentialQwen3Attention(original, arm_two_stage(arch))
    assert wrapper.inner is original
    assert wrapper.inner.q_proj.weight is original.q_proj.weight
    assert wrapper.inner.o_proj.weight.data_ptr() == original.o_proj.weight.data_ptr()


def test_session_restores_on_exception(fixture):
    model, arch, _ = fixture
    originals = [block.self_attn for block in model.model.layers]
    with pytest.raises(ZeroDivisionError), sequential_session(model, "all", arm_four_stage(arch)):
        raise ZeroDivisionError
    assert [block.self_attn for block in model.model.layers] == originals


def test_kv_cache_is_refused_not_silently_degraded(fixture):
    model, arch, input_ids = fixture
    with (
        sequential_session(model, "all", arm_two_stage(arch)),
        pytest.raises(RuntimeError, match="incremental KV cache"),
        torch.inference_mode(),
    ):
        model(input_ids=input_ids[:, :16], use_cache=True)


# ---------------------------------------------------------------------------
# partitions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ARM_NAMES)
def test_groups_partition_every_query_head_exactly_once(fixture, name):
    _, arch, _ = fixture
    arm = build_arm(name, arch)
    flat = [head for group in arm.groups for head in group]
    assert sorted(flat) == list(range(arch.num_q_heads))
    assert len(flat) == len(set(flat))
    assert all(group for group in arm.groups)
    arm.validate_against(arch)


def test_partition_heads_is_contiguous_and_front_loads_the_remainder():
    assert partition_heads(range(16), 3) == (
        (0, 1, 2, 3, 4, 5),
        (6, 7, 8, 9, 10),
        (11, 12, 13, 14, 15),
    )
    assert partition_heads((3, 1, 0, 2), 2) == ((3, 1), (0, 2))
    with pytest.raises(ValueError):
        partition_heads(range(3), 4)


def test_a_repeated_head_is_rejected():
    with pytest.raises(ValueError, match="more than one stage"):
        SequentialArm(name="bad", n_stages=2, groups=((0, 1), (1, 2)), alphas=(1.0, 1.0))


def test_arm_d_is_one_head_per_stage(fixture):
    _, arch, _ = fixture
    arm = arm_fully_sequential(arch)
    assert arm.n_stages == arch.num_q_heads
    assert all(len(group) == 1 for group in arm.groups)
    assert arm.kv_projection_cost() == arch.num_q_heads


def test_arm_e_refuses_a_meaningless_kv_recompute(fixture):
    _, arch, _ = fixture
    with pytest.raises(ValueError, match="recompute_kv"):
        arm_query_refinement(arch, recompute_kv=True)
    assert arm_query_refinement(arch).kv_projection_cost() == 1


def test_arm_factories_and_grid_construct_without_error(fixture):
    _, arch, _ = fixture
    assert len(all_arms(arch)) == len(ARM_NAMES)
    grid = arm_grid(arch)
    assert grid
    for arm in grid:
        arm.validate_against(arch)
        assert not (arm.query_refinement and arm.recompute_kv)


# ---------------------------------------------------------------------------
# alphas
# ---------------------------------------------------------------------------


def test_alphas_for_matches_the_two_mandated_conventions():
    assert alphas_for(4, "compute_matched") == (0.25, 0.25, 0.25, 0.25)
    assert alphas_for(4, "naive") == (1.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="alpha mode"):
        alphas_for(4, "whatever")


def test_zero_alphas_write_nothing(fixture):
    """``alpha_s = 0`` for every stage must produce an exactly zero attention write —
    the sanity check that ``alphas`` multiply the stage output and nothing else."""
    model, arch, _ = fixture
    hidden = torch.randn(1, 32, TINY["hidden_size"])
    cos, sin = model.model.rotary_emb(hidden, torch.arange(32).unsqueeze(0))
    arm = SequentialArm(
        name="four_stage",
        n_stages=4,
        groups=partition_heads(range(arch.num_q_heads), 4),
        alphas=(0.0, 0.0, 0.0, 0.0),
        recompute_kv=True,
    )
    wrapper = SequentialQwen3Attention(model.model.layers[0].self_attn, arm)
    with torch.inference_mode():
        out = wrapper(hidden_states=hidden, position_embeddings=(cos, sin))[0]
    assert torch.count_nonzero(out) == 0


def test_first_stage_write_is_exactly_alpha_times_its_unscaled_write(fixture):
    """``alpha`` enters as a multiplier on the stage output, nothing else."""
    model, arch, _ = fixture
    hidden = torch.randn(1, 24, TINY["hidden_size"])
    cos, sin = model.model.rotary_emb(hidden, torch.arange(24).unsqueeze(0))
    groups = partition_heads(range(arch.num_q_heads), 4)

    def first_stage(alpha: float) -> torch.Tensor:
        arm = SequentialArm(
            name="probe",
            n_stages=4,
            groups=groups,
            alphas=(alpha, 0.0, 0.0, 0.0),
            recompute_kv=True,
        )
        wrapper = SequentialQwen3Attention(model.model.layers[0].self_attn, arm)
        with torch.inference_mode():
            return wrapper(hidden_states=hidden, position_embeddings=(cos, sin))[0]

    base = first_stage(1.0)
    assert torch.allclose(first_stage(0.25), 0.25 * base, atol=ATOL, rtol=0.0)
    assert torch.allclose(first_stage(-2.0), -2.0 * base, atol=ATOL, rtol=0.0)


def test_stages_partition_o_proj_exactly(fixture):
    """``sum_s G_s(X) == parallel(X)`` when the recurrence is switched off.

    This is the analytic claim ``arms.py`` makes about residual scaling, and the whole
    ``alphas`` discussion rests on it, so it is pinned rather than asserted in prose.
    """
    model, arch, _ = fixture
    hidden = torch.randn(1, 24, TINY["hidden_size"])
    cos, sin = model.model.rotary_emb(hidden, torch.arange(24).unsqueeze(0))
    groups = partition_heads(range(arch.num_q_heads), 4)
    attn = model.model.layers[1].self_attn

    total = torch.zeros(1, 24, TINY["hidden_size"])
    for stage in range(4):
        alphas = tuple(1.0 if s == stage else 0.0 for s in range(4))
        arm = SequentialArm(
            name="probe", n_stages=4, groups=groups, alphas=alphas, recompute_kv=True
        )
        wrapper = SequentialQwen3Attention(attn, arm)
        with torch.inference_mode():
            total = total + wrapper(hidden_states=hidden, position_embeddings=(cos, sin))[0]

    parallel = SequentialQwen3Attention(attn, arm_parallel(arch))
    with torch.inference_mode():
        reference = parallel(hidden_states=hidden, position_embeddings=(cos, sin))[0]
    assert torch.allclose(total, reference, atol=ATOL, rtol=0.0)


def test_recompute_kv_actually_changes_the_answer(fixture):
    """The expensive path must not be a no-op relative to the cheap control."""
    model, arch, input_ids = fixture
    with sequential_session(model, "all", arm_four_stage(arch, recompute_kv=True)):
        expensive = _logits(model, input_ids)
    with sequential_session(model, "all", arm_four_stage(arch, recompute_kv=False)):
        cheap = _logits(model, input_ids)
    assert not torch.allclose(expensive, cheap, atol=1e-4)


def test_query_refinement_moves_the_logits_without_moving_the_hidden_state(fixture):
    model, arch, input_ids = fixture
    reference = _logits(model, input_ids)
    with sequential_session(model, "all", arm_query_refinement(arch, n_stages=2)):
        refined = _logits(model, input_ids)
    assert torch.isfinite(refined).all()
    assert not torch.allclose(reference, refined, atol=1e-4)


def test_partial_stage_refuses_a_live_head_indexed_psa_context(fixture):
    model, arch, input_ids = fixture
    ctx = PSAContext(enabled=True, head_scale=HeadScaleSpec.zero_head(arch, 0, 1))
    with (
        sequential_session(model, "all", arm_two_stage(arch)),
        psa_session(model, ctx),
        pytest.raises(RuntimeError, match="renumbers query heads"),
    ):
        _logits(model, input_ids)


# ---------------------------------------------------------------------------
# stability
# ---------------------------------------------------------------------------


def test_baseline_profile_compares_to_itself_as_stable(fixture):
    model, _, input_ids = fixture
    profile = profile_forward(model, input_ids, label="parallel_baseline")
    report = compare_profiles(profile, profile)
    assert report.stable
    assert report.violations == ()
    assert np.allclose(report.rms_ratio, 1.0)
    assert report.kl_next_token == pytest.approx(0.0, abs=1e-9)
    assert len(profile.layer_stats) == TINY["num_hidden_layers"]
    assert profile.finite


def test_arm_a_is_stable_and_kl_zero(fixture):
    model, arch, input_ids = fixture
    profile, report = profile_arm(model, input_ids, "all", arm_parallel(arch))
    assert report.stable
    assert report.kl_next_token == pytest.approx(0.0, abs=1e-6)
    assert report.worst_rms_ratio == pytest.approx(1.0, abs=1e-5)


def test_the_detector_fires_on_a_deliberately_blown_up_alpha(fixture):
    """A destabilised arm is a result: the report is populated, not clamped."""
    model, arch, input_ids = fixture
    blown = SequentialArm(
        name="four_stage",
        n_stages=4,
        groups=partition_heads(range(arch.num_q_heads), 4),
        alphas=(60.0, 60.0, 60.0, 60.0),
        recompute_kv=True,
    )
    _, report = profile_arm(model, input_ids, "all", blown)
    assert not report.stable
    assert any("rms_ratio outside" in v for v in report.violations)
    assert report.worst_rms_ratio > 2.0
    assert np.isfinite(report.rms).all()
    # the numbers survive the verdict rather than being replaced by a flag
    assert report.to_json()["rms_ratio"]


def test_run_arm_with_stability_aborts_and_still_carries_the_measurement(fixture):
    model, arch, input_ids = fixture
    blown = SequentialArm(
        name="four_stage",
        n_stages=4,
        groups=partition_heads(range(arch.num_q_heads), 4),
        alphas=(60.0, 60.0, 60.0, 60.0),
        recompute_kv=True,
    )
    scored: list[str] = []
    with pytest.raises(StabilityAbort) as excinfo:
        run_arm_with_stability(
            model, blown, "all", input_ids, score=lambda m: scored.append("ran")
        )
    assert scored == []  # aborted before spending the scoring budget
    assert excinfo.value.result is not None
    assert excinfo.value.result.aborted is True
    assert excinfo.value.result.records is None
    assert not excinfo.value.report.stable


def test_abort_false_scores_the_unstable_arm_anyway(fixture):
    model, arch, input_ids = fixture
    blown = SequentialArm(
        name="four_stage",
        n_stages=4,
        groups=partition_heads(range(arch.num_q_heads), 4),
        alphas=(60.0, 60.0, 60.0, 60.0),
        recompute_kv=True,
    )
    result = run_arm_with_stability(
        model,
        blown,
        "all",
        input_ids,
        score=lambda m: ["record"],
        config=StabilityConfig(abort=False),
    )
    assert result.records == ["record"]
    assert result.aborted is False
    assert result.report.stable is False
    assert result.to_json()["stability"]["violations"]


def test_scoring_runs_with_the_arm_installed(fixture):
    model, arch, input_ids = fixture
    seen: list[bool] = []

    def score(m):
        seen.append(isinstance(m.model.layers[0].self_attn, SequentialQwen3Attention))
        return seen

    run_arm_with_stability(model, arm_parallel(arch), "all", input_ids, score=score)
    assert seen == [True]
    assert not isinstance(model.model.layers[0].self_attn, SequentialQwen3Attention)


def test_stability_config_rejects_an_impossible_band():
    with pytest.raises(ValueError, match="rms_ratio_band"):
        StabilityConfig(rms_ratio_band=(2.0, 0.5))


def test_nonfinite_activations_are_reported_not_swallowed(fixture):
    model, _, input_ids = fixture
    baseline = profile_forward(model, input_ids, label="parallel_baseline")
    broken = profile_forward(model, input_ids, label="broken")
    poisoned = type(broken)(
        label="broken",
        layer_stats=tuple(
            type(s)(layer=s.layer, rms=s.rms, max_abs=s.max_abs, n_nan=3, n_inf=1)
            for s in broken.layer_stats
        ),
        next_token_logprobs=broken.next_token_logprobs,
        logits_max_abs=broken.logits_max_abs,
        n_nonfinite_logits=2,
        n_tokens=broken.n_tokens,
    )
    report = compare_profiles(poisoned, baseline)
    assert not report.stable
    assert any("non-finite activations" in v for v in report.violations)
    assert any("non-finite logits" in v for v in report.violations)


# ---------------------------------------------------------------------------
# orderings
# ---------------------------------------------------------------------------


def _significance_matrix(arch, values=None) -> SignificanceMatrix:
    rng = np.random.default_rng(7)
    if values is None:
        values = rng.normal(size=(arch.num_layers, arch.num_kv_heads, 4))
    return SignificanceMatrix(
        values=values,
        layers=tuple(range(arch.num_layers)),
        kv_heads=tuple(range(arch.num_kv_heads)),
        bins=tuple(range(4)),
        binning_desc={"kind": "absolute", "n_bins": 4},
        objective="gold_nll",
        model_key="tiny",
        provenance={},
    )


def _head_stats(arch) -> list[HeadStats]:
    rng = np.random.default_rng(11)
    stats: list[HeadStats] = []
    for layer in range(arch.num_layers):
        for head in range(arch.num_q_heads):
            stats.append(
                HeadStats(
                    layer=layer,
                    q_head=head,
                    gold_span_attn=float(rng.normal()),
                    ablation_effect=float(rng.normal()),
                    middle_mass=float(rng.normal()),
                    entropy=float(rng.normal()),
                    margin_effect=float(rng.normal()),
                )
            )
    return stats


def test_every_ordering_is_a_valid_permutation(fixture):
    _, arch, _ = fixture
    roles = assign_roles(_head_stats(arch), n_layers=arch.num_layers)
    orders = all_orders(
        arch, matrix=_significance_matrix(arch), roles=roles, n_random=5, seed=3
    )
    names = [o.name for o in orders]
    assert names[0] == "original"
    assert "significance_desc" in names and "role_guided" in names and "role_reverse" in names
    assert len(names) == len(set(names))
    for order in orders:
        assert is_permutation(order.order, arch.num_q_heads)
        order.validate_against(arch)
        assert order.rule


def test_random_orders_are_reproducible_and_distinct(fixture):
    _, arch, _ = fixture
    a = random_orders(arch, n=4, seed=17)
    b = random_orders(arch, n=4, seed=17)
    assert [o.order for o in a] == [o.order for o in b]
    assert [o.meta["seed"] for o in a] == [17, 18, 19, 20]
    assert random_orders(arch, n=4, seed=99)[0].order != a[0].order or arch.num_q_heads < 3


def test_significance_order_is_descending_and_gqa_aware(fixture):
    _, arch, _ = fixture
    values = np.zeros((arch.num_layers, arch.num_kv_heads, 4))
    # The objective is gold_nll, whose OBJECTIVE_SIGN is -1: an important cell has a
    # *negative* Delta. Ordering must go through `importance()`, so a raw -5 here has
    # to come out on top; a -5 that sorted last would mean the sign was dropped.
    values[:, 1, :] = -5.0
    order = significance_descending_order(_significance_matrix(arch, values), arch)
    lead = set(order.order[: arch.kv_group_size])
    assert lead == set(arch.q_heads_for_kv(1))
    assert order.meta["gqa_degenerate"] is (arch.kv_group_size > 1)
    scores = order.meta["scores"]
    assert [scores[h] for h in order.order] == sorted(
        (scores[h] for h in order.order), reverse=True
    )


def test_original_order_is_the_identity(fixture):
    _, arch, _ = fixture
    assert original_order(arch).order == tuple(range(arch.num_q_heads))


def test_role_rule_refuses_heads_that_clear_nothing(fixture):
    _, arch, _ = fixture
    # Every statistic identical -> every z is 0 -> nothing clears a positive threshold.
    flat = [
        HeadStats(layer=layer, q_head=head, gold_span_attn=1.0, ablation_effect=1.0,
                  middle_mass=1.0, entropy=1.0, margin_effect=1.0)
        for layer in range(arch.num_layers)
        for head in range(arch.num_q_heads)
    ]
    roles = assign_roles(flat, n_layers=arch.num_layers)
    assert roles.n_labelled == 0
    assert all(label is None for label in roles.labels)
    # ...and the ordering still covers every head.
    order = role_guided_order(roles, arch)
    assert is_permutation(order.order, arch.num_q_heads)
    assert order.meta["n_unlabelled"] == arch.num_q_heads


def test_role_rule_labels_a_head_that_clears_it(fixture):
    _, arch, _ = fixture
    stats: list[HeadStats] = []
    for layer in range(arch.num_layers):
        for head in range(arch.num_q_heads):
            retrieval = 5.0 if head == 0 else 0.0
            stats.append(
                HeadStats(
                    layer=layer,
                    q_head=head,
                    gold_span_attn=retrieval,
                    ablation_effect=retrieval,
                    middle_mass=0.0,
                    entropy=0.0,
                    margin_effect=0.0,
                )
            )
    roles = assign_roles(stats, n_layers=arch.num_layers)
    assert roles.head_roles()[0][0] == "retrieval"
    assert all(roles.head_roles()[h][0] is None for h in range(1, arch.num_q_heads))
    rows = roles.rows()
    assert {"z_gold_span_attn", "score_retrieval", "passes_retrieval", "role"} <= set(rows[0])
    assert role_guided_order(roles, arch).order[0] == 0


def test_role_guided_and_reverse_disagree_on_direction(fixture):
    _, arch, _ = fixture
    stats: list[HeadStats] = []
    for layer in range(arch.num_layers):
        for head in range(arch.num_q_heads):
            stats.append(
                HeadStats(
                    layer=layer,
                    q_head=head,
                    gold_span_attn=5.0 if head == 0 else 0.0,
                    ablation_effect=5.0 if head == 0 else 0.0,
                    middle_mass=5.0 if head == 1 else 0.0,
                    entropy=5.0 if head == 2 else 0.0,
                    margin_effect=5.0 if head == 3 else 0.0,
                )
            )
    roles = assign_roles(stats, n_layers=arch.num_layers)
    forward = role_guided_order(roles, arch)
    backward = role_reverse_order(roles, arch)
    labelled = [h for h in forward.order if forward.meta["roles"][h] is not None]
    labelled_back = [h for h in backward.order if backward.meta["roles"][h] is not None]
    assert labelled == list(reversed(labelled_back))
    assert is_permutation(backward.order, arch.num_q_heads)
    assert forward.meta["roles"][0] == "retrieval"
    assert ROLE_SEQUENCE.index("retrieval") == 0


def test_an_ordering_drives_the_arm_partition(fixture):
    _, arch, _ = fixture
    order = random_orders(arch, n=1, seed=5)[0]
    arm = arm_two_stage(arch, order=order.order)
    assert arm.order == order.order
    assert arm.groups == partition_heads(order.order, 2)
    arm.validate_against(arch)


def test_role_rule_accepts_head_stats_rows_and_flags_what_it_could_not_measure(fixture):
    """``activation_analysis.head_stats.HeadStatRow`` spellings feed straight in.

    That module names the middle-mass column ``middle_attn`` and has no
    ``margin_effect`` at all, so the verification criterion is unmeasured — which the
    assignment must announce rather than silently reporting zero verification heads.
    """
    _, arch, _ = fixture
    rows = [
        {
            "model_key": "tiny",
            "layer": layer,
            "q_head": head,
            "kv_head": arch.kv_head_for_q(head),
            "n_items": 8,
            "gold_span_attn": 5.0 if head == 0 else 0.0,
            "edge_attn": 0.1,
            "middle_attn": 5.0 if head == 1 else 0.0,
            "recent_attn": 0.1,
            "entropy": 0.2,
            "qk_logit_mean": 0.0,
            "head_out_norm": 1.0,
            "ablation_effect": 5.0 if head == 0 else 0.0,
        }
        for layer in range(arch.num_layers)
        for head in range(arch.num_q_heads)
    ]
    roles = assign_roles(rows, n_layers=arch.num_layers)
    assert roles.head_roles()[0][0] == "retrieval"
    assert roles.head_roles()[1][0] == "association"
    assert roles.unmeasured == ("margin_effect",)
    assert roles.unevaluable_roles == ("verification",)
    assert roles.to_json()["unevaluable_roles"] == ["verification"]
    order = role_guided_order(roles, arch)
    assert order.order[0] == 0
    assert is_permutation(order.order, arch.num_q_heads)


def test_head_stats_from_row_does_not_alias_patch_effect_to_margin_effect():
    row = {"layer": 0, "q_head": 1, "patch_effect": 9.0, "middle_attn": 0.4}
    stats = HeadStats.from_row(row)
    assert np.isnan(stats.margin_effect)
    assert stats.middle_mass == 0.4
    assert HeadStats.from_row(row, margin_effect=2.0).margin_effect == 2.0

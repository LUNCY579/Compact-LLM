from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from rineya_significance_lab.runtime import (
    CLEAN_COUNT_CONTROL_NAMES_V2,
    COUNT_CONTROL_NAMES,
    CountControlSpec,
    RuntimeCountController,
    SignificanceCache,
    SignificancePolicy,
    boundary_resident_positions,
    chunked_prefill,
)


def _controller(name: str, *, case_id: str = "case-1") -> RuntimeCountController:
    kwargs: dict[str, Any] = {}
    if name == "wrong_token":
        kwargs["circular_shift"] = 1
    return RuntimeCountController(
        CountControlSpec(
            condition_id=f"{name}-condition",
            transform=name,  # type: ignore[arg-type]
            seed=23,
            **kwargs,
        ),
        case_id=case_id,
        chunk_size=2,
    )


def _cache(*, length: int = 6, heads: int = 2) -> SignificanceCache:
    cache = SignificanceCache(num_hidden_layers=1)
    keys = (
        torch.arange(length, dtype=torch.float32)
        .view(1, 1, length, 1)
        .expand(1, heads, length, 1)
        .clone()
    )
    cache.prepare_append(torch.arange(length))
    cache.update(keys, keys + 1000, 0)
    cache.assert_forward_complete()
    layer = cache.layers[0]
    layer.record(torch.ones(1, heads, length))
    layer.state.counts = torch.stack(
        [
            torch.arange(1, length + 1, dtype=torch.float32),
            torch.arange(length, 0, -1, dtype=torch.float32),
        ][:heads]
    ).unsqueeze(0)
    layer.state.scores = torch.arange(length, dtype=torch.float32).view(1, 1, -1).expand(1, heads, -1).clone()
    cache.assert_aligned()
    return cache


def test_all_closed_loop_count_transforms_are_distinct_deterministic_and_aligned() -> None:
    expected_true = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]]
    )
    observed: dict[str, torch.Tensor] = {}
    for name in COUNT_CONTROL_NAMES:
        first = _controller(name).effective_counts(_cache(), boundary_index=0)[0]
        second = _controller(name).effective_counts(_cache(), boundary_index=0)[0]
        assert first.shape == expected_true.shape
        assert torch.equal(first, second)
        assert torch.isfinite(first).all() and (first >= 0).all()
        observed[name] = first

    torch.testing.assert_close(observed["true"], expected_true)
    torch.testing.assert_close(observed["constant"], torch.ones_like(expected_true))
    torch.testing.assert_close(
        observed["age_derived"],
        torch.tensor([[[3, 3, 2, 2, 1, 1], [3, 3, 2, 2, 1, 1]]], dtype=torch.float32),
    )
    for head in range(2):
        assert sorted(observed["shuffled"][0, head].tolist()) == sorted(
            expected_true[0, head].tolist()
        )
        assert not torch.equal(observed["shuffled"][0, head], expected_true[0, head])
    torch.testing.assert_close(
        observed["wrong_token"],
        torch.tensor([[[6, 1, 2, 3, 4, 5], [1, 6, 5, 4, 3, 2]]], dtype=torch.float32),
    )
    torch.testing.assert_close(observed["delayed"], torch.zeros_like(expected_true))
    assert not torch.equal(observed["noisy"], expected_true)
    torch.testing.assert_close(
        observed["head_averaged"], torch.full_like(expected_true, 3.5)
    )


@pytest.mark.parametrize(
    "transform",
    ["eligible_shuffled_v2", "eligible_circular_v2"],
)
def test_v2_identity_controls_preserve_only_eligible_multiset_and_protection(
    transform: str,
) -> None:
    cache = _cache(length=10)
    true_counts = cache.layers[0].state.counts.clone()
    controller = RuntimeCountController(
        CountControlSpec(
            condition_id=transform,
            transform=transform,  # type: ignore[arg-type]
            seed=23,
            circular_shift=1,
        ),
        case_id="clean-pool",
        chunk_size=2,
        protected_sinks=2,
        protected_recent=3,
    )
    effective = controller.effective_counts(cache, boundary_index=0)[0]
    protected = torch.tensor([0, 1, 7, 8, 9])
    eligible = torch.tensor([2, 3, 4, 5, 6])
    torch.testing.assert_close(effective[..., protected], true_counts[..., protected])
    torch.testing.assert_close(
        torch.sort(effective[..., eligible], dim=-1).values,
        torch.sort(true_counts[..., eligible], dim=-1).values,
    )
    # Counts are unique in this fixture, so a value change at every eligible
    # destination also proves semantic entry/count misalignment.
    assert torch.all(effective[..., eligible] != true_counts[..., eligible])


def test_v2_shuffle_is_seeded_deterministic_and_case_specific() -> None:
    def transformed(*, seed: int, case_id: str) -> torch.Tensor:
        return RuntimeCountController(
            CountControlSpec("shuffle-v2", "eligible_shuffled_v2", seed=seed),
            case_id=case_id,
            chunk_size=2,
            protected_sinks=2,
            protected_recent=3,
        ).effective_counts(_cache(length=20), boundary_index=4)[0]

    first = transformed(seed=23, case_id="same")
    torch.testing.assert_close(first, transformed(seed=23, case_id="same"))
    assert not torch.equal(first, transformed(seed=24, case_id="same"))
    assert not torch.equal(first, transformed(seed=23, case_id="different"))


def test_v2_boundary_trace_proves_source_identity_and_standard_true_path() -> None:
    policy = SignificancePolicy(
        budget=7,
        sinks=2,
        recent=3,
        count_exponent=1.0,
        require_heavy=True,
    )
    clean_cache = _cache(length=10)
    clean = RuntimeCountController(
        CountControlSpec("shuffle-v2", "eligible_shuffled_v2", seed=23),
        case_id="trace",
        chunk_size=2,
        protected_sinks=2,
        protected_recent=3,
    )
    clean.compress(clean_cache, policy, 0)
    arrays = clean.diagnostic_arrays()
    source = arrays["b000_l000_effective_count_source_entry_ids"]
    destination = arrays["b000_l000_entry_ids"]
    eligible_mask = arrays["b000_l000_eligible_heavy_mask"].astype(bool)
    assert eligible_mask[..., :2].sum() == 0
    assert eligible_mask[..., -3:].sum() == 0
    assert eligible_mask.sum() == 2 * 5
    assert (source[eligible_mask] != destination[eligible_mask]).all()
    assert (source[~eligible_mask] == destination[~eligible_mask]).all()

    true_cache = _cache(length=10)
    true_counts = true_cache.layers[0].state.counts.clone()
    standard = RuntimeCountController(
        CountControlSpec("true-v2", "true", seed=23),
        case_id="trace",
        chunk_size=2,
        protected_sinks=2,
        protected_recent=3,
    )
    effective = standard.effective_counts(true_cache, boundary_index=0)[0]
    torch.testing.assert_close(effective, true_counts)


def test_v2_controls_fail_closed_without_exact_policy_protection() -> None:
    with pytest.raises(ValueError, match="explicit protected_sinks"):
        RuntimeCountController(
            CountControlSpec("shuffle-v2", "eligible_shuffled_v2", seed=23),
            case_id="missing-protection",
            chunk_size=2,
        )
    controller = RuntimeCountController(
        CountControlSpec("shuffle-v2", "eligible_shuffled_v2", seed=23),
        case_id="mismatch",
        chunk_size=2,
        protected_sinks=2,
        protected_recent=3,
    )
    with pytest.raises(ValueError, match="exactly match"):
        controller.compress(
            _cache(length=10),
            SignificancePolicy(budget=7, sinks=1, recent=3, count_exponent=1.0),
            0,
        )


def test_delayed_counts_follow_semantic_entry_history_not_current_values() -> None:
    cache = _cache()
    controller = _controller("delayed")
    first = controller.effective_counts(cache, boundary_index=0)[0]
    old = cache.layers[0].state.counts.clone()
    cache.layers[0].state.counts.add_(10)
    second = controller.effective_counts(cache, boundary_index=1)[0]
    assert torch.count_nonzero(first) == 0
    torch.testing.assert_close(second, old)


def test_counterfactual_denominator_changes_selection_but_true_count_survives_gather() -> None:
    def ranked_cache() -> SignificanceCache:
        cache = SignificanceCache(num_hidden_layers=1)
        keys = torch.arange(5, dtype=torch.float32).view(1, 1, 5, 1)
        cache.prepare_append(torch.arange(5))
        cache.update(keys, keys + 1000, 0)
        cache.assert_forward_complete()
        state = cache.layers[0].state
        state.scores = torch.tensor([[[0.0, 10.0, 8.0, 1.0, 0.0]]])
        state.counts = torch.tensor([[[1.0, 10.0, 1.0, 1.0, 1.0]]])
        state.scoring_events = 1
        cache.assert_aligned()
        return cache

    policy = SignificancePolicy(
        budget=3,
        sinks=1,
        recent=1,
        count_exponent=1.0,
        require_heavy=True,
    )
    true_cache = ranked_cache()
    _controller("true").compress(true_cache, policy, 0)
    assert true_cache.layers[0].state.positions.tolist() == [[[0, 2, 4]]]

    constant_cache = ranked_cache()
    controller = _controller("constant")
    report = controller.compress(constant_cache, policy, 0)
    assert constant_cache.layers[0].state.positions.tolist() == [[[0, 1, 4]]]
    # The effective denominator was one, but persistent C=10 remains attached to entry 1.
    assert constant_cache.layers[0].state.counts.tolist() == [[[1.0, 10.0, 1.0]]]
    assert report.before_entries == 5 and report.after_entries == 3 and report.evictions == 2
    arrays = controller.diagnostic_arrays()
    selected = arrays["b000_l000_selected"]
    positions = arrays["b000_l000_positions"]
    assert positions[selected.astype(bool)].tolist() == [0, 1, 4]
    assert "b000_l000_true_counts" in arrays and "b000_l000_effective_counts" in arrays
    assert boundary_resident_positions(arrays, boundary_index=0, layer_count=1) == [
        [[0, 1, 4]]
    ]


def _load_runner() -> Any:
    lab = Path(__file__).resolve().parents[1]
    path = lab / "scripts" / "run_halo_count_controls.py"
    spec = importlib.util.spec_from_file_location("halo_count_controls_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_v2_analyzer() -> Any:
    lab = Path(__file__).resolve().parents[1]
    path = lab / "scripts" / "analyze_halo_count_controls_v2.py"
    spec = importlib.util.spec_from_file_location("halo_count_controls_v2_analyzer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v2_persisted_diagnostic_audit_checks_every_pool_invariant(tmp_path: Path) -> None:
    analyzer = _load_v2_analyzer()
    length = 40
    shape = (1, 2, length)
    true = np.broadcast_to(np.arange(length, dtype=np.float32), shape).copy()
    destination = np.broadcast_to(np.arange(length, dtype=np.int64), shape).copy()
    eligible = np.zeros(shape, dtype=np.uint8)
    eligible[..., 4:8] = 1
    rows = []
    for condition in analyzer.V2_CONDITIONS:
        source = destination.copy()
        effective = true.copy()
        if condition != analyzer.REFERENCE:
            source[..., 4:8] = np.roll(source[..., 4:8], 1, axis=-1)
            effective[..., 4:8] = np.roll(effective[..., 4:8], 1, axis=-1)
        path = tmp_path / f"{condition}.npz"
        np.savez_compressed(
            path,
            b000_l000_true_counts=true,
            b000_l000_effective_counts=effective,
            b000_l000_entry_ids=destination,
            b000_l000_effective_count_source_entry_ids=source,
            b000_l000_eligible_heavy_mask=eligible,
        )
        rows.append({"condition": condition, "boundary_diagnostics": path.name})
    result = analyzer.audit_v2_diagnostics(rows, artifact_root=tmp_path)
    assert result["status"] == "PASS"
    assert result["audited_layer_head_cells"] == 6
    assert result["intervention_semantic_fixed_points"] == 0
    assert result["standard_true_path_exact"] is True


def test_halo_count_control_config_and_dry_run_freeze_all_eight_matched_conditions() -> None:
    lab = Path(__file__).resolve().parents[1]
    runner = _load_runner()
    config_path = lab / "configs" / "halo_count_causal_controls.yaml"
    config = runner._load(config_path)
    specs = runner.validate_count_control_config(config)
    assert tuple(spec.transform for spec in specs) == COUNT_CONTROL_NAMES
    assert config["model"]["expected_source_base"] == "Qwen3-0.6B"
    assert ".halo." in config["model"]["core_id"]
    assert "R2" not in config["selection"]["families"]
    plan = runner.dry_run_plan(config_path)
    assert plan["status"] == "DRY_RUN_VALID"
    assert plan["model_loaded"] is False
    assert plan["selected_clusters"] >= 8
    assert plan["planned_model_executions"] == plan["selected_clusters"] * 8


def test_halo_v2_config_freezes_only_clean_eligible_pool_controls() -> None:
    lab = Path(__file__).resolve().parents[1]
    runner = _load_runner()
    config_path = lab / "configs" / "halo_count_causal_controls_v2.yaml"
    config = runner._load(config_path)
    specs = runner.validate_count_control_config(config)
    assert tuple(spec.transform for spec in specs) == CLEAN_COUNT_CONTROL_NAMES_V2
    assert (config["cache"]["sink_tokens"], config["cache"]["recent_tokens"]) == (4, 32)
    plan = runner.dry_run_plan(config_path)
    assert plan["status"] == "DRY_RUN_VALID"
    assert plan["count_control_contract"] == "eligible_heavy_v2"
    assert plan["protected_sink_tokens"] == 4
    assert plan["protected_recent_tokens"] == 32
    assert plan["planned_model_executions"] == plan["selected_clusters"] * 3


def test_matched_execution_contract_detects_query_or_schedule_drift() -> None:
    runner = _load_runner()
    fields = {
        "input_tokens": 384,
        "chunk_size": 128,
        "chunk_count": 3,
        "chunk_sizes": [128, 128, 128],
        "compression_boundaries": 2,
        "boundary_accounting": [{"boundary_index": 0}, {"boundary_index": 1}],
        "retained_history_budget": 128,
        "history_entries_after_last_boundary": 128,
        "final_active_chunk_entries": 128,
        "query_tokens_per_probe": 9,
        "query_probe_token_evaluations": 18,
        "max_transient_resident_entries": 265,
        "resident_entries_after_prefill_expected": 256,
        "resident_entries_after_prefill": 256,
        "scoring_events_per_layer": [2] * 28,
        "compression_reports": [{"before_entries_per_layer": 256, "after_entries_per_layer": 128}],
        "effective_evictions_per_layer": 256,
    }
    ids = ("true_realized_count", "constant_count")
    rows = [
        {
            "case_id": "case-1",
            "condition": condition,
            "prompt_sha256": "prompt",
            "query_token_sha256": "query",
            "accounting": dict(fields),
        }
        for condition in ids
    ]
    runner.assert_matched_execution_contract(rows, condition_ids=ids)
    rows[1]["query_token_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="query tokens changed"):
        runner.assert_matched_execution_contract(rows, condition_ids=ids)
    rows[1]["query_token_sha256"] = "query"
    rows[1]["accounting"] = {**fields, "chunk_size": 64}
    with pytest.raises(RuntimeError, match="chunk_size"):
        runner.assert_matched_execution_contract(rows, condition_ids=ids)


def test_prompt_only_view_rejects_answer_or_evidence_leakage() -> None:
    runner = _load_runner()
    row = {
        "case_id": "case-1",
        "cluster_id": "cluster-1",
        "family": "R1",
        "prompt": "prompt",
        "prompt_sha256": "hash",
        "question": "question",
        "token_count": 1,
        "token_ids_sha256": "tokens",
    }
    runner._validate_inference_rows([row])
    with pytest.raises(RuntimeError, match=r"forbidden=.*accepted_answers"):
        runner._validate_inference_rows([{**row, "accepted_answers": ["secret"]}])


def test_resume_checkpoint_recovers_only_the_common_frozen_prefix() -> None:
    runner = _load_runner()
    expected = [("case-1", "true"), ("case-1", "constant"), ("case-2", "true")]
    predictions = [
        {"case_id": case_id, "condition": condition, "prediction": "x"}
        for case_id, condition in expected[:2]
    ]
    boundaries = [
        {"case_id": case_id, "condition": condition, "boundaries": []}
        for case_id, condition in expected[:1]
    ]
    recovered_predictions, recovered_boundaries = runner.recover_aligned_checkpoint_prefix(
        expected,
        predictions,
        boundaries,
    )
    assert len(recovered_predictions) == len(recovered_boundaries) == 1
    with pytest.raises(RuntimeError, match="frozen-order prefix"):
        runner.recover_aligned_checkpoint_prefix(
            expected,
            [{"case_id": "case-2", "condition": "true"}],
            [],
        )


def _tiny_qwen3() -> Any:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rope_theta=1_000_000,
        attention_dropout=0.0,
        use_cache=True,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(11)
    return Qwen3ForCausalLM(config).eval()


@pytest.mark.model
def test_true_count_callback_is_closed_loop_identity_with_boundary_diagnostics() -> None:
    model = _tiny_qwen3()
    generator = torch.Generator().manual_seed(91)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 12), generator=generator)
    query_ids = torch.randint(0, model.config.vocab_size, (1, 2), generator=generator)
    policy = SignificancePolicy(
        budget=4,
        sinks=1,
        recent=1,
        count_exponent=1.0,
        require_heavy=True,
    )
    reference_cache = SignificanceCache(config=model.config)
    reference = chunked_prefill(
        model,
        input_ids,
        reference_cache,
        chunk_size=4,
        policy=policy,
        query_ids=query_ids,
        verify_query_rollback=True,
    )
    controlled_cache = SignificanceCache(config=model.config)
    controller = RuntimeCountController(
        CountControlSpec("true-realized", "true", seed=23),
        case_id="tiny",
        chunk_size=4,
    )
    controlled = chunked_prefill(
        model,
        input_ids,
        controlled_cache,
        chunk_size=4,
        policy=policy,
        query_ids=query_ids,
        verify_query_rollback=True,
        compression_callback=controller.compress,
    )
    assert reference.compression_reports == controlled.compression_reports
    assert len(controller.summaries) == 2
    assert controlled_cache.scoring_events() == reference_cache.scoring_events() == [2, 2]
    assert controlled_cache.stored_length() == reference_cache.stored_length() == 8
    for left, right in zip(reference_cache.layers, controlled_cache.layers, strict=True):
        torch.testing.assert_close(left.state.scores, right.state.scores)
        torch.testing.assert_close(left.state.counts, right.state.counts)
        assert torch.equal(left.state.positions, right.state.positions)
        assert torch.equal(left.state.entry_ids, right.state.entry_ids)
    arrays = controller.diagnostic_arrays()
    assert "b000_l000_selected" in arrays
    assert "b001_l001_retained_positions" in arrays

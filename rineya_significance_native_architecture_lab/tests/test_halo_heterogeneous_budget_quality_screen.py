from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

LAB_ROOT = Path(__file__).resolve().parents[1]


def _runner():
    path = LAB_ROOT / "scripts" / "run_halo_heterogeneous_budget_quality_screen.py"
    spec = importlib.util.spec_from_file_location("heterogeneous_quality_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_quality_screen_config_reuses_exact_frozen_ids_but_is_not_registered() -> None:
    config = yaml.safe_load(
        (LAB_ROOT / "configs" / "halo_heterogeneous_budget_quality_screen.yaml").read_text()
    )
    registered = yaml.safe_load(
        (LAB_ROOT / "configs" / "halo_endpoint_validation_v2_registered.yaml").read_text()
    )
    assert config["experiment"]["classification"] == "EXPLORATORY_NOT_REGISTERED"
    assert config["frozen_case_source"]["new_conditions_registered"] is False
    assert config["selection"]["selected_case_ids"] == registered["selection"]["selected_case_ids"]
    assert [row["id"] for row in config["conditions"]] == [
        "B3_UNIFORM",
        "B3_ENTROPY",
        "B3_SHUFFLED",
    ]
    assert config["protocol"]["total_cells"] == 36
    assert config["cache"]["fixed_global_entry_budget"] == 28 * 8 * 128
    assert config["protocol"]["memory_saving_claim_allowed"] is False


def test_variable_resident_position_round_trip_has_no_padding(tmp_path: Path) -> None:
    runner = _runner()
    resident = [
        [[0, 4], [0, 2, 4]],
        [[1], [1, 3, 5, 7]],
    ]
    path = tmp_path / "resident.npz"
    digest = runner.save_resident_positions(path, resident)
    assert len(digest) == 64
    assert runner.load_resident_positions(path, layer_count=2, head_count=2) == resident
    with np.load(path, allow_pickle=False) as arrays:
        assert set(arrays.files) == {
            "layer_000_head_000",
            "layer_000_head_001",
            "layer_001_head_000",
            "layer_001_head_001",
        }
        assert all((arrays[name] >= 0).all() for name in arrays.files)
    with pytest.raises(ValueError, match="padding escaped"):
        runner.validate_resident_positions([[[-1], [0]]])


def test_query_access_matching_audit_detects_any_condition_drift() -> None:
    runner = _runner()
    rows = []
    access = {
        "mode": "conditioned",
        "compression_boundaries": 3,
        "query_tokens_per_probe": 7,
        "query_probe_token_evaluations": 21,
    }
    for case_id in ("a", "b"):
        for condition in runner.CONDITION_IDS:
            rows.append({"case_id": case_id, "condition": condition, "query_access": dict(access)})
    assert runner.validate_query_access(rows)["status"] == "PASS"
    rows[-1]["query_access"]["query_probe_token_evaluations"] = 20
    result = runner.validate_query_access(rows)
    assert result["status"] == "FAIL"
    assert result["failures"] == ["b:query_access"]


def test_allocation_summary_requires_exact_budget_and_reports_layer_heterogeneity() -> None:
    runner = _runner()
    summary = runner._allocation_summary(
        np.asarray([[2, 3], [3, 4]]),
        boundary_index=2,
        global_budget=12,
    )
    assert summary["heterogeneous"] is True
    assert summary["layer_total_minimum"] == 5
    assert summary["layer_total_maximum"] == 7
    with pytest.raises(RuntimeError, match="fixed global budget"):
        runner._allocation_summary(
            np.asarray([[2, 3], [3, 4]]),
            boundary_index=0,
            global_budget=11,
        )


def test_paired_statistics_emits_all_exploratory_endpoint_and_answer_contrasts() -> None:
    runner = _runner()
    rows = []
    for index in range(12):
        case_id = f"case-{index:02d}"
        for condition, endpoint, correct in (
            ("B3_UNIFORM", 0.2, False),
            ("B3_ENTROPY", 0.3 if index < 8 else 0.2, index == 0),
            ("B3_SHUFFLED", 0.25 if index < 4 else 0.2, False),
        ):
            rows.append(
                {
                    "case_id": case_id,
                    "scenario_id": f"cluster-{index:02d}",
                    "condition": condition,
                    "evidence": {"layer_fact_exact_completion_rate": endpoint},
                    "correct": correct,
                }
            )
    config = {
        "analysis": {
            "minimum_independent_clusters": 12,
            "confidence": 0.95,
            "bootstrap_samples": 200,
            "randomization_samples": 1000,
        },
        "runtime": {"seed": 9},
    }
    result = runner.paired_statistics(rows, config=config)
    assert result["classification"] == "EXPLORATORY_NOT_REGISTERED"
    assert result["cluster_count"] == 12
    assert set(result["comparisons"]) == {
        "B3_ENTROPY_minus_B3_UNIFORM__endpoint",
        "B3_ENTROPY_minus_B3_UNIFORM__answer_correct",
        "B3_SHUFFLED_minus_B3_UNIFORM__endpoint",
        "B3_SHUFFLED_minus_B3_UNIFORM__answer_correct",
        "B3_ENTROPY_minus_B3_SHUFFLED__endpoint",
        "B3_ENTROPY_minus_B3_SHUFFLED__answer_correct",
    }


def _tiny_qwen3():
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
    torch.manual_seed(59)
    return Qwen3ForCausalLM(config).eval()


@pytest.mark.model
def test_same_process_uniform_identity_gate_is_bitwise_on_tiny_qwen() -> None:
    runner = _runner()
    model = _tiny_qwen3()
    generator = torch.Generator().manual_seed(61)
    prepared = SimpleNamespace(
        case_id="tiny-gate",
        input_ids=torch.randint(0, 128, (1, 9), generator=generator),
        query_ids=torch.randint(0, 128, (1, 2), generator=generator),
    )
    config = {
        "cache": {
            "retained_history_budget_per_layer_head": 3,
            "compression_chunk_tokens": 3,
            "fixed_global_entry_budget": 12,
            "sink_tokens": 1,
            "recent_tokens": 1,
            "count_exponent": 1.0,
        },
        "runtime": {"verify_query_rollback": True},
    }
    result = runner.uniform_identity_gate(model, prepared, config=config)
    assert result == {
        "status": "PASS",
        "case_id": "tiny-gate",
        "performed_before_outcome_access": True,
        "exact_logits": True,
        "max_abs_logit_delta": 0.0,
        "exact_cache_state": True,
        "cache_mismatches": [],
        "uniform_allocation_reports_exact": True,
        "compression_boundaries": 2,
        "global_budget": 12,
    }


def test_canonical_screen_artifacts_if_present_are_self_consistent() -> None:
    run = LAB_ROOT / "results" / "runs" / "halo_heterogeneous_budget_quality_screen_v1"
    if not (run / "summary.json").exists():
        pytest.skip("bounded quality screen has not run yet")
    summary = json.loads((run / "summary.json").read_text())
    rows = [json.loads(line) for line in (run / "cases.jsonl").read_text().splitlines()]
    assert summary["classification"] == "EXPLORATORY_NOT_REGISTERED"
    assert summary["cells"] == 36
    assert len(rows) == 36
    assert {row["condition"] for row in rows} == {
        "B3_UNIFORM",
        "B3_ENTROPY",
        "B3_SHUFFLED",
    }
    assert all(
        row["efficiency"]["physical_total_bytes"]
        >= row["efficiency"]["logical_payload_bytes"]
        for row in rows
    )

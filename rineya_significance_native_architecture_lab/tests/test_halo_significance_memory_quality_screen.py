from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from rineya_significance_lab.runtime import SignificanceMemory

LAB_ROOT = Path(__file__).resolve().parents[1]


def _runner():
    path = LAB_ROOT / "scripts" / "run_halo_significance_memory_quality_screen.py"
    spec = importlib.util.spec_from_file_location("memory_quality_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config() -> dict:
    return yaml.safe_load(
        (LAB_ROOT / "configs" / "halo_significance_memory_quality_screen.yaml").read_text()
    )


def test_config_reuses_frozen_ids_but_registers_no_memory_condition() -> None:
    config = _config()
    registered = yaml.safe_load(
        (LAB_ROOT / "configs" / "halo_endpoint_validation_v2_registered.yaml").read_text()
    )
    runner = _runner()
    assert config["experiment"]["classification"] == "EXPLORATORY_NOT_REGISTERED"
    assert config["frozen_case_source"]["new_conditions_registered"] is False
    assert config["selection"]["selected_case_ids"] == registered["selection"]["selected_case_ids"]
    assert [row["id"] for row in config["conditions"]] == list(runner.CONDITION_IDS)
    assert config["protocol"]["total_cells"] == 72
    assert config["protocol"]["successive_halving_used"] is False
    assert config["protocol"]["latency_claim_allowed"] is False


def test_halo_total_byte_contract_is_recomputed_not_trusted() -> None:
    runner = _runner()
    contract = runner.solve_total_budget_contract(
        _config(),
        num_layers=28,
        batch_size=1,
        num_kv_heads=8,
        head_dim=128,
        kv_dtype=torch.bfloat16,
    )
    assert contract["baseline_ordinary_budget"] == 128
    assert contract["candidate_ordinary_budget"] == 123
    assert contract["ordinary_budget_reduction"] == 5
    assert contract["ordinary_bytes_per_budget_unit"] == 120_064
    assert contract["memory_bytes"] == 491_904
    assert contract["candidate_total_bytes"] == 15_259_776
    assert contract["baseline_total_bytes"] == 15_368_192
    assert contract["discrete_slack_bytes"] == 108_416


def test_slot_artifact_round_trip_preserves_exact_fixed_shapes(tmp_path: Path) -> None:
    runner = _runner()
    for slots in (0, 4):
        memory = SignificanceMemory(
            num_layers=2,
            batch_size=1,
            num_kv_heads=2,
            num_slots=slots,
            head_dim=4,
            variant="mean",
            device="cpu",
            kv_dtype=torch.bfloat16,
        )
        path = tmp_path / f"m{slots}.npz"
        digest = runner.save_memory_slots(path, memory)
        assert len(digest) == 64
        runner.validate_memory_slot_artifact(
            path,
            layers=2,
            heads=2,
            slots=slots,
            head_dim=4,
        )


def test_matching_requires_same_queries_fixed_decode_and_total_bytes() -> None:
    runner = _runner()
    rows = []
    access = {
        "compression_boundaries": 3,
        "query_tokens_per_probe": 15,
        "query_probe_token_evaluations": 45,
    }
    for case_id in ("a", "b"):
        for index, condition in enumerate(runner.CONDITION_IDS):
            rows.append(
                {
                    "case_id": case_id,
                    "condition": condition,
                    "query_access": dict(access),
                    "decode_cache_appends": 31,
                    "budget": {
                        "matched_total_persistent_bytes": True,
                        "condition_realized_total_persistent_bytes": 100 - index,
                    },
                }
            )
    assert runner.validate_matching(rows)["status"] == "PASS"
    rows[-1]["query_access"]["query_probe_token_evaluations"] = 44
    assert runner.validate_matching(rows)["status"] == "FAIL"
    rows[-1]["query_access"] = dict(access)
    rows[-1]["budget"]["condition_realized_total_persistent_bytes"] = 101
    result = runner.validate_matching(rows)
    assert result["status"] == "FAIL"
    assert any("total_bytes" in value for value in result["failures"])


def test_all_15_pairs_use_whole_cluster_statistics() -> None:
    runner = _runner()
    rows = []
    config = _config()
    config["analysis"] = {
        **config["analysis"],
        "bootstrap_samples": 100,
        "randomization_samples": 1000,
    }
    for index, case_id in enumerate(config["selection"]["selected_case_ids"]):
        for condition_index, condition in enumerate(runner.CONDITION_IDS):
            rows.append(
                {
                    "case_id": case_id,
                    "scenario_id": f"cluster-{index:02d}",
                    "condition": condition,
                    "correct": condition_index == index % len(runner.CONDITION_IDS),
                    "evidence": {
                        "layer_fact_exact_completion_rate": 0.1 + 0.01 * condition_index
                    },
                }
            )
    result = runner.paired_statistics(rows, config=config)
    assert result["classification"] == "EXPLORATORY_LOCAL_NOT_REGISTERED"
    assert result["independence_unit"] == "whole_cluster_id"
    assert result["cluster_count"] == 12
    assert result["condition_pair_count"] == 15
    assert len(result["comparisons"]) == 30


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
    torch.manual_seed(79)
    return Qwen3ForCausalLM(config).eval()


@pytest.mark.model
def test_same_process_m0_gate_includes_fixed_decode_identity() -> None:
    runner = _runner()
    model = _tiny_qwen3()
    generator = torch.Generator().manual_seed(83)
    prepared = SimpleNamespace(
        case_id="tiny-memory-gate",
        input_ids=torch.randint(0, 128, (1, 9), generator=generator),
        query_ids=torch.randint(0, 128, (1, 2), generator=generator),
    )
    config = _config()
    config["conditions"][0]["ordinary_budget"] = 3
    config["cache"].update(
        {
            "compression_chunk_tokens": 3,
            "sink_tokens": 1,
            "recent_tokens": 1,
            "selection_count_exponent": 1.0,
        }
    )
    config["runtime"].update({"max_new_tokens": 4, "verify_query_rollback": True})
    result = runner.m0_identity_gate(
        model,
        prepared,
        config=config,
        eos_token_ids=set(),
    )
    assert result["status"] == "PASS"
    assert result["exact_final_logits"] is True
    assert result["max_abs_final_logit_delta"] == 0.0
    assert result["exact_prediction_tokens"] is True
    assert result["exact_fixed_decode_trajectory"] is True
    assert result["exact_cache_state"] is True
    assert result["m0_memory_bytes"] == 0


def test_source_opens_evaluation_only_after_durable_predictions() -> None:
    source = (
        LAB_ROOT / "scripts" / "run_halo_significance_memory_quality_screen.py"
    ).read_text(encoding="utf-8")
    durable = source.index('predictions_path.write_text(')
    evaluation_hash = source.index('sha256_file(resolved["evaluation"])')
    evaluation_load = source.index('_jsonl(resolved["evaluation"])')
    assert durable < evaluation_hash < evaluation_load


def test_canonical_artifacts_if_present_are_complete_and_narrow() -> None:
    run = LAB_ROOT / "results" / "runs" / "halo_significance_memory_quality_screen_v1"
    if not (run / "summary.json").exists():
        pytest.skip("memory quality screen has not completed")
    summary = json.loads((run / "summary.json").read_text())
    cases = [json.loads(line) for line in (run / "cases.jsonl").read_text().splitlines()]
    predictions = [
        json.loads(line)
        for line in (run / "predictions.unscored.jsonl").read_text().splitlines()
    ]
    assert summary["classification"] == "EXPLORATORY_LOCAL_NOT_REGISTERED"
    assert summary["cells"] == 72
    assert len(cases) == len(predictions) == 72
    assert {row["condition"] for row in cases} == set(_runner().CONDITION_IDS)
    assert all(row["fixed_decode_steps"] == 32 for row in cases)
    assert all(
        row["budget"]["condition_realized_total_persistent_bytes"]
        <= row["budget"]["m0_fixed_decode_capacity_bytes"]
        for row in cases
    )
    assert all(not ({"correct", "accepted_answers", "evidence"} & set(row)) for row in predictions)
    assert len(list((run / "cell_checkpoints").glob("*.json"))) == 72
    assert len(list((run / "resident_positions").glob("*.npz"))) == 72
    assert len(list((run / "memory_slots").glob("*.npz"))) == 72
    expected = {
        "M0_HARD": (5, 0.21130952380952386),
        "M4_MEAN": (5, 0.18452380952380953),
        "M4_RECENCY": (5, 0.18601190476190477),
        "M4_RAW_S": (5, 0.18601190476190474),
        "M4_NORMALIZED_S_OVER_C": (5, 0.18303571428571427),
        "M4_RANDOM": (5, 0.19196428571428578),
    }
    for condition, (correct, endpoint) in expected.items():
        observed = summary["by_condition"][condition]
        assert observed["correct"] == correct
        assert observed["mean_layer_fact_exact_completion_rate"] == pytest.approx(endpoint)

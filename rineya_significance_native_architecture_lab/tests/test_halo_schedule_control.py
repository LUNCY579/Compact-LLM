from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from rineya_significance_lab.evaluation.schedule_interaction import (
    clustered_method_round_interaction,
    validate_matched_schedule_rows,
)
from rineya_significance_lab.runtime import SignificanceCache, SignificancePolicy
from rineya_significance_lab.runtime.scheduled_prefill import (
    build_fixed_terminal_schedule,
    fixed_terminal_prefill,
)


def _runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_halo_schedule_control.py"
    spec = importlib.util.spec_from_file_location("halo_schedule_control_test_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    torch.manual_seed(11)
    return Qwen3ForCausalLM(config).eval()


def test_fixed_terminal_schedule_holds_final_cache_size() -> None:
    schedules = [
        build_fixed_terminal_schedule(
            schedule_id=f"c{chunk}",
            input_tokens=425,
            prefix_chunk_tokens=chunk,
            terminal_active_tokens=64,
            retained_history_budget=128,
        )
        for chunk in (64, 96, 128, 192, 512)
    ]
    assert [schedule.compression_rounds for schedule in schedules] == [6, 4, 3, 2, 1]
    assert {schedule.final_resident_entries for schedule in schedules} == {192}
    assert all(schedule.chunk_sizes[-1] == 64 for schedule in schedules)
    assert all(schedule.boundary_positions[-1] == 361 for schedule in schedules)


@pytest.mark.model
def test_fixed_terminal_prefill_realizes_exact_rounds_and_budget() -> None:
    model = _tiny_qwen3()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 13))
    query_ids = torch.randint(0, model.config.vocab_size, (1, 2))
    schedule = build_fixed_terminal_schedule(
        schedule_id="c4",
        input_tokens=13,
        prefix_chunk_tokens=4,
        terminal_active_tokens=3,
        retained_history_budget=5,
    )
    cache = SignificanceCache(config=model.config)
    result = fixed_terminal_prefill(
        model,
        input_ids,
        cache,
        schedule=schedule,
        policy=SignificancePolicy(budget=5, sinks=1, recent=1, count_exponent=1.0),
        query_ids=query_ids,
    )

    assert schedule.chunk_sizes == (4, 4, 2, 3)
    assert schedule.boundary_positions == (4, 8, 10)
    assert len(result.compression_reports) == 3
    assert cache.scoring_events() == [3, 3]
    assert cache.true_length == 13
    assert cache.stored_length() == 8
    assert result.query_probe_token_evaluations == 6
    cache.assert_aligned()


def _interaction_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cluster in ("C1", "C2", "C3"):
        held = {
            "prompt_sha256": f"prompt-{cluster}",
            "input_token_sha256": f"tokens-{cluster}",
            "query_token_sha256": f"query-{cluster}",
            "model_identity_sha256": "model",
            "answer_key_sha256": f"answer-{cluster}",
            "retained_history_budget": 8,
            "terminal_active_tokens": 2,
            "decoding": "manual_greedy",
            "max_new_tokens": 2,
            "seed": 17,
        }
        for rounds in (1, 3):
            schedule = {
                "schedule_id": f"r{rounds}",
                "compression_rounds": rounds,
                "prefix_chunk_tokens": 12 // rounds,
                "terminal_active_tokens": 2,
                "retained_history_budget": 8,
                "final_resident_entries": 10,
                "boundary_positions": list(range(1, rounds + 1)),
            }
            access = {
                "query_tokens_per_probe": 2,
                "query_probes": rounds,
                "query_probe_token_evaluations": 2 * rounds,
                "scoring_events_per_layer": [rounds, rounds],
            }
            for method, exponent in (("B2", 0.0), ("B3", 1.0)):
                value = 0.2 if method == "B2" else 0.2 + 0.1 * rounds
                rows.append(
                    {
                        "case_id": cluster,
                        "cluster_id": cluster,
                        "schedule_id": f"r{rounds}",
                        "method": method,
                        "count_exponent": exponent,
                        "schedule": schedule,
                        "held_constant": held,
                        "query_access": access,
                        "accounting": {
                            "resident_entries_after_prefill": 10,
                            "boundary_accounting": [
                                {"absolute_token_position": value}
                                for value in range(1, rounds + 1)
                            ],
                            "compression_reports": [{} for _ in range(rounds)],
                        },
                        "outcomes": {"value": value},
                    }
                )
    return rows


def test_clustered_interaction_uses_cluster_slopes() -> None:
    result = clustered_method_round_interaction(
        _interaction_rows(),
        outcome_path="outcomes.value",
        minimum_clusters=3,
        bootstrap_samples=200,
        randomization_samples=200,
        seed=7,
    )
    slope = result["linear_interaction_per_additional_compression_round"]
    endpoint = result["highest_common_round_minus_lowest_common_round_method_contrast"]
    assert slope["cluster_count"] == 3
    assert slope["mean"] == pytest.approx(0.1)
    assert endpoint["mean"] == pytest.approx(0.2)
    assert result["row_level_schedule_repeats_treated_as_independent"] is False


def test_matching_audit_rejects_unequal_query_access() -> None:
    rows = _interaction_rows()
    rows[1]["query_access"] = {**rows[1]["query_access"], "query_probes": 99}
    with pytest.raises(ValueError, match="query_access differs"):
        validate_matched_schedule_rows(rows)


def test_config_builds_complete_matched_halo_plan() -> None:
    lab = Path(__file__).resolve().parents[1]
    runner = _runner()
    config = yaml.safe_load((lab / "configs" / "halo_schedule_control.yaml").read_text())
    inference = [
        json.loads(line)
        for line in (lab / "results" / "benchmarks" / "halo_calibration_v3" / "inference.jsonl")
        .read_text()
        .splitlines()
        if line
    ]
    capability = [
        json.loads(line)
        for line in (lab / "results" / "runs" / "halo_capability_calibration_v2" / "cases.jsonl")
        .read_text()
        .splitlines()
        if line
    ]
    plans = runner.build_cell_plans(
        config,
        inference,
        capability,
        model_identity_sha256="0" * 64,
    )
    assert len(plans) == 110  # 11 B0-eligible clusters x 5 schedules x 2 methods
    assert {plan["method"] for plan in plans} == {"B2", "B3"}
    assert {plan["schedule"]["final_resident_entries"] for plan in plans} == {192}
    assert all(plan["held_constant"]["seed"] == 17 for plan in plans)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for plan in plans:
        grouped.setdefault((plan["case_id"], plan["schedule_id"]), []).append(plan)
    assert all([row["method"] for row in pair] == ["B2", "B3"] for pair in grouped.values())


def test_cell_checkpoints_resume_without_reexecution(tmp_path: Path) -> None:
    runner = _runner()
    plans = [
        {"cell_id": "case__c1__B2", "plan_fingerprint": "a"},
        {"cell_id": "case__c1__B3", "plan_fingerprint": "b"},
    ]
    calls: list[str] = []

    def execute(plan: dict[str, Any]) -> dict[str, Any]:
        calls.append(plan["cell_id"])
        return {"cell_id": plan["cell_id"], "value": len(calls)}

    first, first_new = runner.run_checkpointed_cells(
        plans, tmp_path / "checkpoints", execute, max_new_cells=1
    )
    assert len(first) == 1 and first_new == 1
    second, second_new = runner.run_checkpointed_cells(plans, tmp_path / "checkpoints", execute)
    assert len(second) == 2 and second_new == 1
    third, third_new = runner.run_checkpointed_cells(plans, tmp_path / "checkpoints", execute)
    assert len(third) == 2 and third_new == 0
    assert calls == ["case__c1__B2", "case__c1__B3"]

    corrupted_path = tmp_path / "checkpoints" / "case__c1__B2.json"
    corrupted = json.loads(corrupted_path.read_text())
    corrupted["result"]["value"] = 999
    corrupted_path.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(RuntimeError, match="result hash mismatch"):
        runner.run_checkpointed_cells(plans, tmp_path / "checkpoints", execute)

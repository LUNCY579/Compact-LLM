from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

LAB = Path(__file__).resolve().parents[1]
SCRIPT = LAB / "scripts" / "run_published_core_eval.py"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("rineya_published_core_eval", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _prediction_rows(predictions: dict[tuple[str, str], str] | None = None) -> list[dict[str, object]]:
    predictions = predictions or {}
    return [
        {
            "case_id": case_id,
            "condition": condition,
            "prediction": predictions.get((case_id, condition), "wrong"),
            "input_token_sha256": "a" * 64,
            "timing": {"inference_wall_seconds": 0.25},
        }
        for case_id in runner.CASE_IDS
        for condition in runner.CONDITION_IDS
    ]


def _answer_rows() -> list[dict[str, object]]:
    return [
        {
            "id": case_id,
            "scenario_id": case_id.removesuffix("-B"),
            "answer": f"answer-{index}",
            "accepted_answers": [f"answer-{index}"],
        }
        for index, case_id in enumerate(runner.CASE_IDS, 1)
    ]


def test_committed_config_freezes_halo_cases_conditions_and_cpu_bf16() -> None:
    config = runner._load_mapping(LAB / "configs" / "halo_mini_eval.yaml")
    conditions = runner.validate_config(config)

    assert tuple(config["benchmark"]["case_ids"]) == runner.CASE_IDS
    assert [(value.condition_id, value.exponent) for value in conditions] == [
        ("B0", None),
        ("B2", 0.0),
        ("B3", 1.0),
    ]
    assert all(value.query_conditioned for value in conditions)
    assert config["runtime"]["device"] == "cpu"
    assert config["runtime"]["dtype"] == "bfloat16"
    assert config["runtime"]["decoding"] == "manual_greedy"
    assert set(config["model"]["checkpoint_files"]) == set(config["model"]["expected_hashes"])


@pytest.mark.parametrize(
    ("enabled", "width"),
    [(True, 1), (False, 3)],
)
def test_config_rejects_pooling_or_nonunit_pool_width(enabled: bool, width: int) -> None:
    config = runner._load_mapping(LAB / "configs" / "halo_mini_eval.yaml")
    config["significance"]["score_pooling"] = {"enabled": enabled, "width": width}
    with pytest.raises(ValueError, match="score pooling must remain disabled"):
        runner.validate_config(config)


def test_checkpoint_hash_mismatch_fails_before_loading_model(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"not Halo weights")
    model_config = {
        "checkpoint_files": {"weights": checkpoint.name},
        "expected_hashes": {"weights": "0" * 64},
    }
    with pytest.raises(RuntimeError, match="checkpoint hash mismatch for weights"):
        runner._checkpoint_hashes(tmp_path, model_config)


@pytest.mark.parametrize("codename", ["Aegis", "R2", "Halo Uncensored"])
def test_halo_core_gate_rejects_other_family_models(codename: str) -> None:
    config = runner._load_mapping(LAB / "configs" / "halo_mini_eval.yaml")
    core = {
        "core_id": config["model"]["core_id"],
        "codename": codename,
        "directory_name": "Rineya-Halo-1.2.0-devfreeze",
        "source_base": "Qwen3-0.6B",
    }
    with pytest.raises(RuntimeError, match="refusing non-Halo"):
        runner._validate_halo_core(core, config["model"])


def test_prompt_loader_selects_exact_ids_and_rejects_oracle_fields(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    rows = [
        {
            "id": case_id,
            "context": f"context {case_id}",
            "question": "question?",
            "messages": [{"role": "user", "content": "prompt"}],
        }
        for case_id in reversed(runner.CASE_IDS)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    selected = runner._read_prompt_rows(path)
    assert [row["id"] for row in selected] == list(runner.CASE_IDS)

    rows[0]["answer"] = "leaked"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="oracle fields"):
        runner._read_prompt_rows(path)


def test_answer_key_is_opened_only_after_unscored_predictions_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer_path = tmp_path / "answers.jsonl"
    answer_path.write_text(
        "\n".join(json.dumps(row) for row in _answer_rows()) + "\n", encoding="utf-8"
    )
    unscored_path = tmp_path / "predictions.unscored.jsonl"
    events: list[str] = []
    original = runner._read_answer_key

    def inference() -> list[dict[str, object]]:
        events.append("inference_complete")
        return _prediction_rows()

    def guarded_key_load(path: Path) -> dict[str, dict[str, object]]:
        assert events == ["inference_complete"]
        assert unscored_path.is_file()
        persisted = [json.loads(line) for line in unscored_path.read_text(encoding="utf-8").splitlines()]
        assert len(persisted) == 9
        assert not any("correct" in row or "reference_answer" in row for row in persisted)
        events.append("answer_key_opened")
        return original(path)

    monkeypatch.setattr(runner, "_read_answer_key", guarded_key_load)
    predictions, answers = runner.blind_inference_then_load_key(
        inference,
        unscored_path=unscored_path,
        answer_key_path=answer_path,
    )
    assert len(predictions) == 9
    assert set(answers) == set(runner.CASE_IDS)
    assert events == ["inference_complete", "answer_key_opened"]


def test_full_cache_conditioned_correctness_is_joined_per_case() -> None:
    answers = {str(row["id"]): row for row in _answer_rows()}
    predictions = _prediction_rows(
        {
            ("RLC-001-B", "B0"): "Answer-1.",
            ("RLC-001-B", "B2"): "answer-1",
            ("RLC-001-B", "B3"): "wrong",
            ("RLC-010-B", "B0"): "wrong",
            ("RLC-010-B", "B2"): "answer-2",
            ("RLC-019-B", "B0"): "answer-3",
            ("RLC-019-B", "B2"): "answer-3",
            ("RLC-019-B", "B3"): "answer-3",
        }
    )
    scored = runner.score_predictions(
        predictions,
        answers,
        run_id="tiny-test",
        seed=17,
        chunk_size=4,
    )
    by_pair = {(row["case_id"], row["condition"]): row for row in scored}

    assert by_pair[("RLC-001-B", "B2")]["compression_conditioned_eligible"] is True
    assert by_pair[("RLC-001-B", "B2")]["full_cache_conditioned_correct"] is True
    assert by_pair[("RLC-001-B", "B3")]["full_cache_conditioned_correct"] is False
    assert by_pair[("RLC-010-B", "B2")]["compression_conditioned_eligible"] is False
    assert by_pair[("RLC-010-B", "B2")]["full_cache_conditioned_correct"] is None
    summary = runner._descriptive_summary(scored)
    assert summary["inferential_statistics"] is None
    assert summary["descriptive_results"]["B2"]["full_cache_conditioned_correctness"] == {
        "eligible_cases": 2,
        "correct_cases": 2,
        "accuracy": 1.0,
    }


def _tiny_qwen3():
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rope_theta=1_000_000,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=True,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(5)
    return transformers.Qwen3ForCausalLM(config).to(dtype=torch.bfloat16).eval()


def test_tiny_qwen3_manual_decode_and_matched_query_chunk_accounting() -> None:
    model = _tiny_qwen3()
    input_ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]])
    query_ids = torch.tensor([[14, 15]])
    prepared = runner.PreparedCase(
        case_id="RLC-001-B",
        scenario_id="RLC-001",
        metadata={},
        input_ids=input_ids,
        context_ids=input_ids,
        query_ids=query_ids,
        prompt_text_sha256="1" * 64,
        input_token_sha256=runner._token_sha256(input_ids),
        context_token_sha256=runner._token_sha256(input_ids),
        query_token_sha256=runner._token_sha256(query_ids),
    )
    cache_config = {
        "retained_history_budget": 4,
        "sink_tokens": 1,
        "recent_tokens": 1,
        "compression_chunk_tokens": 4,
    }
    runtime_config = {
        "max_new_tokens": 2,
        "verify_query_rollback": True,
    }
    conditions = (
        runner.Condition("B0", "full_cache", None, True),
        runner.Condition("B2", "significance", 0.0, True),
        runner.Condition("B3", "significance", 1.0, True),
    )

    results = {
        condition.condition_id: runner.execute_condition(
            model,
            prepared,
            condition,
            cache_config=cache_config,
            runtime_config=runtime_config,
            eos_token_ids=set(),
        )
        for condition in conditions
    }

    plain_cache = runner.SignificanceCache(config=model.config)
    plain_prefill = runner.chunked_prefill(
        model,
        input_ids,
        plain_cache,
        chunk_size=4,
        policy=None,
        logits_to_keep=1,
    )
    plain_decode = runner._manual_greedy_decode(
        model,
        plain_cache,
        plain_prefill.output.logits,
        max_new_tokens=2,
        eos_token_ids=set(),
    )

    for result in results.values():
        assert result["manual_greedy"] is True
        assert result["generated_tokens"] == 2
        assert result["accounting"]["chunk_sizes"] == [4, 4, 3]
        assert result["accounting"]["query_tokens_per_probe"] == 2
        assert result["accounting"]["query_probe_token_evaluations"] == 4
        assert result["accounting"]["scoring_events_per_layer"] == [2, 2]
        assert result["accounting"]["decode_cache_appends"] == 1
    assert results["B0"]["accounting"]["resident_entries_after_prefill"] == 11
    assert results["B0"]["accounting"]["effective_evictions_per_layer"] == 0
    assert results["B0"]["generated_token_ids"] == plain_decode["token_ids"]
    assert results["B2"]["accounting"]["resident_entries_after_prefill"] == 7
    assert results["B3"]["accounting"]["resident_entries_after_prefill"] == 7
    assert results["B2"]["accounting"]["effective_evictions_per_layer"] > 0
    assert results["B3"]["accounting"]["effective_evictions_per_layer"] > 0
    assert results["B2"]["significance_exponent"] == 0.0
    assert results["B3"]["significance_exponent"] == 1.0


def test_manifest_builder_output_validates_against_committed_schema(tmp_path: Path) -> None:
    config = runner._load_mapping(LAB / "configs" / "halo_mini_eval.yaml")
    config["_source_config_path"] = str(LAB / "configs" / "halo_mini_eval.yaml")
    resolved = tmp_path / "resolved_config.yaml"
    resolved.write_text("schema_version: '1.0'\n", encoding="utf-8")
    rows: list[dict[str, object]] = []
    for case_id in runner.CASE_IDS:
        for condition in runner.CONDITION_IDS:
            rows.append(
                {
                    "case_id": case_id,
                    "condition": condition,
                    "accounting": {
                        "resident_entries_after_prefill": 4,
                        "resident_entries_after_decode": 5,
                    },
                    "efficiency": {
                        "kv_only_bytes": 128,
                        "architecture_total_persistent_bytes": 160,
                    },
                }
            )
    digest = "b" * 64
    files = {
        name: {"path": filename, "sha256": digest}
        for name, filename in config["model"]["checkpoint_files"].items()
    }
    manifest = runner._manifest_document(
        config=config,
        run_id="schema-test",
        resolved_config_path=resolved,
        model_path=Path(config["paths"]["model"]),
        core={"core_id": config["model"]["core_id"]},
        checkpoint={"files": files, "aggregate_sha256": digest},
        benchmark_hashes={"prompts": digest, "answer_key": digest},
        rows=rows,
        artifacts={
            "resolved_config": "resolved_config.yaml",
            "manifest": "manifest.json",
            "cases": "cases.jsonl",
            "summary": "summary.json",
            "unscored_predictions": "predictions.unscored.jsonl",
        },
        completed_at=runner.utc_now(),
    )
    runner._schema_validate(manifest, LAB / "schemas" / "manifest.schema.json")
    assert manifest["status"] == "COMPLETE"
    assert manifest["model"]["checkpoint_aggregate_sha256"] == digest
    assert manifest["benchmark"]["answer_key_join_order"].startswith("opened_after")


def test_existing_output_directory_is_always_rejected(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        runner._create_output_dir(output)

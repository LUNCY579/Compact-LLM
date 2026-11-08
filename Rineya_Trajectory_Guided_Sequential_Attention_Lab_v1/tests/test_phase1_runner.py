from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tgsa.phase1_data import SyntheticVocabulary, generate_split
from tgsa.phase1_model import TinyLMConfig, build_model
from tgsa.phase1_runner import _evaluation_chunks, evaluate_model, load_study_config


def _study_config() -> dict[str, object]:
    return {
        "schema_version": "tgsa.phase1-model-identifiability-study.v1",
        "study_id": "unit-test",
        "confirmatory": False,
        "expected_base_commit": "0" * 40,
        "arms": ["cascade", "history_control", "learned_delta"],
        "seeds": [1, 2],
        "dataset": {},
        "model": {"dropout": 0.0},
        "training": {"steps": 2, "batch_size": 4},
        "evaluation": {},
        "analysis": {},
        "runtime": {},
    }


def _generation_spec() -> dict[str, object]:
    return {
        "families": {
            "direct_lookup": {
                "min_nodes": 5,
                "max_nodes": 7,
                "min_hops": 1,
                "max_hops": 1,
            },
            "pointer_chase": {
                "min_nodes": 6,
                "max_nodes": 9,
                "min_hops": 2,
                "max_hops": 4,
            },
            "state_update": {
                "min_steps": 3,
                "max_steps": 5,
            },
        }
    }


def test_evaluation_chunks_cover_examples_without_singleton_tail() -> None:
    for count, batch_size in ((1, 8), (2, 8), (9, 4), (17, 8), (32, 8)):
        chunks = _evaluation_chunks(count, batch_size)
        flattened = [index for start, end in chunks for index in range(start, end)]
        assert flattened == list(range(count))
        assert all(start < end for start, end in chunks)
        if count > 1:
            assert all(end - start >= 2 for start, end in chunks)
        assert all(end - start <= batch_size for start, end in chunks)


def test_load_study_config_validates_required_identifiability_controls(tmp_path: Path) -> None:
    path = tmp_path / "study.json"
    path.write_text(json.dumps(_study_config()), encoding="utf-8")
    loaded = load_study_config(path)
    assert loaded["study_id"] == "unit-test"

    invalid = _study_config()
    invalid["arms"] = ["cascade", "learned_delta"]
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="required arms"):
        load_study_config(path)

    invalid = _study_config()
    invalid["model"] = {"dropout": 0.1}
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="dropout=0"):
        load_study_config(path)


def test_evaluate_model_preserves_instance_order_and_intervention_records() -> None:
    vocab = SyntheticVocabulary(domain_size=12, max_hops=4, max_abs_step=3)
    examples = generate_split(
        split="test",
        count_by_family={"direct_lookup": 2, "pointer_chase": 2, "state_update": 2},
        dataset_seed=444,
        vocab=vocab,
        generation_spec=_generation_spec(),
    )
    max_sequence_length = max(len(example.tokens) - 1 for example in examples)
    model = build_model(
        TinyLMConfig(
            vocab_size=vocab.vocab_size,
            max_sequence_length=max_sequence_length,
            embed_dim=24,
            num_heads=4,
            num_layers=1,
            ffn_multiplier=2.0,
            dropout=0.0,
            trajectory_rank=6,
            arm="learned_delta",
        ),
        initialization_seed=55,
    )

    rows, summary, records = evaluate_model(
        model,
        examples,
        device=torch.device("cpu"),
        batch_size=4,
        split="test",
        arm="learned_delta",
        seed=1,
        intervention="identity",
    )
    assert [row["example_id"] for row in rows] == [example.id for example in examples]
    assert summary["overall"]["examples"] == len(examples)
    assert records == []

    rolled_rows, rolled_summary, rolled_records = evaluate_model(
        model,
        examples,
        device=torch.device("cpu"),
        batch_size=4,
        split="test",
        arm="learned_delta",
        seed=1,
        intervention="roll_batch",
        capture_intervention_records=True,
    )
    assert [row["example_id"] for row in rolled_rows] == [example.id for example in examples]
    assert rolled_summary["overall"]["examples"] == len(examples)
    assert rolled_records
    assert all(record["mode"] == "roll_batch" for record in rolled_records)
    assert sum(len(module._forward_pre_hooks) for module in model.modules()) == 0

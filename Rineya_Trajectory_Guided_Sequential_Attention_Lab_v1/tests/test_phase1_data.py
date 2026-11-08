from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tgsa.phase1_data import (
    SyntheticVocabulary,
    collate_examples,
    deterministic_batch_stream,
    generate_split,
    load_examples,
    materialize_dataset,
    validate_dataset_manifest,
)


def _dataset_config() -> dict[str, object]:
    return {
        "seed": 12345,
        "vocabulary": {
            "domain_size": 16,
            "max_hops": 6,
            "max_abs_step": 4,
        },
        "families": {
            "direct_lookup": {
                "min_nodes": 5,
                "max_nodes": 8,
                "min_hops": 1,
                "max_hops": 1,
            },
            "pointer_chase": {
                "min_nodes": 6,
                "max_nodes": 10,
                "min_hops": 2,
                "max_hops": 4,
            },
            "state_update": {
                "min_steps": 2,
                "max_steps": 5,
            },
        },
        "split_overrides": {
            "test": {
                "pointer_chase": {"min_hops": 4, "max_hops": 6},
                "state_update": {"min_steps": 4, "max_steps": 6},
            }
        },
        "splits": {
            "train": {
                "direct_lookup": 8,
                "pointer_chase": 8,
                "state_update": 8,
            },
            "dev": {
                "direct_lookup": 4,
                "pointer_chase": 4,
                "state_update": 4,
            },
            "test": {
                "direct_lookup": 4,
                "pointer_chase": 4,
                "state_update": 4,
            },
        },
    }


def test_vocabulary_ranges_are_disjoint_and_bounded() -> None:
    vocab = SyntheticVocabulary(domain_size=16, max_hops=6, max_abs_step=4)
    key_tokens = {vocab.key(index) for index in range(vocab.domain_size)}
    value_tokens = {vocab.value(index) for index in range(vocab.domain_size)}
    operation_tokens = {
        vocab.operation(step)
        for step in range(-vocab.max_abs_step, vocab.max_abs_step + 1)
    }
    hop_tokens = {vocab.hop(count) for count in range(1, vocab.max_hops + 1)}

    assert key_tokens.isdisjoint(value_tokens)
    assert key_tokens.isdisjoint(operation_tokens)
    assert key_tokens.isdisjoint(hop_tokens)
    assert value_tokens.isdisjoint(operation_tokens)
    assert value_tokens.isdisjoint(hop_tokens)
    assert operation_tokens.isdisjoint(hop_tokens)
    assert max(key_tokens | value_tokens | operation_tokens | hop_tokens) < vocab.vocab_size


def test_split_generation_is_byte_deterministic() -> None:
    config = _dataset_config()
    vocab = SyntheticVocabulary(domain_size=16, max_hops=6, max_abs_step=4)
    counts = {"direct_lookup": 5, "pointer_chase": 5, "state_update": 5}

    first = generate_split(
        split="train",
        count_by_family=counts,
        dataset_seed=12345,
        vocab=vocab,
        generation_spec=config,
    )
    second = generate_split(
        split="train",
        count_by_family=counts,
        dataset_seed=12345,
        vocab=vocab,
        generation_spec=config,
    )

    first_payload = [example.to_json_dict() for example in first]
    second_payload = [example.to_json_dict() for example in second]
    assert first_payload == second_payload
    assert len({example.id for example in first}) == len(first)
    assert len({example.latent_signature for example in first}) == len(first)


def test_materialization_hashes_and_cross_split_uniqueness(tmp_path: Path) -> None:
    output_dir = tmp_path / "dataset"
    manifest = materialize_dataset(
        dataset_config=_dataset_config(),
        output_dir=output_dir,
    )
    validated = validate_dataset_manifest(output_dir / "dataset_manifest.json")

    assert manifest["dataset_config_sha256"] == validated["dataset_config_sha256"]
    assert validated["cross_split_unique_latent_signatures"] is True

    signatures: dict[str, set[str]] = {}
    for split in ("train", "dev", "test"):
        examples = load_examples(output_dir / f"{split}.jsonl")
        signatures[split] = {example.latent_signature for example in examples}
        assert len(signatures[split]) == len(examples)
        assert all(example.split == split for example in examples)

    assert signatures["train"].isdisjoint(signatures["dev"])
    assert signatures["train"].isdisjoint(signatures["test"])
    assert signatures["dev"].isdisjoint(signatures["test"])


def test_manifest_validation_detects_tampering(tmp_path: Path) -> None:
    output_dir = tmp_path / "dataset"
    materialize_dataset(dataset_config=_dataset_config(), output_dir=output_dir)
    train_path = output_dir / "train.jsonl"
    train_path.write_text(train_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        validate_dataset_manifest(output_dir / "dataset_manifest.json")


def test_collator_places_exactly_one_answer_target_per_example(tmp_path: Path) -> None:
    output_dir = tmp_path / "dataset"
    materialize_dataset(dataset_config=_dataset_config(), output_dir=output_dir)
    examples = load_examples(output_dir / "train.jsonl")[:6]
    batch = collate_examples(examples)

    assert batch.input_ids.shape == batch.labels.shape == batch.loss_mask.shape
    assert batch.key_padding_mask.shape == batch.input_ids.shape
    assert torch.equal(batch.loss_mask.sum(dim=1), torch.ones(len(examples), dtype=torch.long))
    for row, example in enumerate(examples):
        index = example.answer_input_index
        assert batch.loss_mask[row, index]
        assert batch.labels[row, index].item() == example.answer_token
        assert batch.answer_input_indices[row].item() == index
        assert batch.answer_tokens[row].item() == example.answer_token


def test_training_batch_order_is_arm_independent_and_seeded(tmp_path: Path) -> None:
    output_dir = tmp_path / "dataset"
    materialize_dataset(dataset_config=_dataset_config(), output_dir=output_dir)
    examples = load_examples(output_dir / "train.jsonl")

    first_stream = deterministic_batch_stream(examples, batch_size=5, seed=999)
    second_stream = deterministic_batch_stream(examples, batch_size=5, seed=999)
    third_stream = deterministic_batch_stream(examples, batch_size=5, seed=1000)

    first_ids = [next(first_stream)[1].example_ids for _ in range(3)]
    second_ids = [next(second_stream)[1].example_ids for _ in range(3)]
    third_ids = [next(third_stream)[1].example_ids for _ in range(3)]

    assert first_ids == second_ids
    assert first_ids != third_ids

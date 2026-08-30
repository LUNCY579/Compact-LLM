from __future__ import annotations

from pathlib import Path

from tgsa.phase2_data import (
    collate_phase2_examples,
    load_phase2_examples,
    materialize_phase2_dataset,
    validate_phase2_dataset,
)


def test_materialized_splits_are_disjoint_and_target_mask_starts_at_continuation(
    tmp_path: Path,
) -> None:
    config = {
        "seed": 7,
        "vocabulary": {"domain_size": 16},
        "continuation_lengths": [5, 6],
        "split_generation_order": ["test", "dev", "train"],
        "splits": {
            "train": {"affine": 6, "second_order": 6, "switch": 6},
            "dev": {"affine": 3, "second_order": 3, "switch": 3},
            "test": {"affine": 3, "second_order": 3, "switch": 3},
        },
    }
    manifest = materialize_phase2_dataset(dataset_config=config, output_dir=tmp_path)
    assert manifest["unbiased_split"] == "test"
    validate_phase2_dataset(tmp_path / "dataset_manifest.json")
    train = load_phase2_examples(tmp_path / "train.jsonl")
    dev = load_phase2_examples(tmp_path / "dev.jsonl")
    assert not ({row.latent_signature for row in train} & {row.latent_signature for row in dev})
    batch = collate_phase2_examples(train[:3])
    for row, example in enumerate(train[:3]):
        assert not batch.target_mask[row, : example.prompt_length].any()
        assert batch.target_mask[row, example.prompt_length : len(example.tokens)].all()
        assert not batch.attention_mask[row, len(example.tokens) :].any()

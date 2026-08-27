from __future__ import annotations

from pathlib import Path

import pytest

from rineya_pcsm.data import (
    CorpusAccessPolicy,
    LineageRecord,
    assign_components,
    connected_components,
    encode_chat_example,
    stable_sample_ids,
)
from rineya_pcsm.hashing import canonical_json_hash


def record(index: int, family: str, origins: list[str]) -> LineageRecord:
    return LineageRecord(
        sft_id=f"row-{index}",
        split="train",
        source_family_id=family,
        origin_group_ids=tuple(origins),
        domain="test",
        task_key="lookup",
        stratum="4k",
        evidence_position="early",
        prompt_sha256="a" * 64,
    )


def test_components_close_transitively_over_origins_and_families() -> None:
    rows = [
        record(0, "f0", ["a"]),
        record(1, "f0", ["b"]),
        record(2, "f1", ["b", "c"]),
        record(3, "f2", ["z"]),
    ]
    components = connected_components(rows)
    assert len(components) == 2
    assert sorted(len(component.row_ids) for component in components) == [1, 3]


def test_assignment_is_deterministic_and_semantically_disjoint() -> None:
    rows = [record(i, f"f{i}", [f"o{i}"]) for i in range(8)]
    components = connected_components(rows)
    first = assign_components(
        components, meta_train_count=4, mask_cal_count=2, split_salt="frozen"
    )
    second = assign_components(
        list(reversed(components)),
        meta_train_count=4,
        mask_cal_count=2,
        split_salt="frozen",
    )
    assert first == second
    assert len(first.unused) == 2


def test_stable_sample_ids_is_order_invariant() -> None:
    values = [f"row-{index}" for index in range(20)]
    assert stable_sample_ids(values, 5, "salt") == stable_sample_ids(
        reversed(values), 5, "salt"
    )


def test_sealed_path_is_never_authorized_before_choices_freeze(
    tmp_path: Path,
) -> None:
    sealed = tmp_path / "test"
    sealed.mkdir()
    policy = CorpusAccessPolicy([sealed])
    with pytest.raises(PermissionError):
        policy.authorize(sealed / "shard.jsonl", "selection")
    policy.freeze_choices(canonical_json_hash({"choice": "final"}))
    assert (
        policy.authorize(sealed / "shard.jsonl", "confirmatory").name == "shard.jsonl"
    )


class PrefixTokenizer:
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: object,
    ) -> list[int]:
        del tokenize, add_generation_prompt, kwargs
        return [
            token
            for index, message in enumerate(conversation)
            for token in [index + 1, *range(len(message["content"]))]
        ]

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": list(range(len(text)))}


def test_chat_encoding_labels_only_assistant_and_forbids_truncation() -> None:
    messages = [
        {"role": "user", "content": "ask"},
        {"role": "assistant", "content": "ok"},
    ]
    input_ids, labels = encode_chat_example(
        PrefixTokenizer(), messages, maximum_tokens=16, example_id="x"
    )
    assert len(input_ids) == len(labels)
    assert labels[:4] == [-100] * 4
    assert all(value != -100 for value in labels[4:])
    with pytest.raises(RuntimeError, match="forbids silent truncation"):
        encode_chat_example(
            PrefixTokenizer(), messages, maximum_tokens=2, example_id="x"
        )

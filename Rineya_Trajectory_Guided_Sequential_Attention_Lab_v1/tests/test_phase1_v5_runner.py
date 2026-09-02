"""Determinism and accounting tests for the v5 execution engine."""

from __future__ import annotations

from tgsa.phase1_data import SyntheticVocabulary, generate_split
from tgsa.phase1_v5_runner import _exact_full_batch_stream, _learning_rate


def _examples():
    vocab = SyntheticVocabulary()
    return generate_split(
        split="train",
        count_by_family={"pointer_chase": 13},
        dataset_seed=991,
        vocab=vocab,
        generation_spec={
            "families": {
                "pointer_chase": {
                    "min_nodes": 6,
                    "max_nodes": 8,
                    "min_hops": 1,
                    "max_hops": 3,
                    "reject_trivial": True,
                }
            }
        },
    )


def test_full_batch_stream_has_exact_batch_size_and_resume_identity() -> None:
    examples = _examples()
    stream = _exact_full_batch_stream(
        examples,
        batch_size=4,
        seed=123,
    )
    first = [next(stream) for _ in range(7)]
    assert [step for step, _epoch, _batch in first] == list(range(7))
    assert all(batch.input_ids.shape[0] == 4 for _step, _epoch, batch in first)

    resumed = _exact_full_batch_stream(
        examples,
        batch_size=4,
        seed=123,
        start_step=3,
    )
    replay = [next(resumed) for _ in range(4)]
    assert [
        batch.example_ids for _step, _epoch, batch in replay
    ] == [
        batch.example_ids for _step, _epoch, batch in first[3:7]
    ]


def test_full_batch_stream_never_counts_epoch_remainders() -> None:
    examples = _examples()  # 13 examples; one is dropped from each shuffled epoch.
    stream = _exact_full_batch_stream(examples, batch_size=4, seed=17)
    batches = [next(stream) for _ in range(6)]
    assert [epoch for _step, epoch, _batch in batches] == [0, 0, 0, 1, 1, 1]
    assert sum(batch.input_ids.shape[0] for _step, _epoch, batch in batches) == 24


def test_cosine_schedule_is_warm_then_decays_to_zero() -> None:
    values = [
        _learning_rate(
            step=step,
            total_steps=20,
            peak=1.0,
            warmup_fraction=0.10,
        )
        for step in range(20)
    ]
    assert values[0] == 0.5
    assert values[1] == 1.0
    assert values[-1] == 0.0
    assert all(left >= right for left, right in zip(values[1:], values[2:]))

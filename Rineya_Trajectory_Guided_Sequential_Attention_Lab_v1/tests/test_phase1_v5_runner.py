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


# ---------------------------------------------------------------------------
# Interrupt safety: the emergency snapshot must never become a resume source.
# ---------------------------------------------------------------------------

import json
from pathlib import Path

import pytest
import torch

from tgsa.phase1_data import Phase1Example
from tgsa.phase1_v5 import load_v5_plan, probe_exposure, resolve_rung
from tgsa.phase1_v5_runner import INTERRUPTED_STATE_NAME, _train_cell

LAB_ROOT = Path(__file__).resolve().parents[1]
SMOKE_PLAN = LAB_ROOT / "configs" / "phase1_v5_smoke.json"


def _cell_inputs():
    """Everything `_train_cell` needs, at smoke scale, on CPU.

    The smoke plan is used unmodified: `checkpoint_interval_steps` is 4 and the
    probe exposure is 192 presentations at batch 16, i.e. 12 optimizer steps.
    Periodic checkpoints therefore land at steps 4 and 8, which is what makes
    an interrupt at step 10 a meaningful test.
    """

    plan = load_v5_plan(SMOKE_PLAN)
    vocab = SyntheticVocabulary(domain_size=32, max_hops=8, max_abs_step=8)
    spec = {
        "families": {
            "pointer_chase": {
                "min_nodes": 6,
                "max_nodes": 8,
                "min_hops": 1,
                "max_hops": 1,
                "reject_trivial": True,
            }
        }
    }

    def split(name: str, count: int, seed: int) -> list[Phase1Example]:
        return generate_split(
            split=name,
            count_by_family={"pointer_chase": count},
            dataset_seed=seed,
            vocab=vocab,
            generation_spec=spec,
        )

    train, dev, test = split("train", 96, 41), split("dev", 32, 42), split("test", 32, 43)
    max_len = max(len(e.tokens) - 1 for e in (*train, *dev, *test))
    return dict(
        plan=plan,
        stage="probe",
        rung=resolve_rung(plan, "S1"),
        exposure=probe_exposure(plan),
        arm="cascade",
        seed=7,
        train=train,
        dev=dev,
        test=test,
        vocab=vocab,
        max_sequence_length=max_len,
        device=torch.device("cpu"),
        identity={"test": "interrupt-contract"},
    )


def _raise_after_real_step(monkeypatch, at_call: int) -> None:
    """Reproduce the exact race: the step is APPLIED, then we die.

    `next_step` is assigned on the line after `optimizer.step()` returns, so a
    signal delivered in that window leaves the parameters advanced while
    `next_step` still names the step just applied. Calling through to the real
    `step` and only then raising puts the process in precisely that state.
    """

    real_step = torch.optim.AdamW.step
    seen = {"n": 0}

    def patched(self, *args, **kwargs):
        result = real_step(self, *args, **kwargs)
        seen["n"] += 1
        if seen["n"] == at_call:
            raise KeyboardInterrupt("simulated Ctrl-C after optimizer.step()")
        return result

    monkeypatch.setattr(torch.optim.AdamW, "step", patched)


def test_an_interrupt_after_the_optimizer_step_does_not_corrupt_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cell interrupted mid-step must resume to a bit-identical model.

    Before this was fixed the exception handler wrote its snapshot to
    `training_state.pt` with the stale `next_step`, so resume re-applied one
    optimizer step and the cell diverged from an uninterrupted run while still
    producing a well-formed, fingerprint-matching report. Nothing downstream
    would have caught it.
    """

    inputs = _cell_inputs()

    reference = _train_cell(cell_dir=tmp_path / "clean", resume=True, **inputs)
    expected_hash = reference["model_state_sha256"]

    interrupted_dir = tmp_path / "interrupted"
    _raise_after_real_step(monkeypatch, at_call=10)
    with pytest.raises(KeyboardInterrupt):
        _train_cell(cell_dir=interrupted_dir, resume=True, **inputs)
    monkeypatch.undo()

    # The emergency snapshot exists, and is NOT the resume source.
    assert (interrupted_dir / INTERRUPTED_STATE_NAME).is_file()
    failure = json.loads((interrupted_dir / "failure.json").read_text(encoding="utf-8"))
    assert failure["interrupted_state_is_resumable"] is False
    assert failure["interrupted_state"] == INTERRUPTED_STATE_NAME
    assert failure["error_type"] == "KeyboardInterrupt"

    # The resume source is the last PERIODIC checkpoint, written inside the loop
    # after the bookkeeping, so it names a step the trajectory actually passed.
    state = torch.load(interrupted_dir / "training_state.pt", weights_only=False)
    assert state["next_step"] == 8, "expected the step-8 periodic checkpoint"
    assert failure["resume_source"] == "training_state.pt"

    # The optimizer had in fact applied more steps than the periodic checkpoint
    # records -- which is exactly why the handler must not nominate a resume point.
    applied = failure["optimizer_applied_steps"]
    assert applied["all_parameters_agree"] is True
    assert applied["applied_steps"] > state["next_step"]

    resumed = _train_cell(cell_dir=interrupted_dir, resume=True, **inputs)
    assert resumed["training"]["resumed_from_step"] == 8
    assert resumed["model_state_sha256"] == expected_hash


def test_the_emergency_snapshot_never_overwrites_the_periodic_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins the invariant directly: `training_state.pt` is untouched by the handler."""

    inputs = _cell_inputs()
    cell_dir = tmp_path / "cell"

    _raise_after_real_step(monkeypatch, at_call=6)
    with pytest.raises(KeyboardInterrupt):
        _train_cell(cell_dir=cell_dir, resume=True, **inputs)
    monkeypatch.undo()

    periodic = torch.load(cell_dir / "training_state.pt", weights_only=False)
    emergency = torch.load(cell_dir / INTERRUPTED_STATE_NAME, weights_only=False)
    assert periodic["next_step"] == 4
    # The emergency snapshot carries the stale counter that made resume unsafe.
    assert emergency["next_step"] == 5

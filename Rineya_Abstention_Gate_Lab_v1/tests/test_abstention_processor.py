"""The inference-time gates must match the offline gate they claim to implement."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from abstention_processor import (  # noqa: E402
    LogitOffsetGate,
    NormalisedAbstentionGate,
)

UNKNOWN = 5
PROMPT = 7


def scores(values: dict[int, float], size: int = 12) -> torch.Tensor:
    row = torch.full((1, size), -30.0)
    for index, value in values.items():
        row[0, index] = value
    return row


def ids(length: int) -> torch.LongTensor:
    return torch.zeros((1, length), dtype=torch.long)


def argmax(row: torch.Tensor) -> int:
    return int(row.argmax(dim=-1)[0])


# --------------------------------------------------------------------------
# LogitOffsetGate == "add a constant to one logit"
# --------------------------------------------------------------------------


def test_offset_gate_is_exactly_a_constant_added_to_one_logit():
    bias = 3.0
    gate = LogitOffsetGate(prompt_length=PROMPT, unknown_token_id=UNKNOWN, bias=bias)
    for gap in (0.5, 2.0, 2.999, 3.001, 5.0):
        row = scores({0: 10.0, UNKNOWN: 10.0 - gap})
        gated = gate(ids(PROMPT), row.clone())
        biased = row.clone()
        biased[0, UNKNOWN] += bias
        assert (argmax(gated) == UNKNOWN) == (argmax(biased) == UNKNOWN), gap


def test_offset_gate_fires_exactly_at_the_boundary():
    gate = LogitOffsetGate(prompt_length=PROMPT, unknown_token_id=UNKNOWN, bias=3.0)
    assert argmax(gate(ids(PROMPT), scores({0: 10.0, UNKNOWN: 7.0}))) == UNKNOWN
    assert argmax(gate(ids(PROMPT), scores({0: 10.0, UNKNOWN: 6.99}))) == 0


def test_offset_gate_leaves_an_already_abstaining_row_alone():
    gate = LogitOffsetGate(prompt_length=PROMPT, unknown_token_id=UNKNOWN, bias=3.0)
    row = scores({UNKNOWN: 10.0, 0: 1.0})
    assert argmax(gate(ids(PROMPT), row.clone())) == UNKNOWN


# --------------------------------------------------------------------------
# NormalisedAbstentionGate == the S1 threshold on summed probability mass
# --------------------------------------------------------------------------


def test_normalised_gate_matches_a_hand_computed_probability_mass():
    variants = [UNKNOWN, 6, 7]
    row = scores({0: 4.0, UNKNOWN: 1.0, 6: 0.5, 7: 0.0})
    logprobs = torch.log_softmax(row.float(), dim=-1)
    mass = float(torch.logsumexp(logprobs[0, variants], dim=0))

    below = NormalisedAbstentionGate(
        prompt_length=PROMPT,
        unknown_token_id=UNKNOWN,
        abstention_token_ids=variants,
        threshold=mass + 0.01,
    )
    above = NormalisedAbstentionGate(
        prompt_length=PROMPT,
        unknown_token_id=UNKNOWN,
        abstention_token_ids=variants,
        threshold=mass - 0.01,
    )
    assert argmax(below(ids(PROMPT), row.clone())) == 0
    assert argmax(above(ids(PROMPT), row.clone())) == UNKNOWN


def test_normalised_gate_is_not_a_constant_offset():
    """Two rows with the same z_UNK - z_top1 but different normalisers."""

    variants = [UNKNOWN]
    peaked = scores({0: 4.0, UNKNOWN: 1.0})
    # Same 3.0 gap, but mass spread over many rivals, so P(UNKNOWN) is smaller.
    flat = scores({0: 4.0, UNKNOWN: 1.0, 1: 3.9, 2: 3.8, 3: 3.7, 8: 3.6, 9: 3.5})

    def mass(row):
        lp = torch.log_softmax(row.float(), dim=-1)
        return float(torch.logsumexp(lp[0, variants], dim=0))

    assert mass(peaked) > mass(flat)
    gate = NormalisedAbstentionGate(
        prompt_length=PROMPT,
        unknown_token_id=UNKNOWN,
        abstention_token_ids=variants,
        threshold=(mass(peaked) + mass(flat)) / 2,
    )
    # A constant offset gate would treat these two rows identically; this one
    # must not. That is precisely why S1 cannot be baked into the weights.
    assert argmax(gate(ids(PROMPT), peaked.clone())) == UNKNOWN
    assert argmax(gate(ids(PROMPT), flat.clone())) == 0

    offset = LogitOffsetGate(prompt_length=PROMPT, unknown_token_id=UNKNOWN, bias=3.0)
    assert argmax(offset(ids(PROMPT), peaked.clone())) == UNKNOWN
    assert argmax(offset(ids(PROMPT), flat.clone())) == UNKNOWN


# --------------------------------------------------------------------------
# Both gates act on the first generated token only
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gate",
    [
        LogitOffsetGate(prompt_length=PROMPT, unknown_token_id=UNKNOWN, bias=30.0),
        NormalisedAbstentionGate(
            prompt_length=PROMPT,
            unknown_token_id=UNKNOWN,
            abstention_token_ids=[UNKNOWN],
            threshold=-math.inf,
        ),
    ],
)
def test_gates_stand_down_after_the_first_generated_token(gate):
    row = scores({0: 10.0, UNKNOWN: -20.0})
    assert argmax(gate(ids(PROMPT), row.clone())) == UNKNOWN
    for later in (PROMPT + 1, PROMPT + 5):
        assert argmax(gate(ids(later), row.clone())) == 0


def test_gates_do_not_mutate_the_caller_tensor():
    gate = LogitOffsetGate(prompt_length=PROMPT, unknown_token_id=UNKNOWN, bias=30.0)
    row = scores({0: 10.0, UNKNOWN: -20.0})
    original = row.clone()
    gate(ids(PROMPT), row)
    assert torch.equal(row, original)


def test_prompt_length_must_be_positive():
    with pytest.raises(ValueError):
        LogitOffsetGate(prompt_length=0, unknown_token_id=UNKNOWN, bias=1.0)


def test_normalised_gate_needs_at_least_one_token():
    with pytest.raises(ValueError):
        NormalisedAbstentionGate(
            prompt_length=PROMPT,
            unknown_token_id=UNKNOWN,
            abstention_token_ids=[],
            threshold=0.0,
        )


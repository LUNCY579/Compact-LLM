"""The opt-in confidence capture must be additive and arithmetically right."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "Rineya_Long_Context_Demo_v1"
    / "scripts"
    / "run_local_hf.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("run_local_hf", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load_runner()
EOS = 15  # inside the fake vocabulary so it can be scored


class FakeTokenizer:
    """Single-token surface forms, as the Qwen3 tokenizer gives for UNKNOWN."""

    eos_token_id = EOS
    vocabulary = {
        "UNKNOWN": 5,
        "unknown": 6,
        "Unknown": 7,
        " UNKNOWN": 8,
        " unknown": 9,
        " Unknown": 10,
    }

    def encode(self, text, add_special_tokens=True):
        return [self.vocabulary[text]]

    def decode(self, ids, skip_special_tokens=False):
        return f"<{ids[0]}>"


class MultiTokenTokenizer(FakeTokenizer):
    def encode(self, text, add_special_tokens=True):
        return [self.vocabulary[text], 42]


def logits(values: dict[int, float], size: int = 16) -> torch.Tensor:
    row = torch.full((1, size), -30.0)
    for index, value in values.items():
        row[0, index] = value
    return row


def summarize(step_logits, continuation, tokenizer=None, top_k=3):
    return RUNNER.summarize_confidence(
        torch=torch,
        tokenizer=tokenizer or FakeTokenizer(),
        step_logits=step_logits,
        continuation=torch.tensor(continuation),
        top_k=top_k,
        unknown_tokens=RUNNER.unknown_continuation_tokens(tokenizer or FakeTokenizer()),
    )


# --------------------------------------------------------------------------
# Flag plumbing
# --------------------------------------------------------------------------


def test_confidence_flags_are_opt_in_and_default_off():
    parser_source = RUNNER_PATH.read_text(encoding="utf-8")
    assert '"--emit-logprobs",\n        action="store_true"' in parser_source
    assert "return_dict_in_generate" in parser_source
    # The frozen contract must still be the literal default.
    assert '"do_sample": False' in parser_source
    assert '"use_cache": True' in parser_source


def test_unknown_continuation_tokens_dedupe_and_flag_multi_token_forms():
    entries = RUNNER.unknown_continuation_tokens(FakeTokenizer())
    assert [entry["token_id"] for entry in entries] == [5, 6, 7, 8, 9, 10]
    assert entries[0]["surface"] == "UNKNOWN"
    assert all(entry["single_token"] for entry in entries)
    multi = RUNNER.unknown_continuation_tokens(MultiTokenTokenizer())
    assert not any(entry["single_token"] for entry in multi)


# --------------------------------------------------------------------------
# Arithmetic
# --------------------------------------------------------------------------


def test_step_confidence_matches_hand_computed_softmax():
    # Two live logits, 1.0 apart; everything else is negligible.
    row = logits({0: 1.0, 1: 0.0})[0]
    stats = RUNNER.step_confidence(torch, row, chosen_id=0)
    reference = torch.log_softmax(row.to(torch.float32), dim=-1)
    assert stats["token_logprob"] == pytest.approx(float(reference[0]), abs=1e-5)
    assert stats["top1_logprob"] == pytest.approx(float(reference[0]), abs=1e-5)
    assert stats["top2_logprob"] == pytest.approx(float(reference[1]), abs=1e-5)
    assert stats["logprob_margin"] == pytest.approx(1.0, abs=1e-4)
    assert stats["prob_margin"] == pytest.approx(
        float(reference[0].exp() - reference[1].exp()), abs=1e-5
    )
    probs = reference.exp()
    assert stats["entropy"] == pytest.approx(
        float(-(probs * reference).sum()), abs=1e-4
    )


def test_uniform_distribution_has_maximum_entropy_and_zero_margin():
    row = torch.zeros(8)
    stats = RUNNER.step_confidence(torch, row, chosen_id=3)
    assert stats["entropy"] == pytest.approx(math.log(8), abs=1e-5)
    assert stats["logprob_margin"] == pytest.approx(0.0, abs=1e-6)


def test_unknown_mass_is_read_from_the_first_step_only():
    step0 = logits({0: 4.0, 5: 2.0, 6: 1.0})
    step1 = logits({5: 20.0})  # a late UNKNOWN spike must not count
    result = summarize([step0, step1], [0, EOS])
    reference = torch.log_softmax(step0[0].to(torch.float32), dim=-1)
    assert result["unknown"]["exact_token_id"] == 5
    assert result["unknown"]["exact_logprob"] == pytest.approx(
        float(reference[5]), abs=1e-5
    )
    expected_variant = float(
        torch.logsumexp(reference[torch.tensor([5, 6, 7, 8, 9, 10])], dim=0)
    )
    assert result["unknown"]["variant_logprob"] == pytest.approx(
        expected_variant, abs=1e-5
    )
    # The variant mass is a superset of the exact mass, so never smaller.
    assert result["unknown"]["variant_logprob"] >= result["unknown"]["exact_logprob"]


def test_content_tokens_stop_at_the_first_eos():
    step = logits({0: 3.0, 1: 1.0})
    result = summarize([step, step, step, step], [0, 1, EOS, 0])
    assert result["scored_steps"] == 4
    assert result["content_tokens"] == 2
    assert result["eos_emitted"] is True
    assert len(result["per_step"]) == 4
    assert result["per_step"][2]["is_eos"] is True


def test_answer_statistics_exclude_the_eos_step():
    confident = logits({0: 12.0})
    unconfident = logits({0: 0.05, 1: 0.0})
    result = summarize([confident, unconfident, unconfident], [0, 0, EOS])
    per_step = result["per_step"]
    assert result["content_tokens"] == 2
    answer = result["answer"]
    assert answer["min_logprob"] == pytest.approx(
        min(per_step[0]["token_logprob"], per_step[1]["token_logprob"])
    )
    assert answer["mean_logprob"] == pytest.approx(
        (per_step[0]["token_logprob"] + per_step[1]["token_logprob"]) / 2
    )
    assert answer["sum_logprob"] == pytest.approx(
        per_step[0]["token_logprob"] + per_step[1]["token_logprob"]
    )
    assert answer["max_entropy"] == pytest.approx(
        max(per_step[0]["entropy"], per_step[1]["entropy"])
    )
    # The EOS step is scored but must not enter the answer aggregate.
    assert answer["min_logprob_margin"] == pytest.approx(
        min(per_step[0]["logprob_margin"], per_step[1]["logprob_margin"])
    )


def test_answer_is_none_when_the_model_emits_nothing():
    result = summarize([logits({EOS: 5.0})], [EOS])
    assert result["content_tokens"] == 0
    assert result["answer"] is None
    assert result["first_token"]["is_eos"] is True
    # The step-0 UNKNOWN mass is still well defined.
    assert result["unknown"]["exact_logprob"] < 0.0


def test_first_step_top_k_is_ranked_and_truncated():
    result = summarize([logits({0: 5.0, 1: 4.0, 2: 3.0, 3: 2.0})], [0, EOS], top_k=3)
    top = result["first_token"]["top_k"]
    assert [entry["token_id"] for entry in top] == [0, 1, 2]
    assert top == sorted(top, key=lambda entry: -entry["logprob"])


def test_no_scored_steps_is_an_error():
    with pytest.raises(RuntimeError):
        summarize([], [])


def test_multiple_eos_ids_are_all_honoured():
    class MultiEos(FakeTokenizer):
        eos_token_id = [EOS, 14]

    result = summarize([logits({0: 3.0}), logits({0: 3.0})], [0, 14], MultiEos())
    assert result["content_tokens"] == 1
    assert result["eos_emitted"] is True


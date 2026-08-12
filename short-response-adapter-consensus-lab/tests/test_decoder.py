from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from consensus_lab.decoding.decoder import (
    generate_adaptive_consultation,
    generate_reference,
    generate_separate_cache,
)


@dataclass(frozen=True)
class FakeExpert:
    name: str
    validation_weight: float = 1.0
    temperature: float = 1.0


class FakeTokenizer:
    eos_token_id = 6

    @staticmethod
    def decode(ids: list[int], skip_special_tokens: bool = True) -> str:
        kept = [value for value in ids if not (skip_special_tokens and value == 6)]
        return " ".join(str(value) for value in kept)


class FakeCommittee:
    def __init__(self) -> None:
        self.experts = [FakeExpert("a"), FakeExpert("b"), FakeExpert("c")]
        self.tokenizer = FakeTokenizer()
        self.device = torch.device("cpu")
        self.calls: list[tuple[str, tuple[int, ...], bool]] = []

    def forward(
        self,
        expert: FakeExpert,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, Any | None]:
        incoming = tuple(int(value) for value in input_ids.flatten().tolist())
        self.calls.append((expert.name, incoming, past_key_values is not None))
        if past_key_values is None:
            prefix = incoming
        else:
            prefix = tuple(past_key_values["prefix"]) + incoming
        next_token = 6 if len(prefix) >= 5 else (sum(prefix) + len(prefix)) % 5 + 1
        values = torch.zeros(7)
        values[next_token] = 10.0
        return values, {"prefix": prefix} if use_cache else None

    def subset(self, names: list[str]) -> "FakeCommittee":
        view = FakeCommittee()
        by_name = {expert.name: expert for expert in self.experts}
        view.experts = [by_name[name] for name in names]
        view.calls = self.calls
        return view


def test_cache_parity_and_prefix_synchronization() -> None:
    prompt = torch.tensor([[1, 2]])
    reference_committee = FakeCommittee()
    cached_committee = FakeCommittee()
    reference = generate_reference(reference_committee, prompt, max_new_tokens=8)
    cached = generate_separate_cache(cached_committee, prompt, max_new_tokens=8)
    assert cached.output_token_ids == reference.output_token_ids
    assert cached.output_text == reference.output_text
    assert cached.stop_reason == reference.stop_reason == "eos"
    reference_inputs: dict[int, list[tuple[int, ...]]] = {}
    for index in range(0, len(reference_committee.calls), 3):
        block = reference_committee.calls[index : index + 3]
        reference_inputs[index // 3] = [item[1] for item in block]
    assert all(len(set(values)) == 1 for values in reference_inputs.values())


def test_eos_minimum_guard_overrides_early_eos() -> None:
    class EarlyEOS(FakeCommittee):
        def forward(self, expert: FakeExpert, input_ids: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, Any]:
            values = torch.zeros(7)
            values[6] = 10.0
            values[2] = 9.0
            return values, {"prefix": tuple(input_ids.flatten().tolist())}

    result = generate_reference(EarlyEOS(), torch.tensor([[1]]), max_new_tokens=2, eos_policy="minimum_tokens", minimum_tokens=1)
    assert result.output_token_ids == [2, 6]
    assert result.steps[0]["eos_policy"]["overrode"] is True


def test_eos_strict_majority_rejects_plurality_eos() -> None:
    class SplitEOS(FakeCommittee):
        def __init__(self) -> None:
            super().__init__()
            self.experts = [FakeExpert("a"), FakeExpert("b"), FakeExpert("c"), FakeExpert("d")]

        def forward(self, expert: FakeExpert, input_ids: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, Any]:
            values = torch.zeros(7)
            values[6 if expert.name in {"a", "b"} else (2 if expert.name == "c" else 3)] = 10.0
            values[1] = 9.0
            return values, {"prefix": tuple(input_ids.flatten().tolist())}

    result = generate_reference(
        SplitEOS(), torch.tensor([[1]]), max_new_tokens=1, eos_policy="strict_majority"
    )
    assert result.output_token_ids != [6]
    assert result.steps[0]["eos_policy"]["overrode"] is True


def test_adaptive_consultation_triggers_and_counts_extra_forwards() -> None:
    class Disagreeing(FakeCommittee):
        def __init__(self) -> None:
            super().__init__()
            self.experts = [FakeExpert("a"), FakeExpert("b"), FakeExpert("c"), FakeExpert("d")]

        def subset(self, names: list[str]) -> "Disagreeing":
            view = Disagreeing()
            by_name = {expert.name: expert for expert in self.experts}
            view.experts = [by_name[name] for name in names]
            view.calls = self.calls
            return view

        def forward(self, expert: FakeExpert, input_ids: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, Any]:
            values = torch.zeros(7)
            values[{"a": 1, "b": 2, "c": 2, "d": 2}[expert.name]] = 10.0
            return values, None

    result = generate_adaptive_consultation(
        Disagreeing(),
        torch.tensor([[1]]),
        core_expert_names=["a", "b", "c"],
        agreement_threshold=0.67,
        max_new_tokens=1,
    )
    assert result.output_token_ids == [2]
    assert result.forward_passes == 4
    assert result.steps[0]["adaptive_consultation"]["triggered"] is True

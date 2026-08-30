from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from tgsa.phase2_decode import (
    assert_exact_greedy_equivalence,
    greedy_generate,
    verified_mtp_generate,
)


class ArithmeticProposalModel(nn.Module):
    def __init__(self, vocab: int = 23, *, wrong_auxiliary: bool = False) -> None:
        super().__init__()
        self.vocab = vocab
        self.wrong_auxiliary = wrong_auxiliary

    def _logits_for_offset(self, input_ids: torch.Tensor, offset: int) -> torch.Tensor:
        target = (input_ids + offset) % self.vocab
        if self.wrong_auxiliary and offset > 1:
            target = (target + 3) % self.vocab
        logits = torch.full(
            (*input_ids.shape, self.vocab),
            -1000.0,
            device=input_ids.device,
        )
        return logits.scatter(-1, target.unsqueeze(-1), 1000.0)

    def forward(self, input_ids: torch.Tensor, **_: object) -> SimpleNamespace:
        return SimpleNamespace(
            logits=self._logits_for_offset(input_ids, 1),
            mtp_logits={
                2: self._logits_for_offset(input_ids, 2),
                3: self._logits_for_offset(input_ids, 3),
                4: self._logits_for_offset(input_ids, 4),
            },
        )


def test_all_correct_proposals_reduce_serial_target_rounds() -> None:
    model = ArithmeticProposalModel()
    prompt = torch.tensor([[2, 3, 4]])
    result = assert_exact_greedy_equivalence(model, prompt, max_new_tokens=12)
    assert result["exact"] is True
    stats = result["verified_mtp"]
    assert stats["auxiliary_acceptance_rate"] == 1.0
    assert stats["serial_round_speedup_vs_ar"] > 2.0
    assert stats["serial_round_reduction_fraction_vs_ar"] > 0.5


def test_wrong_proposals_are_corrected_without_changing_greedy_output() -> None:
    model = ArithmeticProposalModel(wrong_auxiliary=True)
    prompt = torch.tensor([[2, 3, 4]])
    greedy = greedy_generate(model, prompt, max_new_tokens=9)
    verified = verified_mtp_generate(model, prompt, max_new_tokens=9)
    assert torch.equal(greedy.generated_ids, verified.generated_ids)
    assert verified.stats.corrected_tokens > 0
    assert verified.stats.accepted_auxiliary_tokens == 0

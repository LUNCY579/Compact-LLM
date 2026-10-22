"""Assistant-only causal language-model loss helpers."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def assistant_only_nll(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Return shifted summed-token NLL divided by supervised token count."""

    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError(
            "Expected logits [batch, sequence, vocab] and labels [batch, sequence]"
        )
    if logits.shape[:2] != labels.shape:
        raise ValueError("Logit and label batch/sequence dimensions differ")
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    supervised = shift_labels.ne(-100)
    count = int(supervised.sum().item())
    if count == 0:
        raise RuntimeError("Batch has zero supervised assistant tokens")
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss / count, count


def assert_no_truncation(token_count: int, maximum: int, example_id: str) -> None:
    if token_count > maximum:
        raise RuntimeError(
            f"Example {example_id} requires {token_count} tokens, exceeding {maximum}; "
            "PCSM forbids silent truncation"
        )

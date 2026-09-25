"""Completion-label mechanics shared by future hard and soft arms."""
from __future__ import annotations

import torch


def completion_labels(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: list[int],
    completion_lengths: list[int],
    eos_ids: tuple[int, ...],
    stop_reasons: list[str],
    max_new_tokens: int,
) -> torch.Tensor:
    """Mask only the actual generated suffix, including first EOS.

    Attention positions, not token IDs, determine padding. This is necessary
    when PAD and EOS share an ID, and it preserves left/right padding behavior.
    """

    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input IDs and attention mask shape mismatch")
    if input_ids.dtype != torch.long or attention_mask.dtype != torch.bool:
        raise ValueError("input IDs must be long and attention mask must be bool")
    if len({len(prompt_lengths), len(completion_lengths), len(stop_reasons)}) != 1:
        raise ValueError("batch metadata length mismatch")
    if not eos_ids or len(set(eos_ids)) != len(eos_ids) or any(token < 0 for token in eos_ids):
        raise ValueError("invalid EOS inventory")
    if max_new_tokens <= 0:
        raise ValueError("invalid generation cap")
    labels = torch.full_like(input_ids, -100)
    for row, (prompt, completion, reason) in enumerate(zip(prompt_lengths, completion_lengths, stop_reasons)):
        if prompt <= 0 or completion <= 0 or completion > max_new_tokens:
            raise ValueError("invalid prompt/completion length")
        positions = attention_mask[row].nonzero(as_tuple=True)[0].tolist()
        if len(positions) != prompt + completion:
            raise ValueError("attention span does not match real lengths")
        if positions != list(range(positions[0], positions[-1] + 1)):
            raise ValueError("attention mask contains a hole")
        start = positions[0] + prompt
        generated = input_ids[row, start:start + completion].tolist()
        eos_positions = [index for index, token in enumerate(generated) if token in eos_ids]
        if reason == "eos":
            if eos_positions != [completion - 1]:
                raise ValueError("first EOS must be the final generated token")
        elif reason == "length":
            if completion != max_new_tokens or eos_positions:
                raise ValueError("length stop must be uncapped and contain no EOS")
        else:
            raise ValueError("unknown stop reason")
        labels[row, start:start + completion] = input_ids[row, start:start + completion]
    return labels

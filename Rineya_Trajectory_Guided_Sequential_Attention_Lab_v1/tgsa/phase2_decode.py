"""Exact greedy block verification for MTP proposals.

This is a correctness-first reference decoder, not a fused deployment kernel.
It measures *serial target-model rounds* and proposal acceptance separately from
wall-clock time.  That distinction matters: MTP can reduce serial decoding
rounds, but it does not magically remove the additional per-forward cost of a
sequential TGSA attention operator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass(slots=True)
class DecodeStats:
    generated_tokens: int = 0
    target_forward_calls: int = 0
    proposal_forward_calls: int = 0
    verification_forward_calls: int = 0
    verification_cycles: int = 0
    proposed_auxiliary_tokens: int = 0
    accepted_auxiliary_tokens: int = 0
    corrected_tokens: int = 0
    confidence_truncations: int = 0
    accepted_by_offset: dict[int, int] = field(default_factory=dict)
    proposed_by_offset: dict[int, int] = field(default_factory=dict)

    @property
    def tokens_per_target_forward(self) -> float:
        if self.target_forward_calls == 0:
            return 0.0
        return self.generated_tokens / self.target_forward_calls

    @property
    def auxiliary_acceptance_rate(self) -> float:
        if self.proposed_auxiliary_tokens == 0:
            return 0.0
        return self.accepted_auxiliary_tokens / self.proposed_auxiliary_tokens

    @property
    def serial_round_speedup_vs_ar(self) -> float:
        """Baseline greedy calls divided by verified-MTP target calls.

        Values above one mean fewer serial target-model calls than one-token AR.
        This is an algorithmic latency proxy, not a wall-clock claim.
        """

        return self.tokens_per_target_forward

    @property
    def serial_round_reduction_fraction_vs_ar(self) -> float:
        """Fractional reduction in target-model serial rounds versus greedy AR.

        A value of ``0.25`` means 25% fewer target-model forwards.  The value may
        be negative on short or low-acceptance generations because proposal
        verification can require more calls than plain greedy decoding.
        """

        if self.generated_tokens == 0:
            return 0.0
        return 1.0 - (self.target_forward_calls / self.generated_tokens)

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value.update(
            {
                "tokens_per_target_forward": self.tokens_per_target_forward,
                "auxiliary_acceptance_rate": self.auxiliary_acceptance_rate,
                "serial_round_speedup_vs_ar": self.serial_round_speedup_vs_ar,
                "serial_round_reduction_fraction_vs_ar": (
                    self.serial_round_reduction_fraction_vs_ar
                ),
            }
        )
        value["accepted_by_offset"] = {
            str(key): count for key, count in sorted(self.accepted_by_offset.items())
        }
        value["proposed_by_offset"] = {
            str(key): count for key, count in sorted(self.proposed_by_offset.items())
        }
        return value


@dataclass(slots=True)
class DecodeResult:
    token_ids: Tensor
    generated_ids: Tensor
    stats: DecodeStats


def _call_model(model: Any, input_ids: Tensor) -> Any:
    try:
        return model(input_ids, telemetry_level="none")
    except TypeError:
        return model(input_ids)


def _argmax_last(logits: Tensor) -> int:
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("reference decoding currently requires logits [1, sequence, vocab]")
    return int(logits[0, -1].argmax(dim=-1).item())


def _confidence_last(logits: Tensor, token: int) -> float:
    probabilities = torch.softmax(logits[0, -1].float(), dim=-1)
    return float(probabilities[token].item())


def greedy_generate(
    model: Any,
    prompt_ids: Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> DecodeResult:
    """One-token greedy reference used as the exactness oracle."""

    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("reference decoding currently supports batch size 1")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    tokens = prompt_ids.clone()
    generated: list[int] = []
    stats = DecodeStats()
    training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                output = _call_model(model, tokens)
                stats.target_forward_calls += 1
                token = _argmax_last(output.logits)
                generated.append(token)
                tokens = torch.cat(
                    (tokens, torch.tensor([[token]], device=tokens.device, dtype=tokens.dtype)),
                    dim=1,
                )
                if eos_token_id is not None and token == eos_token_id:
                    break
    finally:
        if training and hasattr(model, "train"):
            model.train()
    stats.generated_tokens = len(generated)
    generated_tensor = torch.tensor([generated], device=tokens.device, dtype=tokens.dtype)
    return DecodeResult(token_ids=tokens, generated_ids=generated_tensor, stats=stats)


def verified_mtp_generate(
    model: Any,
    prompt_ids: Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    min_confidence: float = 0.0,
) -> DecodeResult:
    """Generate greedily while verifying MTP blocks against the main LM head.

    The output is **exactly** the same as ordinary greedy decoding:

    1. offset 1 always comes from the main LM head;
    2. offsets 2..K are proposals only;
    3. one target-model pass over the proposed block verifies each proposal in
       causal order;
    4. at the first mismatch, the main head's token is emitted and the remaining
       proposals are discarded.

    When an entire block is accepted, the verification output is reused as the
    next proposal output.  This is the reference analogue of cache reuse and is
    essential for measuring serial-round reduction honestly.
    """

    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("reference decoding currently supports batch size 1")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")

    tokens = prompt_ids.clone()
    generated: list[int] = []
    stats = DecodeStats()
    cached_output: Any | None = None
    training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.no_grad():
            while len(generated) < max_new_tokens:
                if cached_output is None:
                    proposal_output = _call_model(model, tokens)
                    stats.target_forward_calls += 1
                    stats.proposal_forward_calls += 1
                else:
                    proposal_output = cached_output
                    cached_output = None

                first = _argmax_last(proposal_output.logits)
                remaining = max_new_tokens - len(generated)
                proposals: list[int] = [first]
                proposal_offsets: list[int] = [1]

                if eos_token_id is None or first != eos_token_id:
                    mtp_logits = getattr(proposal_output, "mtp_logits", None)
                    if not isinstance(mtp_logits, dict):
                        raise TypeError("verified MTP decoding requires output.mtp_logits dict")
                    for offset in sorted(mtp_logits):
                        if len(proposals) >= remaining:
                            break
                        token = _argmax_last(mtp_logits[offset])
                        confidence = _confidence_last(mtp_logits[offset], token)
                        if confidence < min_confidence:
                            stats.confidence_truncations += 1
                            break
                        proposals.append(token)
                        proposal_offsets.append(int(offset))
                        stats.proposed_auxiliary_tokens += 1
                        stats.proposed_by_offset[offset] = stats.proposed_by_offset.get(offset, 0) + 1
                        if eos_token_id is not None and token == eos_token_id:
                            break

                if len(proposals) == 1:
                    generated.append(first)
                    tokens = torch.cat(
                        (tokens, torch.tensor([[first]], device=tokens.device, dtype=tokens.dtype)),
                        dim=1,
                    )
                    if eos_token_id is not None and first == eos_token_id:
                        break
                    continue

                stats.verification_cycles += 1
                candidate = torch.cat(
                    (
                        tokens,
                        torch.tensor([proposals], device=tokens.device, dtype=tokens.dtype),
                    ),
                    dim=1,
                )
                verification = _call_model(model, candidate)
                stats.target_forward_calls += 1
                stats.verification_forward_calls += 1

                prefix_length = tokens.shape[1]
                emitted: list[int] = [first]
                all_auxiliary_accepted = True
                for proposal_index in range(1, len(proposals)):
                    prediction_position = prefix_length + proposal_index - 1
                    correct = int(
                        verification.logits[0, prediction_position].argmax(dim=-1).item()
                    )
                    offset = proposal_offsets[proposal_index]
                    if proposals[proposal_index] == correct:
                        emitted.append(correct)
                        stats.accepted_auxiliary_tokens += 1
                        stats.accepted_by_offset[offset] = stats.accepted_by_offset.get(offset, 0) + 1
                    else:
                        emitted.append(correct)
                        stats.corrected_tokens += 1
                        all_auxiliary_accepted = False
                        break

                # EOS can end an otherwise accepted block before the candidate's
                # final token, in which case the cached verification no longer
                # describes the actual prefix and must not be reused.
                eos_cut = None
                if eos_token_id is not None:
                    for index, token in enumerate(emitted):
                        if token == eos_token_id:
                            eos_cut = index + 1
                            break
                if eos_cut is not None:
                    emitted = emitted[:eos_cut]
                    all_auxiliary_accepted = False

                emitted = emitted[: max_new_tokens - len(generated)]
                generated.extend(emitted)
                tokens = torch.cat(
                    (
                        tokens,
                        torch.tensor([emitted], device=tokens.device, dtype=tokens.dtype),
                    ),
                    dim=1,
                )
                if (
                    all_auxiliary_accepted
                    and len(emitted) == len(proposals)
                    and tokens.shape[1] == candidate.shape[1]
                ):
                    cached_output = verification
                if eos_cut is not None:
                    break
    finally:
        if training and hasattr(model, "train"):
            model.train()

    stats.generated_tokens = len(generated)
    generated_tensor = torch.tensor([generated], device=tokens.device, dtype=tokens.dtype)
    return DecodeResult(token_ids=tokens, generated_ids=generated_tensor, stats=stats)


def assert_exact_greedy_equivalence(
    model: Any,
    prompt_ids: Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    min_confidence: float = 0.0,
) -> dict[str, object]:
    baseline = greedy_generate(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
    )
    verified = verified_mtp_generate(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        min_confidence=min_confidence,
    )
    exact = torch.equal(baseline.generated_ids, verified.generated_ids)
    if not exact:
        raise AssertionError(
            "verified MTP decoding diverged from greedy output: "
            f"greedy={baseline.generated_ids.tolist()} verified={verified.generated_ids.tolist()}"
        )
    return {
        "exact": True,
        "generated_ids": verified.generated_ids.tolist(),
        "greedy": baseline.stats.to_dict(),
        "verified_mtp": verified.stats.to_dict(),
    }

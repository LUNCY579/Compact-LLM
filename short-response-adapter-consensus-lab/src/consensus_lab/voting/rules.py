from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


METHODS = {
    "hard_plurality",
    "weighted_hard",
    "probability_mean",
    "calibrated_probability_mean",
    "logit_mean",
    "calibrated_logit_mean",
    "normalized_logit_mean",
    "top2_approval",
    "borda_top5",
    "reciprocal_rank_top5",
    "confidence_weighted_probability",
    "confidence_weighted_logit",
}


@dataclass(frozen=True)
class AggregationDecision:
    token_id: int
    diagnostics: dict[str, Any]


def _vote_entropy(counts: Counter[int], k: int) -> float:
    return -sum((count / k) * math.log(count / k) for count in counts.values() if count)


def _pairwise_js(probabilities: torch.Tensor, log_probabilities: torch.Tensor) -> tuple[float | None, list[list[float | None]]]:
    k = int(probabilities.shape[0])
    matrix: list[list[float | None]] = [[0.0 if i == j else None for j in range(k)] for i in range(k)]
    values: list[float] = []
    for i in range(k):
        for j in range(i + 1, k):
            mean = 0.5 * (probabilities[i] + probabilities[j])
            log_mean = torch.log(mean.clamp_min(torch.finfo(mean.dtype).tiny))
            left_term = torch.where(
                probabilities[i] > 0,
                probabilities[i] * (log_probabilities[i] - log_mean),
                torch.zeros_like(probabilities[i]),
            )
            right_term = torch.where(
                probabilities[j] > 0,
                probabilities[j] * (log_probabilities[j] - log_mean),
                torch.zeros_like(probabilities[j]),
            )
            js = 0.5 * (
                torch.sum(left_term)
                + torch.sum(right_term)
            )
            value = float(js.item())
            matrix[i][j] = value
            matrix[j][i] = value
            values.append(value)
    return (sum(values) / len(values) if values else None), matrix


def _select_hard(
    *,
    top_tokens: list[int],
    log_probabilities: torch.Tensor,
    weights: Sequence[float],
) -> tuple[int, bool, list[int], dict[int, float]]:
    totals: dict[int, float] = {}
    for token, weight in zip(top_tokens, weights, strict=True):
        totals[token] = totals.get(token, 0.0) + float(weight)
    maximum = max(totals.values())
    tied = sorted(token for token, value in totals.items() if math.isclose(value, maximum, rel_tol=1e-12, abs_tol=1e-12))
    if len(tied) == 1:
        return tied[0], False, tied, totals
    tie_scores = {token: float(log_probabilities[:, token].sum().item()) for token in tied}
    best = max(tie_scores.values())
    finalists = sorted(token for token, value in tie_scores.items() if math.isclose(value, best, rel_tol=1e-8, abs_tol=1e-8))
    return finalists[0], True, tied, totals


def _select_primary_then_secondary(
    primary: torch.Tensor,
    secondary: torch.Tensor,
) -> tuple[int, bool, list[int]]:
    """Select a score winner with deterministic confidence and token-ID ties."""
    maximum = torch.max(primary)
    tied_tensor = torch.nonzero(torch.isclose(primary, maximum, rtol=1e-12, atol=1e-12)).flatten()
    tied = [int(value) for value in tied_tensor.tolist()]
    if len(tied) == 1:
        return tied[0], False, tied
    tied_secondary = secondary[tied_tensor]
    best = torch.max(tied_secondary)
    finalists = tied_tensor[torch.isclose(tied_secondary, best, rtol=1e-8, atol=1e-8)]
    return int(torch.min(finalists).item()), True, tied


def _stable_topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Return full-stable-sort top-k indices without sorting the vocabulary.

    ``torch.topk`` identifies the k-th score using a bounded temporary, but its
    ordering of equal values is unspecified.  For each row, order the fewer
    than k values strictly above that threshold with a stable small sort, then
    fill the remaining slots from threshold ties in ascending token-ID order.
    This is exactly the prefix of ``argsort(..., descending=True, stable=True)``
    while avoiding a vocabulary-sized int64 permutation per expert.
    """
    if scores.ndim != 2:
        raise ValueError("Stable top-k requires a rank-2 score tensor")
    vocabulary_size = int(scores.shape[-1])
    if k <= 0 or k > vocabulary_size:
        raise ValueError(f"k must be in [1, {vocabulary_size}], got {k}")
    rows: list[torch.Tensor] = []
    for row in scores:
        threshold = torch.topk(row, k, largest=True, sorted=False).values.min()
        above = torch.nonzero(row > threshold, as_tuple=False).flatten()
        if above.numel():
            # ``nonzero`` supplies ascending token IDs, so a stable score sort
            # retains the former full-sort token-ID tie break.
            order = torch.argsort(row[above], descending=True, stable=True)
            above = above[order]
        remaining = k - int(above.numel())
        boundary = torch.nonzero(row == threshold, as_tuple=False).flatten()[:remaining]
        selected = torch.cat((above, boundary))
        if int(selected.numel()) != k:
            raise RuntimeError("Stable top-k failed to recover the requested rank depth")
        rows.append(selected)
    return torch.stack(rows, dim=0)


def aggregate_logits(
    logits_by_expert: Mapping[str, torch.Tensor],
    *,
    method: str,
    eos_token_id: int | None,
    validation_weights: Mapping[str, float] | None = None,
    temperatures: Mapping[str, float] | None = None,
) -> AggregationDecision:
    if method not in METHODS:
        raise ValueError(f"Unknown aggregation method {method!r}; expected one of {sorted(METHODS)}")
    if not logits_by_expert:
        raise ValueError("At least one expert distribution is required")
    names = list(logits_by_expert)
    logits = torch.stack([logits_by_expert[name].float() for name in names], dim=0)
    if logits.ndim != 2 or len({int(row.numel()) for row in logits}) != 1:
        raise ValueError("Every expert must provide one compatible vocabulary-sized logit vector")
    temperature_values = torch.tensor(
        [float((temperatures or {}).get(name, 1.0)) for name in names], device=logits.device, dtype=logits.dtype
    )
    if bool(torch.any(temperature_values <= 0)):
        raise ValueError("All temperatures must be positive")
    raw_log_probabilities = torch.log_softmax(logits, dim=-1)
    raw_probabilities = torch.exp(raw_log_probabilities)
    calibrated = logits / temperature_values[:, None]
    calibrated_probabilities = torch.softmax(calibrated, dim=-1)
    # ``torch.topk`` does not guarantee stable ordering when values are tied.
    # Greedy generation uses ``argmax`` semantics, selecting the first (and
    # therefore smallest token id) among exact ties. Preserve that behavior
    # for each expert so K=1 is exactly equivalent to ordinary greedy decode.
    top_tensor = torch.argmax(raw_probabilities, dim=-1)
    top_prob_tensor = raw_probabilities.gather(dim=-1, index=top_tensor[:, None]).squeeze(-1)
    probabilities_without_top = raw_probabilities.clone()
    probabilities_without_top.scatter_(dim=-1, index=top_tensor[:, None], value=float("-inf"))
    top2_prob_tensor = torch.max(probabilities_without_top, dim=-1).values
    top_tokens = [int(value) for value in top_tensor.tolist()]
    top_probs = [float(value) for value in top_prob_tensor.tolist()]
    top2_probs = [float(value) for value in top2_prob_tensor.tolist()]
    margins = [a - b for a, b in zip(top_probs, top2_probs, strict=True)]
    counts = Counter(top_tokens)
    hard_weights = [1.0] * len(names)
    if method == "weighted_hard":
        hard_weights = [float((validation_weights or {}).get(name, 1.0)) for name in names]
    tie = False
    tied_tokens: list[int] = []
    hard_totals: dict[int, float] = {}
    rank_score_totals: dict[int, float] = {}
    rank_depth: int | None = None
    if method in {"hard_plurality", "weighted_hard"}:
        winner, tie, tied_tokens, hard_totals = _select_hard(
            top_tokens=top_tokens,
            log_probabilities=raw_log_probabilities,
            weights=hard_weights,
        )
    elif method == "probability_mean":
        winner = int(torch.argmax(raw_probabilities.mean(dim=0)).item())
    elif method == "calibrated_probability_mean":
        winner = int(torch.argmax(calibrated_probabilities.mean(dim=0)).item())
    elif method == "logit_mean":
        winner = int(torch.argmax(logits.mean(dim=0)).item())
    elif method == "calibrated_logit_mean":
        winner = int(torch.argmax(calibrated.mean(dim=0)).item())
    elif method == "normalized_logit_mean":
        centered = logits - logits.mean(dim=-1, keepdim=True)
        scales = logits.std(dim=-1, keepdim=True, unbiased=False).clamp_min(torch.finfo(logits.dtype).eps)
        winner = int(torch.argmax((centered / scales).mean(dim=0)).item())
    elif method in {"top2_approval", "borda_top5", "reciprocal_rank_top5"}:
        rank_depth = min(2 if method == "top2_approval" else 5, int(logits.shape[-1]))
        ranked = _stable_topk_indices(raw_probabilities, rank_depth)
        primary = torch.zeros(int(logits.shape[-1]), device=logits.device, dtype=logits.dtype)
        if method == "top2_approval":
            points = torch.ones(rank_depth, device=logits.device, dtype=logits.dtype)
        elif method == "borda_top5":
            points = torch.arange(rank_depth, 0, -1, device=logits.device, dtype=logits.dtype)
        else:
            points = 1.0 / torch.arange(1, rank_depth + 1, device=logits.device, dtype=logits.dtype)
        for expert_ranking in ranked:
            primary.scatter_add_(0, expert_ranking, points)
        secondary = raw_log_probabilities.sum(dim=0)
        winner, tie, tied_tokens = _select_primary_then_secondary(primary, secondary)
        rank_score_totals = {
            int(token): float(primary[token].item())
            for token in torch.nonzero(primary).flatten().tolist()
        }
    else:
        confidence = torch.tensor(margins, device=logits.device, dtype=logits.dtype).clamp_min(1e-12)
        confidence = confidence / confidence.sum()
        if method == "confidence_weighted_probability":
            winner = int(torch.argmax(torch.sum(confidence[:, None] * raw_probabilities, dim=0)).item())
        else:
            winner = int(torch.argmax(torch.sum(confidence[:, None] * logits, dim=0)).item())
    ordered_counts = sorted(counts.values(), reverse=True)
    winning_votes = counts.get(winner, 0)
    plurality_votes = ordered_counts[0]
    second_votes = ordered_counts[1] if len(ordered_counts) > 1 else 0
    mean_js, js_matrix = _pairwise_js(raw_probabilities, raw_log_probabilities)
    entropy_terms = torch.where(
        raw_probabilities > 0,
        raw_probabilities * raw_log_probabilities,
        torch.zeros_like(raw_probabilities),
    )
    entropies = [float(value) for value in (-entropy_terms.sum(dim=-1)).tolist()]
    eos_votes = counts.get(int(eos_token_id), 0) if eos_token_id is not None else 0
    coalition = [name for name, token in zip(names, top_tokens, strict=True) if token == winner]
    return AggregationDecision(
        token_id=winner,
        diagnostics={
            "expert_order": names,
            "top_tokens": dict(zip(names, top_tokens, strict=True)),
            "top_probabilities": dict(zip(names, top_probs, strict=True)),
            "top2_probabilities": dict(zip(names, top2_probs, strict=True)),
            "top1_top2_margins": dict(zip(names, margins, strict=True)),
            "expert_entropies": dict(zip(names, entropies, strict=True)),
            "vote_counts": {str(token): count for token, count in sorted(counts.items())},
            "winning_votes": winning_votes,
            "plurality_votes": plurality_votes,
            "agreement": plurality_votes / len(names),
            "vote_entropy": _vote_entropy(counts, len(names)),
            "majority_margin": plurality_votes - second_votes,
            "strict_majority": plurality_votes > len(names) / 2,
            "tie": tie,
            "tied_tokens": tied_tokens,
            "tie_break_rule": "sum_log_probability_then_smallest_token_id" if tie else None,
            "weighted_vote_totals": {str(token): value for token, value in sorted(hard_totals.items())},
            "rank_score_totals": {str(token): value for token, value in sorted(rank_score_totals.items())},
            "rank_depth": rank_depth,
            "winning_expert_coalition": coalition,
            "dissenting_experts": [name for name in names if name not in coalition],
            "mean_pairwise_js": mean_js,
            "pairwise_js_matrix": js_matrix,
            "eos_vote_fraction": eos_votes / len(names),
            "winner_is_eos": eos_token_id is not None and winner == eos_token_id,
            "method": method,
        },
    )

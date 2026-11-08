"""SPEC section 6 metrics: accuracy, positional shape, and teacher-forced scores.

Torch is imported lazily inside the two model-facing functions so this module stays
importable on a torch-free interpreter (SPEC section 2). Everything above
``aligned_answer_logprobs`` is pure numpy.

DEVIATION: SPEC section 6 lists ``worst_position(acc_by_pos) -> float`` without saying
whether the float is the accuracy at the worst position or the position itself. It is
implemented as the *worst-case accuracy* (the level you can promise regardless of where
the evidence lands), because that is the quantity the robustness index and the report
tables need; ``worst_position_at`` returns the location for callers that want it.

DEVIATION: SPEC section 6 does not pin the ``position_robustness_index`` formula beyond
its ``eps`` guard. It is ``(min A + eps) / (max A + eps)``: 1.0 for a flat curve,
0 for a curve that collapses to zero somewhere, and 1.0 (flat, uniformly bad) rather
than 0/0 when every position scores zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from rineya_psa.types import (
    BenchmarkItem,
    TokenizedItem,
    UShapeFit,
    matches_gold,
    normalize_answer,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, never executed
    import torch

__all__ = [
    "TeacherForcedScores",
    "accuracy",
    "accuracy_columns",
    "aligned_answer_logprobs",
    "answer_logit_window",
    "answer_margin",
    "contains_gold",
    "edge_middle_gap",
    "first_nonempty_line",
    "gold_nll",
    "position_robustness_index",
    "teacher_forced_scores",
    "ushape_fit",
    "worst_position",
    "worst_position_at",
]


# --------------------------------------------------------------------------------------
# Surface-level scoring
# --------------------------------------------------------------------------------------


def first_nonempty_line(text: str) -> str:
    """First non-blank line, matching ``Rineya_Long_Context_Demo_v1`` scoring."""
    for line in str(text).splitlines():
        if line.strip():
            return line.strip()
    return ""


def contains_gold(
    prediction: str,
    item: BenchmarkItem | Mapping[str, Any],
) -> bool:
    """Does the gold answer appear as a normalised span inside the prediction?

    SPEC section 13.2. The Phase-2b calibration found two failure modes that are the model
    finding the answer and the scorer refusing it: format non-compliance ("The answer is:
    **The loom overseer**." against gold "the loom overseer") and truncation mid-restatement.
    ``strict`` scores both as wrong.

    **This is a CEILING, not a score.** A sufficiently verbose model inflates it - a
    prediction that lists every candidate contains the gold one - so SPEC 13.2 forbids
    accepting an intervention on this column alone. It is reported beside ``strict`` and
    ``first_line`` and never collapsed into a single headline.
    """
    if isinstance(item, Mapping):
        gold = str(item.get("answer", ""))
        aliases = tuple(item.get("answer_aliases", ()) or ())
    else:
        gold = item.answer
        aliases = item.answer_aliases
    haystack = normalize_answer(prediction)
    if not haystack:
        return False
    for candidate in (gold, *aliases):
        needle = normalize_answer(candidate)
        if needle and needle in haystack:
            return True
    return False


def accuracy_columns(
    prediction: str,
    item: BenchmarkItem | Mapping[str, Any],
) -> dict[str, bool]:
    """The three columns SPEC 13.2 mandates, always together."""
    return {
        "strict": accuracy(prediction, item),
        "first_line": accuracy(prediction, item, first_line=True),
        "contains": contains_gold(prediction, item),
    }


def accuracy(
    prediction: str,
    item: BenchmarkItem | Mapping[str, Any],
    *,
    first_line: bool = False,
) -> bool:
    """Normalised exact match against the gold answer and its aliases.

    ``first_line=True`` scores only the first non-blank line, which is the project's
    ``first_line_exact`` convention for models that append a justification. The default
    is strict, as SPEC section 6 requires. No fuzzy matching in either mode.
    """
    if isinstance(item, Mapping):
        gold = str(item.get("answer", ""))
        aliases = tuple(item.get("answer_aliases", ()) or ())
    else:
        gold = item.answer
        aliases = item.answer_aliases
    text = first_nonempty_line(prediction) if first_line else prediction
    return matches_gold(text, gold, aliases)


# --------------------------------------------------------------------------------------
# Positional shape
# --------------------------------------------------------------------------------------


def _as_sorted_curve(acc_by_pos: Mapping[float, float]) -> tuple[np.ndarray, np.ndarray]:
    if not acc_by_pos:
        raise ValueError("acc_by_pos is empty; no positional curve to summarise")
    positions = np.asarray(sorted(float(p) for p in acc_by_pos), dtype=np.float64)
    values = np.asarray([float(acc_by_pos[p]) for p in positions], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("acc_by_pos contains non-finite accuracies")
    return positions, values


def edge_middle_gap(acc_by_pos: Mapping[float, float]) -> float:
    """(A(p_min) + A(p_max)) / 2 - A(p closest to 0.5). Positive means a U shape."""
    positions, values = _as_sorted_curve(acc_by_pos)
    # Ties in |p - 0.5| resolve to the smaller p so the metric is order-independent.
    middle_idx = int(
        min(range(len(positions)), key=lambda i: (abs(positions[i] - 0.5), positions[i]))
    )
    edges = 0.5 * (values[0] + values[-1])
    return float(edges - values[middle_idx])


def worst_position(acc_by_pos: Mapping[float, float]) -> float:
    """Worst-case accuracy over the position grid (see the module DEVIATION note)."""
    _, values = _as_sorted_curve(acc_by_pos)
    return float(values.min())


def worst_position_at(acc_by_pos: Mapping[float, float]) -> float:
    """The normalized position where accuracy is worst; ties resolve to the smaller p."""
    positions, values = _as_sorted_curve(acc_by_pos)
    return float(positions[int(np.argmin(values))])


def position_robustness_index(acc_by_pos: Mapping[float, float], eps: float = 1e-9) -> float:
    """Flatness of the positional curve in [0, 1]; 1.0 is position-invariant.

    Rejects values outside [0, 1]: the ratio is only bounded for accuracies, and feeding it
    a log-probability curve silently returns numbers above 1 instead of failing.
    """
    if eps < 0:
        raise ValueError("eps must be non-negative")
    _, values = _as_sorted_curve(acc_by_pos)
    if values.min() < 0.0 or values.max() > 1.0:
        raise ValueError(
            "position_robustness_index needs accuracies in [0, 1]; got range "
            f"[{values.min():.4g}, {values.max():.4g}]"
        )
    return float((values.min() + eps) / (values.max() + eps))


def ushape_fit(positions: Sequence[float], acc: Sequence[float]) -> UShapeFit:
    """Least squares of accuracy on ``[1, (p - 0.5), (p - 0.5)^2]``.

    Standard errors come from the residual covariance ``sigma^2 (X'X)^-1`` with
    ``sigma^2 = RSS / (n - 3)``, i.e. real OLS standard errors, not a placeholder.
    ``beta2 > 0`` with a beta2 several standard errors from zero is the lost-in-the-
    middle signature.
    """
    p = np.asarray(list(positions), dtype=np.float64)
    y = np.asarray(list(acc), dtype=np.float64)
    if p.ndim != 1 or p.shape != y.shape:
        raise ValueError("positions and acc must be 1-D and the same length")
    n = int(p.size)
    if n < 3:
        raise ValueError(f"a quadratic fit needs at least 3 points, got {n}")
    if not (np.isfinite(p).all() and np.isfinite(y).all()):
        raise ValueError("positions and acc must be finite")

    c = p - 0.5
    x = np.column_stack([np.ones_like(c), c, c * c])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())

    dof = n - x.shape[1]
    if dof > 0:
        sigma2 = rss / dof
        # pinv rather than inv: a degenerate grid (repeated positions) must not raise.
        cov = sigma2 * np.linalg.pinv(x.T @ x)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    else:
        # Saturated fit: the residuals carry no information about the sampling error.
        se = np.full(3, np.nan)

    # A flat accuracy curve has zero total sum of squares; call a perfect flat fit R^2 = 1
    # rather than 0/0.
    r_squared = (1.0 - rss / tss) if tss > 0 else (1.0 if rss <= 1e-24 else 0.0)

    return UShapeFit(
        beta0=float(beta[0]),
        beta1=float(beta[1]),
        beta2=float(beta[2]),
        se0=float(se[0]),
        se1=float(se[1]),
        se2=float(se[2]),
        r_squared=float(r_squared),
        n=n,
    )


# --------------------------------------------------------------------------------------
# Teacher-forced scores
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TeacherForcedScores:
    """One item's teacher-forced summary. ``argmax_ids`` is the gold-prefix greedy path."""

    item_id: str
    gold_nll: float
    answer_margin: float
    token_nll: tuple[float, ...]
    argmax_ids: tuple[int, ...]
    n_answer_tokens: int


def answer_logit_window(total_len: int, n_answer: int) -> tuple[int, int]:
    """Half-open ``[start, stop)`` slice of a length-``total_len`` logit row that scores
    the final ``n_answer`` tokens.

    This is the single place the teacher-forcing off-by-one is written down: ``logits[t]``
    scores token ``t + 1``, so the distribution over the answer token at absolute index
    ``total_len - n_answer + k`` lives at ``total_len - n_answer - 1 + k``.
    """
    if n_answer < 1:
        raise ValueError("n_answer must be at least 1")
    start = total_len - n_answer - 1
    if start < 0:
        raise ValueError(
            f"sequence of {total_len} tokens cannot teacher-force {n_answer} answer tokens; "
            "the model cannot score the very first token of a sequence"
        )
    return start, total_len - 1


def aligned_answer_logprobs(
    logits: torch.Tensor,
    sequence_ids: torch.Tensor,
    answer_start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-probabilities of the answer tokens, with the off-by-one handled once.

    ``logits[t]`` scores the token at ``t + 1``, so the distribution over the answer
    token at absolute index ``answer_start + k`` lives at ``logits[answer_start - 1 + k]``.
    Returns ``(gold_logprobs[n_answer], answer_logprobs[n_answer, vocab])``.

    Both inputs are single-sequence: ``logits`` is ``[T, V]``, ``sequence_ids`` is ``[T]``
    and already contains the answer tokens (teacher forcing).
    """
    import torch

    if logits.ndim != 2 or sequence_ids.ndim != 1:
        raise ValueError(f"expected [T,V] logits and [T] ids, got {tuple(logits.shape)} / "
                         f"{tuple(sequence_ids.shape)}")
    total = int(sequence_ids.shape[0])
    if logits.shape[0] != total:
        raise ValueError(f"logits length {logits.shape[0]} != sequence length {total}")
    if not 1 <= answer_start < total:
        raise ValueError(
            f"answer_start {answer_start} must satisfy 1 <= start < {total}; the model "
            "cannot score the very first token of a sequence"
        )

    start, stop = answer_logit_window(total, total - answer_start)
    lp = torch.log_softmax(logits[start:stop].to(torch.float32), dim=-1)
    gold = sequence_ids[answer_start:total].to(torch.long)
    gold_lp = lp.gather(-1, gold.unsqueeze(-1)).squeeze(-1)
    return gold_lp, lp


def _margin_from_logprobs(lp_row: torch.Tensor, gold_id: int) -> float:
    """logP(gold) - logP(best non-gold) at one position."""
    import torch

    gold_lp = float(lp_row[gold_id])
    masked = lp_row.clone()
    masked[gold_id] = -torch.inf
    return gold_lp - float(masked.max())


def teacher_forced_scores(
    model: Any,
    tok_item: TokenizedItem,
    *,
    device: Any | None = None,
) -> TeacherForcedScores:
    """Single forward over ``input_ids + answer_ids``; both SPEC section 6 scores at once."""
    import torch

    if not tok_item.answer_ids:
        raise ValueError(f"{tok_item.item_id}: no answer tokens to score")
    if not tok_item.input_ids:
        raise ValueError(f"{tok_item.item_id}: empty prompt; nothing conditions the answer")

    if device is None:
        device = next(model.parameters()).device
    seq = torch.tensor(
        list(tok_item.input_ids) + list(tok_item.answer_ids),
        dtype=torch.long,
        device=device,
    )
    answer_start = len(tok_item.input_ids)
    with torch.inference_mode():
        logits = model(input_ids=seq.unsqueeze(0)).logits[0]

    gold_lp, lp = aligned_answer_logprobs(logits, seq, answer_start)
    token_nll = tuple(float(-v) for v in gold_lp)
    margin = _margin_from_logprobs(lp[0], int(tok_item.answer_ids[0]))
    return TeacherForcedScores(
        item_id=tok_item.item_id,
        gold_nll=float(sum(token_nll)),
        answer_margin=float(margin),
        token_nll=token_nll,
        argmax_ids=tuple(int(i) for i in lp.argmax(dim=-1).tolist()),
        n_answer_tokens=len(tok_item.answer_ids),
    )


def gold_nll(model: Any, tok_item: TokenizedItem, *, device: Any | None = None) -> float:
    """Teacher-forced sum of ``-log P(answer_ids | input_ids)`` in a single forward."""
    return teacher_forced_scores(model, tok_item, device=device).gold_nll


def answer_margin(model: Any, tok_item: TokenizedItem, *, device: Any | None = None) -> float:
    """``logP(gold first answer token) - logP(best non-gold token)`` at the first answer slot."""
    return teacher_forced_scores(model, tok_item, device=device).answer_margin


def normalized_answer_set(item: BenchmarkItem) -> frozenset[str]:
    """Accepted normalised surfaces for ``item``; handy for offline re-scoring."""
    targets = {normalize_answer(item.answer)}
    targets.update(normalize_answer(a) for a in item.answer_aliases)
    targets.discard("")
    return frozenset(targets)

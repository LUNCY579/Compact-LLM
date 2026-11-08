"""Activation patching between matched positional variants (brief section 17).

Take one scenario, two runs of it: a **clean** run with the evidence at an edge, where
the model scores the gold answer well, and a **corrupted** run with the same evidence in
the middle, where it does not. Copy activations from the clean run into the corrupted run
at one site and measure what happens to the gold log-probability. A site whose patch
restores the clean score is carrying the information the middle placement loses.

The alignment problem, stated rather than hidden
------------------------------------------------
The two runs do **not** have the evidence at the same token indices -- that is the whole
manipulation. So "patch position ``i`` into position ``i``" is only well defined where
position ``i`` means the same thing in both runs, and for most of the prompt it does not.
Three alignments are implemented, in descending order of how defensible they are:

``final_query`` (exact)
    The last ``n_final_tokens`` prompt tokens. Both runs end with the identical chat
    template generation prompt, both are scored on the identical ``answer_ids``, so the
    same *offset from the end* is the same token in both runs. This is also the position
    the first answer logit is read from, so it is the tightest bottleneck in the model.
    ``alignment_exact=True``. This is the patch to believe.

``span_aligned`` (approximate, documented)
    The clean evidence span mapped onto the corrupted evidence span. Two approximations,
    both recorded on every row:
    1. when the two spans tokenize to different lengths the donor is resampled by
       nearest neighbour, so a receiver token can receive a donor vector from a
       neighbouring token of the same sentence;
    2. the donor vectors were computed at the clean run's absolute positions and
       therefore carry the clean placement's RoPE phase. The patch transplants the
       evidence's *content and its positional signature together*; a positive effect
       cannot be attributed to either alone. ``random_control`` is the negative control
       for the amount-of-substitution part of that, not for the RoPE part.

``all_positions`` (approximate, weak)
    Every prompt position, defined only when the two runs tokenize to the same length.
    Even then, position ``i`` holds a different token in the two runs everywhere the
    evidence moved. Patching every residual position at layer ``l`` replaces essentially
    the entire computation up to layer ``l``, so a near-total recovery is expected *by
    construction* and localises nothing. It is reported because it upper-bounds what any
    patch at that layer could do, and it is flagged ``alignment_exact=False`` so it is
    never mistaken for evidence that layer ``l`` matters.

``random_control`` (negative control)
    The same number of positions as ``span_aligned``, drawn outside every evidence span
    and outside the final-query window, patched at matching absolute indices. It answers
    "does patching *this many* vectors from a better run help regardless of which ones?".

Answer positions are never patched. Under teacher forcing the answer tokens are in the
sequence, and copying the clean run's activations *at* those positions would inject a
representation computed while attending to the clean context directly into the position
whose logits are being scored. Every alignment is restricted to prompt positions, and
information still reaches the answer positions the honest way: through attention.

Direction is always clean -> corrupted, and the reported quantity is

    delta_gold_logprob   = patched - corrupted            (positive = the patch helped)
    recovery_fraction    = (patched - corrupted) / (clean - corrupted)

with ``recovery_fraction`` left NaN when the clean/corrupted gap is too small to divide
by, because a ratio against a near-zero denominator is noise amplification, not a result.

DEVIATION: donor activations are kept in float32, unlike
``collect.ActivationDataset``'s float16 store. Storage there is bulk and lossy-by-budget;
a patch is a transplant and any rounding shows up directly in the measured delta. The
same ``ActivationCollector`` does both -- it takes a ``store_dtype``.

DEVIATION: SPEC 12.1 puts the ``sdpa_kernel`` backend pin in ``evaluation/runner.py``.
``run_items`` cannot host a module-hook patch (it batches, and the patch is per item and
per layer), so this module re-enters the same context manager through
``collect.forced_sdpa_backends``. No behaviour differs; the pin is a performance choice.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rineya_psa.activation_analysis.collect import (
    SITES,
    ActivationCollector,
    forced_sdpa_backends,
    site_hooks,
)
from rineya_psa.evaluation.metrics import answer_logit_window
from rineya_psa.types import BenchmarkItem, TokenizedItem, stable_id

__all__ = [
    "ALIGNMENTS",
    "DEFAULT_ALIGNMENTS",
    "DEFAULT_SITES",
    "Alignment",
    "PatchPair",
    "PatchRecord",
    "build_alignment",
    "build_alignments",
    "find_matched_pairs",
    "gold_logprob",
    "patch_cost_estimate",
    "patch_records_to_dataframe",
    "record_field",
    "run_patch_pair",
    "run_patch_pairs",
    "site_label",
    "write_patch_records",
]

#: Alignments, most defensible first. See the module docstring.
ALIGNMENTS: tuple[str, ...] = (
    "final_query",
    "span_aligned",
    "random_control",
    "all_positions",
)

#: ``all_positions`` is opt-in: it is the weak one and it triples the forwards.
DEFAULT_ALIGNMENTS: tuple[str, ...] = ("final_query", "span_aligned", "random_control")

DEFAULT_SITES: tuple[str, ...] = SITES

#: Below this clean-minus-corrupted gap (nats) ``recovery_fraction`` is not computed.
MIN_RECOVERY_DENOMINATOR: float = 0.05

_SPAN_LABELS: Mapping[str, str] = {
    "final_query": "final_query_representation",
    "span_aligned": "evidence_span_representation",
    "random_control": "random_control_representation",
    "all_positions": "whole_residual_stream",
}
_SITE_LABELS: Mapping[str, str] = {
    "resid": "residual_stream",
    "attn_out": "attention_output",
    "mlp_out": "mlp_output",
}


def site_label(site: str, alignment: str) -> str:
    """The brief's name for a ``(site, alignment)`` pair.

    Brief section 17 lists five sites, but two of them -- "evidence-span representation"
    and "final query representation" -- are the residual stream *at particular positions*,
    i.e. a site crossed with an alignment. This function is where that identification is
    written down instead of being left for the reader to infer from a column name.
    """
    if site == "resid":
        return _SPAN_LABELS.get(alignment, f"residual_stream_{alignment}")
    base = _SITE_LABELS.get(site, site)
    return f"{base}_{alignment}"


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


def _scored_sequence(tok_item: TokenizedItem) -> tuple[list[int], int, int]:
    """``(input_ids + answer_ids, n_prompt, n_answer)`` with the empty cases rejected."""
    if not tok_item.input_ids:
        raise ValueError(f"{tok_item.item_id}: empty prompt; nothing conditions the answer")
    if not tok_item.answer_ids:
        raise ValueError(f"{tok_item.item_id}: no answer tokens to score")
    ids = [int(i) for i in tok_item.input_ids] + [int(i) for i in tok_item.answer_ids]
    return ids, len(tok_item.input_ids), len(tok_item.answer_ids)


def _logprob_from_logits(logits: torch.Tensor, ids: Sequence[int], n_answer: int) -> float:
    """Sum of ``log P(answer token)`` from a possibly ``logits_to_keep``-trimmed tensor.

    ``answer_logit_window`` owns the teacher-forcing off-by-one; the ``offset`` term
    converts its absolute window into the trimmed tensor's coordinates, so this works
    whether or not the model honoured ``logits_to_keep``.
    """
    total = len(ids)
    start, stop = answer_logit_window(total, n_answer)
    offset = total - int(logits.shape[1])
    lp = torch.log_softmax(logits[0, start - offset : stop - offset].float(), dim=-1)
    gold = torch.tensor(list(ids[total - n_answer :]), dtype=torch.long, device=lp.device)
    return float(lp.gather(-1, gold.unsqueeze(-1)).squeeze(-1).sum())


def gold_logprob(
    model: Any,
    tok_item: TokenizedItem,
    *,
    patch_fn: Any | None = None,
    layer: int | None = None,
    site: str | None = None,
) -> float:
    """Teacher-forced ``sum log P(answer | prompt)``, optionally under one patch hook.

    Only ``logits_to_keep = n_answer + 1`` positions of the LM head are materialised
    (SPEC 11.1): the full ``[1, T, 151936]`` tensor is the memory wall, and nothing here
    needs it.
    """
    ids, _, n_answer = _scored_sequence(tok_item)
    device = next(model.parameters()).device
    tensor = torch.tensor(ids, dtype=torch.long, device=device).reshape(1, -1)

    hook_ctx: Any = contextlib.nullcontext()
    if patch_fn is not None:
        if layer is None or site is None:
            raise ValueError("patch_fn requires both layer and site")
        hook_ctx = site_hooks(model, patch_fn, layers=[int(layer)], sites=[str(site)])

    # The log-softmax stays inside ``inference_mode``: reading an inference tensor from
    # outside the block is a documented trap in torch, and the reduction is trivial.
    with hook_ctx, torch.inference_mode(), forced_sdpa_backends(device):
        out = model(input_ids=tensor, use_cache=False, logits_to_keep=n_answer + 1)
        return _logprob_from_logits(out.logits, ids, n_answer)


# --------------------------------------------------------------------------------------
# Pairs
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PatchPair:
    """One scenario's clean (edge, scores well) and corrupted (middle, scores badly) run."""

    scenario_id: str
    clean_item: BenchmarkItem
    clean_tok: TokenizedItem
    corrupted_item: BenchmarkItem
    corrupted_tok: TokenizedItem
    clean_position: float = float("nan")
    corrupted_position: float = float("nan")
    clean_gold_nll: float | None = None
    corrupted_gold_nll: float | None = None
    selection: str = "manual"

    def __post_init__(self) -> None:
        if tuple(self.clean_tok.answer_ids) != tuple(self.corrupted_tok.answer_ids):
            raise ValueError(
                f"{self.scenario_id}: the two variants have different answer tokens "
                f"({len(self.clean_tok.answer_ids)} vs "
                f"{len(self.corrupted_tok.answer_ids)}); they are not the same question "
                "and the log-probabilities are not comparable"
            )

    @property
    def key(self) -> str:
        return f"{self.clean_tok.item_id}->{self.corrupted_tok.item_id}"

    def describe(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "clean_item_id": self.clean_tok.item_id,
            "corrupted_item_id": self.corrupted_tok.item_id,
            "clean_position": self.clean_position,
            "corrupted_position": self.corrupted_position,
            "clean_gold_nll": self.clean_gold_nll,
            "corrupted_gold_nll": self.corrupted_gold_nll,
            "clean_n_tokens": self.clean_tok.n_tokens,
            "corrupted_n_tokens": self.corrupted_tok.n_tokens,
            "selection": self.selection,
        }


def record_field(record: Any, name: str, default: Any = None) -> Any:
    """Read one field from a record that may be a dataclass row or a plain dict."""
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _position_of(record: Any) -> float:
    """Measured evidence position if the tokenizer found the span, else the requested one."""
    for name in ("actual_positions", "target_positions"):
        values = record_field(record, name) or ()
        values = [float(v) for v in values if np.isfinite(float(v))]
        if values:
            return float(np.mean(values))
    return float("nan")


def find_matched_pairs(
    records: Sequence[Any],
    lookup: Mapping[str, tuple[BenchmarkItem, TokenizedItem]],
    *,
    criterion: str = "gold_nll",
    min_nll_gap: float = 0.25,
    edge_max: float = 0.15,
    middle_lo: float = 0.35,
    middle_hi: float = 0.65,
) -> list[PatchPair]:
    """Scenarios where an edge placement beats a middle placement, as patchable pairs.

    ``criterion="gold_nll"`` is the default because SPEC 13.1 measured exact match at the
    floor on this benchmark: selecting pairs on ``correct`` would keep only the handful of
    scenarios where a 0.6B model happens to emit the gold string verbatim, and would throw
    away the graded signal the study actually has. ``criterion="correct"`` implements the
    brief's literal wording ("edge placement succeeds, middle placement fails") for
    callers who want it and is exact when accuracy is not at the floor.

    ``records`` are :class:`rineya_psa.evaluation.runner.RunRecord` (or dicts with the
    same fields); ``lookup`` maps ``item_id`` to the pair the run came from.
    """
    if criterion not in {"gold_nll", "correct"}:
        raise ValueError(f"unknown criterion {criterion!r}; expected 'gold_nll' or 'correct'")
    if not 0.0 <= edge_max < middle_lo <= middle_hi < 1.0 - edge_max:
        raise ValueError(
            "expected 0 <= edge_max < middle_lo <= middle_hi < 1 - edge_max, got "
            f"{edge_max}, {middle_lo}, {middle_hi}"
        )

    by_scenario: dict[str, list[tuple[Any, float]]] = {}
    for record in records:
        scenario = record_field(record, "scenario_id")
        item_id = record_field(record, "item_id")
        if scenario is None or item_id not in lookup:
            continue
        by_scenario.setdefault(str(scenario), []).append((record, _position_of(record)))

    pairs: list[PatchPair] = []
    for scenario, entries in sorted(by_scenario.items()):
        edges = [
            e for e in entries
            if np.isfinite(e[1]) and (e[1] <= edge_max or e[1] >= 1.0 - edge_max)
        ]
        middles = [e for e in entries if np.isfinite(e[1]) and middle_lo <= e[1] <= middle_hi]
        if not edges or not middles:
            continue

        def nll(entry: tuple[Any, float]) -> float:
            value = record_field(entry[0], "gold_nll")
            return float(value) if value is not None else float("nan")

        if criterion == "correct":
            edges = [e for e in edges if bool(record_field(e[0], "correct"))]
            middles = [e for e in middles if record_field(e[0], "correct") is False]
            if not edges or not middles:
                continue
            best_edge = min(edges, key=lambda e: (np.nan_to_num(nll(e), nan=np.inf), e[1]))
            worst_middle = max(middles, key=lambda e: (np.nan_to_num(nll(e), nan=-np.inf), e[1]))
        else:
            edges = [e for e in edges if np.isfinite(nll(e))]
            middles = [e for e in middles if np.isfinite(nll(e))]
            if not edges or not middles:
                continue
            best_edge = min(edges, key=lambda e: (nll(e), e[1]))
            worst_middle = max(middles, key=lambda e: (nll(e), e[1]))
            if nll(worst_middle) - nll(best_edge) < float(min_nll_gap):
                continue

        clean_id = str(record_field(best_edge[0], "item_id"))
        corrupt_id = str(record_field(worst_middle[0], "item_id"))
        if clean_id == corrupt_id:
            continue
        clean_item, clean_tok = lookup[clean_id]
        corrupt_item, corrupt_tok = lookup[corrupt_id]
        if tuple(clean_tok.answer_ids) != tuple(corrupt_tok.answer_ids):
            continue  # different tokenisation of the gold string; not comparable
        pairs.append(
            PatchPair(
                scenario_id=scenario,
                clean_item=clean_item,
                clean_tok=clean_tok,
                corrupted_item=corrupt_item,
                corrupted_tok=corrupt_tok,
                clean_position=float(best_edge[1]),
                corrupted_position=float(worst_middle[1]),
                clean_gold_nll=nll(best_edge) if np.isfinite(nll(best_edge)) else None,
                corrupted_gold_nll=(
                    nll(worst_middle) if np.isfinite(nll(worst_middle)) else None
                ),
                selection=criterion,
            )
        )
    return pairs


# --------------------------------------------------------------------------------------
# Alignments
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Alignment:
    """Which donor index feeds which receiver index, and how honest that mapping is."""

    name: str
    donor_indices: tuple[int, ...]
    receiver_indices: tuple[int, ...]
    exact: bool
    note: str
    status: str = "ok"

    def __post_init__(self) -> None:
        if len(self.donor_indices) != len(self.receiver_indices):
            raise ValueError(
                f"{self.name}: {len(self.donor_indices)} donor indices for "
                f"{len(self.receiver_indices)} receiver indices"
            )

    @property
    def n_positions(self) -> int:
        return len(self.receiver_indices)

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.n_positions > 0


def _skipped_alignment(name: str, reason: str) -> Alignment:
    return Alignment(
        name=name,
        donor_indices=(),
        receiver_indices=(),
        exact=False,
        note="",
        status=f"skipped:{reason}",
    )


def _clip_spans(spans: Sequence[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
    """Spans clipped into ``[0, limit)``; empty and inverted spans dropped."""
    out = []
    for a, b in spans:
        start, end = max(0, int(a)), min(int(limit), int(b))
        if end > start:
            out.append((start, end))
    return out


def _resample(
    donor_span: tuple[int, int], receiver_span: tuple[int, int]
) -> tuple[list[int], list[int]]:
    """Nearest-neighbour map from a donor span onto a receiver span of any length."""
    d_start, d_end = donor_span
    r_start, r_end = receiver_span
    n_recv = r_end - r_start
    n_don = d_end - d_start
    if n_recv <= 0 or n_don <= 0:
        return [], []
    if n_recv == 1:
        return [d_start], [r_start]
    steps = np.linspace(0.0, n_don - 1, n_recv)
    donor = [int(d_start + int(round(float(s)))) for s in steps]
    receiver = list(range(r_start, r_end))
    return donor, receiver


def build_alignment(
    name: str,
    pair: PatchPair,
    *,
    n_final_tokens: int = 1,
    seed: int = 20260828,
) -> Alignment:
    """One named donor->receiver index map over the two runs' *prompt* positions.

    Indices are into the scored sequence ``input_ids + answer_ids``; because answer
    positions are never patched they are also indices into ``input_ids``.
    """
    if name not in ALIGNMENTS:
        raise ValueError(f"unknown alignment {name!r}; known: {list(ALIGNMENTS)}")
    n_don = pair.clean_tok.n_tokens
    n_recv = pair.corrupted_tok.n_tokens
    if n_don < 1 or n_recv < 1:
        return _skipped_alignment(name, "empty_prompt")

    if name == "final_query":
        k = max(1, int(n_final_tokens))
        k = min(k, n_don, n_recv)
        donor = [n_don - j for j in range(k, 0, -1)]
        receiver = [n_recv - j for j in range(k, 0, -1)]
        return Alignment(
            name=name,
            donor_indices=tuple(donor),
            receiver_indices=tuple(receiver),
            exact=True,
            note=(
                "same offset from the end of the prompt; both runs end with the identical "
                "generation prompt and are scored on identical answer_ids"
            ),
        )

    donor_spans = _clip_spans(pair.clean_tok.evidence_spans, n_don)
    receiver_spans = _clip_spans(pair.corrupted_tok.evidence_spans, n_recv)

    if name == "span_aligned":
        if not donor_spans or not receiver_spans:
            return _skipped_alignment(name, "no_evidence_spans")
        n_common = min(len(donor_spans), len(receiver_spans))
        donor: list[int] = []
        receiver: list[int] = []
        resampled = False
        for i in range(n_common):
            d_span, r_span = donor_spans[i], receiver_spans[i]
            resampled |= (d_span[1] - d_span[0]) != (r_span[1] - r_span[0])
            d_idx, r_idx = _resample(d_span, r_span)
            donor.extend(d_idx)
            receiver.extend(r_idx)
        if not receiver:
            return _skipped_alignment(name, "empty_evidence_span")
        note = (
            "clean evidence span mapped onto corrupted evidence span; donor vectors carry "
            "the clean placement's RoPE phase, so content and position move together"
        )
        if resampled:
            note += "; spans differ in length and the donor was resampled nearest-neighbour"
        if len(donor_spans) != len(receiver_spans):
            note += (
                f"; {len(donor_spans)} donor vs {len(receiver_spans)} receiver spans, "
                f"first {n_common} paired"
            )
        return Alignment(
            name=name,
            donor_indices=tuple(donor),
            receiver_indices=tuple(receiver),
            exact=False,
            note=note,
        )

    if name == "random_control":
        if not donor_spans or not receiver_spans:
            return _skipped_alignment(name, "no_evidence_spans")
        span = build_alignment(
            "span_aligned", pair, n_final_tokens=n_final_tokens, seed=seed
        )
        if not span.ok:
            return _skipped_alignment(name, "span_aligned_unavailable")
        blocked = set()
        for a, b in receiver_spans:
            blocked.update(range(a, b))
        blocked.update(range(max(0, n_recv - max(1, int(n_final_tokens))), n_recv))
        usable = min(n_don, n_recv)  # the donor must have the index too
        candidates = np.asarray(
            sorted(set(range(usable)) - blocked), dtype=np.int64
        )
        if candidates.size == 0:
            return _skipped_alignment(name, "no_control_positions")
        k = int(min(span.n_positions, candidates.size))
        rng = np.random.default_rng(
            int(stable_id(seed, "random_control", pair.key), 16) % (2**63)
        )
        picked = np.sort(rng.choice(candidates, size=k, replace=False))
        return Alignment(
            name=name,
            donor_indices=tuple(int(i) for i in picked),
            receiver_indices=tuple(int(i) for i in picked),
            exact=False,
            note=(
                f"negative control for span_aligned: {k} positions outside every evidence "
                "span and outside the final-query window, patched at matching indices"
            ),
        )

    # all_positions
    if n_don != n_recv:
        return _skipped_alignment(name, f"length_mismatch_{n_don}_vs_{n_recv}")
    idx = tuple(range(n_recv))
    return Alignment(
        name=name,
        donor_indices=idx,
        receiver_indices=idx,
        exact=False,
        note=(
            "every prompt position; position i holds a different token in the two runs "
            "wherever the evidence moved, and patching all of them replaces the whole "
            "computation up to this layer, so a large effect is expected by construction "
            "and localises nothing"
        ),
    )


def build_alignments(
    pair: PatchPair,
    names: Sequence[str] = DEFAULT_ALIGNMENTS,
    *,
    n_final_tokens: int = 1,
    seed: int = 20260828,
) -> dict[str, Alignment]:
    return {
        name: build_alignment(name, pair, n_final_tokens=n_final_tokens, seed=seed)
        for name in names
    }


# --------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PatchRecord:
    """One (pair, alignment, layer, site) patch. The row unit of the parquet table."""

    scenario_id: str
    clean_item_id: str
    corrupted_item_id: str
    family: str
    fmt: str
    length_bucket: str
    split: str
    model_key: str | None
    layer: int
    site: str
    alignment: str
    site_label: str
    alignment_exact: bool
    alignment_note: str
    n_patched_positions: int
    clean_position: float
    corrupted_position: float
    clean_n_tokens: int
    corrupted_n_tokens: int
    n_answer_tokens: int
    clean_gold_logprob: float
    corrupted_gold_logprob: float
    patched_gold_logprob: float
    delta_gold_logprob: float
    recovery_fraction: float
    status: str = "ok"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _recovery(patched: float, corrupted: float, clean: float) -> float:
    denominator = clean - corrupted
    if not np.isfinite(denominator) or abs(denominator) < MIN_RECOVERY_DENOMINATOR:
        return float("nan")
    return float((patched - corrupted) / denominator)


def _skipped_record(
    pair: PatchPair,
    alignment: Alignment,
    layer: int,
    site: str,
    *,
    model_key: str | None,
    clean_lp: float,
    corrupted_lp: float,
) -> PatchRecord:
    nan = float("nan")
    return PatchRecord(
        scenario_id=pair.scenario_id,
        clean_item_id=pair.clean_tok.item_id,
        corrupted_item_id=pair.corrupted_tok.item_id,
        family=pair.corrupted_item.family,
        fmt=pair.corrupted_item.fmt,
        length_bucket=pair.corrupted_item.length_bucket,
        split=pair.corrupted_item.split,
        model_key=model_key,
        layer=int(layer),
        site=str(site),
        alignment=alignment.name,
        site_label=site_label(site, alignment.name),
        alignment_exact=alignment.exact,
        alignment_note=alignment.note,
        n_patched_positions=0,
        clean_position=pair.clean_position,
        corrupted_position=pair.corrupted_position,
        clean_n_tokens=pair.clean_tok.n_tokens,
        corrupted_n_tokens=pair.corrupted_tok.n_tokens,
        n_answer_tokens=len(pair.corrupted_tok.answer_ids),
        clean_gold_logprob=clean_lp,
        corrupted_gold_logprob=corrupted_lp,
        patched_gold_logprob=nan,
        delta_gold_logprob=nan,
        recovery_fraction=nan,
        status=alignment.status,
    )


# --------------------------------------------------------------------------------------
# The patch itself
# --------------------------------------------------------------------------------------


def _patch_fn(donor: torch.Tensor, receiver_index: torch.Tensor):
    """A ``site_hooks`` callback that overwrites ``receiver_index`` rows with ``donor``."""

    def fn(layer: int, site: str, tensor: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            raise RuntimeError(
                f"patching runs at batch size 1 on [1, T, hidden]; got {tuple(tensor.shape)}"
            )
        out = tensor.clone()
        idx = receiver_index.to(out.device)
        out[0, idx, :] = donor.to(device=out.device, dtype=out.dtype)
        return out

    return fn


def run_patch_pair(
    model: Any,
    pair: PatchPair,
    *,
    layers: Sequence[int] | None = None,
    sites: Sequence[str] = DEFAULT_SITES,
    alignments: Sequence[str] = DEFAULT_ALIGNMENTS,
    n_final_tokens: int = 1,
    seed: int = 20260828,
    model_key: str | None = None,
    progress: Any | None = None,
) -> list[PatchRecord]:
    """Every ``(alignment, layer, site)`` patch for one pair.

    Cost is ``2 + n_layers * n_sites * n_alignments`` forwards: one to score the corrupted
    run, one to score *and* capture the clean run (the donor capture returns its own
    logits, so the clean score is free), then one per patch.
    """
    site_list = list(dict.fromkeys(str(s) for s in sites))
    unknown = [s for s in site_list if s not in SITES]
    if unknown:
        raise ValueError(f"unknown sites {unknown}; known: {list(SITES)}")

    maps = build_alignments(pair, alignments, n_final_tokens=n_final_tokens, seed=seed)
    _, n_corrupted_prompt, n_answer = _scored_sequence(pair.corrupted_tok)
    corrupted_lp = gold_logprob(model, pair.corrupted_tok)

    donor_indices = sorted({int(i) for m in maps.values() if m.ok for i in m.donor_indices})
    if not donor_indices:
        donor_indices = [0]  # still need the clean score; capture one throwaway position
    clean_ids, n_clean_prompt, _ = _scored_sequence(pair.clean_tok)

    with ActivationCollector(
        model, layers=layers, sites=site_list, store_dtype=np.float32
    ) as collector:
        donor_blocks = collector.run(
            clean_ids, donor_indices, logits_to_keep=n_answer + 1
        )
        with torch.inference_mode():
            clean_lp = _logprob_from_logits(collector.last_logits, clean_ids, n_answer)
        used_layers = collector.layers

    row_of = {index: row for row, index in enumerate(donor_indices)}
    device = next(model.parameters()).device

    records: list[PatchRecord] = []
    for name in alignments:
        alignment = maps[name]
        for layer in used_layers:
            for site in site_list:
                if not alignment.ok:
                    records.append(
                        _skipped_record(
                            pair, alignment, layer, site,
                            model_key=model_key, clean_lp=clean_lp,
                            corrupted_lp=corrupted_lp,
                        )
                    )
                    continue
                rows = [row_of[int(i)] for i in alignment.donor_indices]
                donor = torch.as_tensor(
                    donor_blocks[(int(layer), site)][rows], dtype=torch.float32
                ).to(device)
                receiver = torch.tensor(
                    [int(i) for i in alignment.receiver_indices],
                    dtype=torch.long,
                    device=device,
                )
                patched_lp = gold_logprob(
                    model,
                    pair.corrupted_tok,
                    patch_fn=_patch_fn(donor, receiver),
                    layer=layer,
                    site=site,
                )
                records.append(
                    PatchRecord(
                        scenario_id=pair.scenario_id,
                        clean_item_id=pair.clean_tok.item_id,
                        corrupted_item_id=pair.corrupted_tok.item_id,
                        family=pair.corrupted_item.family,
                        fmt=pair.corrupted_item.fmt,
                        length_bucket=pair.corrupted_item.length_bucket,
                        split=pair.corrupted_item.split,
                        model_key=model_key,
                        layer=int(layer),
                        site=site,
                        alignment=name,
                        site_label=site_label(site, name),
                        alignment_exact=alignment.exact,
                        alignment_note=alignment.note,
                        n_patched_positions=alignment.n_positions,
                        clean_position=pair.clean_position,
                        corrupted_position=pair.corrupted_position,
                        clean_n_tokens=n_clean_prompt,
                        corrupted_n_tokens=n_corrupted_prompt,
                        n_answer_tokens=n_answer,
                        clean_gold_logprob=clean_lp,
                        corrupted_gold_logprob=corrupted_lp,
                        patched_gold_logprob=patched_lp,
                        delta_gold_logprob=float(patched_lp - corrupted_lp),
                        recovery_fraction=_recovery(patched_lp, corrupted_lp, clean_lp),
                        status="ok",
                    )
                )
                if progress is not None:
                    progress(1)
    return records


def run_patch_pairs(
    model: Any,
    pairs: Sequence[PatchPair],
    **kwargs: Any,
) -> list[PatchRecord]:
    """:func:`run_patch_pair` over many pairs, concatenated."""
    out: list[PatchRecord] = []
    for pair in pairs:
        out.extend(run_patch_pair(model, pair, **kwargs))
    return out


# --------------------------------------------------------------------------------------
# Cost, serialisation
# --------------------------------------------------------------------------------------


def patch_cost_estimate(
    *,
    n_pairs: int,
    n_layers: int,
    n_sites: int = len(DEFAULT_SITES),
    n_alignments: int = len(DEFAULT_ALIGNMENTS),
    seq_len: int = 4096,
) -> dict[str, Any]:
    """Forwards and wall clock this sweep will cost (SPEC 11.3's gate).

    The per-forward second count comes from ``significance.causal.cost_estimate`` rather
    than from a second copy of SPEC 12.1's measured table, so the two estimators cannot
    drift apart.
    """
    from rineya_psa.significance.causal import GPU_HOUR_BUDGET, cost_estimate

    reference = cost_estimate(
        n_items=1, n_layers=1, n_kv_heads=1, n_bins=1, seq_len=int(seq_len), tier="ablation"
    )
    per_forward = float(reference["seconds_per_forward"])
    per_pair = 2 + int(n_layers) * int(n_sites) * int(n_alignments)
    n_forwards = int(n_pairs) * per_pair
    seconds = n_forwards * per_forward
    gpu_hours = seconds / 3600.0
    return {
        "n_pairs": int(n_pairs),
        "forwards_per_pair": per_pair,
        "n_forwards": n_forwards,
        "seq_len": int(seq_len),
        "seconds_per_forward": per_forward,
        "estimated_seconds": seconds,
        "estimated_gpu_hours": gpu_hours,
        "gpu_hour_budget": GPU_HOUR_BUDGET,
        "exceeds_budget": gpu_hours > GPU_HOUR_BUDGET,
        "warning": (
            "over the 2 GPU-hour budget of SPEC 11.3; pass --yes-i-costed-this"
            if gpu_hours > GPU_HOUR_BUDGET
            else None
        ),
    }


def patch_records_to_dataframe(records: Sequence[PatchRecord]) -> Any:
    """Tidy long format, one row per (pair, alignment, layer, site)."""
    import pandas as pd

    if not records:
        return pd.DataFrame(columns=list(PatchRecord.__dataclass_fields__))
    return pd.DataFrame([r.to_json() for r in records])


def write_patch_records(
    records: Sequence[PatchRecord],
    path: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
) -> Path:
    """Write the long table plus its ``*.provenance.json`` sidecar (SPEC section 9)."""
    from rineya_psa.statistics.provenance import write_result

    return write_result(patch_records_to_dataframe(records), path, config=config)

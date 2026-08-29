"""Batched inference driver. Every phase that touches a model goes through ``run_items``.

Design notes a reader would otherwise have to reverse-engineer:

* **torch is imported inside functions.** SPEC section 2 keeps ``evaluation/`` importable on
  a torch-free interpreter so CPU analysis and most tests start instantly.
* **Batches are capped by total tokens, not rows.** The 8.59 GB card cannot hold a
  ``[B, T, vocab]`` logit tensor at T = 8192 for B > 1 (151936 * 8192 * 4 B ~= 5 GB on its
  own), so ``batch_tokens`` defaults to 8192: one 8K item per batch, sixteen 512-token
  items per batch. Items are bucketed by length first so padding waste stays low.
* **Teacher forcing only materialises the logits it needs.** ``logits_to_keep`` trims the
  forward's LM head to the answer window, which is what makes long items affordable.
* **Left padding with explicit ``position_ids``.** ``(attention_mask.cumsum(-1) - 1)``
  clamped at zero reproduces the unpadded RoPE offsets exactly; a padded row's logits match
  the same row run alone to float32 round-off (``tests/test_metrics.py``).
* **PSA hooks are imported lazily**, so this module stays importable before
  ``rineya_psa.hooks.registry`` exists and on machines with no GPU.

DEVIATION: ``latency_s`` is per-batch wall time divided by the rows in that batch. There is
no per-row timing inside a batched forward, and reporting the whole batch's time on every
row would inflate the total by the batch size. ``n_batch_rows`` is recorded alongside so the
amortisation is undoable.

DEVIATION: ``peak_vram_bytes`` is ``torch.cuda.max_memory_allocated`` for the batch the row
belonged to, reset per batch, and is 0 on CPU. It is a batch-level, not row-level, figure.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rineya_psa.backend import efficient_attention
from rineya_psa.evaluation.metrics import accuracy, answer_logit_window
from rineya_psa.paths import assert_readonly_target
from rineya_psa.types import BenchmarkItem, TokenizedItem

__all__ = [
    "DEFAULT_BATCH_TOKENS",
    "DEFAULT_RECORD",
    "RECORD_FIELDS",
    "RunRecord",
    "generate_batch",
    "plan_batches",
    "prepare_items",
    "render_messages",
    "render_prompt",
    "run_items",
    "to_dataframe",
    "tokenize_item",
    "write_parquet",
]

# One 8K item per batch; sixteen 512-token items per batch. See the module docstring.
DEFAULT_BATCH_TOKENS = 8192
DEFAULT_MAX_BATCH_ROWS = 32

RECORD_FIELDS: tuple[str, ...] = (
    "correct",
    "prediction",
    "gold_nll",
    "answer_margin",
    "latency_s",
    "peak_vram_bytes",
)
DEFAULT_RECORD: tuple[str, ...] = RECORD_FIELDS

MODES: tuple[str, ...] = ("teacher_forced", "greedy")


@dataclass(frozen=True)
class RunRecord:
    """One item's outcome under one model/context. The unit of every downstream table."""

    item_id: str
    scenario_id: str
    family: str
    fmt: str
    length_bucket: str
    split: str
    target_positions: tuple[float, ...]
    actual_positions: tuple[float, ...]
    correct: bool | None
    prediction: str | None
    gold_nll: float | None
    answer_margin: float | None
    n_tokens: int
    latency_s: float | None
    peak_vram_bytes: int | None
    mode: str = "teacher_forced"
    n_batch_rows: int = 1

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["target_positions"] = list(self.target_positions)
        d["actual_positions"] = list(self.actual_positions)
        return d


# --------------------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------------------


def _messages_for(item: BenchmarkItem) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if item.system:
        messages.append({"role": "system", "content": item.system})
    messages.append({"role": "user", "content": item.user})
    return messages


def render_prompt(tok: Any, item: BenchmarkItem, *, enable_thinking: bool = False) -> str:
    """Chat-template render of one benchmark item."""
    return render_messages(tok, _messages_for(item), enable_thinking=enable_thinking)


def render_messages(
    tok: Any, messages: Sequence[Mapping[str, str]], *, enable_thinking: bool = False
) -> str:
    """Chat-template render, thinking disabled so the token budget is not eaten by <think>."""
    messages = [{"role": str(m["role"]), "content": str(m["content"])} for m in messages]
    if getattr(tok, "chat_template", None) is None:
        parts = [f"{m['role'].upper()}\n{m['content']}" for m in messages]
        return "\n\n".join(parts) + "\n\nASSISTANT\n"
    try:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # Templates without an ``enable_thinking`` variable reject the kwarg outright.
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _evidence_spans(
    tok: Any, rendered: str, evidence_sentences: Sequence[str], n_tokens: int
) -> tuple[tuple[int, int], ...]:
    """Token spans of the evidence sentences, via the fast tokenizer's offset mapping.

    Returns an empty tuple when offsets are unavailable (slow tokenizer) or a sentence is
    not found verbatim in the rendered prompt; the caller then falls back to the requested
    positions rather than inventing a measurement.
    """
    if not evidence_sentences:
        return ()
    try:
        encoding = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
        offsets = list(encoding["offset_mapping"])
    except (NotImplementedError, TypeError, KeyError):
        return ()
    if len(offsets) != n_tokens:
        return ()

    spans: list[tuple[int, int]] = []
    for sentence in evidence_sentences:
        char_start = rendered.find(sentence)
        if char_start < 0:
            continue
        char_end = char_start + len(sentence)
        tok_start = next(
            (i for i, (a, b) in enumerate(offsets) if b > char_start and a < char_end), None
        )
        if tok_start is None:
            continue
        tok_end = tok_start
        for i, (a, b) in enumerate(offsets):
            if a < char_end and b > char_start:
                tok_end = i + 1
        spans.append((tok_start, tok_end))
    return tuple(spans)


def tokenize_item(
    tok: Any, item: BenchmarkItem, *, enable_thinking: bool = False
) -> TokenizedItem:
    """Render, tokenize, and measure where the evidence actually landed.

    ``actual_positions`` is the evidence-span midpoint divided by the prompt length. When
    the spans cannot be located the requested ``target_positions`` are echoed back, so a
    caller can always compare requested against actual and see when they are identical.
    """
    if not isinstance(item, BenchmarkItem):
        # See the mirror-image guard in ``benchmark.tokenize.tokenize_item``: the two
        # helpers share a name and take their arguments in opposite orders.
        raise TypeError(
            "evaluation.runner.tokenize_item(tok, item); you may want "
            "benchmark.tokenize.tokenize_item(item, tokenizer), which takes them the "
            f"other way round. Got {type(item).__name__} as the second argument."
        )
    rendered = render_prompt(tok, item, enable_thinking=enable_thinking)
    input_ids = tuple(int(i) for i in tok.encode(rendered, add_special_tokens=False))
    answer_ids = tuple(int(i) for i in tok.encode(item.answer, add_special_tokens=False))
    n = len(input_ids)
    spans = _evidence_spans(tok, rendered, item.evidence_sentences, n)
    if spans and n:
        actual = tuple(((a + b) / 2.0) / n for a, b in spans)
    else:
        actual = tuple(item.target_positions)
    return TokenizedItem(
        item_id=item.item_id,
        input_ids=input_ids,
        answer_ids=answer_ids,
        evidence_spans=spans,
        actual_positions=actual,
    )


def prepare_items(
    tok: Any,
    items: Iterable[BenchmarkItem | tuple[BenchmarkItem, TokenizedItem]],
    *,
    enable_thinking: bool = False,
) -> list[tuple[BenchmarkItem, TokenizedItem]]:
    """Normalise the caller's input to ``(BenchmarkItem, TokenizedItem)`` pairs."""
    pairs: list[tuple[BenchmarkItem, TokenizedItem]] = []
    for entry in items:
        if isinstance(entry, BenchmarkItem):
            pairs.append((entry, tokenize_item(tok, entry, enable_thinking=enable_thinking)))
            continue
        if isinstance(entry, tuple) and len(entry) == 2:
            item, tok_item = entry
            if isinstance(item, BenchmarkItem) and isinstance(tok_item, TokenizedItem):
                pairs.append((item, tok_item))
                continue
        raise TypeError(
            "run_items accepts BenchmarkItem or (BenchmarkItem, TokenizedItem) pairs; "
            f"got {type(entry).__name__}"
        )
    return pairs


# --------------------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------------------


def _row_length(tok_item: TokenizedItem, mode: str, max_new_tokens: int) -> int:
    if mode == "teacher_forced":
        return len(tok_item.input_ids) + max(1, len(tok_item.answer_ids))
    return len(tok_item.input_ids) + max_new_tokens


def plan_batches(
    lengths: Sequence[int], *, batch_tokens: int, max_rows: int = DEFAULT_MAX_BATCH_ROWS
) -> list[list[int]]:
    """Length-bucketed batches whose padded token area stays under ``batch_tokens``.

    Sorting by length before packing is what keeps padding waste low: neighbouring items in
    a batch differ by a few tokens rather than by kilotokens. A single item longer than
    ``batch_tokens`` still gets its own batch instead of being dropped.
    """
    if batch_tokens < 1:
        raise ValueError("batch_tokens must be positive")
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for idx in order:
        candidate_max = max(current_max, lengths[idx])
        area = (len(current) + 1) * candidate_max
        if current and (area > batch_tokens or len(current) >= max_rows):
            batches.append(current)
            current, current_max = [idx], lengths[idx]
        else:
            current.append(idx)
            current_max = candidate_max
    if current:
        batches.append(current)
    return batches


def _pad_left(sequences: Sequence[Sequence[int]], pad_id: int, device: Any) -> tuple[Any, Any, Any]:
    import torch

    width = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), width), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, seq in enumerate(sequences):
        ids[row, width - len(seq) :] = torch.tensor(list(seq), dtype=torch.long)
        mask[row, width - len(seq) :] = 1
    position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
    return ids.to(device), mask.to(device), position_ids.to(device)


# --------------------------------------------------------------------------------------
# Forward passes
# --------------------------------------------------------------------------------------


def _teacher_forced_batch(
    model: Any,
    tok_items: Sequence[TokenizedItem],
    *,
    pad_id: int,
    device: Any,
    want_scores: bool,
    want_prediction: bool,
) -> list[dict[str, Any]]:
    """One padded forward; per-row gold NLL, first-token margin, and gold-prefix argmax."""
    import torch

    empty = [t.item_id for t in tok_items if not t.answer_ids]
    if empty:
        raise ValueError(f"teacher forcing needs answer tokens; empty for {empty[:5]}")
    sequences = [list(t.input_ids) + list(t.answer_ids) for t in tok_items]
    n_answers = [len(t.answer_ids) for t in tok_items]
    ids, mask, position_ids = _pad_left(sequences, pad_id, device)
    width = int(ids.shape[1])
    keep = min(width, max(n_answers) + 1)

    with torch.inference_mode():
        logits = model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=position_ids,
            use_cache=False,
            logits_to_keep=keep,
        ).logits
    offset = width - int(logits.shape[1])

    out: list[dict[str, Any]] = []
    for row, tok_item in enumerate(tok_items):
        n_ans = n_answers[row]
        start, stop = answer_logit_window(width, n_ans)
        lp = torch.log_softmax(logits[row, start - offset : stop - offset].float(), dim=-1)
        row_out: dict[str, Any] = {}
        if want_prediction:
            row_out["argmax_ids"] = [int(i) for i in lp.argmax(dim=-1).tolist()]
        if want_scores:
            gold = torch.tensor(list(tok_item.answer_ids), dtype=torch.long, device=lp.device)
            gold_lp = lp.gather(-1, gold.unsqueeze(-1)).squeeze(-1)
            row_out["gold_nll"] = float(-gold_lp.sum())
            first = lp[0].clone()
            gold_id = int(tok_item.answer_ids[0])
            gold_value = float(first[gold_id])
            first[gold_id] = -torch.inf
            row_out["answer_margin"] = gold_value - float(first.max())
        out.append(row_out)
    return out


def _strip_generation(text: str) -> str:
    """Drop a leaked ``<think>`` block and everything after the first newline."""
    body = str(text)
    if "</think>" in body:
        body = body.split("</think>", 1)[1]
    body = body.replace("<think>", "").strip()
    return body.split("\n", 1)[0].strip()


def generate_batch(
    model: Any,
    tok: Any,
    tok_items: Sequence[TokenizedItem],
    *,
    pad_id: int,
    device: Any,
    max_new_tokens: int,
) -> list[str]:
    """Greedy continuation for one left-padded batch, template and newline stripped."""
    import torch
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    ids, mask, _ = _pad_left([list(t.input_ids) for t in tok_items], pad_id, device)
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": pad_id,
    }
    eos = getattr(tok, "eos_token_id", None)
    if eos is not None:
        kwargs["eos_token_id"] = eos
    if isinstance(tok, PreTrainedTokenizerBase):
        # Real tokenizers can stop the batch early on a newline; the fake tokenizers used
        # in tests cannot, and post-truncation covers them.
        kwargs["stop_strings"] = ["\n"]
        kwargs["tokenizer"] = tok

    with torch.inference_mode():
        generated = model.generate(input_ids=ids, attention_mask=mask, **kwargs)
    prompt_width = int(ids.shape[1])
    continuations = generated[:, prompt_width:]
    return [
        _strip_generation(tok.decode(row, skip_special_tokens=True)) for row in continuations
    ]


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def _psa_context(model: Any, ctx: Any) -> Any:
    """Enter ``hooks.registry.psa_session`` only when a context was supplied."""
    if ctx is None:
        return contextlib.nullcontext()
    from rineya_psa.hooks.registry import psa_session  # lazy: no GPU/torch at import time

    return psa_session(model, ctx)


def run_items(
    model: Any,
    tok: Any,
    items: Iterable[BenchmarkItem | tuple[BenchmarkItem, TokenizedItem]],
    mode: str = "teacher_forced",
    ctx: Any | None = None,
    *,
    batch_tokens: int = DEFAULT_BATCH_TOKENS,
    max_new_tokens: int = 16,
    max_batch_rows: int = DEFAULT_MAX_BATCH_ROWS,
    record: Sequence[str] = DEFAULT_RECORD,
    enable_thinking: bool = False,
    progress: Any | None = None,
) -> list[RunRecord]:
    """Run ``items`` through ``model`` and return one :class:`RunRecord` per item.

    ``mode="teacher_forced"`` scores the gold continuation and reports the gold-prefix
    argmax as the prediction. ``mode="greedy"`` generates (``do_sample=False``, stopped at
    a newline or EOS, template stripped) and additionally teacher-forces when ``gold_nll``
    or ``answer_margin`` are requested in ``record``.

    Output order matches input order even though execution order is length-sorted.
    """
    import torch

    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    requested = tuple(record)
    unknown = set(requested) - set(RECORD_FIELDS)
    if unknown:
        raise ValueError(f"unknown record fields {sorted(unknown)}; known: {list(RECORD_FIELDS)}")

    pairs = prepare_items(tok, items, enable_thinking=enable_thinking)
    if not pairs:
        return []

    pad_id = getattr(tok, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tok, "eos_token_id", None)
    if pad_id is None:
        pad_id = 0
    if isinstance(pad_id, (list, tuple)):
        pad_id = pad_id[0]

    device = next(model.parameters()).device
    on_cuda = torch.cuda.is_available() and getattr(device, "type", str(device)) == "cuda"

    want_scores = "gold_nll" in requested or "answer_margin" in requested
    want_prediction = "prediction" in requested or "correct" in requested

    lengths = [_row_length(t, mode, max_new_tokens) for _, t in pairs]
    batches = plan_batches(lengths, batch_tokens=batch_tokens, max_rows=max_batch_rows)

    results: list[RunRecord | None] = [None] * len(pairs)
    # SPEC 12.1: the SDPA dispatcher picks the MATH backend on this machine (no
    # FlashAttention in the Windows wheel), which is 8x slower and 3x heavier and makes
    # 8K contexts look impossible. Pinning here means every caller gets it, rather than
    # each module remembering to.
    with efficient_attention(active=on_cuda), _psa_context(model, ctx):
        for batch in batches:
            tok_items = [pairs[i][1] for i in batch]
            if on_cuda:
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()

            predictions: list[str] | None = None
            scored: list[dict[str, Any]] = [{} for _ in batch]
            if mode == "greedy":
                if want_prediction:
                    predictions = generate_batch(
                        model,
                        tok,
                        tok_items,
                        pad_id=pad_id,
                        device=device,
                        max_new_tokens=max_new_tokens,
                    )
                if want_scores:
                    scored = _teacher_forced_batch(
                        model,
                        tok_items,
                        pad_id=pad_id,
                        device=device,
                        want_scores=True,
                        want_prediction=False,
                    )
            else:
                scored = _teacher_forced_batch(
                    model,
                    tok_items,
                    pad_id=pad_id,
                    device=device,
                    want_scores=want_scores,
                    want_prediction=want_prediction,
                )
                if want_prediction:
                    predictions = [
                        _strip_generation(tok.decode(s["argmax_ids"], skip_special_tokens=True))
                        for s in scored
                    ]

            if on_cuda:
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            per_row_latency = elapsed / len(batch)
            peak = int(torch.cuda.max_memory_allocated(device)) if on_cuda else 0

            for slot, index in enumerate(batch):
                item, tok_item = pairs[index]
                prediction = predictions[slot] if predictions is not None else None
                results[index] = RunRecord(
                    item_id=item.item_id,
                    scenario_id=item.scenario_id,
                    family=item.family,
                    fmt=item.fmt,
                    length_bucket=item.length_bucket,
                    split=item.split,
                    target_positions=tuple(item.target_positions),
                    actual_positions=tuple(tok_item.actual_positions),
                    correct=(
                        accuracy(prediction, item)
                        if ("correct" in requested and prediction is not None)
                        else None
                    ),
                    prediction=prediction if "prediction" in requested else None,
                    gold_nll=scored[slot].get("gold_nll") if "gold_nll" in requested else None,
                    answer_margin=(
                        scored[slot].get("answer_margin") if "answer_margin" in requested else None
                    ),
                    n_tokens=tok_item.n_tokens,
                    latency_s=per_row_latency if "latency_s" in requested else None,
                    peak_vram_bytes=peak if "peak_vram_bytes" in requested else None,
                    mode=mode,
                    n_batch_rows=len(batch),
                )
            if progress is not None:
                progress(len(batch))

    missing = [i for i, r in enumerate(results) if r is None]
    if missing:  # pragma: no cover - defensive; plan_batches is a partition
        raise RuntimeError(f"{len(missing)} items were never scheduled: {missing[:5]}")
    return [r for r in results if r is not None]


# --------------------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------------------


def to_dataframe(records: Sequence[RunRecord | Mapping[str, Any]]) -> Any:
    """Records as a pandas DataFrame; positional tuples become lists (parquet-friendly)."""
    import pandas as pd

    rows = [r.to_json() if isinstance(r, RunRecord) else dict(r) for r in records]
    if not rows:
        return pd.DataFrame(columns=[f.name for f in RunRecord.__dataclass_fields__.values()])
    return pd.DataFrame(rows)


def write_parquet(records: Sequence[RunRecord | Mapping[str, Any]], path: str | Path) -> Path:
    """Write records to parquet, refusing any path inside a source checkpoint."""
    target = Path(path)
    assert_readonly_target(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = to_dataframe(records)
    frame.to_parquet(target, index=False)
    return target

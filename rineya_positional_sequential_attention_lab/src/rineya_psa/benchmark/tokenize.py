"""Chat-template application and evidence localisation in token space.

The measured token midpoint produced here -- not the requested character target -- is
what every downstream analysis uses as the item's position. Character targets only
steer the search in :mod:`rineya_psa.benchmark.assemble`.

Offset availability, verified on this environment (transformers 5.14.1): the Qwen3
tokenizer resolves to a **fast** ``Qwen2Tokenizer``, and
``__call__(..., return_offsets_mapping=True)`` returns real character spans, including
for the chat-template control tokens. The incremental-encoding fallback in
``_span_incremental`` is therefore dead on this machine but kept because the brief
asks for a verified answer rather than an assumption, and because a slow tokenizer
would otherwise raise instead of degrading.

Torch is not imported at module scope: the loader is pulled in lazily so the rest of
the benchmark package stays importable on a torch-free interpreter (SPEC section 2).
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from typing import Any

from rineya_psa.benchmark.assemble import SYSTEM_PROMPT
from rineya_psa.types import BenchmarkItem, TokenizedItem

_TOKENIZERS: dict[str, Any] = {}


def get_tokenizer(key: str = "qwen3") -> Any:
    """Cached tokenizer for a panel key.

    Prefers ``rineya_psa.loading.load_tokenizer`` so the panel's padding policy is
    shared, but that module imports torch at import time; if torch is unavailable we
    fall back to a direct ``AutoTokenizer`` load from the same path.
    """
    if key in _TOKENIZERS:
        return _TOKENIZERS[key]
    try:
        from rineya_psa.loading import load_tokenizer  # noqa: PLC0415 - deliberate

        tok = load_tokenizer(key)
    except Exception:  # noqa: BLE001 - torch-free interpreters land here
        from transformers import AutoTokenizer  # noqa: PLC0415 - deliberate

        from rineya_psa.paths import model_spec

        spec = model_spec(key)
        tok = AutoTokenizer.from_pretrained(
            str(spec.path), trust_remote_code=spec.trust_remote_code
        )
    _TOKENIZERS[key] = tok
    return tok


def offsets_available(tokenizer: Any) -> bool:
    """Whether ``return_offsets_mapping`` works for this tokenizer."""
    if not getattr(tokenizer, "is_fast", False):
        return False
    try:
        enc = tokenizer("probe text", return_offsets_mapping=True, add_special_tokens=False)
    except Exception:  # noqa: BLE001 - slow tokenizers raise NotImplementedError
        return False
    offsets = enc.get("offset_mapping")
    return bool(offsets) and any(e > s for s, e in offsets)


def build_chat_prompt(system: str, user: str, tokenizer: Any | None = None) -> str:
    """Apply the chat template with thinking disabled and a generation prompt appended."""
    tok = tokenizer if tokenizer is not None else get_tokenizer()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:  # a template without the Qwen3 thinking switch
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def encode(text: str, tokenizer: Any | None = None) -> list[int]:
    tok = tokenizer if tokenizer is not None else get_tokenizer()
    return list(tok(text, add_special_tokens=False)["input_ids"])


def count_tokens(text: str, tokenizer: Any | None = None) -> int:
    return len(encode(text, tokenizer))


def count_prompt_tokens(
    system: str, user: str, tokenizer: Any | None = None
) -> int:
    """Token length of the full templated prompt. This is the benchmark's length budget."""
    return count_tokens(build_chat_prompt(system, user, tokenizer), tokenizer)


def prompt_token_counter(tokenizer: Any | None = None):
    """A ``(system, user) -> int`` closure for ``assemble.fit_bank``."""
    tok = tokenizer if tokenizer is not None else get_tokenizer()

    def _count(system: str, user: str) -> int:
        return count_prompt_tokens(system, user, tok)

    return _count


# ---------------------------------------------------------------------------
# Span location
# ---------------------------------------------------------------------------


def _spans_from_offsets(
    offsets: Sequence[tuple[int, int]], char_spans: Sequence[tuple[int, int]]
) -> tuple[tuple[int, int], ...]:
    """Overlap query by bisection; a linear scan per span is too slow at 8k tokens."""
    index: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    for t, (o0, o1) in enumerate(offsets):
        if o1 <= o0:  # zero-width control token
            continue
        index.append(t)
        starts.append(o0)
        ends.append(o1)

    out: list[tuple[int, int]] = []
    for c0, c1 in char_spans:
        lo = bisect.bisect_right(ends, c0)  # first token whose end exceeds c0
        hi = bisect.bisect_left(starts, c1)  # one past the last token starting before c1
        if lo >= hi:
            raise ValueError(f"no token overlaps char span [{c0},{c1})")
        out.append((index[lo], index[hi - 1] + 1))
    return tuple(out)


def _span_incremental(
    text: str, char_spans: Sequence[tuple[int, int]], tokenizer: Any
) -> tuple[tuple[int, int], ...]:
    """Fallback for tokenizers without offsets: encode growing prefixes.

    Prefix lengths are not exactly additive when a merge straddles the cut, so the
    result can be off by a token at each edge. That is why offsets are preferred and
    why the item records which path produced its spans.
    """
    out: list[tuple[int, int]] = []
    for c0, c1 in char_spans:
        start = len(tokenizer(text[:c0], add_special_tokens=False)["input_ids"])
        end = len(tokenizer(text[:c1], add_special_tokens=False)["input_ids"])
        out.append((start, max(end, start + 1)))
    return tuple(out)


def locate_spans(
    prompt: str, char_spans: Sequence[tuple[int, int]], tokenizer: Any
) -> tuple[tuple[tuple[int, int], ...], list[int], str]:
    """Map char spans in ``prompt`` to ``[start, end)`` token spans.

    Returns ``(token_spans, input_ids, method)`` where ``method`` is ``"offsets"`` or
    ``"incremental"``.
    """
    if offsets_available(tokenizer):
        enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
        offsets = [tuple(o) for o in enc["offset_mapping"]]
        return _spans_from_offsets(offsets, char_spans), list(enc["input_ids"]), "offsets"
    ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    return _span_incremental(prompt, char_spans, tokenizer), ids, "incremental"


def tokenize_item(item: BenchmarkItem, tokenizer: Any | None = None) -> TokenizedItem:
    """Tokenize one item and measure where its evidence actually landed.

    ``item.meta["evidence_char_spans"]`` holds spans into ``item.user``; the chat
    template inserts that string verbatim, so shifting by the template prefix is
    exact. The shift is verified rather than assumed.
    """
    if not isinstance(item, BenchmarkItem):
        # ``evaluation.runner`` exports a same-named helper with the arguments the
        # other way round. Both are legitimate -- this one measures spans through the
        # benchmark's recorded char offsets, that one re-finds the sentences in the
        # rendered prompt -- but a swapped call would otherwise fail somewhere deep.
        raise TypeError(
            "benchmark.tokenize.tokenize_item(item, tokenizer); you may want "
            "evaluation.runner.tokenize_item(tok, item), which takes them the other "
            f"way round. Got {type(item).__name__} as the first argument."
        )
    tok = tokenizer if tokenizer is not None else get_tokenizer()
    prompt = build_chat_prompt(item.system, item.user, tok)
    try:
        shift = prompt.rindex(item.user)
    except ValueError as exc:  # pragma: no cover - only if a template rewrites content
        raise ValueError(
            f"{item.item_id}: chat template did not embed the user turn verbatim"
        ) from exc

    raw_spans = item.meta.get("evidence_char_spans")
    if not raw_spans:
        raise ValueError(f"{item.item_id}: meta has no evidence_char_spans")
    char_spans = [(int(a) + shift, int(b) + shift) for a, b in raw_spans]
    for (a, b), sentence in zip(char_spans, item.evidence_sentences, strict=True):
        if prompt[a:b] != sentence:
            raise ValueError(
                f"{item.item_id}: recorded char span does not cover its evidence "
                f"sentence; got {prompt[a:b]!r}"
            )

    token_spans, input_ids, _method = locate_spans(prompt, char_spans, tok)
    n = len(input_ids)
    positions = tuple(((s + e) / 2.0) / n for s, e in token_spans)
    answer_ids = tuple(tok(item.answer, add_special_tokens=False)["input_ids"])
    if not answer_ids:
        raise ValueError(f"{item.item_id}: answer {item.answer!r} encoded to nothing")
    return TokenizedItem(
        item_id=item.item_id,
        input_ids=tuple(input_ids),
        answer_ids=answer_ids,
        evidence_spans=token_spans,
        actual_positions=positions,
    )


def span_method(tokenizer: Any | None = None) -> str:
    """Which localisation path this tokenizer will take. Recorded in the manifest."""
    tok = tokenizer if tokenizer is not None else get_tokenizer()
    return "offsets" if offsets_available(tok) else "incremental"


__all__ = [
    "SYSTEM_PROMPT",
    "build_chat_prompt",
    "count_prompt_tokens",
    "count_tokens",
    "encode",
    "get_tokenizer",
    "locate_spans",
    "offsets_available",
    "prompt_token_counter",
    "span_method",
    "tokenize_item",
]

from __future__ import annotations

"""Shared, auditable response-boundary control for short-response inference.

The model should learn to emit EOS early, but a deployed strict short mode also
needs a deterministic runtime guarantee.  This module provides that guarantee
without pretending that punctuation alone proves semantic adequacy.

The controller is intentionally conservative:
- EOS remains the primary model-native stop signal.
- ``one_sentence`` additionally stops after the first validated sentence
  boundary.
- Decimal points, common abbreviations, URLs/domains, code spans, ellipses, and
  unmatched quotes/brackets are protected from premature termination.
- A hard token cap remains the final fail-safe.

This is output control, not structural model pruning.  It reduces avoided decode
steps and output-side KV growth; it does not remove model weights from memory.
"""

from dataclasses import dataclass
import re
from typing import Any, Iterable

try:
    from transformers import StoppingCriteria, StoppingCriteriaList
except ImportError:  # Pure boundary tests do not require the heavy runtime stack.
    class StoppingCriteria:  # type: ignore[no-redef]
        pass

    class StoppingCriteriaList(list):  # type: ignore[no-redef]
        pass

RESPONSE_CONTROL_MODES = ("eos_only", "one_sentence")

# A deliberately small, reviewable set.  The detector also protects single
# initials and dotted initialisms algorithmically.
_COMMON_ABBREVIATIONS = {
    "a.m.", "p.m.", "approx.", "ave.", "dept.", "dr.", "e.g.", "etc.",
    "fig.", "i.e.", "inc.", "jr.", "mr.", "mrs.", "ms.", "no.", "prof.",
    "rd.", "sr.", "st.", "u.k.", "u.s.", "u.s.a.", "vs.",
}
_SENTENCE_MARKS = frozenset(".!?。！？")
_CLOSERS = frozenset('"\'”’»）)]}】》')
_OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}", "（": "）", "【": "】", "《": "》"}
_QUOTE_PAIRS = {"\"": "\"", "'": "'", "“": "”", "‘": "’", "«": "»"}
_URL_RE = re.compile(r"(?i)(?:https?://|www\.)\S+$")
_DOMAIN_RE = re.compile(r"(?i)\b(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/\S*)?$")
_INITIALISM_RE = re.compile(r"(?i)(?:\b[a-z]\.){2,}$")
_SINGLE_INITIAL_RE = re.compile(r"(?i)\b[a-z]\.$")


@dataclass(frozen=True)
class BoundaryDecision:
    found: bool
    end_index: int | None = None
    reason: str | None = None


@dataclass(frozen=True)
class FinalizedResponse:
    raw_text: str
    text: str
    trimmed: bool
    sentence_count: int
    first_boundary_reason: str | None


def validate_response_control_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in RESPONSE_CONTROL_MODES:
        raise ValueError(
            f"Unsupported response-control mode {mode!r}; expected one of {RESPONSE_CONTROL_MODES}."
        )
    return normalized


def _inside_code(text: str, index: int) -> bool:
    """Return whether ``index`` is inside a Markdown code span/fence."""

    prefix = text[: index + 1]
    # An unmatched fenced block takes precedence over inline backticks.
    if prefix.count("```") % 2 == 1:
        return True
    without_fences = prefix.replace("```", "")
    return without_fences.count("`") % 2 == 1


def _balanced_before_stop(text: str, end_index: int) -> bool:
    """Require quotes/brackets opened before the boundary to be closed.

    Apostrophes inside words are ignored.  Straight quote handling is parity
    based, which is intentionally conservative rather than language-complete.
    """

    candidate = text[:end_index]
    stack: list[str] = []
    for i, char in enumerate(candidate):
        if char in _OPEN_TO_CLOSE:
            stack.append(_OPEN_TO_CLOSE[char])
        elif char in _OPEN_TO_CLOSE.values():
            if stack and stack[-1] == char:
                stack.pop()
        elif char in _QUOTE_PAIRS:
            if char == "'" and 0 < i < len(candidate) - 1:
                if candidate[i - 1].isalnum() and candidate[i + 1].isalnum():
                    continue
            closer = _QUOTE_PAIRS[char]
            if stack and stack[-1] == char:
                stack.pop()
            elif stack and stack[-1] == closer:
                stack.pop()
            else:
                stack.append(closer)
        elif char in _QUOTE_PAIRS.values() and stack and stack[-1] == char:
            stack.pop()
    return not stack


def _token_before(text: str, index: int) -> str:
    start = index
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    return text[start : index + 1]


def _is_protected_period(text: str, index: int) -> bool:
    prev_char = text[index - 1] if index > 0 else ""
    next_char = text[index + 1] if index + 1 < len(text) else ""

    # Decimal/version segments such as 3.14 or 2.1.0.
    if prev_char.isdigit() and next_char.isdigit():
        return True

    # Ordered-list markers such as "1. item" should not terminate before the
    # item text.  Only protect a numeric marker at the start of a line.
    line_start = text.rfind("\n", 0, index) + 1
    line_prefix = text[line_start : index + 1].strip()
    if re.fullmatch(r"\d{1,3}\.", line_prefix) and next_char.isspace():
        return True

    token = _token_before(text, index)
    lowered = token.casefold().strip("".join(_CLOSERS))
    if lowered in _COMMON_ABBREVIATIONS:
        return True
    if _INITIALISM_RE.search(lowered) or _SINGLE_INITIAL_RE.search(lowered):
        return True
    if _URL_RE.search(lowered) or _DOMAIN_RE.search(lowered):
        return True

    # A dot followed immediately by a letter/number is normally internal to a
    # filename, identifier, version, or domain rather than a sentence boundary.
    if next_char and (next_char.isalnum() or next_char in "_-/"):
        return True
    return False


def sentence_boundaries(text: str) -> list[BoundaryDecision]:
    """Return validated sentence boundaries in ``text``.

    A boundary includes trailing quote/bracket closers when they are already
    present.  The function does not claim semantic completeness; it only avoids
    common mechanical false positives better than raw punctuation matching.
    """

    decisions: list[BoundaryDecision] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char == "\n":
            if not _inside_code(text, i):
                preceding = text[:i].rstrip()
                if preceding and _balanced_before_stop(text, len(preceding)):
                    decisions.append(BoundaryDecision(True, len(preceding), "validated_line_boundary"))
            i += 1
            continue
        if char not in _SENTENCE_MARKS and char != "…":
            i += 1
            continue
        if _inside_code(text, i):
            i += 1
            continue

        # Treat a run of ASCII dots as one ellipsis candidate.  A single dot is
        # checked for abbreviation/decimal/domain protections.  Ellipses are
        # protected when ordinary continuation text follows; they become a
        # boundary only when they actually end the visible utterance.
        reason = "validated_sentence_boundary"
        mark_end = i + 1
        ellipsis = False
        if char == ".":
            run_end = i
            while run_end + 1 < len(text) and text[run_end + 1] == ".":
                run_end += 1
            if run_end > i:
                mark_end = run_end + 1
                reason = "validated_ellipsis_boundary"
                ellipsis = True
            elif _is_protected_period(text, i):
                i += 1
                continue
        elif char == "…":
            reason = "validated_ellipsis_boundary"
            ellipsis = True

        end = mark_end
        while end < len(text) and text[end] in _CLOSERS:
            end += 1
        if not _balanced_before_stop(text, end):
            i = mark_end
            continue

        if ellipsis and text[end:].strip():
            i = mark_end
            continue

        # Once a non-protected sentence mark is reached it is a boundary even
        # when the next sentence is attached without ASCII whitespace.  This is
        # required for normal CJK text. Periods internal to domains, versions,
        # identifiers, and decimals were filtered above.
        decisions.append(BoundaryDecision(True, end, reason))
        i = max(end, mark_end)
    return decisions


def first_sentence_boundary(text: str) -> BoundaryDecision:
    boundaries = sentence_boundaries(text)
    return boundaries[0] if boundaries else BoundaryDecision(False)


def finalize_response(raw_text: str, mode: str) -> FinalizedResponse:
    """Apply the visible-output guarantee after generation.

    Tokenizers can decode one token into punctuation plus following text.  The
    stopping callback cannot retract part of a token, so finalization trims the
    user-visible string at the validated boundary when strict mode is active.
    """

    mode = validate_response_control_mode(mode)
    value = raw_text.strip()
    boundaries = sentence_boundaries(value)
    if mode == "one_sentence" and boundaries:
        first = boundaries[0]
        assert first.end_index is not None
        visible = value[: first.end_index].rstrip()
        return FinalizedResponse(
            raw_text=value,
            text=visible,
            trimmed=visible != value,
            sentence_count=1 if visible else 0,
            first_boundary_reason=first.reason,
        )
    return FinalizedResponse(
        raw_text=value,
        text=value,
        trimmed=False,
        sentence_count=len(boundaries),
        first_boundary_reason=boundaries[0].reason if boundaries else None,
    )


class OneSentenceStoppingCriteria(StoppingCriteria):
    """Stop batch-size-one generation at the first validated sentence boundary."""

    def __init__(self, tokenizer: Any, prompt_length: int, min_new_tokens: int = 1) -> None:
        self.tokenizer = tokenizer
        self.prompt_length = int(prompt_length)
        self.min_new_tokens = max(1, int(min_new_tokens))
        self.triggered = False
        self.trigger_reason: str | None = None
        self.trigger_text: str | None = None

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
        del scores, kwargs
        if int(input_ids.shape[0]) != 1:
            raise ValueError("OneSentenceStoppingCriteria currently supports batch size 1 only.")
        continuation = input_ids[0, self.prompt_length :]
        if int(continuation.shape[0]) < self.min_new_tokens:
            return False
        text = self.tokenizer.decode(continuation, skip_special_tokens=True)
        decision = first_sentence_boundary(text)
        if decision.found:
            self.triggered = True
            self.trigger_reason = decision.reason
            self.trigger_text = text
            return True
        return False


def build_stopping_criteria(
    tokenizer: Any,
    prompt_length: int,
    mode: str,
    min_new_tokens: int = 1,
) -> tuple[StoppingCriteriaList | None, OneSentenceStoppingCriteria | None]:
    mode = validate_response_control_mode(mode)
    if mode == "eos_only":
        return None, None
    criterion = OneSentenceStoppingCriteria(tokenizer, prompt_length, min_new_tokens)
    return StoppingCriteriaList([criterion]), criterion


def infer_stop_reason(
    generated_ids: Iterable[int],
    *,
    eos_token_id: int | Iterable[int] | None,
    max_new_tokens: int,
    criterion: OneSentenceStoppingCriteria | None = None,
) -> str:
    ids = [int(token_id) for token_id in generated_ids]
    if criterion is not None and criterion.triggered:
        return criterion.trigger_reason or "validated_sentence_boundary"
    eos_ids: set[int] = set()
    if eos_token_id is not None:
        if isinstance(eos_token_id, int):
            eos_ids.add(eos_token_id)
        else:
            eos_ids.update(int(value) for value in eos_token_id)
    if ids and ids[-1] in eos_ids:
        return "eos"
    if len(ids) >= max_new_tokens:
        return "max_new_tokens"
    return "unknown"

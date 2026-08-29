"""Document assembly, position control, and the four surface formats.

The matched-construction rule of the brief is enforced structurally rather than by
convention: a scenario's chunk bank -- filler paragraphs, evidence sentences and
distractor sentences, all of it -- is generated **once**, and every positional
variant is a *permutation of that same list*. Nothing is regenerated, so the multiset
of filler chunks, the distractor set, the answer, the separators and the sentence
count are identical across variants by construction, and
``filler.filler_fingerprint`` makes that checkable.

Two position knobs are orthogonal:

* **global** -- which index of the ordered chunk list the evidence's host paragraph
  occupies, chosen so the evidence's character midpoint lands near ``target_p``;
* **local** -- ``first`` / ``middle`` / ``last``, where the evidence sentence sits
  *inside* its host paragraph.

Character midpoints are what the placement search optimises because they are cheap
and monotone in the quantity we care about. The authoritative number is the measured
**token** midpoint from :mod:`rineya_psa.benchmark.tokenize`; both are recorded.

DEVIATION: the ``shuffled`` format randomises the order of the *filler* sentences
inside each paragraph and pins evidence and distractor sentences to their slots.
Shuffling those too would destroy the local-position factor and make ``shuffled``
non-comparable with the other three formats, which is the opposite of what the
format axis is for. ``meta["shuffle_scope"]`` records the choice.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rineya_psa.benchmark.families import FamilyExample, build_exact_retrieval
from rineya_psa.benchmark.filler import (
    DOMAIN_LABELS,
    KIND_DISTRACTOR,
    KIND_FILLER,
    MAX_SENTENCES,
    MIN_SENTENCES,
    Chunk,
    chunk_is_inert,
    collisions,
    evidence_kind,
    filler_fingerprint,
    generate_chunks,
    insert_sentence,
    local_slot_index,
    reserved_tokens,
)
from rineya_psa.types import FORMATS, stable_id

# Verbatim from Rineya_Long_Context_Demo_v1/data/showcase_prompts.jsonl so that
# preservation comparisons against the published showcase stay fair.
SYSTEM_PROMPT: str = (
    "You are answering a closed-context reading test. Use only the supplied context. "
    "Reply with the shortest exact answer and no explanation. If the context does not "
    "determine the answer, reply UNKNOWN."
)

SEPARATOR: str = "\n\n"
CONTEXT_HEADER: str = "CONTEXT\n"
MIN_CHUNKS: int = 6


def build_user_prompt(context: str, question: str) -> str:
    """The project's CONTEXT / QUESTION / ANSWER user turn, byte-identical in shape."""
    return f"{CONTEXT_HEADER}{context}\n\nQUESTION\n{question}\n\nANSWER"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _display_order(chunk: Chunk, fmt: str) -> tuple[int, ...]:
    n = len(chunk.sentences)
    if fmt != "shuffled":
        return tuple(range(n))
    filler_slots = [i for i, k in enumerate(chunk.kinds) if k == KIND_FILLER]
    permuted = list(filler_slots)
    # Seeded on the chunk id, not on its index, so a re-ordering does not reshuffle.
    random.Random(int(stable_id("shuffle", chunk.chunk_id)[:12], 16)).shuffle(permuted)
    order = list(range(n))
    for slot, src in zip(filler_slots, permuted, strict=True):
        order[slot] = src
    return tuple(order)


def render_chunk_with_offsets(chunk: Chunk, fmt: str) -> tuple[str, tuple[tuple[int, int], ...]]:
    """Render one paragraph and report each sentence's char span, by original index."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown fmt {fmt!r}; known: {FORMATS}")
    order = _display_order(chunk, fmt)
    shown = [chunk.sentences[i] for i in order]
    spans_by_display: list[tuple[int, int]] = []

    if fmt in ("natural", "shuffled"):
        parts: list[str] = []
        cursor = 0
        for i, sentence in enumerate(shown):
            if i:
                parts.append(" ")
                cursor += 1
            spans_by_display.append((cursor, cursor + len(sentence)))
            parts.append(sentence)
            cursor += len(sentence)
        text = "".join(parts)
    elif fmt == "list":
        parts = []
        cursor = 0
        for i, sentence in enumerate(shown):
            if i:
                parts.append("\n")
                cursor += 1
            parts.append("- ")
            cursor += 2
            spans_by_display.append((cursor, cursor + len(sentence)))
            parts.append(sentence)
            cursor += len(sentence)
        text = "".join(parts)
    else:  # json
        payload = {
            "record": chunk.chunk_id,
            "source": DOMAIN_LABELS[chunk.domain],
            "entries": shown,
        }
        text = json.dumps(payload, ensure_ascii=False)
        cursor = 0
        for sentence in shown:
            try:
                start = text.index(sentence, cursor)
            except ValueError:  # a sentence needed JSON escaping
                escaped = json.dumps(sentence, ensure_ascii=False)[1:-1]
                start = text.index(escaped, cursor)
                spans_by_display.append((start, start + len(escaped)))
                cursor = start + len(escaped)
                continue
            spans_by_display.append((start, start + len(sentence)))
            cursor = start + len(sentence)

    spans = [(0, 0)] * len(order)
    for display_index, original_index in enumerate(order):
        spans[original_index] = spans_by_display[display_index]
    return text, tuple(spans)


def render_chunk(chunk: Chunk, fmt: str) -> str:
    return render_chunk_with_offsets(chunk, fmt)[0]


def render_context(
    chunks: Sequence[Chunk], fmt: str, *, separator: str = SEPARATOR
) -> tuple[str, tuple[tuple[tuple[int, int], ...], ...]]:
    """Render the whole document; return it plus per-chunk, per-sentence char spans."""
    parts: list[str] = []
    all_spans: list[tuple[tuple[int, int], ...]] = []
    cursor = 0
    for i, chunk in enumerate(chunks):
        if i:
            parts.append(separator)
            cursor += len(separator)
        text, spans = render_chunk_with_offsets(chunk, fmt)
        all_spans.append(tuple((s + cursor, e + cursor) for s, e in spans))
        parts.append(text)
        cursor += len(text)
    return "".join(parts), tuple(all_spans)


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Placement:
    """Where one evidence sentence ended up, target and achievement side by side."""

    host_index: int
    host_id: str
    slot: int
    target_p: float
    achieved_char_p: float
    local_position: str


def place_evidence(
    chunks: Sequence[Chunk],
    evidence: str,
    target_p: float,
    *,
    slot: int = 0,
    local_position: str = "middle",
    fmt: str = "natural",
    separator: str = SEPARATOR,
    exclude: Sequence[str] = (),
) -> tuple[tuple[Chunk, ...], Placement]:
    """Insert ``evidence`` into whichever paragraph puts its char midpoint near ``target_p``.

    The search is exact rather than proportional: for every candidate host the
    resulting document length and the evidence offset inside the rewritten host are
    both computed, so a long paragraph near the head does not bias the answer.
    """
    if not 0.0 <= target_p <= 1.0:
        raise ValueError(f"target_p must lie in [0,1], got {target_p}")
    if not chunks:
        raise ValueError("cannot place evidence into an empty chunk list")
    banned = set(exclude)
    lens = [len(render_chunk(c, fmt)) for c in chunks]
    sep = len(separator)
    prefix = [0]
    for length in lens:
        prefix.append(prefix[-1] + length)
    base_total = prefix[-1] + sep * (len(chunks) - 1)

    best: tuple[float, int, Chunk, float] | None = None
    for i, chunk in enumerate(chunks):
        if chunk.chunk_id in banned or chunk.carries_evidence:
            continue
        candidate = insert_sentence(chunk, evidence, evidence_kind(slot), local_position)
        text, spans = render_chunk_with_offsets(candidate, fmt)
        ev_index = candidate.sentence_index_of_kind(evidence_kind(slot))
        start, end = spans[ev_index]
        total = base_total - lens[i] + len(text)
        mid = prefix[i] + sep * i + (start + end) / 2.0
        achieved = mid / total if total else 0.0
        cost = abs(achieved - target_p)
        if best is None or cost < best[0]:
            best = (cost, i, candidate, achieved)
    if best is None:
        raise ValueError("no eligible host paragraph left for this evidence sentence")

    _, index, candidate, achieved = best
    out = list(chunks)
    out[index] = candidate
    return tuple(out), Placement(
        host_index=index,
        host_id=candidate.chunk_id,
        slot=slot,
        target_p=target_p,
        achieved_char_p=achieved,
        local_position=local_position,
    )


@dataclass(frozen=True)
class Layout:
    """A rendered positional variant of one scenario bank."""

    chunks: tuple[Chunk, ...]
    context: str
    user: str
    evidence_char_spans: tuple[tuple[int, int], ...]
    achieved_char_positions: tuple[float, ...]
    distractor_char_positions: tuple[float, ...]
    order_signature: str


def reorder_for_targets(
    chunks: Sequence[Chunk],
    host_ids: Sequence[str],
    targets: Sequence[float],
    *,
    fmt: str = "natural",
    separator: str = SEPARATOR,
) -> tuple[Chunk, ...]:
    """Permute the chunk list so evidence ``k`` lands near ``targets[k]``.

    ``host_ids[k]`` is the paragraph already carrying evidence ``k``. Non-host chunks
    keep their relative order, so the only thing that varies between positional
    variants is where the hosts were spliced back in.

    Because the hosts are re-inserted in ascending target order, the character offset
    of host ``k`` depends only on its own gap and on the total length of the hosts
    placed before it -- both known in closed form -- so a single left-to-right pass is
    exact, not greedy-approximate.
    """
    if len(host_ids) != len(targets):
        raise ValueError(f"{len(host_ids)} hosts vs {len(targets)} targets")
    by_id = {c.chunk_id: c for c in chunks}
    missing = [h for h in host_ids if h not in by_id]
    if missing:
        raise KeyError(f"host chunk(s) not in bank: {missing}")

    host_set = set(host_ids)
    rest = [c for c in chunks if c.chunk_id not in host_set]
    rest_len = [len(render_chunk(c, fmt)) for c in rest]
    rest_prefix = [0]
    for length in rest_len:
        rest_prefix.append(rest_prefix[-1] + length)

    sep = len(separator)
    host_text: list[str] = []
    host_mid: list[float] = []
    for k, hid in enumerate(host_ids):
        text, spans = render_chunk_with_offsets(by_id[hid], fmt)
        idx = by_id[hid].sentence_index_of_kind(evidence_kind(k))
        start, end = spans[idx]
        host_text.append(text)
        host_mid.append((start + end) / 2.0)
    host_len = [len(t) for t in host_text]
    total = rest_prefix[-1] + sum(host_len) + sep * (len(chunks) - 1)

    order = sorted(range(len(targets)), key=lambda k: (targets[k], k))
    gaps: dict[int, int] = {}
    placed_len = 0
    low = 0
    for a, k in enumerate(order):
        best_gap, best_cost = low, None
        for g in range(low, len(rest) + 1):
            offset = rest_prefix[g] + placed_len + sep * (g + a)
            achieved = (offset + host_mid[k]) / total if total else 0.0
            cost = abs(achieved - targets[k])
            if best_cost is None or cost < best_cost:
                best_gap, best_cost = g, cost
        gaps[k] = best_gap
        low = best_gap
        placed_len += host_len[k]

    out: list[Chunk] = []
    cursor = 0
    for k in order:
        out.extend(rest[cursor : gaps[k]])
        cursor = gaps[k]
        out.append(by_id[host_ids[k]])
    out.extend(rest[cursor:])
    if len(out) != len(chunks):
        raise AssertionError("re-ordering lost or duplicated a chunk")
    return tuple(out)


def measure_layout(
    ordered: Sequence[Chunk],
    question: str,
    *,
    fmt: str,
    n_evidence: int,
    separator: str = SEPARATOR,
) -> Layout:
    """Render an ordering and read the achieved positions back out of the text."""
    context, spans = render_context(ordered, fmt, separator=separator)
    user = build_user_prompt(context, question)
    shift = len(CONTEXT_HEADER)
    total = len(context)

    ev_spans: list[tuple[int, int] | None] = [None] * n_evidence
    ev_pos: list[float | None] = [None] * n_evidence
    distractor_pos: list[float] = []
    for chunk, chunk_spans in zip(ordered, spans, strict=True):
        for kind, (start, end) in zip(chunk.kinds, chunk_spans, strict=True):
            if kind == KIND_DISTRACTOR:
                distractor_pos.append(((start + end) / 2.0) / total if total else 0.0)
            elif kind.startswith("evidence:"):
                k = int(kind.split(":", 1)[1])
                if k >= n_evidence:
                    raise AssertionError(f"evidence slot {k} beyond n_evidence={n_evidence}")
                ev_spans[k] = (start + shift, end + shift)
                ev_pos[k] = ((start + end) / 2.0) / total if total else 0.0
    if any(s is None for s in ev_spans):
        missing = [k for k, s in enumerate(ev_spans) if s is None]
        raise AssertionError(f"evidence slots {missing} never made it into the document")

    return Layout(
        chunks=tuple(ordered),
        context=context,
        user=user,
        evidence_char_spans=tuple(s for s in ev_spans if s is not None),
        achieved_char_positions=tuple(p for p in ev_pos if p is not None),
        distractor_char_positions=tuple(distractor_pos),
        order_signature=stable_id("order", *[c.chunk_id for c in ordered]),
    )


# ---------------------------------------------------------------------------
# Scenario banks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioBank:
    """The one-and-only chunk list behind every positional variant of a scenario."""

    chunks: tuple[Chunk, ...]
    host_ids: tuple[str, ...]
    example: FamilyExample
    fmt: str
    local_position: str
    sentences_per_chunk: int
    ballast_sentences: int
    n_chunks: int
    token_target: int
    n_prompt_tokens: int
    filler_fingerprint: str

    @property
    def token_error(self) -> float:
        return (self.n_prompt_tokens - self.token_target) / float(self.token_target)


def _distractor_slots(n_free: int, n_distractors: int) -> tuple[int, ...]:
    """Evenly spread slot indices among the paragraphs not hosting evidence."""
    if n_distractors <= 0 or n_free <= 0:
        return ()
    take = min(n_distractors, n_free)
    return tuple(
        sorted({min(n_free - 1, int((j + 0.5) * n_free / take)) for j in range(take)})
    )


def build_bank(
    example: FamilyExample,
    *,
    n_chunks: int,
    fmt: str,
    local_position: str,
    seed: int,
    sentences_per_chunk: int = 3,
    ballast_sentences: int | None = None,
    canonical_targets: Sequence[float] | None = None,
    token_target: int = 0,
    n_prompt_tokens: int = 0,
) -> ScenarioBank:
    """Generate the chunk bank once: filler, then evidence, then distractors."""
    m = len(example.evidence_sentences)
    if n_chunks < max(MIN_CHUNKS, m + 1):
        raise ValueError(f"n_chunks={n_chunks} too small for {m} evidence sentences")
    reserved = reserved_tokens(example.meta["reserved_terms"])
    chunks = generate_chunks(
        n_chunks,
        seed=seed,
        sentences_per_chunk=sentences_per_chunk,
        ballast_sentences=ballast_sentences,
        reserved=reserved,
    )
    for chunk in chunks:
        if not chunk_is_inert(chunk, reserved):
            raise AssertionError(
                f"filler chunk {chunk.chunk_id} collides with reserved terms "
                f"{sorted(collisions(chunk, reserved))}"
            )

    if canonical_targets is None:
        canonical_targets = tuple((k + 0.5) / m for k in range(m))
    if len(canonical_targets) != m:
        raise ValueError(f"{len(canonical_targets)} canonical targets for {m} evidence")

    host_ids: list[str] = []
    for k, sentence in enumerate(example.evidence_sentences):
        chunks, placement = place_evidence(
            chunks,
            sentence,
            float(canonical_targets[k]),
            slot=k,
            local_position=local_position,
            fmt=fmt,
            exclude=host_ids,
        )
        host_ids.append(placement.host_id)

    free = [i for i, c in enumerate(chunks) if not c.carries_evidence]
    slots = _distractor_slots(len(free), len(example.distractor_sentences))
    working = list(chunks)
    # A short bank can offer fewer free paragraphs than there are distractors, so the
    # tail is dropped rather than doubled up; the surviving count is recorded per item.
    for sentence, slot in zip(example.distractor_sentences, slots, strict=False):
        i = free[slot]
        working[i] = insert_sentence(working[i], sentence, KIND_DISTRACTOR, "middle")
    chunks = tuple(working)

    return ScenarioBank(
        chunks=chunks,
        host_ids=tuple(host_ids),
        example=example,
        fmt=fmt,
        local_position=local_position,
        sentences_per_chunk=sentences_per_chunk,
        ballast_sentences=(
            sentences_per_chunk if ballast_sentences is None else ballast_sentences
        ),
        n_chunks=n_chunks,
        token_target=token_target,
        n_prompt_tokens=n_prompt_tokens,
        filler_fingerprint=filler_fingerprint(chunks),
    )


def render_variant(
    bank: ScenarioBank, targets: Sequence[float], *, separator: str = SEPARATOR
) -> Layout:
    """Produce the positional variant of ``bank`` for ``targets``."""
    ordered = reorder_for_targets(
        bank.chunks, bank.host_ids, targets, fmt=bank.fmt, separator=separator
    )
    return measure_layout(
        ordered,
        bank.example.question,
        fmt=bank.fmt,
        n_evidence=len(bank.example.evidence_sentences),
        separator=separator,
    )


# ---------------------------------------------------------------------------
# Token budgeting
# ---------------------------------------------------------------------------

PromptTokenCounter = Callable[[str, str], int]


@dataclass(frozen=True)
class TokenModel:
    """Affine token cost of a document: ``overhead + per_chunk * n_chunks``."""

    overhead: float
    per_chunk: float
    per_sentence: float

    def n_for(self, target: int) -> int:
        return max(MIN_CHUNKS, int(round((target - self.overhead) / max(self.per_chunk, 1.0))))


def _measure(
    example: FamilyExample,
    *,
    n_chunks: int,
    ballast: int,
    fmt: str,
    local_position: str,
    seed: int,
    sentences_per_chunk: int,
    canonical_targets: Sequence[float] | None,
    count_prompt_tokens: PromptTokenCounter,
) -> tuple[ScenarioBank, int]:
    bank = build_bank(
        example,
        n_chunks=n_chunks,
        fmt=fmt,
        local_position=local_position,
        seed=seed,
        sentences_per_chunk=sentences_per_chunk,
        ballast_sentences=ballast,
        canonical_targets=canonical_targets,
    )
    layout = render_variant(bank, canonical_targets or tuple(
        (k + 0.5) / len(example.evidence_sentences)
        for k in range(len(example.evidence_sentences))
    ))
    return bank, count_prompt_tokens(SYSTEM_PROMPT, layout.user)


_PROBE_SEED = 0
_PROBE_LOCAL = "middle"


def _probe_example() -> FamilyExample:
    """A fixed example used only to calibrate the token model.

    Calibrating on the caller's own example would make the cached model depend on
    which scenario happened to be built first, which is a determinism hazard for no
    benefit -- the model is only a starting guess and the secant search below does the
    real work.
    """
    return build_exact_retrieval(random.Random(_PROBE_SEED))


def calibrate_token_model(
    *,
    fmt: str,
    sentences_per_chunk: int,
    count_prompt_tokens: PromptTokenCounter,
) -> TokenModel:
    """Two probe builds give the affine token cost of a document in this format."""
    example = _probe_example()
    probes: dict[int, int] = {}
    for n in (8, 24):
        _, probes[n] = _measure(
            example,
            n_chunks=n,
            ballast=sentences_per_chunk,
            fmt=fmt,
            local_position=_PROBE_LOCAL,
            seed=_PROBE_SEED,
            sentences_per_chunk=sentences_per_chunk,
            canonical_targets=None,
            count_prompt_tokens=count_prompt_tokens,
        )
    per_chunk = max((probes[24] - probes[8]) / 16.0, 1.0)
    return TokenModel(
        overhead=probes[8] - 8 * per_chunk,
        per_chunk=per_chunk,
        per_sentence=per_chunk / max(sentences_per_chunk, 1),
    )


def fit_bank(
    example: FamilyExample,
    *,
    token_target: int,
    tolerance: float,
    fmt: str,
    local_position: str,
    seed: int,
    count_prompt_tokens: PromptTokenCounter,
    sentences_per_chunk: int = 3,
    canonical_targets: Sequence[float] | None = None,
    model_cache: dict[tuple[str, int], TokenModel] | None = None,
    max_iterations: int = 12,
) -> ScenarioBank:
    """Choose ``(n_chunks, ballast_sentences)`` so the prompt lands inside ``tolerance``.

    Two nested searches. The outer one is a secant on chunk count: the affine model is
    only a seed, because extrapolating a per-chunk cost measured at 24 paragraphs out
    to the 550 a 32k document needs amplifies the sampling error past 2 percent. Each
    measurement re-estimates the slope from the two most recent points, which converges
    in two or three builds at every length.

    The inner one is the fine knob: chunk count moves the budget in ~65-token steps,
    which is 3 percent at 2k, so the trailing ballast paragraph's sentence count (3-6)
    fills in the gaps at ~20 tokens each.
    """
    floor = max(MIN_CHUNKS, len(example.evidence_sentences) + 2)
    key = (fmt, sentences_per_chunk)
    if model_cache is not None and key in model_cache:
        model = model_cache[key]
    else:
        model = calibrate_token_model(
            fmt=fmt,
            sentences_per_chunk=sentences_per_chunk,
            count_prompt_tokens=count_prompt_tokens,
        )
        if model_cache is not None:
            model_cache[key] = model

    def measure(n: int, ballast: int) -> tuple[ScenarioBank, int]:
        return _measure(
            example,
            n_chunks=n,
            ballast=ballast,
            fmt=fmt,
            local_position=local_position,
            seed=seed,
            sentences_per_chunk=sentences_per_chunk,
            canonical_targets=canonical_targets,
            count_prompt_tokens=count_prompt_tokens,
        )

    def error(tokens: int) -> float:
        return abs(tokens - token_target) / float(token_target)

    n = max(floor, model.n_for(token_target))
    bank, tokens = measure(n, sentences_per_chunk)
    best: tuple[float, ScenarioBank, int] = (error(tokens), bank, tokens)
    slope = model.per_chunk
    seen: set[int] = {n}
    for _ in range(max_iterations):
        if error(tokens) <= tolerance:
            break
        step = int(round((token_target - tokens) / max(slope, 1.0)))
        if step == 0:
            step = 1 if tokens < token_target else -1
        nxt = max(floor, n + step)
        if nxt in seen:
            break
        prev_n, prev_tokens = n, tokens
        bank, tokens = measure(nxt, sentences_per_chunk)
        n = nxt
        seen.add(n)
        if error(tokens) < best[0]:
            best = (error(tokens), bank, tokens)
        if n != prev_n:
            slope = max((tokens - prev_tokens) / float(n - prev_n), 1.0)

    ballasts = [sentences_per_chunk] + [
        b for b in range(MIN_SENTENCES, MAX_SENTENCES + 1) if b != sentences_per_chunk
    ]
    anchor = n
    for dn in (0, -1, 1):
        candidate_n = anchor + dn
        if candidate_n < floor:
            continue
        for ballast in ballasts:
            trial, trial_tokens = measure(candidate_n, ballast)
            if error(trial_tokens) < best[0]:
                best = (error(trial_tokens), trial, trial_tokens)
            if best[0] <= tolerance:
                break
        if best[0] <= tolerance:
            break

    _, bank, tokens = best
    return ScenarioBank(
        chunks=bank.chunks,
        host_ids=bank.host_ids,
        example=bank.example,
        fmt=bank.fmt,
        local_position=bank.local_position,
        sentences_per_chunk=bank.sentences_per_chunk,
        ballast_sentences=bank.ballast_sentences,
        n_chunks=bank.n_chunks,
        token_target=token_target,
        n_prompt_tokens=tokens,
        filler_fingerprint=bank.filler_fingerprint,
    )


__all__ = [
    "CONTEXT_HEADER",
    "Layout",
    "MIN_CHUNKS",
    "Placement",
    "PromptTokenCounter",
    "SEPARATOR",
    "SYSTEM_PROMPT",
    "ScenarioBank",
    "TokenModel",
    "build_bank",
    "build_user_prompt",
    "calibrate_token_model",
    "fit_bank",
    "local_slot_index",
    "measure_layout",
    "place_evidence",
    "render_chunk",
    "render_chunk_with_offsets",
    "render_context",
    "render_variant",
    "reorder_for_targets",
]

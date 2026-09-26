"""Frozen Phase 0 context-refinement operators for the Supa long-context pilot.

Four input conditions are built from the SAME original context:

    A  full        -- original context, unchanged (canonical baseline)
    B  query       -- BM25-scored chunks under a fixed token budget
    C  random      -- same chunk COUNT as B, chosen uniformly at random
    D  oracle      -- chunks localised from dataset ``supporting_facts`` metadata

Every condition is emitted in ORIGINAL DOCUMENT ORDER.

Design rules frozen before any scoring run (see ``preregistrations/``):

* one chunk size, no sweep;
* the structural anchor is the document's FIRST SENTENCE only -- tiny, ~30
  tokens -- and it is prepended to B, C and D alike, so it can never be the
  thing that separates them;
* C matches B's chunk count per example, so B - C isolates *relevance* rather
  than *brevity*;
* the selector is lexical only -- no embedding model, no second LLM.

Sizing note. Demo-suite contexts run 1048-1481 tokens (median 1264), which is
6-9 chunks at the 192-token target. An earlier draft used a whole-chunk anchor
with a 768-token ceiling; that left B at 56% of the original and made Oracle
collapse onto Query on the first example inspected. The budget below was
tightened once, before any model was run, and then frozen.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

# --------------------------------------------------------------------------
# Frozen selector configuration
# --------------------------------------------------------------------------

CHUNK_TARGET_TOKENS = 192
"""Low end of the mandated 192-384 band. At 384 a context would hold only 3-4
chunks and the budget would barely be able to exclude anything."""

CHUNK_OVERLAP_SENTENCES = 0
"""Chunks are packed on sentence boundaries, so a sentence is never split and
overlap buys nothing. Neighbour expansion covers evidence that straddles a
boundary."""

SELECT_BUDGET_TOKENS = 384
"""~30% of the median context. Fixed and absolute, not a ratio, so the
reduction is comparable across the three length buckets."""

NEIGHBOUR_RADIUS = 1
"""Each scored-in chunk pulls in its immediate neighbours, subject to the
ceiling below. Two-hop items often place the second hop adjacent to the first."""

HARD_CEILING_TOKENS = 576
"""~45% of the median context: room for exactly one neighbour chunk beyond the
scored budget."""

USE_SENTENCE_ANCHOR = True
ANCHOR_SENTENCES = 1
"""Every demo-suite context opens with a framing sentence naming the document
and its scope. Without it the retained chunks read as orphaned fragments. One
sentence is ~30 tokens, so the anchor costs ~8% of the budget rather than the
~35% a whole leading chunk would."""

BM25_K1 = 1.5
BM25_B = 0.75

ORACLE_MIN_CONFIDENCE = 0.34
"""Minimum lexical overlap for a ``supporting_facts`` entry to count as
localised. ALL facts for a row must clear it or the row is Oracle-ineligible."""

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

STOPWORDS = frozenset(
    """a an and are as at be been but by did do does for from had has have he her his
    how i if in into is it its of on or our she that the their them then there these
    they this to was were what when where which who whom why will with you your""".split()
)


def lexical_tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with stopwords removed."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    text: str
    n_tokens: int
    sentence_span: tuple[int, int]
    terms: tuple[str, ...] = field(repr=False, default=())


def chunk_context(
    context: str,
    count_tokens: Callable[[str], int],
    target_tokens: int = CHUNK_TARGET_TOKENS,
) -> list[Chunk]:
    """Pack sentences into ~``target_tokens`` chunks without splitting sentences.

    A sentence longer than the target becomes its own chunk rather than being
    truncated -- the evidence must survive chunking intact.
    """
    sentences = split_sentences(context)
    if not sentences:
        return []

    sent_tokens = [count_tokens(s) for s in sentences]
    chunks: list[Chunk] = []
    start = 0
    running = 0

    for idx, n in enumerate(sent_tokens):
        # Close the open chunk *before* adding a sentence that would overshoot,
        # unless the chunk is still empty (an oversized lone sentence).
        if running and running + n > target_tokens:
            text = " ".join(sentences[start:idx])
            chunks.append(
                Chunk(len(chunks), text, running, (start, idx), tuple(lexical_tokens(text)))
            )
            start = idx
            running = 0
        running += n

    if running:
        text = " ".join(sentences[start:])
        chunks.append(
            Chunk(
                len(chunks),
                text,
                running,
                (start, len(sentences)),
                tuple(lexical_tokens(text)),
            )
        )
    return chunks


# --------------------------------------------------------------------------
# BM25 relevance
# --------------------------------------------------------------------------


def bm25_scores(chunks: Sequence[Chunk], query: str) -> list[float]:
    """Okapi BM25 of each chunk against the query, IDF computed over the chunks
    of this one document (the document *is* the corpus here)."""
    if not chunks:
        return []

    query_terms = lexical_tokens(query)
    if not query_terms:
        return [0.0] * len(chunks)

    n_docs = len(chunks)
    avg_len = sum(c.n_tokens for c in chunks) / n_docs
    scores = [0.0] * n_docs

    for term in set(query_terms):
        containing = sum(1 for c in chunks if term in c.terms)
        if containing == 0:
            continue
        idf = math.log(1.0 + (n_docs - containing + 0.5) / (containing + 0.5))
        for i, chunk in enumerate(chunks):
            tf = chunk.terms.count(term)
            if not tf:
                continue
            denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * chunk.n_tokens / avg_len)
            scores[i] += idf * (tf * (BM25_K1 + 1)) / denom
    return scores


# --------------------------------------------------------------------------
# Shared assembly
# --------------------------------------------------------------------------


def anchor_text(context: str) -> str:
    """The tiny structural anchor: the document's opening sentence(s)."""
    if not USE_SENTENCE_ANCHOR:
        return ""
    return " ".join(split_sentences(context)[:ANCHOR_SENTENCES])


def _expand_and_cap(
    chunks: Sequence[Chunk],
    core_ids: Iterable[int],
    radius: int = NEIGHBOUR_RADIUS,
    ceiling: int = HARD_CEILING_TOKENS,
) -> list[int]:
    """Add +/-``radius`` neighbours around ``core_ids``, dropping any addition
    that would breach ``ceiling``.

    Core chunks are never dropped: the ceiling only governs how much neighbour
    padding survives.
    """
    core = set(core_ids)
    total = sum(chunks[i].n_tokens for i in core)

    candidates: list[int] = []
    for offset in range(1, radius + 1):
        for cid in sorted(core):
            for neighbour in (cid - offset, cid + offset):
                if 0 <= neighbour < len(chunks) and neighbour not in core:
                    if neighbour not in candidates:
                        candidates.append(neighbour)

    selected = set(core)
    for neighbour in candidates:
        cost = chunks[neighbour].n_tokens
        if total + cost > ceiling:
            continue
        selected.add(neighbour)
        total += cost
    return sorted(selected)


def assemble(
    chunks: Sequence[Chunk],
    chunk_ids: Iterable[int],
    anchor: str = "",
) -> str:
    """Join chunks in ORIGINAL DOCUMENT ORDER, prefixed by the anchor.

    Non-adjacent runs are separated by a blank line so the model sees a section
    break rather than a false sentence adjacency. The anchor is suppressed when
    chunk 0 is selected, because chunk 0 already opens with it. Both markers are
    identical across conditions B, C and D.
    """
    ordered = sorted(set(chunk_ids))
    parts: list[str] = []
    if anchor and 0 not in ordered:
        parts.extend([anchor, ""])
    previous: int | None = None
    for cid in ordered:
        if previous is not None and cid != previous + 1:
            parts.append("")  # gap marker -> blank line after join
        parts.append(chunks[cid].text)
        previous = cid
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------


@dataclass
class Selection:
    condition: str
    chunk_ids: list[int]
    text: str
    n_tokens: int
    n_chunks: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _finalise(
    condition: str,
    chunks: Sequence[Chunk],
    chunk_ids: Sequence[int],
    anchor: str,
    count_tokens: Callable[[str], int] | None,
    diagnostics: dict[str, Any],
) -> Selection:
    text = assemble(chunks, chunk_ids, anchor)
    n_tokens = (
        count_tokens(text)
        if count_tokens is not None
        else sum(chunks[i].n_tokens for i in chunk_ids)
    )
    return Selection(
        condition=condition,
        chunk_ids=list(chunk_ids),
        text=text,
        n_tokens=n_tokens,
        n_chunks=len(chunk_ids),
        diagnostics=diagnostics,
    )


def select_query(
    chunks: Sequence[Chunk],
    query: str,
    anchor: str = "",
    budget: int = SELECT_BUDGET_TOKENS,
    count_tokens: Callable[[str], int] | None = None,
) -> Selection:
    """Condition B: BM25 chunks under ``budget``, then neighbour expansion."""
    scores = bm25_scores(chunks, query)
    order = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))

    core: set[int] = set()
    total = 0
    for cid in order:
        if scores[cid] <= 0.0:
            continue  # a chunk sharing no query term is filler, not a selection
        if total + chunks[cid].n_tokens > budget:
            continue
        core.add(cid)
        total += chunks[cid].n_tokens

    if not core and chunks:
        # Degenerate query (no lexical overlap anywhere). Fall back to the
        # leading chunk so the condition is still a valid, non-empty context.
        core = {0}

    chunk_ids = _expand_and_cap(chunks, core)
    return _finalise(
        "query",
        chunks,
        chunk_ids,
        anchor,
        count_tokens,
        {
            "scored_core_ids": sorted(core),
            "bm25_scores": [round(s, 4) for s in scores],
            "n_chunks_available": len(chunks),
            "degenerate_query": not any(s > 0 for s in scores),
        },
    )


def select_random(
    chunks: Sequence[Chunk],
    n_chunks: int,
    seed: int,
    anchor: str = "",
    count_tokens: Callable[[str], int] | None = None,
) -> Selection:
    """Condition C: ``n_chunks`` chunks drawn uniformly at random.

    The anchor is prepended exactly as for B, so the two conditions differ only
    in which chunks fill the slots. No neighbour expansion is applied here --
    B's neighbours are already counted in ``n_chunks``, so re-expanding would
    overshoot the match.
    """
    rng = random.Random(seed)
    pool = [c.chunk_id for c in chunks]
    take = max(0, min(n_chunks, len(pool)))
    chosen = sorted(rng.sample(pool, take))
    return _finalise(
        "random",
        chunks,
        chosen,
        anchor,
        count_tokens,
        {"target_n_chunks": n_chunks, "seed": seed},
    )


def localise_fact_exact(
    chunks: Sequence[Chunk], fact: str, context: str
) -> tuple[int, float]:
    """Exact-span localisation: find the chunk whose text contains ``fact``.

    Available only where the dataset states supporting facts verbatim. On the
    LCR dev split every one of 2688 facts is an exact substring of its context;
    on the demo suite none of 424 are.
    """
    if fact in context:
        for chunk in chunks:
            if fact in chunk.text:
                return chunk.chunk_id, 1.0
        # The fact straddles a chunk boundary: attribute it to the chunk holding
        # the longest overlapping prefix rather than dropping it.
        best_id, best_len = -1, 0
        for chunk in chunks:
            for cut in range(len(fact), 0, -1):
                if fact[:cut] in chunk.text:
                    if cut > best_len:
                        best_id, best_len = chunk.chunk_id, cut
                    break
        if best_id >= 0:
            return best_id, best_len / len(fact)
    return -1, 0.0


def localise_fact(chunks: Sequence[Chunk], fact: str) -> tuple[int, float]:
    """Best-matching chunk for one ``supporting_facts`` entry, by lexical overlap.

    Used where facts are abstracted rather than quoted. The demo suite states
    supporting facts in abstracted form -- 0 of 424 appear verbatim in their
    context -- so localisation is approximate and its confidence is reported per
    example rather than assumed.
    """
    fact_terms = set(lexical_tokens(fact))
    if not fact_terms or not chunks:
        return -1, 0.0
    best_id, best_score = -1, -1.0
    for chunk in chunks:
        overlap = len(fact_terms & set(chunk.terms)) / len(fact_terms)
        if overlap > best_score:
            best_id, best_score = chunk.chunk_id, overlap
    return best_id, best_score


def select_oracle(
    chunks: Sequence[Chunk],
    supporting_facts: Sequence[str],
    answerable: bool,
    anchor: str = "",
    min_confidence: float = ORACLE_MIN_CONFIDENCE,
    count_tokens: Callable[[str], int] | None = None,
    context: str | None = None,
    mode: str = "lexical",
) -> Selection | None:
    """Condition D: chunks localised from dataset evidence metadata.

    Returns ``None`` -- Oracle UNAVAILABLE -- when either:

    * the row is unanswerable, whose ``supporting_facts`` is a constant
      boilerplate string ("The decisive record is missing, damaged, ambiguous,
      or explicitly unresolved.") carrying no localisation signal; or
    * any single fact fails to localise at ``min_confidence``. A two-hop oracle
      missing one hop is not an oracle.

    Only ``supporting_facts`` is read. The gold ``answer`` field is never
    consulted, and the emitted text is always original context, never the
    annotation string itself.

    Note on what D is and is not. Gold evidence for a question of the form
    "which item is X" must name the item, so an evidence oracle necessarily
    contains the answer -- 87.5% of demo-suite and 92.2% of LCR-dev annotations
    do. D is therefore an upper bound on *what any selector could retrieve*,
    computed with privileged information. It is not a model-capability ceiling,
    and it is only a retrieval ceiling if it actually scores above the other
    refined conditions -- which is measured here, not assumed.

    ``mode`` is ``"exact"`` where the dataset quotes facts verbatim and
    ``"lexical"`` where it abstracts them.
    """
    if not answerable:
        return None
    if not chunks or not supporting_facts:
        return None

    if mode == "exact":
        if context is None:
            raise ValueError("exact oracle localisation needs the original context")
        hits = [localise_fact_exact(chunks, f, context) for f in supporting_facts]
    else:
        hits = [localise_fact(chunks, f) for f in supporting_facts]

    if any(cid < 0 or conf < min_confidence for cid, conf in hits):
        return None

    core = {cid for cid, _ in hits}
    chunk_ids = _expand_and_cap(chunks, core)
    return _finalise(
        "oracle",
        chunks,
        chunk_ids,
        anchor,
        count_tokens,
        {
            "evidence_chunk_ids": sorted(core),
            "localisation_mode": mode,
            "localisation_confidence": [round(c, 4) for _, c in hits],
            "n_facts": len(supporting_facts),
        },
    )

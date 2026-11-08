"""Deterministic, semantically inert filler prose.

Everything here is invented for this lab. No sentence is copied from any corpus in
the repository; the generator is a combinatorial template engine over five made-up
clerical domains (survey logs, depot manifests, observatory notes, guild ledgers,
transit schedules).

Two invariants make the filler safe as a positional carrier, and both are tested:

1. **No digits.** Every quantity is spelled out or omitted. Evidence identifiers,
   codes, timestamps and counts all carry digits, so a digit-free filler can never
   reproduce one by accident.
2. **Lexical disjointness.** The filler vocabulary shares no content word with the
   entity banks in :mod:`rineya_psa.benchmark.families`. A per-item reserved-token
   check re-verifies this at generation time against the terms actually drawn.

Paragraph size is a stable sentence count (3-6, default 3) so that a token budget can
be met by choosing a chunk count. Exactly one "ballast" chunk per document is allowed
a different sentence count inside the same 3-6 range; that is the fine-grained knob
that lets the budget land inside +/-2 percent (see ``assemble.fit_bank``).
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from rineya_psa.types import stable_id

KIND_FILLER = "filler"
KIND_DISTRACTOR = "distractor"
EVIDENCE_KIND_PREFIX = "evidence:"

MIN_SENTENCES = 3
MAX_SENTENCES = 6

# Words too common to be diagnostic of anything. Excluded from the reserved-token
# check so that shared function words do not make every chunk look like a collision.
# Written as prose and split rather than as a list literal: the point of the set is
# that a reader can scan it, and 120 quoted strings would defeat that.
FUNCTION_WORDS: frozenset[str] = frozenset(
    """
    about above after against also although always among another any are around
    because been before being below beside besides between both but came come
    could does done down during each either else enough even ever every from
    further had has have here hers hides him his how into itself just kept least
    less like made make many might more most much must near neither next none nor
    not now often once only onto other others ought over own past perhaps rather
    same shall should since some still such than that the their them then there
    these they this those though through thus together too under until upon very
    was were what when where whether which while who whom whose will with within
    without would yet your
    """.split()  # noqa: SIM905 - readability beats a 120-element literal
)

# ---------------------------------------------------------------------------
# Chunk container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """One paragraph of the assembled document.

    ``kinds[i]`` labels ``sentences[i]`` as filler, a distractor, or the k-th
    evidence sentence (``"evidence:0"``, ``"evidence:1"``, ...). Keeping the label on
    the sentence rather than on the chunk is what lets a chunk travel through a
    re-ordering while the analysis still knows which fact it carries.
    """

    chunk_id: str
    domain: str
    sentences: tuple[str, ...]
    kinds: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.sentences) != len(self.kinds):
            raise ValueError(
                f"chunk {self.chunk_id}: {len(self.sentences)} sentences vs "
                f"{len(self.kinds)} kinds"
            )
        if not self.sentences:
            raise ValueError(f"chunk {self.chunk_id}: empty")

    @property
    def filler_sentences(self) -> tuple[str, ...]:
        return tuple(
            s for s, k in zip(self.sentences, self.kinds, strict=True) if k == KIND_FILLER
        )

    @property
    def carries_evidence(self) -> bool:
        return any(k.startswith(EVIDENCE_KIND_PREFIX) for k in self.kinds)

    def evidence_slot(self) -> int | None:
        for k in self.kinds:
            if k.startswith(EVIDENCE_KIND_PREFIX):
                return int(k[len(EVIDENCE_KIND_PREFIX) :])
        return None

    def sentence_index_of_kind(self, kind: str) -> int:
        for i, k in enumerate(self.kinds):
            if k == kind:
                return i
        raise KeyError(f"chunk {self.chunk_id} has no sentence of kind {kind!r}")

    def with_sentence(self, sentence: str, kind: str, index: int) -> Chunk:
        sentences = list(self.sentences)
        kinds = list(self.kinds)
        sentences.insert(index, sentence)
        kinds.insert(index, kind)
        return Chunk(self.chunk_id, self.domain, tuple(sentences), tuple(kinds))


def evidence_kind(slot: int) -> str:
    return f"{EVIDENCE_KIND_PREFIX}{slot}"


def local_slot_index(chunk: Chunk, local_position: str) -> int:
    """Insertion index inside a paragraph for ``first`` / ``middle`` / ``last``.

    ``middle`` is the floor of half the current sentence count, which for the default
    3-sentence paragraph puts the inserted sentence second of four.
    """
    n = len(chunk.sentences)
    if local_position in ("first",):
        return 0
    if local_position in ("last",):
        return n
    if local_position in ("middle", "na"):
        return n // 2
    raise ValueError(f"unknown local_position {local_position!r}")


def insert_sentence(chunk: Chunk, sentence: str, kind: str, local_position: str) -> Chunk:
    return chunk.with_sentence(sentence, kind, local_slot_index(chunk, local_position))


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

DOMAINS: tuple[str, ...] = (
    "survey",
    "depot",
    "observatory",
    "guild",
    "transit",
)

DOMAIN_LABELS: dict[str, str] = {
    "survey": "field-survey log",
    "depot": "depot manifest",
    "observatory": "observatory notebook",
    "guild": "guild ledger",
    "transit": "transit timetable",
}

_DOMAIN_NOUNS: dict[str, tuple[str, ...]] = {
    "survey": (
        "baseline offset",
        "gradient reading",
        "control marker",
        "vertical datum",
        "chain interval",
        "azimuth check",
        "levelling staff",
        "traverse loop",
        "backsight pair",
        "closure residual",
    ),
    "depot": (
        "pallet rotation",
        "shelf clearance",
        "loading window",
        "carton grade",
        "stacking order",
        "dispatch lane",
        "returns bay",
        "handling class",
        "wrapping stock",
        "forklift roster",
    ),
    "observatory": (
        "seeing quality",
        "dome rotation",
        "focus drift",
        "cloud fraction",
        "guiding lock",
        "exposure cadence",
        "dew heater",
        "mirror cover",
        "filter wheel",
        "tracking rate",
    ),
    "guild": (
        "dues cycle",
        "apprentice hours",
        "bench allocation",
        "workshop rota",
        "materials draw",
        "audit column",
        "membership roll",
        "sundry account",
        "bursary fund",
        "attendance figure",
    ),
    "transit": (
        "headway plan",
        "layover margin",
        "platform assignment",
        "service band",
        "connection window",
        "running allowance",
        "turnback move",
        "boarding order",
        "relief crew",
        "depot balance",
    ),
}

_OPENERS: dict[str, tuple[str, ...]] = {
    "survey": (
        "This part of the field book covers routine traverse work and carries no exceptions.",
        "The pages below hold ordinary levelling material kept only for completeness.",
        "What follows is a plain transcription of the daily worksheet, with nothing disputed.",
    ),
    "depot": (
        "The following block is standard stock movement kept for continuity of the manifest.",
        "These lines describe ordinary warehouse handling with no held or flagged goods.",
        "The paragraph below repeats the customary loading material for an uneventful shift.",
    ),
    "observatory": (
        "The observing book continues here with ordinary conditions and no unusual events.",
        "These remarks cover routine dome and enclosure housekeeping and nothing further.",
        "The passage below is a plain summary of sky conditions with nothing set aside.",
    ),
    "guild": (
        "This part of the guild book covers ordinary administration and settles nothing.",
        "The lines that follow are customary bookkeeping without any contested figure.",
        "Below is a routine passage of guild housekeeping kept for the sake of continuity.",
    ),
    "transit": (
        "The timetable material below covers ordinary running with no service alteration.",
        "This passage lists customary scheduling detail and reports no disruption at all.",
        "The following lines are plain operating material for an unremarkable period.",
    ),
}

_PERIODS: tuple[str, ...] = (
    "the opening interval",
    "the second interval",
    "the middle interval",
    "the closing interval",
    "the overnight window",
    "the weekly cycle",
    "the quiet spell",
    "the handover period",
)

_QUALIFIERS: tuple[str, ...] = (
    "within the usual band",
    "inside the ordinary margin",
    "at the expected level",
    "close to the running mean",
    "no wider than before",
    "well inside the working range",
)

_PREDICATES: tuple[str, ...] = (
    "was left unchanged",
    "was carried forward",
    "was reviewed and kept",
    "was folded into the summary",
    "was marked routine",
    "was passed over without comment",
    "was held at its previous setting",
    "was mentioned only in passing",
)

# Every skeleton is a flat statement about two administrative quantities. None of
# them asserts an action by an identifiable agent, and none of them retracts or
# amends anything, which keeps the filler from imitating the state-constraint family.
_SKELETONS: tuple[str, ...] = (
    "The {a} for {period} {pred}, and the {b} stayed {qual}.",
    "Nothing in {period} altered the {a}, so the {b} {pred}.",
    "A short annotation observed that the {a} and the {b} continued to agree {qual}.",
    "Because {period} produced no exception, the {a} {pred} and the {b} was left open.",
    "The {a} was compared against the {b}, and both figures sat {qual}.",
    "Routine handling of the {a} during {period} required no adjustment to the {b}.",
    "The summary for {period} lists the {a} first and the {b} afterwards, with no comment.",
    "Neither the {a} nor the {b} moved far enough during {period} to justify a closer look.",
    "Following {period}, the {a} {pred} while the {b} remained {qual}.",
    "The working file keeps the {a} beside the {b} so that later intervals compare easily.",
    "The {a} appears in the same column as the {b}, which keeps the layout easy to scan.",
    "Handling material for {period} treats the {a} and the {b} as ordinary with no condition.",
)


def filler_vocabulary() -> frozenset[str]:
    """Every content token the generator can ever emit.

    Used by the disjointness test rather than at runtime; the runtime guard is the
    per-item reserved-token check, which is strictly narrower and cheaper.
    """
    pieces: list[str] = []
    for bank in _DOMAIN_NOUNS.values():
        pieces.extend(bank)
    for bank in _OPENERS.values():
        pieces.extend(bank)
    pieces.extend(_PERIODS)
    pieces.extend(_QUALIFIERS)
    pieces.extend(_PREDICATES)
    pieces.extend(s.replace("{a}", "").replace("{b}", "") for s in _SKELETONS)
    pieces.extend(DOMAINS)
    pieces.extend(DOMAIN_LABELS.values())
    return content_tokens(*pieces)


# ---------------------------------------------------------------------------
# Reserved-token machinery
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z]+")
_DIGIT_RE = re.compile(r"[0-9]")


def content_tokens(*texts: str) -> frozenset[str]:
    """Lower-cased alphabetic tokens of length >= 4 that are not function words.

    Length 4 is the cutoff because every entity word in ``families`` is at least four
    letters long by construction, while most accidental overlaps ("day", "log") are
    shorter.
    """
    out: set[str] = set()
    for text in texts:
        for word in _WORD_RE.findall(str(text)):
            low = word.casefold()
            if len(low) >= 4 and low not in FUNCTION_WORDS:
                out.add(low)
    return frozenset(out)


def reserved_tokens(terms: Iterable[str]) -> frozenset[str]:
    """Tokens the filler must not contain, derived from the terms an item actually drew."""
    return content_tokens(*terms)


def has_digits(text: str) -> bool:
    return _DIGIT_RE.search(text) is not None


def chunk_is_inert(chunk: Chunk, reserved: frozenset[str]) -> bool:
    """True when no filler sentence in the chunk touches a reserved term or a digit."""
    for sentence, kind in zip(chunk.sentences, chunk.kinds, strict=True):
        if kind != KIND_FILLER:
            continue
        if has_digits(sentence):
            return False
        if content_tokens(sentence) & reserved:
            return False
    return True


def collisions(chunk: Chunk, reserved: frozenset[str]) -> frozenset[str]:
    tokens: set[str] = set()
    for sentence, kind in zip(chunk.sentences, chunk.kinds, strict=True):
        if kind == KIND_FILLER:
            tokens |= content_tokens(sentence)
    return frozenset(tokens & reserved)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

_MAX_RESAMPLE = 64


def _sentence(rng: random.Random, domain: str) -> str:
    nouns = _DOMAIN_NOUNS[domain]
    a, b = rng.sample(nouns, 2)
    return rng.choice(_SKELETONS).format(
        a=a,
        b=b,
        period=rng.choice(_PERIODS),
        qual=rng.choice(_QUALIFIERS),
        pred=rng.choice(_PREDICATES),
    )


def _build_chunk(
    index: int,
    domain: str,
    n_sentences: int,
    rng: random.Random,
    reserved: frozenset[str],
    used: set[str],
    bank_seed: int,
) -> Chunk:
    sentences: list[str] = []
    opener_pool = _OPENERS[domain]
    for slot in range(n_sentences):
        for _ in range(_MAX_RESAMPLE):
            text = rng.choice(opener_pool) if slot == 0 else _sentence(rng, domain)
            if text in used:
                continue
            if has_digits(text):  # defensive; the banks contain no digits
                continue
            if content_tokens(text) & reserved:
                continue
            used.add(text)
            sentences.append(text)
            break
        else:
            raise ValueError(
                f"filler generator exhausted {_MAX_RESAMPLE} attempts for domain "
                f"{domain!r}; reserved terms are too broad: "
                f"{sorted(reserved)[:12]}"
            )
        if slot == 0:
            # Openers are a small pool; allow reuse across chunks after the first
            # document-wide use so long documents do not starve.
            used.discard(text)
    return Chunk(
        chunk_id=stable_id("chunk", bank_seed, index),
        domain=domain,
        sentences=tuple(sentences),
        kinds=tuple([KIND_FILLER] * n_sentences),
    )


def generate_chunks(
    n_chunks: int,
    *,
    seed: int,
    sentences_per_chunk: int = 3,
    ballast_sentences: int | None = None,
    reserved: frozenset[str] = frozenset(),
    domains: Sequence[str] = DOMAINS,
) -> tuple[Chunk, ...]:
    """Generate ``n_chunks`` inert paragraphs.

    ``ballast_sentences`` overrides the sentence count of the final chunk only; it is
    the fine token-budget knob. It must still lie in ``[MIN_SENTENCES, MAX_SENTENCES]``.
    """
    if n_chunks < 1:
        raise ValueError(f"n_chunks must be >= 1, got {n_chunks}")
    if not MIN_SENTENCES <= sentences_per_chunk <= MAX_SENTENCES:
        raise ValueError(
            f"sentences_per_chunk must lie in [{MIN_SENTENCES},{MAX_SENTENCES}], "
            f"got {sentences_per_chunk}"
        )
    if ballast_sentences is not None and not (
        MIN_SENTENCES <= ballast_sentences <= MAX_SENTENCES
    ):
        raise ValueError(
            f"ballast_sentences must lie in [{MIN_SENTENCES},{MAX_SENTENCES}], "
            f"got {ballast_sentences}"
        )
    if not domains:
        raise ValueError("at least one domain is required")

    rng = random.Random(seed)
    used: set[str] = set()
    rotation = rng.randrange(len(domains))
    chunks: list[Chunk] = []
    for i in range(n_chunks):
        domain = domains[(i + rotation) % len(domains)]
        n_sent = (
            ballast_sentences
            if (ballast_sentences is not None and i == n_chunks - 1)
            else sentences_per_chunk
        )
        chunks.append(_build_chunk(i, domain, n_sent, rng, reserved, used, seed))
    return tuple(chunks)


def filler_fingerprint(chunks: Sequence[Chunk]) -> str:
    """Order-independent hash of the filler multiset.

    Two positional variants of one scenario must produce the same fingerprint; that is
    the machine-checkable form of the matched-construction rule.
    """
    lines = sorted(s for c in chunks for s in c.filler_sentences)
    return stable_id("filler-multiset", len(lines), "\n".join(lines))

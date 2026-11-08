"""The four question families.

Each builder returns ``(evidence_sentences, distractor_sentences, question, answer,
aliases, meta)`` as a :class:`FamilyExample`, which is a ``NamedTuple`` so the tuple
form in the brief and attribute access both work.

Design constraints that the rest of the benchmark leans on:

* every evidence item is exactly **one sentence**, so it can be inserted into a host
  paragraph at a controlled local slot;
* the answer is a short single-line string that survives
  :func:`rineya_psa.types.normalize_answer` unchanged, so exact-match scoring works;
* every entity word is drawn from banks that are lexically disjoint from the filler
  vocabulary (``tests/test_benchmark_generation.py`` enforces this), and every
  identifier, timestamp and count contains a digit, which the digit-free filler
  cannot reproduce;
* ``meta["reserved_terms"]`` lists exactly the entity strings the example drew, which
  is what the runtime inertness check is run against.

DEVIATION: ``types.BenchmarkItem.target_positions`` is annotated "1 or 2 for
multi-hop". ``state_constraint`` needs at least three distributed updates by
definition, so its items carry one target per evidence sentence (three or four).
``meta["decisive_index"]`` marks which of them is the correction being swept, and
``meta["primary_target"]`` repeats that value for callers that want a single scalar.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from rineya_psa.benchmark.filler import content_tokens
from rineya_psa.types import FAMILIES, normalize_answer


class FamilyExample(NamedTuple):
    """The six-tuple the brief specifies, with names attached."""

    evidence_sentences: tuple[str, ...]
    distractor_sentences: tuple[str, ...]
    question: str
    answer: str
    aliases: tuple[str, ...]
    meta: Mapping[str, Any]


# ---------------------------------------------------------------------------
# Entity banks. Word-disjoint from filler and, within reason, from each other.
# ---------------------------------------------------------------------------

ROLES: tuple[str, ...] = (
    "harbour archivist",
    "tide-gate steward",
    "kiln inspector",
    "lantern warden",
    "orchard bailiff",
    "quarry factor",
    "ferry reeve",
    "saltworks keeper",
    "vellum binder",
    "beacon marshal",
    "loom overseer",
    "cellar reckoner",
)

OBJECTS: tuple[str, ...] = (
    "brass alidade rod",
    "copper sighting plate",
    "ivory scale rule",
    "slate weather dial",
    "amber lens cap",
    "iron plumb bob",
    "walnut specimen case",
    "phial rack",
    "bronze counterweight",
    "leather map tube",
    "pewter inkwell",
    "cedar sample dish",
)

PLACES: tuple[str, ...] = (
    "the north annexe",
    "the lower cistern",
    "the stone byre",
    "the flint undercroft",
    "the willow jetty",
    "the ash granary",
    "the pitch works",
    "the reed causeway",
    "the elm counting room",
    "the birch scriptorium",
)

TASKS: tuple[str, ...] = (
    "night calibration",
    "damp inspection",
    "annual recount",
    "hearth sweep",
    "brine weighing",
    "roof gauging",
    "ink assay",
    "wax sealing",
    "rope proofing",
    "flue testing",
)

UNITS: tuple[str, ...] = (
    "crates",
    "reels",
    "sacks",
    "drums",
    "punnets",
    "bales",
)

LABEL_KINDS: tuple[str, ...] = (
    "verification code",
    "strongbox key",
    "seal number",
    "docket cipher",
    "release token",
    "tally mark",
)

CARRIERS: tuple[str, ...] = (
    "consignment",
    "canister",
    "hamper",
    "spool",
    "casket",
    "portfolio",
)

LEDGERS: tuple[str, ...] = (
    "custody sheet",
    "index slip",
    "register strip",
    "deposit chit",
    "holding card",
    "escrow docket",
)

_ID_LETTERS = "BCDFGHJKLMNPQRSTVWXZ"
_ID_DIGITS = "23456789"


def entity_banks() -> tuple[tuple[str, ...], ...]:
    return (ROLES, OBJECTS, PLACES, TASKS, UNITS, LABEL_KINDS, CARRIERS, LEDGERS)


def entity_tokens() -> frozenset[str]:
    """Every content token that can appear as an entity in any family."""
    flat: list[str] = []
    for bank in entity_banks():
        flat.extend(bank)
    return content_tokens(*flat)


# ---------------------------------------------------------------------------
# Small deterministic helpers
# ---------------------------------------------------------------------------


def _code(rng: random.Random, groups: Sequence[int] = (2, 4, 3)) -> str:
    """Nonce identifier such as ``QX-4417-TLM``. Always contains digits."""
    parts: list[str] = []
    for gi, size in enumerate(groups):
        alphabet = _ID_DIGITS if gi % 2 == 1 else _ID_LETTERS
        parts.append("".join(rng.choice(alphabet) for _ in range(size)))
    return "-".join(parts)


# Answers are reproduced by the model verbatim, so they stay short: "QX-4417" is
# 7 characters and 3-4 tokens. 20 letters x 8 digits gives 1.6M distinct codes, so
# collision with any of the ~9 distractor codes in one item is negligible.
_ANSWER_GROUPS: tuple[int, ...] = (2, 4)


def _uuidish(rng: random.Random) -> str:
    hexd = "0123456789abcdef"
    blocks = (8, 4, 4, 12)
    return "-".join("".join(rng.choice(hexd) for _ in range(n)) for n in blocks)


def _times(rng: random.Random, n: int) -> tuple[str, ...]:
    """``n`` ascending wall-clock stamps inside one working day."""
    start = rng.randrange(6 * 60, 8 * 60)
    out: list[str] = []
    t = start
    for _ in range(n):
        out.append(f"{t // 60:02d}:{t % 60:02d}")
        t += rng.randrange(45, 150)
    return tuple(out)


def _bare(role_or_place: str) -> str:
    """Article-free surface form, a legitimate alternative answer spelling."""
    for prefix in ("the ", "a ", "an "):
        if role_or_place.startswith(prefix):
            return role_or_place[len(prefix) :]
    return role_or_place


def _check_answer(answer: str) -> str:
    if "\n" in answer or not answer.strip():
        raise ValueError(f"answer must be a non-empty single line, got {answer!r}")
    if normalize_answer(answer) != normalize_answer(normalize_answer(answer)):
        raise ValueError(f"answer {answer!r} is not stable under normalisation")
    return answer


def _terms(*groups: object) -> tuple[str, ...]:
    out: list[str] = []
    for g in groups:
        if isinstance(g, str):
            out.append(g)
        elif isinstance(g, (list, tuple)):
            out.extend(str(x) for x in g)
        else:
            out.append(str(g))
    return tuple(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# Family A - exact retrieval
# ---------------------------------------------------------------------------


def build_exact_retrieval(rng: random.Random, *, n_distractors: int = 8) -> FamilyExample:
    """Nonce key lookup. Minimal reasoning: find the line, copy the token.

    The *lookup key* is long and may be UUID-like, because that is what makes the
    search hard. The *answer* is deliberately short (SPEC §13.4): the calibration
    pilot found the model returning ``902eb5c6-...`` for gold ``402eb5c6-...``, a
    single wrong hex character scoring identically to retrieving the wrong record
    entirely. That conflates transcription failure with the positional retrieval
    failure this study is about, so the reproduced string is kept to one short group.
    """
    label = rng.choice(LABEL_KINDS)
    carrier = rng.choice(CARRIERS)
    use_uuid = rng.random() < 0.35
    subject_id = _uuidish(rng) if use_uuid else _code(rng)
    answer_code = _code(rng, _ANSWER_GROUPS)

    evidence = (f"The {label} for {carrier} {subject_id} is {answer_code}.",)

    other_carriers = [c for c in CARRIERS if c != carrier]
    distractors: list[str] = []
    for i in range(n_distractors):
        other_id = _uuidish(rng) if use_uuid else _code(rng)
        other_code = _code(rng, _ANSWER_GROUPS)
        if i % 3 == 0:
            c = rng.choice(other_carriers)
            distractors.append(f"The {label} for {c} {other_id} is {other_code}.")
        elif i % 3 == 1:
            other_label = rng.choice([x for x in LABEL_KINDS if x != label])
            distractors.append(
                f"The {other_label} for {carrier} {other_id} is {other_code}."
            )
        else:
            distractors.append(
                f"A withdrawn draft listed {other_code} against {carrier} {other_id}, "
                "and that draft was never adopted."
            )

    question = f"What is the {label} for {carrier} {subject_id}?"
    answer = _check_answer(answer_code)
    meta = {
        # Names the LOOKUP KEY format; the answer is always a short code now.
        "subtype": "uuid_key" if use_uuid else "nonce_key",
        "answer_chars": len(answer_code),
        "label_kind": label,
        "carrier": carrier,
        "subject_id": subject_id,
        "n_evidence": 1,
        "decisive_index": 0,
        "reserved_terms": list(_terms(label, carrier, subject_id, answer_code)),
        "reasoning_hops": 1,
    }
    return FamilyExample(evidence, tuple(distractors), question, answer, (), meta)


# ---------------------------------------------------------------------------
# Family B - semantic single hop
# ---------------------------------------------------------------------------


def build_semantic_single_hop(
    rng: random.Random, *, n_distractors: int = 8
) -> FamilyExample:
    """One natural statement answers the question; the distractors are near misses."""
    task = rng.choice(TASKS)
    place = rng.choice(PLACES)
    role = rng.choice(ROLES)

    evidence = (f"The {task} at {place} was completed by the {role}.",)

    other_tasks = [t for t in TASKS if t != task]
    other_places = [p for p in PLACES if p != place]
    other_roles = [r for r in ROLES if r != role]
    distractors: list[str] = []
    for i in range(n_distractors):
        if i % 4 == 0:
            distractors.append(
                f"The {rng.choice(other_tasks)} at {place} was completed by "
                f"the {rng.choice(other_roles)}."
            )
        elif i % 4 == 1:
            distractors.append(
                f"The {task} at {rng.choice(other_places)} was completed by "
                f"the {rng.choice(other_roles)}."
            )
        elif i % 4 == 2:
            distractors.append(
                f"The {rng.choice(other_roles)} offered to take on the {task} at "
                f"{place} but was assigned elsewhere."
            )
        else:
            distractors.append(
                f"An early plan gave the {task} at {place} to the "
                f"{rng.choice(other_roles)}; the plan was replaced before any work began."
            )

    question = f"Who completed the {task} at {place}?"
    answer = _check_answer(f"the {role}")
    meta = {
        "subtype": "agent_of_task",
        "task": task,
        "place": place,
        "role": role,
        "n_evidence": 1,
        "decisive_index": 0,
        "reserved_terms": list(_terms(task, place, role, *other_roles[:0])),
        "reasoning_hops": 1,
    }
    return FamilyExample(
        evidence, tuple(distractors), question, answer, (_bare(role),), meta
    )


# ---------------------------------------------------------------------------
# Family C - multi hop
# ---------------------------------------------------------------------------

MULTI_HOP_ANCHORS: dict[str, float] = {"start": 0.02, "middle": 0.50, "end": 0.98}

MULTI_HOP_POSITION_PAIRS: tuple[tuple[str, str], ...] = (
    ("start", "start"),
    ("start", "middle"),
    ("start", "end"),
    ("middle", "middle"),
    ("middle", "end"),
    ("end", "end"),
)


def multi_hop_targets(pair: tuple[str, str]) -> tuple[float, float]:
    return (MULTI_HOP_ANCHORS[pair[0]], MULTI_HOP_ANCHORS[pair[1]])


def build_multi_hop(
    rng: random.Random, *, n_distractors: int = 8, guided: bool = False
) -> FamilyExample:
    """Two facts, neither sufficient alone: item -> ledger reference -> custodian.

    ``guided=True`` is the tractable variant required by SPEC §13.3. The calibration
    pilot scored the hard variant 0/36 on both strict and containment matching, at
    NLL 4.4-5.1 — a floor, against which no intervention could be shown to help or
    hurt, which would make the brief's §41 multi-hop criterion unmeasurable.

    Guiding changes the *signposting*, not the structure: it is still two hops
    through the same bridge reference, and the bridge is still the only path from the
    question to the answer. What changes is that each evidence sentence names the
    other explicitly ("see the custody note for ..."), the question names the route,
    and the distractor count drops. The hard variant is retained and both are
    reported, so the difficulty manipulation is itself measurable.
    """
    ledger = rng.choice(LEDGERS)
    ref = _code(rng)
    role = rng.choice(ROLES)
    obj = rng.choice(OBJECTS)
    if guided:
        n_distractors = min(n_distractors, 3)

    if guided:
        hop_a = (
            f"Custody note for {ledger} {ref}: the entry filed as {ledger} {ref} "
            f"was placed in the keeping of the {role}."
        )
        hop_b = (
            f"The item recorded under {ledger} {ref} is the {obj}; "
            f"for its keeper see the custody note for {ledger} {ref}."
        )
    else:
        hop_a = f"The entry filed as {ledger} {ref} was placed in the keeping of the {role}."
        hop_b = f"The item recorded under {ledger} {ref} is the {obj}."
    evidence = (hop_a, hop_b)

    other_roles = [r for r in ROLES if r != role]
    other_objects = [o for o in OBJECTS if o != obj]
    distractors: list[str] = []
    for i in range(n_distractors):
        other_ref = _code(rng)
        if i % 3 == 0:
            distractors.append(
                f"The entry filed as {ledger} {other_ref} was placed in the keeping of "
                f"the {rng.choice(other_roles)}."
            )
        elif i % 3 == 1:
            distractors.append(
                f"The item recorded under {ledger} {other_ref} is the "
                f"{rng.choice(other_objects)}."
            )
        else:
            other_ledger = rng.choice([x for x in LEDGERS if x != ledger])
            distractors.append(
                f"A cross-reference from {other_ledger} {other_ref} points at the "
                f"{rng.choice(other_objects)} and names no keeper."
            )

    question = (
        f"Look up the ledger reference recorded for the {obj}, then use that "
        f"reference to say who has it in keeping. Who has the {obj} in keeping?"
        if guided
        else f"Who has the {obj} in keeping?"
    )
    answer = _check_answer(f"the {role}")
    meta = {
        "subtype": "reference_chain_guided" if guided else "reference_chain",
        "guided": guided,
        "ledger": ledger,
        "reference": ref,
        "role": role,
        "object": obj,
        "n_evidence": 2,
        "decisive_index": 1,
        "hop_order": ["custodian_of_reference", "item_of_reference"],
        "reserved_terms": list(_terms(ledger, ref, role, obj)),
        "reasoning_hops": 2,
    }
    return FamilyExample(
        evidence, tuple(distractors), question, answer, (_bare(role),), meta
    )


# ---------------------------------------------------------------------------
# Family D - state / constraint tracking
# ---------------------------------------------------------------------------


def _state_custody(rng: random.Random, n_distractors: int) -> FamilyExample:
    obj = rng.choice(OBJECTS)
    r1, r2, r3 = rng.sample(ROLES, 3)
    t1, t2, t3, t4 = _times(rng, 4)

    updates = (
        f"At {t1} the {obj} was signed over to the {r1}.",
        f"At {t2} the {r1} passed the {obj} to the {r2}.",
        f"At {t3} the {r2} handed the {obj} to the {r3}.",
        f"A later note posted at {t4} voids the transfer timed {t3}; "
        f"the {obj} never left the {r2}.",
    )
    question = f"Who held the {obj} at the close of the day?"
    answer = _check_answer(f"the {r2}")

    other_objects = [o for o in OBJECTS if o != obj]
    other_roles = [r for r in ROLES if r not in (r1, r2, r3)]
    distractors: list[str] = []
    for i in range(n_distractors):
        ot = _times(rng, 1)[0]
        if i % 3 == 0:
            distractors.append(
                f"At {ot} the {rng.choice(other_objects)} was signed over to "
                f"the {rng.choice(other_roles)}."
            )
        elif i % 3 == 1:
            distractors.append(
                f"A proposal timed {ot} would have moved the {obj} to the "
                f"{rng.choice(other_roles)}, but it was never signed."
            )
        else:
            distractors.append(
                f"At {ot} the {rng.choice(other_roles)} countersigned a receipt for the "
                f"{rng.choice(other_objects)} and kept no further part."
            )

    meta = {
        "subtype": "custody_chain",
        "object": obj,
        "holders": [r1, r2, r3],
        "final_holder": r2,
        "times": [t1, t2, t3, t4],
        "n_evidence": 4,
        "decisive_index": 3,
        "n_updates": 4,
        "correction_overrides": t3,
        "reserved_terms": list(_terms(obj, r1, r2, r3, t1, t2, t3, t4)),
        "reasoning_hops": 4,
    }
    return FamilyExample(
        updates, tuple(distractors), question, answer, (_bare(f"the {r2}"),), meta
    )


def _state_inventory(rng: random.Random, n_distractors: int) -> FamilyExample:
    place = rng.choice(PLACES)
    unit = rng.choice(UNITS)
    t1, t2, t3, t4 = _times(rng, 4)
    n0 = rng.randrange(30, 91)
    n_in = rng.randrange(12, 31)
    n_in_corrected = rng.randrange(3, n_in - 2)
    n_out = rng.randrange(4, 16)
    total = n0 + n_in_corrected - n_out
    naive = n0 + n_in - n_out
    if total <= 0 or total == naive:  # pragma: no cover - ranges make this impossible
        raise ValueError("degenerate inventory draw")

    updates = (
        f"At {t1} the standing figure for {place} was {n0} {unit}.",
        f"At {t2} a further {n_in} {unit} were moved into {place}.",
        f"At {t3} the entry timed {t2} was amended: only {n_in_corrected} "
        f"{unit} were actually moved.",
        f"At {t4} exactly {n_out} {unit} were taken out of {place}.",
    )
    question = f"How many {unit} were in {place} after {t4}?"
    answer = _check_answer(str(total))

    other_places = [p for p in PLACES if p != place]
    other_units = [u for u in UNITS if u != unit]
    distractors: list[str] = []
    for i in range(n_distractors):
        ot = _times(rng, 1)[0]
        if i % 3 == 0:
            distractors.append(
                f"At {ot} the standing figure for {rng.choice(other_places)} was "
                f"{rng.randrange(20, 120)} {rng.choice(other_units)}."
            )
        elif i % 3 == 1:
            distractors.append(
                f"A draft timed {ot} suggested moving {rng.randrange(5, 40)} "
                f"{rng.choice(other_units)} into {place}, but the move was not carried out."
            )
        else:
            distractors.append(
                f"At {ot} a count of {rng.randrange(10, 90)} {rng.choice(other_units)} "
                f"was posted against {rng.choice(other_places)} and closed the same day."
            )

    meta = {
        "subtype": "inventory_count",
        "place": place,
        "unit": unit,
        "opening": n0,
        "moved_in_claimed": n_in,
        "moved_in_actual": n_in_corrected,
        "moved_out": n_out,
        "naive_answer": str(naive),
        "times": [t1, t2, t3, t4],
        "n_evidence": 4,
        "decisive_index": 2,
        "n_updates": 4,
        "correction_overrides": t2,
        "reserved_terms": list(
            _terms(place, unit, n0, n_in, n_in_corrected, n_out, total, t1, t2, t3, t4)
        ),
        "reasoning_hops": 4,
    }
    return FamilyExample(
        updates, tuple(distractors), question, answer, (f"{total} {unit}",), meta
    )


def build_state_constraint(rng: random.Random, *, n_distractors: int = 9) -> FamilyExample:
    """Four distributed updates where a later correction overrides an earlier one.

    Two sub-types alternate by draw: a custody chain (answer is a role) and an
    inventory count (answer is a number). Both need every update: dropping any one of
    them, or reading the updates in text order instead of stamp order, gives a
    different answer.
    """
    if rng.random() < 0.5:
        return _state_custody(rng, n_distractors)
    return _state_inventory(rng, n_distractors)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_BUILDERS = {
    "exact_retrieval": build_exact_retrieval,
    "semantic_single_hop": build_semantic_single_hop,
    "multi_hop": build_multi_hop,
    "state_constraint": build_state_constraint,
}
assert tuple(_BUILDERS) == FAMILIES, "family dispatch drifted from types.FAMILIES"


def build_family(family: str, rng: random.Random, **kwargs: Any) -> FamilyExample:
    try:
        builder = _BUILDERS[family]
    except KeyError:
        raise KeyError(f"unknown family {family!r}; known: {sorted(_BUILDERS)}") from None
    example = builder(rng, **kwargs)
    _validate(family, example)
    return example


def _validate(family: str, example: FamilyExample) -> None:
    if not example.evidence_sentences:
        raise ValueError(f"{family}: no evidence sentences")
    for sentence in example.evidence_sentences + example.distractor_sentences:
        if "\n" in sentence:
            raise ValueError(f"{family}: multi-line sentence {sentence!r}")
        if '"' in sentence or "\\" in sentence:
            raise ValueError(f"{family}: quote/backslash breaks the json format layout")
    if len(set(example.evidence_sentences)) != len(example.evidence_sentences):
        raise ValueError(f"{family}: duplicate evidence sentences would alias spans")
    if set(example.evidence_sentences) & set(example.distractor_sentences):
        raise ValueError(f"{family}: a distractor duplicates an evidence sentence")
    if "reserved_terms" not in example.meta:
        raise ValueError(f"{family}: meta must carry reserved_terms")
    if "decisive_index" not in example.meta:
        raise ValueError(f"{family}: meta must carry decisive_index")


def n_evidence(family: str) -> int:
    """How many independently positioned evidence sentences a family emits."""
    return {
        "exact_retrieval": 1,
        "semantic_single_hop": 1,
        "multi_hop": 2,
        "state_constraint": 4,
    }[family]

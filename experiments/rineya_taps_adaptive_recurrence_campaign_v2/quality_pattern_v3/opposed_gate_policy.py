"""Three fixed prompt/probability-only K0/K1 rules; no model or labels."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
import re

CONFIDENT = "CONFIDENT_RECURRENCE"
UNCERTAIN = "UNCERTAIN_RECURRENCE"
GUARD = "EVIDENCE_CONSISTENCY_GUARD"
POLICIES = (CONFIDENT, UNCERTAIN, GUARD)
THRESHOLD = 0.5
HEADER = "DOCUMENTS: use only A-approved records"
RULE = ". Equal approved values give that value; absent/conflicting values give UNKNOWN."
LATEST = "; latest approved date per entity"
INTEGER = r"(?:0|-?[1-9][0-9]*)"
UNDATED_RECORD = re.compile(r"([A-Z]{3}) ([AD]) (" + INTEGER + r")")
DATED_RECORD = re.compile(r"([A-Z]{3}) ([AD]) d([1-9][0-9]*) (" + INTEGER + r")")
QUESTION = re.compile(r"(?:Question: ([A-Z]{3})\?|Query: ([A-Z]{3})\.) Reply one integer or UNKNOWN\.")


@dataclass(frozen=True)
class Decision:
    k: int
    recognized: bool
    certificate: str


def document_certificate(raw_prompt: str) -> str:
    """Strict complete user-content parsing; never return a solved value."""
    if type(raw_prompt) is not str:
        raise TypeError("raw user-content string required")
    lines = raw_prompt.split("\n")
    if len(lines) != 3:
        return "UNRECOGNIZED"
    header, records_line, question = lines
    if header == HEADER + RULE:
        dated = False
    elif header == HEADER + LATEST + RULE:
        dated = True
    else:
        return "UNRECOGNIZED"
    match = QUESTION.fullmatch(question)
    if match is None or not records_line.endswith("."):
        return "UNRECOGNIZED"
    query = match[1] or match[2]
    records = records_line[:-1].split(";")
    relevant = []
    for record in records:
        row = (DATED_RECORD if dated else UNDATED_RECORD).fullmatch(record)
        if row is None:
            return "UNRECOGNIZED"
        entity, authority = row[1], row[2]
        date, value = (row[3], row[4]) if dated else (None, row[3])
        if entity == query and authority == "A":
            relevant.append((date, value))
    if not relevant:
        return "ABSENT"
    if dated:
        # Positive canonical decimal strings have exact integer order by length
        # and lexicographic order; no floating conversion or integer-size limit.
        latest = max((date for date, _ in relevant), key=lambda x: (len(x), x))
        relevant = [(date, value) for date, value in relevant if date == latest]
    values = {value for _, value in relevant}
    return "CONSISTENT" if len(values) == 1 else "CONFLICT"


def decide(policy: str, raw_prompt: str, probability: float | None = None) -> Decision:
    """No metadata dictionaries, arbitrary kwargs, labels or response inputs."""
    if type(raw_prompt) is not str:
        raise TypeError("raw user-content string required")
    if policy not in POLICIES:
        raise ValueError("unregistered policy")
    if policy == GUARD:
        if probability is not None:
            raise ValueError("prompt guard has no probability input")
        certificate = document_certificate(raw_prompt)
        return Decision(int(certificate not in ("ABSENT", "CONFLICT")),
                        certificate != "UNRECOGNIZED", certificate)
    if type(probability) is not float or not math.isfinite(probability) or not 0. <= probability <= 1.:
        raise ValueError("finite original-input float probability in [0,1] required")
    confident = probability >= THRESHOLD
    return Decision(int(confident if policy == CONFIDENT else not confident), False,
                    "CONFIDENCE_RULE")


def header_cue(raw_prompt: str) -> Decision:
    if type(raw_prompt) is not str:
        raise TypeError("raw user-content string required")
    return Decision(int(not raw_prompt.split("\n", 1)[0].startswith("DOCUMENTS:")),
                    False, "HEADER_CUE")


def matched_random_decisions(rate: float, count: int, seed: int) -> tuple[int, ...]:
    """One frozen-order draw per item; rate is frozen without outcomes."""
    if type(rate) is not float or not math.isfinite(rate) or not 0. <= rate <= 1.:
        raise ValueError("finite float rate in [0,1] required")
    if type(count) is not int or count < 0 or type(seed) is not int or seed not in (17, 29):
        raise ValueError("registered count and seed17/29 required")
    rng = random.Random(seed)
    return tuple(int(rng.random() < rate) for _ in range(count))

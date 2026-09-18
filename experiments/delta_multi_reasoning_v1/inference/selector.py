"""The frozen selector: one interface, three registered variants, no gold anywhere.

S0  agreement/majority over candidate answers. **Zero extra model calls.**
S1  model selector WITHOUT checker findings. Diagnostic only, on a prospectively fixed subset.
S2  model selector WITH checker findings. **The main selector.**

S1 exists solely to isolate what the audit roles' findings are worth. It costs an extra call
per item, so it runs on a fixed subset and its calls are accounted separately -- the whole
experiment must not be described as five calls per item when part of it is six.

EVERY DEGENERATE CASE IS FROZEN HERE, BEFORE ANY RESULT IS SEEN
--------------------------------------------------------------
Ties, malformed candidates, malformed selector output, abstention and the empty candidate set
all have one written rule. They are the paths where a selector quietly becomes an oracle or a
coin flip, so none of them is decided at analysis time.

  empty candidate set      -> emit the registered abstention; flag `empty_candidate_set`
  all candidates malformed -> emit the registered abstention; flag `all_malformed`
  S0 tie                   -> the tied answer whose LOWEST-indexed role produced it
                              (deterministic, independent of dict order); flag `tie_broken`
  malformed selector output -> fall back to S0 over the same candidates; flag
                              `selector_malformed`. Falling back rather than scoring zero is
                              the deployable behaviour, and the flag keeps the failure visible.
  selector names a non-candidate -> treated as malformed, same fallback
  abstention as a candidate -> a legitimate answer, eligible for selection, never auto-ranked
                              up or down

The selector receives candidate answers and, for S2, checker findings. It never receives a
reference, a scorer verdict, task metadata, or an independent solver's answer. Candidate-oracle
accuracy is computed AFTERWARDS, in analysis, and gets no deployment credit.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from .roles import ABSTAIN

MALFORMED = "__MALFORMED__"


@dataclass(frozen=True)
class Candidate:
    """One role's contribution. `answer` is the PARSED answer; `raw` is kept for the record."""

    rid: str
    raw: str
    answer: str | None          # None == unparseable
    finding: str | None = None  # only audit roles produce one; never shown to S1

    @property
    def malformed(self) -> bool:
        return self.answer is None

    @property
    def abstained(self) -> bool:
        return self.answer is not None and self.answer.strip() == ABSTAIN


@dataclass
class Selection:
    """What the selector did, with every degenerate path flagged rather than smoothed."""

    selected_rid: str | None
    answer: str
    variant: str
    extra_model_calls: int = 0
    flags: dict[str, bool] = field(default_factory=dict)
    agreement: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"selected_rid": self.selected_rid, "answer": self.answer,
                "variant": self.variant, "extra_model_calls": self.extra_model_calls,
                "flags": {k: v for k, v in self.flags.items() if v},
                "agreement": self.agreement}


def _ordered(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Role order fixes every tie break. Sorting by rid makes it independent of call order."""
    return sorted(candidates, key=lambda c: c.rid)


def _abstain(variant: str, flag: str) -> Selection:
    return Selection(selected_rid=None, answer=ABSTAIN, variant=variant,
                     flags={flag: True, "abstained": True})


def s0_agreement(candidates: Sequence[Candidate], variant: str = "S0") -> Selection:
    """Majority over parsed answers. No model call.

    Abstention is counted as an ordinary answer: a set of roles that all abstain agrees on
    abstention, which is the correct behaviour on an unanswerable item.
    """
    cands = _ordered(candidates)
    if not cands:
        return _abstain(variant, "empty_candidate_set")
    usable = [c for c in cands if not c.malformed]
    if not usable:
        return _abstain(variant, "all_malformed")

    counts = Counter(c.answer for c in usable)
    top = max(counts.values())
    winners = [a for a, n in counts.items() if n == top]
    flags: dict[str, bool] = {}
    if len(winners) > 1:
        flags["tie_broken"] = True
        # The tied answer produced by the lowest-indexed role. Deterministic.
        answer = next(c.answer for c in usable if c.answer in winners)
    else:
        answer = winners[0]
    if len(usable) < len(cands):
        flags["some_malformed"] = True
    chosen = next(c for c in usable if c.answer == answer)
    if chosen.abstained:
        flags["abstained"] = True
    return Selection(selected_rid=chosen.rid, answer=answer, variant=variant,
                     extra_model_calls=0, flags=flags,
                     agreement={str(a): n for a, n in counts.items()})


def model_selector(candidates: Sequence[Candidate],
                   generate: Callable[[list[dict], int], str],
                   prompt: list[dict],
                   include_checker_findings: bool,
                   variant: str,
                   max_new_tokens: int = 8) -> Selection:
    """S1 (findings withheld) or S2 (findings shown), on one common interface.

    `generate(messages, max_new_tokens) -> decoded text` is the native backend call. The
    selector must run on the SAME backend the roles ran on; a different model here would make
    the condition a two-model system rather than an inference condition.
    """
    from .roles import selector_prompt

    cands = _ordered(candidates)
    if not cands:
        return _abstain(variant, "empty_candidate_set")
    usable = [c for c in cands if not c.malformed]
    if not usable:
        return _abstain(variant, "all_malformed")

    payload = [{"id": c.rid, "answer": c.answer,
                **({"finding": c.finding} if include_checker_findings and c.finding else {})}
               for c in usable]
    text = generate(selector_prompt(prompt, payload, include_checker_findings), max_new_tokens)

    by_rid = {c.rid: c for c in usable}
    hits = [rid for rid in by_rid if rid in (text or "")]
    if len(hits) != 1:
        # No id, or several ids, is malformed. Fall back to S0 and keep the flag.
        fb = s0_agreement(cands, variant=variant)
        fb.extra_model_calls = 1
        fb.flags["selector_malformed"] = True
        fb.flags["selector_raw_kept"] = True
        return fb

    chosen = by_rid[hits[0]]
    flags: dict[str, bool] = {}
    if len(usable) < len(cands):
        flags["some_malformed"] = True
    if chosen.abstained:
        flags["abstained"] = True
    counts = Counter(c.answer for c in usable)
    return Selection(selected_rid=chosen.rid, answer=chosen.answer, variant=variant,
                     extra_model_calls=1, flags=flags,
                     agreement={str(a): n for a, n in counts.items()})


def select(candidates: Sequence[Candidate], variant: str,
           generate: Callable[[list[dict], int], str] | None = None,
           prompt: list[dict] | None = None) -> Selection:
    """Dispatch to the registered variant. The one entry point REP and MR both use."""
    if variant == "S0":
        return s0_agreement(candidates)
    if variant in ("S1", "S2"):
        if generate is None or prompt is None:
            raise ValueError(f"{variant} needs a native generate() and the original prompt")
        return model_selector(candidates, generate, prompt,
                              include_checker_findings=(variant == "S2"), variant=variant)
    raise ValueError(f"unregistered selector variant {variant!r}; "
                     f"registered: S0, S1, S2")


def policy() -> dict:
    """The frozen policy, for the run receipt."""
    return {
        "variants": {
            "S0": "agreement/majority over parsed answers; 0 extra model calls",
            "S1": "model selector, checker findings WITHHELD; diagnostic subset only; 1 extra call",
            "S2": "model selector, checker findings SHOWN; MAIN selector; 1 extra call",
        },
        "main_selector": "S2",
        "selection_output": "a candidate ID, never a synthesised novel answer",
        "frozen_degenerate_paths": {
            "empty_candidate_set": "registered abstention, flagged",
            "all_candidates_malformed": "registered abstention, flagged",
            "s0_tie": "tied answer from the lowest-indexed role; deterministic",
            "malformed_selector_output": "fall back to S0, flagged selector_malformed",
            "selector_names_non_candidate": "treated as malformed, same fallback",
            "abstention_as_candidate": "an ordinary eligible answer, never re-ranked",
        },
        "label_isolation": ("the selector sees candidate answers and, for S2, checker findings; "
                            "never a reference, scorer verdict, task metadata or independent "
                            "solver output"),
        "oracle_note": ("candidate-oracle accuracy is computed afterwards in analysis and gets "
                        "no deployment credit"),
        "backend": "the selector runs on the same backend as the roles it selects among",
    }

"""Independent CPU certificate verifier over student-visible R/S messages.

The public contract parser is shared deliberately.  Reference binding uses
fixed-point terminal propagation; subset selection enumerates combinations.
Neither path imports a compiler, primary solver, model, or protected bank.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

try:
    from . import rs_contract as contract
except ImportError:  # Direct invocation with this package directory on sys.path.
    import rs_contract as contract


VERSION = "synth-rs-verifier-1"


def solve_visible(messages: Any) -> dict[str, Any]:
    """Enumerate every valid unordered certificate in a bounded visible task.

    Invalid messages raise ValueError under the public contract.  Unrequested
    aliases can remain unresolved: only requested aliases determine eligibility.
    A missing requested link or requested cycle is insufficient reference
    evidence; both lack a valid list and therefore permit the UNKNOWN sentinel.
    Detailed missing/cycle diagnosis belongs to the independent primary path.
    """
    spec = contract.parse_visible(messages)
    values = spec["initial"]
    eligible = set(range(len(values)))

    if "lookup" in spec:
        lookup = spec["lookup"]
        links = lookup["links"]
        resolved = {name: target for name, target in links.items()
                    if type(target) is int}
        # Each iteration propagates at least one terminal or reaches a fixed
        # point.  Cycles and missing targets never acquire a terminal index.
        while True:
            additions = {name: resolved[target] for name, target in links.items()
                         if name not in resolved and type(target) is str
                         and target in resolved}
            if not additions:
                break
            resolved.update(additions)
        if any(name not in resolved for name in lookup["requested"]):
            return {"status": "insufficient_evidence",
                    "reason": "unresolved_requested_reference", "solutions": []}
        eligible = {resolved[name] for name in lookup["requested"]}

    if "selection" not in spec:
        return {"status": "solved", "reason": "valid",
                "solutions": [sorted(eligible)]}

    selection = spec["selection"]
    solutions = [list(chosen) for chosen in combinations(sorted(eligible), selection["k"])
                 if sum(values[index] for index in chosen) == selection["target"]]
    return {"status": "solved" if solutions else "impossible",
            "reason": "valid" if solutions else "no_exact_subset",
            "solutions": sorted(solutions)}


def verify(messages: Any, candidate: str) -> dict[str, Any]:
    """Accept any valid certificate, or UNKNOWN exactly when none exists."""
    try:
        outcome = solve_visible(messages)
        # Length is a public schema property, never compiler metadata.
        item_count = len(contract.parse_visible(messages)["initial"])
    except (ValueError, TypeError, KeyError, IndexError, RecursionError, UnicodeError):
        return {"accepted": False, "reason": "invalid_prompt", "outcome": None}

    try:
        answer = contract.parse_answer(candidate, item_count)
    except (ValueError, TypeError, KeyError, IndexError, RecursionError, UnicodeError):
        return {"accepted": False, "reason": "invalid_answer", "outcome": outcome}

    accepted = (not outcome["solutions"] if answer == "UNKNOWN"
                else answer in outcome["solutions"])
    return {"accepted": accepted,
            "reason": "verified" if accepted else "incorrect_answer",
            "outcome": outcome}

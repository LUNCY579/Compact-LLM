"""Compiler reference path: direct alias walks and cardinality/sum DP.

This module never imports the independent verifier. Bounds come from the public
contract; no pruning assumptions about positive values are made.
"""
from __future__ import annotations

from rs_contract import validate_spec

VERSION = "synth-rs-walk-dp-1"


def solve(spec: dict) -> dict:
    sig = validate_spec(spec)
    eligible = list(range(len(spec["initial"])))
    if "R" in sig:
        eligible_set = set()
        links = spec["lookup"]["links"]
        for requested in spec["lookup"]["requested"]:
            current = requested
            visited = set()
            while type(current) is str:
                if current in visited:
                    return {"status": "insufficient_evidence", "reason": "requested_cycle", "solutions": []}
                if current not in links:
                    return {"status": "insufficient_evidence", "reason": "missing_requested_link", "solutions": []}
                visited.add(current)
                current = links[current]
            eligible_set.add(current)
        eligible = sorted(eligible_set)
    if "S" not in sig:
        return {"status": "solved", "reason": "valid", "solutions": [eligible]}
    k, target = spec["selection"]["k"], spec["selection"]["target"]
    # Each state retains every index witness. Descending cardinality ensures an
    # item is never consumed twice, including when its value is zero or negative.
    states = [{0: [()]}] + [{} for _ in range(k)]
    for index in eligible:
        for count in range(k, 0, -1):
            for total, witnesses in list(states[count - 1].items()):
                new_total = total + spec["initial"][index]
                states[count].setdefault(new_total, []).extend(w + (index,) for w in witnesses)
    solutions = [list(w) for w in sorted(states[k].get(target, []))]
    return {"status": "solved" if solutions else "impossible",
            "reason": "valid" if solutions else "no_exact_subset", "solutions": solutions}

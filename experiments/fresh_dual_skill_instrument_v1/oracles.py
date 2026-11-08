"""Oracles for the three transition families, reimplemented from the system prompts.

Validated by reproducing all 384 recorded `reference` strings in
rineya-dataset-retention-records/process-transition-probe-001/CANDIDATE_VIEWS.jsonl.
That collision is the only reason to trust them; they are not copied from any solver.
"""
from __future__ import annotations
import json

MOD = 97


def dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


def ledger_replay(before: dict, event: dict) -> str:
    bal = dict(before)
    if event.get("status") == "posted":
        kind, acct, amt = event["kind"], event["account"], event["amount"]
        if kind == "add":
            bal[acct] = bal[acct] + amt
        elif kind == "scale":
            bal[acct] = bal[acct] * amt
        elif kind == "transfer":
            other = event["other"]
            bal[acct] = bal[acct] - amt
            bal[other] = bal[other] + amt
        else:
            raise ValueError(f"unknown ledger kind {kind}")
    return dumps(bal)


def prerequisite_plan(before: int, event: dict) -> str:
    state = before
    pre_on, pre_off = event["pre_on"], event["pre_off"]
    applicable = (state & pre_on) == pre_on and (state & pre_off) == 0
    if applicable:
        state = (state & ~event["delete"]) | event["add"]
    return dumps(state)


def register_machine(before: dict, event: dict) -> str:
    reg = dict(before)
    kind, r = event["kind"], event["register"]
    other = event.get("other")
    val = event.get("value", 0)
    if kind == "swap":
        reg[r], reg[other] = reg[other], reg[r]
    elif kind == "add":
        reg[r] = (reg[r] + val) % MOD
    elif kind == "mix":
        reg[r] = (reg[r] + 2 * reg[other] + val) % MOD
    elif kind == "conditional":
        delta = val if reg[other] > event["threshold"] else -val
        reg[r] = (reg[r] + delta) % MOD
    else:
        raise ValueError(f"unknown register kind {kind}")
    return dumps(reg)


ORACLES = {
    "ledger_replay": lambda u: ledger_replay(u["before"], u["event"]),
    "prerequisite_plan": lambda u: prerequisite_plan(u["before"], u["event"]),
    "register_machine": lambda u: register_machine(u["before"], u["event"]),
}

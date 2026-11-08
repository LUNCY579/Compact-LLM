"""PARENTQ qualification rows, built by the frozen GEO section H constructors (no model).

Every row comes from the frozen `benchmark._transition_h` / `benchmark._answerability_h`
(GEO FREEZE_SECTIONH.json), with fresh operands, fresh worlds and a registered seed.
- T block: 6 rows for each of the screen's four T subtypes, inside T's trained support
  with no modulo-97 wrap, ranges bracketing the screen row of that subtype.
- H block ON: the screen's H template under the HSKILL answerable contract (the screen's
  own system prompt), 12 answerable, 10 distractor and 10 cue rows.
- H block OFF: the same 32 user texts under H's own abstention training prompts (A or B,
  verbatim from delta_H's training rows, balanced within each row kind).
The arm-identity canary rows are built separately (build_parentq.py) from the exposed
HSKILL_VIEWS_V2 bank and CONFIRM-002's stored predictions.
"""
from __future__ import annotations

from dataclasses import replace
import random

SPLIT = "parentq"
ROW_SEED = 20261002
T_SUBTYPES = ("ledger_replay:transfer", "ledger_replay:scale", "register_machine:add",
              "register_machine:conditional")
T_PER_SUBTYPE = 6
H_KINDS = {"answerable": 12, "distractor": 10, "cue": 10}
# Fresh worlds: none occurs as a word in any GEO screen, development, legacy, format or
# conjunction row (checked by build_parentq.validate, with delta_H's training prompts).
WORLDS = ("Aspen", "Cypress", "Elm", "Fir", "Hemlock", "Holly", "Laurel", "Linden", "Oak", "Olive",
          "Tamarack", "Poplar", "Spruce", "Sumac", "Walnut", "Yew", "Acacia", "Banyan", "Catalpa", "Dogwood",
          "Ebony", "Ginkgo", "Hickory", "Ironwood", "Jacaranda", "Kapok", "Locust", "Magnolia", "Mahogany",
          "Mulberry", "Quince", "Redwood")
# delta_H's only abstention-gold training prompts: every one of the 56 exact
# "I don't know based on the available information." replies sits under one of these two
# (28 each, compose001-20260918/H/H-prepared/microfit-train.jsonl, sha256 cb3df333...).
H_NATIVE_PROMPTS = {
    "native_a": ("Solve the problem silently and give only the concise final answer. Use only the supplied "
                 "facts; do not show chain-of-thought or invent missing information."),
    "native_b": "Use the stated facts to reach the result. Return only the compact conclusion, not the reasoning steps.",
}
H_TRAINING_ROWS_SHA256 = "cb3df33343056ca51e26616d27ea18b84306f411a4491be7439de8338cdb71a8"


def subtype(row) -> str:
    """The screen subtype of a T row, or the kind of an H row."""
    if row.task_kind == "transition_parent":
        return f"{row.metadata['transition_family']}:{row.metadata['event_kind']}"
    return row.metadata["h_row_kind"]


def _ledger(rng: random.Random, kind: str, tick: int) -> dict:
    """Balances -30..40 (the screen's ledger rows use -27..37); amount inside T's -9..12."""
    accounts = ["A", "B", "C", "D"]
    before = {name: rng.randint(-30, 40) for name in accounts}
    account, other = rng.sample(accounts, 2)
    amount = rng.randint(1, 12) if kind == "transfer" else rng.randint(2, 4)   # screen: transfer 8, scale 3
    return {"before": before, "event": {"account": account, "amount": amount, "kind": kind, "other": other,
                                        "status": "posted", "tick": tick}}


def _register(rng: random.Random, kind: str, tick: int) -> dict:
    """Values 0..96, operand inside T's -8..11 (never 0), and no modulo-97 wrap."""
    names = ["X", "Y", "Z"]
    while True:
        before = {name: rng.randint(0, 96) for name in names}
        register, other = rng.sample(names, 2)
        value = rng.choice([v for v in range(-8, 12) if v != 0])
        threshold = rng.randint(0, 96)
        if kind == "add":
            raw = before[register] + value
        else:
            raw = before[register] + (value if before[other] > threshold else -value)
        if 0 <= raw < 97:
            return {"before": before, "event": {"kind": kind, "other": other, "register": register,
                                                "threshold": threshold, "tick": tick, "value": value}}


def build_t_rows(benchmark, rng: random.Random) -> list:
    rows, index = [], 0
    for _ in range(T_PER_SUBTYPE):
        for name in T_SUBTYPES:
            family, kind = name.split(":")
            index += 1
            tick = rng.randint(1, 99)
            payload = _ledger(rng, kind, tick) if family == "ledger_replay" else _register(rng, kind, tick)
            rows.append(benchmark._transition_h(SPLIT, index, family, payload))
    return rows


def build_h_rows(benchmark, rng: random.Random) -> list:
    worlds = list(WORLDS)
    rng.shuffle(worlds)
    kinds = [kind for kind, count in H_KINDS.items() for _ in range(count)]
    rows = []
    for world, kind in zip(worlds, kinds, strict=True):
        number = rng.choice([n for n in range(2, 97) if n != 23])
        rows.append(benchmark._answerability_h(SPLIT, world, kind, number))
    return rows


def native_variants(benchmark, h_rows) -> list:
    """The same user texts under H's own abstention training prompts, A/B alternating per kind."""
    seen, out = {}, []
    for row in h_rows:
        kind = row.metadata["h_row_kind"]
        label = ("native_a", "native_b")[seen.get(kind, 0) % 2]
        seen[kind] = seen.get(kind, 0) + 1
        out.append(replace(row, row_id=f"{row.row_id}-{label}", system=H_NATIVE_PROMPTS[label],
                           metadata={**row.metadata, "contract": f"h_training_prompt_{label}",
                                     "on_row_id": row.row_id}))
    return out


def build_bank(benchmark) -> dict:
    """Deterministic: the same frozen constructors and seed always give the same rows."""
    rng = random.Random(ROW_SEED)
    t_rows = build_t_rows(benchmark, rng)
    h_on = build_h_rows(benchmark, rng)
    order = random.Random(ROW_SEED + 1)
    order.shuffle(t_rows)
    order.shuffle(h_on)
    return {"T": tuple(t_rows), "H_ON": tuple(h_on), "H_OFF": tuple(native_variants(benchmark, h_on))}


def all_rows(bank: dict) -> tuple:
    return (*bank["T"], *bank["H_ON"], *bank["H_OFF"])

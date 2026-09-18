"""Build DMRF-SCOUT-001: a small OPEN-DEVELOPMENT bank with independently checkable answers.

WHY A NEW BANK
--------------
The MR / REP / selector / routing cells cannot run on anything that already exists here:

* SELECT-VIEWS-001 (384 views) is selection-admissible but covers **acquisition only** -- no
  arithmetic, no multi-step, no long context, no unanswerable rows, and no ambiguous or
  shortcut cases for a router.
* The 256-row Long-Context Demonstration Suite is `SPENT` -- `QUARANTINE.json` bars it for
  TRAIN, SELECTION and CONFIRMATION -- and the 64 abstention rows live inside it and are not
  independent of its score.
* The MRS lane already spent a scout whose checkpoints sat **at or below the constant-answer
  baseline**, which made every arm contrast on it uninterpretable.

EVERY ANSWER IS COMPUTED BY CODE IN THIS FILE, never by a language model. The generator draws
a world, simulates it, and records the result plus the oracle that produced it.

A DESIGN ERROR THIS FILE'S OWN FLOOR COMPUTATION CAUGHT, RECORDED BECAUSE IT IS THE POINT
------------------------------------------------------------------------------------------
The first version scored unanswerable long-context rows as their **own stratum**. Its
answer-agnostic floor came out at **1.000**: when every reference is `UNKNOWN`, a model that
abstains unconditionally scores 100% and the stratum discriminates nothing. That is the MRS
failure reproduced in miniature, and it was caught before a single model call.

The fix is the structure the real suite already uses: answerable and unanswerable rows live in
**one scored pool**, so unconditional abstention is penalised by the answerable rows. Abstention
is then reported as sub-analyses of that pool -- correct abstention, false abstention -- and
never as a stratum with its own floor.

The first version also put router cases in separate strata, which left only 16 items each and
let `copy_before` reach 0.375 on one of them by sampling variance. Router cases are now
**flags on ordinary items** (`router_case`, `router_expected`), which is what the protocol
actually asks for -- ambiguous requests and shortcut checks present in the bank -- and it keeps
four strata of 32 CHECK items instead of eight of 16.

STRUCTURE
---------
A **world** is one scenario. Each world emits exactly ONE item, so the world is the resampling
unit for every confidence interval; calls, tokens and paraphrases are not independent units.

Sizes frozen HERE, before any CHECK outcome: **24 CAL worlds, 128 CHECK worlds**, four strata,
so 6 CAL and 32 CHECK items per stratum. Sized for the inference budget -- roughly 11 calls per
item per system after sharing R0 between O and MR, four systems, ~180 items -- not for
per-stratum power. Strata this small are registered as **exploratory**.

PERMISSION: permanently OPEN DEVELOPMENT / SELECTION-EXPOSED. It may support prompt, role,
router and selector decisions on CAL. It may never confirm anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

CAL_WORLDS = 24
CHECK_WORLDS = 128
SEED = 20260918
ABSTAIN = "UNKNOWN"

STRATA = ("transition", "multi_step", "arith_competence", "longctx_mixed")

# Fraction of drawn events that are `held` (do not apply). Kept low so `copy_before` is a weak
# predictor, but non-zero because detecting "this event does not apply" is exactly the
# operation MR's premise-audit role R2 exists to perform.
HELD_RATE = 0.125
# Unanswerable share of the long-context pool. Matches the real suite's 64/256.
UNANSWERABLE_SHARE = 0.25

ACCOUNTS = ("A", "B", "C", "D")
CITIES = ("Lindholm", "Varna", "Kettering", "Otsu", "Mirabel", "Zaragoza")
GOODS = ("valves", "ballast", "rotors", "gaskets", "spindles", "couplings")

# max_context is 4096 tokens in the qualified harness. The largest prompt this bank must
# survive is the MR selector on a long-context item: the document, four candidate answers and
# the selector instruction. Documents are capped so that fits with room to spare -- no arm may
# silently drop evidence or shorten a long-context problem.
DOC_TARGET_CHARS = 2400
SELECTOR_RESERVE_CHARS = 900


def h(*parts: object) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# --------------------------------------------------------------------------------------- #
# World and oracles
# --------------------------------------------------------------------------------------- #

def _draw_event(rng: random.Random) -> dict:
    """One event, with the degenerate self-transfer excluded.

    A posted `transfer` with account == other subtracts and re-adds the same amount to the same
    account, so the state is unchanged and `copy_before` scores it correctly. That handed the
    first build a 0.25 copy_before floor on the transition stratum -- the drawn held rate is
    only 0.125, and self-transfers supplied the rest (1/3 x 1/4 = 0.083). Excluding them is not
    a convenience: a self-transfer is a degenerate instruction, not an interesting gated case,
    and the gated cases R2 exists to detect are the `held` ones, which are retained.
    """
    op = rng.choice(("add", "scale", "transfer"))
    account = rng.choice(ACCOUNTS)
    other = account
    if op == "transfer":
        other = rng.choice([x for x in ACCOUNTS if x != account])
    return {"op": op, "account": account, "other": other,
            "amount": rng.choice((-5, -3, -2, 2, 3, 4, 5)),
            "status": "held" if rng.random() < HELD_RATE else "posted"}


def make_world(rng: random.Random, wid: str) -> dict:
    return {
        "world_id": wid,
        "balances": {a: rng.randint(-20, 20) for a in ACCOUNTS},
        "events": [_draw_event(rng) for _ in range(4)],
        "city": rng.choice(CITIES),
        "good": rng.choice(GOODS),
        "units_shipped": rng.randint(120, 900),
        "defect_count": rng.randint(3, 40),
        "shift_start_min": rng.randrange(0, 1440, 15),
        "shift_len_min": rng.choice((75, 90, 135, 150, 210, 225)),
        "crew": rng.randint(4, 19),
    }


def apply_event(state: dict, ev: dict) -> dict:
    """The transition oracle. A non-posted event leaves every balance unchanged."""
    out = dict(state)
    if ev["status"] != "posted":
        return out
    a, o, amt = ev["account"], ev["other"], ev["amount"]
    if ev["op"] == "add":
        out[a] += amt
    elif ev["op"] == "scale":
        out[a] *= amt
    elif ev["op"] == "transfer":
        out[a] -= amt
        out[o] += amt
    return out


# The transition system prompt is SELECT-VIEWS-001's, verbatim, including the "simultaneously"
# clause and the closing output contract. Matching it is not cosmetic: the first CAL pass used
# a paraphrase without the contract line and both cores scored 0/6 on this stratum while
# reasoning correctly, because nothing asked them for a bare JSON object.
RULES = ("Apply one event to the supplied account balances. If status is not posted, leave all "
         "balances unchanged. add: add amount to account. scale: multiply account by amount. "
         "transfer: subtract amount from account and add it to other, simultaneously. "
         "Return only the complete resulting balances as a JSON object.")
RULES_MULTI = ("Apply each event to the supplied account balances, in order. If an event's "
               "status is not posted, leave all balances unchanged for that event. add: add "
               "amount to account. scale: multiply account by amount. transfer: subtract "
               "amount from account and add it to other, simultaneously. Return only the "
               "complete resulting balances as a JSON object.")
# Output contracts for the other strata, in the same shape: say exactly what to return.
CONTRACT_ARITH = "Return only the final answer."
CONTRACT_LONGCTX = ("Answer using only the document. Return only the answer. If the document "
                    f"does not state the answer, return exactly {ABSTAIN}.")


def ev_obj(ev: dict, tick: int = 1) -> dict:
    """The event encoding SELECT-VIEWS-001 uses: `kind` not `op`, and a tick index."""
    o = {"account": ev["account"], "amount": ev["amount"], "kind": ev["op"],
         "status": ev["status"], "tick": tick}
    if ev["op"] == "transfer":
        o["other"] = ev["other"]
    return o


FILLER = (
    "The {city} depot logs {good} movements at the end of each shift.",
    "Reconciliation is performed by the duty supervisor before the log is closed.",
    "Pallets of {good} are counted twice when the seal is broken in transit.",
    "Held entries remain in the queue until a supervisor releases them.",
    "The ledger is append-only; corrections are entered as new lines.",
    "Crews rotate between the {city} floor and the outer yard on alternate days.",
    "Yard scales are calibrated weekly and the certificate is filed with the log.",
    "Entries that fail validation are annotated but not removed.",
    "The night shift hands over a written summary to the morning supervisor.",
    "Seasonal demand shifts the balance of inbound and outbound lines.",
)


def build_doc(rng: random.Random, w: dict, needle: str | None) -> str:
    """Distractor prose with the needle (if any) at a random interior position."""
    lines = [rng.choice(FILLER).format(**w) for _ in range(26)]
    if needle is not None:
        lines.insert(rng.randrange(4, len(lines) - 3), needle)
    doc = " ".join(lines)
    while len(doc) < DOC_TARGET_CHARS:
        doc += " " + rng.choice(FILLER).format(**w)
    return doc[:DOC_TARGET_CHARS + 200]


# --------------------------------------------------------------------------------------- #
# Item builders
# --------------------------------------------------------------------------------------- #

def item_transition(rng, w, idx):
    """One gated state transition. Some items carry a router flag."""
    ev = w["events"][0]
    after = apply_event(w["balances"], ev)
    core = canon({"before": w["balances"], "event": ev_obj(ev)})
    # 1 in 8 is a shortcut check: a long document whose real task is the transition. A
    # length-first router sends this to the general model and loses it.
    if idx % 8 == 3:
        case, body = "shortcut_long", f"{build_doc(rng, w, None)}\n\n{core}"
    # 1 in 8 is ambiguous: prose framing around a structured state.
    elif idx % 8 == 6:
        case, body = "ambiguous", (
            f"The {w['city']} depot handles {w['good']}. Overnight the duty supervisor recorded "
            f"one event against the balances below.\n{core}")
    else:
        case, body = "plain", core
    msgs = [{"role": "system", "content": RULES}, {"role": "user", "content": body}]
    return msgs, canon(after), "oracle.apply_event", {
        "steps": 1, "event_applies": ev["status"] == "posted",
        "copy_before_correct": canon(after) == canon(w["balances"]),
        "router_case": case, "router_expected": "small_specialist"}


#: Chained steps in the multi_step stratum. TWO, not three, and the reason is a CAL
#: measurement: at three events both cores scored 0/6 while emitting perfectly well-formed
#: JSON with the right keys -- genuine chain-arithmetic failure, not a format problem. A
#: stratum nobody can score is as uninterpretable as one an answer-agnostic predictor wins,
#: so it would have contributed nothing to an MR contrast. Two chained steps is still
#: genuinely multi-step -- the second step's input is the first step's output -- and at a
#: ~50% one-step rate it should land in a measurable band. Decided on CAL, before CHECK was
#: ever scored, and recorded here rather than left as a silent constant.
MULTI_STEP_EVENTS = 2


def _chain_events(rng: random.Random) -> list[dict]:
    """Two ops on ONE account, so the chain is real and exact match still has range.

    CAL measured the alternative. With two events drawn freely across four accounts, both cores
    scored 0/6 on exact match while emitting perfectly well-formed JSON -- and inspection showed
    three of four balances correct on several items, failing on exactly one. The stratum was
    informative and the scoring could not see it, which is a measurement-design failure rather
    than a capability finding.

    Constraining both ops to one account keeps what makes this multi-step -- the second op's
    input is the first op's output, and order matters, since scale-then-add differs from
    add-then-scale -- while leaving only one field to get right. `transfer` is excluded here
    because it touches two accounts and would reintroduce the spread. Decided on CAL, before
    CHECK was ever scored.
    """
    acct = rng.choice(ACCOUNTS)
    return [{"op": rng.choice(("add", "scale")), "account": acct, "other": acct,
             "amount": rng.choice((-5, -3, -2, 2, 3, 4, 5)),
             "status": "held" if rng.random() < HELD_RATE else "posted"}
            for _ in range(MULTI_STEP_EVENTS)]


def item_multi_step(rng, w, idx):
    """Chained transitions on one account; each step's input is the previous step's output."""
    st, evs = dict(w["balances"]), _chain_events(rng)
    for ev in evs:
        st = apply_event(st, ev)
    body = canon({"before": w["balances"],
                  "events": [ev_obj(e, i + 1) for i, e in enumerate(evs)]})
    msgs = [{"role": "system", "content": RULES_MULTI}, {"role": "user", "content": body}]
    return msgs, canon(st), "oracle.apply_event_chain", {
        "steps": 3, "n_applied": sum(1 for e in evs if e["status"] == "posted"),
        "copy_before_correct": canon(st) == canon(w["balances"]),
        "router_case": "plain", "router_expected": "small_specialist",
        "chained_steps": MULTI_STEP_EVENTS}


def item_arith(rng, w, idx):
    """Source competence, mirroring the 150-row panel's three classes."""
    # 1 in 8 is a shortcut check: short, no JSON -- surface says small -- but two dependent hops.
    if idx % 8 == 5:
        ref = str(w["crew"] * w["defect_count"])
        q = (f"Each of the {w['crew']} crew members at {w['city']} logged {w['defect_count']} "
             f"defective {w['good']} during the audit week. No other defects were logged. "
             f"How many defective units were logged in total?")
        return ([{"role": "system", "content": CONTRACT_ARITH},
                 {"role": "user", "content": q}], ref, "oracle.product",
                {"arith_class": "multi_hop", "steps": 2,
                 "router_case": "shortcut_short", "router_expected": "large_general"})
    kind = ("percentage", "ratio", "duration")[idx % 3]
    if kind == "percentage":
        ref = f"{round(w['defect_count'] * 1000 / w['units_shipped'], 2):.2f}"
        q = (f"{w['units_shipped']} units of {w['good']} shipped from {w['city']} and "
             f"{w['defect_count']} were defective. Defects per 1000 units, to two decimals?")
    elif kind == "ratio":
        ref = str(w["units_shipped"] // w["crew"])
        q = (f"{w['units_shipped']} units were handled by a crew of {w['crew']}. Units per "
             f"crew member, rounded down to a whole number?")
    else:
        end = (w["shift_start_min"] + w["shift_len_min"]) % 1440
        hh, mm = divmod(end, 60)
        ref = f"{hh % 12 or 12}:{mm:02d} {'AM' if hh < 12 else 'PM'}"
        q = (f"A shift starts at {w['shift_start_min'] // 60:02d}:{w['shift_start_min'] % 60:02d} "
             f"and runs {w['shift_len_min']} minutes. What clock time does it end, as h:mm AM/PM?")
    return ([{"role": "system", "content": CONTRACT_ARITH},
             {"role": "user", "content": q}], ref, f"oracle.arith.{kind}",
            {"arith_class": kind, "steps": 1,
             "router_case": "plain", "router_expected": "large_general"})


def item_longctx(rng, w, idx):
    """ONE pooled stratum: answerable and unanswerable rows scored together.

    Scoring these separately gave the unanswerable stratum a 1.000 floor, because
    unconditional abstention is perfect when every reference is UNKNOWN. Pooling is what makes
    abstention measurable rather than free.
    """
    unanswerable = (idx % 4 == 2)          # exactly 25%, deterministic in the index
    seal = 100000 + (int(h(w["world_id"], "seal")[:6], 16) % 899999)
    needle = (f"For the record, the {w['city']} depot seal number for this consignment of "
              f"{w['good']} is {seal}.")
    doc = build_doc(rng, w, None if unanswerable else needle)
    msgs = [{"role": "system", "content": CONTRACT_LONGCTX},
            {"role": "user", "content":
                f"{doc}\n\nWhat is the {w['city']} depot seal number for this consignment?"}]
    return msgs, (ABSTAIN if unanswerable else str(seal)), (
        "oracle.absent_by_construction" if unanswerable else "oracle.needle_seal"), {
        "needle_present": not unanswerable, "unanswerable": unanswerable,
        "doc_chars": len(doc), "steps": 1,
        "router_case": "plain", "router_expected": "large_general"}


BUILDERS = {"transition": item_transition, "multi_step": item_multi_step,
            "arith_competence": item_arith, "longctx_mixed": item_longctx}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite an existing scout: {a.out}")
    a.out.mkdir(parents=True)

    rows: list[dict] = []
    for split, n in (("CAL", CAL_WORLDS), ("CHECK", CHECK_WORLDS)):
        per = {s: 0 for s in STRATA}
        for i in range(n):
            wid = f"DMRFW-{split}-{i:04d}"
            rng = random.Random(h(SEED, wid))     # one RNG per world, seeded from its id
            w = make_world(rng, wid)
            stratum = STRATA[i % len(STRATA)]
            idx = per[stratum]
            per[stratum] += 1
            msgs, ref, oracle, extra = BUILDERS[stratum](rng, w, idx)
            pc = sum(len(m["content"]) for m in msgs)
            rows.append({
                "id": f"DMRF-{split}:{stratum}:{h(wid, stratum)[:16]}",
                "world_id": wid, "split": split, "stratum": stratum,
                "prompt": msgs, "reference": ref, "gold_verified_by": oracle,
                "prompt_sha256": h(canon(msgs)), "prompt_chars": pc,
                "max_selector_prompt_chars": pc + SELECTOR_RESERVE_CHARS,
                "view_role": "OPEN_DEVELOPMENT_SELECTION_EXPOSED", **extra,
            })

    (a.out / "SCOUT.jsonl").write_text("".join(canon(r) + "\n" for r in rows), encoding="utf-8")

    # --------------------------------------------------------------- answer-agnostic floors
    #
    # Floors are computed per split AND pooled over the whole bank. The POOLED floor is the one
    # the gate compares against, because the binding floor is a property of the item GENERATOR,
    # not of a split, and a per-split floor on 6 CAL items is an artifact: with n = 6,
    # `constant_majority` is mechanically >= 1/6 = 0.1667, so even a perfect 6/6 barely clears
    # its own lower bound and a genuinely good 3/6 cannot. Estimating the same quantity on 38
    # items per stratum is strictly better and does not depend on which split is being scored.
    floors: dict[str, dict] = {}
    for stratum in STRATA:
        for split in ("CAL", "CHECK", "POOLED"):
            sub = [r for r in rows if r["stratum"] == stratum
                   and (split == "POOLED" or r["split"] == split)]
            if not sub:
                continue
            counts: dict[str, int] = {}
            for r in sub:
                counts[r["reference"]] = counts.get(r["reference"], 0) + 1
            const_k = max(counts.values())
            copy_k = sum(1 for r in sub if r.get("copy_before_correct"))
            abst_k = sum(1 for r in sub if r["reference"] == ABSTAIN)
            binding = max(const_k, copy_k, abst_k)
            pred = ("constant_majority" if const_k == binding
                    else "copy_before" if copy_k == binding else "always_abstain")
            floors[f"{stratum}/{split}"] = {
                "n": len(sub), "distinct_references": len(counts),
                "constant_majority": const_k, "constant_majority_rate": round(const_k / len(sub), 4),
                "copy_before": copy_k, "copy_before_rate": round(copy_k / len(sub), 4),
                "always_abstain": abst_k, "always_abstain_rate": round(abst_k / len(sub), 4),
                "binding_floor_count": binding,
                "binding_floor_rate": round(binding / len(sub), 4),
                "binding_predictor": pred,
            }

    router_cases: dict[str, int] = {}
    for r in rows:
        if r["split"] == "CHECK":
            router_cases[r["router_case"]] = router_cases.get(r["router_case"], 0) + 1

    manifest = {
        "schema": 1, "record": "DMRF_SCOUT_001_BUILD", "instrument": "DMRF-SCOUT-001",
        "what_this_is": (
            "A small open-development bank whose answers are computed by the oracles in "
            "build_scout.py. Built because no existing instrument here covers acquisition, "
            "arithmetic, multi-step reasoning, long context, abstention AND the ambiguous and "
            "shortcut cases a router must be scored on."),
        "admissible_uses": ["SELECTION", "DEV"],
        "inadmissible_uses": ["CONFIRMATION"],
        "training_admitted": False,
        "why_not_confirmation": ("An instrument is spent for confirmation the moment it is used "
                                 "for selection. This one is built to be selected on."),
        "seed": SEED, "worlds": {"CAL": CAL_WORLDS, "CHECK": CHECK_WORLDS},
        "sizes_frozen_before_any_check_outcome": True,
        "items": len(rows), "one_item_per_world": True, "resampling_unit": "world",
        "strata": list(STRATA), "small_strata_are_exploratory": True,
        "held_event_rate": HELD_RATE, "unanswerable_share": UNANSWERABLE_SHARE,
        "abstention_sentinel": ABSTAIN,
        "abstention_is_not_a_separate_stratum": (
            "Answerable and unanswerable long-context rows are scored in ONE pool. Scored "
            "separately, the unanswerable stratum's answer-agnostic floor is 1.000 -- "
            "unconditional abstention is perfect when every reference is UNKNOWN. Pooling is "
            "what makes abstention measurable. Correct and false abstention are reported as "
            "sub-analyses of the pool, never as a stratum with its own floor."),
        "router_cases_in_CHECK": router_cases,
        "router_case_meaning": {
            "plain": "surface form and best component agree",
            "ambiguous": "prose framing around a structured state; genuinely either-way",
            "shortcut_long": "long document whose real task is a state transition; a "
                             "length-first router loses these",
            "shortcut_short": "short, no JSON, but needs two dependent hops; surface says "
                              "small, truth says large",
        },
        "context_budget": {
            "harness_max_context_tokens": 4096,
            "doc_target_chars": DOC_TARGET_CHARS,
            "selector_reserve_chars": SELECTOR_RESERVE_CHARS,
            "max_prompt_chars_in_bank": max(r["prompt_chars"] for r in rows),
            "max_selector_prompt_chars_in_bank": max(r["max_selector_prompt_chars"] for r in rows),
            "note": ("Documents are capped so the largest MR selector prompt fits inside "
                     "max_context. No arm may drop evidence or shorten a long-context problem."),
        },
        "floors": floors,
        "floor_rule": ("A system clears a stratum only if the exact two-sided binomial 95% "
                       "lower bound on its correct count exceeds that stratum's binding floor "
                       "rate, where the binding floor is the strongest of constant_majority, "
                       "copy_before and always_abstain."),
        "floors_published_before_any_model_call": True,
        "gate_before_use": ("NOT USABLE until floor_report.py confirms a real model clears the "
                            "binding floor on CAL."),
        "design_error_caught_by_this_floor_computation": (
            "v1 scored unanswerable rows as their own stratum and its floor came out 1.000; v1 "
            "also put router cases in separate strata of 16, letting copy_before reach 0.375 by "
            "sampling variance. Both were fixed before any model call. Recorded because the "
            "floor computation existing at all is the point."),
        "scout_sha256": h(canon(rows)), "model_calls": 0, "optimizer_steps": 0,
    }
    (a.out / "SCOUT_BUILD.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                            encoding="utf-8")
    print(json.dumps({
        "items": len(rows), "scout_sha256": manifest["scout_sha256"],
        "router_cases_in_CHECK": router_cases,
        "max_selector_prompt_chars": manifest["context_budget"]["max_selector_prompt_chars_in_bank"],
        "floors_CHECK": {k.split("/")[0]: {
            "n": v["n"], "binding_floor_rate": v["binding_floor_rate"],
            "binding_predictor": v["binding_predictor"],
            "distinct_references": v["distinct_references"]}
            for k, v in floors.items() if k.endswith("/CHECK")},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

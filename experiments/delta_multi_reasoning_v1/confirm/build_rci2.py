"""RCI-002 — a confirmation instrument with TWO pre-registered endpoints, built from RCI-001's
defects rather than from its design.

    python confirm/build_rci2.py --out <records>/rci002-20260918

Everything here exists because RCI-001 lost something specific.

    RCI-001 published four floors and not one READ THE CONTEXT. `last_candidate_in_context`
    scored 0.3711 against the published 0.2773 and 100% on two of five task types, because
    every correction was emitted after the thing it corrected. Found after the seal was spent.
        -> clause order and block order are RANDOMISED, extraction predictors are in the
           published set, and THE BUILD REFUSES TO SHIP if any stratum's shortcut floor
           exceeds --max-floor. The check is a gate, not a report.

    `temporal_correction` came back 78/78 for both systems and 78/78 for the recency rule --
    above the ceiling and at the extraction floor at once. A fifth of the answerable bank was
    decoration and the aggregate floor did not notice.
        -> per-stratum ceiling and floor are computed at build time.

    The largest effect in RCI-001 was invisible for a week because `wh`-form was confounded
    with `variant`. `rule_exception` asks "Which role..." in base and "Who..." in paraphrase;
    the candidate scored 25/25 on the first and 2/25 on the second. Two sessions proposed four
    mechanisms off the task-type marginal and all four were wrong.
        -> **wh-congruence is an explicit crossed factor**, and `paraphrase` now varies wording
           while holding the interrogative FIXED. What used to be an accident is a design.

    52 of 384 answerable rows turned on a leading definite article, and two sessions spent real
    effort reconciling numbers that never disagreed about the model.
        -> references are stored BARE, the normaliser is declared in the manifest, and
           convention sensitivity is reported.

## The 2x2 that discriminates the three surviving hypotheses

Same carve-out reasoning in all four cells: find the one role exempted from a rule.

| task type | interrogative | answer |
|---|---|---|
| `carve_out_role_congruent`   | "Which role ..."        | the ROLE   |
| `carve_out_role_incongruent` | "Who ..."               | the ROLE   |
| `carve_out_badge_congruent`  | "Which badge number ..."| the BADGE  |
| `carve_out_badge_incongruent`| "Whose badge ..."       | the BADGE  |

The incongruent cells type the answer as a PERSON while the context determines no person for
that role -- asserted at build time, not assumed. Predicted damage patterns:

| hypothesis | role/cong | role/incong | badge/cong | badge/incong |
|---|---|---|---|---|
| **wh-compliance** (answers the interrogative, not the context) | none | DAMAGE | none | DAMAGE |
| **role-occupant binding** | damage | damage | damage | damage |
| **output-type shift** (worse at entity spans) | damage | damage | none | none |

Three hypotheses, four cells, no two patterns alike.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "method"))
from floors import floors as gate_floors, immunity, require_pack_structure  # noqa: E402

SENTINEL = "UNKNOWN"
SYSTEM = ("You are answering a closed-context reading test. Use only the supplied context. "
          "Reply with the shortest exact answer and no explanation. If the context does not "
          f"determine the answer, reply {SENTINEL}.")

# --------------------------------------------------------------------------------------- #
# Entity pools. Partitioned before generation, and ALSO disjoint from RCI-001's pools -- an
# entity the candidate has already been asked about on a spent bank is not a fresh entity.
# --------------------------------------------------------------------------------------- #
RCI001_ENTITIES = {
    "Ardel Vance", "Noor Halliday", "Peregrine Oyelaran", "Sifra Lindqvist", "Tobias Renn",
    "Umeko Farrow", "Casimir Blythe", "Delphine Arana", "Imogen Castellanos", "Rashid Okonjo",
    "Beatrix Vollmer", "Hiran Devakumar", "Solveig Brandt", "Marcus Ttofi", "Yolanda Pais",
    "Kenji Aldarrow", "Freya Mbeki", "Aurelio Santangelo", "Nadia Kowalczyk", "Emeka Lindgren",
    "night steward", "records clerk", "visiting auditor", "duty warden", "assistant curator",
    "survey lead", "morning registrar", "conservation officer", "standing inspector",
    "gallery marshal", "acquisitions secretary", "field recorder",
    "ivory seal", "brass astrolabe", "lacquer casket", "vellum ledger", "copper orrery",
    "jade stylus", "onyx tablet", "silver quadrant", "walnut reliquary", "linen portfolio",
    "pewter censer", "garnet signet", "ebony caliper", "alabaster urn",
    "Cedar Vault", "the West Annexe", "the Glasshouse", "Reading Room Four",
    "the Under-Stair Store", "the Cartography Loft", "Larkspur Hold", "the North Undercroft",
    "the Orangery", "Consultation Room Nine", "the Belfry Store", "the Plan Chest Room",
    "Asterhall Institute", "Marrowgate Observatory", "the Thorne Repository",
    "Quillbourne Athenaeum", "Halvard Station", "the Pellingham Collection",
}

POOLS = {
    "CALIB": {
        "people": ["Anselm Dreyfus", "Ingrid Tamboli", "Oskar Benavides", "Wren Achterberg",
                   "Lucia Farrington", "Dmitri Oyelowo", "Saoirse Kandel", "Bertram Iwu"],
        "roles": ["stores supervisor", "reading-room usher", "loans coordinator",
                  "fabric warden", "print-room deputy", "estates liaison"],
        "objects": ["the cinnabar flask", "the maple theodolite", "the sable portfolio",
                    "the flint counterweight", "the zinc armature", "the porphyry weight",
                    "the obsidian rule", "the walrus-ivory comb", "the tallow lamp",
                    "the chalcedony bowl"],
        "places": ["Hollis Vestry", "the Lower Gallery", "the Seed Room", "Study Carrel Two",
                   "the Bell Passage", "the Map Loft", "the Kiln Room", "the Drying Loft"],
        "sites": ["Ravenscourt Lyceum", "Dunmarrow Field Station", "the Alderney Bequest"],
    },
    "SEALED": {
        "people": ["Cordelia Nkemdirim", "Bastian Kohlrausch", "Tamsin Ojukwu",
                   "Leopold Machiavelli", "Anouk Vasilenko", "Ravi Thorsteinsson",
                   "Mireille Baptiste", "Hektor Sandoval", "Perpetua Adeyemi",
                   "Gunnar Halloway", "Xiomara Pettigrew", "Aurick Nakamura"],
        "roles": ["dispatch registrar", "vellum conservator", "roving invigilator",
                  "cloister marshal", "bequests secretary", "topographic recorder"],
        "objects": ["the basalt tablet", "the nickel sextant", "the cherrywood casket",
                    "the hessian folio", "the tin thurible", "the amber seal-ring",
                    "the boxwood divider", "the serpentine ewer", "the gypsum medallion",
                    "the rattan coffer", "the slate palette", "the horn beaker"],
        "places": ["Wychelm Keep", "the South Crypt", "the Fernhouse",
                   "Consultation Room Three", "the Clock Loft", "the Chart Press Room",
                   "the Slip Room", "the Winding Stair"],
        "sites": ["Pentacost Athenaeum", "Vardenger Station", "the Ashcombe Bequest"],
    },
}

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

#: A SHARED badge pool, deliberately TINY, and the exempt badge is rotated through all of it.
#:
#: This number is load-bearing and was got wrong twice. The threat is not that a badge is
#: guessable -- it is that the answer may be the only member of its stratum answer-vocabulary
#: present in its own context, which makes `first_candidate_in_context` correct by default
#: without reading anything. With 20 badges the first draft measured 1.000; with the register
#: decoys added it was still 0.8667, because a decoy only helps if it is itself a possible
#: ANSWER somewhere in the bank. Six badges, rotated so every one of them is the reference in
#: some pack, means every badge in every context is a live candidate and position decides
#: nothing.
BADGE_POOL = [str(x) for x in (14, 27, 38, 46, 59, 63)]

#: (task type -> the interrogative class its questions use, and whether that class MATCHES the
#: entity class of the answer). `incongruent` means the interrogative types the answer as a
#: person while the context determines no person -- the factor RCI-001 varied by accident.
TASK_TYPES = {
    "carve_out_role_congruent":    {"answer": "ROLE",  "wh": "congruent",   "needs": "rule"},
    "carve_out_role_incongruent":  {"answer": "ROLE",  "wh": "incongruent", "needs": "rule"},
    "carve_out_badge_congruent":   {"answer": "BADGE", "wh": "congruent",   "needs": "rule"},
    "carve_out_badge_incongruent": {"answer": "BADGE", "wh": "incongruent", "needs": "rule"},
    "exclusion_count_number":      {"answer": "COUNT", "wh": "congruent",   "needs": "count"},
    "alias_resolution":            {"answer": "PERSON", "wh": "congruent",  "needs": "alias"},
    "object_state_tracking":       {"answer": "PLACE", "wh": "congruent",   "needs": "moves"},
    "temporal_correction":         {"answer": "DAY",   "wh": "congruent",   "needs": "temporal"},
}
VARIANTS = ("base", "paraphrase", "counterfactual", "unanswerable")

#: The two endpoints, fixed here and copied into the prereg verbatim.
ENDPOINTS = {
    "E1_abstention": {
        "rows": "every unanswerable row",
        "metric": "correct literal UNKNOWN",
        "why_convention_free": "a literal sentinel admits no article or punctuation ambiguity",
    },
    "E2_wh_incongruence": {
        "rows": "answerable rows of carve_out_role_incongruent and carve_out_badge_incongruent",
        "metric": "exact match under the declared normaliser",
        "why": "the factor that was confounded with `variant` in RCI-001 and carried its "
               "largest effect",
    },
}

FILLER = [
    "The account was compiled from duty books and is presented without editorial comment.",
    "Several pages carry marginal notes in a second hand; these are reproduced where legible.",
    "Routine movements of consumable stores are omitted as they were not logged individually.",
    "The building was open to visitors on three of the days covered by this account.",
    "Weather for the period was unremarkable and is noted only where it affected access.",
    "A lock was replaced during the period; the old key was retained in the strongbox.",
    "The compiler notes that nothing in this account should be read as an allegation.",
    "Page numbering restarts partway through and the original sequence is not recoverable.",
    "Two appendices referenced in the margin were not attached to the copy transcribed here.",
    "Costs were recorded separately and do not appear in this summary.",
]


def h(*p) -> str:
    return hashlib.sha256("|".join(str(x) for x in p).encode("utf-8")).hexdigest()


def canon(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"))


ARTICLE = re.compile(r"^(the|a|an)\s+")


def normalise(v: str) -> str:
    """THE declared scorer. Fixed here, copied into the manifest, and used by every consumer.

    Article-tolerant and punctuation-insensitive, first line only. RCI-001 scored `the morning
    registrar` against `morning registrar` as wrong and so measured article usage on a third of
    one task type; two sessions then reconciled ~50 rows by hand, twice.
    """
    s = str(v).strip().splitlines()[0] if str(v).strip() else ""
    s = " ".join(s.lower().replace(".", " ").replace(",", " ").split())
    return ARTICLE.sub("", s)


# --------------------------------------------------------------------------------------- #
# World
# --------------------------------------------------------------------------------------- #

def make_world(rng: random.Random, pool: dict, sid: str, density: str, idx: int) -> dict:
    people = rng.sample(pool["people"], 4)
    # The exempt role is ROTATED through the pool rather than sampled. Sampled, the modal
    # reference carried `per_stratum_majority` to 0.375 on the E2 endpoint at calibration n --
    # a guess with no reading. Rotation makes the reference distribution flat by construction.
    _exempt = pool["roles"][idx % len(pool["roles"])]
    _rest = rng.sample([r for r in pool["roles"] if r != _exempt], 3)
    roles = [_rest[0], _exempt, _rest[1], _rest[2]]     # slot 0 alias, slot 1 exempt
    objects = rng.sample(pool["objects"], min(9, len(pool["objects"])))
    # SIX places, split into two disjoint roles. The rule block names its place two or three
    # times; when that place could also be a movement endpoint it inflated the frequency of an
    # `object_state_tracking` answer and carried that stratum to a 0.625 floor.
    places = rng.sample(pool["places"], 8)
    move_places, rule_places = places[:4], places[4:8]
    site = rng.choice(pool["sites"])

    # Badges: every role carries one, so a badge answer needs the same carve-out reasoning as a
    # role answer and differs only in the entity class emitted.
    # Rotate BOTH the exempt role and its badge by pack index, so each is the reference in
    # roughly equal numbers of packs and the answer vocabulary covers the whole pool.
    exempt_badge = BADGE_POOL[idx % len(BADGE_POOL)]
    others = rng.sample([b for b in BADGE_POOL if b != exempt_badge], len(roles) - 1)
    # slot 1 is the exempt role, so the rotated badge goes there
    badge_choice = [others[0], exempt_badge] + others[1:]
    badges = dict(zip(roles, badge_choice))

    # Aliasing: exactly ONE role is bound to a person, and it is never the exempt role. This is
    # what makes the incongruent question determinate-but-person-free, and it is asserted later.
    # roles[0] is the alias role and carries the only person binding in the text; roles[1] is
    # the exempt role and never has an occupant. That separation is what makes an incongruent
    # question determinate, and it is asserted per pack in build_bank.
    alias_person, alias_role = people[0], roles[0]

    moves = []
    for ob in objects[:3]:
        frm, to = rng.sample(move_places, 2)
        moves.append({"object": ob, "from": frm, "to": to, "day": rng.choice(DAYS[:4]),
                      "by": rng.choice(people[1:]), "cancelled": rng.random() < 0.5})
    if all(m["cancelled"] for m in moves):
        moves[0]["cancelled"] = False
    if not any(m["cancelled"] for m in moves):
        moves[0]["cancelled"] = True
    tracked = rng.choice(moves)

    corr_obj = objects[3]
    stated_day, true_day = rng.sample(DAYS, 2)

    # ONE CARVE-OUT PER ROLE. Four rules, four places, four exempt roles, and the question
    # names which place it is asking about.
    #
    # This is the only construction that makes frequency uninformative rather than merely
    # noisy. Any entity named in an exception is mentioned once more than an entity that is
    # not, so with two carve-outs the answer and its decoy sat at 3 mentions against everyone
    # else's 2 and `most_frequent_candidate_in_context` reached 0.75 on the badge stratum. With
    # one exception per role every candidate has identical frequency, identical first position
    # and identical last position, and the only way to the answer is to match the place.
    # It also removes a ceiling: matching four carve-outs is a real task, where picking the
    # single exception in the text was 24/24 for the frozen source.
    rule_place = rule_places[0]
    exempt_role = roles[1]          # never roles[0]; the alias role has an occupant
    carve = list(zip(rule_places, roles))    # (place, exempt role) -- one per role
    rng.shuffle(carve)
    carve = [(rule_place, exempt_role)] + [(pl, r) for pl, r in carve
                                           if pl != rule_place and r != exempt_role][:3]

    # A wide count range on purpose: with `inspected` at 3-4 and one exclusion the answer took
    # only two values and `constant_majority` scored 0.70 on that stratum by guessing.
    # ROTATED count, and a VARYING number of exclusions.
    #
    # Two separate defects met here. With 3-4 items the answer took two values and
    # `constant_majority` guessed 0.80 of the stratum. With exactly one exclusion always, the
    # answer was `len(listing) - 1` -- countable off the listing sentence without ever reading
    # the withdrawal clause, which is the entire task. Both are fixed by rotating the count and
    # the exclusion count independently.
    n_insp = 4 + (idx % 3)          # 4-6 items: wide enough to spread, narrow enough to count
    inspected = rng.sample(objects, min(n_insp, len(objects)))
    n_excl = 1 + ((idx // 3) % 3)   # 1-3 withdrawn, so list length alone determines nothing
    excluded = inspected[:min(n_excl, len(inspected) - 1)]

    return {
        "scenario_id": sid, "site": site, "people": people, "roles": roles, "badges": badges,
        "needs": None,   # set per pack in build_bank; the answerable lacuna must avoid it
        "objects": objects, "places": places, "move_places": move_places,
        "rule_places": rule_places, "density": density,
        "alias_person": alias_person, "alias_role": alias_role,
        "alias_action": {"object": objects[0], "place": places[0]},
        "moves": moves, "tracked": tracked,
        "corr_obj": corr_obj, "stated_day": stated_day, "true_day": true_day,
        "rule_place": rule_place, "exempt_role": exempt_role, "carve": carve,
        "inspected": inspected, "excluded": excluded,
    }


# --------------------------------------------------------------------------------------- #
# Render. Block order AND within-block clause order are randomised -- this is the fix for the
# defect that broke RCI-001's floor.
# --------------------------------------------------------------------------------------- #

def _blk_alias(w, rng, cf, insufficient=False):
    keep = (f"It was the {w['alias_role']} who carried {w['alias_action']['object']} to "
            f"{w['alias_action']['place']}, unaccompanied.")
    if insufficient:
        return [f"The {w['alias_role']} that week was a member of staff whose name is torn "
                f"from the page, and who is not named again in this account.", keep]
    who = w["people"][1] if cf else w["alias_person"]
    return [f"The {w['alias_role']} that week was {who}, who is not named again in this "
            f"account.", keep]


def _blk_moves(w, rng, cf, insufficient=False):
    """Both places are named exactly TWICE per object, whichever way the move went.

    In the first draft the cancelled branch named the origin twice and the destination once,
    so `most_frequent_candidate_in_context` read the answer straight off and scored 0.625 on
    this stratum. Frequency now carries no information about the outcome; only the cancellation
    clause does.
    """
    out = []
    for m in sorted(w["moves"], key=lambda _m: rng.random()):
        out.append(f"{m['object'].capitalize()} was kept in {m['from']} at the start of the "
                   f"period. On {m['day']}, {m['by']} was assigned to move it to {m['to']}.")
        if insufficient and m is w["tracked"]:
            out.append(f"The line recording whether the move of {m['object']} out of "
                       f"{m['from']} was cancelled before it was carried out, or completed as "
                       f"assigned, is water-damaged and cannot be read.")
            continue
        out.append(
            f"That assignment for {m['object']} was cancelled before it was carried out; the "
            f"{m['to']} entry was struck through and the item did not leave {m['from']}."
            if m["cancelled"] else
            f"The move of {m['object']} out of {m['from']} was completed as assigned, and it "
            f"was held in {m['to']} thereafter.")
    return out


def _blk_temporal(w, rng, cf, insufficient=False):
    """RANDOMISED ORDER. RCI-001 always put the correction after the error, so `answer the last
    day mentioned` scored 78/78 here. Half the packs now state the correct date FIRST."""
    err = (f"An earlier page gives {w['stated_day']} as the date the {w['corr_obj']} was "
           f"examined; that entry is a transcription error.")
    if insufficient:
        return [err, f"The examination of {w['corr_obj']} took place on a day whose entry has "
                     f"been obliterated and cannot be recovered."]
    d = w["stated_day"] if cf else w["true_day"]
    fix = f"The examination of {w['corr_obj']} took place on {d}."
    return [err, fix] if rng.random() < 0.5 else [fix, err]


def _blk_rule(w, rng, cf, insufficient=False):
    """RANDOMISED ORDER, same reason: the exception used to always follow the rule.

    Two carve-outs, not one. With a single rule the congruent question scored 24/24 for the
    frozen source on the calibration twin -- above the ceiling, so the cell could not have
    shown damage even if damage were there, and it is the control cell the whole 2x2 leans on.
    A second rule for a different place forces the answer to be matched to the place asked
    about instead of being the only exception in the text.

    In BADGE packs the exception identifies its holder by badge number and never by role, so
    the incongruent question can be a bare "Who ..." with no answer-type instruction in it.
    Instructing the type ("Answer with the badge number") re-types the answer and destroys the
    incongruence -- measured on the twin, where that cell showed no damage while its role
    counterpart lost 9 of 17.
    """
    by_badge = w["answer_class"] == "BADGE"

    def _who(role):
        return (f"the holder of badge {w['badges'][role]}" if by_badge else f"the {role}")

    pairs = list(w["carve"])
    rng.shuffle(pairs)
    out = []
    for pl, role in pairs:
        rule = (f"Access to {pl} required a countersigned docket from every member of staff.")
        if insufficient:
            # The RULE stays, so every word the question uses -- access, countersigned, docket,
            # permitted, enter, and the place itself -- is still in the context. Only the line
            # naming WHO was excepted is gone. Deleting the whole block instead removed the
            # question's own vocabulary and made answerability detectable by set difference.
            out += [rule, f"One standing exception applied there: a role whose name is "
                          f"torn from the page was permitted to enter {pl} without one."]
            continue
        exc = (f"One standing exception applied there: {_who(role)} was permitted to enter "
               f"{pl} without one.")
        out += [rule, exc] if rng.random() < 0.5 else [exc, rule]
    # ORDER RANDOMISED. Listing badges in role order put the exempt role second every time,
    # which is a positional cue an extractor can learn without reading.
    items = list(w["badges"].items())
    rng.shuffle(items)
    badges = ("Duty badges for the period were issued as follows: "
              + "; ".join(f"the {r}, badge {b}" for r, b in items) + ".")
    pos = rng.randint(0, len(out))
    return out[:pos] + [badges] + out[pos:]


def _blk_count(w, rng, cf, insufficient=False):
    """Both orders read naturally, so which comes first is randomised. In RCI-001 the
    withdrawal always came last and `last_candidate_in_context` read it straight off."""
    ins = list(w["inspected"])
    rng.shuffle(ins)
    ex = w["excluded"][0]
    listing = (f"The items formally inspected during the period were {', '.join(ins[:-1])} and "
               f"{ins[-1]}.")
    if insufficient:
        return [listing, "Of those, an unrecorded number were withdrawn from the inspection "
                         "record after the fact and are not counted as inspected; the lines "
                         "saying which and how many are missing."]
    if rng.random() < 0.5:
        return [listing,
                f"Of those, {ex} was withdrawn from the inspection record after the fact and "
                f"is not counted as inspected."]
    return [f"{ex.capitalize()} was withdrawn from the inspection record after the fact and "
            f"is not counted as inspected, though it appears in the listing below.",
            listing]


#: Every block renders two ways. The `insufficient` rendering is what an unanswerable row
#: gets, and it differs from its sufficient twin ONLY by replacing the answer token with an
#: illegibility marker -- same template, same predicate vocabulary, same entities.
#:
#: Two separate leaks forced this, and the second only appeared after the first was fixed.
#: DELETING the block removed the question's own vocabulary, so counting question tokens absent
#: from the context detected answerability at 40/64 with zero false abstentions -- beating the
#: frozen baseline. Replacing the block with loosely-worded prose still left a LEXICAL tell: a
#: rule checking for each task's characteristic phrase scored 0.750. Only a rendering matched
#: phrase-for-phrase against its twin closes both.
#:
#: The cost is real and is reported rather than hidden: with the surface cue gone the frozen
#: source's correct abstentions fell from 35/64 to 15/64. Most of what the first draft measured
#: as abstention was the cue.
BLOCKS = {"alias": _blk_alias, "moves": _blk_moves, "temporal": _blk_temporal,
          "rule": _blk_rule, "count": _blk_count}


#: What is unreadable. Present in EVERY row -- see `_lacuna`.
LACUNA = {
    "rule": "The lines recording the access requirements for {rule_place} are torn away.",
    "temporal": "The line giving the date on which {corr_obj} was examined is obliterated.",
    "moves": "The movement log for {tracked_object} is water-damaged and cannot be read.",
    "alias": "The line naming the {alias_role} for that week has been cut out.",
    "count": "The lines listing which items were formally inspected are missing.",
}


def _lacuna(w, rng, omit):
    """Every row says something is unreadable. Only in the UNANSWERABLE row is it the thing asked about.

    THE ABSTENTION FIX, and it is a power fix rather than a fairness one. On the first
    calibration twin the frozen source abstained on 0 of 8 unanswerable rows in seven of nine
    strata: silently deleting a block leaves prose that reads complete, so the model answers
    from the decoys and never notices. With no correct abstentions there are no discordant
    pairs, and E1 came out at 11 of 72 against RCI-001's 40 of 128 -- an endpoint with no power.

    A lacuna note in EVERY row makes the omission findable without making it detectable.
    "Does this row contain an illegibility note" is constant across the bank and therefore
    worth nothing to an attacker; "is the illegible thing the thing I was asked about" is the
    abstention judgement itself.

    Returns (block_to_drop, sentence). For an answerable row the named block is dropped too,
    so the note is never false -- and it is never the block the question needs.
    """
    fmt = {"rule_place": w["rule_place"], "corr_obj": w["corr_obj"],
           "tracked_object": w["tracked"]["object"], "alias_role": w["alias_role"]}
    key = omit if omit else rng.choice([k for k in LACUNA if k != w["needs"]])
    return key, LACUNA[key].format(**fmt)


def _density_noise(w, rng, omit):
    """Same-type distractors left behind after the determining sentence is removed.

    A registered secondary on E1: if abstention loss scales with distractor density, the
    mechanism is capture; if it is flat, it is a genuine confidence shift. RCI-001 could not ask
    this because density was never varied.
    """
    if omit is None:
        return []
    n = 3 if w["density"] == "high" else 0
    if omit == "temporal":
        return [f"A note elsewhere records unrelated activity on {d}."
                for d in rng.sample(DAYS, min(n, len(DAYS)))]
    if omit == "moves":
        return [f"{p} was swept and re-shelved during the period."
                for p in rng.sample(w["places"], min(n, len(w["places"])))]
    if omit == "rule":
        return [f"The {r} signed the daily attendance book."
                for r in rng.sample(list(w["badges"]), min(n, len(w["badges"])))]
    if omit == "count":
        return [f"{o.capitalize()} appears in the general holdings list."
                for o in rng.sample(w["objects"], min(n, len(w["objects"])))]
    if omit == "alias":
        return [f"{p} was on the premises at some point during the period."
                for p in rng.sample(w["people"], min(n, len(w["people"])))]
    return []


def _register(w, rng):
    """Same-class entities mentioned OUTSIDE the determining sentences, in every row.

    Positional extractors -- first / last / most-frequent candidate in context -- only work
    when the answer is the sole member of its answer vocabulary present. Measured on the first
    draft of this bank: `first_candidate_in_context` scored 1.000 on both badge strata because
    nothing else in the text was a badge number. The register puts same-class decoys in front
    of and behind the real answer, so those rules have to guess.

    Nothing here is determining. It never states who holds a role, where an item ended up, when
    anything happened, or which item was withdrawn.
    """
    # SAMPLED, never the complement. The first version listed exactly the NON-exempt roles
    # and the NON-exempt badges, so "the member of the badge list missing from the roster list"
    # recovered the answer with no reading at all -- a set-complement leak introduced by the
    # very block that was added to close a positional one, and invisible to every positional
    # floor predictor. A red-team pass found it before the bank was sealed; RCI-001 had no such
    # pass and shipped its defect.
    # EVERY member of each class, shuffled -- not a sample, and never the complement.
    #
    # Two mistakes have now been made in this one block. Listing the complement of the answer
    # handed it over outright. Listing a random SUBSET then broke ties in
    # `most_frequent_candidate_in_context`: whichever candidate the sample happened to include
    # got +1, and on the badge and place strata that was enough to push the floor past 0.60.
    # Listing all of them adds exactly +1 to every candidate, which preserves ties and leaves
    # frequency carrying no information at all.
    def _all(xs):
        xs = list(xs)
        rng.shuffle(xs)
        return xs

    def block():
        ls = [
            f"Duty rosters list the {', the '.join(_all(w['badges']))} among the staff.",
            f"Badges {', '.join(_all(w['badges'].values()))} were in circulation this period.",
            f"Holdings referenced elsewhere include {', '.join(_all(w['objects']))}.",
            f"Rooms cleaned during the period were {', '.join(_all(w['move_places']))}.",
        # THE QUESTION-TOKEN FIX. Every entity a question can name must appear somewhere
        # OUTSIDE the block that determines the answer, or deleting that block removes the
        # question's own vocabulary and answerability becomes detectable by set difference.
        # Measured on the first sealed draft: a rule counting question tokens absent from the
        # context scored 40/64 correct abstentions at ZERO false abstentions -- beating the
        # frozen baseline's 35 -- and was perfect on the five strata whose questions name a
        # `rule_place` or say "formally inspected". Those two are the only vocabulary the
        # register did not already carry.
        f"Keys were signed out for {', '.join(_all(w['rule_places']))} during the period.",
        ("Lists of items formally inspected in earlier periods are held separately and are "
         "not reproduced in this account."),
            f"The duty book carries initials for {', '.join(_all(w['people']))}.",
            f"Entries were made on {', '.join(_all(DAYS))}.",
        ]
        rng.shuffle(ls)
        return ls

    # Every answer class is enumerated BEFORE and AFTER the determining sentences, in two
    # independent random orders. `first_candidate_in_context` and `last_candidate_in_context`
    # then both resolve inside a shuffled full listing and are uniform over the class, which
    # is the only way to make position carry nothing. Sampling a subset was not enough: the
    # sample decided the tie.
    return block(), block()


def render(w: dict, rng: random.Random, omit: str | None, cf: bool) -> str:
    head = (f"This is a consolidated account of the audit at {w['site']}, covering six "
            f"consecutive days. Entries appear in the order they were found in the duty books, "
            f"which is not the order in which they occurred.")
    order = list(BLOCKS)
    rng.shuffle(order)                       # BLOCK ORDER RANDOMISED
    lac_key, lac = _lacuna(w, rng, omit)
    if omit is not None:
        # The insufficient rendering states its own lacuna in place, so the standalone note
        # would be a second, redundant tell. Answerable rows keep theirs.
        lac = None
    if omit is None:
        order = [b for b in order if b != lac_key]   # keep the note true
    reg_head, reg_tail = _register(w, rng)
    body = list(reg_head)                    # every class enumerated BEFORE
    for name in order:
        body += BLOCKS[name](w, rng, cf, insufficient=(name == omit))
        if rng.random() < 0.5:
            body.append(rng.choice(FILLER))
    body += _density_noise(w, rng, omit)
    if lac:
        body.append(lac)
    body += reg_tail                         # and AFTER
    # Order WITHIN a block stays as rendered -- the randomisation that matters is which block
    # comes first and which clause leads inside it. Shuffling sentences would produce prose no
    # reader could follow and would change what the instrument measures.
    return " ".join([head] + body)


# --------------------------------------------------------------------------------------- #
# Questions. The interrogative is a property of the TASK TYPE, never of the variant -- that
# separation is the whole point of this build.
# --------------------------------------------------------------------------------------- #

def ask(w: dict, task: str, cf: bool, paraphrase: bool) -> tuple[str, str]:
    p, badge = w["rule_place"], w["badges"][w["exempt_role"]]
    role = w["exempt_role"]
    if task == "carve_out_role_congruent":
        q = (f"Which role was permitted to enter {p} without a countersigned docket?"
             if not paraphrase else
             f"Which role did the standing exception for {p} apply to?")
        return q, role
    if task == "carve_out_role_incongruent":
        q = (f"Who was permitted to enter {p} without a countersigned docket?"
             if not paraphrase else
             f"Who did the standing exception for {p} apply to?")
        return q, role
    if task == "carve_out_badge_congruent":
        q = (f"Which badge number was permitted to enter {p} without a countersigned docket?"
             if not paraphrase else
             f"Which badge number did the standing exception for {p} apply to?")
        return q, badge
    if task == "carve_out_badge_incongruent":
        # A BARE "Who". The context identifies the exempt person only by badge number, so this
        # is determinate and person-free, exactly as the role/incongruent cell is. Naming the
        # answer type in the question would remove the incongruence being tested.
        q = (f"Who was permitted to enter {p} without a countersigned docket?"
             if not paraphrase else
             f"Who did the standing exception for {p} apply to?")
        return q, badge
    if task == "exclusion_count_number":
        q = ("How many items are counted as formally inspected? Answer with a number."
             if not paraphrase else
             "Give the number of items that count as formally inspected.")
        return q, str(len(w["inspected"]) - len(w["excluded"]))
    if task == "alias_resolution":
        o, pl = w["alias_action"]["object"], w["alias_action"]["place"]
        q = (f"Who personally carried {o} to {pl}?" if not paraphrase else
             f"By name, who took {o} to {pl} on their own?")
        return q, (w["people"][1] if cf else w["alias_person"])
    if task == "object_state_tracking":
        t = w["tracked"]
        end = t["from"] if t["cancelled"] else t["to"]
        if cf:
            end = t["to"] if t["cancelled"] else t["from"]
        q = (f"Where was {t['object']} at the end of the period?" if not paraphrase else
             f"Where was {t['object']} held once the period closed?")
        return q, end
    if task == "temporal_correction":
        d = w["stated_day"] if cf else w["true_day"]
        q = (f"On which day was {w['corr_obj']} examined?" if not paraphrase else
             f"On which day did the examination of {w['corr_obj']} actually take place?")
        return q, d
    raise ValueError(task)


#: Task types whose answer changes under the counterfactual. The carve-out and exclusion
#: families have no counterfactual reading, so their `counterfactual` variant re-renders the
#: world with a different block order and keeps the same answer -- an order-robustness cell
#: rather than a filler row.
CF_SENSITIVE = {"alias_resolution", "object_state_tracking", "temporal_correction"}


def build_bank(partition: str, packs_per_task: int, seed: int) -> list[dict]:
    pool = POOLS[partition]
    rows, n = [], 0
    for task in TASK_TYPES:
        for i in range(packs_per_task):
            sid = f"R2-{partition[:1]}{n:04d}"
            n += 1
            density = "high" if i % 2 else "low"
            rng = random.Random(h(seed, partition, sid))
            w = make_world(rng, pool, sid, density, idx=i)
            # The rule block renders differently for BADGE packs, so the world has to know
            # which task it is serving. One task type per pack, so this is unambiguous.
            w["answer_class"] = TASK_TYPES[task]["answer"]
            w["needs"] = TASK_TYPES[task]["needs"]

            # ASSERTION, not an assumption: an incongruent question types the answer as a
            # person, and it is only determinate because NO person is bound to the exempt role.
            if TASK_TYPES[task]["wh"] == "incongruent" and w["exempt_role"] == w["alias_role"]:
                raise SystemExit(f"{sid}: exempt role has an occupant; incongruence unsound")

            need = TASK_TYPES[task]["needs"]
            for variant in VARIANTS:
                r = random.Random(h(seed, partition, sid, variant))
                cf = variant == "counterfactual" and task in CF_SENSITIVE
                omit = need if variant == "unanswerable" else None
                ctx = render(w, r, omit=omit, cf=cf)
                q, ans = ask(w, task, cf, paraphrase=(variant == "paraphrase"))
                if variant == "unanswerable":
                    ans = SENTINEL
                if SENTINEL.lower() in ctx.lower() or SENTINEL.lower() in q.lower():
                    raise SystemExit(f"sentinel leaked into item content: {sid}/{variant}")
                msgs = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": f"CONTEXT\n{ctx}\n\nQUESTION\n{q}"}]
                rows.append({
                    "id": f"{sid}-{variant[:4].upper()}", "scenario_id": sid,
                    "partition": partition, "task_type": task, "variant": variant,
                    "stratum": task,
                    "answer_class": TASK_TYPES[task]["answer"],
                    "wh_congruence": TASK_TYPES[task]["wh"],
                    "distractor_density": density if variant == "unanswerable" else None,
                    "answerable": variant != "unanswerable",
                    "endpoint": ("E1_abstention" if variant == "unanswerable" else
                                 ("E2_wh_incongruence"
                                  if TASK_TYPES[task]["wh"] == "incongruent" else None)),
                    "prompt": msgs,
                    "reference": ans,
                    "reference_normalised": normalise(ans),
                    "context": ctx,
                    "gold_verified_by": f"oracle.{task}",
                    "context_words": len(ctx.split()),
                    "prompt_sha256": h(canon(msgs)), "context_sha256": h(ctx),
                })
    return rows


# --------------------------------------------------------------------------------------- #
# Floors -- computed here, and ENFORCED. RCI-001 reported its floors; this one refuses.
# --------------------------------------------------------------------------------------- #

def _detector(rows: list[dict]) -> dict:
    """The best zero-false-abstention question/context token rule, swept over its parameters.

    Reported as a first-class build quantity rather than left to an auditor, because an
    auditor found it and the seal was nearly spent with it in place. A rule is only counted
    when it raises NO false abstentions on answerable rows -- a detector that over-abstains is
    not a threat, and scoring it without that constraint would flatter the bank.
    """
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    best = {"correct": 0, "n": len(una), "rate": 0.0, "predictor": "none", "false": 0}
    for minlen in (3, 4, 5, 6):
        for thr in range(1, 9):
            def fires(r):
                q = r["prompt"][1]["content"].split("QUESTION\n")[-1].lower()
                toks = {t for t in re.findall(r"[a-z]+", q) if len(t) >= minlen}
                return sum(1 for t in toks if t not in r["context"].lower()) >= thr
            if any(fires(r) for r in ans):
                continue
            c = sum(1 for r in una if fires(r))
            if c > best["correct"]:
                best = {"correct": c, "n": len(una),
                        "rate": round(c / len(una), 4) if una else 0.0,
                        "predictor": f"question_tokens_absent(minlen>={minlen}, thr>={thr})",
                        "false": 0}
    best["why"] = ("if the unanswerable variant deletes the block that determines the answer, "
                   "it also deletes the entities the question names, and answerability becomes "
                   "detectable without reading")
    return best


def floor_report(rows: list[dict]) -> dict:
    """Aggregate, per stratum, and per endpoint -- every one through the shared gate."""
    def _f(sub):
        if not sub:
            return None
        f = gate_floors(sub, abstain=SENTINEL)
        return {"n": len(sub), "binding_class": f["binding_class"],
                "binding_predictor": f["binding_predictor"],
                "binding_floor_count": f["binding_floor_count"],
                "binding_floor_rate": f["binding_floor_rate"],
                "extraction": f["extraction"], "distributional": f["distributional"]}

    strata = {t: _f([r for r in rows if r["task_type"] == t]) for t in TASK_TYPES}
    e1 = [r for r in rows if r["endpoint"] == "E1_abstention"]
    e2 = [r for r in rows if r["endpoint"] == "E2_wh_incongruence"]
    return {
        "aggregate": _f(rows),
        "per_stratum": strata,
        "per_stratum_answerable": {t: _f([r for r in rows if r["task_type"] == t
                                          and r["answerable"]]) for t in TASK_TYPES},
        "E1_abstention": _f(e1),
        # Predictors calibrated on the FULL bank and scored on the subset. Scoring an
        # all-UNKNOWN subset against itself gives `constant_majority` = 1.000, which says
        # nothing about the endpoint and would refuse every instrument that has one.
        "E1_abstention_immunity": immunity(rows, lambda r: not r["answerable"],
                                           abstain=SENTINEL),
        "E1_answerability_detector": _detector(rows),
        "E2_wh_incongruence": _f(e2),
        "pack_structure": require_pack_structure(rows, min_variants=len(VARIANTS)),
        "note": ("Extraction predictors are in the published set at BUILD time. RCI-001's were "
                 "added by an outside audit after the seal was spent."),
    }


def enforce(fr: dict, max_floor: float, max_endpoint_floor: float,
            max_detector: float = 0.35) -> list[str]:
    """The gate. Returns the reasons to REFUSE; an empty list means the bank may ship.

    The two endpoints are held to DIFFERENT standards, because only one of them can meet the
    stronger one. An abstention endpoint is **structurally** extraction-immune: an extractor
    answers something from the context, so it can never emit the sentinel and scores exactly
    zero. That is why RCI-001 survived its own floor defect. An ANSWERABLE endpoint has no such
    property -- any extractor gets some rows right - so demanding a zero floor there is
    unachievable, and the first draft of this gate demanded exactly that and refused its own
    bank. It is a bounded floor instead, which is the honest version.
    """
    bad = []
    if not fr["pack_structure"]["satisfied"]:
        bad.append(f"pack structure unsatisfied: {fr['pack_structure']}")

    e1 = fr["E1_abstention_immunity"]
    if not e1["extraction_immune"]:
        bad.append(f"E1 is not extraction-immune: {e1['extraction']} -- an extractor should be "
                   f"unable to emit the sentinel at all")

    # Extraction immunity was true and TOO NARROW. No extractor can emit the sentinel, but a
    # rule that detects ANSWERABILITY can -- and on the first sealed draft one did, at 40/64
    # with zero false abstentions, beating the frozen baseline. Immunity to emitting the answer
    # is not immunity to knowing when there is none.
    det = fr["E1_answerability_detector"]
    if det["rate"] > max_detector:
        bad.append(f"E1 answerability is structurally detectable: "
                   f"{det['predictor']} scores {det['rate']:.4f} "
                   f"({det['correct']}/{det['n']}) > {max_detector}. The unanswerable variant "
                   f"is removing the question's own vocabulary along with the answer")

    f2 = fr["E2_wh_incongruence"]
    if f2["binding_floor_rate"] > max_endpoint_floor:
        bad.append(f"E2 floor {f2['binding_floor_rate']:.4f} > {max_endpoint_floor} via "
                   f"{f2['binding_predictor']} ({f2['binding_class']}) -- an answerable "
                   f"endpoint cannot be shortcut-immune, but it must leave room to measure")

    for t, f in fr["per_stratum_answerable"].items():
        if f and f["binding_floor_rate"] > max_floor:
            bad.append(f"stratum {t} answerable floor {f['binding_floor_rate']:.4f} "
                       f"> {max_floor} via {f['binding_predictor']} "
                       f"({f['binding_class']}) -- this is the RCI-001 defect")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--calib-packs-per-task", type=int, default=8)
    ap.add_argument("--sealed-packs-per-task", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--max-floor", type=float, default=0.60,
                    help="refuse to ship if any answerable stratum shortcut floor exceeds this")
    ap.add_argument("--max-endpoint-floor", type=float, default=0.35,
                    help="E2 is held to a tighter bound than an ordinary stratum")
    ap.add_argument("--max-answerability-detector", type=float, default=0.35,
                    help="refuse if question/context token overlap detects answerability")
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite: {a.out}")

    calib = build_bank("CALIB", a.calib_packs_per_task, a.seed)
    sealed = build_bank("SEALED", a.sealed_packs_per_task, a.seed)

    # Disjointness is a requirement, not an expectation.
    if {r["context_sha256"] for r in calib} & {r["context_sha256"] for r in sealed}:
        raise SystemExit("calibration and sealed contexts overlap")
    ce = set()
    for k in ("people", "objects", "places", "sites", "roles"):
        ce |= set(POOLS["CALIB"][k]) & set(POOLS["SEALED"][k])
    if ce:
        raise SystemExit(f"entity pools overlap: {sorted(ce)}")
    reused = set()
    for part in POOLS.values():
        for k in ("people", "objects", "places", "sites", "roles"):
            reused |= {e for e in part[k] if e in RCI001_ENTITIES
                       or ARTICLE.sub("", e) in RCI001_ENTITIES}
    if reused:
        raise SystemExit(f"entities reused from RCI-001: {sorted(reused)}")

    fr_c, fr_s = floor_report(calib), floor_report(sealed)
    refusals = (enforce(fr_c, a.max_floor, a.max_endpoint_floor, a.max_answerability_detector)
                + enforce(fr_s, a.max_floor, a.max_endpoint_floor,
                          a.max_answerability_detector))
    if refusals:
        print("BUILD REFUSED -- the bank would not be interpretable:")
        for r in refusals:
            print("  -", r)
        return 2

    a.out.mkdir(parents=True)
    (a.out / "RCI_CALIBRATION.jsonl").write_text(
        "".join(canon(r) + "\n" for r in calib), encoding="utf-8")
    (a.out / "RCI_SEALED.jsonl").write_text(
        "".join(canon(r) + "\n" for r in sealed), encoding="utf-8")

    man = {
        "schema": 1, "record": "RCI_002_BUILD", "instrument": "RCI-002",
        "supersedes": "RCI-001 (SPENT 2026-09-18) -- generator reused, defects fixed",
        "what_this_is": (
            "A confirmation instrument with TWO pre-registered endpoints. Answers are computed "
            "from generated structure by oracle and never parsed from prose."),
        "seed": a.seed,
        "declared_normaliser": {
            "source": "confirm/build_rci2.py::normalise",
            "rule": "first line, lowercased, '.' and ',' to space, whitespace collapsed, "
                    "leading definite/indefinite article stripped",
            "why": ("RCI-001 scored `the morning registrar` against `morning registrar` as "
                    "wrong; 52 of 384 answerable rows turned on an article and two sessions "
                    "reconciled them by hand twice. References are stored BARE as well."),
        },
        "endpoints": ENDPOINTS,
        "design": {
            "crossed_factors": {
                "answer_class": sorted({v["answer"] for v in TASK_TYPES.values()}),
                "wh_congruence": ["congruent", "incongruent"],
                "note": ("wh-congruence is a property of the TASK TYPE. `paraphrase` varies "
                         "wording and holds the interrogative FIXED -- in RCI-001 the two were "
                         "confounded and that confound carried its largest effect."),
            },
            "discriminating_2x2": {
                "cells": ["carve_out_role_congruent", "carve_out_role_incongruent",
                          "carve_out_badge_congruent", "carve_out_badge_incongruent"],
                "reasoning_structure": "identical in all four: find the role exempted from a rule",
                "predicted_damage": {
                    "wh_compliance": ["none", "DAMAGE", "none", "DAMAGE"],
                    "role_occupant_binding": ["damage", "damage", "damage", "damage"],
                    "output_type_shift": ["damage", "damage", "none", "none"],
                },
                "soundness_assertion": ("no person is bound to the exempt role in any pack, so "
                                        "an incongruent question is determinate and its "
                                        "person-typed reading has no answer; asserted per pack"),
            },
            "clause_and_block_order_randomised": True,
            "distractor_density_on_unanswerable": ["low", "high"],
            "variants_per_pack": list(VARIANTS),
            "task_types": {k: v for k, v in TASK_TYPES.items()},
            "counterfactual_note": ("only alias/state/temporal have a counterfactual reading; "
                                    "for the rest the cell re-renders with a different block "
                                    "order and the same answer, making it an order-robustness "
                                    "row rather than filler"),
        },
        "banks": {
            "CALIBRATION": {"packs": a.calib_packs_per_task * len(TASK_TYPES),
                            "rows": len(calib), "sha256": h(canon(calib)),
                            "role": "OPENLY SPENDABLE -- scored to verify difficulty is in band"},
            "SEALED": {"packs": a.sealed_packs_per_task * len(TASK_TYPES),
                       "rows": len(sealed), "sha256": h(canon(sealed)),
                       "role": "CONFIRMATION ONLY -- never scored in this build"},
        },
        "partitions_disjoint": {
            "entity_pools": "asserted disjoint CALIB vs SEALED",
            "contexts": "asserted zero shared context_sha256",
            "vs_RCI_001": "asserted: no entity from either RCI-001 pool is reused",
            "assigned_before_generation": True,
        },
        "floors_published_before_any_model_call": True,
        "floors": {"CALIBRATION": fr_c, "SEALED": fr_s},
        "floor_enforcement": {
            "max_answerable_stratum_floor": a.max_floor,
            "max_E2_endpoint_floor": a.max_endpoint_floor,
            "E1_must_be_extraction_immune": True,
            "max_answerability_detector": a.max_answerability_detector,
            "why_extraction_immunity_was_not_enough": (
                "no extractor can EMIT the sentinel, but a rule that detects answerability can. "
                "On the first sealed draft, counting question tokens absent from the context "
                "scored 40/64 at zero false abstentions -- beating the frozen baseline's 35. "
                "Immunity to producing the answer is not immunity to knowing there isn't one."),
            "E1_required_control": ("false abstention on answerable rows, reported and near "
                                    "zero -- an abstention endpoint is NOT distributionally "
                                    "immune and endpoint choice cannot make it so"),
            "build_refuses_otherwise": True,
            "why": ("RCI-001 REPORTED its floors. A report cannot stop a defective bank "
                    "shipping, and one did."),
        },
        "floor_rule": ("a system clears only if the exact two-sided binomial 95% lower bound on "
                       "its correct count exceeds the binding floor rate"),
        "read_budget": {
            "CALIBRATION": "unlimited",
            "SEALED": {"reads_authorised": 1, "reads_used": 0,
                       "condition": "after a candidate is frozen; never for selection",
                       "claim_before_render": True},
        },
        "model_calls_in_this_build": 0,
    }
    (a.out / "RCI_BUILD.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "calibration_rows": len(calib), "sealed_rows": len(sealed),
        "task_types": len(TASK_TYPES),
        "sealed_floor_aggregate": fr_s["aggregate"]["binding_floor_rate"],
        "sealed_floor_predictor": fr_s["aggregate"]["binding_predictor"],
        # NOT `E1_abstention.binding_floor_rate`. That scores an all-UNKNOWN subset against
        # itself, returns 1.000 for `always_abstain`, and says nothing about the endpoint.
        # The figure that matters is whether an extractor calibrated on the WHOLE bank can
        # reach it -- structurally it cannot, and that is the property being claimed.
        "E1_n": fr_s["E1_abstention"]["n"],
        "E1_extraction_immune": fr_s["E1_abstention_immunity"]["extraction_immune"],
        "E1_extraction_max": fr_s["E1_abstention_immunity"]["extraction_max"],
        "E1_required_control": "false abstention on answerable rows, reported at calibration",
        "E1_answerability_detector": fr_s["E1_answerability_detector"],
        "E2_floor": fr_s["E2_wh_incongruence"]["binding_floor_rate"],
        "E2_n": fr_s["E2_wh_incongruence"]["n"],
        "worst_answerable_stratum": max(
            ((t, f["binding_floor_rate"], f["binding_predictor"])
             for t, f in fr_s["per_stratum_answerable"].items() if f),
            key=lambda x: x[1]),
        "sealed_sha256": man["banks"]["SEALED"]["sha256"],
        "context_words": {"min": min(r["context_words"] for r in sealed),
                          "max": max(r["context_words"] for r in sealed)},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

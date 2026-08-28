#!/usr/bin/env python3
"""Generate fresh long-context scenarios in the Rineya Demo v1 schema.

Why this exists
---------------
All 256 rows of `Rineya_Long_Context_Demo_v1` have been consumed: showcase64 was
selection data for the published family, and stability192 was inspected when the
S4 abstention threshold was chosen. The data card additionally stamps every row
`intended_use: "demo/evaluation only; do not train on this exact suite"`.

So any claim of the form "Rineya beats a larger model" cannot be substantiated on
that suite. This generator produces statistically comparable but entirely fresh
scenarios, so a sealed TEST split can exist.

Design contract
---------------
Every scenario is built from one *decisive binding* that can be expressed three
ways, mirroring the original suite exactly:

  base           the binding is stated plainly
  counterfactual the binding is SWAPPED; every entity stays present, so surface
                 statistics barely move and only the relation flips
  unanswerable   the binding is DESTROYED; every entity still stays present, so
                 the model must notice a missing *link*, not a missing name

`paraphrase` reuses the base context with a reworded question.

That "entities stay, relation moves" property is what makes the original suite
hard, and it is reproduced here deliberately.

Determinism
-----------
Fully seeded. Same seed and count -> byte-identical output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Callable

SYSTEM_PROMPT = (
    "You are answering a closed-context reading test. Use only the supplied "
    "context. Reply with the shortest exact answer and no explanation. If the "
    "context does not determine the answer, reply UNKNOWN."
)

UNKNOWN_ACCEPTED = ["UNKNOWN", "unknown", "not determinable", "cannot be determined"]

TASK_TYPES = [
    "alias_resolution",
    "object_state_tracking",
    "temporal_correction",
    "rule_exception",
    "count_with_exclusions",
    "causal_disambiguation",
    "multi_condition_identity",
    "latest_verified_record",
]

NEEDLE_PROFILES = ["early", "middle", "late", "distributed"]
LENGTH_BUCKETS = {"compact_long": 860, "standard_long": 955, "extended_long": 1055}

# Confusable first-name pairs drive `hard` difficulty: the counterfactual swap
# then moves the answer between two near-homographs, exactly like the original
# suite's Talia Frost / Thalia Lark.
CONFUSABLE_FIRSTS = [
    ("Talia", "Thalia"), ("Karin", "Karina"), ("Elena", "Elina"),
    ("Alina", "Alena"), ("Dana", "Dena"), ("Nora", "Nyra"),
    ("Cara", "Kara"), ("Sofia", "Sonia"), ("Mira", "Myra"),
    ("Lena", "Lina"), ("Rhea", "Thea"), ("Iona", "Iona"),
    ("Corin", "Corrin"), ("Silas", "Silus"), ("Emmet", "Emmett"),
    ("Rowan", "Roan"), ("Marek", "Marec"), ("Ilya", "Ilia"),
]
DISTINCT_FIRSTS = [
    "Beatrix", "Quinn", "Uma", "Desmond", "Priya", "Tomas", "Hana", "Otto",
    "Yusuf", "Greta", "Malik", "Ingrid", "Rafael", "Noor", "Caleb", "Simone",
    "Anwar", "Delia", "Fen", "오", "Bram", "Cleo", "Idris", "Juno",
]
DISTINCT_FIRSTS = [n for n in DISTINCT_FIRSTS if n.isascii()]

LASTS = [
    "Frost", "Lark", "Nash", "Gray", "Quill", "Shaw", "Vane", "Roth", "Pike",
    "Wren", "Ash", "Doyle", "Marsh", "Bell", "Crane", "Hale", "Vogel", "Sato",
    "Okafor", "Reyes", "Byrne", "Holt", "Nadir", "Sorrel", "Tarn", "Ives",
]

DOMAINS: dict[str, dict[str, Any]] = {
    "arcane_academy": {
        "label": "Original fantasy academy chronicle",
        "site": ["Asterhall Institute", "Cinderwick College", "Vell Observatory School"],
        "places": ["Cedar Vault", "Map Room", "North Reliquary", "Glass Gallery", "Bell Undercroft"],
        "objects": ["ivory seal", "resonance prism", "star-glass lens", "folded ward map", "opal compass"],
        "roles": ["astronomy tutor", "ward engineer", "relic conservator", "archive proctor", "bell warden"],
        "event": "archive audit",
    },
    "detective_bureau": {
        "label": "Original municipal investigation file",
        "site": ["Harbour Precinct", "Ninth District Bureau", "Calder Street Office"],
        "places": ["Evidence Bay", "Interview Room 2", "Cold Store", "Records Annex", "Property Cage"],
        "objects": ["brass key", "case ledger", "witness tape", "sealed envelope", "field notebook"],
        "roles": ["duty sergeant", "evidence clerk", "case analyst", "liaison officer", "records keeper"],
        "event": "evidence reconciliation",
    },
    "orbital_station": {
        "label": "Original orbital operations log",
        "site": ["Meridian Station", "Kepler Ring", "Halcyon Platform"],
        "places": ["Instrument Bay", "Spectra Lab", "Docking Spine", "Cryo Hold", "Command Annex"],
        "objects": ["calibration rod", "sample canister", "optics cassette", "coolant valve", "telemetry drive"],
        "roles": ["flight engineer", "payload specialist", "systems officer", "life-support tech", "comms officer"],
        "event": "systems handover",
    },
    "expedition": {
        "label": "Original field expedition record",
        "site": ["Kestrel Traverse", "Umber Basin Survey", "Latham Ridge Camp"],
        "places": ["Base Camp", "Ridge Cache", "River Depot", "Snow Hut", "Survey Post"],
        "objects": ["survey theodolite", "core sample", "weather drum", "sealed ration crate", "signal mirror"],
        "roles": ["field lead", "quartermaster", "surveyor", "camp medic", "route scout"],
        "event": "cache reconciliation",
    },
    "museum": {
        "label": "Original museum conservation record",
        "site": ["Thorne Museum", "Lowlands Collection", "Ardent Hall"],
        "places": ["Conservation Lab", "Loan Store", "East Gallery", "Quarantine Room", "Mount Shop"],
        "objects": ["gilded reliquary", "lacquer panel", "bronze stylus", "vellum charter", "glass negative"],
        "roles": ["senior conservator", "registrar", "mount maker", "collections manager", "loans officer"],
        "event": "condition survey",
    },
    "research_institute": {
        "label": "Original research institute audit",
        "site": ["Brackwater Institute", "Nyle Research Centre", "Ostrander Labs"],
        "places": ["Cold Room", "Assay Bench", "Sample Vault", "Instrument Hall", "Data Annex"],
        "objects": ["reference standard", "assay plate", "sealed reagent", "logbook volume", "calibration weight"],
        "roles": ["principal analyst", "lab manager", "quality officer", "instrument tech", "data steward"],
        "event": "quality audit",
    },
    "coastal_council": {
        "label": "Original coastal council proceeding",
        "site": ["Wraymouth Council", "Saltgate Authority", "Pellan Harbour Board"],
        "places": ["Harbour Office", "Tide Store", "Committee Room", "Slipway Shed", "Beacon House"],
        "objects": ["harbour charter", "tide register", "mooring permit", "survey plate", "beacon log"],
        "roles": ["harbour clerk", "works inspector", "committee secretary", "beacon keeper", "moorings officer"],
        "event": "permit review",
    },
    "family_enterprise": {
        "label": "Original family enterprise ledger",
        "site": ["Ardley & Sons", "theВарго Works", "Halloway Provisioners"],
        "places": ["Front Office", "Back Store", "Loading Yard", "Strong Room", "Sample Room"],
        "objects": ["order book", "brass scale", "signet ring", "stock ledger", "delivery slip"],
        "roles": ["managing partner", "book-keeper", "yard foreman", "buyer", "shop steward"],
        "event": "annual reconciliation",
    },
}
for _d in DOMAINS.values():
    _d["site"] = [s for s in _d["site"] if s.isascii()]

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# --------------------------------------------------------------------------
# Task builders
#
# Each returns a dict with, per variant, the decisive sentences that carry the
# binding, plus the question/answer pair. Non-decisive text is filler and is
# identical across variants, so a diff between variants isolates the binding.
# --------------------------------------------------------------------------


def _names(rng: random.Random, difficulty: str, count: int) -> list[str]:
    """Return `count` person names. `hard` makes the first two confusable."""
    lasts = rng.sample(LASTS, count)
    if difficulty == "hard":
        a, b = rng.choice(CONFUSABLE_FIRSTS)
        firsts = [a, b] + rng.sample(DISTINCT_FIRSTS, count - 2)
    else:
        firsts = rng.sample(DISTINCT_FIRSTS, count)
    return [f"{f} {l}" for f, l in zip(firsts, lasts)]


def build_alias_resolution(rng, dom, difficulty) -> dict[str, Any]:
    people = _names(rng, difficulty, 4)
    codes = rng.sample(["Mosaic", "Kite", "Lantern", "Harrow", "Pennant"], 2)
    obj = rng.choice(dom["objects"])
    place = rng.choice(dom["places"])
    a, b = people[0], people[1]
    q = f"Who personally carried the {obj} to {place}?"
    qp = f"Give the ordinary name of the person recorded as transporting the {obj} to {place}."
    base = [
        f"A sealed communications appendix records a temporary call-sign assignment for the operation.",
        f"{a} was assigned “{codes[0]},” and {b} was assigned “{codes[1]}.”",
        f"The appendix states that the assignments remained unchanged until the final debrief.",
        f"The transfer log shows that {codes[0]} carried the {obj} to {place}.",
    ]
    cf = list(base)
    cf[1] = f"{b} was assigned “{codes[0]},” and {a} was assigned “{codes[1]}.”"
    un = list(base)
    un[1] = (
        f"The appendix shows that the temporary call signs “{codes[0]}” and "
        f"“{codes[1]}” were issued, but the page binding each name to a "
        f"participant was removed before filing. No other source reconstructs which "
        f"participant received which name."
    )
    del un[2]
    return {
        "question": q, "question_paraphrase": qp,
        "base_sents": base, "cf_sents": cf, "un_sents": un,
        "answer_base": a, "answer_cf": b,
        "facts_base": [f"{a} was assigned {codes[0]}.", f"{codes[0]} carried the {obj}."],
        "facts_cf": [f"{b} was assigned {codes[0]}.", f"{codes[0]} carried the {obj}."],
        "entities": people,
    }


def build_object_state_tracking(rng, dom, difficulty) -> dict[str, Any]:
    obj = rng.choice(dom["objects"])
    places = rng.sample(dom["places"], 3)
    q = f"Where was the {obj} located after the final transfer?"
    qp = f"State the location holding the {obj} once the last transfer was completed."
    base = [
        f"The {obj} was first held in {places[0]} during the opening inventory.",
        f"A midweek movement slip relocated it to {places[1]}.",
        f"The final transfer placed the {obj} in {places[2]}, and it remained there after the debrief.",
    ]
    cf = list(base)
    cf[2] = f"The final transfer placed the {obj} in {places[1]}, and it remained there after the debrief."
    un = list(base)
    un[2] = (
        f"A final transfer was carried out after the midweek movement, but the "
        f"destination field on the slip was left blank and the counter-signature "
        f"page is missing. {places[0]}, {places[1]} and {places[2]} all appear as "
        f"candidate destinations elsewhere in the file."
    )
    return {
        "question": q, "question_paraphrase": qp,
        "base_sents": base, "cf_sents": cf, "un_sents": un,
        "answer_base": places[2], "answer_cf": places[1],
        "facts_base": [f"The final transfer placed the {obj} in {places[2]}."],
        "facts_cf": [f"The final transfer placed the {obj} in {places[1]}."],
        "entities": places,
    }


def build_temporal_correction(rng, dom, difficulty) -> dict[str, Any]:
    obj = rng.choice(dom["objects"])
    d_planned, d_actual, d_other = rng.sample(WEEKDAYS, 3)
    q = f"On which day did the examination of the {obj} actually begin?"
    qp = f"Identify the weekday on which examination of the {obj} genuinely started."
    # The swap exchanges which date was *planned* and which was *corrected to*,
    # so both weekdays appear in both answerable variants and only the binding
    # between "planned" and "actual" moves.
    base = [
        f"The circulated schedule listed the examination of the {obj} as beginning on {d_planned}.",
        f"That entry was only a plan and was never treated as a record of events.",
        f"A signed correction notice states that the examination actually began on {d_actual}.",
    ]
    cf = [
        f"The circulated schedule listed the examination of the {obj} as beginning on {d_actual}.",
        f"That entry was only a plan and was never treated as a record of events.",
        f"A signed correction notice states that the examination actually began on {d_planned}.",
    ]
    un = [
        f"The circulated schedule listed the examination of the {obj} as beginning on {d_planned}.",
        f"That entry was only a plan and was never treated as a record of events.",
        f"A correction notice was raised because the planned date was known to be "
        f"wrong, and {d_actual} was among the replacements discussed, but the notice "
        f"was filed unsigned and its date field is illegible. No countersigned copy survives.",
    ]
    return {
        "question": q, "question_paraphrase": qp,
        "base_sents": base, "cf_sents": cf, "un_sents": un,
        "answer_base": d_actual, "answer_cf": d_planned,
        "facts_base": [f"The initial {d_planned} date was only planned.", f"The signed correction gives {d_actual}."],
        "facts_cf": [f"The initial {d_actual} date was only planned.", f"The signed correction gives {d_planned}."],
        "entities": [d_planned, d_actual, d_other],
    }


def build_rule_exception(rng, dom, difficulty) -> dict[str, Any]:
    people = _names(rng, difficulty, 3)
    obj = rng.choice(dom["objects"])
    a, b = people[0], people[1]
    q = f"Who had authority to approve removal of the {obj} during the alarm?"
    qp = f"Name the person empowered to authorise removing the {obj} while the alarm was active."
    base = [
        f"Standing rules reserve removal approval to the duty committee as a whole.",
        f"An exception clause applies whenever emergency conditions are declared.",
        f"Emergency conditions were declared during the alarm, activating the exception.",
        f"Both {a} and {b} held the clearance required to be named under that clause.",
        f"The delegation form names {a} as sole delegate under that clause.",
    ]
    cf = list(base)
    cf[4] = f"The delegation form names {b} as sole delegate under that clause."
    un = list(base)
    un[4] = (
        f"The delegation form for that clause was issued, but the delegate field "
        f"was left blank and the register of delegates for that period was never "
        f"completed."
    )
    return {
        "question": q, "question_paraphrase": qp,
        "base_sents": base, "cf_sents": cf, "un_sents": un,
        "answer_base": a, "answer_cf": b,
        "facts_base": ["Emergency conditions activated the exception.", f"The form names {a} as sole delegate."],
        "facts_cf": ["Emergency conditions activated the exception.", f"The form names {b} as sole delegate."],
        "entities": people,
    }


def build_count_with_exclusions(rng, dom, difficulty) -> dict[str, Any]:
    obj = rng.choice(dom["objects"])
    total = rng.choice([7, 8, 9, 10])
    invalid = rng.choice([1, 2])
    duplicate = rng.choice([1, 2])
    valid = total - invalid - duplicate
    valid_cf = valid - 1
    q = f"How many of the {total} entries associated with the {obj} qualified as valid?"
    qp = f"Out of the {total} listed entries for the {obj}, state the number counted as valid."
    # A preliminary tally and a superseding reconciliation are both stated, so
    # both candidate counts appear in both answerable variants; the swap moves
    # only which tally is authoritative.
    base = [
        f"The register lists {total} entries associated with the {obj}.",
        f"A preliminary tally recorded {valid_cf} valid entries before the review closed.",
        f"On review, {invalid} entries were marked invalid and {duplicate} were struck as duplicates.",
        f"The reconciliation sheet, which supersedes the preliminary tally, counts {valid} valid entries.",
    ]
    cf = [
        f"The register lists {total} entries associated with the {obj}.",
        f"A preliminary tally recorded {valid} valid entries before the review closed.",
        f"On review, {invalid} entries were marked invalid and {duplicate} were struck as duplicates.",
        f"The reconciliation sheet, which supersedes the preliminary tally, counts {valid_cf} valid entries.",
    ]
    un = [
        f"The register lists {total} entries associated with the {obj}.",
        f"A preliminary tally recorded {valid_cf} valid entries before the review closed.",
        f"On review some entries were marked invalid and others struck as duplicates, but the "
        f"review column was truncated when the register was copied, so neither category total "
        f"can be recovered.",
        f"The reconciliation sheet that would supersede the preliminary tally was never filed, "
        f"and {valid} appears in the working papers only as an unconfirmed estimate.",
    ]
    return {
        "question": q, "question_paraphrase": qp,
        "base_sents": base, "cf_sents": cf, "un_sents": un,
        "answer_base": str(valid), "answer_cf": str(valid_cf),
        "facts_base": [f"{total} listed; {invalid} invalid and {duplicate} duplicate."],
        "facts_cf": [f"{total} listed; {invalid+1} invalid and {duplicate} duplicate."],
        "entities": [str(total)],
    }


def build_causal_disambiguation(rng, dom, difficulty) -> dict[str, Any]:
    obj = rng.choice(dom["objects"])
    causes = rng.sample(
        [
            "a power relay that failed its second diagnostic",
            "a coolant line that lost pressure overnight",
            "a hinge bracket that sheared under load",
            "a timer board that reset without warning",
            "a seal gasket that perished in storage",
        ],
        2,
    )
    q = f"What was the primary cause of the work stoppage involving the {obj}?"
    qp = f"State the cause identified as primary for the stoppage concerning the {obj}."
    # Both causes are always named in mid-sentence position so their surface
    # form is identical across variants and only the primary/secondary binding
    # moves. Capitalising one at a sentence start would change the string.
    base = [
        f"Two faults were open at the time of the stoppage involving the {obj}.",
        f"The review identifies {causes[0]} as the primary cause.",
        f"It records {causes[1]} as a secondary contributing issue.",
    ]
    cf = [
        f"Two faults were open at the time of the stoppage involving the {obj}.",
        f"The review identifies {causes[1]} as the primary cause.",
        f"It records {causes[0]} as a secondary contributing issue.",
    ]
    un = [
        f"Two faults were open at the time of the stoppage involving the {obj}.",
        f"The review lists {causes[0]} and {causes[1]} as the open faults.",
        f"The panel could not agree which was primary and closed the item as "
        f"unresolved, so neither fault is ranked in any surviving record.",
    ]
    return {
        "question": q, "question_paraphrase": qp,
        "base_sents": base, "cf_sents": cf, "un_sents": un,
        "answer_base": causes[0], "answer_cf": causes[1],
        "facts_base": [f"The review identifies {causes[0]} as primary."],
        "facts_cf": [f"The review identifies {causes[1]} as primary."],
        "entities": causes,
    }


def build_multi_condition_identity(rng, dom, difficulty) -> dict[str, Any]:
    people = _names(rng, difficulty, 4)
    a, b = people[0], people[1]
    q = "Who performed the final check under the two-condition rule?"
    qp = "Name the person who satisfied both conditions and performed the final check."
    base = [
        "The two-condition rule requires signing before the cutoff and entering later with the token.",
        f"{a} signed before the cutoff, and {a} also entered later carrying the token.",
        f"{b} signed before the cutoff but did not return with the token.",
        f"{people[2]} entered with the token but had not signed before the cutoff.",
    ]
    cf = list(base)
    cf[1] = f"{b} signed before the cutoff, and {b} also entered later carrying the token."
    cf[2] = f"{a} signed before the cutoff but did not return with the token."
    un = list(base)
    un[1] = (
        f"{a}, {b} and {people[2]} all signed before the cutoff, but the token "
        f"register for the later entry was destroyed, so no participant can be "
        f"shown to satisfy the second condition."
    )
    del un[3]
    del un[2]
    return {
        "question": q, "question_paraphrase": qp,
        "base_sents": base, "cf_sents": cf, "un_sents": un,
        "answer_base": a, "answer_cf": b,
        "facts_base": [f"{a} signed before the cutoff.", f"{a} entered later with the token."],
        "facts_cf": [f"{b} signed before the cutoff.", f"{b} entered later with the token."],
        "entities": people,
    }


def build_latest_verified_record(rng, dom, difficulty) -> dict[str, Any]:
    people = _names(rng, difficulty, 3)
    obj = rng.choice(dom["objects"])
    a, b = people[0], people[1]
    q = f"According to the final audit, who applied the seal to the {obj}?"
    qp = f"Per the concluding audit, identify the person who sealed the {obj}."
    # The swap exchanges who the superseded provisional accounts named and who
    # the final audit identifies, so both names appear in both variants.
    base = [
        f"Provisional witness accounts named {b} as having applied the seal to the {obj}.",
        "Those accounts were taken before the register was recovered and are expressly superseded.",
        f"The final audit identifies {a}.",
    ]
    cf = [
        f"Provisional witness accounts named {a} as having applied the seal to the {obj}.",
        "Those accounts were taken before the register was recovered and are expressly superseded.",
        f"The final audit identifies {b}.",
    ]
    un = [
        f"Provisional witness accounts named {b} as having applied the seal to the {obj}, "
        f"while a competing statement named {a}.",
        "Those accounts were taken before the register was recovered and are expressly superseded.",
        "The final audit was commissioned and its cover sheet survives, but the "
        "findings pages were never filed, so no superseding identification exists.",
    ]
    return {
        "question": q, "question_paraphrase": qp,
        "base_sents": base, "cf_sents": cf, "un_sents": un,
        "answer_base": a, "answer_cf": b,
        "facts_base": [f"The final audit identifies {a}.", "Provisional accounts are superseded."],
        "facts_cf": [f"The final audit identifies {b}.", "Provisional accounts are superseded."],
        "entities": people,
    }


BUILDERS: dict[str, Callable] = {
    "alias_resolution": build_alias_resolution,
    "object_state_tracking": build_object_state_tracking,
    "temporal_correction": build_temporal_correction,
    "rule_exception": build_rule_exception,
    "count_with_exclusions": build_count_with_exclusions,
    "causal_disambiguation": build_causal_disambiguation,
    "multi_condition_identity": build_multi_condition_identity,
    "latest_verified_record": build_latest_verified_record,
}


# --------------------------------------------------------------------------
# Filler and assembly
# --------------------------------------------------------------------------

# Filler templates. Each carries slots so that repeated draws differ in surface
# form; a scenario needs ~40 filler sentences and duplicate sentences would be an
# obvious artifact, so the effective pool must be far larger than the template
# count. Templates x slot values gives several thousand distinct sentences.
FILLER_TEMPLATES = [
    "The record was assembled from {src1}, {src2}, and {src3} rather than from a single narrator.",
    "Timestamps follow the local convention, and any entry without a countersignature was treated as {status}.",
    "The {role} maintained a running list of open items, of which {n1} were closed without incident.",
    "Routine movements between {p1} and {p2} were logged but are not material to the disputed points.",
    "A separate appendix covering {topic} is retained on file and is not reproduced here.",
    "{n1} participants submitted written statements after the fact; these were filed for completeness.",
    "Conditions in {p1} were noted each morning and did not interrupt the schedule.",
    "The {role} confirmed that the numbering scheme had not changed since the {ord} cycle.",
    "{n1} items were reclassified during the period for reasons unrelated to the determinations below.",
    "Correspondence concerning {topic} was archived separately under the general file.",
    "The {p1} inventory was recounted on {day} and matched the opening figure.",
    "Handover notes were exchanged at each shift change and countersigned by the {role}.",
    "An earlier draft contained a transcription error in the {topic} header, since corrected.",
    "Attendance was recorded for each session, and no session between {day} and {day2} was cancelled.",
    "The {role} noted that access arrangements for {p2} had been revised before the period began.",
    "Minor clerical amendments affecting {topic} are listed in the closing schedule.",
    "Access to {p2} required a countersigned pass for the duration of the period.",
    "The filing order follows the date of receipt rather than the date of the event.",
    "A duplicate set of {topic} slips was retained off-site as a precaution.",
    "Nothing in the general correspondence bears on the determinations recorded here.",
    "A reconciliation of {topic} was carried out on {day} and raised no exceptions.",
    "The {role} circulated a draft index on {day2}; {n1} corrections were returned.",
    "Storage conditions in {p1} were checked against the standing schedule and found compliant.",
    "A short addendum concerning {topic} was appended after the {ord} review.",
    "Queries raised during the {ord} session were answered in writing by the {role}.",
    "Supporting photographs of {p2} were catalogued but are held in a separate series.",
    "The {role} recorded that {n1} routine checks fell due within the period.",
    "Transport arrangements between {p1} and {p2} were confirmed on {day}.",
    "A note on {topic} was added to the margin and initialled by the {role}.",
    "The {ord} inventory pass covered {p1} and raised no discrepancy.",
]
FILLER_SLOTS = {
    "src1": ["meeting minutes", "duty rosters", "committee papers", "shift diaries"],
    "src2": ["transfer slips", "access logs", "movement dockets", "gate registers"],
    "src3": ["interviews", "written depositions", "handover notes", "telephone memoranda"],
    "status": ["provisional", "unconfirmed", "open", "pending review"],
    "topic": ["consumable stock", "insurance cover", "staffing levels", "utilities",
              "external contractors", "training records", "waste disposal", "signage"],
    "n1": ["two", "three", "four", "five", "six", "seven", "nine", "eleven"],
    "ord": ["previous", "preceding", "earlier", "last recorded", "prior"],
}

OPENERS = [
    "{title} is a consolidated account of the {event} at {site}.",
    "{title} records the {event} conducted at {site}.",
    "{title} sets out the findings of the {event} held at {site}.",
]

CLOSERS = [
    "The certificate was the last document added, so no later event changed the statuses summarized above.",
    "No supplementary filing was accepted after the closing date, and the summary above is final.",
    "The file was sealed at the close of the period, and nothing later amends the determinations above.",
]

TITLE_WORDS_A = ["Record", "Account", "Register", "Report", "Digest", "Summary"]
TITLE_WORDS_B = [
    "of the Hidden Key", "of the Closed Period", "of the Quiet Transfer",
    "of the Second Audit", "of the Late Return", "of the Sealed Appendix",
    "of the Divided Ledger", "of the Final Handover",
]


def _wc(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def assemble(rng, dom, spec, needle_profile, target_words) -> tuple[str, str, str]:
    """Return (base_ctx, cf_ctx, un_ctx) sharing identical filler."""
    site = rng.choice(dom["site"])
    title = f"{rng.choice(TITLE_WORDS_A)} {rng.choice(TITLE_WORDS_B)}"
    role = rng.choice(dom["roles"])
    p1, p2 = rng.sample(dom["places"], 2)
    opener = rng.choice(OPENERS).format(
        title=title, event=dom["event"], site=site
    )

    closer = rng.choice(CLOSERS)
    # Decisive sentences and the closer are added on top of the filler, so the
    # filler target must leave room for them or every context overshoots.
    longest_decisive = max(
        _wc(" ".join(spec[k])) for k in ("base_sents", "cf_sents", "un_sents")
    )
    filler_target = max(60, target_words - longest_decisive - _wc(closer) - _wc(opener))

    # Draw distinct filler sentences: templates x slot values gives a pool far
    # larger than the ~40 sentences a scenario needs, so nothing repeats.
    seen: set[str] = set()
    filler: list[str] = []
    guard = 0
    while _wc(" ".join(filler)) < filler_target and guard < 4000:
        guard += 1
        tpl = rng.choice(FILLER_TEMPLATES)
        slots = {k: rng.choice(v) for k, v in FILLER_SLOTS.items()}
        slots.update(
            role=rng.choice(dom["roles"]),
            p1=rng.choice(dom["places"]),
            p2=rng.choice(dom["places"]),
            day=rng.choice(WEEKDAYS),
            day2=rng.choice(WEEKDAYS),
        )
        sentence = tpl.format(**slots)
        if sentence in seen:
            continue
        seen.add(sentence)
        filler.append(sentence)
    while len(filler) > 1 and _wc(" ".join(filler)) > filler_target:
        filler.pop()

    def place(decisive: list[str]) -> str:
        n = len(filler)
        if needle_profile == "early":
            cut = max(1, n // 6)
            body = filler[:cut] + decisive + filler[cut:]
        elif needle_profile == "middle":
            cut = n // 2
            body = filler[:cut] + decisive + filler[cut:]
        elif needle_profile == "late":
            cut = max(1, (5 * n) // 6)
            body = filler[:cut] + decisive + filler[cut:]
        else:  # distributed
            body = list(filler)
            step = max(1, n // (len(decisive) + 1))
            for i, s in enumerate(decisive):
                body.insert(min(len(body), step * (i + 1) + i), s)
        return " ".join([opener] + body + [closer])

    return place(spec["base_sents"]), place(spec["cf_sents"]), place(spec["un_sents"])


def make_row(scenario_id, variant, dom_key, dom, spec, ctx, question, answer,
             accepted, facts, answerable, task_type, difficulty, needle, bucket,
             split) -> dict[str, Any]:
    suffix = {"base": "B", "paraphrase": "P", "counterfactual": "C", "unanswerable": "U"}[variant]
    prompt = f"CONTEXT\n{ctx}\n\nQUESTION\n{question}\n\nANSWER"
    return {
        "id": f"{scenario_id}-{suffix}",
        "scenario_id": scenario_id,
        "variant": variant,
        "domain": dom_key,
        "domain_label": dom["label"],
        "title": ctx.split(" is ")[0].split(" records ")[0].split(" sets out ")[0],
        "task_type": task_type,
        "difficulty": difficulty,
        "needle_profile": needle,
        "length_bucket": bucket,
        "answerable": answerable,
        "context": ctx,
        "question": question,
        "prompt": prompt,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "answer": answer,
        "accepted_answers": accepted,
        "normalized_accepted_answers": [a.strip().lower() for a in accepted],
        "supporting_facts": facts,
        "context_words": _wc(ctx),
        "prompt_words": _wc(prompt),
        "context_sha256": hashlib.sha256(ctx.encode("utf-8")).hexdigest(),
        "license": "CC0-1.0 (synthetic original text)",
        "intended_use": "evaluation and controlled training for the Rineya invisible-limit program",
        "split": split,
    }


def ensure_candidates_survive(rng, dom, spec) -> None:
    """Guarantee every candidate answer still appears in the unanswerable context.

    This is the property that makes the original suite hard: the unanswerable
    variant destroys the *binding* while leaving every entity in place, so the
    model cannot shortcut by noticing that a name has gone missing. Without this
    the unanswerable variant degenerates into "is the entity absent?", which is a
    far easier and quite different task, and would inflate abstention scores.
    """
    candidates = [spec["answer_base"], spec["answer_cf"]]
    un_text = " ".join(spec["un_sents"])
    missing = [c for c in candidates if c not in un_text]
    if not missing:
        return
    listed = ", ".join(candidates[:-1]) + f" and {candidates[-1]}"
    if all(c.replace(".", "").replace(" ", "").isdigit() for c in candidates):
        roster = (
            f"Working notes preserve {listed} as figures considered during the "
            f"review, but none was confirmed as the reconciled total."
        )
    elif any(c[:1].isupper() and " " in c for c in candidates):
        roles = rng.sample(dom["roles"], min(len(candidates), len(dom["roles"])))
        parts = [f"{c}, the {r}" for c, r in zip(candidates, roles)]
        roster = (
            "The principal participants recorded for the period were "
            + "; ".join(parts)
            + "."
        )
    else:
        roster = (
            f"Both {listed} appear elsewhere in the file as candidates, and "
            f"neither is excluded by any surviving record."
        )
    spec["un_sents"] = list(spec["un_sents"]) + [roster]


def generate(count: int, seed: int, split: str, prefix: str) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    dom_keys = list(DOMAINS)
    for i in range(count):
        sid = f"{prefix}-{i+1:04d}"
        dom_key = dom_keys[i % len(dom_keys)]
        dom = DOMAINS[dom_key]
        task = TASK_TYPES[(i // len(dom_keys)) % len(TASK_TYPES)]
        needle = NEEDLE_PROFILES[i % len(NEEDLE_PROFILES)]
        bucket = list(LENGTH_BUCKETS)[i % len(LENGTH_BUCKETS)]
        difficulty = "hard" if i % 8 < 5 else "medium"
        spec = BUILDERS[task](rng, dom, difficulty)
        ensure_candidates_survive(rng, dom, spec)
        base_ctx, cf_ctx, un_ctx = assemble(
            rng, dom, spec, needle, LENGTH_BUCKETS[bucket]
        )
        common = dict(
            scenario_id=sid, dom_key=dom_key, dom=dom, spec=spec,
            task_type=task, difficulty=difficulty, needle=needle,
            bucket=bucket, split=split,
        )
        rows.append(make_row(variant="base", ctx=base_ctx, question=spec["question"],
                             answer=spec["answer_base"], accepted=[spec["answer_base"]],
                             facts=spec["facts_base"], answerable=True, **common))
        rows.append(make_row(variant="paraphrase", ctx=base_ctx,
                             question=spec["question_paraphrase"],
                             answer=spec["answer_base"], accepted=[spec["answer_base"]],
                             facts=spec["facts_base"], answerable=True, **common))
        rows.append(make_row(variant="counterfactual", ctx=cf_ctx, question=spec["question"],
                             answer=spec["answer_cf"], accepted=[spec["answer_cf"]],
                             facts=spec["facts_cf"], answerable=True, **common))
        rows.append(make_row(variant="unanswerable", ctx=un_ctx, question=spec["question"],
                             answer="UNKNOWN", accepted=list(UNKNOWN_ACCEPTED),
                             facts=["The decisive record is missing, damaged, ambiguous, or explicitly unresolved."],
                             answerable=False, **common))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--prefix", required=True, help="scenario id prefix, e.g. RIL")
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    rows = generate(args.scenarios, args.seed, args.split, args.prefix)
    args.outdir.mkdir(parents=True, exist_ok=True)

    full = args.outdir / f"{args.split}_full.jsonl"
    prompts = args.outdir / f"{args.split}_prompts.jsonl"
    key = args.outdir / f"{args.split}_answer_key.jsonl"

    with full.open("w", encoding="utf-8", newline="\n") as f, \
         prompts.open("w", encoding="utf-8", newline="\n") as p, \
         key.open("w", encoding="utf-8", newline="\n") as k:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            p.write(json.dumps({
                "id": r["id"], "prompt": r["prompt"], "messages": r["messages"][:2],
            }, ensure_ascii=False) + "\n")
            k.write(json.dumps({
                key_: r[key_] for key_ in (
                    "id", "scenario_id", "variant", "domain", "domain_label",
                    "task_type", "difficulty", "needle_profile", "length_bucket",
                    "answerable", "answer", "accepted_answers", "split",
                )
            }, ensure_ascii=False) + "\n")

    print(json.dumps({
        "split": args.split, "scenarios": args.scenarios, "rows": len(rows),
        "seed": args.seed,
        "files": {p_.name: hashlib.sha256(p_.read_bytes()).hexdigest()
                  for p_ in (full, prompts, key)},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

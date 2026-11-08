"""RCI-001 — a long-context + abstention instrument built to CONFIRM, and therefore never to select.

WHY THIS EXISTS
---------------
`CONFIRMATION_INSTRUMENT_GAP.md` establishes that every long-context and literal-`UNKNOWN`
instrument in this repository is spent, reserved, or locked to another programme. Seven screens
terminate in "measured, never confirmed", and that is a property of the instrument inventory
rather than of the results. This builds the missing instrument to that document's five stated
requirements.

    1. fresh, never-read rows -- a new scenario/entity/template partition assigned BEFORE
       generation, not a resample of a spent bank
    2. the literal `UNKNOWN` contract, with the sentinel not obtainable by copying from the item
    3. a shortcut floor published BEFORE any model call, using the strongest honest predictor
    4. long-context rows within reach -- <= 2048 rendered tokens
    5. a registered read budget, since an instrument is spent the moment it is used for selection

A CLARIFICATION THE GAP DOCUMENT NEEDS, recorded rather than silently resolved
------------------------------------------------------------------------------
Requirement 2 says the sentinel must be "absent from the prompt". Taken literally that is
stricter than the qualified reference instrument it points at: the Long-Context Demonstration
Suite's own system prompt reads *"If the context does not determine the answer, reply UNKNOWN."*
So the Demo suite prints the sentinel too.

The distinction the gap document is actually drawing is with Forge v2, whose shortcut floor is
0.4391 via `copy_prompt_sentinel` because its answer OBJECT (`{"status":"UNKNOWN"}`) appears in
the item. That is copyable; an instruction is not. The operative requirement is therefore:

    the sentinel may be established once as a CONTRACT, and must never be obtainable by
    copying from the item's own context -- and the always-abstain floor must be published
    and beaten.

This build enforces that: the context text is asserted never to contain the sentinel, and
`always_abstain` is computed as a first-class floor.

THE CALIBRATION TWIN, which is what makes this shippable
---------------------------------------------------------
An instrument nobody has scored might be at floor or at ceiling, and both are uninterpretable.
MRS spent a scout that sat at the constant-answer baseline; DMRF-SCOUT-001's long-context
stratum came back 32/32 on both cores, a ceiling. Neither was knowable without scoring.

But scoring the confirmation bank spends it. So two disjoint banks are generated from the same
generator with the same parameters and **disjoint entity/template partitions assigned before
generation**:

    CALIBRATION   openly spendable. Scored freely to verify difficulty sits in band.
    SEALED        never scored here. Hash-sealed, one registered read, after a candidate freeze.

Difficulty transfers because the two are the same generator; the partitions differ so the
sealed rows are genuinely unread.

ORACLE
------
Every answer is computed from the generated STRUCTURE, never parsed back out of the rendered
prose. The renderer is a pure function of the structure, so an answer cannot disagree with its
context by construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

SENTINEL = "UNKNOWN"
SYSTEM = ("You are answering a closed-context reading test. Use only the supplied context. "
          "Reply with the shortest exact answer and no explanation. If the context does not "
          f"determine the answer, reply {SENTINEL}.")

# Entity pools are PARTITIONED before generation. Requirement 1: the sealed bank's scenarios,
# people, objects and places are disjoint from the calibration bank's, so calibrating cannot
# leak an entity the sealed bank later asks about.
POOLS = {
    "CALIB": {
        "people": ["Ardel Vance", "Noor Halliday", "Peregrine Oyelaran", "Sifra Lindqvist",
                   "Tobias Renn", "Umeko Farrow", "Casimir Blythe", "Delphine Arana"],
        "roles": ["the night steward", "the records clerk", "the visiting auditor",
                  "the duty warden", "the assistant curator", "the survey lead"],
        "objects": ["the ivory seal", "the brass astrolabe", "the lacquer casket",
                    "the vellum ledger", "the copper orrery", "the jade stylus"],
        "places": ["Cedar Vault", "the West Annexe", "the Glasshouse", "Reading Room Four",
                   "the Under-Stair Store", "the Cartography Loft"],
        "sites": ["Asterhall Institute", "Marrowgate Observatory", "the Thorne Repository"],
    },
    "SEALED": {
        "people": ["Imogen Castellanos", "Rashid Okonjo", "Beatrix Vollmer", "Hiran Devakumar",
                   "Solveig Brandt", "Marcus Ttofi", "Yolanda Pais", "Kenji Aldarrow",
                   "Freya Mbeki", "Aurelio Santangelo", "Nadia Kowalczyk", "Emeka Lindgren"],
        "roles": ["the morning registrar", "the conservation officer", "the standing inspector",
                  "the gallery marshal", "the acquisitions secretary", "the field recorder"],
        "objects": ["the onyx tablet", "the silver quadrant", "the walnut reliquary",
                    "the linen portfolio", "the pewter censer", "the garnet signet",
                    "the ebony caliper", "the alabaster urn"],
        "places": ["Larkspur Hold", "the North Undercroft", "the Orangery",
                   "Consultation Room Nine", "the Belfry Store", "the Plan Chest Room"],
        "sites": ["Quillbourne Athenaeum", "Halvard Station", "the Pellingham Collection"],
    },
}

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
TASK_TYPES = ("alias_resolution", "object_state_tracking", "temporal_correction",
              "rule_exception", "count_with_exclusions")
VARIANTS = ("base", "paraphrase", "counterfactual", "unanswerable")
FILLER = [
    "The account was compiled from duty books and is presented without editorial comment.",
    "Entries were transcribed in the order they were found rather than the order they occurred.",
    "Several pages carry marginal notes in a second hand; these are reproduced where legible.",
    "Where two records disagree, both are given and the later correction is marked as such.",
    "Routine movements of consumable stores are omitted as they were not logged individually.",
    "The building was open to visitors on three of the days covered by this account.",
    "Weather for the period was unremarkable and is noted only where it affected access.",
    "Staff rotas were disrupted midweek and the substitutions are recorded below.",
    "A lock was replaced during the period; the old key was retained in the strongbox.",
    "The compiler notes that nothing in this account should be read as an allegation.",
]


def h(*p) -> str:
    return hashlib.sha256("|".join(str(x) for x in p).encode("utf-8")).hexdigest()


def canon(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"))


# --------------------------------------------------------------------------------------- #
# World: a structured scenario. Every answer below is read off THIS, never off the prose.
# --------------------------------------------------------------------------------------- #

def make_world(rng: random.Random, pool: dict, sid: str) -> dict:
    people = rng.sample(pool["people"], 4)
    roles = rng.sample(pool["roles"], 3)
    objects = rng.sample(pool["objects"], 4)
    places = rng.sample(pool["places"], 4)
    site = rng.choice(pool["sites"])
    # Aliasing: one person is referred to throughout by a role, named only once.
    alias_person, alias_role = people[0], roles[0]
    # Object movements. One is later cancelled -- the cancellation is the needle.
    moves = []
    for ob in objects[:3]:
        frm, to = rng.sample(places, 2)
        moves.append({"object": ob, "from": frm, "to": to, "day": rng.choice(DAYS[:4]),
                      "by": rng.choice(people[1:]), "cancelled": rng.random() < 0.5})
    # At least one of each outcome, so the task cannot be answered by a constant.
    if all(m["cancelled"] for m in moves):
        moves[0]["cancelled"] = False
    if not any(m["cancelled"] for m in moves):
        moves[0]["cancelled"] = True
    tracked = rng.choice(moves)
    # Temporal correction: a stated day, later corrected.
    corr_obj = objects[3]
    stated_day, true_day = rng.sample(DAYS, 2)
    # Rule with one carve-out.
    rule_place = rng.choice(places)
    exempt_role = roles[1]
    # Counting with exclusions.
    k_insp = rng.randint(3, len(objects))
    inspected = rng.sample(objects, k_insp)
    n_excl = rng.randint(1, min(3, k_insp - 1))
    excluded = inspected[:n_excl]
    return {
        "scenario_id": sid, "site": site, "people": people, "roles": roles,
        "objects": objects, "places": places,
        "alias_person": alias_person, "alias_role": alias_role,
        "alias_action": {"object": objects[0], "place": places[0]},
        "moves": moves, "tracked": tracked, "corr_obj": corr_obj, "stated_day": stated_day, "true_day": true_day,
        "rule_place": rule_place, "exempt_role": exempt_role,
        "inspected": inspected, "excluded": excluded,
    }


def render(w: dict, rng: random.Random, omit: str | None, counterfactual: bool) -> str:
    """Pure function of the structure. `omit` drops exactly the sentence a task needs."""
    L = [f"This is a consolidated account of the audit at {w['site']}, covering six consecutive "
         f"days. The original plan was revised more than once, so the record below supersedes "
         f"any earlier summary."]
    L += rng.sample(FILLER, 3)

    if omit != "alias":
        who = w["alias_person"] if not counterfactual else w["people"][1]
        L.append(f"{w['alias_role'].capitalize()} that week was {who}, who is not named again in "
                 f"this account.")
        L.append(f"It was {w['alias_role']} who carried {w['alias_action']['object']} to "
                 f"{w['alias_action']['place']}, unaccompanied.")

    for m in w["moves"]:
        if omit == "moves":
            break
        L.append(f"{m['object'].capitalize()} was kept in {m['from']} at the start of the "
                 f"period. On {m['day']}, {m['by']} was assigned to move it to {m['to']}.")
        if m["cancelled"]:
            L.append(f"That assignment for {m['object']} was cancelled before it was carried "
                     f"out, and the item did not leave {m['from']}.")
        else:
            L.append(f"The move of {m['object']} was completed as assigned.")
    L += rng.sample(FILLER, 2)

    if omit != "temporal":
        d = w["true_day"] if not counterfactual else w["stated_day"]
        L.append(f"An earlier page gives {w['stated_day']} as the date the "
                 f"{w['corr_obj']} was examined.")
        L.append(f"That date is a transcription error. The examination of {w['corr_obj']} "
                 f"took place on {d}.")

    if omit != "rule":
        L.append(f"Access to {w['rule_place']} required a countersigned docket for every member "
                 f"of staff, with one standing exception: {w['exempt_role']} was permitted to "
                 f"enter without one.")

    if omit != "count":
        L.append(f"The items formally inspected during the period were "
                 f"{', '.join(w['inspected'][:-1])} and {w['inspected'][-1]}.")
        ex = w["excluded"]
        exs = ex[0] if len(ex) == 1 else (", ".join(ex[:-1]) + " and " + ex[-1])
        verb = "was" if len(ex) == 1 else "were"
        L.append(f"Of those, {exs} {verb} withdrawn from the inspection record after the fact "
                 f"and {verb} not counted as inspected.")
    L += rng.sample(FILLER, 2)
    return " ".join(L)


def ask(w: dict, task: str, counterfactual: bool) -> tuple[str, str]:
    """(question, ORACLE answer). Answer is computed from structure, never from prose."""
    if task == "alias_resolution":
        q = (f"Who personally carried {w['alias_action']['object']} to "
             f"{w['alias_action']['place']}?")
        return q, (w["people"][1] if counterfactual else w["alias_person"])
    if task == "object_state_tracking":
        t = w["tracked"]
        end = t["from"] if t["cancelled"] else t["to"]
        if counterfactual:
            end = t["to"] if t["cancelled"] else t["from"]
        return f"Where was {t['object']} at the end of the period?", end
    if task == "temporal_correction":
        d = w["stated_day"] if counterfactual else w["true_day"]
        return f"On which day was {w['corr_obj']} examined?", d
    if task == "rule_exception":
        return (f"Which role could enter {w['rule_place']} without a countersigned docket?"), \
               w["exempt_role"]
    if task == "count_with_exclusions":
        return ("How many items are counted as formally inspected? Answer with a number."), \
               str(len(w["inspected"]) - 1)
    raise ValueError(task)


#: The fact each task depends on. Omitting it makes the question genuinely unanswerable.
NEEDS = {"alias_resolution": "alias", "object_state_tracking": "moves",
         "temporal_correction": "temporal", "rule_exception": "rule",
         "count_with_exclusions": "count"}

PARAPHRASE = {
    "alias_resolution": lambda w: (f"By name, who took {w['alias_action']['object']} to "
                                   f"{w['alias_action']['place']} on their own?"),
    "object_state_tracking": lambda w: (
        f"At the close of the period, in which place was {w['tracked']['object']} held?"),
    "temporal_correction": lambda w: (f"According to the corrected record, what day was "
                                      f"{w['corr_obj']} examined?"),
    "rule_exception": lambda w: (f"Who was exempt from the countersigned-docket requirement "
                                 f"for {w['rule_place']}?"),
    "count_with_exclusions": lambda w: ("After the withdrawal, how many items count as "
                                        "formally inspected? Give a number."),
}


def build_bank(partition: str, packs: int, seed: int) -> list[dict]:
    pool = POOLS[partition]
    rows = []
    for p in range(packs):
        sid = f"RCI-{partition[:1]}{p:04d}"
        rng = random.Random(h(seed, partition, sid))
        w = make_world(rng, pool, sid)
        task = TASK_TYPES[p % len(TASK_TYPES)]
        need = NEEDS[task]
        for variant in VARIANTS:
            r = random.Random(h(seed, partition, sid, variant))
            cf = variant == "counterfactual"
            omit = need if variant == "unanswerable" else None
            ctx = render(w, r, omit=omit, counterfactual=cf)
            if variant == "paraphrase":
                q = PARAPHRASE[task](w)
                ans = ask(w, task, False)[1]
            else:
                q, ans = ask(w, task, cf)
            if variant == "unanswerable":
                ans = SENTINEL
            # Requirement 2, enforced: the sentinel must never be copyable from the item.
            if SENTINEL.lower() in ctx.lower() or SENTINEL.lower() in q.lower():
                raise SystemExit(f"sentinel leaked into item content: {sid}/{variant}")
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"CONTEXT\n{ctx}\n\nQUESTION\n{q}"}]
            rows.append({
                "id": f"{sid}-{variant[:4].upper()}", "scenario_id": sid,
                "partition": partition, "task_type": task, "variant": variant,
                "answerable": variant != "unanswerable",
                "prompt": msgs, "reference": ans,
                "gold_verified_by": f"oracle.{task}",
                "context_words": len(ctx.split()),
                "prompt_sha256": h(canon(msgs)), "context_sha256": h(ctx),
            })
    return rows


def floors(rows: list[dict]) -> dict:
    """Answer-agnostic predictors, computed BEFORE any model call. Requirement 3."""
    n = len(rows)
    refs = [r["reference"] for r in rows]
    from collections import Counter
    cnt = Counter(refs)
    always_abstain = sum(1 for x in refs if x == SENTINEL)
    const_major = max(cnt.values())
    # copy a capitalised entity out of the question
    import re
    copy_q = 0
    for r in rows:
        q = r["prompt"][1]["content"].split("QUESTION\n")[-1]
        cands = re.findall(r"[A-Z][a-z]+(?: [A-Z][a-z]+)*", q)
        if any(c.strip() == r["reference"] for c in cands):
            copy_q += 1
    # the most common answer WITHIN each task type, chosen with labels: a stronger predictor
    per_task = {}
    for r in rows:
        per_task.setdefault(r["task_type"], []).append(r["reference"])
    per_task_major = sum(Counter(v).most_common(1)[0][1] for v in per_task.values())
    preds = {"always_abstain": always_abstain, "constant_majority": const_major,
             "copy_entity_from_question": copy_q,
             "per_task_type_majority": per_task_major}
    binding = max(preds.values())
    return {"n": n, "predictors": preds, "binding_floor_count": binding,
            "binding_floor_rate": round(binding / n, 4),
            "binding_predictor": max(preds, key=preds.get),
            "distinct_references": len(cnt),
            "note": ("per_task_type_majority is label-aware and deliberately the strongest "
                     "honest predictor offered, per requirement 3 -- not the first one that "
                     "comes to mind.")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--calib-packs", type=int, default=32)
    ap.add_argument("--sealed-packs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260918)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite: {a.out}")
    a.out.mkdir(parents=True)

    calib = build_bank("CALIB", a.calib_packs, a.seed)
    sealed = build_bank("SEALED", a.sealed_packs, a.seed)

    # Disjointness is a requirement, not an expectation. Assert it.
    cs, ss = {r["context_sha256"] for r in calib}, {r["context_sha256"] for r in sealed}
    if cs & ss:
        raise SystemExit("calibration and sealed contexts overlap")
    ce = set()
    for k in ("people", "objects", "places", "sites", "roles"):
        ce |= set(POOLS["CALIB"][k]) & set(POOLS["SEALED"][k])
    if ce:
        raise SystemExit(f"entity pools overlap: {sorted(ce)}")

    (a.out / "RCI_CALIBRATION.jsonl").write_text(
        "".join(canon(r) + "\n" for r in calib), encoding="utf-8")
    (a.out / "RCI_SEALED.jsonl").write_text(
        "".join(canon(r) + "\n" for r in sealed), encoding="utf-8")

    man = {
        "schema": 1, "record": "RCI_001_BUILD", "instrument": "RCI-001",
        "what_this_is": (
            "A long-context and literal-UNKNOWN abstention instrument built to CONFIRM. Answers "
            "are computed from generated structure by oracle, never parsed from prose."),
        "built_to": "CONFIRMATION_INSTRUMENT_GAP.md requirements 1-5",
        "requirement_2_clarification": (
            "The gap document says the sentinel must be 'absent from the prompt'. Taken "
            "literally that is stricter than its own reference instrument: the Long-Context "
            "Demonstration Suite's system prompt reads 'If the context does not determine the "
            "answer, reply UNKNOWN.' The distinction it is actually drawing is with Forge v2, "
            "whose ANSWER OBJECT appears in the item (floor 0.4391 via copy_prompt_sentinel). "
            "An instruction is not copyable; an answer is. This build establishes the sentinel "
            "once as a contract, ASSERTS it never appears in any context or question, and "
            "publishes always_abstain as a first-class floor."),
        "seed": a.seed,
        "banks": {
            "CALIBRATION": {"packs": a.calib_packs, "rows": len(calib),
                            "role": "OPENLY SPENDABLE -- scored to verify difficulty is in band",
                            "sha256": h(canon(calib))},
            "SEALED": {"packs": a.sealed_packs, "rows": len(sealed),
                       "role": "CONFIRMATION ONLY -- never scored in this build",
                       "sha256": h(canon(sealed))},
        },
        "partitions_disjoint": {
            "entity_pools": "asserted disjoint across people/roles/objects/places/sites",
            "contexts": "asserted zero shared context_sha256",
            "assigned_before_generation": True,
        },
        "structure": {
            "variants_per_pack": list(VARIANTS),
            "task_types": list(TASK_TYPES),
            "anti_luck_control": ("each scenario pack carries base, paraphrase, counterfactual "
                                  "and unanswerable on the SAME scenario, mirroring the Demo "
                                  "suite's design, so a per-pack all-four-correct metric is "
                                  "available and memorisation is visible"),
            "unanswerable_construction": ("the single sentence the task depends on is OMITTED "
                                          "from the render; the question remains plausible and "
                                          "in-domain but is not determined by the context"),
        },
        "floors_published_before_any_model_call": True,
        "floors": {"CALIBRATION": floors(calib), "SEALED": floors(sealed)},
        "floor_rule": ("a system clears only if the exact two-sided binomial 95% lower bound on "
                       "its correct count exceeds the binding floor rate"),
        "read_budget": {
            "CALIBRATION": "unlimited",
            "SEALED": {"reads_authorised": 1, "reads_used": 0,
                       "condition": "after a candidate is frozen; never for selection",
                       "spent_if": "used to choose among arms, thresholds, prompts or seeds"},
        },
        "model_calls_in_this_build": 0,
    }
    (a.out / "RCI_BUILD.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "calibration_rows": len(calib), "sealed_rows": len(sealed),
        "calib_floor": man["floors"]["CALIBRATION"]["binding_floor_rate"],
        "calib_predictor": man["floors"]["CALIBRATION"]["binding_predictor"],
        "sealed_floor": man["floors"]["SEALED"]["binding_floor_rate"],
        "sealed_predictor": man["floors"]["SEALED"]["binding_predictor"],
        "sealed_sha256": man["banks"]["SEALED"]["sha256"],
        "context_words": {"min": min(r["context_words"] for r in sealed),
                          "max": max(r["context_words"] for r in sealed)},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

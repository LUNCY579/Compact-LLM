"""ALIAS-001 — does the alias regression track SALIENCE, or is it a uniform weakening?

    python probe/build_alias_probe.py --out <records>/alias001-20260918

**This is a MECHANISM SCOUT, not a confirmation instrument.** No seal, no read budget, openly
spendable and re-runnable. Building a sealed bank for an exploratory mechanism question would
misuse the one thing a seal is for.

## What it discriminates

`alias_resolution` is a replicated, purely subtractive regression — pooled across two sealed
banks, **0 repairs / 17 damages, p = 1.53e-05** — with no mechanism. See
`ALIAS_RESOLUTION_INVESTIGATION.md`.

Both prior generators named the aliased person exactly ONCE while the distractors were named up
to three times, so the correct answer was always the **least salient** person present. I
proposed salience-over-binding and it failed its control: the delta does reach for the salient
distractor, but so does the frozen source, on every one of its own errors, with an identical
profile (Fisher p = 1.000 on both banks). That refutes salience as a *differential* account —
it does not test whether salience drives the effect at all, because salience was never varied.

Here it is a **designed factor**. Three conditions, identical task and identical interrogative,
differing only in how often the answer is named relative to its distractors:

| condition | alias named | each distractor named | the answer is |
|---|---:|---:|---|
| `low` | 1 | 3 | the least salient person — the prior design |
| `equal` | 2 | 2 | no salience advantage either way |
| `high` | 3 | 1 | the most salient person |

**Registered predictions, written before any generation:**

    UNIFORM WEAKENING   the B-D gap is roughly constant across all three conditions. The delta
                        weakens a binding step; how loud the distractors are does not matter.

    SALIENCE            the gap shrinks monotonically low -> equal -> high, and may vanish at
                        `high`, where reaching for the loudest person gives the right answer.

These diverge on the `equal` condition, which is the cleanest cell: it removes the answer's
disadvantage without making it extractable.

**A floor caveat that is structural and is stated up front.** In `high` the answer IS the most
frequent person, so `most_frequent_candidate_in_context` scores near 1.000 there **by
construction**. That inflates the ABSOLUTE rates in that condition and says nothing about the
paired contrast: a deterministic rule emits the same string for both systems, yields zero
discordant pairs, and cannot manufacture a B-D difference. The question asked here is the
paired one, and `high`'s absolute numbers are not evidence of competence.

Extra mentions of a person never restate the role binding. They are neutral activity lines
("X signed the attendance book on Tuesday"), so salience moves without the answer becoming any
easier to *derive* — only easier to *reach for*.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "method"))
from floors import floors as gate_floors, gate  # noqa: E402

SENTINEL = "UNKNOWN"
SYSTEM = ("You are answering a closed-context reading test. Use only the supplied context. "
          "Reply with the shortest exact answer and no explanation. If the context does not "
          f"determine the answer, reply {SENTINEL}.")

#: A THIRD entity partition, disjoint from both RCI-001 and RCI-002 pools. Those banks are
#: spent, but reusing an entity the candidate has already been asked about confounds a fresh
#: measurement with whatever it did last time.
PEOPLE = ["Marisol Everdene", "Kwabena Lindholm", "Persephone Ratcliffe", "Yusuf Almeida-Stern",
          "Ingeborg Tanaka", "Octavian Mbeki-Ross", "Rosalind Feodorov", "Teodoro Nakashima",
          "Wilhelmina Achebe", "Sandor Villalobos", "Anouska Fairweather", "Dmitri Oyelaran-Cole"]
ROLES = ["night bindery clerk", "quadrant surveyor", "seal-room deputy",
         "provisions marshal", "muniments warden", "rota superintendent"]
OBJECTS = ["the malachite ewer", "the ashwood tripod", "the vermilion casket",
           "the pewter quadrant", "the jasper weight", "the cedar reliquary"]
PLACES = ["Thornbury Cellar", "the East Rotunda", "the Palm Store", "Study Alcove Six",
          "the Winch House", "the Sail Loft"]
SITES = ["Merridew Athenaeum", "Calderstone Field House", "the Ferrers Bequest"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

#: alias mentions -> each distractor's mentions. Symmetric by construction.
CONDITIONS = {"low": (1, 3), "equal": (2, 2), "high": (3, 1)}
#: `reorder` is a second independent shuffle rather than a different construction -- with
#: every line already shuffled per variant, it is an order-robustness cell.
VARIANTS = ("base", "paraphrase", "reorder")

FILLER = [
    "The account was compiled from duty books and carries no editorial comment.",
    "Several pages bear marginal notes in a second hand, reproduced where legible.",
    "Weather for the period was unremarkable and is noted only where it affected access.",
    "Page numbering restarts partway through and the original sequence is not recoverable.",
    "Costs were recorded separately and do not appear in this summary.",
]
ACTIVITY = [
    "{p} signed the attendance book on {d}.",
    "{p} was recorded as present for the {d} stocktake.",
    "{p} countersigned the {d} duty sheet.",
    "{p} collected keys from the porter on {d}.",
    "{p} appears in the {d} visitor register.",
]


def h(*parts) -> str:
    return hashlib.sha256("|".join(str(x) for x in parts).encode("utf-8")).hexdigest()


def canon(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"))


def build(condition: str, packs: int, seed: int) -> list[dict]:
    k_alias, k_dist = CONDITIONS[condition]
    rows = []
    for i in range(packs):
        sid = f"AL-{condition[:1].upper()}{i:04d}"
        rng = random.Random(h(seed, condition, sid))
        people = rng.sample(PEOPLE, 4)
        alias_person, distractors = people[0], people[1:]
        alias_role = rng.choice(ROLES)
        obj, place, site = rng.choice(OBJECTS), rng.choice(PLACES), rng.choice(SITES)

        for variant in VARIANTS:
            r = random.Random(h(seed, condition, sid, variant))
            # THE BINDING, stated exactly once in every condition. Only the neutral activity
            # lines below change in number, so salience moves and derivability does not.
            binding = (f"The {alias_role} that week was {alias_person}, who is not named again "
                       f"in this account.")
            action = (f"It was the {alias_role} who carried {obj} to {place}, unaccompanied.")

            lines = [binding, action]
            for p, k in [(alias_person, k_alias - 1)] + [(d, k_dist) for d in distractors]:
                for j in range(k):
                    lines.append(rng.choice(ACTIVITY).format(p=p, d=rng.choice(DAYS)))
            lines += rng.sample(FILLER, 2)

            head = (f"This is a consolidated account of the audit at {site}, covering six "
                    f"consecutive days, transcribed in the order the entries were found.")
            # EVERY line shuffled, including the binding and the action.
            #
            # The first draft kept those two at the front in two of three variants, and
            # `first_candidate_in_context` promptly scored 0.667 and 0.742 on the `low` and
            # `equal` conditions -- the answer was simply the first person named. The paired
            # contrast is immune to that, but absolute rates at a 0.7 floor are not worth
            # reading, and this probe is cheap enough that there is no excuse.
            #
            # The two sentences are independent assertions, so nothing depends on their order:
            # "The <role> that week was <name>" and "It was the <role> who carried X to Y" can
            # be read in either sequence and still determine the answer.
            body = lines[:]
            rng.shuffle(body)
            ctx = " ".join([head] + body)

            q = (f"Who personally carried {obj} to {place}?" if variant != "paraphrase" else
                 f"By name, who took {obj} to {place} on their own?")
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"CONTEXT\n{ctx}\n\nQUESTION\n{q}"}]

            # Assertions, not assumptions: the manipulation must have actually happened.
            n_alias = ctx.count(alias_person)
            n_dist = [ctx.count(d) for d in distractors]
            if n_alias != k_alias or any(x != k_dist for x in n_dist):
                raise SystemExit(f"{sid}/{variant}: salience not as specified -- alias "
                                 f"{n_alias} (want {k_alias}), distractors {n_dist} "
                                 f"(want {k_dist})")
            if SENTINEL.lower() in ctx.lower() or SENTINEL.lower() in q.lower():
                raise SystemExit(f"sentinel leaked into item content: {sid}/{variant}")

            rows.append({
                "id": f"{sid}-{variant[:3].upper()}", "scenario_id": sid,
                "condition": condition, "variant": variant, "stratum": condition,
                "partition": "PROBE",
                "task_type": "alias_resolution", "answerable": True,
                "alias_mentions": k_alias, "distractor_mentions": k_dist,
                "prompt": msgs, "reference": alias_person,
                "reference_normalised": alias_person.lower(),
                "context": ctx, "question": q,
                "gold_verified_by": "oracle.alias_resolution",
                "context_words": len(ctx.split()),
                "prompt_sha256": h(canon(msgs)), "context_sha256": h(ctx),
            })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--packs-per-condition", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260918)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite: {a.out}")

    rows = []
    for c in CONDITIONS:
        rows += build(c, a.packs_per_condition, a.seed)
    if len({r["context_sha256"] for r in rows}) != len(rows):
        raise SystemExit("duplicate contexts across the probe")

    fl = {c: gate_floors([r for r in rows if r["condition"] == c]) for c in CONDITIONS}
    a.out.mkdir(parents=True)
    (a.out / "ALIAS_PROBE.jsonl").write_text(
        "".join(canon(r) + "\n" for r in rows), encoding="utf-8")

    man = {
        "schema": 1, "record": "ALIAS_001_BUILD", "instrument": "ALIAS-001",
        "class": "MECHANISM SCOUT -- openly spendable, no seal, no read budget",
        "question": ("does the replicated alias regression track SALIENCE, or is it a uniform "
                     "weakening of a binding step?"),
        "registered_predictions": {
            "uniform_weakening": "the B-D gap is roughly constant across low / equal / high",
            "salience": "the gap shrinks monotonically low -> equal -> high and may vanish at high",
            "they_diverge_on": "the `equal` condition, which removes the answer's disadvantage "
                               "without making it extractable",
        },
        "design": {
            "conditions": {c: {"alias_mentions": k[0], "distractor_mentions": k[1]}
                           for c, k in CONDITIONS.items()},
            "held_fixed": ["the interrogative (congruent throughout)", "the binding sentence",
                           "the action sentence", "the answer", "the number of distractors"],
            "varied": "only the count of NEUTRAL activity lines naming each person",
            "why_neutral": ("extra mentions never restate the role binding, so salience moves "
                            "while derivability does not -- the answer becomes easier to reach "
                            "for, not easier to work out"),
            "variants_per_pack": list(VARIANTS),
        },
        "floors": {c: {k: v for k, v in f.items() if k != "note"} for c, f in fl.items()},
        "structural_floor_caveat": (
            "in `high` the answer IS the most frequent person, so "
            "`most_frequent_candidate_in_context` scores near 1.000 there BY CONSTRUCTION. That "
            "inflates the ABSOLUTE rates in that condition and cannot touch the paired "
            "contrast: a deterministic rule emits the same string for both systems, yields zero "
            "discordant pairs, and cannot manufacture a B-D difference. Read `high` paired only."),
        "entity_pools": "a THIRD partition, disjoint from both RCI-001 and RCI-002",
        "rows": len(rows), "packs_per_condition": a.packs_per_condition,
        "seed": a.seed, "model_calls_in_this_build": 0,
    }
    (a.out / "ALIAS_BUILD.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")

    print(f"{len(rows)} rows, {a.packs_per_condition} packs x {len(VARIANTS)} variants x "
          f"{len(CONDITIONS)} conditions\n")
    print(f"{'condition':<10}{'n':>5}{'floor':>8}  binding predictor")
    for c, f in fl.items():
        print(f"{c:<10}{f['n']:>5}{f['binding_floor_rate']:>8.3f}  "
              f"{f['binding_predictor']} ({f['binding_class']})")
    print(f"\ncontext words: {min(r['context_words'] for r in rows)}-"
          f"{max(r['context_words'] for r in rows)}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

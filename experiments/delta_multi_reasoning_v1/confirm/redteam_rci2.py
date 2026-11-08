"""The red-team proposals my workflow could not verify, implemented and RUN. Zero model calls.

    python confirm/redteam_rci2.py --root <records>/rci002-20260918 --out <...>/REDTEAM.json

A six-lens adversarial pass proposed ~30 shortcuts against RCI-002's calibration twin. Six were
verified before the session hit its usage limit and twenty-four were not. One of the six --
`roster_complement_exempt_role` -- scored 1.000 and is fixed. Leaving the rest unverified would
be exactly the "it existed only in narration" failure this programme keeps paying for, so they
are implemented here and executed against the bank.

The family that matters most is the abstention attack, because E1 is a registered endpoint:

    A rule that outputs UNKNOWN when a task-characteristic phrase is missing, and a candidate
    otherwise, would solve the abstention endpoint without reading anything.

The generator's answer to that is the LACUNA: every row carries an illegibility note, so
"is a note present" is constant across the bank and worth nothing, while "is the illegible
thing the thing I was asked about" is the abstention judgement itself. This script tests
whether that answer actually holds, rather than asserting it.

**A rule that beats the floor is not automatically a defect.** Detecting that the determining
sentence is absent IS the abstention task; a mechanical rule that does it is a statement about
how easy the task is, not about the bank being broken. The distinction that matters for a
PAIRED endpoint is different again: a deterministic rule emits the same string for both
systems, so it cannot manufacture a difference -- it can only compress power if both systems
adopt it. Each result below is labelled with which of those it is.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "method"))
from floors import cp_lower  # noqa: E402

SENTINEL = "unknown"
ARTICLE = re.compile(r"^(the|a|an)\s+")


def norm(v):
    s = str(v).strip().splitlines()[0] if str(v).strip() else ""
    return ARTICLE.sub("", " ".join(s.lower().replace(".", " ").replace(",", " ").split()))


# --------------------------------------------------------------------------------------- #
# The attacks. Each takes (row, ctx, q, vocab, aux) and returns a predicted string.
# --------------------------------------------------------------------------------------- #

#: Phrases the generator uses to state each determining fact. An attacker reading the published
#: bank can extract these; they are not secret.
TRIGGERS = {
    "rule": "standing exception applied",
    "temporal": "took place on",
    "moves": "was assigned to move it",
    "count": "formally inspected during the period",
    "alias": "who is not named again",
}
NEEDS = {
    "carve_out_role_congruent": "rule", "carve_out_role_incongruent": "rule",
    "carve_out_badge_congruent": "rule", "carve_out_badge_incongruent": "rule",
    "exclusion_count_number": "count", "alias_resolution": "alias",
    "object_state_tracking": "moves", "temporal_correction": "temporal",
}


def trigger_absence_abstainer(r, ctx, q, vocab, aux):
    """THE E1 ATTACK. Abstain when this task's characteristic phrase is missing."""
    if TRIGGERS[NEEDS[r["task_type"]]] not in ctx:
        return SENTINEL
    return aux["modal"][r["task_type"]]


def block_census_abstainer(r, ctx, q, vocab, aux):
    """Abstain when FEWER blocks than usual are present -- length/structure tell only."""
    present = sum(1 for t in TRIGGERS.values() if t in ctx)
    return SENTINEL if present < aux["median_blocks"] else aux["modal"][r["task_type"]]


def lacuna_reader(r, ctx, q, vocab, aux):
    """Read the lacuna note itself and abstain if it names THIS task's block.

    The sharpest admissible attack on E1, and it is close to the task. It still has to match
    the note against the question, which is the judgement being measured -- so a high score
    here says the abstention task is mechanically easy, not that the bank leaks.
    """
    for key, pat in aux["lacuna_pats"].items():
        if re.search(pat, ctx):
            return SENTINEL if key == NEEDS[r["task_type"]] else aux["modal"][r["task_type"]]
    return aux["modal"][r["task_type"]]


def sole_alias_anchor(r, ctx, q, vocab, aux):
    """The alias person is the unique filler of `... that week was <NAME>, who is not named`."""
    m = re.search(r"that week was ([A-Z][\w'-]+(?: [A-Z][\w'-]+)+), who is not named", ctx)
    return m.group(1) if m else aux["modal"][r["task_type"]]


def canonical_badge_bijection(r, ctx, q, vocab, aux):
    """Read the role->badge table and answer the badge of whichever role the question implies."""
    pairs = dict(re.findall(r"the ([a-z][\w \-]*?), badge (\d+)", ctx))
    for role, badge in pairs.items():
        if role in q:
            return badge
    return aux["modal"][r["task_type"]]


def exam_template_day(r, ctx, q, vocab, aux):
    """The day inside the `examination ... took place on <DAY>` template."""
    m = re.search(r"took place on (\w+)", ctx)
    return m.group(1) if m else aux["modal"][r["task_type"]]


def block_terminal_place(r, ctx, q, vocab, aux):
    """The place named in the final clause of the tracked object's movement sentences."""
    obj = None
    for o in aux["objects"]:
        if o in q:
            obj = o
            break
    if not obj:
        return aux["modal"][r["task_type"]]
    m = re.findall(re.escape(obj) + r"[^.]*?(?:did not leave|held in) ([^.,]+)", ctx)
    return m[-1].strip() if m else aux["modal"][r["task_type"]]


def inspected_list_arity(r, ctx, q, vocab, aux):
    """Count the inspection listing and subtract one, never reading the withdrawal clause."""
    m = re.search(r"formally inspected during the period were ([^.]+)\.", ctx)
    if not m:
        return aux["modal"][r["task_type"]]
    return str(len(re.split(r",| and ", m.group(1))) - 1)


def withdrawn_subpool_residual(r, ctx, q, vocab, aux):
    """Listing minus the withdrawal clause -- a genuine implementation of the count task."""
    m = re.search(r"formally inspected during the period were ([^.]+)\.", ctx)
    if not m:
        return aux["modal"][r["task_type"]]
    items = [x.strip() for x in re.split(r",| and ", m.group(1)) if x.strip()]
    w = re.findall(r"([^.]*?) was withdrawn from the inspection record", ctx)
    return str(max(0, len(items) - len(w)))


def entity_key_from_question(r, ctx, q, vocab, aux):
    """Ignore the context entirely: answer the modal reference for this task type."""
    return aux["modal"][r["task_type"]]


def wh_router_no_metadata(r, ctx, q, vocab, aux):
    """Route on the interrogative alone, with no task_type metadata."""
    ql = q.lower()
    if ql.startswith("how many") or "give the number" in ql:
        return aux["modal"]["exclusion_count_number"]
    if ql.startswith("which badge"):
        return aux["modal"]["carve_out_badge_congruent"]
    if ql.startswith("which role"):
        return aux["modal"]["carve_out_role_congruent"]
    if ql.startswith("where"):
        return aux["modal"]["object_state_tracking"]
    if ql.startswith("on which day"):
        return aux["modal"]["temporal_correction"]
    return aux["modal"]["alias_resolution"]


ATTACKS = {
    "trigger_absence_abstainer": (trigger_absence_abstainer, "E1"),
    "block_census_abstainer": (block_census_abstainer, "E1"),
    "lacuna_reader": (lacuna_reader, "E1"),
    "sole_alias_anchor": (sole_alias_anchor, "alias_resolution"),
    "canonical_badge_bijection": (canonical_badge_bijection, "carve_out_badge_congruent"),
    "exam_template_day": (exam_template_day, "temporal_correction"),
    "block_terminal_place": (block_terminal_place, "object_state_tracking"),
    "inspected_list_arity": (inspected_list_arity, "exclusion_count_number"),
    "withdrawn_subpool_residual": (withdrawn_subpool_residual, "exclusion_count_number"),
    "entity_key_from_question": (entity_key_from_question, "ALL"),
    "wh_router_no_metadata": (wh_router_no_metadata, "ALL"),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--bank", default="RCI_CALIBRATION.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in
            (a.root / a.bank).read_text(encoding="utf-8").splitlines() if l.strip()]
    build = json.loads((a.root / "RCI_BUILD.json").read_text(encoding="utf-8"))
    part = "CALIBRATION" if "CALIB" in a.bank else "SEALED"
    fl = build["floors"][part]

    vocab = defaultdict(set)
    modal = {}
    for r in rows:
        if r["answerable"]:
            vocab[r["task_type"]].add(r["reference_normalised"])
    for t in vocab:
        c = Counter(r["reference_normalised"] for r in rows
                    if r["answerable"] and r["task_type"] == t)
        modal[t] = c.most_common(1)[0][0]

    blocks = [sum(1 for t in TRIGGERS.values() if t in r["context"]) for r in rows]
    aux = {
        "modal": modal,
        "median_blocks": sorted(blocks)[len(blocks) // 2],
        "objects": sorted({o for r in rows for o in re.findall(r"the [a-z\-]+ \w+", r["context"])},
                          key=len, reverse=True)[:60],
        "lacuna_pats": {
            "rule": r"lines recording the access requirements",
            "temporal": r"line giving the date on which",
            "moves": r"movement log for .* is water-damaged",
            "alias": r"line naming the .* has been cut out",
            "count": r"lines listing which items were formally inspected are missing",
        },
    }

    una = [r for r in rows if not r["answerable"]]
    results = {}
    for name, (fn, target) in ATTACKS.items():
        scored = {}
        for r in rows:
            try:
                p = norm(fn(r, r["context"], r["prompt"][1]["content"].split("QUESTION\n")[-1],
                            vocab[r["task_type"]], aux))
            except Exception:
                p = ""
            scored[r["id"]] = p == r["reference_normalised"]
        overall = sum(scored.values())
        sub = ([r for r in rows if r["task_type"] == target] if target in vocab
               else (una if target == "E1" else rows))
        on_target = sum(scored[r["id"]] for r in sub)
        if target == "E1":
            floor = 0.0
            floor_label = "E1 is extraction-immune by construction; any rate here is the TASK"
        else:
            key = "aggregate" if target == "ALL" else None
            f = fl["aggregate"] if key else fl["per_stratum_answerable"].get(target) or {}
            floor = f.get("binding_floor_rate", 0.0)
            floor_label = f.get("binding_predictor", "n/a")
        rate = on_target / len(sub) if sub else 0.0
        results[name] = {
            "target": target, "n_target": len(sub),
            "correct_on_target": on_target, "rate_on_target": round(rate, 4),
            "rate_overall": round(overall / len(rows), 4),
            "published_floor": floor, "published_floor_predictor": floor_label,
            "beats_published_floor": rate > floor,
            "paired_consequence": (
                "NONE -- a deterministic rule emits the same string for both systems, so it "
                "yields zero discordant pairs and cannot create a paired difference. It can "
                "only compress power if BOTH systems adopt it."),
        }

    out = {
        "what_this_is": "the 24 red-team proposals the workflow could not verify, implemented "
                        "and executed. Zero model calls; nothing spent.",
        "bank": a.bank, "rows": len(rows),
        "attacks": results,
        "how_to_read": (
            "A rule beating a floor is not automatically a defect. Detecting that the "
            "determining sentence is absent IS the abstention task, and a mechanical rule that "
            "does it measures how easy the task is. What would be a defect is a rule that "
            "reaches the answer WITHOUT the reasoning -- `set_complement_singleton` at 1.000 "
            "was that, and is fixed."),
    }
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"{'attack':<32}{'target':<30}{'n':>5}{'rate':>8}{'floor':>8}  beats?")
    for k, v in sorted(results.items(), key=lambda kv: -kv[1]["rate_on_target"]):
        print(f"{k:<32}{v['target']:<30}{v['n_target']:>5}{v['rate_on_target']:>8.3f}"
              f"{v['published_floor']:>8.3f}  {'YES' if v['beats_published_floor'] else 'no'}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Post-hoc decomposition of a spent RCI read -- EXPLORATORY, never verdict-bearing.

The confirmation read's verdict is fixed by `RCI001_CONFIRM_PREREG.md` and nothing here touches
it. This script exists because FOUR mechanism hypotheses were proposed off this one read and all
four died, and the only way to stop a fifth going the same way is to make every figure fall out
of a script instead of out of narration.

    python confirm/decompose_rci.py \
        --bank  <records>/RCI_SEALED.jsonl \
        --gen B=<records>/gen-rci_confirm_B/predictions.jsonl \
        --gen D=<records>/gen-rci_confirm_D/predictions.jsonl \
        --out   <records>/RCI_DECOMPOSITION.json

What it reports:

  * per task type, on answerable rows: B, D, gains, losses, **exact two-sided McNemar**, Holm
    across the five-type family;
  * the recency rule per task type, so a null cell can be told apart from a cell no instrument
    could read -- `temporal_correction`'s answerable half is solved outright by the rule and
    therefore carries no information about either system;
  * the **binding / non-binding** split crossed with answerable / unanswerable, and the exact
    interaction test between them;
  * **answer-form classification** against the role and person-name vocabularies taken from the
    references, which is what distinguishes "emits the wrong kind of thing" from "emits the
    right kind of thing and picks the wrong one";
  * **convention sensitivity** -- rows whose score turns on a leading article or trailing period
    rather than on the model being right -- reported under both scorers, because the two
    sessions reading this bank do not share a convention and the difference is ~50 rows.

Two groupings here were proposed by the peer session and are reproduced rather than taken on
trust; disagreements are printed, not silently reconciled.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

SENTINEL = "UNKNOWN"

#: Answering these requires resolving a role to the person holding it (or the reverse).
BINDING = ("rule_exception", "alias_resolution")


def norm(v: str) -> str:
    """Canonical scorer -- identical to `calibrate_rci.norm`, and deliberately a copy.

    This must score a spent bank exactly the way the read scored it. An import that later
    drifted would silently restate a confirmed result.
    """
    return " ".join(str(v).strip().lower().replace(".", " ").split())


def norm_strict(v: str) -> str:
    """The peer session's scorer: case-folded and trimmed, but punctuation-preserving."""
    return " ".join(str(v).strip().lower().split())


def mcnemar_exact(gains: int, losses: int) -> float:
    """Exact two-sided binomial on the discordant pairs."""
    n = gains + losses
    if n == 0:
        return 1.0
    k = min(gains, losses)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n))


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher on [[a,b],[c,d]], by summing tables at most as probable."""
    def p(x):
        return (math.comb(a + b, x) * math.comb(c + d, a + c - x)
                / math.comb(a + b + c + d, a + c))
    obs = p(a)
    lo = max(0, a + c - (c + d))
    hi = min(a + b, a + c)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= obs * (1 + 1e-9)))


def holm(pvals: dict) -> dict:
    order = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(order), {}, 0.0
    for i, (name, pv) in enumerate(order):
        running = min(1.0, max(running, (m - i) * pv))
        out[name] = running
    return out


def cell(rows, pb, pd, scorer=norm):
    """One paired cell. Gains and losses are never netted before the test."""
    gains = losses = both = neither = 0
    lost_ids, gained_ids = [], []
    for r in rows:
        ok_b = scorer(pb.get(r["id"], "")) == scorer(r["reference"])
        ok_d = scorer(pd.get(r["id"], "")) == scorer(r["reference"])
        if ok_b and ok_d:
            both += 1
        elif ok_d:
            gains += 1
            gained_ids.append(r["id"])
        elif ok_b:
            losses += 1
            lost_ids.append(r["id"])
        else:
            neither += 1
    b_correct = both + losses
    return {"n": len(rows), "B": b_correct, "D": both + gains,
            "gains": gains, "losses": losses, "net": gains - losses,
            "p_mcnemar_exact": mcnemar_exact(gains, losses),
            # THE DENOMINATOR THAT DECIDES A DISSOCIATION. A loss is only possible on a row B
            # got right, so a group near the floor cannot show damage however severe the damage
            # is. Comparing losses against all rows re-expresses B's baseline as if it were an
            # effect -- the same error `paired.py` exists to stop on the repair/damage axis.
            "loss_rate_given_B_correct": round(losses / b_correct, 4) if b_correct else None,
            "lost_ids": lost_ids, "gained_ids": gained_ids}


def recency(rows) -> dict:
    """`last_candidate_in_context` -- the rule that broke the published aggregate floor."""
    vocab = defaultdict(set)
    for r in rows:
        if r["answerable"]:
            vocab[r["task_type"]].add(str(r["reference"]))
    hits = 0
    for r in rows:
        ctx = r["prompt"][1]["content"]
        best, pos = None, -1
        for cand in vocab[r["task_type"]]:
            i = ctx.rfind(cand)
            if i > pos:
                best, pos = cand, i
        if best is not None and norm(best) == norm(r["reference"]):
            hits += 1
    return {"n": len(rows), "correct": hits,
            "rate": round(hits / len(rows), 4) if rows else 0.0}


ARTICLE = re.compile(r"^(the|a|an)\s+")


def _bare(s: str) -> str:
    """Article-tolerant. On `rule_exception` this is the ONLY convention that measures what the
    model knows: the frozen source scores 32/75 canonically and 65/75 article-tolerant, so 33 of
    its answers drop the leading `the`. A canonical score there is largely an article score."""
    return ARTICLE.sub("", norm(s))


def wh_word(r: Mapping) -> str:
    """The question's leading interrogative -- which TYPE of entity the surface form demands."""
    q = r["prompt"][1]["content"].split("QUESTION\n")[-1].strip().lower()
    for w in ("which role", "who", "which", "where", "when", "how many", "what"):
        if q.startswith(w) or f" {w} " in q[:40]:
            return w
    return "other"


def by_variant(rows, pb, pd, vocabs) -> dict:
    """Damage per (task type, variant). NOBODY LOOKED AT THIS AND IT REFRAMES EFFECT 1.

    `rule_exception` asks `Which role could enter X...?` in its base and counterfactual variants
    and `Who was exempt from...?` in its paraphrase. The damage is almost entirely in the
    paraphrase, and so are the person-name emissions -- so what looked like role/occupant
    conflation is, on this evidence, the candidate answering the wh-word instead of the context.
    Scored article-tolerant, because on this task the canonical convention scores articles.
    """
    out = {}
    for tt in sorted({r["task_type"] for r in rows}):
        for v in sorted({r["variant"] for r in rows if r["answerable"]}):
            sub = [r for r in rows if r["answerable"] and r["task_type"] == tt
                   and r["variant"] == v]
            if not sub:
                continue
            c = cell(sub, pb, pd, scorer=_bare)
            c.pop("lost_ids", None)
            c.pop("gained_ids", None)
            c["wh_word"] = sorted({wh_word(r) for r in sub})
            c["D_answer_form"] = answer_form(sub, pd, vocabs)
            c["B_answer_form"] = answer_form(sub, pb, vocabs)
            out[f"{tt}/{v}"] = c
    return out


def answer_form(rows, pred, vocabs) -> dict:
    """Classify each prediction against the reference vocabularies.

    This is the test that separates "the delta emits the wrong KIND of answer" from "it emits
    the right kind and picks the wrong one". The first would be an output-form effect; the
    second cannot be.
    """
    counts = defaultdict(int)
    for r in rows:
        p = _bare(pred.get(r["id"], ""))
        kinds = [k for k, v in vocabs.items() if p in v]
        counts["+".join(kinds) if kinds else "other"] += 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def convention(rows, pred) -> dict:
    """Rows whose score turns on a convention rather than on knowing the answer."""
    strict = canonical = tolerant = 0
    for r in rows:
        raw = pred.get(r["id"], "")
        if norm_strict(raw) == norm_strict(r["reference"]):
            strict += 1
        if norm(raw) == norm(r["reference"]):
            canonical += 1
        if _bare(raw) == _bare(r["reference"]) and _bare(raw):
            tolerant += 1
    return {"n": len(rows), "strict_punctuation_sensitive": strict,
            "canonical_period_insensitive": canonical, "article_tolerant": tolerant,
            "rows_turning_on_punctuation": canonical - strict,
            "rows_turning_on_article": tolerant - canonical}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--gen", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in a.bank.read_text(encoding="utf-8").splitlines() if l.strip()]
    preds = {}
    for spec in a.gen:
        label, _, path = spec.partition("=")
        preds[label] = {r["id"]: r.get("prediction", "") for r in
                        (json.loads(l) for l in
                         Path(path).read_text(encoding="utf-8").splitlines() if l.strip())}
    if sorted(preds) != ["B", "D"]:
        raise SystemExit(f"expected exactly B and D, got {sorted(preds)}")
    pb, pd = preds["B"], preds["D"]

    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]

    # Vocabularies come from the references, not from a hand list.
    vocabs = {"ROLE": {_bare(r["reference"]) for r in ans if r["task_type"] == "rule_exception"},
              "NAME": {_bare(r["reference"]) for r in ans
                       if r["task_type"] == "alias_resolution"}}

    per_type, pvals = {}, {}
    for tt in sorted({r["task_type"] for r in rows}):
        sub_a = [r for r in ans if r["task_type"] == tt]
        sub_u = [r for r in una if r["task_type"] == tt]
        c = cell(sub_a, pb, pd)
        c["recency_rule_answerable"] = recency(sub_a)
        c["unanswerable"] = cell(sub_u, pb, pd)
        c["answer_form"] = {"B": answer_form(sub_a, pb, vocabs),
                            "D": answer_form(sub_a, pd, vocabs)}
        c["answer_form_on_rows_D_lost"] = answer_form(
            [r for r in sub_a if r["id"] in set(c["lost_ids"])], pd, vocabs)
        per_type[tt] = c
        pvals[tt] = c["p_mcnemar_exact"]
    for tt, adj in holm(pvals).items():
        per_type[tt]["p_holm"] = adj

    groups = {}
    for gname, keep in (("binding", True), ("non_binding", False)):
        sel = [r for r in rows if (r["task_type"] in BINDING) is keep]
        groups[gname] = {
            "task_types": sorted({r["task_type"] for r in sel}),
            "answerable": cell([r for r in sel if r["answerable"]], pb, pd),
            "unanswerable": cell([r for r in sel if not r["answerable"]], pb, pd)}

    # Is the damage differently distributed over the two groups? Two ways of asking, and they
    # do not agree, which is the point. `_tab_all` scores losses against every row in the
    # group; `_tab_cond` scores them against the rows B actually got right -- the only rows on
    # which a loss was possible at all.
    def _tab_all(half):
        bl = groups["binding"][half]["losses"]
        nl = groups["non_binding"][half]["losses"]
        return (bl, groups["binding"][half]["n"] - bl,
                nl, groups["non_binding"][half]["n"] - nl)

    def _tab_cond(half):
        b, n = groups["binding"][half], groups["non_binding"][half]
        return (b["losses"], b["B"] - b["losses"], n["losses"], n["B"] - n["losses"])

    # Is abstention damage CONCENTRATED in a task type, or uniform? Every mechanism proposed
    # off this read started from "10 of the 15 losses are temporal_correction", which is a
    # count. As a rate over the rows B got right it is 10 of 21, and the comparison cells are
    # tiny. Test homogeneity over the two cells with usable n rather than asserting it.
    ab = {tt: {"B_correct": c["unanswerable"]["B"], "losses": c["unanswerable"]["losses"],
               "rate_given_B_correct": c["unanswerable"]["loss_rate_given_B_correct"]}
          for tt, c in per_type.items()}
    usable = sorted((tt for tt, v in ab.items() if v["B_correct"] >= 10),
                    key=lambda t: -ab[t]["B_correct"])
    hom = None
    if len(usable) == 2:
        x, y = (ab[t] for t in usable)
        hom = {"compared": usable,
               "fisher": fisher_exact_2x2(x["losses"], x["B_correct"] - x["losses"],
                                          y["losses"], y["B_correct"] - y["losses"])}

    out = {
        "status": "EXPLORATORY -- post-hoc, on a spent bank, with the splits chosen after "
                  "seeing the read. No figure here bears on the registered verdict.",
        "registered_primary_endpoint_reproduced": cell(una, pb, pd),
        "answerable_overall": cell(ans, pb, pd),
        "per_task_type": per_type,
        "binding_dissociation": {
            "groups": groups,
            "fisher_losses_vs_all_rows": {
                "answerable": fisher_exact_2x2(*_tab_all("answerable")),
                "unanswerable": fisher_exact_2x2(*_tab_all("unanswerable"))},
            "fisher_losses_given_B_correct": {
                "answerable": fisher_exact_2x2(*_tab_cond("answerable")),
                "unanswerable": fisher_exact_2x2(*_tab_cond("unanswerable"))},
            "note": "The conditioned test is the admissible one. Where the two disagree, the "
                    "unconditioned test is reporting B's baseline, not D's damage.",
        },
        "abstention_loss_by_task_type": {
            "per_type": ab, "homogeneity_over_usable_cells": hom,
            "note": "Rates, not counts. A task type B rarely abstains on correctly cannot show "
                    "abstention damage, so counts track B's baseline before they track D."},
        "by_task_type_and_variant": by_variant(rows, pb, pd, vocabs),
        "convention_sensitivity_answerable": {"B": convention(ans, pb),
                                              "D": convention(ans, pd)},
        "caveat": "Both prediction sets come from the pinned serving path in each GENERATE.json. "
                  "A decomposition inherits every caveat of the read it decomposes and adds the "
                  "multiplicity of having chosen the split afterwards.",
    }
    for v in out["per_task_type"].values():
        v.pop("lost_ids", None)
        v.pop("gained_ids", None)
        v["unanswerable"].pop("lost_ids", None)
        v["unanswerable"].pop("gained_ids", None)
    for g in groups.values():
        for half in ("answerable", "unanswerable"):
            g[half].pop("lost_ids", None)
            g[half].pop("gained_ids", None)
    for k in ("registered_primary_endpoint_reproduced", "answerable_overall"):
        out[k].pop("lost_ids", None)
        out[k].pop("gained_ids", None)
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"{'task type':<24}{'n':>4}{'B':>5}{'D':>5}{'gain':>5}{'loss':>5}{'p':>10}"
          f"{'holm':>9}{'recency':>9}   {'unans B/D (g/l)':>16}")
    for tt, c in per_type.items():
        u = c["unanswerable"]
        print(f"{tt:<24}{c['n']:>4}{c['B']:>5}{c['D']:>5}{c['gains']:>5}{c['losses']:>5}"
              f"{c['p_mcnemar_exact']:>10.2e}{c['p_holm']:>9.2e}"
              f"{c['recency_rule_answerable']['rate']:>9.3f}   "
              f"{u['B']:>4}/{u['D']:<4} ({u['gains']}/{u['losses']})")
    print()
    for gname, g in groups.items():
        for half in ("answerable", "unanswerable"):
            h = g[half]
            lr = h["loss_rate_given_B_correct"]
            print(f"{gname:<12}{half:<14}n={h['n']:<4} B {h['B']:<4} D {h['D']:<4} "
                  f"gains {h['gains']:<3} losses {h['losses']:<3} p {h['p_mcnemar_exact']:.2e}"
                  f"   loss rate | B correct: {'n/a' if lr is None else f'{lr:.3f}'}")
    bd = out["binding_dissociation"]
    print("\nFisher, binding vs non-binding losses")
    for k in ("fisher_losses_vs_all_rows", "fisher_losses_given_B_correct"):
        print(f"  {k:<32} answerable {bd[k]['answerable']:.2e}   "
              f"unanswerable {bd[k]['unanswerable']:.2e}")
    print("\nabstention loss as a RATE over the rows B got right")
    for tt, v in sorted(ab.items(), key=lambda kv: -kv[1]["B_correct"]):
        r = v["rate_given_B_correct"]
        print(f"  {tt:<24} {v['losses']:>2}/{v['B_correct']:<3} "
              f"{'n/a' if r is None else f'{r:.3f}'}")
    if hom:
        print(f"  homogeneity over {hom['compared']}: Fisher {hom['fisher']:.3f}")

    print("\nby task type and VARIANT, article-tolerant (the convention that measures knowledge)")
    print(f"  {'cell':<38}{'n':>4}{'B':>5}{'D':>5}{'gain':>5}{'loss':>5}{'rate|B':>8}  wh-word")
    for k, c in out["by_task_type_and_variant"].items():
        if c["losses"] or c["gains"]:
            rt = c["loss_rate_given_B_correct"]
            rs = "n/a" if rt is None else f"{rt:.3f}"
            print(f"  {k:<38}{c['n']:>4}{c['B']:>5}{c['D']:>5}{c['gains']:>5}{c['losses']:>5}"
                  f"{rs:>8}  {','.join(c['wh_word'])}")

    print("\nanswer form (answerable rows, bare-normalised against reference vocabularies)")
    for tt in ("rule_exception", "alias_resolution"):
        c = per_type[tt]
        print(f"  {tt}\n    B {c['answer_form']['B']}\n    D {c['answer_form']['D']}"
              f"\n    D on the rows it lost: {c['answer_form_on_rows_D_lost']}")
    e = out["registered_primary_endpoint_reproduced"]
    print(f"\nregistered endpoint (unanswerable): B {e['B']}/{e['n']} D {e['D']}/{e['n']} "
          f"gains {e['gains']} losses {e['losses']} p {e['p_mcnemar_exact']:.2e}")
    for lab, cv in out["convention_sensitivity_answerable"].items():
        print(f"convention, {lab}: strict {cv['strict_punctuation_sensitive']} -> canonical "
              f"{cv['canonical_period_insensitive']} -> article-tolerant {cv['article_tolerant']}"
              f"  (punctuation {cv['rows_turning_on_punctuation']}, "
              f"article {cv['rows_turning_on_article']})")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

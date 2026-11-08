"""THE GATE. Score a model on DMRF-SCOUT-001 and decide whether the bank is usable at all.

A scout is worthless if the models under test score at or below what an answer-agnostic
predictor gets. The MRS lane spent one that did, and every arm contrast on it was
uninterpretable. This file is the check that was missing there, and nothing in the MR /
routing study may run until it returns USABLE.

THE RULE, published in SCOUT_BUILD.json before any model call
-------------------------------------------------------------
A system clears a stratum only if the **exact two-sided binomial 95% lower bound** on its
correct count exceeds that stratum's **binding floor rate** -- the strongest of
`constant_majority`, `copy_before` and `always_abstain`. Same rule, same test and same
lower-bound construction the acquisition instrument uses, so a clearance here means what a
clearance there means.

PER-STRATUM SCORING RULES, each mirroring an existing qualified instrument
-------------------------------------------------------------------------
  transition, multi_step   canonical JSON equality -- `process_probe.py`'s `canonical` rule
  arith_competence         `scoring.answer_matches` -- IMPORTED from the pilot clone, not
                           copied, because that module exists precisely because copied
                           scorers drifted and inverted a published conclusion
  longctx_mixed            first line, normalised, exact membership -- the suite's own rule;
                           plus literal `UNKNOWN` for abstention

Abstention is reported as sub-analyses of the pooled long-context stratum -- correct
abstention, false abstention on answerable rows -- and never as a stratum with its own floor.

Also performs the AUTHORIZED contamination check: the scout's prompts and references against
SELECT-VIEWS-001 and the RRP-015 training file, both of which this experiment is permitted to
read. Quarantined banks are NOT opened; they are handled by manifest elsewhere.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PILOT_SCORER = Path(r"C:/Users/clclc/BusinessProject/ChatBubble-retention-pilot-20260917/"
                    r"experiments/rab001_exploration")
SELECT_VIEWS = Path(r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/"
                    r"select-views-001/SELECT_VIEWS.jsonl")
MICROFIT_TRAIN = Path(r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/"
                      r"rrp015-20260917/control-prepared/microfit-train.jsonl")

ABSTAIN = "UNKNOWN"


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def binom_at_least(k: int, n: int, p: float) -> float:
    if k <= 0:
        return 1.0
    if p <= 0.0:
        return 0.0
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def cp_lower(k: int, n: int, alpha: float = 0.05) -> float:
    """Exact two-sided 95% lower bound, same construction as the acquisition instrument."""
    if k <= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if binom_at_least(k, n, mid) > alpha / 2:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def min_k_to_clear(n: int, floor: float) -> int:
    """Smallest correct count at this n whose exact 95% lower bound exceeds `floor`.

    At n = 6 with a floor of 0.0526 this is 3, so a stratum on which the model scores 1/6 or
    2/6 CANNOT be certified however real its signal is. That is a power limit of the split, not
    a property of the bank, and conflating the two would either discard a usable stratum or
    invite tuning the task until a six-item test says yes.
    """
    for k in range(0, n + 1):
        if cp_lower(k, n) > floor:
            return k
    return n + 1


def norm_lc(v: str) -> str:
    """The long-context suite's normalisation, verbatim."""
    return " ".join(str(v).strip().lower().replace(".", " ").split())


def score_item(row: dict, pred: str, answer_matches) -> dict:
    """Return the per-item outcome under that stratum's registered rule."""
    s = row["stratum"]
    ref = row["reference"]
    text = (pred or "").strip()
    out = {"parsed": None, "abstained": text == ABSTAIN}
    if s in ("transition", "multi_step"):
        try:
            obj = json.loads(text)
            out["parsed"] = True
            out["correct"] = canon(obj) == ref
        except (ValueError, TypeError):
            out["parsed"] = False
            out["correct"] = False
    elif s == "arith_competence":
        out["parsed"] = bool(text)
        out["correct"] = bool(answer_matches(text, ref))
    elif s == "longctx_mixed":
        first = text.splitlines()[0] if text else ""
        out["parsed"] = bool(text)
        out["correct"] = norm_lc(first) == norm_lc(ref)
    else:
        raise ValueError(f"unregistered stratum {s!r}")
    return out


def contamination_check(rows: list[dict]) -> dict:
    """Authorized overlap check against the two banks this experiment may read."""
    scout_prompts = {r["prompt_sha256"] for r in rows}
    scout_refs: dict[str, int] = {}
    for r in rows:
        scout_refs[r["reference"]] = scout_refs.get(r["reference"], 0) + 1

    def load(p: Path) -> list[dict]:
        if not p.exists():
            return []
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

    out = {"scout_items": len(rows), "scout_distinct_prompts": len(scout_prompts)}
    for name, path, ref_key in (("select_views_001", SELECT_VIEWS, "reference"),
                                ("rrp015_microfit_train", MICROFIT_TRAIN, "reference")):
        other = load(path)
        if not other:
            out[name] = {"status": "ABSENT", "path": str(path)}
            continue
        # Prompt-text overlap. The scout is generated from a different generator entirely, so
        # any hit would be a real problem rather than a coincidence.
        other_text = set()
        for o in other:
            p = o.get("prompt")
            other_text.add(canon(p) if isinstance(p, list) else str(p))
        scout_text = {canon(r["prompt"]) for r in rows}
        shared_refs = {k for k in scout_refs if k in {str(o.get(ref_key)) for o in other}}
        out[name] = {
            "status": "CHECKED", "rows": len(other),
            "identical_prompt_texts": len(scout_text & other_text),
            "shared_reference_values": len(shared_refs),
            "shared_reference_note": ("reference-value coincidence, not item overlap; both "
                                      "banks emit small JSON states and short numerics"),
        }
    out["quarantined_banks"] = ("handled by QUARANTINE.json manifest, not opened here; the "
                                "three ANY_READ-barred banks were not read")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scout", type=Path, required=True)
    ap.add_argument("--build", type=Path, required=True, help="SCOUT_BUILD.json")
    ap.add_argument("--split", required=True, choices=("CAL", "CHECK"))
    ap.add_argument("--gen", type=Path, nargs="+", required=True,
                    help="one or more gen dirs, each with predictions.jsonl and GENERATE.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    sys.path.insert(0, str(PILOT_SCORER))
    from scoring import answer_matches, self_test          # noqa: E402
    self_test()                                            # refuse to score with a broken scorer

    build = json.loads(a.build.read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in a.scout.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["split"] == a.split]
    by_id = {r["id"]: r for r in rows}

    systems: dict[str, dict] = {}
    for gd in a.gen:
        gmeta = json.loads((gd / "GENERATE.json").read_text(encoding="utf-8"))
        preds = {json.loads(l)["id"]: json.loads(l)
                 for l in (gd / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
                 if l.strip()}
        label = gmeta["label"]
        missing = [i for i in by_id if i not in preds]
        detail = []
        for iid, row in by_id.items():
            if iid in preds:
                sc = score_item(row, preds[iid].get("prediction", ""), answer_matches)
                sc.update({"id": iid, "stratum": row["stratum"], "world_id": row["world_id"],
                           "router_case": row.get("router_case"),
                           "unanswerable": bool(row.get("unanswerable")),
                           "prediction": preds[iid].get("prediction", "")})
            else:
                sc = {"id": iid, "stratum": row["stratum"], "world_id": row["world_id"],
                      "router_case": row.get("router_case"),
                      "unanswerable": bool(row.get("unanswerable")),
                      "correct": False, "parsed": False, "abstained": False,
                      "prediction": None, "absent": True}
            detail.append(sc)

        per_stratum = {}
        for s in build["strata"]:
            sub = [d for d in detail if d["stratum"] == s]
            if not sub:
                continue
            k, n = sum(1 for d in sub if d["correct"]), len(sub)
            # POOLED floor, not the split's own: see SCOUT_BUILD.json. A floor estimated on
            # 6 CAL items is mechanically >= 1/6 and says nothing about the generator.
            fl = build["floors"][f"{s}/POOLED"]
            lb = cp_lower(k, n)
            floor = fl["binding_floor_rate"]
            need = min_k_to_clear(n, floor)
            clears = bool(lb > floor)
            if clears:
                verdict_s = "CLEARS"
            elif (k / n) > floor:
                # Point estimate above the floor but the interval cannot exclude it at this n.
                verdict_s = "INCONCLUSIVE_UNDERPOWERED"
            else:
                verdict_s = "BELOW_FLOOR"
            per_stratum[s] = {
                "n": n, "correct": k, "rate": round(k / n, 4),
                "ci_lower_95": round(lb, 4),
                "binding_floor_rate": floor,
                "binding_predictor": fl["binding_predictor"],
                "clears_floor": clears,
                "verdict": verdict_s,
                "min_correct_to_clear_at_this_n": need,
                "parsed": sum(1 for d in sub if d["parsed"]),
            }
        # Abstention as sub-analyses of the pooled long-context stratum, never its own stratum.
        lc = [d for d in detail if d["stratum"] == "longctx_mixed"]
        un = [d for d in lc if d["unanswerable"]]
        an = [d for d in lc if not d["unanswerable"]]
        abstention = {
            "pool_n": len(lc), "answerable_n": len(an), "unanswerable_n": len(un),
            "correct_abstention": sum(1 for d in un if d["abstained"]),
            "false_abstention_on_answerable": sum(1 for d in an if d["abstained"]),
            "answerable_correct": sum(1 for d in an if d["correct"]),
            "note": ("Reported inside the pooled stratum. Scored alone, unconditional "
                     "abstention would be perfect and the floor would be 1.000."),
        }
        systems[label] = {
            "model": gmeta["model"], "dtype": gmeta["dtype"],
            "transformers": gmeta["transformers"],
            "rows_skipped_context_overflow": gmeta["rows_skipped_context_overflow"],
            "decoding": gmeta["decoding"], "numerics": gmeta.get("numerics"),
            "items_absent_from_predictions": len(missing),
            "total_correct": sum(1 for d in detail if d["correct"]),
            "total_n": len(detail),
            "per_stratum": per_stratum, "abstention": abstention,
            "by_router_case": {
                c: {"n": sum(1 for d in detail if d["router_case"] == c),
                    "correct": sum(1 for d in detail if d["router_case"] == c and d["correct"])}
                for c in sorted({d["router_case"] for d in detail if d["router_case"]})},
            "per_item": detail,
        }

    # ------------------------------------------------------------------------- the verdict
    def worst(s):
        vs = [sy["per_stratum"].get(s, {}).get("verdict") for sy in systems.values()]
        if "CLEARS" in vs:
            return "CLEARS"
        if "INCONCLUSIVE_UNDERPOWERED" in vs:
            return "INCONCLUSIVE_UNDERPOWERED"
        return "BELOW_FLOOR"

    by_stratum = {s: worst(s) for s in build["strata"]}
    usable = [s for s, v in by_stratum.items() if v == "CLEARS"]
    inconclusive = [s for s, v in by_stratum.items() if v == "INCONCLUSIVE_UNDERPOWERED"]
    unusable = [s for s, v in by_stratum.items() if v == "BELOW_FLOOR"]
    verdict = ("USABLE" if len(usable) == len(build["strata"])
               else "PARTIALLY_USABLE" if usable else "NOT_USABLE")

    report = {
        "schema": 1, "record": "DMRF_SCOUT_FLOOR_REPORT",
        "instrument": "DMRF-SCOUT-001", "split": a.split,
        "scout_sha256": build["scout_sha256"],
        "floor_rule": build["floor_rule"],
        "floor_estimated_on": ("POOLED -- all 152 items per stratum. The binding floor is a "
                               "property of the generator, not of a split; a per-split floor on "
                               "6 CAL items is mechanically >= 1/6 and cannot be cleared by any "
                               "honest score."),
        "scorer_source": str(PILOT_SCORER / "scoring.py"),
        "scorer_self_test": "PASSED",
        "verdict": verdict,
        "strata_clearing_their_floor_by_at_least_one_system": usable,
        "strata_inconclusive_underpowered_at_this_n": inconclusive,
        "strata_below_floor": unusable,
        "per_stratum_verdict": by_stratum,
        "power_note": ("At n = 6 per stratum on CAL the exact 95% lower bound cannot exceed a "
                       "0.0526 floor below 3/6, so CAL can only certify strata where a system "
                       "scores >= 50%. A stratum marked INCONCLUSIVE_UNDERPOWERED has a point "
                       "estimate above its floor and must be decided on its CHECK score, at "
                       "n = 32, BEFORE any contrast on it is reported. Insufficient sample "
                       "size means inconclusive, not pass and not fail."),
        "what_the_verdict_means": (
            "USABLE: every stratum has at least one system whose exact 95% lower bound exceeds "
            "the answer-agnostic floor, so arm contrasts on it are interpretable. NOT_USABLE or "
            "a stratum in strata_no_system_clears means contrasts on that stratum are NOT "
            "interpretable and must not be reported -- that is the MRS failure this gate exists "
            "to catch."),
        "systems": systems,
        "contamination_check": contamination_check(rows),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "verdict": verdict,
        "strata_below_floor": unusable,
        "strata_inconclusive": inconclusive,
        "systems": {k: {"total": f"{v['total_correct']}/{v['total_n']}",
                        "per_stratum": {s: f"{d['correct']}/{d['n']} lb={d['ci_lower_95']} "
                                           f"floor={d['binding_floor_rate']} "
                                           f"need>={d['min_correct_to_clear_at_this_n']} "
                                           f"{d['verdict']}"
                                        for s, d in v["per_stratum"].items()},
                        "abstention": v["abstention"]["correct_abstention"],
                        "false_abstention": v["abstention"]["false_abstention_on_answerable"]}
                    for k, v in systems.items()},
        "contamination": {k: v for k, v in report["contamination_check"].items()
                          if isinstance(v, dict)},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

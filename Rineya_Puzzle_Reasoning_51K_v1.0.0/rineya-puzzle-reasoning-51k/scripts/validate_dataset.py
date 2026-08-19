#!/usr/bin/env python3
"""Full symbolic validation for every Rineya Puzzle Reasoning row."""
from __future__ import annotations

import argparse
import collections
import itertools
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import generate_dataset as g


def accepted(row: dict[str, Any], expected: str) -> bool:
    target = g.normalize_answer(expected)
    return any(g.normalize_answer(x) == target for x in row["accepted_answers"])


def verify_hats(row):
    s=row["structure"]; colors=s["colors"]; stock=collections.Counter(s["inventory"]); n=len(s["people"])
    worlds=[w for w in itertools.product(colors,repeat=n) if all(collections.Counter(w)[c]<=stock[c] for c in colors)]
    actual=tuple(s["actual_world"]); current=list(worlds)
    for ann in s["announcements"]:
        speaker=ann["speaker"]; status=ann["knows"]
        assert g.hat_knows(current,actual,speaker)==status
        current=g.hat_update(current,speaker,status)
    q=s["query"]
    if q["type"]=="color":
        vals={w[q["target"]] for w in current}; assert len(vals)==1; expected=next(iter(vals))
    else:
        vals={g.hat_knows(current,w,q["target"]) for w in current}; assert len(vals)==1; expected="Yes" if next(iter(vals)) else "No"
    assert accepted(row,expected)


def verify_knights(row):
    s=row["structure"]; n=len(s["people"]); sols=[]
    for a in itertools.product([False,True],repeat=n):
        if all(g.eval_knight_statement(st,a)==a[st["speaker"]] for st in s["statements"]): sols.append(a)
    assert len(sols)==1
    expected="Knight" if sols[0][s["query_target"]] else "Knave"
    assert accepted(row,expected)


def verify_alibis(row):
    s=row["structure"]; n=len(s["people"]); k=s["required_true"]
    sols=[c for c in range(n) if sum(g.eval_alibi(st,c) for st in s["statements"])==k]
    assert len(sols)==1; assert accepted(row,s["people"][sols[0]])


def verify_boxes(row):
    s=row["structure"]; n=len(s["boxes"])
    cand=[p for p in itertools.permutations(s["contents"]) if sum(p[i]==s["labels"][i] for i in range(n))==s["correct_labels"]]
    for o in s["observations"]:
        if o["kind"]=="is": cand=[p for p in cand if p[o["box"]]==o["item"]]
        else: cand=[p for p in cand if p[o["box"]]!=o["item"]]
    vals={p[s["query_box"]] for p in cand}; assert len(vals)==1; assert accepted(row,next(iter(vals)))


def solve_grid(s):
    people,pets,cities=s["people"],s["pets"],s["cities"]
    cand=[(p,c) for p in itertools.permutations(pets) for c in itertools.permutations(cities)]
    cand=[x for x in cand if all(g.eval_grid_clue(cl,x,people) for cl in s["clues"])]
    assert len(cand)==1; return cand[0]

def verify_grid(row):
    s=row["structure"]; pet,city=solve_grid(s); q=s["query"]; people=s["people"]
    if q["kind"]=="pet_of_person": expected=pet[q["person"]]
    elif q["kind"]=="city_of_person": expected=city[q["person"]]
    elif q["kind"]=="owner_of_pet": expected=people[pet.index(q["pet"])]
    else: expected=city[pet.index(q["pet"])]
    assert accepted(row,expected)


def verify_order(row):
    s=row["structure"]; n=len(s["names"]); cand=[p for p in g.order_perms(n) if all(g.eval_order_clue(c,p) for c in s["clues"])]
    assert len(cand)==1; p=cand[0]; pos={x:i for i,x in enumerate(p)}; q=s["query"]
    if q["kind"]=="position": expected=str(pos[q["entity"]]+1)
    elif q["kind"]=="who_at": expected=s["names"][p[q["position"]]]
    elif q["kind"]=="first": expected=s["names"][p[0]]
    else: expected=s["names"][p[-1]]
    assert accepted(row,expected)


def verify_rules(row):
    s=row["structure"]
    facts={(x["entity"],x["predicate"],x["polarity"]):["given"] for x in s["facts"]}
    rules=[]
    for r in s["rules"]:
        pre=tuple((x["predicate"],x["polarity"]) for x in r["premises"]); c=r["conclusion"]
        rules.append((pre,(c["predicate"],c["polarity"])))
    known,_,_=g.rule_closure(facts,rules,len(s["entities"])); q=s["query"]; key=(q["entity"],q["predicate"],q["polarity"])
    expected="Yes" if key in known else "No" if (q["entity"],q["predicate"],not q["polarity"]) in known else "Unknown"
    assert accepted(row,expected)


def verify_tracking(row):
    s=row["structure"]; state=list(s["initial"])
    for op in s["operations"]: state=g.apply_tracking_op(state,op)
    q=s["query"]
    expected=state[q["position"]] if q["kind"]=="object_at" else str(state.index(q["object"])+1)
    assert accepted(row,expected)


def verify_code(row):
    s=row["structure"]; cand,matrix=g.code_space(s["length"],list(s["alphabet"]),s["repeats"])
    idx=list(range(len(cand)))
    lookup={c:i for i,c in enumerate(cand)}
    for cl in s["clues"]:
        gi=lookup[tuple(cl["guess"])]; fb=cl["exact"]*10+cl["misplaced"]
        idx=[i for i in idx if int(matrix[gi,i])==fb]
    assert len(idx)==1; expected="".join(cand[idx[0]])
    assert accepted(row,expected)


def verify_calendar(row):
    s=row["structure"]; n=len(s["events"]); days=[None]*n; days[s["root"]]=s["root_day"]
    changed=True
    while changed:
        changed=False
        for e in s["edges"]:
            a,b,d=e["a"],e["b"],e["delta"]
            if days[b] is not None and days[a] is None: days[a]=days[b]+d; changed=True
            elif days[a] is not None and days[b] is None: days[b]=days[a]-d; changed=True
            elif days[a] is not None and days[b] is not None: assert days[a]==days[b]+d
    assert all(x is not None for x in days); q=s["query"]; day=days[q["target"]]
    expected=f"Day {day}" if q["kind"]=="day_number" else g.WEEKDAYS[(s["start_weekday"]+day-1)%7]
    assert accepted(row,expected)


def family_relation(s,x,y):
    gender=s["gender"]; parents={k:set(v) for k,v in s["parents"].items()}
    def siblings(a,b): return a!=b and bool(parents.get(a,set())&parents.get(b,set()))
    def gps(a): return {z for p in parents.get(a,set()) for z in parents.get(p,set())}
    if x in parents.get(y,set()): return "father" if gender[x]=="M" else "mother"
    if y in parents.get(x,set()): return "son" if gender[x]=="M" else "daughter"
    if siblings(x,y): return "brother" if gender[x]=="M" else "sister"
    if x in gps(y): return "grandfather" if gender[x]=="M" else "grandmother"
    if y in gps(x): return "grandson" if gender[x]=="M" else "granddaughter"
    if any(siblings(x,p) for p in parents.get(y,set())): return "uncle" if gender[x]=="M" else "aunt"
    if any(siblings(y,p) for p in parents.get(x,set())): return "nephew" if gender[x]=="M" else "niece"
    if any(siblings(px,py) for px in parents.get(x,set()) for py in parents.get(y,set())): return "first cousin"
    return None

def verify_family(row):
    s=row["structure"]; q=s["query"]; expected=family_relation(s,q["x"],q["y"]); assert expected; assert accepted(row,expected)


def verify_path(row):
    s=row["structure"]; q=s["query"]; dist,count,prev,_,_=g.dijkstra_unique(len(s["nodes"]),[tuple(e) for e in s["edges"]],q["start"],s["directed"])
    assert count[q["target"]]==1; path=[];cur=q["target"]
    while cur is not None:
        path.append(cur)
        if cur==q["start"]: break
        cur=prev[cur]
    path=path[::-1]
    if q["kind"]=="distance": expected=str(dist[q["target"]])
    elif q["kind"]=="route": expected=" → ".join(s["nodes"][i] for i in path)
    else: expected=s["nodes"][path[1]]
    assert accepted(row,expected)


def gaussian_solve(equations):
    n=len(equations[0])-1; a=[[Fraction(x) for x in row] for row in equations]
    r=0
    for c in range(n):
        pivot=next((i for i in range(r,len(a)) if a[i][c]),None); assert pivot is not None
        a[r],a[pivot]=a[pivot],a[r]; v=a[r][c]; a[r]=[x/v for x in a[r]]
        for i in range(len(a)):
            if i!=r and a[i][c]:
                v=a[i][c]; a[i]=[x-v*y for x,y in zip(a[i],a[r])]
        r+=1
    return [a[i][-1] for i in range(n)]

def verify_arithmetic(row):
    s=row["structure"]; sol=gaussian_solve(s["equations"]); assert all(x.denominator==1 for x in sol)
    q=s["query"]
    if s["kind"]=="animals": idx=0 if q=="x" else 1
    elif s["kind"]=="three_types": idx=int(q)
    else: idx=s["variables"].index(q)
    assert accepted(row,str(sol[idx].numerator))


def verify_boolean(row):
    s=row["structure"]; q=s["query"]
    if q["kind"]=="output": expected="On" if g.eval_bool_expr(s["expression"],s["known"]) else "Off"
    else:
        unknown=q["unknown"]; matches=[]
        for v in [False,True]:
            a=dict(s["known"]);a[unknown]=v
            if g.eval_bool_expr(s["expression"],a)==q["output"]: matches.append(v)
        assert len(matches)==1; expected="On" if matches[0] else "Off"
    assert accepted(row,expected)


def verify_hanoi(row):
    s=row["structure"]; start=tuple(s["start"]);goal=tuple(s["goal"]);d=g.hanoi_distances(s["n_disks"],goal)[start];q=s["query"]
    if q["kind"]=="min_moves": expected=f"{d} moves"
    else:
        opts=[v for v in g.hanoi_adj(s["n_disks"])[start] if g.hanoi_distances(s["n_disks"],goal)[v]==d-1]
        assert len(opts)==1; expected=g.hanoi_move(start,opts[0],s["peg_names"])
    assert accepted(row,expected)


VERIFIERS={
 "epistemic_hats":verify_hats,"knights_knaves":verify_knights,"prisoner_alibis":verify_alibis,
 "mislabeled_boxes":verify_boxes,"logic_grid":verify_grid,"linear_order":verify_order,
 "rule_deduction":verify_rules,"object_tracking":verify_tracking,"code_breaking":verify_code,
 "calendar_reasoning":verify_calendar,"family_relations":verify_family,"shortest_path":verify_path,
 "arithmetic_constraints":verify_arithmetic,"boolean_switches":verify_boolean,"hanoi_planning":verify_hanoi,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--limit", type=int, default=None, help="Validate only the first N rows globally; layout checks are skipped")
    parser.add_argument(
        "--expect-default-layout",
        action="store_true",
        help="Require the canonical 2,720/272/272/136 rows per family layout",
    )
    args = parser.parse_args()
    root = args.root

    ids: set[str] = set()
    prompts: set[str] = set()
    structures: set[str] = set()
    counts: collections.Counter[tuple[str, str]] = collections.Counter()
    templates: dict[str, set[str]] = collections.defaultdict(set)
    failures: list[dict[str, Any]] = []
    checked = 0

    def record_failure(**failure: Any) -> None:
        if len(failures) < 20:
            failures.append(failure)

    stop = False
    for split in g.SPLIT_COUNTS:
        path = root / "data" / f"{split}.jsonl"
        if not path.exists():
            record_failure(file=path.name, line=None, id=None, family=None, error="missing split file")
            continue
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if args.limit is not None and checked >= args.limit:
                    stop = True
                    break
                row: dict[str, Any] = {}
                try:
                    row = json.loads(line)
                    checked += 1
                    assert row["split"] == split
                    assert row["family"] in VERIFIERS
                    assert row["id"] not in ids
                    assert row["prompt_sha256"] not in prompts
                    assert row["structural_signature"] not in structures
                    assert row["prompt_sha256"] == g.signature(row["prompt"])
                    assert row["structural_signature"] == g.signature(
                        {"family": row["family"], "structure": row["structure"]}
                    )
                    assert row["reasoning_steps"]
                    assert row["final_answer"].strip()
                    assert len(row["final_answer"].split()) <= 16
                    assert any(
                        g.normalize_answer(row["final_answer"]) == g.normalize_answer(value)
                        for value in row["accepted_answers"]
                    )
                    VERIFIERS[row["family"]](row)
                    ids.add(row["id"])
                    prompts.add(row["prompt_sha256"])
                    structures.add(row["structural_signature"])
                    counts[(split, row["family"])] += 1
                    templates[split].add(row["template_id"])
                except Exception as exc:
                    record_failure(
                        file=path.name,
                        line=line_number,
                        id=row.get("id"),
                        family=row.get("family"),
                        error=repr(exc),
                    )
                    if len(failures) >= 20:
                        stop = True
                        break
        if stop or len(failures) >= 20:
            break

    split_counts: dict[str, int] = {
        split: sum(counts[(split, family)] for family in VERIFIERS)
        for split in g.SPLIT_COUNTS
    }
    family_counts: dict[str, dict[str, int]] = {
        split: {family: counts[(split, family)] for family in VERIFIERS}
        for split in g.SPLIT_COUNTS
    }

    if not failures and args.limit is None:
        for split in g.SPLIT_COUNTS:
            observed = family_counts[split]
            values = list(observed.values())
            if not values or min(values) <= 0:
                record_failure(
                    file=f"{split}.jsonl",
                    line=None,
                    id=None,
                    family=None,
                    error=f"missing one or more families: {observed}",
                )
            elif len(set(values)) != 1:
                record_failure(
                    file=f"{split}.jsonl",
                    line=None,
                    id=None,
                    family=None,
                    error=f"unbalanced family counts: {observed}",
                )
            if args.expect_default_layout:
                expected = g.SPLIT_COUNTS[split]
                for family, value in observed.items():
                    if value != expected:
                        record_failure(
                            file=f"{split}.jsonl",
                            line=None,
                            id=None,
                            family=family,
                            error=f"expected {expected} rows, observed {value}",
                        )

        split_names = list(g.SPLIT_COUNTS)
        for i, left in enumerate(split_names):
            for right in split_names[i + 1 :]:
                overlap = templates[left] & templates[right]
                if overlap:
                    record_failure(
                        file="template_layout",
                        line=None,
                        id=None,
                        family=None,
                        error=f"template overlap between {left} and {right}: {sorted(overlap)}",
                    )

        stats_path = root / "reports" / "dataset_stats.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            expected_split_counts = stats.get("split_counts", {})
            if expected_split_counts and expected_split_counts != split_counts:
                record_failure(
                    file=str(stats_path.relative_to(root)),
                    line=None,
                    id=None,
                    family=None,
                    error=f"split counts disagree with statistics: expected {expected_split_counts}, observed {split_counts}",
                )

    report = {
        "status": "PASS" if not failures else "FAIL",
        "checked_rows": checked,
        "failure_count": len(failures),
        "failures": failures,
        "unique_ids": len(ids),
        "unique_prompts": len(prompts),
        "unique_structures": len(structures),
        "split_counts": split_counts,
        "family_counts": family_counts,
        "layout_check": "skipped_for_limit" if args.limit is not None else (
            "canonical" if args.expect_default_layout else "balanced_auto"
        ),
    }
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)

if __name__=="__main__": main()

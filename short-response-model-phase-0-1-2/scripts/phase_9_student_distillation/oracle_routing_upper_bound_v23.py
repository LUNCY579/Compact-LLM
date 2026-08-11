from __future__ import annotations

"""Compute best-of-candidate oracle routing upper bounds for v2.3 / v2.3.1 diagnostics."""
import argparse, json
from itertools import combinations
from pathlib import Path
from typing import Any


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def direct_row_map(p: Path) -> dict[str, dict[str, Any]]:
    return {str(r.get("id")): r for r in read_jsonl(p)}


def direct_correct_map(p: Path) -> dict[str, bool]:
    return {i: bool(r.get("correct")) for i, r in direct_row_map(p).items()}


def direct_answerable_correct_map(p: Path) -> dict[str, bool]:
    rows = direct_row_map(p)
    return {i: bool(r.get("correct")) for i, r in rows.items() if str(r.get("decision") or "answer") == "answer"}


def rag_correct(r: dict[str, Any]) -> bool:
    if str(r.get("condition")) != "real_rag":
        return False
    k = str(r.get("case_class") or "")
    if k == "supported":
        return bool(r.get("supported_answer_correct"))
    if k == "unsupported":
        return bool(r.get("abstained"))
    if k == "conflict":
        return bool(r.get("conflict_safe_response"))
    if k == "social":
        return bool(r.get("social_usable"))
    return False


def rag_correct_map(p: Path) -> dict[str, bool]:
    return {str(r.get("id")): rag_correct(r) for r in read_jsonl(p) if str(r.get("condition")) == "real_rag"}


def union(maps: dict[str, dict[str, bool]]) -> dict[str, Any]:
    ids = sorted(set().union(*(set(m) for m in maps.values())))
    per = {n: sum(bool(m.get(i)) for i in ids) for n, m in maps.items()}
    u = sum(any(bool(m.get(i)) for m in maps.values()) for i in ids)
    best = max(per.values()) if per else 0
    return {
        "case_count": len(ids),
        "candidate_correct_counts": per,
        "candidate_accuracies": {n: (c / len(ids) if ids else None) for n, c in per.items()},
        "oracle_union_correct": u,
        "oracle_union_accuracy": u / len(ids) if ids else None,
        "oracle_gain_over_best_single": (u - best) / len(ids) if ids and per else None,
    }


def unique_wins(maps: dict[str, dict[str, bool]]) -> dict[str, Any]:
    ids = sorted(set().union(*(set(m) for m in maps.values())))
    names = list(maps)
    unique = {n: 0 for n in names}
    for i in ids:
        winners = [n for n in names if bool(maps[n].get(i))]
        if len(winners) == 1:
            unique[winners[0]] += 1
    pairwise = {}
    for a, b in combinations(names, 2):
        both = sum(1 for i in ids if bool(maps[a].get(i)) and bool(maps[b].get(i)))
        either = sum(1 for i in ids if bool(maps[a].get(i)) or bool(maps[b].get(i)))
        pairwise[f"{a}|{b}"] = {"intersection": both, "union": either, "jaccard": (both / either) if either else None}
    three_way = None
    if len(names) >= 3:
        three_way = {}
        for trio in combinations(names, 3):
            key = "|".join(trio)
            three_way[key] = sum(1 for i in ids if all(bool(maps[n].get(i)) for n in trio))
    return {"unique_wins": unique, "pairwise": pairwise, "three_way_intersections": three_way, "case_count": len(ids)}


def parse_specs(specs: list[str], mode: str) -> dict[str, dict[str, bool]]:
    if mode == "rag":
        fn = rag_correct_map
    elif mode == "direct_answerable":
        fn = direct_answerable_correct_map
    else:
        fn = direct_correct_map
    out = {}
    for s in specs:
        n, p = s.split("=", 1)
        out[n] = fn(Path(p))
    return out


def section_md(title: str, v: dict[str, Any]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        f"- Cases: **{v['case_count']}**",
        f"- Oracle union accuracy: **{v['oracle_union_accuracy']}**",
        f"- Gain over best single: **{v['oracle_gain_over_best_single']}**",
        "",
        "Candidate accuracies:",
    ]
    for n, x in (v.get("candidate_accuracies") or {}).items():
        lines.append(f"- {n}: `{x}`")
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", action="append", default=[])
    ap.add_argument("--rag", action="append", default=[])
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    ap.add_argument("--section-label", default="")
    a = ap.parse_args()
    out: dict[str, Any] = {"schema_version": "2.3.1"}
    if a.section_label:
        out["section_label"] = a.section_label
    if a.direct:
        dmaps = parse_specs(a.direct, "direct")
        out["direct"] = union(dmaps)
        out["direct_answerable_only"] = union(parse_specs(a.direct, "direct_answerable"))
        out["direct_overlap"] = unique_wins(dmaps)
    if a.rag:
        rmaps = parse_specs(a.rag, "rag")
        out["rag"] = union(rmaps)
        out["rag_overlap"] = unique_wins(rmaps)
    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# v2.3 Oracle Routing Upper Bound", ""]
    if a.section_label:
        lines += [f"Section: **{a.section_label}**", ""]
    for k in ("direct", "direct_answerable_only", "rag"):
        if k in out:
            lines += section_md(k, out[k])
    if "direct_overlap" in out:
        ov = out["direct_overlap"]
        lines += ["## direct unique / pairwise / three-way", "", f"Unique wins: `{ov['unique_wins']}`", ""]
        for k, v in (ov.get("pairwise") or {}).items():
            lines.append(f"- {k}: intersection=`{v['intersection']}`, jaccard=`{v['jaccard']}`")
        lines.append("")
        if ov.get("three_way_intersections"):
            lines.append(f"Three-way: `{ov['three_way_intersections']}`")
            lines.append("")
    a.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

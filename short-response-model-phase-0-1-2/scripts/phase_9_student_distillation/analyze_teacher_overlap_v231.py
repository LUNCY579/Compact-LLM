from __future__ import annotations

"""Teacher unique-win / pairwise / three-way overlap for v2.3.1 diagnostics."""
import argparse, json
from itertools import combinations
from pathlib import Path
from typing import Any


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def answerable_correct(p: Path) -> dict[str, bool]:
    out = {}
    for r in read_jsonl(p):
        if str(r.get("decision") or "answer") != "answer":
            continue
        out[str(r.get("id"))] = bool(r.get("correct"))
    return out


def analyze(maps: dict[str, dict[str, bool]], greedy: dict[str, bool] | None = None) -> dict[str, Any]:
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
        pairwise[f"{a}|{b}"] = {
            "intersection": both,
            "union": either,
            "jaccard": (both / either) if either else None,
            "a_only": sum(1 for i in ids if bool(maps[a].get(i)) and not bool(maps[b].get(i))),
            "b_only": sum(1 for i in ids if bool(maps[b].get(i)) and not bool(maps[a].get(i))),
        }
    three = None
    if len(names) >= 3:
        three = {}
        for trio in combinations(names, 3):
            three["|".join(trio)] = sum(1 for i in ids if all(bool(maps[n].get(i)) for n in trio))
    per = {n: sum(bool(m.get(i)) for i in ids) for n, m in maps.items()}
    union_n = sum(any(bool(m.get(i)) for m in maps.values()) for i in ids)
    recoverable = None
    if greedy is not None:
        gids = [i for i in ids if i in greedy]
        recoverable = {
            "greedy_wrong_answerable": sum(1 for i in gids if not bool(greedy.get(i))),
            "recoverable_by_any_teacher": sum(
                1 for i in gids if (not bool(greedy.get(i))) and any(bool(m.get(i)) for m in maps.values())
            ),
            "teacher_only_correct_counts": {
                n: sum(1 for i in gids if (not bool(greedy.get(i))) and bool(maps[n].get(i))) for n in names
            },
        }
    return {
        "schema_version": "2.3.1",
        "answerable_case_count": len(ids),
        "teacher_correct_counts": per,
        "teacher_accuracies": {n: (c / len(ids) if ids else None) for n, c in per.items()},
        "oracle_union_correct": union_n,
        "oracle_union_accuracy": union_n / len(ids) if ids else None,
        "unique_wins": unique,
        "pairwise": pairwise,
        "three_way_intersections": three,
        "vs_greedy": recoverable,
    }


def markdown(out: dict[str, Any]) -> str:
    lines = [
        "# v2.3.1 Teacher Overlap Analysis",
        "",
        f"- Answerable cases: **{out['answerable_case_count']}**",
        f"- Teacher oracle union: **{out['oracle_union_accuracy']}**",
        "",
        "## Per-teacher answerable accuracy",
        "",
    ]
    for n, x in (out.get("teacher_accuracies") or {}).items():
        lines.append(f"- {n}: `{x}` (correct `{out['teacher_correct_counts'][n]}`, unique wins `{out['unique_wins'][n]}`)")
    lines += ["", "## Pairwise overlap", ""]
    for k, v in (out.get("pairwise") or {}).items():
        lines.append(
            f"- {k}: intersection=`{v['intersection']}`, jaccard=`{v['jaccard']}`, "
            f"a_only=`{v['a_only']}`, b_only=`{v['b_only']}`"
        )
    lines += ["", f"Three-way intersections: `{out.get('three_way_intersections')}`", ""]
    if out.get("vs_greedy"):
        vg = out["vs_greedy"]
        lines += [
            "## Vs greedy failures",
            "",
            f"- Greedy wrong (answerable): **{vg['greedy_wrong_answerable']}**",
            f"- Recoverable by any teacher: **{vg['recoverable_by_any_teacher']}**",
            f"- Per-teacher recoverable counts: `{vg['teacher_only_correct_counts']}`",
            "",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", action="append", default=[], help="name=path to direct results.jsonl")
    ap.add_argument("--greedy", type=Path, help="optional greedy direct results.jsonl")
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    a = ap.parse_args()
    maps = {}
    for spec in a.teacher:
        name, path = spec.split("=", 1)
        maps[name] = answerable_correct(Path(path))
    greedy = answerable_correct(a.greedy) if a.greedy else None
    out = analyze(maps, greedy)
    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    a.output_md.write_text(markdown(out), encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Read completed cells, verify their artifacts, and report development tradeoffs.

No best-arm selection, pooled p-values, equivalence claims, or release permissions.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from .integrity import file_hash, read_json, write_new


def summarize(directory: Path) -> dict:
    result = read_json(directory / "result.json")
    if result.get("status") != "ENGINEERING_COMPLETE_QUALITY_UNDECIDED":
        raise ValueError("incomplete/failed attempts cannot enter a completed-cell table")
    for filename, field in (("baseline.json", "baseline_sha256"), ("final_dev.json", "final_dev_sha256"),
                            ("steps.jsonl", "steps_sha256")):
        if file_hash(directory / filename) != result[field]:
            raise ValueError("completed-cell artifact hash mismatch")
    base, final = read_json(directory / "baseline.json"), read_json(directory / "final_dev.json")
    if ([r["id"] for r in base["records"]] != [r["id"] for r in final["records"]]
            or base["response_targets"] != final["response_targets"]):
        raise ValueError("evaluation population changed")
    totals = {"scored": 0, "correct_before": 0, "correct_after": 0,
              "gained": 0, "lost": 0, "false_abstentions_after": 0, "limit_hits_after": 0}
    for a, b in zip(base["records"], final["records"], strict=True):
        totals["limit_hits_after"] += b["limit_hit"]
        sa, sb = a["scores"], b["scores"]
        if (sa["scorer"], sa.get("answerable")) != (sb["scorer"], sb.get("answerable")):
            raise ValueError("scoring contract changed")
        if sa["correct"] is None:
            continue
        totals["scored"] += 1
        totals["correct_before"] += bool(sa["correct"])
        totals["correct_after"] += bool(sb["correct"])
        totals["gained"] += not sa["correct"] and sb["correct"]
        totals["lost"] += sa["correct"] and not sb["correct"]
        totals["false_abstentions_after"] += bool(sb.get("false_abstention", False))
    return {"arm": result["arm"], "binding": result["binding"], "synthetic": result["synthetic"],
            "progress": result["progress"], "baseline_response_ce": base["response_ce"],
            "final_response_ce": final["response_ce"], "metrics": totals,
            "quality_improvement_established": False, "release_permission": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    rows = [summarize(p) for p in args.directories]
    if len({r["arm"] for r in rows}) != len(rows):
        raise ValueError("one attempt per arm; do not select or duplicate retries silently")
    common = [{k: v for k, v in r["binding"].items() if k != "arm"} for r in rows]
    if any(c != common[0] for c in common):
        raise ValueError("different plans, sources, environments or code cannot form a matched screen")
    write_new(args.out, {"schema": 1, "experiment": "MQ-001", "scope": "DEVELOPMENT_ONLY",
                         "rows": rows, "all_six_present": len(rows) == 6,
                         "statistical_confirmation": "NOT_PERFORMED", "release_permission": False})


if __name__ == "__main__":
    main()

"""Is the beta=1.0 collapse a capability loss or a format break?

Exact-match scoring conflates two very different failures:
  (a) the model still emits a well-formed state object, with wrong values  -> capability loss
  (b) the model stops emitting the output contract at all                  -> format break

These have opposite implications. (a) says composition destroyed the skill. (b) says composition
destroyed INSTRUCTION FOLLOWING, and the skill underneath may be intact and recoverable by
prompting or by a lighter coefficient — a much cheaper problem.

Classifies every generation of every arm into a small set of shapes, so the collapse can be
attributed rather than merely observed.
"""
from __future__ import annotations

import argparse, json, re
from collections import Counter
from pathlib import Path


def shape_of(gen: str, row: dict) -> str:
    g = gen.strip()
    if not g:
        return "empty"
    if g == row["reference"]:
        return "correct"
    if g == row["copy_baseline"]:
        return "copied_input_state"
    # well-formed JSON of the right TYPE and KEYS but wrong values?
    try:
        obj = json.loads(g)
    except json.JSONDecodeError:
        # partial/invalid JSON that still looks like it is trying
        if g.startswith("{") or g.lstrip("-").isdigit():
            return "malformed_json_attempt"
        return "prose_or_other"
    try:
        gold = json.loads(row["reference"])
    except json.JSONDecodeError:
        return "unclassified"
    if isinstance(gold, int):
        return "wellformed_wrong_value" if isinstance(obj, int) else "wrong_type"
    if isinstance(obj, dict):
        if set(obj) == set(gold):
            return "wellformed_wrong_value"
        if set(gold) <= set(obj):
            return "extra_keys"            # e.g. echoing before/event alongside after
        return "wrong_keys"
    return "wrong_type"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--examples", type=int, default=2)
    a = ap.parse_args()

    bank = {r["id"]: r for r in json.loads(a.bank.read_text(encoding="utf-8"))["rows"]}
    out = {"bank_n": len(bank), "arms": {}}
    for p in a.runs:
        d = json.loads(p.read_text(encoding="utf-8"))
        for arm in d["arms"]:
            shapes = Counter()
            by_family = {}
            examples = {}
            for pr in arm["predictions"]:
                row = bank[pr["id"]]
                s = shape_of(pr["generation"], row)
                shapes[s] += 1
                by_family.setdefault(row["family"], Counter())[s] += 1
                if s != "correct" and len(examples.setdefault(s, [])) < a.examples:
                    examples[s].append({"family": row["family"],
                                        "gold": row["reference"],
                                        "generation": pr["generation"][:160]})
            out["arms"][arm["arm"]] = {
                "correct": arm["correct"],
                "shapes": dict(shapes.most_common()),
                "by_family": {k: dict(v.most_common()) for k, v in sorted(by_family.items())},
                "examples": examples,
            }
    a.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    order = ["correct", "wellformed_wrong_value", "extra_keys", "copied_input_state",
             "malformed_json_attempt", "wrong_keys", "wrong_type", "prose_or_other", "empty"]
    hdr = ["arm"] + [o[:14] for o in order]
    print("  " + "".join(f"{h:<16}" for h in hdr))
    for name, v in out["arms"].items():
        row = [name] + [str(v["shapes"].get(o, 0)) for o in order]
        print("  " + "".join(f"{c:<16}" for c in row))
    print()
    print("  KEY: 'wellformed_wrong_value' = still obeys the output contract, wrong answer")
    print("       -> capability loss. High 'prose_or_other'/'extra_keys' -> format break.")


if __name__ == "__main__":
    main()

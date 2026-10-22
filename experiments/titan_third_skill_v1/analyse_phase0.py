"""TRISKILL-001 Phase 0 analysis: the registered gates of PHASE0_PREREG.md, nothing else.

Reads PHASE0_SCORES.json and writes PHASE0_RESULT.json. Refuses to overwrite.
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLOORS = {"A": 23, "C": 18, "P": 22, "R": 20, "S": 17}      # registered, from the keys
HEADROOM_MAX = 48                                          # BASE strict <= 48 of 96
ORDER = ["C", "P"]
FLAG_ROWS = 20


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def json_rows(arm_rec, bank):
    return {r["id"]: r for r in arm_rec["banks"][bank]["rows"] if r["variant"] != "alternate_format"}


def main() -> int:
    src = json.loads((HERE / "PHASE0_SCORES.json").read_text(encoding="utf-8"))
    out_path = HERE / "PHASE0_RESULT.json"
    if out_path.exists():
        raise SystemExit("refusing to overwrite PHASE0_RESULT.json")
    arms = src["arms"]
    table, contrasts = {}, {}
    for bank in FLOORS:
        base = json_rows(arms["BASE"], bank)
        if len(base) != 96:
            raise SystemExit(f"{bank}: expected 96 JSON rows")
        table[bank] = {}
        for arm in arms:
            rows = json_rows(arms[arm], bank)
            alt = [r for r in arms[arm]["banks"][bank]["rows"] if r["variant"] == "alternate_format"]
            table[bank][arm] = {
                "strict": sum(r["strict_correct"] for r in rows.values()),
                "semantic": sum(r["semantic_correct"] for r in rows.values()),
                "format_valid": sum(r["format_valid"] for r in rows.values()),
                "by_variant": {v: sum(r["strict_correct"] for r in rows.values() if r["variant"] == v)
                               for v in ("full", "counterfactual", "distractor")},
                "alternate_format_strict_of_32": sum(r["strict_correct"] for r in alt),
            }
        for arm in ("SPARSE", "TT"):
            rows = json_rows(arms[arm], bank)
            gained = sum(rows[i]["strict_correct"] and not base[i]["strict_correct"] for i in base)
            lost = sum(base[i]["strict_correct"] and not rows[i]["strict_correct"] for i in base)
            contrasts[f"{arm}-BASE/{bank}"] = {"net": gained - lost, "gained": gained, "lost": lost,
                                              "p_exact_mcnemar": mcnemar_exact(gained, lost)}
    headroom = {b: {"base_strict": table[b]["BASE"]["strict"], "floor": FLOORS[b],
                    "pass": table[b]["BASE"]["strict"] <= HEADROOM_MAX} for b in ORDER}
    selected = next((b for b in ORDER if headroom[b]["pass"]), None)
    flag = None
    if selected:
        c = contrasts[f"TT-BASE/{selected}"]
        flag = bool(c["net"] >= FLAG_ROWS and c["p_exact_mcnemar"] < 0.05)
    # Amendment 1 determinism canary: every unit finished in attempt 1 must be byte-identical now.
    determinism = {}
    a1 = HERE / "attempt1" / "phase0_predictions"
    if a1.exists():
        for p in sorted(a1.glob("*.jsonl")):
            now = HERE / "phase0_predictions" / p.name
            determinism[p.stem] = now.exists() and now.read_bytes() == p.read_bytes()
    result = {
        "schema": 1, "experiment": "TRISKILL-001", "phase": 0,
        "determinism_vs_attempt1": {"units": determinism, "all_identical": bool(determinism) and all(determinism.values())},
        "registered_in": "PHASE0_PREREG.md", "scores_file": "PHASE0_SCORES.json",
        "population": "96 JSON-contract rows per bank (full, counterfactual, distractor); strict = SIF scorer",
        "table": table, "contrasts": contrasts, "G0_HEADROOM": headroom,
        "selected_candidate": selected,
        "stop": selected is None,
        "partially_present_before_donation_flag": flag,
        "quality_claim": False, "release_permitted": False,
    }
    out_path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"table_strict": {b: {a: table[b][a]["strict"] for a in table[b]} for b in table},
                      "headroom": headroom, "selected": selected, "flag": flag}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

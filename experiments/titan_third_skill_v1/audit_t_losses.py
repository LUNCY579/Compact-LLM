"""Post-STOP descriptive audit: T losses as value errors versus format errors.

This never changes the registered T exact-match endpoint or Stage A verdict.
It checks whether a lost prediction still parses to the same JSON value as gold.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from analyse_stage_a import V3X, t_rows

HERE = Path(__file__).resolve().parent
OUT = HERE / "T_LOSS_AUDIT.json"


def category(generation: str, reference: str) -> str:
    text = generation.strip()
    if text.endswith("."):
        text = text[:-1]
    try:
        parsed = json.loads(text)
    except (ValueError, json.JSONDecodeError):
        return "unparseable"
    return "same_json_value" if parsed == json.loads(reference) else "parseable_wrong_value"


def main() -> int:
    if OUT.exists():
        raise SystemExit("T_LOSS_AUDIT.json exists (create-only)")
    bank = json.loads(V3X.read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in bank["rows"]}
    tt = t_rows("TT")
    result = {"experiment": "TRISKILL-001", "kind": "post-STOP descriptive T loss audit",
              "registered_T_endpoint_unchanged": True, "arms": {}}
    for name in ("TTC25", "TTC50", "TTC100"):
        arm = t_rows(name)
        lost = [i for i in tt if tt[i]["primary"] and not arm[i]["primary"]]
        gained = [i for i in tt if not tt[i]["primary"] and arm[i]["primary"]]
        categories = Counter(category(arm[i]["generation"], rows[i]["reference"]) for i in lost)
        examples = defaultdict(list)
        for i in lost:
            family = rows[i]["family"]
            if len(examples[family]) < 5:
                examples[family].append({"id": i, "gold": rows[i]["reference"],
                                         "TT": tt[i]["generation"], "composite": arm[i]["generation"],
                                         "category": category(arm[i]["generation"], rows[i]["reference"])})
        result["arms"][name] = {"lost_primary": len(lost), "gained_primary": len(gained),
                                 "net_T_loss": len(lost) - len(gained),
                                 "loss_categories": dict(categories), "examples_by_family": dict(examples)}
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({name: {k: v for k, v in rec.items() if k != "examples_by_family"}
                      for name, rec in result["arms"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

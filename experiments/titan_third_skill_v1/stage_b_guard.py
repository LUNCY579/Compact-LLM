"""Gate every Stage B entry point on the frozen Stage A decision."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def selected() -> tuple[str, str, float]:
    path = HERE / "STAGE_A_RESULT.json"
    if not path.exists():
        raise SystemExit("Stage B refused: Stage A result is missing")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("verdict") != "PROCEED_TO_STAGE_B" or not result.get("canaries", {}).get("pass") or not result.get("A1", {}).get("pass"):
        raise SystemExit("Stage B refused: registered Stage A gates did not pass")
    choice = result.get("A3") or {}
    beta = choice.get("beta_star")
    composite = choice.get("arm")
    names = {0.25: ("TTC25", "C25"), 0.50: ("TTC50", "C50"), 1.00: ("TTC100", "C100")}
    if beta not in names or composite != names[beta][0] or not result.get("A2", {}).get(composite, {}).get("eligible"):
        raise SystemExit("Stage B refused: Stage A coefficient or eligibility is inconsistent")
    return composite, names[beta][1], beta

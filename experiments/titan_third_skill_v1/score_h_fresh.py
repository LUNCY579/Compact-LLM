"""Run CONFIRM-002's H scorer on the registered TRISKILL H_FRESH bank.

Its rendering, decoding and boundary_rule are unchanged. The only added allowed
bank is the committed H_FRESH sha256. The sealed HSKILL-CONFIRM-002 stays barred.
"""
from __future__ import annotations

import sys
from pathlib import Path

from stage_b_guard import selected

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "composition_confirmation_v2"))
import score_boundary  # noqa: E402

H_SHA = "471c7a48a3821038e9e401f710526d01f0cef4b3d70b9c8625cff15ad8ed9cc4"


def main() -> int:
    selected()
    score_boundary.ALLOWED = {H_SHA: "TRISKILL-001 H_FRESH (registered held-out bank)"}
    return score_boundary.main()


if __name__ == "__main__":
    raise SystemExit(main())

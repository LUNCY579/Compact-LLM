"""CLOCKFIX-001: the clock-repair lane, registered separately from the transition lanes.

Trains one repair delta, delta_R, on the time_duration class with every retention-probe item
removed by CONTENT (not merely by id) -- CLOCKFIX001_AMENDMENT_4_CONTENT_EXPOSURE.md -- and
restricted to the single-turn rows `serialization.compile_row` accepts. delta_R is trained from
Supa like every other delta in this programme and composed onto doseparam001 rung4 at a fixed,
pre-registered coefficient of 1.0 (CLOCKFIX001_AMENDMENT_5_LAUNCH_PATH.md section 5).

Registration terms are the transition lanes' terms unchanged, exactly as GEN2-001 registered: the
ceiling is `preflight.LIMITS` and the phases are ("train", "resume") -- no "qualify", because this
lane inherits the already-qualified 2048 window rather than re-qualifying it.

Registered limitation, with its direction, in Amendment 5 section 4: two of rung4's three lost
time_duration references are multi-turn while this corpus is single-turn only. It biases toward
the primary NOT firing, so a PASS is conservative and a FAIL is confounded with the mismatch.
"""
from __future__ import annotations

from . import budget
from .preflight import LIMITS

EXPERIMENT = "CLOCKFIX-001-CLOCK-REPAIR"
EXCLUDES = ("RL", "larger_teacher", "cloud", "release", "architectural_change")

budget.register(EXPERIMENT, ceiling=LIMITS, phases=("train", "resume"), excludes=EXCLUDES)

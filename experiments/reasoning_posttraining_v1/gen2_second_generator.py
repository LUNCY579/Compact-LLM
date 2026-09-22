"""GEN2-001: the second GENERATOR lane, registered separately from the transition lanes.

COMPOSITION_IS_A_REPAIR P1 -- "a second exact-match GENERATIVE delta composed with delta_T at
matched norm loses MORE than 16 acquisition rows" -- has been untestable because no second
generator delta exists. Every trained checkpoint on disk is delta_T (three seed draws of one
skill), delta_H (a GATE: wide-margin abstention), or a mixture of those two corpora.

This lane trains one delta on a PURE `role_person_binding` corpus: exact-match single-name
answers, a narrow-margin generative contract, categorically distinct from delta_H.

Registration terms are the transition lanes' terms unchanged. The ceiling is `preflight.LIMITS`
and the phases are ("train", "resume") -- no "qualify", because this lane inherits the already
qualified 2048 window rather than re-qualifying it.

Supervision shortfall and its DIRECTION are registered in GEN2001_PREREG.md: 2,384 supervised
tokens over 768 rows (3.1/row) against delta_T's 13.7 and delta_H's 9.7. It biases toward a
SMALLER measured cost, so P1's falsifier branch is DISABLED and a cost of <=16 is reported
UNINTERPRETABLE rather than as a falsification.
"""
from __future__ import annotations

from . import budget
from .preflight import LIMITS

EXPERIMENT = "GEN2-001-SECOND-GENERATOR"
EXCLUDES = ("RL", "larger_teacher", "cloud", "release", "architectural_change")

budget.register(EXPERIMENT, ceiling=LIMITS, phases=("train", "resume"), excludes=EXCLUDES)

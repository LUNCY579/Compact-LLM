"""TRISKILL-001: the rules-with-exceptions skill lane, registered separately from the transition lanes.

Trains one generator-skill delta, delta_C, from exact Supa on SIF-001 TRAIN atomic_C (JSON-contract
variants only), at delta_T's processed-token dose, for composition onto Titan-Transition at its own
coefficient. Registered in ChatBubble-titan-model-soup-mini-v1
experiments/titan_third_skill_v1/PREREG.md (research/titan-model-soup-mini-v1 0d2b64b4).

Registration terms are the transition lanes' terms unchanged, exactly as GEN2-001 and CLOCKFIX-001
registered: the ceiling is `preflight.LIMITS` and the phases are ("train", "resume").
"""
from __future__ import annotations

from . import budget
from .preflight import LIMITS

EXPERIMENT = "TRISKILL-001-RULES-SKILL"
EXCLUDES = ("RL", "larger_teacher", "cloud", "release", "architectural_change")

budget.register(EXPERIMENT, ceiling=LIMITS, phases=("train", "resume"), excludes=EXCLUDES)

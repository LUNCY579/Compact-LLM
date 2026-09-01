"""Rineya significance modulation. Not part of FALCON.

Kept as a named module so Arm D's extra parameters are obvious in audits.
The actual math lives in ``FastWeightBlock.rates`` when ``significance_gate=True``.
"""

from __future__ import annotations

# Documented form (PREREGISTRATION / PAPER_NOTES part C):
#   s_t in (0, 1)
#   β_t = β_min + s_t (β_max − β_min)
#   λ_t = λ_max − s_t (λ_max − λ_min)

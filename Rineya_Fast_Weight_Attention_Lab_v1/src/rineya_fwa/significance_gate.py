"""Rineya significance modulation. Not part of FALCON.

Kept as a named module so Arm D's extra parameters are obvious in audits.
The actual math lives in ``FastWeightBlock.rates`` when ``significance_gate=True``.

P1/P2 form (absolute ranges)::

    s_t in (0, 1)
    beta_t   = beta_min + s_t (beta_max - beta_min)      # 0.05 .. 1.9
    lambda_t = lambda_max - s_t (lambda_max - lambda_min)  # 0.5 .. 0.01

That form has two defects, both found in the P2 post-mortem:

1. it *replaces* beta/lambda rather than modulating them, so ``beta_raw`` and
   ``lambda_raw`` receive no gradient in arm D -- in the P2 adapters they are
   bit-identical to their initialisation;
2. its ranges are absolute, so arm D and arm C do not share a retention horizon.
   Arm D would then differ from arm C in the horizon *and* the gate, and the
   preregistered D-C contrast could not attribute a difference to the gate.

P2b form (symmetric multiplicative deviation about arm C's own rates)::

    u_t      = 2 s_t - 1                       in (-1, 1)
    beta_t   = beta0   * (1 + span * u_t)
    lambda_t = lambda0 * (1 - span * u_t)

with ``span = FWAConfig.significance_span`` (default 0.9) and ``sig_gate``
zero-initialised, so ``s_t = 0.5`` and arm D is *exactly* arm C at step 0. The
semantics of the brief are preserved -- high significance means a stronger write
and slower forgetting -- while the arms differ by the minimum mechanism.
"""

from __future__ import annotations

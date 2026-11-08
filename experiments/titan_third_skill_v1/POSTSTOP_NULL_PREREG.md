# TRISKILL-POSTSTOP-001 — one norm-matched random control after the Stage A stop

Registered 2026-09-26 **after** TRISKILL-001's `STOP_NO_ELIGIBLE_BETA`, and
before this control arm is built or scored. This is a separate exploratory
diagnostic. It cannot revise A2/A3, select a β, or read any Stage B bank.

## Question

TTC25 gained 47 C semantic rows but lost 16 T primary rows against TT, exceeding
the registered eight-row development limit. Is that T loss a generic consequence
of adding an update with norm `||0.25*δ_C|| = 0.2143793`, or does δ_C's direction
matter? The observed 37× same-block write-out subspace collision motivates
this control, but does not answer the functional question.

## Frozen control and instruments

- **Arm:** `TTR25 = TT + R` via the unchanged `build_arm.py` TTR construction:
  independent Gaussian seed 20260926, each trained tensor rescaled to the norm
  of that tensor in `0.25*δ_C`; FP32(float64 sum). One draw, no seed search.
- **Banks:** only the already exposed Stage A C DEV+CAL, T_BANK_V3X and
  HSKILL_VIEWS_V2. The identical scorers and decoding settings used for
  TRISKILL-001 Stage A are reused. The 192 JSON C rows are the C endpoint;
  T primary and H boundary use the registered Stage A definitions.
- **Controls:** TT and TTC25's saved Stage A predictions. Do not rescore them.
- **Resource floor:** at least 5.0 GiB free host RAM and 6.0 GiB free disk at
  launch; one exclusive GPU lease; arm weights deleted after all score files
  and the build hash have been checked.

## Readout and limits

Report the net T loss, H loss and C semantic gain of TTR25 versus TT, alongside
TTC25's same three contrasts. Report paired gains and losses on T. A TTR25 T
loss much smaller than 16 would support a directional contribution beyond
per-tensor norm; a similar loss would leave generic magnitude plausible.
These are descriptions, **not** a new pass/fail threshold or causal proof:
there is one random draw, the coefficient was chosen after viewing Stage A,
and all banks are development material. No Stage B, broad retention, release,
or promotion result follows from this control.

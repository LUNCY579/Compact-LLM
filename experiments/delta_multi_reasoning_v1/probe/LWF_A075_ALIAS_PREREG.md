# LWF α=0.75 alias — a point prediction from the already-measured collinear ray

**Written and committed BEFORE the arm is scored.** The generation is queued behind the peer
session's lease and has not run. MECHANISM SCOUT.

The arm is the peer's `lwf001-20260918/ctrl-own-alpha0.75` — **their build, their bytes, not
rebuilt here.** Scoring the same bytes matters more than owning the build: a second build risks
a different arm answering the same question.

## Why a point prediction is available where a band would normally be

`ctrl-own-alpha0.75` lies on the **same collinear ray** this session has already measured at four
norms — `theta_supa + a*(theta_lambda0 - theta_supa)`, cosine 1.000000 by construction:

| ‖δ‖ | alias /360 | arm |
|---:|---:|---|
| 0.0000 | 270 | Supa |
| 0.4161 | 237 | `ownA05`, a = 0.50 |
| 0.6800 | 211 | `sc_m05` |
| 0.7427 | 201 | `sc_m025` |
| 0.8323 | 199 | λ = 0 |

The new arm's norm is **0.624217** (= 0.75 × 0.832289, confirmed by the peer's build receipt),
bracketed by `ownA05` and `sc_m05` at fraction 0.7886.

## Registered

> **POINT PREDICTION: 216.5 alias rows of 360 (0.6014). Band ±6 rows: 210–222.**

The peer's registered bands are **≥210 transfers / 203–209 partial / <203 native-specific**. My
prediction sits inside their "transfers" band and is roughly **twelve times narrower**, so it
can fail while theirs passes — which is the point of stating it.

## What each outcome means

| outcome | reading |
|---|---|
| **210–222** | the collinear ladder is **predictive**, and interpolating a measured ray substitutes for building an arm |
| **outside ±6, still monotone** | the ray is measured but not linear enough to interpolate; every residual computed against it inherits that error, including DIRCTL-001's **+7.2 / +9.4 / +4.4** |
| **non-monotone** (above 237 or below 199) | the LWF ray is not monotone in ‖δ‖, which contradicts DIRCTL-001's ρ = −1.000 and would be the most serious outcome |

**The second row is the one that costs me something**, and it is why this is worth registering:
DIRCTL's residuals are differences against *measured* twins, not interpolated ones, so they
survive — but `analysis/coupling_ladder.py` and the conservation arithmetic both lean on ray
interpolation, and a miss would put a tolerance on them.

## The arm-versus-rule point, in concrete form

**`ownA05` at a = 0.50 already scores 237.** So if α=0.75 lands **below** 237 — which the
prediction says it will, by about 20 rows — then **on the LWF lane the knee is at or below 0.50**
and the native lane's α = 0.75 does **not** transfer as a rule.

That is the null that **confirms** the arm-versus-rule distinction rather than retracting the
native result: α = 0.75 would remain the best measured operating point among the native arms,
and "keep three quarters" would remain unsupported as a general recommendation.

Acquisition and long context on this arm are the peer session's instruments and are not scored
here.

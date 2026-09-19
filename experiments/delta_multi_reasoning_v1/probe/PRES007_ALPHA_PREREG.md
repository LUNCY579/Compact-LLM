# PRES-007 × α — does interpolation repair the BINDING deficit, and does it repair all types alike?

**Registered before any arm is generated on this bank.** MECHANISM SCOUT — no seal, no read
budget. Openly spendable; nothing here promotes anything.

## Why this and not another α

The peer session's closing assessment, which I agree with: **the most valuable unmeasured thing
is no longer an α.** Locating the optimum (α = 0.625, 0.875) is refinement nobody has asked for.
**A general role→filler binding deficit that interpolation partially repairs is a mechanism
claim**, and α is only the knob that exposes it.

Established so far: the delta damages person-, object- and place-valued binding **purely
subtractively** (0 repairs, 58 damages, 3/3 strata, PRES-005), and on the alias bank
interpolation recovers damage **purely additively** (α=0.75 vs λ=0: 21 gains, **0 losses**).
**Nobody has asked whether the repair is uniform across answer types.**

## Design

PRES-007 — 360 rows, three strata differing only in the answer's **type**, surface-matched after
the capitalisation cue was found and fixed (person-vs-object: lowercase starts 0.0000,
capitalised-token share 0.0261). Mechanically clean, stratum-outlier improved, and it passes the
answer-sentence locator **cold** — that check was written after the bank was built and never used
to tune it.

Four arms, all on one collinear ray, all already on disk: **Supa** (0.000), **`ownA05`** (0.416),
**`ctrl-own-alpha0.75`** (0.624), **λ=0** (0.832).

`place_binding` is **dropped in advance** — Supa sits at 0.925 there, above ceiling, and a
stratum with no headroom cannot show repair either.

## Registered predictions

| account | pattern across strata |
|---|---|
| **UNIFORM REPAIR** | gains vs λ=0 roughly proportional to each stratum's damage; repair fraction similar for person and object |
| **TYPE-SELECTIVE REPAIR** | repair fraction differs materially between person and object — interpolation restores one type better |
| **NO REPAIR** | gains ≈ 0 on this bank, which would contradict the alias result and be the most interesting outcome |

**Primary: repair fraction per stratum** = (arm − λ=0) / (Supa − λ=0), the share of the delta's
damage recovered. **Reported with gains and losses split and never netted** — the peer's 13/13
hid a +8/−5, and mine hid nothing; only decomposition tells you which kind you have.

**Secondary, free:** whether `ownA05` (α=0.50) beats α=0.75 here as it does on alias. The LWF
lane's best α is 0.50 on alias by 17 rows; if the binding bank agrees, that is a second endpoint
locating the same lane optimum.

## What would disconfirm

- **Any stratum showing repairs AND losses vs λ=0** — alias repair was purely additive; a
  two-sided pattern here means the additivity is task-specific.
- **α=0.75 beating α=0.50** — would contradict the alias result on the same lane and same ray.
- **Repair fraction > 1** — an arm recovering more than the delta destroyed, which no account
  predicts and which would be reported as unexplained.

## Caveats

One draw per arm, one lane, one bank. PRES-007 is **20 points easier than ALIAS-001** for
reasons three tested hypotheses failed to explain, so **every number here is within-bank** and
no rate is comparable to the alias bank's. `high`-style readability caveats do not apply — this
bank has no salience conditions — but `place_binding`'s ceiling does, which is why it is dropped
rather than reported.

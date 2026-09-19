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

---

# AMENDMENT 1 — three pre-data corrections from the peer session, all verified here

**Written while the first arm was still generating.** All three are corrections to *my* prereg,
and all three are right.

## 1. The uniformity test is UNDERPOWERED, and a null must report as such

Repair fraction is `repaired / damaged`, so the denominators are PRES-005's damage counts:
**person 31, object 17.** Computed here:

| person repair | object repair | gap | Fisher p | detectable? |
|---|---|---:|---:|---|
| 19/31 = 0.60 | 5/17 = 0.30 | **30 pp** | **0.0687** | **no** |
| 16/31 = 0.50 | 5/17 = 0.30 | 20 pp | 0.2238 | no |
| 22/31 = 0.70 | 5/17 = 0.30 | 40 pp | 0.0074 | yes |
| 25/31 = 0.80 | 5/17 = 0.30 | 50 pp | 0.0013 | yes |

> **Only a gap of ~40 pp or more is detectable. A 30-point difference — a large effect — lands
> at p ≈ 0.069 and would be invisible.**

**So a null on uniformity is registered NOW as "not powered to distinguish", never as "repair is
uniform."** This is the `references_retained` trap in a new coordinate: that endpoint needed ≥6
discordant pairs and could not have reached significance even if the effect were entirely real,
and I corrected the peer on it this morning before walking into the same shape myself.

**The structural part, which more α points cannot fix:** person has the larger repair
denominator *precisely because it is damaged more*, so the well-estimated fraction and the
poorly-estimated one are systematically the wrong way round. **Only more `object_binding` rows
fix it**, not more arms.

## 2. "Repair fraction" is ambiguous, and my primary is a CHOICE

Two readings that diverge:

| reading | definition | what it asks |
|---|---|---|
| **(a)** — my primary | `repaired / damaged` | does the repair **mechanism** behave the same way |
| **(b)** | `repaired / baseline-correct` | is repair a fixed **per-row** probability |

Person is damaged 0.304 and object 0.168. **If repair is proportional to damage, (a) comes out
equal while (b) differs by 1.81×. If repair is a fixed per-row probability, the reverse.** They
are not two estimates of one quantity.

**(a) stays the primary** because the question is about the mechanism — but **(b) is reported
alongside**, and choosing (a) is recorded as a choice rather than presented as the only reading.

## 3. My reason for dropping `place_binding` was wrong; the decision stands on a different one

I wrote: *"Supa at 0.925 is above ceiling, and a stratum with no headroom cannot show repair any
more than damage."* **That is incorrect.** Repair is measured against **λ=0's damage**, not
against Supa's ceiling. PRES-005 has `place_binding` at Supa 0.925 → λ=0 0.842 — **10 damaged
rows of 111**, and every one of those 10 has full room to be repaired.

> **The actual disqualifier is that a denominator of 10 is too thin for a repair fraction.**
> Ceiling blocks **damage** detection; a thin damage count blocks **repair** detection. They are
> different constraints and `place_binding` has only the second.

The distinction matters because the ceiling framing would also argue for dropping it from the
*damage* analysis, where it produced a real result — **0 repairs / 10 damages, p = 1.9e-03**.
Recorded as thinness so the band rule is not stretched to cover a case it does not govern.

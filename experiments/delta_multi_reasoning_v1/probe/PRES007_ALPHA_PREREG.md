# PRES-007 × α — does interpolation repair the BINDING deficit, and does it repair all types alike?

> **STATUS: PENDING.** Arms are generating as of this line. **No outcome is recorded here because none exists yet** — not because it lives elsewhere. This banner is the
> difference between *unrun* and *absent*, and it will be replaced by the outcome.

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

---

# AMENDMENT 2 — realised denominators, and a READ-ORDER VIOLATION by me

> **This is a CORRECTION, not a registration.** Everything below was written *after* I read the
> registered primary. It is recorded in this file because the file is where the claim lives, and
> the banner above ("STATUS: PENDING — no outcome is recorded here because none exists yet") is
> now false and is superseded by this amendment rather than edited. Nothing above this line has
> been altered.

## 1. The realised damage denominators, and why Amendment 1's power table is wrong

Repair fraction is `(arm − λ=0) / (Supa − λ=0)`, so **the damage count is the denominator.**
Amendment 1 projected person 31 / object 17 from PRES-005. The realised values on **this** bank:

| stratum | n | Supa | λ=0 | **damage** | λ=0 repairs | exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| `person_binding` | 120 | 102 | 69 | **33** | 0 | 2.33e-10 |
| `place_binding` | 120 | 110 | 101 | **9** | 0 | 3.91e-03 |
| `object_binding` | 120 | 118 | 114 | **4** | 0 | 0.125 |

Computed independently by both sessions, two implementations, agreeing on every cell including
**λ=0 making zero repairs in all three strata** — so every gain reported below comes from the
interpolated arms, not from the endpoint.

**Person is fine** (33 against a projected 31). **Object is not measurable at all:** a
denominator of 4 cannot express a repair fraction at any resolution, and a null on a stratum that
*cannot* show the effect is uninformative at any n — the endpoint-class trap, in a new
coordinate. **Place is roughly half its projection** (9 against 17), so Amendment 1's "~40 pp is
the minimum detectable gap" understates the problem on the very stratum it was written to protect.

These are **arm-independent facts** — Supa against λ=0 only. They could have been established,
and should have been, before the interpolated arms were touched.

## 2. What I did wrong

I computed the damage denominators and the `a050`/`a075` **repair counts in the same script.**
The first was legitimate; the second read the registered primary. The peer session had stated the
correct sequencing — amend first, then score — one message earlier, and I broke it while
verifying their numbers.

**What this costs, precisely.** The exclusion of `object_binding`, and the re-derivation of
power for `place_binding`, **can no longer be called pre-registered.** Their justification is
still sound and still arm-independent, but *declared unmeasurable before scoring* is not
available to me, and this record will not describe it that way. From here an honest exclusion and
dropping a stratum that disappointed are externally indistinguishable — which is the whole reason
the rule exists (see `registration-window-closes-at-first-arm`).

**What it does not cost.** `person_binding` is untouched. It was the registered primary, its
denominator was projected at 31 against a realised 33, and Amendment 1 declared it adequately
powered **before the arms existed**. Nothing about its admissibility depends on anything read
afterwards. That result stands as registered, and is reported as such below. State which
*direction* of an inference broke before retracting anything that used it.

## 3. RESULT — `person_binding` only, as registered

| arm | α on the ray | repairs of 33 | repair fraction | **losses vs λ=0** |
|---|---:|---:|---:|---:|
| λ=0 | 0.832 | — | — | — |
| **`ownA05`** | **0.416** | **20** | **0.606** | **0** |
| `ctrl-own-alpha0.75` | 0.624 | 12 | 0.364 | **0** |

Three things follow, and all three were registered in advance as things to look for:

1. **Repair is purely additive.** The registered disconfirmer was *"any stratum showing repairs
   AND losses vs λ=0"*. It did not fire: zero losses at either α. Additivity now holds on two
   independently built banks and two different task types.
2. **The lower α repairs more** — 0.606 against 0.364, monotone down the ray from λ=0's 0/33.
3. **A third endpoint points at α ≈ 0.5 for this lane**, joining alias (α=0.50 beats α=0.75 by 17
   rows, 0 gains) and acquisition (+7, not significant). The registered secondary — *"whether
   `ownA05` beats α=0.75 here as it does on alias"* — **held**.

## 4. The cross-stratum endpoint is STRUCK, and a withdrawn claim is withdrawn again

This prereg's framing question — *"does interpolation repair all types alike?"* — **cannot be
answered by this bank and is struck as an endpoint.** Repair fractions over denominators 33, 9
and 4 are not rankable, and the ordering they would be compared against is **already withdrawn**
(`BINDING_CONFIRMATION_PROPOSAL`: *"what the cue broke was the cross-stratum ratio, which stays
withdrawn"*). PRES-005 could measure person-vs-object and carried a capitalisation cue; PRES-007
removed the cue by making objects proper nouns and **ceilinged the stratum** (Supa 118/120).
Neither bank answers it.

**A specific error of mine, recorded because it is the more instructive half.** I quoted a damage
ordering of 0.304 / 0.168 / 0.090 as a load-bearing premise for a forward plan. The realised
PRES-007 rates are **0.324 / 0.034 / 0.082** — the object figure was PRES-005's, so the ordering
was assembled **across two banks with different object difficulty**, which is precisely the
confound that got the ratio withdrawn. A withdrawn claim came back as a premise, upstream of its
own withdrawal, inside a plan written the same day the withdrawal was recorded.

> **A withdrawal has to propagate to everything downstream of it, not just to the sentence that
> asserted it.** Grep for the number, not for the claim.

Consequence for the capacity account of binding damage, which this was being used to pressure:
the account is **not** wounded by the ordering, because there is no sound ordering for it to fail
to explain. A predictor with no variance (every prose skill is equally absent from a 100%-JSON
training corpus — `CORPUS_BINDING_AUDIT.json`, 0/384) and a withdrawn ordering cancel out.

## 5. What a bank that COULD answer it would need

Not more arms and not more α points — **more `object_binding` damage**, which means an object
stratum that is neither cue-carrying (PRES-005) nor ceilinged (PRES-007). Supa at 118/120 leaves
2 rows of headroom; the bank has to be harder on objects while staying surface-matched to
persons. That is a build problem, and Amendment 1 already said it: *"only more `object_binding`
rows fix it, not more arms."* It was right, and the realised denominator of 4 makes it sharper.

---

# AMENDMENT 3 — conclusion 2 was arithmetic; its replacement is UNDER-POWERED

Amendment 2 §3 reported *"the lower α repairs more — 0.606 against 0.364, monotone down the
ray"* as a finding. **It is a definitional identity, and it is struck.** Damage is defined as
Supa-right-and-λ=0-wrong, so repair at α=0 is 33/33 = 1.000 *by construction*; any monotone path
from λ=0 back to Supa must show repair rising. Monotonicity would hold of a useless lever too.

The peer session proposed the right replacement — the **shape**, not the direction. Both interior
points sit above a linear `repair = 1 − α` reference, which would make repair concave in α and
would be the operationally meaningful claim: at half the delta you would get back more than half
the binding damage, so an interior operating point would strictly dominate.

**Measured against its own denominator, that claim is not detectable here:**

| arm | nominal α | observed | linear `1−α` | excess | two-sided binomial p |
|---|---:|---:|---:|---:|---:|
| `ownA05` | 0.50 | 20/33 = 0.606 | 0.500 | **+3.5 rows** | 0.2962 |
| `ctrl-own-alpha0.75` | 0.75 | 12/33 = 0.364 | 0.250 | **+3.8 rows** | 0.1573 |

The smallest detectable excess at n = 33 is **+6.5 rows** against linear-0.50 and **+5.8** against
linear-0.25. Observed excesses are roughly half of detectable. Both points fall on the same side,
but the joint sign probability under the linear null is 0.25 with two points — the floor of that
test, and not evidence.

> **Registered here, in the same form Amendment 1 used for uniformity: a null on concavity
> reports as "not powered to distinguish", and a positive-looking pair reports the same way.**
> This is the `references_retained` shape for the third time in this programme.

**And the reference curve is itself a choice.** Nothing derives `repair = 1 − α` as the null;
INT-001 shows every endpoint along this ray is steeply non-linear in α (acquisition
14 → 145 → 219 → 234 → 235). Measuring concavity against an underived reference is a tolerance
wearing a null's name — see [[nulls-carry-detectable-effect]].

## What survives without a reference curve

On **this lane**, α = 0.50 beats α = 0.75 on every axis measured:

| axis | α=0.50 | α=0.75 | source |
|---|---:|---:|---|
| alias | 237 | 220 | 17 losses, **0 gains**, p = 1.5e-05 |
| binding repair (person) | **20/33** | 12/33 | this bank |
| acquisition | 215 | 208 | +7, p = 0.281 — *not worse*, not better |

Read directly off measurements, with no interpolation and no assumed shape. It does **not**
locate the knee; it says α = 0.75 is not it on this lane. And it is a **third endpoint on the
SAME lane**, corroborating the known LWF optimum rather than independent of it — INT-001 already
established the knee is lane-dependent (native 0.75, LWF 0.50) and recorded *"never a rule"*.

## The part of Amendment 2 §3 that stands unchanged

**Zero losses, both arms, all three strata.** The registered disconfirmer — *"any stratum showing
repairs AND losses vs λ=0"* — did not fire anywhere. Additivity now holds on two independently
built banks across two task types. That needs no reference curve, no assumed shape and no
denominator argument: it is a count of zero in six opportunities.

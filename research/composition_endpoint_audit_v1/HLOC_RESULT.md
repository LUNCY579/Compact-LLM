# HLOC-001 RESULT — **INDETERMINATE.** The design could only have fired if an ablation destroyed ~44 % of δ_H's entire effect.

**2026-09-23 · four arms, zero training, zero budget grant · 4/4 scored, all gates passed.**
`quality_claim: false`, `release_permitted: false`. **Promotes nothing, claims nothing.**

Registered `ac86bab0`, **Amendment 1 `9b164314` filed with zero artifacts on disk**, scorer
committed **before the battery finished** (`f8ab1a5d`) with P2 as a precondition.

---

## 1. The registered outcome

```
comparison            abl  scalar   net   b   c  nd          p       Holm          limb
hattn  vs hattn_s      73      76    -3   1   4   5   3.750e-01  3.750e-01   UNRESOLVED (P2)
hmlp   vs hmlp_s       73      79    -6   0   6   6   3.125e-02  6.250e-02   resolved, ns
```

> **INDETERMINATE.** The `hattn` limb is UNRESOLVED, so no registered outcome — H_ATTN, H_MLP,
> H_BOTH, H_DOSE or H_RESTRICTION — is satisfied. **My registered prediction H_ATTN did not
> fire, and neither did the H_RESTRICTION I had moved to.**

INDETERMINATE was registered *as a real outcome*, precisely so an underpowered result could not
be read as H_DOSE. It is being used for the thing it was registered for.

## 2. Why it could not fire — and this was computable before the run

Holm over two comparisons requires the smaller-p limb to reach `p < 0.025`:

| nd | min attainable p | ×2 for Holm | can fire? |
|---:|---:|---:|---|
| 5 | 0.0625 | 0.1250 | **no** |
| 6 | 0.0312 | 0.0625 | **no** |
| **7** | 0.0156 | 0.0312 | **YES** |

**The design needed 7 one-directional discordant rows.** And:

> **The `hmlp` limb produced THE MOST EXTREME RESULT AVAILABLE at its discordance — b=0, c=6,
> all six discordant rows one way, `p = 0.03125` which IS its minimum attainable — and
> multiplicity correction still refused it.**

**δ_H's entire boundary effect is +16** (85 vs 69 — the one certified positive in ARMPERP-001).
HLOC asks whether ablating part of that second component costs some *fraction* of 16. Needing 7
clean rows means **the design could only fire if an ablation destroyed ≈ 44 % of δ_H's whole
effect.** It destroyed **3 and 6 rows — 19 % and 38 %.** Ordered, in the predicted direction,
and under the bar.

**That number was in my own result document.** I borrowed a design from a lane whose effects
were 2–4× larger in relative terms (+20 and +27 on 192 rows) — because *their* ablation removes
a **donor** from a two-donor composition at full strength, while mine removes part of a second
component carrying 0.12–0.13 of a 0.69 total norm. **Borrowing a design without re-deriving its
power for a second-order quantity is the error, and it is mine.** A power calculation against
δ_H's own +16 would have shown this before any arm was built.

## 3. What the data descriptively shows — reported, claimed as nothing

```
hattn 73   hattn_s 76      hmlp 73   hmlp_s 79      h025 85   control 69
```

**Both ablations sit below their own norm-matched scalar controls (−3, −6), in the direction
H_RESTRICTION predicts.** Neither reaches the registered bar. This is a *direction*, not a
result, and the distinction is the whole reason P2 exists.

**And the descriptive cell registered with no prediction:**

```
hattn 73  vs  hmlp 73      net 0      nd 2      p 1.000
```

**Two ablations keeping completely disjoint module families land on the identical count.** The
sibling lane's signature — attention-only 68 against MLP-only 67 across a **50 %** energy gap —
appears here across a **~10 %** gap, on a different skill and a different instrument. At nd = 2
it is unresolvable and I claim nothing from it, but it is the second lane in which *which*
family you keep makes no detectable difference while *restricting at all* costs something.

## 4. What this does NOT license

- **Not H_DOSE.** "No registered outcome fired" is not "location does not matter". The
  `hattn` limb was never readable.
- **Not a refutation of the sibling lane's P4** (the trailing-period convention *is*
  attention-carried in their lane) nor of their P6 (output *shape* is module-insensitive). This
  endpoint could not resolve which kind the boundary skill is.
- **Not "the ablation axis is closed".** It is closed *at this effect size on this endpoint*.
  A second component with a larger own-effect than δ_H's +16 could still be localised here.
- **Not a retention result.** Excluded in advance, same reason as ARMPERP-001.
- **`hattn` vs `hmlp` is not norm-matched** (0.1196 vs 0.1311) and carries no prediction.

## 5. Gates — all passed, and the positive control fired first

```
GATE 2  ||P16(dT)|| 0.674029797822137 reproduces MAG-001's 0.674029920773733
GATE 4  POSITIVE CONTROL, before any correct arm: scalar coefficient +1% -> 0.120753282
        vs target 0.119557705, err 1.196e-03 against tol 1e-9 -- REJECTED. The gate can fail.
GATE 1  all four second-component norms match target: err 0.00e+00 on three, 2.78e-17 on hattn_s
GATE 5  ablations verified REAL and BITWISE zero (a peer's design note -- a scalar-times-zero
        bug yields -0.0, and float equality on the untouched set is the cheapest way to see it):
          hattn keeps attn -> zeroed 60 mlp + 80 norm tensors
          hmlp  keeps mlp  -> zeroed 80 attn + 80 norm tensors
GATE 6  all four output sha256 distinct
CANARY  h025 reproduces 85 and the control 69 from per_row, asserted before any arm was read
```

Weights deleted after each arm; all four rebuild from `run_hloc.py` plus two committed
checkpoints to the recorded shas. Peak disk one arm.

## 6. What the episode is actually worth

The instrument behaved correctly and the registration did its job: **a prediction that moved
twice before any score then failed to fire, and the machinery said so instead of finding a
reading.** The scorer was committed before the numbers existed, so its verdict logic could not
be shaped by them — which is the specific repair for what ARMPERP-001's control flow did.

**The transferable lesson is about power, not about modules:**

> **A design borrowed from another lane inherits its effect size, not just its structure.**
> Re-derive the power against the effect *you* are testing. Mine was a second-order quantity —
> a fraction of an effect that was itself only 16 rows — and the bar was 7.

---

# Amendment 1 — P2's threshold was miscalibrated for my own correction, and the scorer let an unresolved limb satisfy a negative clause

**Both found by the peer who scored the artifacts COLD, from the code and the prereg, before
opening an arm. Both verified here. Neither changes the verdict — INDETERMINATE either way,
and now more firmly.**

## (a) P2 registered the UNADJUSTED threshold

P2 says *"discordance < 6 is UNRESOLVED"*. **That is the threshold for an unadjusted exact
test.** This design applies Holm across two comparisons, so the smallest-p limb must clear
α/m = 0.025:

| nd | min attainable p | ×2 for Holm | resolvable? |
|---:|---:|---:|---|
| 5 | 0.0625 | 0.1250 | no |
| **6** | **0.03125** | **0.0625** | **NO** |
| 7 | 0.015625 | 0.03125 | yes |

**Under the correction actually applied, the smallest resolvable discordance is nd ≥ 7.** The
threshold is now **derived** in the scorer from `m` and `α` (`MIN_ND_FOR`) rather than
hardcoded, so it cannot drift from the procedure again.

`hmlp` is the proof rather than a hypothetical: **b = 0, c = 6 is the most extreme split
obtainable at that discordance** — it produced exactly its minimum attainable p — and Holm took
it to 0.0625. **That limb was admitted as RESOLVED by P2 and was unfalsifiable at
significance.**

> **Gates must be able to fail AND pass, one level up: P2 can do both, but the COMPOSITE of P2
> and the multiplicity correction could not.** My own §2 computed "the design needed 7" and
> read it as a *power* finding; I did not turn it on the registered precondition, which is
> where it bites.

**Effect on the verdict:** with the corrected threshold **both** limbs are UNRESOLVED, so
INDETERMINATE is doubly correct rather than marginally.

## (b) An unresolved limb could satisfy a negative clause — third occurrence of this shape

`not above(x)` is TRUE for an unresolved limb. Only H_DOSE required both limbs resolved, so
H_ATTN, H_MLP and H_RESTRICTION could each fire with one limb unreadable. Enumerated over all
sixteen limb-state pairs:

```
BEFORE:  4 of 16 states fired a substantive verdict with an UNRESOLVED limb
           UNRESOLVED / resolved-above  -> H_MLP
           UNRESOLVED / resolved-below  -> H_RESTRICTION
           resolved-above / UNRESOLVED  -> H_ATTN
           resolved-below / UNRESOLVED  -> H_RESTRICTION
AFTER:   0 of 16
```

**It did not fire here** — it is flagged because the peer found it from the code before any
number, and because correction (a) makes one-limb-unresolved states *more* common.

**This is the same defect as ARMPERP-001's `not sig(a)`, for the third time.** So the repair is
structural rather than per-branch: a `both_resolved` guard **hoisted above every branch**.
Patching individual clauses is what failed the first two times.

> **A decision procedure is a function and should be TESTED LIKE ONE.** Enumerating sixteen
> limb-state pairs took one script and found in seconds what reading the branches twice did
> not. Any registered outcome table with ≥ 2 limbs should ship with that enumeration.

## Provenance of the catch

The objection *"P2's resolution threshold and how it is calibrated"* was raised by my own
adversarial panel and was **among the 32 verifiers that died on the usage limit** — reported at
the time as INSUFFICIENT COVERAGE rather than as a verdict. It was then found independently by
a peer. **The class was flagged by the panel and the instance by the peer**, which is the
cross-check working in the direction we had both said it could not.

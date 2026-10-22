# DPACQ-001 RESULT — **THE PRIMARY RETRACTS.** The matched-norm retention advantage was BOUGHT.

**2026-09-22 · 4 arms × 384 views × 128 new tokens · `BATTERY COMPLETE 4/4`, 0 failed · zero
training, zero building, no budget grant.** Registered in `DPACQ_PREREG.md` with two amendments,
all three commits made with **zero acquisition artifacts on disk**, verified at each.

`quality_claim: false`. `release_permitted: false`. **No arm promoted.**

---

## 1. The registered retraction, first

> **P1 PRIMARY: `acq(rung4) = 266` vs `dense a100 = 298`, both at norm 1.8464. Net −32.**
> Registered: **`net ≤ −20` RETRACTS.** −32 ≤ −20.
> **P1 RETRACTS. The +12 retention advantage at matched norm was BOUGHT, at a cost of 32 views
> of acquisition.**

DOSEPARAM-001's §1 headline — *"parameterization is the lever at matched norm"* — stands as a
**retention** result and must now carry its price. *"LoRA retains more at matched norm"* is true;
*"LoRA is better at matched norm"* is **false on this evidence**.

**P2 also fails.** Registered: LoRA saturates by rung 2, rungs 2–4 within ±10.

```
          norm 0.4616   0.9232   1.3848   1.8464      steps
LoRA            52       159      224      266     +107, +65, +42   still climbing
dense          188       291      300      298     +103,  +9,  -2   saturated by rung 2
```

Rungs 2–4 span **107**, not ±10. **The two limbs do not saturate alike** — dense is done by
α = 0.50, LoRA is still climbing at β = 0.62.

## 2. The full picture

| norm | LoRA acq | dense acq | Δacq | LoRA ret | dense ret | Δret |
|---:|---:|---:|---:|---:|---:|---:|
| 0.4616 | 52 | 188 | **−136** | 44 | 40 | +4 |
| 0.9232 | 159 | 291 | **−132** | 43 | 37 | +6 |
| 1.3848 | 224 | 300 | **−76** | 42 | 31 | +11 |
| **1.8464** | **266** | **298** | **−32** | **42** | **30** | **+12** |
| 2.9753 | 296 | — | — | 38 | — | — | ← LoRA β=1.00 |

> **At every rung LoRA pays acquisition for retention.** The trade narrows as norm rises
> (−136/+4 → −32/+12) but **never vanishes at matched norm.**

## 3. P3 was promoted to load-bearing before the data, and it is what explains the result

`rung4` vs `LoRA β=1.00` — transforms of **one** delta, floorless:

```
rung4 266  vs  beta=1.00 296        net -30    scaling the LoRA delta to 62% costs 30 views
```

**Chain check, and it closes exactly:**

```
rung4 -> beta1.00   -30   floorless, one delta
beta1.00 -> a100     -2   cross-draw, ALREADY MEASURED
                    ----
                     -32   vs direct rung4 -> a100 = -32      CONSISTENT
```

Amendment 2 promoted P3 so the floored cross-draw step would be spent **once, on a number
already on disk.** It worked: the −32 is reached by two independent routes that agree exactly,
and the floored step contributes 2 of the 32.

## 4. THE FINDING — the two matchings give OPPOSITE answers, and both are now measured

```
matched RELATIVE dose   beta=1.00 vs alpha=1.00     acq -2    ret +8    retention nearly FREE
matched ABSOLUTE norm   norm 1.8464 both            acq -32   ret +12   retention BOUGHT
```

A peer registered this as a **bracket** before either end was measured, on the grounds that
LoRA's trained norm is 1.611× the dense limb's, so **matching absolute norm necessarily unmatches
fraction-of-own-trained-strength**. Both ends now exist and they disagree.

**The mechanism, and it is visible in §2's shape.** The two parameterizations have different
**acquisition-per-unit-norm efficiency**: dense reaches its acquisition ceiling by norm ≈ 0.92,
LoRA has not reached its by 1.85 and needs ≈ 2.98. So **holding norm equal handicaps LoRA on
acquisition** — the matched-norm cell is not a neutral comparison, it is one that costs LoRA 30
views of its own delta's strength.

> **"Is the retention free?" has no answer independent of the matching.** At its own operating
> point LoRA gets +8 retention for −2 acquisition. Forced to dense's norm it gets +12 for −32.
> Neither is the true answer; both are true at their own cell.

## 5. Rung 1 is NOT interpreted here — the anchor does not exist yet

Rung 1 (52/384, **1 of 3 family floors**) is the only rung failing floors; rungs 2–4 clear 3/3.

`ANCHOR128_PREREG.md` registered — before any anchor number — that **I will not call rung 1 a
"no-op" or "actively degraded"** until a source baseline exists at 128 tokens. There is none:
the bank's only baselines (`rrp015_supa`, `sv001_supa`, both **14/384**) are at **96**, and
quoting them across the decode boundary is the borrowed-scale error this lane amended twice for.

What *is* established, because floors are a **bank** property and byte-identical across budgets:
the source clears **0 of 3** floors; rung 1 clears **`register_machine` at 25 against a floor of
9**, where the source scores **0**. So rung 1 **acquired something real and insufficient** —
neither prior reading survives. The anchor runs next under a **registered reading rule**.

## 6. What this does NOT license

- **Not a refutation of the retention result.** DOSEPARAM-001's +12 is real and independently
  verified; what is refuted is reading it as "better".
- **Not "LoRA is worse".** At matched relative dose it is −2 acq / +8 ret. The claim is about the
  matched-**norm** cell only.
- **Not a general parameterization result.** One delta pair, one base, rank 16, one seed, and
  **cross-draw** — one LoRA draw against one dense draw.
- **Not a statement about the 96-token family.** Every number here is at 128; the two families
  are separate and the 20-view floor used in P1 is **borrowed from the 96-token family** and
  labelled as such in Amendment 2.
- **Rung 1 is uninterpreted** pending the anchor.

## 7. Provenance

`process_probe.py` **unmodified**, `implementation_digest a329eb7bb06f` — the same script and
digest that produced all five comparators. Only the driver is new. Every artifact asserted after
writing: bank `7dd53f2a`, `max_new_tokens 128`, `views 384`, 384 predictions, digest match —
**4/4 comparable**. The probe is **create-only** (refuses if the artifact exists) rather than
skip-on-exists, so this battery could not have reported success having computed nothing.

Arms are DOSEPARAM-001's, independently rebuild-verified earlier: rung 4's data-section sha256
matches its recorded value **bit-for-bit**. One lease held for the whole battery,
`hostbudget.require(3.0)` — the probe's own gate, unchanged. ~19 minutes of card time.

---

# Amendment 1 — §3's "chain check" is a TELESCOPING IDENTITY. It cannot fail, and I published it as a consistency check.

**Raised by the peer whose result this experiment retracted.** Verified here rather than accepted.

## What I wrote, and why it is worthless as a check

```
rung4 -> beta1.00   -30            (a - b)
beta1.00 -> a100     -2            (b - c)
                    ----
                     -32           vs direct rung4 -> a100 = -32    "CONSISTENT"
```

`(a − b) + (b − c) = a − c`. **The intermediate cancels. This is arithmetic, not measurement,
and it returns agreement for any three numbers whatever they are.** Demonstrated on random
triples rather than argued: it held on 5 of 5, as it must on all of them.

> **It is the class this record has been cataloguing all night: a check whose passing answer is
> the only answer it can return.** I described it as a two-route agreement and it is one route
> written twice. **Withdrawn as a check.**

## What the decomposition DOES buy, and this part stands

It **attributes** the gap, which is a real and useful statement:

| step | contribution | axis |
|---|---:|---|
| within-limb scaling, `rung4 ↔ β=1.00` | **30 of 32** | floorless, paired, transforms of one delta |
| cross-draw, `β=1.00 ↔ a100` | **2 of 32** | floored, already measured |

**Nearly the whole deficit lives in the within-limb scaling step**, on the axis where the
comparison is paired and floorless. That is why the result is readable at all despite resting on
a borrowed floor: the floored step contributes 2.

**And promoting P3 was still right** — for the reason registered at the time, that the sharp
number comes off the floorless axis and the floored step is spent once on a number already on
disk. **Not** because it produced an independent second route, because it did not.

## The peer's verification, recorded

Nine arms, **three routes each** — `aggregate.canonical_correct`, a direct count over the per-row
predictions where `canonical` is True, and an independent sum over `per_family` — all agreeing,
384 predictions each. **P1's retraction stands on their extraction as well as mine.**

## The hypothesis that fits, which neither of us registered

At matched **norm** the two limbs sit at different points on **their own** acquisition curves:
dense has saturated by 0.92 and is losing retention steadily; LoRA is still climbing at 1.85 and
has barely lost any. **The retention advantage and the acquisition deficit are the same fact seen
twice.**

> All three registered rivals — H_NORM, H_BEND, H_PARAM — were about the **retention axis
> alone**. The variable that explains the result was on the endpoint that had not been measured.
> **A discriminating test among the wrong candidate set is still a discriminating test of the
> wrong question.**

## And the design criticism that follows, which is theirs and is correct

The **matched-acquisition** comparison predates this entire design and already supported the
conclusion:

```
three LoRA arms   296 / 297 / 299   spread 3, inside the 20-view floor
dense a100        298
retention          38 /  38 /  37   against 30
```

Three arms, three experiments, no new builds. **DOSEPARAM-001 and DPACQ-001 together cost four
arms and ~40 minutes of card time to reach a conclusion those three already supported.** What the
new arms added is the *price* at the matched-norm cell — which is worth having, and is a smaller
addition than the effort implies.

---

# Amendment 2 — the mechanism, as MEASUREMENT: fraction-of-own-ceiling across all four rungs

§4 argued the mechanism from the *shape* of the acquisition curves. A peer supplied a single
measured point for it; extending that point to all four rungs makes it a table rather than an
inference.

Each limb expressed as a percentage of **its own** acquisition ceiling — LoRA's 296 at β = 1.00,
dense's 300 at α = 0.75:

| norm | LoRA | % of its 296 | dense | % of its 300 | dense/LoRA |
|---:|---:|---:|---:|---:|---:|
| 0.4616 | 52 | **17.6 %** | 188 | **62.7 %** | **3.57×** |
| 0.9232 | 159 | 53.7 % | 291 | 97.0 % | 1.81× |
| 1.3848 | 224 | 75.7 % | 300 | 100.0 % | 1.32× |
| **1.8464** | 266 | **89.9 %** | 298 | **99.3 %** | **1.11×** |

> **At any shared norm, dense sits further along its own curve than LoRA sits along its — and
> the disparity closes monotonically as norm rises: 3.57 → 1.81 → 1.32 → 1.11.**

**This reframes the −32 at the primary cell.** At rung 4, dense is essentially *at* its ceiling
(99.3 %) while LoRA is **10 points short of its own** (89.9 %). So the matched-norm acquisition
deficit is mostly **LoRA not yet being at full strength** — not a deficiency of the
parameterization, but a consequence of where the shared norm lands on each limb's own dose curve.

**Which is why the two matchings disagree**: matched *norm* compares a limb at 89.9 % of itself
against one at 99.3 % of itself; matched *relative dose* compares both at 100 %. The bracket in
§4 is the same fact stated without the percentages.

## One wording withdrawn, and it was mine

I described rung 1 as making the limb *"an axis with one end unusable"*, and a peer adopted it.
**They withdrew it and they are right to.** Rung 1 is interpretable and measurable: **+38 over
the source at matched budget, one family floor cleared that the source cannot touch.**

> **The narrower true statement:** rung 1 sits very early on the LoRA limb's own dose curve —
> **17.6 % of its own ceiling.** Its 44/47 retention is high because it changed the model
> little: **not because it is broken, and not because it is inert.**

The §3 retraction in DOSEPARAM-001 is unaffected — the limb reads flat across all four rungs
because it is early on its curve throughout, which is a dose fact and not an advantage.

*(Checked: no committed document in this lane carries the withdrawn wording. It appeared only in
a message, and the peer's document that adopted it has been amended.)*

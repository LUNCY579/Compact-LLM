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

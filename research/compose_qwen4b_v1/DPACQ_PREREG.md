# DPACQ-001 — do the scaled LoRA rungs BUY their retention with acquisition?

> **STATUS: REGISTERED, NOTHING SCORED.** No acquisition artifact exists for any DOSEPARAM rung
> as of this line — verified by search across the records tree and all ChatBubble checkouts.
> The arms already exist (built and independently rebuild-verified), so the registration window
> that is still open is the one before the first **score**, and this file precedes it.

**Class: FOLLOW-ON MEASUREMENT. Zero training, zero building.** Four existing arms, one
instrument, one battery. `quality_claim: false`, `release_permitted: false`. Promotes nothing.

---

## 1. Why, and what is at stake

DOSEPARAM-001 measured **retention** on four LoRA rungs at norms matched to the dense limb:

| rung | norm | LoRA retained | dense retained | diff |
|---|---:|---:|---:|---:|
| 1 | 0.4616 | 44 | 40 | +4 |
| 2 | 0.9232 | 43 | 37 | +6 |
| 3 | 1.3848 | 42 | 31 | +11 |
| 4 | 1.8464 | **42** | **30** | **+12** |

Its own registered limit, which this experiment exists to close:

> *"Until acquisition exists for the four scaled LoRA rungs, 'LoRA retains more at matched norm'
> is not 'LoRA is better at matched norm' — a scaled rung could be buying retention with
> acquisition the way `a025` does at 188/384 with retention 40."*

That failure mode is **real and already on record in the comparator limb**: dense `a025` sits at
norm 0.4616 with retention 40 and acquisition **188/384**, against `a100`'s 298. Low norm buys
retention with acquisition on the dense side. **Nobody has checked whether it does so on the
LoRA side.**

## 2. The floor, and why it makes the primary a NON-INFERIORITY test

The programme records that SELECT-VIEWS cross-draw claims need **|net| ≥ 20**, and that **the
floor does not apply to transforms of one delta**. Both halves bind here, in opposite directions:

| comparison | kind | floor |
|---|---|---|
| rung ↔ rung | transforms of ONE LoRA delta | **no 20-view floor**; paired |
| rung ↔ dense | **cross-draw** (different training runs) | **20-view floor applies** |
| rung ↔ LoRA β=1.00 | transforms of ONE delta | no floor |

So the primary cannot be "is LoRA better on acquisition" — it must be **"is LoRA detectably
worse"**, because the claim at risk is that the retention is *free*.

## 3. Registered predictions

### P1 — PRIMARY. `acq(rung4)` vs `acq(dense a100) = 298`, both at norm 1.8464.

> **FIRES (retention is free) if `|net| < 20`** — no detectable cross-draw acquisition
> difference at matched norm.
> **RETRACTS if `net ≤ −20`** — the +12 retention was **bought**, and DOSEPARAM-001's headline
> needs qualification. This is the retracting outcome and it is registered as such.
> `net ≥ +20` would be a LoRA acquisition *advantage*, unpredicted; reported, claimed as nothing.

**Anchor:** LoRA at β = 1.00 (norm 2.9753) scores **296**; dense is flat from 0.92 → 1.85
(291, 300, 298). Point prediction for rung 4: **290–300.**

### P2 — SHAPE. Acquisition across rungs 1 → 4, paired within the LoRA limb (no floor).

Dense saturates by norm ≈ 0.92 — **188, 291, 300, 298**. Registered prediction: **LoRA
saturates too, by rung 2**, i.e. rungs 2–4 within ±10 of each other and rung 1 lowest.

> **The outcome that matters most is the bad one:** if **rung 1 lands near 188**, the low-norm
> rungs are buying retention with acquisition exactly as the dense limb does, and "the LoRA
> retention limb is flat" reads very differently — flat retention at a collapsing acquisition
> cost is not an advantage.

### P3 — ANCHOR, descriptive, no prediction.

`rung4` ↔ `LoRA β = 1.00` (296). Same delta, two magnitudes, no floor. Measures what scaling the
LoRA delta to 62 % costs on its own terms.

## 4. Comparability gates, all before any score is read

1. **Instrument** `select-views-001`, **384 views**, bank `views_sha256 7dd53f2a…` — the
   acquisition bank, **not** the dose bank `45674251`. Every comparator artifact is on
   `7dd53f2a`; scoring a rung on the other bank would reproduce my own label-for-bank error.
2. **`--max-new-tokens 128`**, matching all five comparators. The probe's default is **96** and
   using it would silently break comparability.
3. **`implementation_digest a329eb7bb06f…`** must match the comparators'.
4. **Per-family floors** recomputed from the frozen views and checked against
   `PROCESS_PROBE_FLOORS.md` before any model loads — the probe does this itself and refuses on
   mismatch.
5. **`aggregate.canonical_correct` is the reported endpoint**, with per-family floors reported
   beside it. The probe's own note: *"the aggregate is REPORTED, never the gate."*
6. **Create-only**: the probe refuses if `PROCESS_<label>.json` exists. It does **not** skip —
   the opposite of the DOSEPARAM builder's vacuous guard, and verified as such at line 263.

## 5. What this does NOT license, whatever the numbers

- **No promotion.** One base, one delta pair, rank 16, one seed.
- **Not a retention result.** Retention is already measured and is not re-run.
- **P1 firing does not prove equivalence** — it bounds the difference below the programme's own
  cross-draw floor, which is a detection limit, not a demonstration of no effect.
- **Not a statement about ranks other than 16**, nor about LoRA in general.
- **Rung 1 is the weakest cell** — it is the furthest from both limbs' trained strength.

## 6. Envelope

4 arms × 384 views × 128 new tokens, greedy, one lease. `hostbudget.require(3.0)` is the probe's
own authoritative gate — **used unchanged, not tightened.** Zero training, zero building,
no budget grant. Outputs are one JSON per arm in this worktree's own evidence directory; the
peer's create-only tree is not written to.

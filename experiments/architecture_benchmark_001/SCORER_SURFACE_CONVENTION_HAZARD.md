# HAZARD: this scorer tolerates a trailing period. The acquisition probe does not.

**Placed next to `retention.py` because that is where a future reader trips over it.**
Measured 2026-09-23. Costs nothing to read; cost another lane 91 rows to discover.

---

## The two conventions

| scorer | domain | trailing period |
|---|---|---|
| `retention.py` (`_strip_one_period`, line 131) | sentence-like answers — `"60 units."` | **STRIPPED — at most one** |
| `experiments/rab001_exploration/process_probe.py` | JSON / numeric — `{"A":-15,...}`, `106` | **NOT stripped** (`.strip()` only) |

**This is not an inconsistency.** Each scorer matches its own domain, measured:

```
RETENTION  domain references ending '.'    48 / 144  (33%)
ACQUISITION domain references ending '.'    0 / 384  ( 0%)
```

## But `_strip_one_period` is LOAD-BEARING, not decorative

```
sampled 144 retention rows across 6 artifacts
  SOURCE answers ending '.'   48  (33%)
  ARM    answers ending '.'   79  (55%)   <-- arms emit MORE than the source
  rows where the stripped period changes the verdict   22  (15%)
```

> **Trained arms drift toward emitting a trailing period relative to the source (55% vs 33%),
> and in 15% of sampled rows the tolerance is what keeps the answer scoring as correct.**
> Every retention number in this programme is conditional on that one line.

## THE HAZARD

Composition **across** the two domains. A delta trained where answers end in periods, merged
and then scored by the strict probe, imports the surface convention and is counted wrong.

**This is not hypothetical.** A sibling lane (`fresh_dual_skill_instrument_v1`, see
`CROSS_LANE_HAZARD_scorer_pair.md` there) found **91 of 96 "lost" rows in one composed arm were
the gold answer plus exactly one trailing period.** The apparent result was a capability cliff —
a score of 6/192 where the tolerant score was 97/192. It was a punctuation mark.

**It is invisible to every weight-space statistic.** That lane tabulated six that missed it:
elementwise cosine 0.0025, in-subspace 0.012, per-direction 1.12× null, subspace collision
3.15×, effective rank ~18, and `‖ΔΘ‖`. Tolerant scores were *flat* across norm 1.23–1.52 while
strict scores were not, so it is not magnitude either.

## Does it reach the arms in THIS tree? Audited: NO

All four ARMPERP-001 comparators, 384 predictions each, every non-canonical row tested under
five trivial normalisations (strip whitespace / one period / all trailing periods / casefold /
drop all whitespace):

```
recovered rows:  0   on every arm, every normalisation
trailing periods emitted:  0 / 384  on every arm
```

**With controls, because a zero from an absence check is what a broken check also returns:**

```
POSITIVE CONTROL  25 injected gold+'.' rows  ->  recovered 25/25   the check CAN fire
NEGATIVE CONTROL  25 genuinely-wrong rows    ->  recovered  0/25   no false positives
NAMESPACE         384/384 rows have both prediction and reference populated
```

Clean **because δ_T and δ_H both come from the non-period domain.** That is a property of the
donors, not a property of the method.

## What to do about it

1. **Before composing a donor trained in another domain, check its answer format**, not just its
   norm and cosine. `grep` the donor's training answers for a terminal period.
2. **When a composed arm scores unexpectedly low, diff the generations against the golds before
   believing the number.** One character moved 91 rows in the sibling lane, and both the
   geometry and the aggregate score were consistent with the wrong conclusion.
3. **Never report an absence of this defect without a positive control.** The check returns 0
   when it works and 0 when it is broken.
4. If `_strip_one_period` is ever removed or tightened, **retention numbers across this whole
   programme move** — see the 15% figure above. It is not a formatting nicety.

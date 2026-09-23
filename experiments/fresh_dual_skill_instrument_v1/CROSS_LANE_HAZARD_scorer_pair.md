# Cross-lane hazard: the output contract is a composed quantity

**Status:** one positive case (FRESH-DUAL-SKILL-001, this lane) and one negative control
(ARMPERP-001, DOSELORA lane, offered for co-signature). Dated 2026-09-23.

This note lives in this experiment's directory because writing into another lane's checkout is
not this session's to do. **If you maintain a scorer that a merged model might be scored by,
link this from next to the scorer** — that is where a future lane trips over it, not here.

## The hazard in one line

> **When you merge two donors trained under different answer formats, the output contract is a
> composed quantity too — and the surviving convention may not be the one your scorer expects.**

It is not a property of any arm. It is a property of the **scorer pair**.

## The positive case

Titan-Sparse donates the transition skill under a **strict** contract: the answer is a bare JSON
state object, scored by exact string match. Opallo donates arithmetic under a **period-tolerant**
convention: its training answers are `64.`, `40.`, `$4.`, `10:25 AM.`, and the scorer it was
measured with — `architecture_benchmark_001/retention.py` — contains

```python
def _strip_one_period(text: str) -> str:
    """Remove at most one closing sentence period. `5...` keeps two and will fail to parse."""
```

Compose them and score with Titan's scorer:

| arm | β (Opallo) | strict | period-tolerant | trailing periods emitted |
|---|---|---|---|---|
| SPARSE | 0 | 102 | 102 | **0** / 192 |
| AVG | 0.5 | 69 | 69 | **0** / 192 |
| ADD-O75 | 0.75 | 63 | 98 | **63** / 192 |
| ADD-100 | 1.0 | **6** | **97** | **178** / 192 |

**91 of ADD-100's 96 lost rows are `gold + "."` exactly.**

```
gold {"A":27,"B":22,"C":-7,"D":33}     gen {"A":27,"B":22,"C":-7,"D":33}.
gold 67                                 gen 67.
```

Read strictly, that is a 96-row capability collapse. It is one character, dose-gated, and it
appears only at β ≥ 0.75.

## Why nothing caught it

Every weight-space statistic computed on this pair was blind to it, and several were computed
specifically to predict composition cost:

| statistic | value | saw the hazard? |
|---|---|---|
| elementwise cosine | 0.0025 | no |
| cosine inside the contended subspace | 0.012 | no |
| per-direction alignment vs Haar null | 1.12× | no |
| subspace collision index | 3.15× chance | no |
| effective rank | ~18 | no |
| ‖ΔΘ‖ | 1.23 → 1.52 across the cliff | no — tolerant scores are **flat** across that range |

The aggregate **score column** was also consistent with the wrong answer, in two different
directions on the same night. Only diffing generations against gold found it.

## The negative control that makes the positive case interpretable

DOSELORA-001 ran the same check on four ARMPERP-001 arms, 384 predictions each
(`perp`, `h025`, `a0554`, `control`), under five normalisations:

```
strip whitespace | strip one period | strip all trailing periods | casefold | drop whitespace
    0                    0                      0                     0            0
trailing-period emission: 0/384 on all four arms; references 0/384 too
```

with controls on the check itself:

```
POSITIVE   injected 25 gold+'.' rows  ->  recovered 25/25   the check CAN fire
NEGATIVE   25 genuinely-wrong rows    ->  recovered  0/25   no false positives
NAMESPACE  384/384 rows have BOTH prediction and reference populated
```

Their misses are real arithmetic (`gold '106'` → `gen '110'`), not formatting. So the defect is
**absent** in a lane that scales a single delta, and **present** in a lane that composes two
donors with mismatched conventions. That contrast is what makes the positive case a finding
rather than an anecdote.

## The check, with its controls

Cheap enough that there is no reason to skip it. It found 91 rows here in one pass.

```python
# for each arm, over all rows:
strict      = gen.strip() == gold
one_period  = gen.strip() == gold + "."
other_norm  = gen.strip().rstrip(".").casefold() == gold.casefold()   # widen as needed
```

Then, before believing a zero — an absence check returns the passing answer when it is broken:

1. **Positive control** — inject N synthetic `gold + "."` rows and require the check recovers N.
2. **Negative control** — inject N genuinely wrong rows and require it recovers 0.
3. **Namespace control** — assert every row has both a prediction and a reference populated.

Widen beyond the period to whatever your donors' conventions differ in: casing, quoting,
thousands separators, currency symbols, unit words, field wrappers, a leading `The answer is`.

## What to do when it fires

- **Keep the registered rule as primary.** Do not retroactively relax a scorer to rescue a
  result. Report a strict column and a convention-tolerant column, and label the tolerant one
  secondary and post-hoc.
- **Report the convention transfer as a real regression.** It is not capability loss, but any
  downstream exact-match consumer breaks completely. It happens to be repairable by stripping
  one character; that is a fact about this instance, not a general licence.
- **Do not attribute it to magnitude without checking.** Here tolerant scores are flat across
  ‖ΔΘ‖ 1.23 → 1.52 while strict scores are not; a norm effect would move both columns.

## Open

Where the convention *lives* is being tested in `../module_disjoint_composition_v1`
(Amendment 2): an arm taking Opallo's contribution only from MLP and Titan's only from
attention localises it to one module class, or shows it is distributed. Registered before that
arm was scored.

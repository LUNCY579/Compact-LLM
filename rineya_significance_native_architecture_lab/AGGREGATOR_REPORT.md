# P4 aggregator screen, and the shuffled-S control

Run: [`halo_showcase64_aggregator_v1`](results/runs/halo_showcase64_aggregator_v1)
Classification: `EXPLORATORY_SCREEN`, unregistered. Halo 1.2.0 only.
256 case-conditions, four arms, single process.

Two questions in one screen: does a spike-preserving `S` aggregator help (audit
repair P4), and — the control this program had never run — does the
significance score carry any usable signal at all?

## Results

| Arm | aggregator | `e` | `S` shuffled | Exact | early | middle | late | dist | within B0-correct |
|---|---|---:|:--:|---:|---:|---:|---:|---:|---:|
| `B0` full cache* | — | — | — | **30/64** | 3 | 6 | 10 | 11 | — |
| `R192` positional-heavy* | sum | 1.0 | no | 25/64 | 4 | 2 | 9 | 10 | 22/30 |
| `B4_POS` position* | — | — | — | 22/64 | 0 | 0 | 11 | 11 | 20/30 |
| **`SUM_SHUF`** | sum | 1.0 | **yes** | **21/64** | 0 | 1 | 10 | 10 | 20/30 |
| `MAX_E0` | max | 0.0 | no | 20/64 | 1 | 2 | 11 | 6 | 19/30 |
| `EMA_E0` | ema α=0.5 | 0.0 | no | 20/64 | 2 | 3 | 11 | 4 | 18/30 |
| `B3` shipped* | sum | 1.0 | no | 19/64 | 1 | 3 | 10 | 5 | 19/30 |
| `B2`* | sum | 0.0 | no | 17/64 | 2 | 2 | 10 | 3 | 17/30 |
| `MAX_SHUF` | max | 0.0 | **yes** | 17/64 | 0 | 0 | 9 | 8 | 16/30 |

\* reference arms from earlier runs, same protocol.

## Finding 1 — the shipped significance score is inert

This is the headline, and it is the control the program had been missing. Every
prior shuffle test in this lab permuted the *counts* (`N-021`, `N-028`,
`N-033`) or an allocation statistic. None permuted the score itself.

```text
B3 vs SUM_SHUF   case 4W/6L p=0.7539   cluster 2better/3worse/11tied  p=1.0000
                 predictions differ on 44/64
```

`SUM_SHUF` is bit-for-bit the shipped B3 configuration — summed `S`, `e=1`,
same budget, sinks, and recent window — with `S` permuted among eligible
entries at every boundary, preserving the per-row multiset and both protected
denominators exactly while destroying which entry holds which score.

It scored **21/64 against B3's 19/64**. The real scores did not beat a random
permutation of themselves; they came in slightly behind.

This is not an inactive-feature artifact. The permutation changes predictions on
44 of 64 cases — it thoroughly rearranges the cache. Accuracy simply does not
depend on getting the arrangement right.

Combined with the earlier finding that content selection is indistinguishable
from a position counter (`B4_POS` 22/64 vs B3 19/64, cluster `p=1.0`), the
picture is consistent: for the shipped configuration on this suite, the
retention behaviour is carried by the **sink-plus-recent structure**, not by the
attention-derived scores that structure is wrapped around.

## Finding 2 — P4 is directional and not significant

```text
MAX_E0 vs B2   case 5W/2L p=0.4531   cluster 4better/2worse/10tied  p=0.6875
EMA_E0 vs B2   case 4W/1L p=0.3750   cluster 3better/1worse/12tied  p=0.6250
```

`B2` is the matched comparison: same `e=0`, differing only in whether `S`
accumulates by sum or by running max. Max scored 20/64 against 17/64 and EMA
also 20/64. Both improve on the summed aggregator directionally, neither
survives cluster-level testing, and neither reaches `B4_POS` (22/64) or `R192`
(25/64).

## Finding 3 — max is the first arm to beat its own shuffle

The specificity margin — real score minus its own eligible-pool shuffle — has
been zero or negative everywhere in this program until now:

```text
sum, e=1   B3      19/64  vs  SUM_SHUF   21/64   =  -2
max, e=0   MAX_E0  20/64  vs  MAX_SHUF   17/64   =  +3
```

```text
MAX_E0 vs MAX_SHUF   case 7W/4L p=0.5488   cluster 6better/2worse/8tied  p=0.2891
```

Six scenario packs improved and two degraded. That is **not significant**
(`p=0.2891`) and must not be reported as a positive result. But it is the first
configuration tested in which the true score outperforms a permutation of
itself at all, and it is the only positive specificity sign the program has
produced. The sign changed; the evidence did not arrive.

If anything here is worth a follow-up with real statistical power, it is this
contrast — a preregistered max-aggregator specificity test on a larger,
independent sample.

## Finding 4 — the compression gap is unchanged

```text
B0 vs MAX_E0   case 11W/1L p=0.0063   cluster 8better/1worse/7tied  p=0.0391
```

No arm in this screen approaches full cache, and `R192` — positional-heavy
allocation with no aggregator change at all — remains the best compressed
configuration tested at 25/64 and 22/30.

## Limits

- One core, one suite, one chunk schedule, one budget family.
- One value of `alpha` for EMA and one shuffle seed per arm.
- Exploratory and unregistered; no frozen endpoint, no advance gate, no
  multiplicity control across arms or across the four screens now run.
- Scenario packs are correlated; all claims use the 16-pack unit.
- The shuffle is applied at selection time to the ranking input. It does not
  perturb the persistent state that later boundaries inherit, so it isolates
  per-boundary selection rather than trajectory effects.
- `record_attention` was monkeypatched in-process. No file under `src/` was
  modified, so the Experiment F content-hash provenance is intact.
- MAX and EMA are applied as deltas through the audited `update()` path
  (`delta = clamp(contribution - S, min=0)` and `alpha * (contribution - S)`),
  verified against direct references before the run.

## Consequence for the program

Three mechanism components have now been tested against matched controls:

| Component | Result |
|---|---|
| Denominator `C` under uniform exposure | identical to arrival-chunk age (`N-044`) |
| Denominator `C` under salience gating | active but fails specificity (`N-052`) |
| Numerator `S` as shipped | inert; loses to its own shuffle (this report) |

The remaining untested audit item is P6, contiguous-span preservation, which is
a Cornerstone-2 factor and addresses the four multi-token entity fragmentations
observed in the showcase-64 evaluation. It is not a repair of Cornerstone 1.

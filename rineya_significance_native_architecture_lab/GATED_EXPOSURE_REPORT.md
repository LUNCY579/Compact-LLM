# P1: non-uniform realized exposure — result

Run: [`halo_showcase64_gated_exposure_v1`](results/runs/halo_showcase64_gated_exposure_v1)
Classification: `EXPLORATORY_SCREEN`, unregistered. Halo 1.2.0 only.
256 case-conditions, four arms, single process.

## What P1 was for

The [implementation-theory audit](IMPLEMENTATION_THEORY_AUDIT.md) showed that
the live path records a uniform scalar exposure of 1.0 for every resident
entry, making `C` identical to arrival-chunk age. Under that identity,
hypothesis **H2** — "realized exposure contains information beyond age" — is not
false, it is *vacuous*: exposure is *defined* as age, so the question cannot be
asked.

P1 was the minimum repair that makes H2 answerable. Exposure becomes salience
gated:

```text
exposure_i = 1  if  contribution_i > tau * mean_j(contribution_j)
exposure_i = 0  otherwise
```

`C` then counts the events at which an entry was actually salient, so
`S/max(C,1)^e` becomes mean attention *per salient event* — a spike magnitude,
which is age-neutral by construction. An early entry that spiked once and a
recent entry that spiked once score identically, which is exactly the property
the uniform implementation destroyed.

## The repair worked mechanically

| Arm | `tau` | `e` | distinct `C` | `C=0` fraction | `C` varies across KV heads |
|---|---:|---:|---:|---:|:--:|
| `SG10_E1` | 1.0 | 1.0 | 4.2 | 0.77 | **yes** |
| `SG10_E05` | 1.0 | 0.5 | 4.2 | 0.77 | **yes** |
| `SG20_E1` | 2.0 | 1.0 | 3.6 | 0.89 | **yes** |
| `SG_SHUF` | 1.0 | 1.0 | 3.6 | 0.81 | yes |

Under uniform exposure, `C` was identical across all KV heads for a given
absolute position. Under gating it is **head-specific and content-dependent**
for the first time. About 77% of entries never draw above-mean attention at any
event, so the gate is genuinely selective. H2 is now a real question.

## The answer is negative

| Arm | Exact | early | middle | late | dist | within B0-correct |
|---|---:|---:|---:|---:|---:|---:|
| `B0` full cache | **30/64** | 3 | 6 | 10 | 11 | — |
| `R192` positional-heavy | 25/64 | 4 | 2 | 9 | 10 | 22/30 |
| `B4_POS` | 22/64 | 0 | 0 | 11 | 11 | 20/30 |
| `CMEAN` cohort-relative | 22/64 | 3 | 4 | 10 | 5 | 18/30 |
| `B3` uniform exposure | 19/64 | 1 | 3 | 10 | 5 | 19/30 |
| `SG20_E1` gated | 17/64 | 2 | 2 | 10 | 3 | 17/30 |
| `SG10_E1` gated | 16/64 | 2 | 1 | 10 | 3 | 16/30 |
| `SG_SHUF` shuffled gate | 16/64 | 2 | 1 | 10 | 3 | 16/30 |
| `SG10_E05` gated | 15/64 | 2 | 1 | 10 | 2 | 15/30 |

**The specificity gate failed maximally.**

```text
SG10_E1 vs SG_SHUF   case 0W/0L p=1.0000   cluster 0better/0worse/16tied  p=1.0000
```

Permuting the gate among eligible entries — preserving the per-row gate
multiset exactly while destroying entry identity — produced **the same
correctness on all 64 cases**.

This is not an inactive-feature artifact. The intervention demonstrably changes
behaviour: `SG10_E1` and `SG_SHUF` give different predictions on 13 of 64
cases, and `SG10_E1` differs from uniform-exposure `B3` on 26 of 64. The gate
moves the cache around. It just moves it no more usefully than a random
permutation of itself does.

Gated exposure was also not an improvement on the thing it repairs:

```text
SG10_E1 vs B3     case 1W/4L p=0.3750   cluster 1b/2w/13t p=1.0000
SG10_E1 vs R192   case 2W/11L p=0.0225  cluster 1b/6w/9t  p=0.1250
B0      vs SG10_E1 case 14W/0L p=0.0001 cluster 9b/0w/7t  p=0.0039
```

Neither gate strength (`tau` 1.0 vs 2.0) nor exponent (`e` 0.5 vs 1.0) changed
anything at cluster level.

## What this settles

H2 moves from **not testable** to **not supported**.

That is the substantive outcome. Before P1, "realized exposure contains
information beyond age" could not be evaluated, because the implementation made
exposure equal to age. Now exposure is genuinely non-uniform, head-specific,
and content-derived — and the entry-identity component of it carries no usable
retention signal, because a shuffled gate matches it exactly.

The narrow research object that the [literature audit](LITERATURE_NOVELTY_AUDIT.md)
identified as the lab's only potential distinction was:

> explicit realized scoring-exposure state `C`, carried with each surviving
> semantic cache entry through repeated inter-chunk gather and eviction

Two implementations of that state have now been tested. The uniform one is age
in disguise. The salience-gated one is genuinely non-uniform and fails its
specificity control. Neither supports the claim.

## Limits

- One core, one suite, one chunk schedule, one budget family.
- One gate family. `contribution > tau * mean` is a natural first choice but not
  the only way to make exposure non-uniform; a continuous, rank-based, or
  cross-layer gate could behave differently.
- Exploratory and unregistered; no frozen endpoint, no advance gate, and no
  multiplicity control across arms.
- Scenario packs are correlated; all claims use the 16-pack unit.
- `record_attention` was monkeypatched in-process. No file under `src/` was
  modified, so the Experiment F content-hash provenance is intact.
- The `C == 0` tolerance in this screen's compression differs deliberately from
  the shipped guard, which rejects `C <= 0` to prevent a missing-significance
  run degrading into a positional policy. Under gating, `C == 0` is a
  legitimate "never salient" state.

## Execution note

The first attempt died at 249/256 rows with no Python traceback, during the
final control arm, on a machine with about 6 GB free. It was discarded rather
than spliced: ledger row `N-034` records that separate CPU/bf16 processes are
not bitwise reproducible here, and `SG10_E1` versus `SG_SHUF` is precisely the
contrast that cross-process drift would corrupt. The recorded run completed all
256 rows in one process. The aborted attempt never opened the answer key.

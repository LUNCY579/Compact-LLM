# Results (P2)

> **WITHDRAWN.** The contrasts below are retained as the record of what P2
> measured, but they do not support the conclusion P2 drew from them. Every arm
> was already correct on every scored item, the fast memory's retention horizon
> (87.5 tokens) was two orders of magnitude shorter than the context, and one
> item per cell could not have reached significance. See
> `reports/MECHANISM_DIAGNOSTIC.md` and `reports/FINAL_INTERPRETATION.md`.
> P2b re-runs the matched arms; its results will be added below.

Classification: `EXPLORATORY_NOT_CONFIRMATORY`.

Matched arms A/B/C/D. Seeds **11 and 22** completed. Seed **33** was not finished
(CUDA OOM after six sequential train+eval loads on the 8 GB laptop GPU).
Arm A was evaluated once after the GPU was free.

Position suite: 14 unsealed synthetic items (7 positions × lengths 256 and 512).
Interference battery: 7 items (A–G).

## Aggregate exact match

| Arm | Seeds | Position exact | Interference exact |
|---|---|---|---|
| A | — | 13/14 = 0.929 | 6/7 = 0.857 |
| B | 11, 22 | 14/14 = 1.000 | 6/7 = 0.857 |
| C | 11, 22 | 14/14 = 1.000 | 6/7 = 0.857 |
| D | 11, 22 | 14/14 = 1.000 | 6/7 = 0.857 |

Arm A's only miss is `POS-512-0.95-0013`: gold `slate`, prediction `color slate`
(malformed prefix, not a wrong colour). B/C/D emit the bare colour. That is not
a middle-context retrieval effect.

## Contrasts (preregistered)

| Contrast | Result |
|---|---|
| C − A | +1/14 exact, entirely the malformed-prefix item. Not a memory gain. |
| C − B | **0**. Identical predictions on every scored item. |
| D − C | **0**. Identical predictions on every scored item. |

## Position-conditioned

B/C/D: exact = 1.0 at every listed position and both lengths. Middle−edge gap = 0.
A: 1.0 everywhere except the 0.95 / 512 malformed row.

Lengths 2K / 4K / 8K / 16K were **not** evaluated (suite items were 256/512 so
the needle stayed inside the window; longer eval was skipped after training
already filled the card).

## Interference

All four arms, both seeds:

- A, B, C, E, F: retained (correct)
- G: late correction overwritten correctly (`teal`)
- D (contradictory update): **wrong** on every arm

No arm-specific overwrite or blend pattern appeared.

## Training

Each completed B/C/D run: 24 steps, ~0.08–0.10 GPU-hours, peak train VRAM
~3.0 GB. P1 Arm C: 0.093 GPU-hours, 2.34 GB.

Total completed training GPU-hours ≈ **0.64** (P1 + six P2 runs). Eval/profile
is extra wall time, not counted in that sum.


---

# Results (P2b) — complete

Full tables and figures: `reports/RESULTS_P2B.md`, `reports/figures/`.
Raw: `artifacts/supa_a1_1_fast_weight_attention_v1/p2b_longctx/`.

Suite: two-hop `vault -> unit -> code`, 24 confusable decoys, token-exact
1K/2K/4K, 8 items per (length, position) cell, 552-code single-token answer
space, candidate-argmax scoring. 168 position items, 56 interference items.
Seeds 11 and 22. Layer 14, rank 16. 4.20 training GPU-hours.

## Totals

| Arm | position | interference |
|---|---:|---:|
| A frozen backbone | 138/168 (0.821) | 55/56 |
| B static control, s11 / s22 | 159 / 159 (0.946) | 55 / 51 |
| C recurrent, s11 / s22 | 160 / 161 (0.952 / 0.958) | 50 / 46 |
| D significance-gated, s11 / s22 | 161 / 160 (0.958 / 0.952) | 55 / 51 |
| C-short (original horizon), s11 | **163 (0.970)** | 55 |

## Preregistered contrasts, within seed, paired exact McNemar

| Contrast | seed 11 | seed 22 |
|---|---|---|
| C vs A, position | 25/3, p < 0.0001 | 26/3, p < 0.0001 |
| **C vs B, position** | 2/1, p = 1.00 | 3/1, p = 0.63 |
| **D vs C, position** | 3/2, p = 1.00 | 2/3, p = 1.00 |
| C vs B, interference | 0/5 against C | 0/5 against C |
| C vs A, interference | 0/5, p = 0.0625 | 0/9, **p = 0.0039** |
| D vs C, interference | 5/0 for D | 5/0 for D |
| D vs B, interference | 0/0 identical | 1/1 identical |
| C-short vs C, position | 5/2, p = 0.45 | not run |
| C-short vs B, position | 4/0, p = 0.125 | not run |

## What replicated

1. **The whole gain over the frozen backbone comes from added parameters, not
   memory.** Arm B — static, no state carry — takes 138 -> 159, including the
   primacy repair (position 0.05: 0.417 -> 0.917). Arm C adds 1-2 further items.
2. **H2 fails at both seeds.** C vs B is 2/1 and 3/1 discordant out of 168.
3. **H4 fails on position at both seeds** (3/2 then 2/3, opposite directions).
4. **Recurrence carries a replicated interference cost**: C loses 5 items to B at
   both seeds with zero gains in 10 discordant pairs, and is worse than the
   *untrained* backbone (0/9, p = 0.0039 at seed 22).
5. **Arm D is identical to arm B on interference** at both seeds (0/0, 1/1). The
   significance gate's entire demonstrated value is cancelling a cost that the
   recurrence introduced.

## The horizon result, which inverts the P2b rationale

Arm C-short keeps the ORIGINAL 87.5-token horizon that Amendment 1 argued was too
short. It is the **best arm in the run**: 163/168, the only arm with zero losses
against the baseline (25/0), no interference degradation (55/56), and ahead of
long-horizon C on both dimensions (5/2 position, 5/0 interference).

The interference cost tracks the **horizon**, not recurrence. Within each seed,
category F is: B 8/8 and 4/8, C 4/8 and 2/8, C-short 8/8. A longer-horizon state
accumulates more and the accumulation interferes. **The horizon change made in
Amendment 1 caused the only harm attributable to the fast memory.** Single seed
for C-short, 8 items per category: suggestive, not established.

## Decision: NO-GO

Against Amendment 1's rule. GO required C > B at p < 0.05 in both seeds (got
p = 1.00 and 0.63) or D > C at p < 0.05 in both seeds (got p = 1.00 and 1.00).

WEAK GO was considered and rejected: C is nominally ahead of B at both seeds, but
by +1 and +2 items out of 168, against a replicated -5 interference cost and a
measured 1.5-4.2x prefill slowdown with no VRAM saving. A directional edge that
small, bought at that price, is not a weak go.

REDESIGN was considered and rejected: it requires a working memory that
interference is spoiling. There is no positional benefit here to rescue.

## Limits on this NO-GO, stated with it

- **Compressed range** (Amendment 6): the suite was calibrated so the *frozen*
  arm was off ceiling, but the headline contrast is C vs B and B trains to
  159/168, leaving 9 items of headroom. This is a null in a narrow window.
- **One layer.** Everything is at layer 14. The brief's layer-location pilot was
  never run. This is a NO-GO *at layer 14*.
- **Interference battery** is ceilinged for gains and seed-noisy (Amendment 9);
  only the within-seed paired cost is reportable, and at p = 0.0625 per seed.
- **Two seeds**, not three.
- The task is synthetic vault/unit/code retrieval. Nothing here transfers
  automatically to natural long-context work.

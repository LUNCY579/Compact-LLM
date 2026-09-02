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

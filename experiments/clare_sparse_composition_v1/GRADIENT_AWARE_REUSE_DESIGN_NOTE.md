# Design note (NOT part of CLARE-RINEYA-001): gradient-aware coordinate reuse

**Status: design only. No code in `clare/` implements it, and no 001 arm uses it.** It is recorded
now so that the 001 result can decide whether it is worth registering as a separate study.

## Motivation

Strict CLARE disjointness forbids every reuse of a claimed coordinate. For R/U/S/P that may be the
wrong prior: the four skills share one answer format (`{"answer": [...]}`) and plausibly shared
intermediate computation (item indexing, value bookkeeping). Arm F (shared + private) reserves a
shared support up front; gradient-aware reuse would instead decide reuse *per coordinate, per skill*,
from whether the new skill wants to move a claimed coordinate in the same direction as its owner.

## Rule sketch

At the start of skill t, with the frozen accumulated delta installed, estimate per-coordinate
gradients of the new skill's loss, `g_t`, and of each earlier skill's loss, `g_s` (a few TRAIN
batches each, no update), on the coordinates claimed by `s`:

```
reuse(j)   if  <g_t, g_s>_j > 0   (aligned: the new skill would push j the way s already did)
private(j) if  <g_t, g_s>_j < 0   (conflict: keep j frozen for s; t must use free coordinates)
```

Per-coordinate products are noisy, so the decision would use a smoothed statistic, e.g. the sign
agreement of `g_t` and `g_s` over K batches, and a margin (reuse only if agreement ≥ a registered
threshold). Reused coordinates are trainable by t but stay attributed to s; t's delta on them is its
own additive increment, so the per-skill deltas remain separable for the 2×2 interaction measurement.

## Why it is excluded from 001

1. It changes two mechanisms at once (allocation AND sharing); 001 must first isolate learned vs
   random allocation at identical capacity.
2. Reuse breaks exact capacity matching with arm D, the crucial control.
3. Gradient alignment is itself a GEOMETRIC quantity; like cosine, it does not by itself establish
   functional compatibility. It would need its own functional endpoint.

## What 001 must show first for this to be worth registering

- If E ≈ D and neither beats A: separation itself is not the lever, and reuse rules are unlikely
  to rescue composition. Low priority.
- If E or D beats A on retention but loses composition relative to A or B: strict disjointness is
  too strict, which is exactly the case reuse targets. Register it, with F as the intermediate
  control.
- Always measure the per-skill ‖Δ‖ first. The parent line showed that structured removal at
  matched norm generally loses to a uniform scalar on a saturating dose curve.

## Minimal future registration outline

Arms: E (strict), F (shared + private), G (gradient-aware reuse), all at identical *total* capacity.
Endpoints: identical to 001 §7. Additional diagnostic: per-coordinate agreement histograms by layer,
reported as geometry.

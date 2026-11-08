# Validation pass 1 — adversarial audit and the defects it found

An adversarial read-only audit was run over the lab before committing: six independent
auditors (claims-vs-artifacts for each report, the hooks equivalence invariant, the
attribution math, silent failure modes, and SPEC drift) plus a synthesis pass.

This document records what the **significance audit** found, all of which is fixed and
pinned by tests. The other five auditors were interrupted by a usage limit and have been
re-run; their findings will land in `validation_pass_2.md`.

None of these defects affected a published number. Phases 1–3 do not touch
`significance/`. All five would have corrupted Phase 4 onward, and the first would have
corrupted the study's headline method.

## 1. Mode 1 pointed the wrong way — `invalidates_result` if it had run

`SignificanceMatrix.to_logit_bias_spec` scattered the **raw** Deltas into the intervention
table, while its sibling `to_value_scale_spec` used `importance()`. On a loss objective
`Delta = L_normal − L_perturbed` is *negative* at an important cell, so the two modes
pointed in opposite directions.

Reproduced on a one-layer, one-head, four-bin matrix with bin 3 as the important cell:

```
raw values       [ 0.5   0.    0.   -1.0 ]
importance()     [-0.5  -0.   -0.    1.0 ]      <- bin 3 correctly ranked first
to_logit_bias    [ 0.5   0.    0.   -1.0 ]      <- bin 3 gets -1.0: SUPPRESSED
to_value_scale   [ 0.75  1.    1.    1.5  ]     <- bin 3 gets 1.5:  BOOSTED
```

Mode 1 is the brief's §24 *primary* intervention candidate. Had this run, PSA would have
suppressed attention to exactly the bins the significance matrix identified as mattering
most, and `sign_flip()` — a **negative control** — would have been the correctly-signed
intervention. The likely outcome is a confident, well-controlled, entirely wrong negative
result for the study's headline method.

Nothing in the old test suite checked direction, only shape. Fixed by routing Mode 1
through `importance()` exactly as Mode 2 does, plus two regression tests that pin the
direction against `top_cells` for all three objectives and assert the two modes agree in
sign on every bin.

## 2. The coarse-to-fine ladder recorded group deltas as cell deltas

`mode="ladder"` passed `group_size` straight to the cell runner, which assigns a group's
**joint** delta to every member cell, marks them all evaluated, and reports
`coverage() == 1.0` with no caveat. On a tiny two-layer model, cell `(0, 1, 2)` had a
true delta of `+4.43e-3` recorded as `−7.22e-3` — wrong magnitude and wrong **sign**.

Worse, `cost_estimate` divided the ladder's pass count by `group_size`, so the SPEC §11.3
budget gate advertised the broken path as the affordable one: the same 28-layer plan
reported **9.52 GPU-hours** at `group_size=1` and **1.21 GPU-hours** at `group_size=8`,
`exceeds_budget` flipping from True to False.

`mode="grouped"` was already correct — it screens groups, then re-runs survivors one cell
at a time, and records the caveat. Only the ladder was wrong. Fixed by forcing one cell
per forward in the ladder, removing the `group_size` division from its cost arm, and
raising a pointed `ValueError` if a caller combines `mode="ladder"` with `group_size > 1`
rather than silently ignoring it.

## 3. The advertised speedup was inconsistent with its own cost estimate

For `tier="attribution"`, `speedup_vs_brute_force` was a ratio of *passes* while
`estimated_seconds` folded in the backward multiplier, so the two disagreed by exactly
4×: **1793× reported against an actual 448×**. `n_forwards` also reported `256` for a plan
that is 256 forwards *plus* 256 backwards.

Fixed: the speedup is now a ratio of seconds and matches `estimated_gpu_hours` exactly
(verified: 36.72 / 0.0819 = 448.25 both ways), and `n_backwards` is reported alongside
`n_forwards`.

## 4. Explained variance and reconstruction error used different denominators

`cumulative_explained_variance` was computed on the **centred** matrix while
`relative_reconstruction_error` divided by the norm of the **uncentred** matrix. The two
appear side by side in the same JSON block. On a 60×8 matrix with a strong shared
profile: rank-1 explained variance `0.2116` next to a rank-1 relative error of `0.0122` —
a **73× discrepancy**, and precisely the mean-profile inflation that centring exists to
remove.

Fixed by normalising the residual on the centred matrix, so
`reconstruction_error(k) == sqrt(1 − cumulative_at(k))` now holds exactly at every rank
(verified at k = 1, 2, 4).

This one matters for a headline the brief specifically asks for: "are the significance
matrices low-rank?" The two numbers disagreeing by 73× is the difference between "rank 1
explains 21% of the variation" and "rank 1 reconstructs it to 1%".

## 5. The Tier-1 acceptance gate had an anticonservative interval

`validate_attribution_against_ablation` bootstrapped its Spearman CI by resampling
**cells**, whose docstring called them "the independent units of this comparison". They
are not: every attribution value in a layer is a contraction of that layer's `v_proj`
gradients, so residuals share a layer-level component by construction. The CI was
therefore too narrow, and the `accept_rho` gate on `ci_low` — the gate that decides
whether the cheap estimator may be used at all — was looser than advertised.

Fixed by clustering the resample on layer, the same discipline SPEC §7 applies to
`scenario_id`, and recording `ci_clustered_on` and `n_layers` in the output. The function
now also refuses to run when the two estimators were measured at different `eps`, since
its Bland-Altman block reports their difference "in objective units".

## What this pass says about the codebase

Five real defects in one module, none of which any of the 585 existing tests caught,
because all five produce **plausible numbers rather than exceptions**. Four of the five
are sign or normalisation errors — the class of bug that survives code review, passes
shape assertions, and inverts a conclusion.

The one structural lesson worth carrying: `to_value_scale_spec` had a docstring
explicitly warning about the sign trap ("`importance` and not `values`: the gate must
*boost* the cells that matter, and for a loss objective the raw Delta is negative
there") — and `to_logit_bias_spec`, twenty lines above it, did the thing the docstring
warned against. A comment describing an invariant is not an enforcement of it. All five
fixes ship with a test.

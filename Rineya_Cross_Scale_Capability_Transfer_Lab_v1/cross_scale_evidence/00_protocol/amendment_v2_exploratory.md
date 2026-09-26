# Amendment v2 — Exploratory H5/H6 Follow-up, Then Better T

Protocol version: `2.0.0-exploratory`

Parent: `preregistration.md` version `1.0.0` (immutable)

Frozen: 2026-08-30, after v1 DEV-FULL no-go and before any v2 candidate construction.

Research branch: `research/rineya-cross-scale-qwen4b-8b`

Base commit of this amendment: `8942d4d8936dff0e0fdbd6732bf22e12c70b92d2`

## Visible evidence at freeze

v1 already observed:

- P1/P2/P3 projections failed the functional gate.
- Only `P4-D2-W4-rank8-strength0.10` produced coherent 0.6B-shaped donors.
- Δ4 and Δ8 agree strongly (cosine 0.932; sign agreement 0.926).
- Mean consensus α=0.15 improved MERGEDEV 48/59 vs Supa 47/59 and tied DEV-FULL 144/180.
- TIES, DARE, DARE-TIES, Fisher, and layer/module injection were implemented but not run on real checkpoints.
- The fresh sealed holdout was never accessed.

Because DEV-FULL was used to evaluate the v1 mean-consensus candidate, it is no longer a clean confirmatory set for that candidate. This amendment reclassifies DEV-FULL as an **intermediate development set** for new methods. The unused sealed bank remains the only confirmatory set.

All v2 outcome runs are **exploratory** under the parent amendment policy. They cannot upgrade the v1 confirmatory negative finding unless a new method later wins a frozen sealed run.

## Scientific objective

Test whether the skipped H5/H6 operators can extract a robust gain from the existing P4-s010 deltas. If they cannot, invest in one better projection operator T.

## Frozen two-control selection

A v2 candidate is eligible only if it:

1. passes the parent preservation gates versus ungated Supa (10-point family drop, 2% invalid, 5% completion/format drop);
2. strictly beats ungated Supa on overall semantic score;
3. strictly beats v1 `CONS_MEAN_A1500` on overall semantic score;
4. then maximizes worst-family, then scenario macro, then prefers the simpler method.

DEV-FULL promotion uses the same two-control rule. Sealed access requires a frozen finalist packet and is unauthorized until that packet exists.

## Frozen grids

Do not rerun the v1 coarse α grids. Use only:

- α ∈ `{0.10, 0.15, 0.20}`
- consensus: `sign_agree`; `tensor_cosine` at τ ∈ `{0.3, 0.5, 0.7}`
- TIES keep ∈ `{0.20, 0.50, 0.80}` on `{Δ8}`, `{Δ4,Δ8}`, `{Δ_Supa,Δ4,Δ8}`
- DARE p ∈ `{0.10, 0.20, 0.30}`, seeds `{17, 23, 42}`
- DARE-TIES: same p/seeds, keep-fraction `0.50` only, on `{Δ4,Δ8}`
- one-layer: α `0.15`, donors `{consensus, 8B}`, layers `0..27`
- groups: early `0–8`, middle `9–18`, late `19–27`
- group coefficients ∈ `{0, 0.10, 0.15}`
- Fisher λ ∈ `{0.25, 1.0}` if compute permits; otherwise `not_run_compute`

Successive halving: first wave at α=0.15 on MERGEDEV-fast → expand α only for methods that are not preservation failures → MERGEDEV → at most 3 DEV-FULL.

MERGEDEV-fast composition remains the frozen v1 subset. Do not rebuild it.

## Localization GO

Proceed to one-layer/group/module maps only if at least one H3/H6 candidate is two-control eligible on MERGEDEV.

## Better-T GO

If Localization GO fails, implement and test:

- D4 linear CKA monotonic depth alignment on calibration prompts only;
- one calibration-only residual direction-transfer SVD.

These are exploratory and not training-free confirmatory evidence. Re-enter the H3/H6 funnel for at most one new T that passes the functional/EOS gate.

## Leakage

Invisible-Limit TEST2, Stage 6 outputs, and the v2/v1 sealed holdout remain denylisted for development. Calibration uses prompts only.

## Claim boundary

v2 may report development-selected exploratory comparisons. It may not claim sealed confirmation, 8B-equivalence, or that a MERGEDEV-only gain is a robust Supa improvement.

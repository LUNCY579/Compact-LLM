# Phase 2 Completion Notes

## Review finding

The prior Phase 2 folder was only a preliminary coarse static scan. It could train single-layer adapters and optionally retrain the all-layer reference, but it did not implement the fine scan, pair/top-four/random/bottom controls, contribution metric, repeated seeds, importance-signal report, or Stop Gate 2.

## Added

- Complete coarse-to-fine static study.
- Best-single, adjacent/complementary pair, TOP4, RANDOM4, and BOTTOM4 variants.
- Project contribution metric using the saved Phase 1 base and all-layer summaries.
- Training VRAM/runtime comparison and category-floor checks.
- Three-seed stability stage.
- Automated Stop Gate 2 report requiring human approval.
- Optional dynamic IST-style experiment plan with a clear evidence boundary.

## Preserved

- Existing Phase 0 and Phase 1 files.
- Legacy root wrappers and the earlier coarse scan.
- Exclusion of generated `artifacts/` and checkpoints from this ZIP.

# Implementation changelog — Phase 2B progressive reduction

## Reason for the change

The existing 50-case heuristic reached 1.0 before Phase 4, so model soup could not show additional contrast. The original Phase 2A also found no small selective-layer adapter that matched the all-layer reference. The repository now preserves that negative result and adds a separate progressive-removal branch instead of repeatedly searching for one nonexistent dominant training layer.

## New behavior

- Tests every current decoder layer as a physical one-layer deletion.
- Generates and stores 50 outputs for each branch.
- Adds teacher-forced target NLL to detect changes hidden by a saturated generation score.
- Recovery-tunes the strongest deletion branches.
- Progressively removes more layers while the weight-memory reduction percentage remains greater than the heuristic-score-loss percentage and hard floors pass.
- Preserves full-quality, balanced, and minimum-memory profiles.
- Allows Phase 3 and Phase 4 to continue from a materialized reduced architecture without breaking adapter compatibility.
- Keeps Phase 4B as post-alignment structural revalidation/refinement.

## Evidence boundary

The new 50-case probe is development-only and intended for trend discovery. It does not replace sealed evaluation, safety review, or final runtime memory measurement.

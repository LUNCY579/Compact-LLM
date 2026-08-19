# Changelog

## v1.0 — 2026-08-18

- Generated 48,000 unique original synthetic problems across 30 balanced categories.
- Produced 48,000 full-context records and 189,922 compact-state transition records.
- Assigned TRAIN/DEV/TEST before transition expansion using an exact 90/5/5 split per category.
- Globally shuffled transition files and removed adjacent rows sharing a problem ID.
- Added direct next-step and optional stateful-rollout message views.
- Added schemas, deterministic regeneration scripts, problem-count-matched epoch views, paired examples, hashes, and full validation.

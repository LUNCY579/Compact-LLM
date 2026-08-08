# Phase 9B memory-reconstruction data

## v1.7.5 canonical reconstruction corpus

The default serious-run corpus is `reconstruction_corpus_v2_large/`. It contains 1,500 explicit memories and 3,600 canonical current-state reconstruction rows across 300 topic families, with 2,700 train / 456 dev / 444 sealed-test rows. Every memory carries source-state and validity-condition metadata; every canonical row supplies current state.

Decision coverage is 900 ACCEPT, 1,200 ADAPT, 600 REJECT, 300 CONFLICT, and 600 INSUFFICIENT. The corpus also includes 600 counterfactual state pairs, 10,800 curated preference pairs, and 14,400 evaluator candidates. Gold rows use `review_status: auto_verified` and are explicitly provenance-marked as deterministic schema-verified examples rather than human judgments.

The builder materializes three supervised views per canonical row: explicit reconstruction, answer-from-reconstruction, and direct-memory-to-answer distillation. Known-wrong replay/over-rejection/wrong-source variants remain preference/evaluator data and never enter ordinary SFT as targets.

## Memory schema

Each memory keeps reusable content together with its historical source state, validity conditions, provenance, timestamp, trust level, and memory type. The reconstruction input compares those historical conditions with the current state before producing ACCEPT / ADAPT / REJECT / CONFLICT / INSUFFICIENT guidance.

## Split and gate rule

Topic families are authored into train/dev/test. Strategy selection happens entirely on DEV. Exactly one reconstruction strategy is frozen before the sealed TEST split is evaluated.

`memories.sample.jsonl` and `reviewed.sample.jsonl` remain tiny smoke examples only; serious training uses `reconstruction_corpus_v2_large/`.

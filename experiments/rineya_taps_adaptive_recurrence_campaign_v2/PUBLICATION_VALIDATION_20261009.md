# Publication CPU validation

The original combined publication run is retained without relabeling:
**647 passed,4 failed in196.12seconds**. It used canonical Python, bytecode/cache
disabled, a campaign-local temporary directory and both top-level test trees.
Its JUnit and complete log are in `reviews/publication_check_20261009_001/`.

Independent review classified the failures as test-context/lifecycle assumptions,
not demonstrated executable-source defects:

| Original failure | Scoped follow-up | Outcome |
|---|---|---|
| Fresh bootstrap parent-path fixture |Same node with V3-local basetemp|1passed|
| Probe reader clean-import check |Same node in a fresh Python process|1passed|
| Fresh-bank clean-import/pre-generation test |Same node in a fresh process|Stillfails its later assertion that the already-generated corpus is absent|
| Passive study originally unused |Read-only actual scope/spent-study witness|Scopevalid; reused completed study correctly refuses|

Follow-up logs/commands/JUnit are in
`quality_pattern_v3/reviews/publication_isolated_20261009_001/`.
The fresh-bank original node is **not** reported as passing. Its pre-generation
assumption is stale after the legitimate fixed432-case corpus export.

A separate clean-process witness confirms no Torch/Transformers import from the
bank/verifier, preserves the actual corpus hash before/after, accepts the current
fresh-V2 consumer and confirms the completed passive attempt remains spent. The
witness neither loads a model nor regenerates data. Its first invocation had a
nested-script import-path error; that log is retained, and the explicit repository
path correction passed. No frozen test/source bytes were edited to obtain these
results.

Independent interpretation is recorded in `publication_validation_review_20261009.json/.md`.
These are apparatus/publication checks; they do not perform native qualification
or measure the pending432-case model evaluation. The publication does not claim
that the entire mixed test invocation is green.

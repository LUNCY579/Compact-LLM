# Advanced Merging and Localization Outcome Gate

## Implementation status

- TIES: implemented with exact deterministic per-tensor magnitude retention,
  sign election, disjoint agreement mean, conflict statistics, FP32 arithmetic,
  and candidate manifests.
- DARE: implemented with SHA-256-derived per-tensor seeds, Bernoulli masks,
  `1/(1-p)` rescaling, actual keep-rate reporting, and manifests.
- DARE-TIES: implemented as independently seeded DARE deltas followed by TIES.
- Fisher: implemented diagonal empirical-Fisher estimation from calibration-only
  prompt-token NLL, clipping/protected denominators, and weighted merging.
- Layer and module injection: implemented for arbitrary layer sets and module
  families (`attention`, Q/K/V/O, `mlp`, gate/up/down, norms, embeddings/head).

Pure and end-to-end synthetic tests cover deterministic masks, sign resolution,
Fisher fallback, selectors, streaming serialization, and state-dict outputs.

## Outcome-experiment status

**Not run on real checkpoints after DEV-FULL.**

The preregistered localization/advanced funnel required a positive signal on a
larger development subset. The selected arithmetic consensus tied Supa at
144/180 on DEV-FULL, traded one fix for one break, and reduced scenario-macro
score. Running TIES/DARE/Fisher or choosing layers from later benchmarks would
expand an already failed method family and increase model-selection bias.

Accordingly:

- no TIES trim fraction was selected from outcomes;
- no DARE seed was sampled for a reported lucky result;
- no Fisher checkpoint consumed ~2.4 GB per model without promotion evidence;
- no one-layer/module map was optimized against DEV-FULL or sealed data;
- sealed evaluation remains untouched.

The tooling is ready for a separately preregistered exploratory follow-up, not
evidence that these methods improved Supa in this study.

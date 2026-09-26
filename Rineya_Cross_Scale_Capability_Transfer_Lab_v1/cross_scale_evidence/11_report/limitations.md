# Limitations

- The procedural banks cover all preregistered families but are smaller and
  more templated than a broad public benchmark battery.
- MERGEDEV and DEV-FULL are development evidence; no sealed result was run.
- Full donor inference baselines were not run on CPU because 4B/8B generation
  cost is not needed to validate the 0.6B construction claim.
- P1 tests channel/head subsets, not an exhaustive neuron permutation search.
- Ridge/Procrustes alignment uses shared-vocabulary embeddings, not hidden-state
  activation alignment on an external corpus.
- Spectral transfer carries donor singular-value profiles while retaining
  target singular vectors; it does not transfer donor vector directions.
- The strongest candidate changed behavior on only two DEV-FULL cases, leaving
  effect estimates highly uncertain.
- TIES, DARE, DARE-TIES, and Fisher are not authorized for outcome experiments
  after the DEV-FULL no-go unless treated as a new exploratory method family.

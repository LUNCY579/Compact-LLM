# TGSA Phase 1 preregistration

**Registered:** 2026-08-28  
**Parent branch:** `research/trajectory-guided-sequential-attention`  
**Pinned parent commit:** `aefb70ebed2e7f57b9cbfe25c24dfdc0aee2fd4e`  
**Configuration:** `configs/phase1_model_identifiability.json`

## Research question

Does the learned finite-difference signal add held-out capability beyond the sequential cascade itself and beyond an information-matched learned history pathway?

## Hypotheses

- **H1a:** `learned_delta` exceeds `history_control` on the macro-average of held-out pointer-chasing and state-update strata.
- **H1b:** `learned_delta` exceeds `cascade` on the same estimand.
- **H1c:** aligned `learned_delta` exceeds the same trained model when trajectory tensors are batch-rolled across examples.
- **Guardrail:** learned delta is non-inferior to history control on direct lookup.

Failure of H1a prevents a claim that finite differences themselves add capability. Passing only H1b permits, at most, the interpretation that an extra learned recurrent pathway helps. Failure of H1c prevents a claim that example-aligned trajectory content is causal.

## Fixed design

- Five paired seeds: 101, 202, 303, 404, 505.
- Three required arms: cascade, history control, learned delta.
- The exact same materialized examples, minibatch order, update count, optimizer, model width/depth, and initial shared tensors are used within each seed.
- History and learned-delta state dictionaries and parameter counts must be exactly identical at initialization.
- Dropout is zero.
- Training stops after exactly 2,500 updates. No dev-based checkpoint selection or early stopping is permitted.
- The fixed final checkpoint is evaluated once by the registered runner.
- Test predictions are retained at the instance level.

## Data

The dataset is generated deterministically from seed 731928 and materialized before training. Cross-split latent signatures are disallowed. The full registered counts are 5,000 train, 900 dev, and 1,800 test examples, balanced within dev/test across direct lookup, pointer chasing, and state update.

Test pointer chains, state sequences, and context sizes are shifted upward from training. The test split is held out from optimization. This split is controlled and generated, not independently authored or sealed; that limitation is part of every claim.

## Primary estimand

For each paired seed and each registered non-guardrail stratum:

1. compute the paired per-example correctness difference;
2. average within stratum;
3. macro-average strata within seed;
4. average the five paired seed effects.

The confidence interval is a hierarchical paired bootstrap that resamples training seeds and then resamples examples within each stratum of each selected seed. The bootstrap uses 10,000 draws, 95% confidence, and seed 91273.

Answer-NLL improvement is reported as a supportive endpoint, not a replacement for accuracy.

## Fixed gates

- delta minus history: observed accuracy delta at least +0.02 and lower 95% bound above 0;
- delta minus cascade: observed delta at least +0.02 and lower bound above 0;
- aligned minus batch-rolled delta: observed delta at least +0.015 and lower bound above 0;
- direct lookup delta minus history: lower bound at least -0.01;
- all arm/seed runs, initialization proofs, hashes, and lineage checks valid.

No threshold may be changed after inspecting confirmatory test outcomes under the same study ID.

## Diagnostic interventions

- `roll_batch`: primary alignment falsifier; preserves tensor distribution and sequence position while breaking example identity;
- `zero`: tests dependence on the learned control path;
- `negate`: tests directional sensitivity.

Only batch-roll enters the advancement decision. Zero and negate remain descriptive because they introduce out-of-distribution control values.

## Exclusions and failures

There are no post-hoc example exclusions. A corrupted, missing, duplicated, or hash-mismatched artifact invalidates the affected run. A numerical failure is reported with its failure bundle; it is not silently replaced. A replacement run requires a new registered seed or a documented rerun with identical code/config and preserved failed artifacts.

A partial matrix may be analyzed only with `--allow-partial`; its decision must remain `smoke_only` and cannot support advancement.

## Claim ceiling

Even if every gate passes, the strongest supported statement is:

> In a controlled tiny decoder trained on the registered synthetic iterative tasks, an aligned finite-difference representation improved held-out capability beyond an information-matched history representation and the sequential cascade alone.

The study does not establish natural-language generalization, long-context performance, Qwen/GQA compatibility, wall-clock efficiency, memory efficiency, scaling behavior, novelty, or an arXiv-primary claim.

## Change control

Any change to task generation, counts, arms, seeds, architecture, optimizer, training steps, interventions, estimand, bootstrap, or thresholds after confirmatory outcomes are visible requires a new study ID, new output directory, and amended preregistration. The original artifacts remain immutable.

# Research Protocol

## Primary question

Does layer-wise interpolation between a preserved Anchor and a stronger Greedy checkpoint improve Rineya’s short-response quality beyond either endpoint and beyond the best global interpolation?

## Required controls

At minimum, preserve evidence for:

1. Anchor endpoint (`α=0`)
2. Greedy endpoint (`α=1`)
3. Uniform average (`α=0.5`)
4. Best global interpolation
5. Layer-wise seed before coordinate search
6. Final layer-wise candidate

Without these controls, a final gain cannot be attributed to layer-wise coefficients rather than ordinary interpolation.

## Development and sealed test

All coefficients are selected on development data. The pipeline freezes:

- candidate ID;
- coefficient recipe;
- source hashes;
- evaluator policy hash;
- development data hash;
- objective definition;
- paired comparisons.

Only then can the sealed test run. The access ledger is part of the evidence. A test result must not trigger a coefficient change.

## Global interpolation

Use a predeclared coarse grid, then a local refinement around the best coarse point. Record the entire curve, not only the winner. A smooth curve suggests compatible endpoints; a deep interior collapse suggests an error barrier or incompatible parameter geometry.

## Layer interventions

For each layer `l`:

- Forward swap: start from Anchor and replace only layer `l` with Greedy.
- Reverse swap: start from Greedy and replace only layer `l` with Anchor.
- Centered grid: start from the globally selected coefficient and vary only `α_l`.

The forward and reverse swaps answer whether the Greedy version of a layer has stable behavioral value in two contexts. The centered grid answers which coefficient works around the global solution. These questions are related but not identical.

## Per-response influence

The primary case-level utility combines:

- v4.5 quality score;
- complete-utterance status;
- observable-defect status;
- brevity appropriateness.

The exact weights are versioned in configuration. Per-response effects are stored before aggregation so domain, category, tail, and sign-consistency analyses remain possible.

## Noise treatment

The lab does not “correct” endpoint scores by hand. It uses:

- stable paired case IDs;
- paired bootstrap confidence intervals;
- probability of improvement;
- sign consistency across cases;
- shrinkage of weak layer estimates toward the global coefficient;
- a bounded evaluation budget;
- acceptance thresholds for coordinate updates.

## Replication plan

A strong finding should be repeated under at least:

- three random development subsets or seeds;
- a second short-response corpus;
- one broader capability-retention suite;
- no-thinking and thinking protocols reported separately;
- raw human review of cases where the layer-wise model changes the answer.

## Claims that the experiment can support

With adequate replication, the experiment may support statements such as:

- a global interpolation interior to the two endpoints outperformed both endpoints;
- layer-wise coefficients improved the frozen development objective beyond the global interpolation;
- behavioral intervention effects were concentrated in specific layers under this task distribution;
- some layers favored Anchor while others favored Greedy;
- coefficient profiles were or were not stable across datasets/protocols.

## Claims it cannot establish by itself

It cannot establish that:

- a layer uniquely caused a particular answer;
- one layer contains a fixed percentage of total reasoning;
- a coefficient is universally optimal outside the evaluated distribution;
- the “Is One Layer Enough?” RL result has been reproduced exactly;
- activation magnitude and causal importance are equivalent;
- test-set gains remain valid after repeated test-guided changes.

## Pre-registration packet

Before the first development run, archive the resolved configuration, dataset manifest, objective weights, candidate grid, search budget, selection strategy, seed, and evaluator-policy hashes. Any later modification starts a new run and must be labeled as a new experiment.

## Multiple-comparison caution

Layer grids and coordinate search evaluate many candidates. Paired bootstrap intervals describe uncertainty for a comparison but do not erase adaptive-search bias. Treat the development winner as selected evidence, confirm it once on the sealed split, and replicate on a fresh split or seed before making a strong mechanistic claim.

## Probability diagnostics

Teacher-forced token statistics can reveal whether the endpoint difference concentrates around target tokens, EOS calibration, or broader distribution changes. They are descriptive. The coarse JS value is a lower bound after collapsing unobserved mass, and must not be interpreted as exact full-vocabulary divergence.

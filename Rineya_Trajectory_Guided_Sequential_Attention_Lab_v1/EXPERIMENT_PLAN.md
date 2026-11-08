# Rigorous implementation and experiment pipeline

## Objective

Determine whether a learned finite-difference state trajectory across sequential attention heads contributes useful computation beyond:

1. ordinary parallel multi-head attention;
2. sequential/cascaded heads alone;
3. access to the previous two states without explicit differencing;
4. a raw linear finite difference;
5. generic additional depth or parameters.

The pipeline is staged so that every phase unlocks only claims supported by its evidence.

---

## Stage 0 — semantic reference kernel

### Goal

Establish a small, auditable operator with exact mechanism switches before touching a pretrained model.

### Deliverables

- shared carrier and finite-difference trajectory;
- `none`, `raw`, and `learned` trajectory modes;
- cascade, Q/K/V conditioning, predictor, correction, and importance switches;
- conventional parallel control;
- summary/full telemetry;
- causal, mask, shape, deterministic, finite-gradient, and ablation tests;
- machine-readable smoke manifest.

### Exit criteria

- all tests pass on CPU;
- all arms produce finite forward/backward values;
- future suffix perturbation does not alter prefix output;
- learned trajectory parameters receive nonzero gradients;
- telemetry can be disabled and does not contaminate timing.

### Current state

Implemented in this branch. This stage proves engineering viability only.

---

## Stage 1 — mechanistic tiny decoder

### Goal

Test whether the mechanism can learn useful sequential refinements under tightly controlled capacity and data.

### Model

Train a decoder-only model from scratch at a scale small enough for five seeds and full ablations. Use identical embeddings, FFNs, layer count, tokenizer, initialization family, optimizer, data order, and token budget across arms.

Recommended two tracks:

- **single-TGSA-layer probe:** replace one middle attention block;
- **all-layer architecture:** replace every attention block after the single-layer behavior is understood.

### Tasks

Use generated, leakage-free tasks whose mechanisms are directly measurable:

- associative recall;
- passkey retrieval at controlled positions;
- two- and three-hop entity composition;
- contradiction resolution;
- distractor suppression;
- state tracking with updates and stale facts;
- order-sensitive reasoning;
- middle-position evidence retrieval;
- answer revision where later evidence invalidates an early hypothesis.

Every generator must emit the ground-truth evidence chain, token positions, distractor classes, and difficulty parameters.

### Arms

| ID | Arm | Purpose |
|---|---|---|
| P | Parallel MHA/GQA | Conventional control |
| C | Cascade | Tests sequential head depth alone |
| H2 | Parameter-identical history-mean controller | Tests whether directional differencing is necessary |
| R | Raw delta | Tests weak/reparameterizable difference |
| L | Learned delta | Tests nonlinear trajectory encoding |
| Q | Learned delta + Q/K/V conditioning | Tests retrieval redirection |
| PC | Q + predictor–corrector | Tests state extrapolation/correction |
| F | PC + soft token significance | Full Phase-1 operator |
| RD | Recurrent-depth block | Matches sequential computation outside head structure |
| DL | Deeper standard Transformer | Matches effective depth/latency |

The current executable presets cover P, C, H2, R, L, Q, PC, and F. RD and DL must be added before a paper-quality mechanism claim.

### Fairness views

Report all of these rather than selecting the most favorable:

- native parameter count;
- trainable parameter matched;
- total parameter matched;
- training-token matched;
- optimizer-step matched;
- estimated training-FLOP matched;
- inference-FLOP matched;
- measured prefill-latency matched;
- measured decode-latency matched;
- peak-memory and KV-cache matched where feasible.

### Exit criteria

Proceed only if learned trajectory control beats cascade on a preregistered development endpoint across seeds and the gain is not erased by H2/RD/DL controls.

---

## Stage 2 — Qwen/Supa compatibility layer

### Goal

Preserve a native causal LLM’s RoPE, GQA, cache, and generation semantics while introducing one TGSA target layer.

### Work items

- inspect exact selected checkpoint revision and code;
- implement disabled-control native equivalence;
- implement cache-safe grouped sequential heads;
- add deterministic conversion manifest;
- train controls first through native-output reconstruction;
- validate full-sequence versus cached decode equivalence;
- run all repository and architecture tests.

### Exit criteria

- disabled path matches native output and generation within frozen tolerances;
- save/load and cache round trips pass;
- one target layer trains without destabilizing validation NLL;
- no hidden parameter-capacity mismatch between arms.

See `QWEN_INTEGRATION_PLAN.md`.

---

## Stage 3 — controlled language modeling

### Goal

Determine whether TGSA improves general language modeling rather than only synthetic tasks.

### Data protocol

- use a documented permissive corpus with content hashes and license cards;
- produce leakage-resistant train/dev/test partitions;
- use the same tokenizer and exact token stream across arms;
- forbid benchmark test text from training or hyperparameter selection;
- freeze the primary evaluation before the final run.

### Metrics

- validation/test NLL and perplexity;
- token accuracy where meaningful;
- convergence versus tokens and FLOPs;
- calibration;
- short-context generation quality;
- trajectory utilization and gate saturation as diagnostics only.

### Controls

Include native attention fine-tuning with identical trainable parameter count and a standard deeper/recurrent control with comparable realized compute.

### Exit criteria

- no registered NLL guardrail failure;
- effect replicates across seeds;
- benefit is not confined to one sequence length or one domain.

---

## Stage 4 — long-context and lost-in-the-middle evaluation

### Goal

Test the central practical hypothesis: iterative head-level refinement can recover or integrate evidence that one-shot parallel heads underuse.

### Evaluation strata

- context lengths: 2K, 4K, 8K, 16K, and selected 32K stress cases;
- answer evidence positions: beginning, early-middle, exact middle, late-middle, end;
- one-hop retrieval, multi-hop synthesis, update/override, contradiction, and distractor-heavy cases;
- answer-length and evidence-count strata;
- clean and messy-record formats.

### Data separation

Use three banks:

1. **training generators/corpora**;
2. **visible development bank** for debugging and thresholds;
3. **independently authored sealed bank** for the primary paper endpoint.

The existing Rineya long-context demo is diagnostic only because it has already influenced design decisions.

### Primary endpoint

Sealed long-context macro accuracy, stratified so large easy categories cannot dominate. The registered comparison is full TGSA minus cascade, using paired bootstrap over identical examples.

### Secondary endpoints

- exact and partial-credit accuracy;
- evidence citation/selection precision and recall;
- position sensitivity curve;
- calibration and abstention;
- NLL/perplexity guardrail;
- latency, memory, and cache bytes;
- failure taxonomy.

---

## Stage 5 — trajectory intervention and interpretation

### Goal

Determine whether the trajectory signal is causal to behavior rather than merely correlated.

### Interventions

- zero velocity at selected heads;
- shuffle velocities across examples while preserving magnitude;
- reverse velocity sign;
- replace velocity with matched Gaussian noise;
- clamp predictor steps;
- disable Q, K, or V conditioning independently;
- flatten token-importance scores;
- force early exit at selected heads;
- transplant trajectories between correct and incorrect runs where shape-compatible.

### Analyses

- correctness change under intervention;
- evidence-attention movement;
- intermediate-logit change;
- convergence and oscillation diagnostics;
- whether simple entropy/margin baselines explain the same outcomes.

No telemetry-based “reasoning” claim is allowed without intervention and held-out replication.

---

## Stage 6 — efficiency engineering

### Goal

Recover enough throughput that quality gains can produce a useful Pareto frontier.

### Optimization ladder

1. profile unfused reference implementation;
2. remove unnecessary materialization and Python overhead;
3. fuse carrier/gate projections;
4. share low-rank trajectory encoders within K/V groups;
5. batch operations across independent head groups;
6. add soft progressive token attenuation;
7. evaluate differentiable hard-concrete/top-k compression as a distinct arm;
8. add adaptive head stopping or routing;
9. implement Triton/CUDA kernels only after semantics freeze.

### Measurement protocol

- identical hardware and power mode;
- fixed software/driver versions;
- warm-up excluded;
- enough repetitions for confidence intervals;
- prefill and decode reported separately;
- batch sizes 1 and production-relevant larger batches;
- multiple context lengths;
- median, p90, throughput, peak memory, KV bytes, and energy if available.

The Phase-0 CPU latency ratio is a smoke observation, not a benchmark.

---

## Stage 7 — replication and paper release

### Goal

Produce evidence that another researcher can audit and reproduce.

### Required package

- frozen code commit and tagged release;
- environment lock and hardware manifest;
- full configurations and hashes;
- base-model revision and conversion record;
- data cards, licenses, generators, and split hashes;
- raw per-example predictions;
- raw timing samples;
- all seeds, including failed runs;
- statistical scripts and table-generation code;
- negative results and deviations from preregistration;
- final related-work audit;
- limitations, compute cost, and broader-impact discussion.

### Release gate

Every claim in the abstract must map to one preregistered result row and one immutable evidence bundle. See `PAPER_EVIDENCE_CHECKLIST.md`.

---

## Statistical analysis

### Primary test

For each sealed example, calculate the paired full-TGSA minus cascade score. Use a 10,000-sample stratified paired bootstrap, stratifying by task family, context length, and evidence position. Report:

- mean absolute difference;
- 95% confidence interval;
- probability that the difference is positive;
- practical-threshold comparison;
- seed-level distribution.

### Multiple comparisons

The single primary comparison is exempt from secondary-family correction. Apply Holm correction within each registered secondary family. Label all unregistered analyses exploratory.

### Seeds

Use five fixed seeds for paper-quality small/medium experiments unless compute makes that impossible; any reduction must be declared before viewing the primary result. Never drop a seed because it is unfavorable.

### Missing or failed runs

- retain logs and failure cause;
- retry only under a predeclared infrastructure-failure rule;
- distinguish numerical failure from infrastructure failure;
- count architecture-specific numerical failure as evidence, not missing data.

---

## Evidence ledger

For every experiment row, record:

- question and hypothesis ID;
- arm and fairness view;
- exact commit and resolved config hash;
- model/data/tokenizer hashes;
- seed and data-order seed;
- trainable/total parameters;
- tokens, steps, FLOPs, time, memory, cache bytes;
- primary and guardrail metrics;
- raw artifact locations;
- status and exclusion rationale;
- claim unlocked or falsified.

The ledger must be append-only after the sealed evaluation begins.

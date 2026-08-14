# Limitations and Ethics

## Limitations

### Paper-ready limitations narrative

This study evaluates a single seven-member committee of rank-16 LoRA adapters
sharing a custom 344M-parameter student backbone on primarily synthetic,
templated short-response tasks. The resulting experts are not representative of
all adapter ensembles, and the study does not establish that its findings extend
to larger or public backbones. The observed behavior may change across backbone
architectures and parameter scales, adapter-training recipes, datasets and task
distributions, and generation lengths; these factors must therefore be tested
before treating the mechanism as general.

The expert pool also does not causally isolate why committee members differ.
Specialization, random seed, competence, calibration, training mixture, and
policy behavior vary together rather than in a matched factorial design. Because
all experts share one backbone lineage, their errors may also be more correlated
than those of independently trained models. Matched specialist and same-task
multi-seed committees are needed to distinguish benefits of intentional
specialization from generic variation among adapters.

The main v2.2 benchmark contains unique example identifiers but substantial
synthetic and structural templating, so example-level sample size may overstate
the number of independent latent tasks. Exact-match accuracy can additionally
confound semantic error with formatting variation. The archived 32-example
pilot exposed part of v2.2 TEST before the full evaluation, and the v2.1.2
transfer corpus has documented historical use and train/test overlap. These
provenance constraints, together with the failure of hard voting to transfer to
v2.1.2, limit claims of external validity and motivate a fresh, preregistered
evaluation corpus.

Hard plurality is mathematically resistant to confidence-scale domination when
there is a unique plurality, but that structural property is not a guarantee of
higher accuracy. Confidence-sensitive tie breaking remains, correlated experts
can agree on the same wrong token, and calibrated probability fusion outperforms
hard voting on DEV. Accordingly, the v2.2 TEST advantage should be interpreted
as a benchmark-dependent outcome of this committee and task distribution, not
as a general local-decision advantage or a universal theorem about hard voting.

Aggregate exact-match accuracy also combines opposing answerable and abstention
effects. Relative to different comparators, hard voting can rescue abstentions
while sacrificing answerable cases, or reduce false abstention while losing some
genuine-abstention accuracy. The most appropriate operating point therefore
depends on class prevalence and the relative costs of an incorrect answer, an
unnecessary abstention, and a failure to abstain; those deployment utilities are
not identified by aggregate benchmark accuracy.

Finally, full K=7 decoding is computationally expensive, and reported latency
and memory measurements are specific to the evaluated laptop GPU, software
stack, and reference decoder. The study does not establish comparable behavior
for other PEFT methods, weight-space merging, learned routing, long-form
generation, natural user distributions, or high-stakes settings. Before such
deployment, future work should test larger public backbones, matched causal
controls, non-templated and longer-form tasks, utility-weighted abstention,
strong routing and merging baselines, and hardware-aware inference.

### One primary backbone lineage

The clean K=7 committee uses one 344M-parameter Qwen3-derived frozen backbone
and seven compatible LoRA specialists. Results do not establish behavior for
larger models, other architecture families, or separately trained backbones.

### Seven LoRA experts

All primary experts are rank-16 LoRAs with common target modules. Software
support for DoRA, PiSSA, AdaLoRA, or mixed PEFT does not constitute quality
evidence. No claim should generalize across PEFT methods without matched trained
committees.

### Specialization is not yet isolated causally

The official pool lacks a matched same-task multi-seed family. Diversity may
arise from specialization, random seed, data mixture, policy differences, or
unequal expert competence. H9 versus H10 remains unresolved without controlled
training.

### Benchmark specificity and templating

The primary corpus is deterministic and templated. Although TEST contains 4,050
unique IDs and no exact row duplicates, those examples are not necessarily
4,050 independent latent tasks. Example-level p-values may overstate effective
power if prompt templates induce correlated failures. The completed reasoning-
family and structural-template cluster bootstraps are materially wider than
the finite-corpus example bootstrap, and the deterministic family definitions
remain sensitivity analyses rather than a definitive effective sample size.

### Prior TEST exposure

An archived n=32 pilot sampled the same TEST split before the full-corpus run.
The canonical method/config hash is identical between runs, reducing but not
eliminating selection-exposure concerns. This history must be disclosed; newly
selected mechanisms and thresholds use DEV and require a fresh frozen protocol
before any further TEST statistic.

The added calibrated-probability TEST condition satisfies that procedural
freeze but is not a wholly untouched comparison: hard and raw-probability
anchor outcomes were already known when it was frozen. Its two paired tests
must therefore be reported as a selected confirmation with both the
known-anchor and prior n=32 caveats, not as a fresh three-way benchmark search.

The v2.1.2 result is a fixed-method cross-corpus transfer check, not a globally
untouched holdout. Although it has no exact prompt overlap with v2.2, 1,008 of
1,320 examples repeat exact v2.1.2 TRAIN prompt+gold pairs and the TEST split
was historically used by another model. Its negative hard-vote result limits
generalization; it cannot establish template-independent external validity.

### Exact-match scoring

Exact match is the official primary metric and is independently reproduced.
It can mark semantically correct paraphrases wrong or formatting deviations as
failures. The preselected secondary audit found zero semantic rescues among 100
exact-match failures, but it used one reviewer and does not replace the official
metric or rule out a small residual semantic-error rate.

### Short-response and abstention mixture

Aggregate accuracy combines 3,510 answerable and 540 abstention examples. The
full ranking can hide large, opposing answer-class effects. Balanced or policy-
specific metrics are descriptive unless predeclared, and deployment utility
depends on the intended class mixture and error cost.

### Autoregressive dependence

Tokens within a sequence are not independent. Teacher-forced token counts
cannot be treated as independent sample size for sequence-level claims.
Free-running methods visit different prefixes, complicating local comparison.

### Inference cost and cache state

K=7 requires up to seven expert evaluations per token. The reference decoder
recomputes the full prefix because the separate-cache bf16 implementation
failed an exact decision-parity gate. Latency is hardware/software specific;
shared-cache approximations could change both cost and decisions.

### Hardware-specific efficiency

Measurements come from an RTX 4070 Laptop GPU under Windows, Python 3.11,
torch 2.12.1+cu126, transformers 5.14.1, and PEFT 0.19.1. Driver state, competing
display workloads, precision, adapter loading, and implementation details limit
portability of timing and VRAM numbers.

### Multiple hypotheses and analyses

Phase 2 intentionally evaluates many mechanisms. DEV exploration and TEST
confirmation must remain separate, and planned comparison families require
Holm correction. Descriptive subgroup patterns should generate hypotheses, not
be promoted post hoc as independent discoveries.

### No broad intelligence claim

The results do not prove general intelligence emergence, human-like reasoning,
grokking, universal superiority of hard voting, optimality of seven experts,
or compute-efficient deployment.

## Ethics

### Compute and environmental cost

Multi-expert decoding multiplies expert-token forwards and energy use relative
to one adapter. The work should report quality per forward and measured latency,
not only accuracy. Adaptive consultation, quantization, and hardware-aware
parallelization are efficiency research questions rather than excuses to omit
cost.

### Reproducibility and transparency

Every reported result is linked to raw/config/dataset/model/adapter/script
hashes. Negative findings, failed cache parity, prior TEST exposure, exact-match
limitations, and untrained controls are retained. This reduces the risk that a
larger result is presented without its selection history.

The long reciprocal-rank sweep also crossed an implementation-maintenance
boundary after CUDA allocator failures. Pre- and post-change rows share one
mathematical rule only because the new stable partial top-k is exhaustively and
realistically parity-checked against the original stable full sort. The parity
artifact and current voting-source hash are mandatory audit inputs; without
that evidence, the mixed-runtime raw file should not be treated as one
condition.

The completed-row boundary and any actual pre-append retries are reported in
`selected_dev_runtime_boundaries.json`; this artifact must exist and match the
selected analysis hash before final synthesis is verified.

### Dataset provenance and privacy

The reviewed v2.2 corpus appears deterministic/synthetic from its source and
verification metadata. The paper must verify and document that provenance
before publication and must not infer that no privacy risk exists merely from
example style. Any external cross-benchmark data require separate license,
privacy, and contamination review.

### Bias and policy composition

All experts inherit biases from the shared backbone and their adaptation data.
Combining an abstention specialist with answer-capable experts changes the
system's refusal/answer policy and may affect groups or topics unevenly. Domain
and answer-class analyses are necessary but not a substitute for a dedicated
bias evaluation.

### Misuse

Stronger small models can lower the cost of automated content generation and
decision support. The present short-response benchmark does not establish
high-stakes reliability. Results should not justify medical, legal, financial,
or safety-critical deployment without domain-specific validation and human
oversight.

### Contamination disclosure

Training exposure, model selection exposure, and evaluation exposure are
separate fields in the dataset table. The prior n=32 TEST access and any later
cross-benchmark overlap must be disclosed rather than folded into a generic
“held out” label.

## Final evidence binding

Bounded conclusions are tied to `final_synthesis_facts.json`. Runtime
equivalence and historical execution boundaries are independently recorded in
`rank_topk_parity_verification.json` and
`selected_dev_runtime_boundaries.json`; these are provenance controls, not
additional scientific observations.

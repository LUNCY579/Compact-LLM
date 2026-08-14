# Paper Analysis and Draft Framework

## Candidate titles

1. **Hard Token Consensus for Inference-Time Composition of Low-Rank Language-Model Specialists**
2. **Decision-Space Composition of Parameter-Efficient Experts by Autoregressive Token Voting**
3. **When Expert Choices Beat Expert Confidence: Hard Voting Across Specialized Language-Model Adapters**
4. **Token-Level Consensus Among Parameter-Efficient Specialists for Short-Response Language Models**
5. **Composing Specialized Adapters at Inference Time Through Hard Next-Token Voting**
6. **Hard Versus Soft Fusion of Low-Rank Expert Adapters in Autoregressive Decoding**
7. **Inference-Time Capability Composition with Discrete Adapter Consensus**
8. **Calibration-Robust Token Aggregation for Ensembles of Specialized LoRA Adapters** *(use only if calibration interventions support the mechanism)*
9. **Coalitions of Low-Rank Specialists: Mechanisms of Hard Token-Level Consensus**
10. **From Specialist Decisions to Shared Trajectories: Hard Consensus in Adapter Ensembles**
11. **Short-Response Generation by Hard Consensus Across Parameter-Efficient Experts**
12. **Analyzing Decision-Space Ensembles of Specialized Language-Model Adapters**

No title uses “grokking.” The experiment concerns inference-time committee behavior,
not delayed training-time generalization.

## Abstract draft framework

Small language models can be adapted cheaply to different capabilities, but it
is unclear whether independently specialized adapters can compose those
capabilities without merging their weights or training a router. We study a
shared 344M-parameter short-response backbone with seven compatible rank-16
LoRA specialists. At each autoregressive step, every specialist evaluates the
same accepted prefix; hard consensus selects the plurality of expert top-1
tokens, while soft comparators average probabilities or logits.

On the complete 4,050-example v2.2 TEST split, the independently verified hard
K=7 condition reaches **17.259% exact accuracy**, compared with **14.593%** for
the strongest predeclared broad individual control and **14.099%** for raw
probability mean. The paired gains are **+2.667 percentage points** over the
broad control (exact McNemar (p=4.54\times10^{-6}), Holm-adjusted
(p=1.36\times10^{-5})) and **+3.160 points** over probability mean
((p=8.72\times10^{-13}), Holm-adjusted (p=4.36\times10^{-12})). These
figures trace to the raw result SHA256
`d3733391b53dc53e7d9ca899d7015ba6ff0b2faa7c28e52a2778a0827c45c32d`.

Paired analysis shows that hard voting rescues 328 Track-A failures while
harming 220 Track-A successes, and rescues 226 probability-mean failures while
harming 98 probability-mean successes. Its aggregate advantage over Track A is
policy-compositional: it is lower on answerable cases but much stronger on
abstention cases. Against probability mean, hard voting improves answerable
accuracy while reducing abstention accuracy. Hard and soft fusion produce
identical complete token sequences on 55.46% of TEST examples; 83.3% of their
first divergences occur at the initial token.

On teacher-forced DEV gold prefixes (4,050 examples; 18,760 token decisions),
hard K=7 is not the best local rule: it obtains 78.33% token accuracy and 25.26%
all-gold-token example accuracy, while raw probability mean obtains 78.91% and
26.62%, and per-expert calibrated probability mean obtains 79.23% and 26.77%.
The best tested rank-only rule, reciprocal-rank aggregation, reaches only 76.34%
and 18.47%, contradicting the local hypothesis that ordinal information alone
reproduces the benefit. Independent temperature fits reveal substantial
inter-expert calibration differences; positive rescaling leaves hard proposals
and unique-plurality winners invariant but can move its confidence-based
tie-break, while materially changing soft decisions more broadly. Exact Shapley analysis assigns
positive hard-vote local value to Track A, Track B, General, Procedure, and
Abstention, and negative value to Grounding and Social. These findings imply
that the TEST free-running hard advantage is not a uniform improvement in local
gold-prefix classification. The completed same-split DEV run preserves the
teacher-forced ordering: calibrated probability 26.790%, raw probability
26.593%, hard 25.309%, and reciprocal rank 18.321%. Calibrated soft beats hard
on DEV by +1.481 pp, while reciprocal rank trails by −6.988 pp. The
near-identical teacher/free gaps do not support trajectory correction as the
source of hard's gain.

The frozen calibrated v2.2 TEST confirmation reverses the DEV comparison:
calibrated probability reaches 13.827% versus hard's 17.259% (−3.432 pp) and
does not differ significantly from raw probability. A fixed-method v2.1.2
cross-corpus run is also negative for hard (0.076% versus raw probability's
3.030%). The result is therefore benchmark-specific and does not establish
cross-corpus superiority. The v2.2 TEST confirmation retains the prior n=32
and known-anchor caveats, and v2.1.2 retains documented TRAIN repetition and
historical TEST-use caveats.

Hard K=7 uses seven expert forwards per generated token and is therefore not a
free deployment improvement. The frozen adaptive policy is statistically
indistinguishable from full K=7 on overall DEV accuracy (25.111% versus
25.309%; p=0.546) while reducing mean expert forwards by 56.579% and mean
latency by about half; it also trades +1.168 pp answerable accuracy for −9.074
pp abstention accuracy. The study is limited to one primary backbone lineage,
seven LoRA experts, exact-match short responses, deterministic/templated data
with correlated families, and prior exposure to an archived n=32 TEST sample.

## Introduction

### Core question and motivation

Parameter-efficient fine-tuning creates task- or policy-specialized variants of
a shared language model without duplicating all backbone weights. This makes it
cheap to train and store specialists, but it does not by itself explain how to
compose their capabilities. Weight merging compresses several deltas into one
model but can produce interference. Learned routing requires additional
training and may commit to one expert before seeing how specialists disagree at
the current token. Conventional ensembles preserve experts but aggregate
confidence magnitudes whose scales may not be comparable after independent
specialization.

ELREA is the closest prior method: it already combines LoRA experts at each
token under a shared autoregressive prefix using input-gradient relevance and
weighted preactivation logits. The question here is therefore narrower. We ask
whether a router-free committee can compose predefined specialists by equal
plurality over their top-1 token identities, and which measurable interactions
explain the resulting gains and failures.

Hard token consensus offers a deliberately small interface: each expert
contributes only its preferred next token. Every expert has one vote, and the
accepted token becomes the shared context for the following step. The method
therefore composes decisions rather than weights and can change the entire
autoregressive trajectory after one early consensus choice.

The primary empirical question is:

> Can independently specialized parameter-efficient adapters compose
> capability at inference time by voting on every autoregressive token?

The Phase 2 question is mechanistic:

> What property of hard, discrete consensus lets it outperform both individual
> specialists and arithmetic probability fusion in the verified full-TEST
> observation?

Candidate explanations include calibration-scale invariance, complementary
errors, task-dependent specialist coalitions, ordinal-information bottlenecks,
robustness to overconfident dissenters, answer/abstain policy composition, and
autoregressive path stabilization. These mechanisms make overlapping
predictions and must be separated experimentally rather than bundled into a
post hoc story.

### Why the pilot reversal matters

The archived 32-example feasibility packet ranked probability mean above hard
voting (37.5% versus 25.0%) and made the abstention specialist appear strongest
(50.0%). The full 4,050-example corpus instead ranks hard K=7 above probability
mean (17.259% versus 14.099%) and estimates abstention at 13.012%. The pilot was
useful for engineering and feasibility, but it was too small and distribution-
sensitive for method ranking. The reversal motivates high-powered paired
evaluation, explicit stratification, and careful separation of exploratory
selection from confirmation.

### Decision-space versus weight-space composition

Weight-space composition constructs one parameter vector from several
specialist deltas and then emits one prediction. Decision-space composition
keeps specialists separate, asks each for a token decision at the current
prefix, and aggregates those decisions. These paradigms differ in storage,
latency, cache behavior, calibration sensitivity, and potential interference.
The existing D_greedy dense merge has a different adapter lineage, so any
comparison is contextual rather than a clean causal test unless matched
composition inputs are established.

## Falsifiable hypotheses

### H1 — Calibration-robust consensus

Experts have materially different confidence/entropy scales. Per-expert DEV
temperature calibration, logit normalization, or confidence clipping should
improve soft fusion and reduce the hard-vote advantage. If calibrated soft
fusion remains worse, scale mismatch is not sufficient.

### H2 — Complementary error correction

Expert errors are imperfectly correlated, and hard voting rescues examples
missed by strong individual experts. Useful committee gain should relate to a
competence–diversity tradeoff rather than maximum disagreement alone.

### H3 — Specialist coalition formation

Winner-support masks and marginal contributions vary systematically across task
categories and answer classes. Removing aligned specialists should cause
category-specific losses larger than generic committee-size effects.

### H4 — Ordinal bottleneck / noise suppression

Discarding raw confidence magnitude removes harmful information. Rank-only
aggregation should approach or exceed raw soft fusion; if only top-1 plurality
works, a broader ordinal-bottleneck account is weakened.

### H5 — Robustness to overconfident experts

Sharpening one expert should move probability fusion while leaving hard top-1
votes and unique-plurality winners invariant; the confidence-based hard
tie-break is tested separately. Naturally high-confidence dissent should be enriched at
hard/soft divergences, and intervention damage should depend on which expert is
sharpened.

### H6 — Committee-size scaling or threshold

Best-at-(K) and mean-across-committees-at-(K) may show smooth gain,
saturation, degradation, or a threshold-like transition. A single good K=7
committee is insufficient evidence for monotonic scaling. “Phase transition”
language requires statistical support; “grokking” is inapplicable.

### H7 — Trajectory correction

The hard-vote advantage is larger under same-split free-running inference than
under teacher-forced local gold-prefix decisions. If local and free-running
gaps are similar, path stabilization is not central.

### H8 — Abstention/EOS policy composition

Answer-capable and abstention-capable specialists create a better aggregate
policy. Removing the abstention expert should specifically alter the
answerable/abstention frontier. The existing full-TEST result already shows
large answer-class tradeoffs but cannot identify the abstention expert's causal
contribution without leave-one-out inference.

### H9 — Diversity, not specialization

Matched same-task random-seed adapters should reproduce the gain. This remains
unresolved because no compatible matched seed family exists.

### H10 — Specialization matters

Matched specialist committees should outperform equally diverse seed-only
committees, and category alignment should predict marginal contribution. H9 and
H10 require newly trained controls if the official repository has no matched
family.

## Contributions ledger for the eventual paper

1. **SUPPORTED:** An exact token-level hard-plurality decoder for independently
   activated, compatible parameter-efficient experts that share every accepted
   prefix, with deterministic tie and EOS behavior.
2. **SUPPORTED:** A complete 4,050-example paired TEST evaluation showing hard
   K=7 above the frozen backbone, Track A, the abstention individual, and raw
   probability K=7 under the declared comparison family.
3. **SUPPORTED:** An independently reproducible provenance and integrity audit
   linking raw rows, dataset/config/model/adapter hashes, scoring, and paired
   statistics.
4. **SUPPORTED:** A rescue/harm decomposition showing that aggregate gains
   coexist with substantial harms and that answer/abstain composition is a
   central part of the observed ranking.
5. **SUPPORTED:** A DEV teacher-forced cache enabling calibrated fusion, rank
   aggregation, all 127 committees, leave-one-out, exact Shapley values, and
   controlled temperature interventions without storing full-vocabulary
   tensors.
6. **SUPPORTED WITH BOUNDS:** A mechanistic conclusion that separates
   calibration-resistant directional aggregation and answer/abstain policy
   composition from unsupported trajectory-correction and universal-superiority
   accounts.
7. **SUPPORTED ON DEV:** A frozen adaptive consultation policy that preserves
   overall K=7 accuracy within paired uncertainty while using 56.579% fewer
   expert forwards, with an explicit answer/abstain tradeoff.
8. **CONTRADICTED FOR THE TESTED TRANSFER:** Hard K=7 does not generalize to the
   fixed-method v2.1.2 cross-corpus benchmark. Generalization to other
   backbones, model scales, PEFT types, or matched seed-only committees remains
   unestablished.

Final numerical bindings are `selected_dev_free_run_analysis.json`,
`adaptive_free_run_dev_analysis.json`, `selected_committee_dev_analysis.json`,
`v22_calibrated_test_analysis.json`, `cross_v212_analysis.json`, and
`final_synthesis_facts.json`.

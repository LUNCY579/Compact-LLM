# Significance causal analysis

Status: published-family execution-path smokes `SUPPORTED` on 5/5 cores; Halo
endpoint v2 locally registered retained-state contrast `SUPPORTED_NARROW`;
answer mediation `NOT_SUPPORTED`; count/schedule and Experiments B-F remain
exploratory development; preserved Tier-A interpretation
`UNDER_METHODOLOGY_REVIEW`.

## Current evidence boundary

The completed smokes support two engineering observations on one identical
bf16, 192-token prompt-only case for each of Halo, Halo Uncensored, Lyra,
Nimbus, and Pulse:

1. disabled chunked and single-shot recorded logits were exactly identical;
2. query-conditioned `S/C` state survived exact rollback and aligned
   compression to **B48+C64**, with two scoring events per layer.

They decoded no answers and had no frozen exact evidence annotations. They therefore
do not establish any link from ranking to evidence retention, later access,
or answer quality.

Runtime status: local recorder implementing audited semantics; four audited
source primitives match exactly, while end-to-end source benchmark reproduction
is `NOT_RUN`. Primitive parity is an implementation gate, not causal evidence.
The registered endpoint-v2 contrast is paired inside a
Torch 2.13.0/Transformers 5.15.0 process and has not been reproduced under the
current 2.12.1/5.14.1 lock used by later architecture screens.

The existing model-free Tier-A run and figures are preserved for audit, but its
experimental unit, pairing, cluster structure, controls, and interpretation are
under review. Its prior toy contrasts are not current causal evidence and may
not be generalized to Halo or the published Qwen3-0.6B family.

Rineya/R2 and Aegis models are outside current causal conclusions. RineyaBeta
appears only as immutable mechanism provenance.

## Bounded Halo answer result

The corrected `halo_mini_eval_20260823_v2` decoded three fixed cases. B0, B2,
and B3 each scored 1/3; B2 and B3 each scored 1/1 on the sole B0-eligible case.
Thus there was no measurable B3 gain over B2 in this slice. Two B0 capability
failures, one eligible case, no inferential statistics, and no exact-evidence
scoring prevent a rank-to-retention-to-answer claim or an equivalence claim.
The result supplies a bounded negative/descriptive screen, not causal support.

## Calibration and retention-pilot result

`halo_calibration_v3` established 32 independent exact-evidence development
clusters. Full-cache Halo solved 18/32, but capability was sharply
family-dependent (`R1=7/8`, `R2=0/8`, `R3=3/8`, `R4=8/8`). The balanced gate
failed because the R2 benchmark family contributed no eligible cluster. The
later pilot therefore selected only 11 B0-correct R1/R3/R4 clusters and cannot
estimate the full calibration population or any R2-family effect.

In that selected pilot, B2/B3/B4 answer counts were 5/6/1 out of 11. B3 had one
answer win and no loss versus B2, but a two-sided exact cluster sign-flip test
gave `p=1.0`. B3 versus B4 was descriptively much larger (six wins, one loss;
mean paired correctness difference `+0.4545`, exploratory `p=0.125`). Neither
answer contrast supports a causal or confirmatory claim.

The proposed strict mediator—every required fact token available at every
evaluated layer—was false for all 11 clusters in B2, B3, and B4. This all-zero
saturation makes it unsuitable as the primary mediator in the current
implementation/budget.

An outcome-blind evidence-only calibration froze
`layer-fact-retention-v1`, the macro-mean over atomic complete-fact-by-layer
indicators, for future evaluations. It passed a predeclared non-saturation
screen. Applying it back to its endpoint-design pilot gave B3-B2 `+0.061688`
(95% interval `[0.030844, 0.094156]`, `p=0.0078125`). This is necessarily
post hoc because the same 33 evidence rows were used to select the endpoint.
It shows retained-state separation in this development sample, not causal
mediation of answers.

The first new prospective dataset preserved that endpoint but failed the
pre-answer B0 gate (`9/24`), so no policy comparison ran. After explicit
difficulty calibration of task templates, v2 passed B0 (`18/24`). A local
registration then froze 12 B0-correct clusters, B2/B3/B4 conditions, and B3-B2
as the sole primary endpoint contrast before compressed execution. B3-B2 was
`+0.0401786`, 95% interval `[0.0133929, 0.0758929]`, exact `p=0.01953125`.
This is prospective evidence for the rank-to-retained-state link in that local
sample. The next link did not hold: B3 scored `5/12` answers versus B2 `6/12`,
one loss and no wins, `p=1.0`. Thus the registered sequence supports retained-
state separation while directly withholding rank-to-answer and mediation
claims. Difficulty tuning after the v1 gate failure and the small local
registration also prevent broad population inference.

## Proposed causal chain

For each exact evidence unit and major distractor, a registered run should
reconstruct:

```text
entry enters cache
  -> eligible scoring observations
  -> raw S and realized C
  -> normalized/raw/age rank
  -> retention at each boundary
  -> later attention access
  -> intermediate/final answer outcome
```

Record layer, KV head, absolute position, stable entry ID, score, count,
normalized score, age, protected status, rank, decision, and the mapping to the
frozen evidence unit.

## Intended primary contrast

Hold checkpoint, prompt, query access, retained history, active chunk,
transient query, byte allocation, and randomization fixed. Change only:

```text
B3: R = S / max(C, 1)
B2: R = S
```

The intended paired direction is B3 minus B2. The tested strict evidence
endpoint is `REJECTED_AS_PRIMARY_FOR_CONFIRMATION_SATURATED`. The replacement
`layer-fact-retention-v1` endpoint now has one completed, locally registered
prospective v2 test. That estimand and result apply only to its 12 difficulty-
calibrated clusters. The broader confirmatory protocol remains unregistered,
and capability-conditioned exact answer correctness remains a distinct
secondary outcome with no detected improvement.

## Cluster-aware dependence

All prompt renderings, policies, depths, and compression schedules derived from
one latent generated world remain one cluster. Splits and uncertainty operate
on whole clusters. A set of many rendered variants from one world is not many
independent replications. The registration must freeze `cluster_id`, eligible
cluster rules, and whole-cluster resampling/randomization before outcomes.

## Completed compression-boundary interaction

`halo_schedule_control_v2` executed this intervention on 11 B0-selected
R1/R3/R4 clusters. Five schedules per case and B2/B3 produced 110 cells and 55
matched method pairs. The prompt, question, checkpoint, decode contract,
128-entry retained history, 64-token terminal active suffix, and within-pair
query access/final pre-decode prompt-cache size were held fixed. All derivatives of one latent
case remained one cluster.

For the frozen endpoint, the B3-B2 cluster-level linear interaction was
`+0.00936074` per additional realized compression round, 95% interval
`[0.00310399, 0.01759850]`, exact sign-flip `p=0.00390625`. The analogous
answer interaction was `+0.02248157`, interval
`[-0.03599509, 0.07641278]`, `p=0.4375`. Thus this development run detects a
schedule-dependent retained-state contrast but not an answer interaction.
Because clusters were selected on B0, reused from calibration, and the
analysis was not registered, the result is exploratory rather than a causal
population or mediation claim.

## Completed count interventions

The one-case `results/halo_count_control_identity.json` engineering check first
verified that the standard B3 compressor and true-count callback produced
identical generated token IDs, resident positions, accounting, policy, and
exponent in the same real-Halo process. This supports the intervention hook,
not a quality or schedule-generalization claim.

`halo_count_causal_controls_v1` completed 88 cells across true, constant,
age-derived, shuffled, wrong-token, delayed, noisy, and head-averaged counts on
the same 11 selected clusters. Query access and retained-history capacity were
matched, true C was preserved as an audit trajectory, and 264 boundary-evidence
rows excluded facts not yet fully observed.

Age-derived and head-averaged counts were exact trajectory equivalents of true
C in all 11 clusters: effective counts, scores, selections, final residents,
predictions, frozen endpoint, and correctness matched exactly. This answers a
narrow causal question for the fixed 128-token/one-probe-per-boundary schedule:
explicit granular C added no information beyond deterministic age in that
regime. It does not establish equivalence under irregular scoring access.

Constant and shuffled transforms reduced the frozen endpoint by `-0.055195`
and `-0.058442`. Nominal `p=0.015625` and `0.046875` became Holm-adjusted
`p=0.109375` and `0.28125`; neither survives multiplicity control. The delayed
condition was descriptively higher (`+0.063312`, nominal `p=0.1171875`, Holm
`p=0.5859375`) and supports no directional claim. All answer-comparison
`p` values were `1.0`.

Shuffled and wrong-token transforms are not clean entry-identity interventions.
Their global permutations crossed the four protected sink and 32 protected
recent entries; on the audited case, the eligible-heavy-pool C multiset changed
in 224/224 layer-head cells. Their correct classification is
`GLOBALLY_DISTRIBUTION_PRESERVING_BUT_ELIGIBLE_POOL_CONFOUNDED`, so their
outcomes cannot be attributed solely to score/count identity.

`halo_count_causal_controls_v2` then implemented the missing clean
intervention: C values are permuted only inside the eligible heavy pool, while
protected sink/recent denominators stay fixed and the eligible C and semantic-
ID multisets are preserved separately per layer/head. Both eligible-shuffled
and eligible-circular transforms have zero semantic fixed points. The runtime
audit passed 22,176/22,176 layer/head cells.

Against true C, eligible-shuffled changed the endpoint by `-0.021104`
(`p=0.4921875`, Holm `0.4921875`) and eligible-circular by `-0.014610`
(`p=0.125`, Holm `0.25`). Answer deltas were -3/11 (`p=0.25`) and -1/11
(`p=1.0`). These clean controls therefore detect no identity-specific effect
under their frozen multiplicity family. They do not establish equivalence.

The true-C path was exact in a same-process identity replay. Separate v1 and
v2 CPU-bf16/SDPA runs, however, matched only 10/11 predictions, 9/11 endpoint
values, and 0/11 resident trajectories. Because the cause of that process-to-
process drift is unestablished, causal interpretation is restricted to paired
conditions executed in the same v2 process.

## Mediation categories

- `RANK_TO_RETENTION_TO_ANSWER`;
- `RANK_CHANGED_NO_RETENTION_CHANGE`;
- `RETENTION_CHANGED_NO_LATER_ACCESS`;
- `ACCESS_CHANGED_NO_ANSWER_CHANGE`;
- `ANSWER_CHANGED_WITHOUT_TRACE_EXPLANATION`;
- `FULL_CACHE_CAPABILITY_FAILURE`.

Unexplained outcomes are evidence against the proposed mechanism, not cases to
discard.

## Open-loop boundary

Open-loop replay can compare policies only on recorded support. It cannot
estimate attention priors, memory slots, or controllers that change future
hidden states. Those require `CLOSED_LOOP_MODEL_EXECUTION`. The old
`tier_a_20260823` figures remain review artifacts and cannot be repurposed as
Halo plots.

The new Experiment-B screen is a valid boundary-local open-loop result over 132
rows (11 clusters x three boundaries x four strategies), with exact 28,672-entry
global allocation and 33/33 uniform-runtime parity checks. At its primary
boundary, entropy-proportional allocation was `+0.055195` versus uniform
(`p=0.107422`, Holm `0.214844`), while shuffled-statistic was `+0.115260`
(`p=0.014648`, Holm `0.043945`). Because shuffled exceeded entropy, the
specificity gate failed: `NO_SIGNIFICANCE_SPECIFIC_ADVANTAGE`. No model was
loaded and no earlier allocation change was propagated to later hidden states.

## Closed-loop architecture interventions B-F

Experiment B closes the allocation loop, but not the specificity argument. Its
36 cells propagate uniform, entropy, or shuffled allocations through later
hidden states. Entropy produced three additional correct answers versus
uniform (`8/12` vs `5/12`), but the paired test was `p=0.25`; shuffled had the
highest mean endpoint and entropy-minus-shuffled retention was negative. The
intended chain “significance statistic -> allocation -> retained evidence ->
answer” therefore lacks statistic specificity and answer inference. Padding
also makes physical bytes larger than the matched logical payload.

Experiment C directly changes later attention and therefore future S/C. All
four nonzero priors changed logits and cache state, yet all five conditions
decoded the same answers (`5/12`). Normalized, raw, and shuffled significance
had the same aggregate endpoint. This is an active-intervention instance of
`ACCESS_CHANGED_NO_ANSWER_CHANGE`, but exact distractor persistence was not
token-aligned and no safety claim follows.

Experiment D replaces exact ordinary positions with blended memory slots. All
six byte-matched conditions solved the same five clusters. Normalized memory
had a lower exact-token endpoint than M0 and random and beat no simpler M4
control. Because the endpoint deliberately gives blended slots no exact-token
credit, the negative retention contrast combines ordinary-capacity replacement
with any memory effect; the zero answer contrast supplies no recovery evidence.

Experiment E's fixed answer-blind risk signal had error AUC `0.50`, so the
Cornerstone diagnostics did not order recovery need. Its live smoke supports
only the action link: a recover decision can safely discard compressed state
and recompute the full context, while accept is literal identity.

Experiment F changes the decision score, not the recovery action. Its frozen
5/3/3 split and seven-parameter logistic model perfectly ordered three held-out
clusters and counterfactually recovered both errors. With only two positive-
negative pairs, one constant training feature, and a tree control that visibly
overfit, this is `UNDERPOWERED_OVERFIT_SCREEN_ONLY` with high overfit risk. It cannot support
the causal statement that Cornerstone state generally determines when recovery
is necessary.

## Still `NOT_RUN`

- registered exact evidence survival and adequately powered answer accuracy
  under B0/B2/B3 controls;
- broad registered replication of the locally validated endpoint on a
  materially larger, diverse cluster split;
- stale-information and distractor behavior;
- registered long-horizon head/layer concentration, churn, and exact
  distractor/stale persistence;
- registered replication of the exploratory compression-round interaction;
- answer-quality benefit or mechanism specificity for adaptive allocation,
  attention prior, memory slots, recovery, or a learned controller;
- truly ragged adaptive storage and deployment-grade overhead;
- quality versus measured CPU/GPU latency, RSS, or VRAM;
- cross-core generalization within the published family.

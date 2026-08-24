# Failure analysis

Status: published-family runtime engineering gates `PASS` on 5/5 cores; the
locally registered Halo endpoint contrast is complete but did not improve
answers; R7 v2, eligible-count v2, and Experiments B-F are settled development
evidence; broad registered failure analysis remains `NOT_RUN`; old Tier-A
interpretation is `UNDER_METHODOLOGY_REVIEW`.

## Observed so far

The one-case bf16 smokes for Halo, Halo Uncensored, Lyra, Nimbus, and Pulse
reported no implementation failure: disabled-path recorded logits were exactly
identical, each query probe rolled back exactly, and aligned state compressed
from 192 true positions to **B48+C64**. Every layer recorded two scoring events
in every run.

No answer was generated in any smoke and no exact evidence outcome was scored. Therefore
`FULL_CACHE_CAPABILITY_FAILURE`, reasoning, evidence, distractor, stale-fact,
and decode categories are `NOT_RUN`, not “zero failures.” The local recorder
implements audited semantics. Four audited source primitives match exactly;
end-to-end source benchmark reproduction remains `NOT_RUN`.

The corrected bounded exploratory `halo_mini_eval_20260823_v2` did decode
answers. B0 was correct on 1/3 cases, so two cases are
`FULL_CACHE_CAPABILITY_FAILURE` for compression-conditioned interpretation.
B2 and B3 were each correct on 1/3 overall and on the sole B0-eligible case.
There was no measurable B3 gain over B2, but three cases and one eligible case
cannot establish equivalence. Because the run did not score the pending exact
evidence metric, evidence/rank-mediated failure labels remain unassigned.

The 32-cluster full-cache capability calibration supplied a stronger base-model
screen. Halo was correct on 18/32 overall: R1 `7/8`, R2 `0/8`, R3 `3/8`, and R4
`8/8`. The balanced advance gate failed because the R2 benchmark family had no
B0-eligible cluster. Those eight R2 outcomes are
`FULL_CACHE_CAPABILITY_FAILURE` for compression-conditioned analysis; R2 here
is a benchmark-family name, not an in-scope Rineya/R2 checkpoint.

The exploratory retention pilot selected 11 B0-correct clusters from R1, R3,
and R4. Exact-answer correctness was B2 `5/11`, B3 `6/11`, and B4 `1/11`.
B3 had one answer rescue and no harm relative to B2, but the exact two-sided
cluster sign-flip result was `p=1.0`; it does not support an answer effect.

The candidate strict endpoint—every required fact token available in every one
of 28 evaluated layers—failed on all 11 clusters under each condition. This is
0/33 condition-cluster observations, so it remains only a strict diagnostic.
It also makes the pilot's aggregate `EVIDENCE_EVICTED` labels too coarse to
establish a causal explanation: partial token availability remained nonzero
and differed between policies.

Outcome-blind evidence-only calibration froze
`layer-fact-retention-v1`, which scores complete facts within each layer and
then macro-averages layer/fact cells. Its application to the same endpoint-
design pilot is post hoc: B3-B2 `+0.061688`, interval
`[0.030844, 0.094156]`, `p=0.0078125`.

The first prospective endpoint dataset then stopped at B0 (`9/24`), correctly
assigning base-capability failure before any compression label. Difficulty-
calibrated v2 passed B0 (`18/24`) and locally registered 12 B0-correct clusters.
There B3 retained more exact layer/fact evidence than B2 (`+0.0401786`,
`p=0.01953125`) but answered fewer cases (`5/12` vs `6/12`, one loss, no wins,
`p=1.0`). This is an observed
`RETENTION_CHANGED_NO_DETECTED_ANSWER_IMPROVEMENT` pattern, not proof that
retention caused a failure. The strict diagnostic remained 0/12 in all arms.

The eight-row R5-R8 full-cache screen cannot be summarized as five valid
successes. R7's raw `2/2` is `INVALID_UNINFORMATIVE`: the generator is 6 NO /
1 YES across its seven seeds, both screen seeds are NO, the prompt contains an
exact lock-sentence shortcut, and all seven facts were marked required. Those
two rows receive a design-invalid label, not capability success or failure.
The valid R5/R6/R8 subset is descriptively 3/6, with R8 `0/2`; R8 stops before
compression-conditioned failure assignment. This bounded screen has no
inferential statistics.

R7 v2 repairs the task itself. Its 80-case dataset is balanced and passes
solver/necessity/counterfactual audits, but Halo answered `YES` on all eight
frozen screen clusters. The `4/8` total consists of `4/4` YES and `0/4` NO,
matching an always-YES baseline and failing the predeclared two-sided gate.
These are `FULL_CACHE_CAPABILITY_FAILURE` outcomes on the four NO clusters and
a capability-gate stop overall; no R7 compression failure label exists.

The count-control run completed 88 condition-cluster cells. Its corrected
descriptive labels are `EVIDENCE_PARTIALLY_RETAINED`: true C 5, age 5,
head-average 5, constant 6, shuffled 6, wrong-token 6, delayed 4, and noisy 5.
The endpoint observes final partial retention; it does not prove complete
eviction or identify the causal failure boundary. These rows must not be
treated as independent events or causal rates. Age
and head-average reproduced every true-C trajectory exactly in this fixed
schedule. Constant and shuffled frozen-endpoint deltas were nominal but not
Holm-significant; delayed is descriptive only. The 264 boundary-evidence rows
exclude facts that were not yet fully observed, preventing future facts from
being mislabeled evicted.

Shuffled and wrong-token C are additionally confounded as entry-identity
controls: their global permutations cross protected sink/recent positions, and
the eligible-heavy-pool C multiset changed in 224/224 audited layer-head cells.
Their failure-label differences cannot be attributed solely to identity.

V2 corrects this with eligible-only shuffled/circular permutations. Protected
denominators and eligible C/identity multisets were preserved in every one of
22,176 audited layer/head interventions. Neither endpoint contrast survived
the two-control Holm family, and answer contrasts were not detected. These are
clean null controls, not evidence that count identity is equivalent or never
matters. Separate v1/v2 processes did not reproduce resident trajectories
bitwise, so labels and contrasts are interpreted within the matched v2 process
only; the cross-process drift remains `UNKNOWN`, not `IMPLEMENTATION_ERROR`
without an established cause.

The held-constant schedule run completed 110 cells but has only 11 independent
clusters. A positive B3-B2 frozen-endpoint interaction was detected
(`+0.00936074`/round, `p=0.00390625`), while the answer interaction was not
(`p=0.4375`). This pattern is consistent with retention changing without a
detected answer change, but it is not a registered mediation category or a
population failure rate. Its 57 affected cell labels are likewise descriptive
`EVIDENCE_PARTIALLY_RETAINED`, not proven complete evictions.

The Experiment-B adaptive allocation screen failed its mechanism-specificity
gate. Entropy-proportional allocation improved the boundary-local endpoint by
`+0.055195` versus uniform, but shuffled-statistic allocation improved it by
`+0.115260` and was the only primary comparison surviving Holm correction
(`p=0.043945`). The appropriate decision is
`NO_SIGNIFICANCE_SPECIFIC_ADVANTAGE`, not an adaptive-budget success. Because
the screen is open loop, it does not assign an answer failure or predict what a
closed-loop allocation would do.

The closed-loop Experiment-B screen produced entropy `8/12`, uniform `5/12`,
and shuffled `5/12`, but entropy's paired answer contrast was `p=0.25` and
shuffled had the highest mean retention. This is a failed mechanism-
specificity screen, not three established rescues. Padded physical storage also
exceeded matched logical payload, so no efficiency failure/success label can be
inferred from the logical budget alone.

Experiment C changed logits and cache state in every nonzero-prior comparison
but left all 12 decoded answers unchanged; all arms scored `5/12`. Normalized,
raw, and shuffled priors also shared the aggregate endpoint. The descriptive
category is `ACCESS_CHANGED_NO_ANSWER_CHANGE`, not
`ATTENTION_FEEDBACK_COLLAPSE`: exact distractor/stale positions were unavailable
and no harmful-feedback chain was demonstrated.

Experiment D's six conditions all solved the same five clusters. Every M4 arm
retained fewer exact ordinary evidence tokens than M0, and normalized `S/C`
was below random and beat no simpler M4 control. `MEMORY_SLOT_INFORMATION_LOSS`
is not established per answer because blended slots receive no exact-token
credit and correctness did not change. The supported statement is simply no
observed recovery gain under the matched-total-capacity M4 design.

Experiment E's fixed risk score had error AUC `0.50`; therefore
controller-trigger errors cannot be assigned to a predictive Cornerstone risk
signal. The live accept/recompute smoke had no implementation failure.

Experiment F's logistic model correctly handled all three frozen evaluation
clusters, but this means only two positive-negative orderings. Its perfect
held-out metrics are `UNDERPOWERED_OVERFIT_SCREEN_ONLY` with high overfit risk, while the
depth-two tree's train-to-evaluation collapse is direct overfit evidence. No
general controller failure rate, rescue rate, or Finding D is assigned.

The preserved Tier-A run previously illustrated a constructed exposure process,
but its experimental unit, controls, clustering, and inference are under review.
It cannot currently establish an exposure-bias failure rate or a conclusion
about Halo. Rineya/R2 and Aegis are outside active failure analysis.

## Taxonomy

| Label | Operational meaning |
|---|---|
| `FULL_CACHE_CAPABILITY_FAILURE` | Halo full-cache reference is wrong or structurally invalid |
| `BASE_MODEL_REASONING_FAILURE` | exact required evidence remains accessible but Halo fails the task |
| `EVIDENCE_NEVER_SCORED` | required evidence receives no eligible score observation |
| `EVIDENCE_LOW_SIGNIFICANCE` | evidence is observed but ranks below the retained threshold |
| `EXPOSURE_BIAS` | raw accumulation's extra realized observations alter rank/retention relative to aligned normalization |
| `EVIDENCE_EVICTED` | the frozen exact evidence unit is removed before required use |
| `EVIDENCE_PARTIALLY_RETAINED` | only part of a frozen multi-unit requirement survives |
| `RETENTION_CHANGED_NO_DETECTED_ANSWER_IMPROVEMENT` | a paired retained-state endpoint changes, but the answer contrast is not improved or not detected |
| `DISTRACTOR_OUTRANKED_EVIDENCE` | an annotated distractor displaces required evidence |
| `STALE_INFORMATION_PERSISTED` | obsolete evidence survives and contributes to failure |
| `QUERY_SIMILARITY_FALSE_POSITIVE` | query-conditioned scoring keeps a near-match over decisive evidence |
| `ATTENTION_FEEDBACK_COLLAPSE` | a soft prior reinforces false positives and reduces diversity |
| `MEMORY_SLOT_INFORMATION_LOSS` | slot compression loses distinctions required for the answer |
| `INSUFFICIENT_KV_BUDGET` | oracle/matched controls show retained history cannot hold required evidence |
| `POSITIONAL_ENCODING_FAILURE` | absolute position, RoPE, or causal-mask handling is inconsistent |
| `DECODE_FAILURE` | evidence is adequate but output formation/scoring fails |
| `IMPLEMENTATION_ERROR` | invariant, checksum, hash, non-finite, or runtime contract fails |
| `UNKNOWN` | evidence is insufficient to assign a stronger label |

The tested strict evidence unit is auditable but saturated. Endpoint v1 is
frozen and has one narrow local prospective validation, yet evidence-related
labels cannot become broad confirmatory failure rates until a materially larger
and diverse protocol is prospectively registered. The local v2 execution also
used Torch 2.13.0/Transformers 5.15.0 and has not been reproduced under the
locked 2.12.1/5.14.1 runtime used by later B-D screens.

## Assignment sequence

1. Validate generator, solver, hashes, alignment, masks, rollback, and numeric
   invariants. An implementation failure invalidates the affected comparison.
2. Check `B0`. If it fails, assign `FULL_CACHE_CAPABILITY_FAILURE`; do not infer
   a compression failure.
3. Apply frozen endpoint v1 before reviewing answer prose. Calibration outcomes
   used to design the scorer cannot enter the
   confirmatory split.
4. Compare oracle and larger retained-history controls before assigning
   `INSUFFICIENT_KV_BUDGET`.
5. Use `BASE_MODEL_REASONING_FAILURE` only when all required evidence remains
   accessible to later attention.
6. Preserve multiple supported labels as a causal chain, with the earliest
   supported failure marked primary.

## Required aggregate views

Summaries must use the registered independent cluster as the denominator and
report prompt counts separately. Report condition, family, depth, context,
retained history, active chunk, query length, and compression-boundary count;
paired B3/B2 rescue/harm; stale/distractor failures; implementation-invalid
clusters; and `UNKNOWN` without forced adjudication.

Failures are never excluded merely because they weaken a claim. Only frozen
pre-answer exclusion rules in `CONFIRMATORY_PROTOCOL.md` apply.

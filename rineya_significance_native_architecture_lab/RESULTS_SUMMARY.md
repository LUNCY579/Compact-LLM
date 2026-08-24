# Results summary

## Active evidence scope

The current model scope is the published Qwen3-0.6B Rineya family, with Halo as
the primary target. Compatibility smokes have completed on Halo, Halo
Uncensored, Lyra, Nimbus, and Pulse. Rineya/R2 and Aegis are excluded from
current evidence and conclusions. RineyaBeta remains read-only provenance.

| Workstream | Status | Evidence unit | Supported conclusion |
|---|---|---:|---|
| Audited source-primitive parity | `PASS` on 4/4 checks | 4 deterministic primitive comparisons | primitive tensors/indices agree exactly; no end-to-end parity claim |
| Published-family CPU runtime smokes | `PASS` on 5/5 cores | 1 prompt-only case/core | local runtime compatibility only |
| Halo calibration dataset v3 | `CALIBRATION_DATASET_BUILT` | 32 independent clusters | exact-evidence construction/alignment only |
| Halo full-cache capability calibration v2 | execution `PASS`; advance gate `FAILED` | 32 clusters | B0 capability is uneven; R2 family is not eligible |
| Endpoint prospective v1 | `STOPPED_AT_B0_GATE` | 24 new clusters | B0 9/24; no compression run |
| Endpoint prospective v2 + local registration | `COMPLETE` | B0 18/24; 12 registered clusters | B3-B2 endpoint `+0.0401786`, `p=0.01953125`; no answer gain |
| Halo R1-R8 ladder v1 | `TIER_A_LADDER_BUILT` | 1,176 cases / 84 clusters | dataset construction/alignment only |
| Halo R5-R8 + corrected R7 v2 | R7 v1 invalid; R7 v2 and R8 `STOP` | R7 v2 8 balanced clusters | Halo all-YES, 4/8; valid benchmark but capability gate failed; no compression |
| Halo retention pilot v2 | `EXPLORATORY_POSTHOC` | 11 B0-selected clusters | endpoint-design reuse favors B3 retention; no confirmatory claim |
| Real-Halo count callback identity | `PASS` | 1 case | exact standard-B3/callback engineering identity only |
| Halo count controls v1 | `COMPLETE_EXPLORATORY` | 88 cells / 11 clusters | age/head-average exactly equal true C in this fixed schedule; adjusted contrasts null |
| Halo eligible-pool count controls v2 | `COMPLETE_EXPLORATORY` | 33 cells / 11 clusters | two clean identity interventions; Holm-adjusted endpoint contrasts null; paired process-local evidence |
| Halo schedule control v2 | `COMPLETE_EXPLORATORY` | 110 cells / 11 clusters | positive retention interaction; answer interaction not detected |
| Halo adaptive-budget screen v1 | `NO_SIGNIFICANCE_SPECIFIC_ADVANTAGE` | 132 boundary rows / 11 clusters | shuffled-statistic beats entropy; no closed-loop or answer claim |
| Experiment B heterogeneous budget | runtime `PASS`; specificity `FAILED` | 36 closed-loop cells / 12 clusters | entropy 8/12 vs uniform 5/12 (`p=0.25`); shuffled retention higher; padded overhead |
| Experiment C attention prior | runtime `PASS`; `NO_MEASURABLE_ANSWER_GAIN` | 60 cells / 12 clusters | all 5/12; normalized endpoint equals raw/shuffled; specificity failed |
| Experiment D memory slots | runtime `PASS`; `NEGATIVE_OR_NEUTRAL` | 72 byte-matched cells / 12 clusters | all 5/12; normalized endpoint below M0/random; no recovery gain |
| Experiment E recovery | mechanism `PASS`; risk signal `NO_MEASURABLE_GAIN` | 11-cluster open-loop + 2-case smoke | risk AUC 0.50; accept/recompute mechanics verified only |
| Experiment F tiny controller | `UNDERPOWERED_OVERFIT_SCREEN_ONLY` | high overfit risk; frozen 5/3/3 split; held-out n=3 | logistic AUC/BA 1.0 on two pos-neg pairs; no Finding D or advancement |
| Halo figure collections | `PASS` | 50 figures (28 controls + 22 deep architecture) | verified visualizations only; individual claim limits apply |
| Literature/novelty audit | `NO_NOVELTY_CLAIM` | scoped primary-source audit | experimental distinction, not broad novelty |
| Broad Halo confirmatory evaluation | `DRAFT_NOT_REGISTERED`; `NOT_RUN` | 0 broad-protocol cases | local endpoint registration does not register the broader protocol |
| Halo bounded mini evaluation | `PASS` execution | 3 fixed cases | `NO_MEASURABLE_GAIN` B3 vs B2, descriptive only |
| Tier-A synthetic/open-loop run | `INVALID_OR_CONFOUNDED` | preserved run artifacts | machinery audit only |
| Deferred GPU confirmation | `NOT_RUN` | 0 | none |

Runtime status: local recorder implementing audited semantics; audited
source-primitive parity `PASS`, end-to-end source benchmark reproduction
`NOT_RUN`. Source-repository results are not pooled with ChatBubble evidence.

[`results/source_parity.json`](results/source_parity.json) records exact tensor
equality for the bottom-right causal mask, post-RoPE GQA score, and count-lag
gather, plus exact tie-free indices for sink/heavy/recent selection. Its own
claim limit is primitive parity only.

## Published-family CPU smokes

Runs:

| Core | Run | Verified weight SHA-256 |
|---|---|---|
| Halo (primary) | [`halo_cpu_smoke_20260823_v2`](results/runs/halo_cpu_smoke_20260823_v2) | `38fb96fe286776919032c4418eef0a2534e2c2c2865c1465cc9dd4cfe7e36544` |
| Halo Uncensored | [`halo_uncensored_cpu_smoke_20260823`](results/runs/halo_uncensored_cpu_smoke_20260823) | `c5d4a17f69ca932a5d4ad29e4f9ba1a29d7a137c08951fd89f61b4cb95444503` |
| Lyra | [`lyra_cpu_smoke_20260823`](results/runs/lyra_cpu_smoke_20260823) | `87c57b8eef24238cbba8bc98a38298c9c200a0fa05102b368d1c6cdaf363c5aa` |
| Nimbus | [`nimbus_cpu_smoke_20260823`](results/runs/nimbus_cpu_smoke_20260823) | `7b66b8d0ef06cf7f71bc9001b0c9a330d5f8a0eec1d11eb173ece70ca66ef49c` |
| Pulse | [`pulse_cpu_smoke_20260823`](results/runs/pulse_cpu_smoke_20260823) | `2357574db5557c288f57a6b97b2761830ea265baf3c6a1c0270357a7d0eaa322` |

All expected weight, tokenizer, and config hashes were verified. Halo, Halo
Uncensored, Lyra, and Pulse used tokenizer hash
`be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506`
and config hash
`2284cc1687a4f4f6e3a64a13dfcf7562e592b951afa2f91244cc7b212c9cfb85`.
Nimbus used tokenizer hash
`aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`
and config hash
`660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd`.

Each run had the same scope:

| Field | Measured value |
|---|---:|
| Cores | five published 28L Qwen3-0.6B family checkpoints |
| Architecture | Qwen3-0.6B-derived, 28L, 1024 hidden, 16 query / 8 KV heads |
| Runtime | CPU, bf16, local SDPA recorder |
| Prompt-only cases | 1 per core |
| True context tokens | 192 |
| Retained history + active chunk | B48+C64 |
| Stored after final chunk | 112 entries |
| Transient query | 10 tokens used; 32 configured maximum |
| Answer generation | no |
| Disabled-path max / mean absolute logit delta | 0.0 / 0.0 |
| Disabled-path argmax agreement | 1.0 |
| Query-conditioned scoring events | 2 per layer, per core |
| Aggregate layer-token evictions | 2,240 |
| KV bytes | 12,845,056 |
| S+C bytes | 200,704 |
| Position/ID metadata bytes | 401,408 |
| Total persistent bytes | 13,447,168 |

The disabled chunked and single-shot paths produced exactly identical recorded
logits on all five cases. Query-conditioned state also survived exact rollback
and aligned compression in each. These are one-case-per-core implementation
checks, not proof of end-to-end source-runtime parity.

The budget is reported as **retained history + active chunk + transient query**.
Each run's final `112` stored states are `48 + 64`; the rolled-back query is transient.
Every manifest is `matched=false`. No smoke decoded an answer or measured
accuracy, evidence survival, long-context reasoning, throughput, deployment
latency, or GPU behavior.

## Halo calibration dataset and B0 capability

Dataset: [`halo_calibration_v3`](results/benchmarks/halo_calibration_v3)

The dataset contains 32 independent clusters, eight each in benchmark families
R1-R4, at evidence depth 0.50. Each case has a clean inference view and a
separate evaluation view containing solver outputs and exact rendered-prompt
evidence-token positions. The manifest explicitly limits this artifact to
dataset construction and alignment; it contains no model result.

Run: [`halo_capability_calibration_v2`](results/runs/halo_capability_calibration_v2)

| Benchmark family | B0 correct / clusters |
|---|---:|
| R1 | 7/8 |
| R2 | 0/8 |
| R3 | 3/8 |
| R4 | 8/8 |
| **Overall** | **18/32 (0.5625)** |

The advance rule was at least 12 correct overall and at least two per family.
The overall threshold passed, but the gate as a whole failed because R2 was
0/8. R2 here denotes a benchmark family, not a Rineya/R2 model. The result
prevents the 32 clusters from supporting a balanced compression-conditioned
claim and motivates prospective task-family repair or replacement.

## Exploratory Halo retention pilot

Run: [`halo_retention_pilot_v2`](results/runs/halo_retention_pilot_v2)

The pilot selected 11 B0-correct calibration clusters from R1, R3, and R4,
capped at four per family, then ran matched 128-entry retained-history B2, B3,
and B4 conditions. Selection on B0 and omission of the R2 family make all
estimates development-only.

| Condition | Correct | Strict all-token evidence survival |
|---|---:|---:|
| B2 raw accumulation | 5/11 | 0/11 |
| B3 observation normalization | 6/11 | 0/11 |
| B4 sink-plus-recent position | 1/11 | 0/11 |

The strict endpoint required every required fact token to remain available in
every evaluated layer. It was all-zero across all three conditions and is
therefore retained only as a strict diagnostic. The all-zero result must not be
described as evidence that all policies retain no useful information.

[`layer-fact-retention-v1`](EVIDENCE_ENDPOINT_CALIBRATION.md) was then selected
without reading condition labels or answer outcomes. It averages atomic
complete-fact indicators over layer/fact cells, passed its non-saturation
screen, and is frozen for future evaluation. Reapplying it to its design data
gave B2 `0.147727`, B3 `0.209416`, and B4 `0.000000`; B3-B2 was `+0.061688`
(95% interval `[0.030844, 0.094156]`, exact sign-flip `p=0.0078125`). Because
the same pilot supplied endpoint-design data, this is post hoc and not an
independent validation. B3 had one answer win and no answer loss versus B2, but
the exact two-sided answer test was `p=1.0`.

## Prospective endpoint validation

The first independent attempt,
[`halo_endpoint_validation_capability_v1`](results/runs/halo_endpoint_validation_capability_v1),
kept endpoint v1 frozen but failed its B0 gate: Halo solved `9/24` (`R1=1/8`,
`R3=1/8`, `R4=7/8`) against a rule requiring at least 12 overall and at least
three per family. It stopped before B2/B3/B4, as required.

Difficulty-calibrated v2 modified the task templates before compressed outcomes
and reran the same 24-cluster B0 structure. Halo solved `18/24` (`R1=6/8`,
`R3=4/8`, `R4=8/8`), passing the gate. Twelve B0-correct clusters—four per
family—were then frozen in the local registration
[`halo_endpoint_validation_v2.json`](results/registrations/halo_endpoint_validation_v2.json)
before any compressed cell ran.

Run: [`halo_endpoint_validation_v2_registered`](results/runs/halo_endpoint_validation_v2_registered)

| Condition | Correct | Mean endpoint v1 |
|---|---:|---:|
| B2 raw | 6/12 | 0.172619 |
| B3 normalized | 5/12 | 0.212798 |
| B4 positional | 1/12 | 0.000000 |

The sole registered primary contrast, B3 minus B2 endpoint, was `+0.0401786`
with a 95% cluster-bootstrap interval `[0.0133929, 0.0758929]` and exact
two-sided cluster sign-flip `p=0.01953125`. This is a prospective retained-state
separation in the locally registered sample. It is not an answer gain: B3 lost
one cluster and won none versus B2 (`5/12` vs `6/12`, `p=1.0`). The strict
diagnostic remained `0/12` in every condition. Difficulty tuning after the v1
capability failure, the 12-cluster scope, and local rather than public
registration limit the result; it is not a broad confirmatory or population
claim. The internally paired run used Python 3.11.15, Torch 2.13.0+cpu, and
Transformers 5.15.0. It has not been reproduced under the current
Torch 2.12.1+cu126 / Transformers 5.14.1 lock used by B-D.

## R1-R8 ladder and bounded R5-R8 capability

Dataset: [`halo_r1_r8_ladder_v1`](results/benchmarks/halo_r1_r8_ladder_v1)

The controlled ladder contains 1,176 cases in 84 independent latent clusters.
R1 contributes 490 cases across five required-fact counts; R2-R8 each
contribute 98. It covers seven clustered evidence depths plus seven fixed
layouts, preserves inference/evaluation separation, and exact-aligns evidence
to Halo tokenizer positions. No model outcome is attached to the ladder as a
whole.

Run: [`halo_r5_r8_capability_screen_v1`](results/runs/halo_r5_r8_capability_screen_v1)

| Benchmark family | B0 correct / clusters |
|---|---:|
| R5 | 2/2 |
| R6 | 1/2 |
| R7 | 2/2 raw; `INVALID_UNINFORMATIVE` |
| R8 | 0/2 |
| **Raw artifact** | **5/8 (0.625)** |
| **Currently valid R5/R6/R8 subset** | **3/6 (0.500)** |

This was a full-cache screen only. R7 v1 is not capability evidence: the seven
ladder seeds are 6 NO / 1 YES, both screened seeds are NO, an exact rendered
lock sentence exposes a solver shortcut, and all seven facts were marked
required. R8 stops at capability and cannot support a compression-conditioned
result from these clusters. `R2` elsewhere in this ladder is a benchmark label,
never an in-scope Rineya/R2 checkpoint.

The corrected [`halo_r7_v2`](results/benchmarks/halo_r7_v2) benchmark contains
80 balanced cases over eight semantic clusters and passes independent rendered-
solver, necessity, and counterfactual audits. Its frozen B0 screen used one
case per cluster, balanced `4 YES / 4 NO`. Halo returned `YES` on all eight and
scored `4/8`, equal to the always-YES baseline. Although the overall minimum
was met, the prospective gate also required at least one correct case on each
answer side; the `0/4` NO-side result stopped compression. R7 v2 is now a valid
benchmark with a Halo capability failure, not a Significance Matrix quality
result.

## Count causal controls

Run: [`halo_count_causal_controls_v1`](results/runs/halo_count_causal_controls_v1)

The prerequisite one-case
[`halo_count_control_identity.json`](results/halo_count_control_identity.json)
check found exact generated token IDs, resident positions, accounting, policy,
and exponent between the standard B3 compressor and the true-count callback.
It validates the control hook on that real-Halo case only.

All 88 planned cells completed: eight count transforms across the same 11
B0-selected R1/R3/R4 clusters. The 128-entry history budget and query schedule
were matched. Age-derived and head-averaged count transforms were exact
trajectory equivalents of true realized C in every cluster: effective and true
counts, scores, boundary selections, final residents, predictions, endpoint,
and correctness all matched exactly. This result is specific to fixed
128-token chunks with one query-scoring event at every non-final boundary.

| Count transform vs true C | Frozen-endpoint delta | Nominal p | Holm p | Interpretation |
|---|---:|---:|---:|---|
| age-derived | 0.000000 | 1.0 | 1.0 | exact trajectory equivalence |
| head-averaged | 0.000000 | 1.0 | 1.0 | exact trajectory equivalence |
| constant | -0.055195 | 0.015625 | 0.109375 | nominal only; adjusted null |
| shuffled | -0.058442 | 0.046875 | 0.28125 | adjusted null; eligible-pool confounded |
| wrong-token | -0.009740 | 0.5 | 1.0 | no detected contrast; eligible-pool confounded |
| noisy | +0.006494 | 0.3125 | 1.0 | no detected endpoint contrast |
| delayed | +0.063312 | 0.1171875 | 0.5859375 | descriptive only |

Answer deltas were not detected (`p=1.0` for every count comparison). The
264-row boundary-evidence artifact scores only facts fully observed by the
boundary; future and boundary-crossing facts are not called evicted. These are
exploratory, multiplicity-aware development results, not proof that count state
is generally useless or beneficial. Shuffled and wrong-token transforms
preserved the global C distribution but moved values across four protected sink
and 32 protected recent entries; the eligible-heavy-pool C multiset differed in
224/224 layer-head cells on the audited case. They are confounded controls and
cannot isolate entry identity.

Run: [`halo_count_causal_controls_v2`](results/runs/halo_count_causal_controls_v2)

V2 replaces the confounded global permutations with two clean eligible-heavy-
pool interventions. Protected sink/recent denominators are immutable;
eligible C and semantic-ID multisets are independently preserved per layer/KV
head; and every intervention has zero semantic fixed points. All 22,176 audited
layer/head cells passed. The 33 cells comprise true C, eligible-shuffled, and
eligible-circular controls on the same 11 development clusters.

| Clean count control vs true C | Endpoint delta | Nominal p | Holm p | Answer delta / p |
|---|---:|---:|---:|---:|
| eligible-shuffled | -0.021104 | 0.4921875 | 0.4921875 | -3/11 / 0.25 |
| eligible-circular | -0.014610 | 0.125 | 0.25 | -1/11 / 1.0 |

Both primary retention contrasts are null after the frozen two-control Holm
family. A same-process standard/true-count replay was exact. By contrast,
separately launched v1 and v2 CPU-bf16/SDPA reference processes were not
bitwise reproducible: 10/11 predictions, 9/11 endpoint values, and 0/11 final
resident trajectories matched exactly. The cause is not established, so v2's
paired conditions are interpreted only within their common process; no cross-
run trajectory-reproduction claim is made.

## Held-constant compression schedule

Run: [`halo_schedule_control_v2`](results/runs/halo_schedule_control_v2)

The run completed 110 cells—11 clusters x five schedules x two methods—and 55
matched B3-B2 method pairs. Every pair held the final pre-decode prompt-cache
size and query access equal while varying realized compression rounds with a fixed 64-token
terminal active suffix and 128 retained-history entries.

| Cluster-level interaction | Slope per additional round | 95% interval | p |
|---|---:|---:|---:|
| Frozen layer-fact endpoint, B3-B2 | +0.00936074 | [0.00310399, 0.01759850] | 0.00390625 |
| Answer correctness, B3-B2 | +0.02248157 | [-0.03599509, 0.07641278] | 0.4375 |

The retention interaction is positive in this selected development sample;
the answer interaction is not detected. Repeated schedule rows are not treated
as independent replicates. Calibration reuse, B0 selection, eleven clusters,
and the unregistered protocol prevent a confirmatory or population claim.

## Adaptive budget allocation

Run: [`halo_adaptive_budget_screen_v1`](results/runs/halo_adaptive_budget_screen_v1)

Experiment B is now screened at recorded boundaries. Four allocation strategies
cross 11 clusters and three boundaries for 132 rows while conserving an exact
28,672-entry global layer/head budget. Uniform replay matched runtime selection
in all 33 boundary checks.

| Primary-boundary strategy minus uniform | Mean delta | Nominal p | Holm p |
|---|---:|---:|---:|
| entropy-proportional | +0.055195 | 0.107422 | 0.214844 |
| inverse-concentration | +0.025974 | 0.345703 | 0.345703 |
| shuffled-statistic | +0.115260 | 0.014648 | 0.043945 |

The shuffled-statistic control exceeded the significance-aligned entropy
allocation, so the specificity gate failed and the decision is
`NO_SIGNIFICANCE_SPECIFIC_ADVANTAGE`. This is boundary-local open-loop replay:
no model was loaded, no answers were decoded, earlier allocation changes were
not propagated through later hidden states, and sparse physical storage
overhead was not measured.

## Closed-loop Experiment B: heterogeneous budgets

The padded heterogeneous runtime passed a same-process uniform identity gate,
exact global logical-budget conservation, GQA validity masking, query rollback,
decode append, and byte-ledger checks. It is semantically heterogeneous but not
a truly ragged storage kernel.

Run: [`halo_heterogeneous_budget_quality_screen_v1`](results/runs/halo_heterogeneous_budget_quality_screen_v1)

| Condition | Correct | Mean endpoint | Mean post-decode padding overhead |
|---|---:|---:|---:|
| uniform | 5/12 | 0.21131 | 0 |
| entropy | 8/12 | 0.22321 | 2,403,854 B |
| shuffled statistic | 5/12 | 0.23810 | 3,121,756 B |

Entropy minus uniform correctness was `+0.25`, interval `[0, 0.5]`, exact
`p=0.25`. Its endpoint contrast was `+0.01190`, interval
`[-0.01637, 0.03869]`, `p=0.4834`. Shuffled allocation had the highest mean
retention and entropy minus shuffled was `-0.01488` (`p=0.2988`), so mechanism
specificity failed. Logical prefill payload was exactly matched; padded
physical storage was about 9.7% larger for entropy and 12.5% larger for
shuffled than the respective logical payload. The result is 36 exploratory,
unregistered cells under Transformers 5.14.1/Torch 2.12.1; the earlier local
endpoint run used 5.15.0/2.13.0, so only the within-run uniform comparator is
used.

## Experiment C: soft attention prior

The compatibility smoke established exact `lambda=0` identity and bounded,
finite, history-only nonzero bias. The 60-cell quality screen used 12 frozen
clusters and five conditions: no prior, normalized `S/C`, raw `S`, shuffled
normalized significance, and age-only. Every condition scored `5/12`, with
text-identical predictions across all conditions. Every nonzero comparison was
active—logits trajectories and final cache states changed in 12/12 clusters.

Normalized, raw, and shuffled each had mean endpoint `0.21577`; no-prior was
`0.21131` and age `0.21280`. Normalized minus no-prior was `+0.00446`
(`p=0.5`), while normalized minus raw and shuffled was zero at aggregate scale
(`p=1.0`). The specificity rule failed. Small attention-entropy shifts are
mechanistic diagnostics only and did not translate into answers. Classification:
`NO_MEASURABLE_ANSWER_GAIN / SPECIFICITY_NOT_SUPPORTED`.

## Experiment D: significance memory slots

The memory runtime passed exact `M=0` hard-eviction identity and a finite
published-Halo `M=4` smoke with 491,904 bytes of slot state. The quality screen
ran all 72 cells: 12 clusters times M0 plus five deterministic M4 controls.
Total capacity used B128 for M0 and B123 plus four slots/head for M4, leaving
108,416 bytes of unavoidable integer slack below M0.

Every condition scored the same five clusters (`5/12`). Mean endpoint was M0
`0.211310`, mean `0.184524`, recency/raw `0.186012`, normalized `0.183036`, and
random `0.191964`. Normalized beat no simpler M4 control and was below both M0
and random. Every M4 arm loses exact-token endpoint credit partly because it
replaces five exact ordinary positions with blended slots; the answer outcome
still showed no recovery gain. Classification:
`EXPLORATORY_LOCAL_NOT_REGISTERED / NEGATIVE_OR_NEUTRAL`.

## Experiments E and F: recovery and tiny controller

Experiment E's deterministic answer-blind six-feature risk signal had
compressed-error ROC AUC `0.50` on 11 B0-selected clusters. It therefore
supports no risk-ranking quality claim. The live two-case smoke did establish
the mechanism: accept invokes no recomputation and preserves the compressed
output literally; recover discards the compressed cache, restores all input
tokens at positions from zero, recomputes once, and matches the prior full-
cache prediction.

Experiment F reused the frozen pre-answer feature view and committed a SHA-256
whole-cluster 5/3/3 train/calibration/evaluation split with both classes in
every partition and no salt retry. A seven-parameter L2 logistic model achieved
held-out AUC `1.0`, balanced accuracy `1.0`, selective coverage `1/3`, risk
`0`, and counterfactual system accuracy `3/3` with two recoveries and a
recorded wall multiplier `1.172x`. Equal-weight and hash-budget baselines had
AUC `0`/balanced accuracy `0.25`; a depth-two tree had AUC/balanced accuracy
`0.5` and visibly overfit.

The held-out sample has only two positive-negative pairs (two errors and one
success), training/calibration have only five/three clusters, and one feature
is constant in training. Classification is
`UNDERPOWERED_OVERFIT_SCREEN_ONLY` with high overfit risk. Finding D, generalization, a
deployable threshold, and advancement are not supported.

## Figures and literature boundary

[`halo_controls_v1`](results/figures/halo_controls_v1) contains 28 PNG figures,
and [`halo_deep_architecture_v1`](results/figures/halo_deep_architecture_v1)
adds 22 panels for the locally registered endpoint, count v2, R7 v2, and
Experiments B-F. Both manifests are `PASS`; all 50 plots have source-data
exports, hashes, and per-panel claim limits. They summarize existing evidence
and add no independent evidence.

The scoped [literature/novelty audit](LITERATURE_NOVELTY_AUDIT.md) finds direct
prior art for the broad surrounding mechanisms. Its verdict is
`NO_NOVELTY_CLAIM`. The defensible current object is the behavior of carried
realized-exposure state under matched count and schedule controls, not a claim
that cumulative-attention KV retention or related architectures were invented
here.

## Bounded exploratory Halo mini evaluation

Run: [`halo_mini_eval_20260823_v2`](results/runs/halo_mini_eval_20260823_v2)

The verified Halo core ran three fixed early/middle/late demo cases on CPU in
bf16 with manual greedy decoding (`max_new_tokens=16`). Inference completed and
unscored predictions were written before the answer key was opened.

| Condition | Correct / cases | Exact accuracy | Correct / B0-eligible cases |
|---|---:|---:|---:|
| B0 full cache | 1/3 | 0.3333 | not applicable |
| B2 raw accumulation | 1/3 | 0.3333 | 1/1 |
| B3 observation normalization | 1/3 | 0.3333 | 1/1 |

The B0 capability gate failed for two of three cases, leaving one eligible case
for compression-conditioned correctness. B3 had no measurable gain over B2:
both were correct on the same sole eligible case and both scored 1/3 overall.
This is not evidence of equivalence. The run intentionally reports no
confidence interval, hypothesis test, or population estimate, and it did not
score the pending exact evidence metric. B0 is not memory matched; B2 and B3
share the same retained-history entry allowance. Timing is local accounting,
not a speed comparison or deployment estimate.

## Invalidated Tier-A run

[`tier_a_20260823`](results/runs/tier_a_20260823) is a completed open-loop,
model-free artifact containing deterministic generators, traces, replay output,
statistics, and figures. Its recorded configuration does not match the executed
hard-coded trace geometry/budget, and its three round settings are not
exchangeable replications. It is `INVALID_OR_CONFOUNDED` for scientific
inference; prior B2/B3 contrasts are not active conclusions. No published
Rineya core was loaded in that run.

## Architecture decision matrix

| Proposal | Halo quality | Evidence survival | Engineering status | Decision |
|---|---|---|---|---|
| Significance-native retention | locally registered B3 5/12 vs B2 6/12, `p=1.0` | prospective B3-B2 `+0.0401786`, `p=0.01953125` | 5/5 compatibility; count/schedule controls complete | narrow local retained-state support; no answer or broad confirmatory claim |
| Adaptive layer/head budgets | entropy 8/12 vs uniform 5/12, `p=0.25` | shuffled retention exceeded entropy | padded closed-loop runtime and 36-cell screen complete | specificity failed; no ragged-memory or quality claim |
| Soft attention prior | all five arms 5/12 | normalized equals raw/shuffled endpoint | zero-bias identity + 60 cells | `NO_MEASURABLE_ANSWER_GAIN`; specificity failed |
| Significance memory slots | all six arms 5/12 | normalized below M0/random endpoint | M0 identity + 72 byte-matched cells | `NEGATIVE_OR_NEUTRAL` for tested M4 design |
| Controlled recovery | risk AUC 0.50 | not an evidence-retention intervention | accept/recompute branches verified | mechanism only; fixed risk signal failed |
| Tiny learned controller | held-out 3/3 counterfactual at 1/3 coverage | uses pre-answer state diagnostics | frozen 5/3/3 split; 7 parameters | `UNDERPOWERED_OVERFIT_SCREEN_ONLY`; high overfit risk; no advancement |

The strongest positive result is now the narrow, locally registered endpoint
separation. The broad protocol still needs a materially larger prospectively
split sample, family/depth diversity, a frozen difficulty contract, and an
answer/mediation plan. R7 v2 and R8 remain stopped at B0; benchmark-family R2
also remains unresolved. No answer-improvement, deployment, novelty,
family-wide quality, or broad confirmatory conclusion is available.

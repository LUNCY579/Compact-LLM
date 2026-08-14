# Results and Comparative Analysis

## Evidence status

All frozen analyses are complete. The 4,050-example DEV cache covers 18,760
gold-prefix token decisions and all 127 nonempty expert subsets; the selected
free-running DEV, committee, adaptive, calibrated v2.2 TEST, and fixed-method
v2.1.2 panels are complete and source-bound. The calibrated TEST condition was
frozen before its output existed; its hard/raw anchors were already known and
the archived n=32 exposure caveat remains.

## Primary v2.2 TEST benchmark

Source: E-CORNERSTONE-RAW / E-CORNERSTONE-STATS. Exact accuracy is primary.
Latency and cost are descriptive hardware measurements from the same raw rows.

| Method | N | Exact acc. | Answerable | Abstention | Mean tokens | Mean latency | Mean expert forwards |
|---|---:|---:|---:|---:|---:|---:|---:|
| Backbone K=1 | 4,050 | 0.025% | 0.028% | 0.000% | 6.96 | 0.304 s | 6.96 |
| Track A K=1 | 4,050 | 14.593% | 16.068% | 5.000% | 4.03 | 0.151 s | 4.03 |
| Abstention K=1 | 4,050 | 13.012% | 0.000% | 97.593% | 10.92 | 0.402 s | 10.92 |
| Probability K=7 | 4,050 | 14.099% | 7.949% | 54.074% | 7.50 | 1.873 s | 52.47 |
| **Hard K=7** | 4,050 | **17.259%** | 12.593% | 47.593% | 5.73 | 1.432 s | 40.11 |

Bold denotes the highest exact accuracy in this frozen comparison, not the best
latency or a universal deployment optimum.

## Paired global comparisons

| Comparison | Difference | Rescue | Harm | 95% paired bootstrap CI | Exact McNemar p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| Hard K=7 − Backbone | +17.235 pp | 699 | 1 | [16.049, 18.395] pp | (2.67\times10^{-208}) | (1.87\times10^{-207}) |
| Hard K=7 − Track A | +2.667 pp | 328 | 220 | [1.531, 3.778] pp | (4.54\times10^{-6}) | (1.36\times10^{-5}) |
| Hard K=7 − Abstention | +4.247 pp | 444 | 272 | [2.963, 5.531] pp | (1.36\times10^{-10}) | (5.42\times10^{-10}) |
| Hard K=7 − Probability K=7 | +3.160 pp | 226 | 98 | [2.321, 4.025] pp | (8.72\times10^{-13}) | (4.36\times10^{-12}) |

The full comparison family and exact stored values are in
`results/summaries/sealed_statistics.json`. The difference between the table's
independent 100,000-resample verification intervals and stored 20,000-resample
intervals is bounded by the audit tolerance.

### Dependence sensitivity

The confirmatory tests above describe the evaluated finite corpus under
example-level pairing. They do not imply 4,050 independent latent tasks. When
entire reasoning families are resampled, the 95% cluster-bootstrap intervals
are [−4.52, +13.70] pp for Hard−Track A and [−1.63, +6.89] pp for
Hard−Probability. Under 114 finer heuristic structural families, the intervals
are [−2.62, +9.26] pp and [+0.33, +5.80] pp, respectively. The finite-corpus
ranking is unchanged; generalization to unseen task/template families is more
uncertain, especially versus Track A.

## What the aggregate ranking hides

Hard K=7 versus Track A:

- answerable: 12.593% versus 16.068% (**−3.476 pp**);
- abstention: 47.593% versus 5.000% (**+42.593 pp**).

Hard K=7 versus probability mean:

- answerable: 12.593% versus 7.949% (**+4.644 pp**);
- abstention: 47.593% versus 54.074% (**−6.481 pp**).

Thus the hard committee is not simply a better reasoning model or a better
abstention model. Relative to Track A it purchases abstention behavior at some
answerable cost; relative to probability mean it recovers answerable behavior
while giving back some abstention accuracy. Leave-one-out and calibrated
free-running tests are required to identify which expert/aggregation choices
cause this frontier.

## Pilot versus full corpus

| Method | n=32 pilot | N=4,050 full TEST |
|---|---:|---:|
| Backbone | 0.000% | 0.025% |
| Track A | 15.625% | 14.593% |
| Abstention | 50.000% | 13.012% |
| Probability K=7 | 37.500% | 14.099% |
| Hard K=7 | 25.000% | 17.259% |

The reversal is a sample-size and benchmark-stratification result. The pilot is
preserved as a feasibility experiment and must not be described as incompetent
or deceptive.

## Free-running hard/soft trace comparison

- exact complete-sequence matches: 2,246/4,050 (55.46%);
- divergent sequences: 1,804/4,050 (44.54%);
- first divergence at (t=0): 1,502/1,804 (83.26%);
- first divergence at (t\le2): 1,781/1,804 (98.73%);
- shared-prefix expert top-token mismatches between reruns: 0;
- at first divergence, a hard-dissenting expert has higher top-1 confidence
  than every hard-winner supporter in 1,645/1,804 cases.

The last statistic is compatible with confidence-magnitude conflict but is not
proof that the dissenting expert caused the probability winner. Full candidate
probabilities and DEV interventions are needed.

## Coalition description

The Hard-K7 trace contains 23,204 token decisions and 107 observed winner-
support masks. The most frequent masks are:

| Mask | Members | Token share | Sequence-correct association |
|---|---|---:|---:|
| `1111111` | all seven | 35.05% | 20.30% |
| `1111100` | Track A, Track B, General, Grounding, Procedure | 11.87% | 16.49% |
| `1110011` | Track A, Track B, General, Abstention, Social | 10.64% | 24.86% |
| `1111000` | Track A, Track B, General, Grounding | 4.18% | 11.95% |
| `1110000` | Track A, Track B, General | 3.86% | 41.41% |

Sequence-correct association assigns a final-output label to each token and is
not local coalition accuracy. Coalition causality requires counterfactual
committee evaluation. The complete supplement retains all 107 observed masks
and stratifies their 23,204 tokens by reasoning family, difficulty, answer
class, domain, token-position bucket, EOS status, and final sequence
success/failure. These cuts close the descriptive inventory but do not make
frequency or sequence association causal.

## Secondary semantic audit

Source: E-SEMANTIC. The official exact score remains primary. We manually
reviewed a deterministic, task-stratified sample of 100 full-K=7 exact-match
failures: 50 hard-vote outputs and 50 probability-mean outputs. None was a
semantically correct answer hidden by formatting or paraphrase (combined
0/100; exact 95% CI 0.00%-3.62%; each method 0/50, upper bound 7.11%). The
combined reviewed errors were 30 wrong-content answers, 38 false abstentions,
16 failures to abstain, and 16 malformed outputs. This does not relabel the
full benchmark, but it gives no evidence that benign exact-match artifacts
explain the hard/probability difference. The audit used one reviewer, so no
inter-rater reliability estimate is available.

## Full teacher-forced DEV mechanism study

### Expert calibration

| Expert | Fitted T | Top-1 token acc. | Mean confidence | Confidence gap | ECE-15 |
|---|---:|---:|---:|---:|---:|
| Track A | 1.00 | 80.01% | 82.94% | +2.93 pp | 0.0422 |
| Track B | 1.00 | 78.74% | 75.30% | -3.44 pp | 0.0438 |
| General | 1.00 | 76.28% | 77.78% | +1.50 pp | 0.0250 |
| Grounding | 1.25 | 59.59% | 73.18% | +13.59 pp | 0.1388 |
| Procedure | 1.25 | 54.08% | 68.55% | +14.47 pp | 0.1447 |
| Abstention | 2.00 | 45.72% | 72.96% | +27.24 pp | 0.2724 |
| Social | 1.50 | 41.84% | 50.98% | +9.13 pp | 0.0913 |

The abstention adapter is the clearest calibration outlier. These are token
calibration measurements on DEV gold prefixes, unlike the directional
free-running confidence profiles reported earlier.

Conditioning descriptively on token correctness reveals a large confidence
separation for every expert. Mean top-1 confidence on correct versus incorrect
tokens is 94.09%/38.28% for Track A, 87.60%/29.75% for Track B,
92.13%/31.62% for General, 87.59%/51.93% for Grounding,
87.71%/45.99% for Procedure, 97.31%/52.45% for Abstention, and
88.52%/23.96% for Social. These are dependent gold-prefix token summaries, not
sequence-level effects or a causal calibration test.

### Teacher-forced K=7 aggregation

| Rule | Token acc. | All-token example acc. | Candidate certificate |
|---|---:|---:|---:|
| Hard plurality | 78.33% | 25.26% | 100% |
| Raw probability mean | 78.91% | 26.62% | 100% |
| Per-expert calibrated probability | **79.23%** | **26.77%** | 100% |
| Shared NLL-optimal temperature | 78.24% | 23.85% | 100% |
| Logit mean | 73.07% | 17.51% | 99.9947% |
| Normalized logit mean | 73.33% | 17.80% | 100% |
| Top-2 approval | 73.01% | 13.98% | 100% |
| Borda top-5 | 72.77% | 11.31% | 100% |
| Reciprocal rank top-5 | 76.34% | 18.47% | 100% |

Hard voting is not the strongest local gold-prefix rule. This falsifies the
simple account that its free-running TEST advantage follows from uniformly
better local token decisions. Rank-only rules also fail to reproduce hard
voting, showing that top-1 support counts plus their autoregressive use are not
interchangeable with generic rank aggregation.

### Entropy normalization and confidence clipping

Source: E-ROBUST-SOFT. Two additional full-K=7 cache-only robustness families
were evaluated without changing the frozen free-running selection. Inverse-
entropy expert weighting performs poorly (74.43% token, 8.25% all-token
example). Capping each expert's top-1 probability and exactly renormalizing has
little effect on raw probability pooling: caps 0.5, 0.7, and 0.9 yield 26.44%,
26.64%, and 26.62% all-token example accuracy, versus 26.62% without clipping.
The cap-0.7 difference from raw is only +0.025 pp (2 raw-to-clipped rescues, 1
harm; exploratory exact McNemar p=1). Every winner is certified against the
omitted vocabulary. Thus neither this entropy normalization nor bounded
confidence clipping explains a local hard-vote advantage; indeed raw soft
fusion already exceeds hard locally. These post-DEV controls are exploratory
and were not promoted to free-running confirmation.

Three further named robust controls were evaluated under the same post-DEV
boundary. Validation-weighted hard voting reaches 79.10% token accuracy but
only 24.77% all-token example accuracy; weighting proposals therefore does not
improve the hard example criterion. Dropping each candidate's largest and
smallest expert probability yields 78.89% token and 26.20% example accuracy,
slightly below raw pooling. Median probability pooling performs poorly at
73.67%/13.73%; moreover, 2/18,760 median decisions cannot be certified against
the omitted vocabulary and that row is explicitly a top-64-union
approximation. Every validation-weighted, trimmed, entropy-weighted, clipped,
hard, and raw winner is fully certified. None of these controls was promoted.

### Committee-size scaling

For hard voting, mean all-token example accuracy over all subsets is 13.92%,
13.72%, 17.09%, 18.22%, 19.39%, 22.22%, and 25.26% for K=1 through K=7.
The best subset peaks at K=4 (27.31%; Track A, Track B, Procedure,
Abstention), above full K=7. This is gradual, nonmonotonic scaling—not evidence
of a K=7 phase transition. The all-127 diagnostic supplement adds descriptive
example-level intervals to both mean-at-K and best-at-K. The latter are
conditional on choosing the best committee on the same DEV outcomes and are
not selection-adjusted confirmatory intervals. For full hard K=7, answerable
and abstention all-token example accuracy are 24.16% and 32.41%, mean proposal
vote entropy is 0.7154 nats, and the primary tie rate is 5.85%. Cached
counterfactual subsets have forward proxies but no imputed latency or VRAM.

### Leave-one-out and Shapley

| Expert | Hard full-minus-K6 | Exact hard Shapley |
|---|---:|---:|
| Track A | +8.27 pp | +10.23 pp |
| Track B | +5.90 pp | +8.26 pp |
| General | +4.47 pp | +6.52 pp |
| Abstention | +2.35 pp | +1.95 pp |
| Social | +1.56 pp | -1.32 pp |
| Procedure | -0.12 pp | +0.64 pp |
| Grounding | -1.16 pp | -1.02 pp |

Leave-one-out is context-specific; Shapley averages over every predecessor
coalition. Their disagreement for Social and Procedure is therefore evidence
of interaction, not a numeric inconsistency.

### Complementarity and diversity

The dynamic gold-prefix oracle reaches 86.36% token accuracy and 48.12%
all-token example accuracy, a +6.35 pp token gain over the best individual.
This proves there is complementary *potential*. It does not prove the voting
rule extracts all of it. At every fixed K=2 through K=6, mean pairwise top-1
disagreement is negatively correlated with hard all-token example accuracy
(Pearson r from -0.56 to -0.78). Generic diversity is therefore not sufficient;
expert quality and the structure of errors matter.

The extended 21-pair audit reaches the same boundary using distributional and
binary-error measures. Along the full-TEST hard trajectory, the largest mean
pairwise JS is Procedure/Abstention (0.4048), followed by
Grounding/Abstention (0.3788); these highly divergent pairs also have high DEV
top-1 disagreement (0.6719 and 0.6498) but substantial double-fault rates
(0.3133 and 0.2832). Correctness mutual information and success/failure
Jaccard values are retained for every pair. Thus high JS or disagreement does
not by itself identify useful complementarity; competence must be considered.

### Temperature interventions

Hard top-1 proposals, vote counts, and unique-plurality winners are invariant
to every positive temperature intervention. The confidence-sum tie-break is
not invariant to unequal expert scaling: plurality ties occur at 1,097/18,760
positions, and applying all fitted per-expert temperatures to the tie-break
changes 259 local decisions (23.61% of tied positions), lowering token/example
accuracy from 78.33%/25.26% to 78.18%/24.54%. A shared temperature changes zero
hard decisions. Probability fusion is sensitive beyond this tie boundary. With all experts jointly rescaled, all-token
example accuracy ranges from 26.91% at T=0.8 to 9.09% at T=2.0. Holding six
experts fixed and softening Grounding improves probability example accuracy
from 26.62% at T=1 to 28.69% at T=3; softening Abstention to T=1.5-2 improves
it to 26.91%. These interventions establish local calibration sensitivity,
but not yet sufficiency for free-running performance.

### Tie and EOS counterfactuals

The current confidence-sum tie-break occurs at 1,097/18,760 DEV gold-prefix
token decisions (5.85%). Replacing it with smallest-token-ID changes 404 local
decisions, improves token accuracy by 0.09 pp, but lowers all-token example
accuracy by 0.25 pp. A Track-A-priority tie rule changes 675 decisions and
improves token/example accuracy by +0.46/+0.05 pp. The negligible example-level
effect provides no basis for claiming that the existing tie-break explains the
full sequence gain or for promoting another expensive tie run.

Ordinary EOS is strongly favored locally. Strict-majority EOS lowers all-token
example accuracy from 25.26% to 15.58%; mean-EOS-probability ≥0.5 lowers it to
12.47%; and minimum three tokens lowers it to 10.12%. These gold-prefix
counterfactuals do not reproduce free-running path effects, but all three
alternatives fail the DEV promotion gate.

## Final long-run results

### Teacher-forced versus free-running

On the same 4,050 DEV examples, the teacher-forced all-token-example and
free-running sequence accuracies are nearly identical:

| Method | Teacher-forced | Free-running | Free − teacher (pp) |
|---|---:|---:|---:|
| Hard plurality | 25.259% | 25.309% | +0.049 |
| Raw probability mean | 26.617% | 26.593% | −0.024 |
| Calibrated probability mean | 26.765% | 26.790% | +0.025 |
| Reciprocal-rank top-5 | 18.469% | 18.321% | −0.148 |

Calibrated probability beats hard by +1.481 pp (95% CI +0.741 to +2.222;
Holm p=2.773e-4), raw probability beats hard by +1.284 pp (95% CI +0.420 to
+2.148; Holm p=0.00382), and rank trails hard by −6.988 pp (95% CI −7.951 to
−6.025; Holm p=2.19e-49). The stable ordering contradicts a strong
trajectory-correction account: hard does not acquire a hidden free-running
advantage on the selection split.

### Selected committee-size and leave-one-out trajectories

The complete 48,600-row panel contains the teacher-forced best hard committee
at K=1…5 and all seven K=6 leave-one-out committees; full K=7 is reused from the
selected four-method run. Free-running best-at-K accuracy is nonmonotonic:
26.099%, 25.012%, 27.185%, 27.333%, 25.877%, 26.395%, and 25.309% for K=1…7.
K=4 is best, rejecting a simple threshold or monotonic scaling claim.

In the paired seven-test leave-one-out family, removing Track A (full-minus-K6
+8.420 pp), Track B (+5.852), General (+4.617), Abstention (+2.370), or Social
(+1.605) harms overall accuracy after Holm correction. Removing Grounding
improves accuracy by 1.086 pp (Holm p=0.00274), while Procedure is neutral
(−0.099 pp; Holm p=0.813). The abstention expert is strongly class-specific:
it improves abstention accuracy by 28.889 pp while reducing answerable accuracy
by 1.709 pp. These are deployed-committee removal effects, not matched-training
causal estimates.

### Adaptive consultation and quality/compute frontier

The predeclared Track A + Track B + General core escalates on disagreement under
the frozen rule. Free-running adaptive accuracy is 25.111% versus hard K=7 at
25.309%, a nonsignificant −0.198 pp (95% CI −0.765 to +0.370; McNemar
p=0.546). Mean expert forwards fall from 34.150 to 14.828 (−56.579%) and mean
latency falls from 1.919 s to 0.928 s. The aggregate similarity hides a policy
shift: adaptive improves answerable accuracy by 1.168 pp and reduces abstention
accuracy by 9.074 pp. It is a favorable DEV compute frontier, not evidence of
unchanged class-specific utility.

### Selected calibrated probability on v2.2 TEST

The DEV-selected full-K=7 calibrated rule reaches 560/4,050 (13.827%), versus
hard at 699/4,050 (17.259%) and raw probability at 571/4,050 (14.099%).
Calibrated-minus-hard is −3.432 pp (95% CI −4.272 to −2.617; Holm
p=1.073e-15); calibrated-minus-raw is −0.272 pp (95% CI −0.988 to +0.444;
p=0.499). Calibration improves local and free-running DEV performance but does
not close the v2.2 TEST hard gap. The result is not wholly untouched: hard/raw
anchors were known before freezing and the archived n=32 pilot exposed the
same TEST split.

### Cross-benchmark

The provenance gate passed all 27 mandatory checks, but fixed-method transfer is
negative. On 1,320 v2.1.2 examples, Backbone scores 0.455%, Track A 0.379%,
Abstention 9.091%, hard K=7 0.076%, raw probability 3.030%, and calibrated
probability 0.152%. Hard-minus-raw is −2.955 pp (Holm p=1.528e-10). This
contradicts cross-corpus hard superiority. It remains a transfer benchmark—not
a globally untouched holdout—because 1,008 examples repeat exact v2.1.2 TRAIN
prompt+gold pairs and this TEST was used historically by another model.

Authoritative numerical packets are `selected_dev_free_run_analysis.json`,
`adaptive_free_run_dev_analysis.json`, `selected_committee_dev_analysis.json`,
`v22_calibrated_test_analysis.json`, `cross_v212_analysis.json`, and
`final_synthesis_facts.json`.

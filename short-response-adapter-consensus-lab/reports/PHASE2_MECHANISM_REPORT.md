# Phase 2 Mechanism Report

**Status:** FINAL MECHANISM SYNTHESIS. The cornerstone, full-TEST trace
analysis, DEV teacher-forced study, all 127 cached committees, every frozen
free-running panel, adaptive consultation, calibrated v2.2 TEST confirmation,
fixed-method v2.1.2 transfer, failure/semantic audits, figures, and tables are
complete. The primary-source adjacency audit and 25-entry verified bibliography
are complete; ELREA is treated as the closest prior shared-prefix token-level
LoRA ensemble. Claims below remain bounded to the recorded lineages and data.

## Most defensible current explanation

Hard K=7's 17.259% v2.2 TEST accuracy is real and paired-significant, but the
evidence does not support a simple claim that plurality is the best local
next-token classifier. On common DEV gold prefixes, raw and calibrated
probability means both outperform hard plurality. The observed sequence-level
advantage is instead consistent with an interaction of three mechanisms:

1. **Policy composition.** Relative to Track A, hard voting gains 230 correct
   abstention cases and loses 122 correct answerable cases, for the observed net
   +108 sequences. Its advantage over the strongest individual is therefore an
   answer/abstention rebalancing, not uniform reasoning improvement.
2. **Directional aggregation without confidence-scale dominance.** Hard voting
   preserves top-1 proposals, vote counts, and unique-plurality winners under
   positive per-expert temperature scaling. Its confidence tie-break is a
   measured exception under unequal temperatures. Expert
   calibration differs substantially, and a high-confidence dissenter is
   present at 1,645 of 1,804 first hard/soft divergence points. Controlled
   temperature changes move soft decisions while leaving hard proposals fixed.
3. **Early path and policy interaction, without evidence of trajectory
   correction as the source of the gain.** Hard and probability fusion diverge
   on 1,804 TEST sequences, with 1,502 first divergences at t=0 and 1,781 by
   t=2. Against probability mean, hard gains 163 correct answerable cases and
   loses 35 correct abstention cases, net +128. Yet the DEV teacher-forced and
   free-running orderings and gaps are nearly unchanged, so early divergence is
   descriptive rather than evidence that hard repairs trajectories.

Calibration mismatch is a contributor but not a sufficient explanation.
Calibrated probability remains above hard on same-split free-running DEV
(26.790% versus 25.309%) but falls below hard on v2.2 TEST (13.827% versus
17.259%). The cornerstone is therefore best explained as benchmark-specific
policy composition plus robust directional aggregation—not as universal hard
superiority, calibration alone, or trajectory correction. Fixed-method
v2.1.2 transfer is negative: hard reaches 0.076% versus raw probability's
3.030%, precluding a generalization claim.

## Verified cornerstone

- Raw SHA256: `d3733391b53dc53e7d9ca899d7015ba6ff0b2faa7c28e52a2778a0827c45c32d`.
- 4,050 unique TEST examples and 32,400 condition rows; no missing or duplicate
  condition keys.
- Backbone 1/4,050 (0.0247%), abstention 527/4,050 (13.0123%), probability K=7
  571/4,050 (14.0988%), Track A 591/4,050 (14.5926%), hard K=7 699/4,050
  (17.2593%).
- Hard versus Track A: +2.6667 pp, rescue/harm 328/220, exact McNemar
  p=4.543e-6, Holm p=1.363e-5.
- Hard versus probability K=7: +3.1605 pp, rescue/harm 226/98, exact McNemar
  p=8.715e-13, Holm p=4.358e-12.
- Independent verification: `reports/PHASE2_CORNERSTONE_VERIFICATION.md`, PASS
  30/30.

## Answers to the 20 required questions

### 1. Why does hard voting outperform the strongest individual adapter?

On full TEST, the net gain over Track A is entirely a policy-composition trade:
hard is 3.476 pp worse on 3,510 answerable cases but 42.593 pp better on 540
abstention cases. In counts, +230 abstention successes offset −122 answerable
successes, yielding +108 overall. Teacher-forced DEV also does not show a local
all-token advantage over standalone Track A (25.26% versus 26.07%). The answer
is therefore **abstention-capable coalition composition**; the completed
teacher/free comparison does not support an additional trajectory-correction
explanation.

### 2. Why does hard voting outperform probability mean?

Hard rescues 226 probability failures and harms 98 probability successes. It is
4.644 pp better on answerable TEST cases (+163 correct) and 6.481 pp worse on
abstention cases (−35 correct). First divergence is usually early and commonly
includes a high-confidence dissenting expert. This supports resistance to
confidence-scale dominance plus path/policy differences, not a claim of
uniformly superior local decisions.

### 3. Is calibration mismatch sufficient to explain the gain?

**No sufficiency claim is supported yet.** Expert confidence gaps range from
−3.44 pp (Track B) to +27.24 pp (Abstention), and fitted temperatures range from
1.0 to 2.0. Per-expert calibration raises full-K7 DEV local accuracy from
78.91%/26.62% token/example for raw probability to 79.23%/26.77%, above hard's
78.33%/25.26%. Bounded confidence clipping barely changes raw pooling (26.44%,
26.64%, and 26.62% example accuracy at caps 0.5, 0.7, and 0.9), while inverse-
entropy expert weighting falls to 8.25%. Trimmed pooling reaches 26.20%,
median pooling only 13.73% (with 2/18,760 decisions uncertified), and
validation-weighted hard reaches 24.77%. This proves scale sensitivity and the
qualified hard-vote invariance property but does not make calibration mismatch
sufficient. In free-running DEV, calibrated probability reaches 26.790% versus
hard's 25.309% (+1.481 pp, 95% CI +0.741 to +2.222; Holm p=2.773e-4). On v2.2
TEST it reaches only 13.827% versus hard's 17.259% (−3.432 pp, 95% CI −4.272
to −2.617; Holm p=1.073e-15) and is statistically indistinguishable from raw
probability (−0.272 pp; p=0.499). Calibration mismatch is therefore real but
insufficient.

### 4. How much gain comes from abstention composition?

Against Track A, all +108 net sequence successes arise after +230 abstention
successes offset −122 answerable successes. Hard's absolute abstention accuracy
is 47.59%, between Track A's 5.00% and the abstention specialist's 97.59%.

### 5. How much gain comes from reasoning-answer rescue?

Against raw probability mean, hard adds 163 correct answerable sequences and
loses 35 abstention sequences, net +128. Against Track A, hard loses 122
answerable sequences. “Reasoning rescue” is therefore comparator-dependent.

### 6. Which experts have positive marginal contribution?

Under exact hard-plurality Shapley values for DEV all-gold-token example
accuracy: Track A +10.23 pp, Track B +8.26 pp, General +6.52 pp, Abstention
+1.95 pp, and Procedure +0.64 pp. The free-running leave-one-out panel agrees
that Track A, Track B, General, Abstention, and Social make positive
context-dependent contributions to K=7; it does not convert these observational
removals into matched-training causal effects.

### 7. Which experts are harmful?

Grounding (−1.02 pp) and Social (−1.32 pp) have negative exact cached Shapley
values. Free-running removal of Grounding improves accuracy by 1.086 pp (Holm
p=0.00274), confirming harm in the deployed K=7 context. Procedure is
free-running neutral (K7−K6 −0.099 pp; Holm p=0.813), while Social reverses
from negative cached Shapley to a positive K=7 contribution (+1.605 pp; Holm
p=3.70e-7), demonstrating context dependence of marginal attribution.

### 8. Which coalitions dominate successful tokens?

Across 23,204 hard-vote TEST token steps, unanimity (`1111111`) is most common
at 8,133 steps (35.05%). The next largest masks are Track A+Track B+General+
Grounding+Procedure (`1111100`, 11.87%) and Track A+Track B+General+Abstention+
Social (`1110011`, 10.64%). Coalition-to-sequence-correct associations are
descriptive and must not be read causally.

### 9. Does specialization causally matter?

**Unresolved.** Functional contributions and category-varying coalition
frequencies are present, but there is no matched compatible random-seed family.
In a bounded eight-example DEV control, three exact General copies reproduce
the General K=1 sequence on all 24 method-example comparisons, while weak and
fixed-pool injections change trajectories. The fixed-pool proxy reuses trained
adapters and is not competence matched, so it cannot identify semantic
specialization. H9 versus H10 still requires a compatible matched lineage.

### 10. Does committee size exhibit smooth or threshold-like scaling?

No defensible threshold is established. Mean hard local example accuracy over
all committees is 13.92%, 13.72%, 17.09%, 18.22%, 19.39%, 22.22%, and 25.26%
for K=1…7. It dips at K=2 and then rises, while the selected best hard committee
peaks at K=4 (27.31%) rather than K=7. Free-running selected committees
reproduce the nonmonotonic pattern: K=1…7 accuracy is 26.099%, 25.012%,
27.185%, 27.333%, 25.877%, 26.395%, and 25.309%. K=4 is best; there is no
threshold or monotonic committee-size claim.

### 11. Is there evidence of trajectory correction?

Early divergence is real, but the trajectory-correction hypothesis is not
supported. Hard and probability sequences differ on 1,804 of 4,050 TEST cases;
83.26% of first divergences occur at t=0, and same-prefix reruns found zero
top-token mismatches. On the decisive same 4,050 DEV examples, however, the
teacher-forced all-token-example ordering (calibrated 26.765%, raw 26.617%,
hard 25.259%, rank 18.469%) is essentially preserved free-running (26.790%,
26.593%, 25.309%, 18.321%). Divergence does not imply corrective steering.

### 12. Does teacher-forced token accuracy predict free-running gains?

Yes, at the method-ordering level in this DEV panel. The local ordering
calibrated probability > raw probability > hard > reciprocal rank survives
free-running generation, with changes of only +0.025, −0.024, +0.049, and
−0.148 percentage points respectively. This close tracking contradicts H7's
prediction that hard would acquire a materially larger advantage through
autoregressive correction.

### 13. Does hard voting suppress overconfident mistakes?

**Partially supported.** Positive temperature scaling cannot change hard
proposals or unique plurality winners, while it materially changes probability
fusion. At TEST first
divergence, 1,645/1,804 cases have a high-confidence dissenter. This establishes
the opportunity and robustness property. On DEV gold prefixes, clipping top-1
mass at 0.5, 0.7, or 0.9 changes raw probability all-token accuracy by at most
0.17 pp; trimming is slightly harmful, and inverse-entropy/median pooling are
strongly harmful. Validation-weighted hard also trails equal hard on the
all-token example criterion. The deployed
confidence tie-break is not fully scale invariant: fitted unequal temperatures
change 259/18,760 local decisions, all within 1,097 plurality ties. It therefore
does not prove every suppressed high-confidence proposal was erroneous or that
confidence suppression alone causes the gain.

### 14. Can rank-based aggregation reproduce the benefit?

Not locally. Reciprocal-rank top-5 is the best tested rank rule but reaches
76.34% token and 18.47% all-token example accuracy, below hard's 78.33% and
25.26%. Top-2 approval and Borda are worse. The local rank-sufficiency
hypothesis is contradicted, and free-running reciprocal-rank top-5 remains far
below hard (18.321% versus 25.309%; −6.988 pp, 95% CI −7.951 to −6.025; Holm
p=2.19e-49).

### 15. Can calibrated soft fusion reproduce the benefit?

It exceeds hard locally and free-running on DEV: 26.790% versus 25.309%, a
+1.481 pp paired gain. It does not reproduce the hard benefit on v2.2 TEST:
13.827% versus 17.259%, a −3.432 pp paired deficit. Thus calibrated soft fusion
can beat hard under same-split DEV selection but cannot explain or close the
cornerstone TEST result. The TEST confirmation retains known-anchor and
archived n=32 exposure caveats.

### 16. Does Hard K7 remain better on an independent benchmark?

No. The v2.1.2 gate passed 27/27 mandatory checks and has zero exact v2.2 prompt
overlap, but the fixed-method result is negative: hard scores 1/1,320 (0.076%)
versus raw probability 40/1,320 (3.030%) and the abstention specialist
120/1,320 (9.091%). Hard versus raw is −2.955 pp (Holm p=1.53e-10). This is a
cross-corpus transfer result, not a globally untouched/template-independent
holdout: 1,008/1,320 TEST rows repeat exact prompt+gold from v2.1.2 TRAIN, and a
historical dense model used this TEST. Results are not pooled with v2.2.

### 17. Can adaptive consultation preserve the gain at lower compute?

Yes for overall DEV accuracy, with a changed answer/abstain tradeoff. The frozen
Track A + Track B + General core escalates on disagreement. Free-running
adaptive accuracy is 25.111% versus 25.309% for hard K=7 (−0.198 pp, 95% CI
−0.765 to +0.370; McNemar p=0.546), while mean expert forwards fall 56.579%
(14.828 versus 34.150) and latency falls from 1.919 s to 0.928 s. It improves
answerable accuracy by 1.168 pp but reduces abstention accuracy by 9.074 pp.

### 18. What is the best quality/latency frontier?

On v2.2 TEST, the nondominated frozen points are Track A (14.59%, 0.151 s mean
latency, 4.03 forwards) and hard K=7 (17.26%, 1.432 s, 40.11 forwards). On the
DEV-selected systems panel, adaptive hard offers the strongest compute
reduction near full-K7 accuracy (25.111%, 0.928 s, 14.828 forwards), while the
selected K=4 committee has the highest measured DEV accuracy (27.333%). These
are split-specific frontiers; answer/abstain utility differs sharply.

### 19. Which original hypotheses were falsified?

- Local hard-rule superiority is contradicted by raw and calibrated probability
  fusion on gold prefixes.
- Rank/ordinal information sufficiency is contradicted locally.
- Generic diversity as a sufficient explanation is contradicted: at each fixed
  K=2…6, hard-committee top-1 disagreement correlates negatively with local
  example accuracy.
- Strictly monotonic best-committee scaling is contradicted; the best hard
  committee peaks at K=4.
- Deployment efficiency of full K=7 is contradicted by measured latency and
  forward counts.
- Trajectory correction as the source of hard's advantage is not supported by
  the near-identical DEV teacher-forced and free-running ordering.
- Cross-corpus superiority is contradicted by the negative v2.1.2 transfer.
- Calibrated soft fusion closing the v2.2 TEST hard gap is contradicted.

### 20. What is the most defensible mechanism?

The evidence supports **benchmark-specific policy composition plus robust
directional aggregation**. Hard voting uses equal top-1 opinions, so unique
plurality winners cannot be dominated by one expert's calibration scale;
confidence re-enters only at tied vote leaders. This matters because expert
confidence behavior differs substantially. Calibration alone is insufficient:
calibrated soft is better on same-split DEV but worse on v2.2 TEST. The completed
teacher/free comparison also rejects a strong trajectory-correction account.
Hard plurality implements a calibration-resistant, composition-dependent
answer/abstain policy that fit v2.2 TEST; it is not a universally better local
rule and does not transfer to v2.1.2.

## Hypothesis adjudication snapshot

| Hypothesis | Current status | Decisive evidence still needed |
|---|---|---|
| H1 Calibration-robust consensus | PARTIALLY SUPPORTED; sufficiency contradicted | calibrated soft beats hard on DEV but not v2.2 TEST |
| H2 Complementary error correction | SUPPORTED descriptively and by K=7 removals | matched-training causal controls remain absent |
| H3 Specialist coalitions | PARTIALLY SUPPORTED | matched/category-specific causal controls |
| H4 Information bottleneck/rank sufficiency | CONTRADICTED | rank remains far below hard free-running |
| H5 Overconfident-expert robustness | PARTIALLY SUPPORTED | free-running perturbation link |
| H6 Committee threshold | CONTRADICTED | free-running K=4 is best and scaling is nonmonotonic |
| H7 Trajectory correction | NOT SUPPORTED | teacher/free method ordering is nearly identical |
| H8 Abstention/EOS composition | SUPPORTED contextually | removal strongly changes the answer/abstain frontier |
| H9 Diversity, not expertise | UNRESOLVED | compatible matched seed committees |
| H10 Specialization matters | PARTIALLY SUPPORTED functionally; causal claim unresolved | matched specialization controls |

## Evidence and reproduction

Numerical sources and exact hashes are indexed in `paper/EVIDENCE_LEDGER.md`.
Machine-generated paper tables are in `paper/TABLES.md` and `paper/tables/`;
figure source/output hashes are in `paper/figures/figure_metadata.json`.
Exact command order and resume behavior are documented in
`paper/EXPERIMENTAL_PROTOCOL.md`. Live progress is generated by
`experiments/phase2_mechanism/update_live_report.py` into
`reports/PHASE2_LIVE_RESULTS.md`.

Final long-run numerical sources are `selected_dev_free_run_analysis.json`,
`adaptive_free_run_dev_analysis.json`, `selected_committee_dev_analysis.json`,
`v22_calibrated_test_analysis.json`, `cross_v212_analysis.json`, and
`final_synthesis_facts.json`. Runtime integrity is additionally bound by
`rank_topk_parity_verification.json`, `selected_dev_resume_history.json`, and
`selected_dev_runtime_boundaries.json`.

## Completion state

All predeclared automated inference and analysis gates are complete. The final
package is subject to the machine-readable synthesis verifier, full test suite,
hash-bound figure/table regeneration, official-source immutability check, and
completion audit; those checks are reported separately rather than inferred
from this narrative.

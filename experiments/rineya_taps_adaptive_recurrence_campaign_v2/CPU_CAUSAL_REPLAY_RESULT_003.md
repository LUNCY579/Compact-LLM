# Full 192-case causal replay: no direction advantage from coherence

The frozen input-only coherence feature is positive on all192 DEVELOPMENT cases
(range0.4373650848865509–0.5430105924606323). Its sign therefore always predicts
increasing the full L12–L13 residual scale. It agrees with91/188 numerically
reliable local loss directions, exactly the same as constant increase; constant
decrease agrees with97/188. Four neutral cases remain in the192-case population,
with zero numerically uncertain nonneutral cases. Exploratory tied-rank
Spearman(balance,-mean derivative)=-0.09544292379893118. This establishes no added
sign discrimination for the frozen rule on this endpoint; it is not a claim
about all features, schedules, continuous scales or generated task correctness.

| Canonical answerability | Increase favored | Decrease favored | Neutral | Total |
|---|---:|---:|---:|---:|
| Answerable |91|52|1|144|
| Unanswerable |0|45|3|48|
| All cases |91|97|4|192|

All48 unanswerable cases use canonical UNKNOWN, one target token on the original
input: increasing alpha1→1.01 increases its loss in every case, while reducing
alpha to0.99 decreases it. Three have a derivative within the fixed neutral
threshold. These are source-grounded local likelihood comparisons, not observed
generated-answer flips. The same unsigned feature predicts increase in every
case. This is consistent with the earlier provisional generation finding that
larger scales added unsupported answers, but does not repair that run's missing
validation or final GPU-state proof.

Directions also vary within paired worlds:34/96 parents have opposite reliable
variant signs (19/80 same-gold pairs and15/16 evidence-transfer pairs). Increasing
harms at least one variant in67 parents; decreasing does so in63. None of these
pairs or token targets supplies an additional independent source. All six
categories retain32 cases; the96 parents share six authored generators.

Equal-category/equal-parent mean canonical answer-prefix loss is1.2062422250
at alpha1,1.2059134134 at1.01 and1.2065944572 at0.99. The small average improvement
for increase coexists with harm on many cases; neither sign counts nor this
proxy mean establishes an accuracy gain, preservation or deployment value.
Per-category counts, paired rows, exact finite deltas and strata are retained
in the corrected source-bound REPORT.json.

## Native mechanics, preservation and costs

The directly approved attempt003 completed all192 cases/96 parents,440 answer
positions and192 separate EOS positions. Every target uses a fresh complete
unit/.99/1.01 rollout; the unit state adjoint includes the actual L14–L27 suffix
and abstention head. Current/future canonical positions are excluded from the
input; preceding gold positions are explicit teacher conditioning. Input-only
prediction features freeze before the first loss; later teacher quantities are
not online/A5 inputs and no fit was performed.

Fresh representative/max256 scalar/EOS qualification passed eight exact parity
pairs, cache refusals, normal/forced restoration and three persisted G/raw-update
vector pairs. Maximum actual causal prefix236 is within the qualified256. Every
632-target and192-case mean FD gate passes unchanged .001+.03*abs(adjoint).
The maximum absolute target discrepancy is0.00135207 (the relative term matters);
maximum mean discrepancy0.000963569. Neutrality remains abs(mean derivative)<=.001.

Actual work:1925 API attempts,1923 full-forward attempts,1922 complete forwards,
635 state adjoints and252696 completed prefix positions. One expected partial
exception has unknown partial compute; two cache rejections occur before
computation. These are physical work counts, not independent cases.

Stored checkpointa99 and loaded FP32 stateb23 bind the canonical Supa-A1.1
checkpoint and native UNKNOWN51378 readout. Final before/after loaded stateb23,
all97 dispatched sources and ten canonical assets match; parameter gradients
NONE, trainable parameters0, optimizer steps0. CPU FP32/eager, deterministic,
four threads, batch1/full-prefix/no-padding/no-KV and one extra whole block remain
fixed. This qualifies this surrogate host; it is not new generation qualification
or retroactive validation of the original CUDA fault.

The fresh conservative complete forecast was4297.4992531seconds including180
internal reserve, within the approved4500 cap. Parent charged1389.328seconds
(23.16minutes), versus worker-reported1386.6926846. Worker peak RSS4457250816
bytes (4.15GiB), parent sampled tree peak4348514304; minimum recorded worker host
availability14571302912bytes. Resource floors remained unchanged. No worker/lock
remains. Cumulative native use2837.843/21600seconds; remaining18762.157. The da328
one-attempt allowance is consumed. No new retry, fullbulk restart or model job
follows from its success.

## EOS precision and the preserved analyzer correction

EOS remains a separate accepted-stop-mass surrogate after supplied canonical
text; canonical EOT151645 CE is descriptive. All192 EOS rows and raw values are
retained. Mean raw mass loss is0.0141806901 at1,0.0143579543 at1.01 and0.0140068233
at0.99; increases occur in132 cases for plus and20 for minus. This is not on-policy
termination or a primary answer+EOS composite.

The frozen bfe6 analyzer reported primaryNULL because one EOS target at three
scales exceeded its extra1e-5 cross-kernel ordering assumption (six duplicate
views). A no-model representable-logit shift fixture demonstrates that this
fixed inequality tolerance can reject valid FP32 computation. Actual logits/LSE
operands were not stored, so the native cause/error bound is not proven.

The separate reviewed5a624 successor preserves oldbfe6/tests/plan/native bytes,
all original native/FD/finite/count/source/causal gates and all numerical values.
It marks EOS ordering UNVERIFIED_FINITE_PRECISION for every row and retains the
original1e-5 exceedances descriptively:335 positive raw gaps, three exceedances,
max2.4199485778808594e-5. No enlarged tolerance, clipping, case removal, endpoint
switch or inferred exact rounding cause was used. The old null report and raw
bundle remain preserved; corrected analysis is a new exclusive output. Ten
independent changed fixture probes passed, and19 analyzer helper functions
remain byte/AST equivalent.

## Interpretation and next decision

This is a restricted negative mechanism result for the frozen coherence sign
rule on exposed canonical teacher-prefix likelihood. It adds evidence to the
provisional generation comparison(bestfixed69/192,TAPS67,finiteoracle70), whose
preservation failed and validation remains null. No controller fit or wider
scale/depth search is justified by these observations. A stronger ordinary
source/query-grounding or execution baseline is a more justified future task
comparison; its effect, cost and implementation are unmeasured. Any intervention
outside initial scope needs direct approval plus its own registration, matching
controls, qualification and complete forecast. No new candidate or deployment
change is selected here.

Independent raw recomputation matches all case means, features, prefix hashes,
counts, FD gates, ranks, parents and strata. Worker-author and analyzer-author
audits disclose their roles; final adversarial review94071d independently
recomputed the complete corrected report. See the exact native/math/review
receipts, correction document, old null analysis and raw journals. Semantic
performance, broader source confirmation and useful matched-quality latency
remain unproved. The full mission completion assessment remains separate.

# Track A: native ARMR generalization v2.1

This protocol is prospective. Registration is effective only after the dataset,
scorer, source and configuration hashes are frozen and the protocol is committed.
The shared execution guard verifies registration before any efficacy generation.
No results from this new bank have been used to select these arms or thresholds.

## Question and historical basis

Does bounded native recurrence improve performance across fresh scenario
structures, and does it outperform an approximate compute-matched residual?
The historical 16 inspected items yielded A0=12, A1=12, A3=13 with one paired
win and zero losses. They provide a hypothesis, not confirmatory evidence.
The historical code and evidence directories remain unchanged.

Use CPU FP32, eager attention, seed 17, four threads, full-prefix recomputation,
no KV cache, frozen canonical Supa-A1.1 including the abstention offset. Preserve
token positions, masks, rotary embeddings, the stem L0-L2 and all ordinary layers.
Generation is greedy, up to 48 new tokens (or a smaller frozen item cap; the
initial bank caps items at 32), with
a 900-second per-item budget checked between forwards. A Torch forward cannot
be preempted safely; any elapsed-budget failure is recorded and invalidates
the shard. Score and NLL are recorded separately from response validity.
Loading requires at least 3 GiB of available system memory, and the process
must remain below the registered 8 GiB RSS bound. Recheck available memory at
execution time; a previous machine inventory does not authorize a load.

## Frozen candidates

| Arm | Added operation | Selection basis |
|---|---|---|
| A0 | None | Intact paired control |
| A1 | One additional L12 | Historical single-layer control |
| A3 | Additional L12 then L13 after ordinary L13 | Historical strongest fixed hypothesis |
| A16 | One additional L16 | Atlas repeatability at an independent middle depth |
| AM | Independent frozen SwiGLU residual after L13 | Generic extra transformation with A3-matched matrix work |

L16 was the only tested middle layer passing the historical smoke gate at both
two and three total applications. At two applications its mean logit cosine was
0.9787 and KL 0.0806; at three, cosine 0.9430 and KL 0.3046. Neither improved
the smoke score. A16 here means **two total applications**, frozen prospectively.
No additional native layers or doses will be added after efficacy inspection.

AM computes `h + M(h)` at the output of ordinary L13. `M` applies unit-scale
RMS normalization (canonical epsilon), then `Wdown(silu(Wgate h) * Wup h)`.
All three matrices are independent normal draws, standard deviation .02, local
CPU generator seed 7301. Alpha is fixed at 1. No labels, activation fitting,
development accuracy, or confirmatory outcomes select or calibrate this control.
Its 1024→10240→1024 geometry has 31,457,280 matrix parameters and 62,914,560
multiply-plus-add FLOPs per token, exactly matching the dominant linear work of
two native decoder layers. Additional active matrix parameters equal total
added parameters; RMS normalization has no learned coefficients.

This is an approximate compute control: attention, normalization, instrumentation
and optional gold NLL are excluded from the estimate. Report measured latency,
peak RSS, active parameters and model-level estimated work. AM is untrained;
beating it cannot establish superiority over a learned same-scale operator.
Its outcome, including inertness or degradation, must be retained. Do not choose
a new initialization because this one performs inconveniently.

## Data and analysis units

The Track A bank is independently seeded and manifest-frozen. Its primary
confirmatory portion contains 1,024 original examples across multiple families
and templates. An additional 128 diagnostic variants remain separate from the
primary denominator. Architecture-development inputs serve mechanical rejection
only. Track B calibration and fitting never consume Track A answers.

All variants of one latent scenario share a split and scenario ID. Primary
inference uses one original per scenario. Diagnostic variants test paraphrase,
changed names/directions, counterfactual/context alterations and evidence order
where the family supports them. Analyze these paired against their originals;
do not count them as independent examples or increase nominal sample size.
Report template-level and family-level effects, longer chains and distractors
as exploratory strata. No failed questions may be edited after model output.

Data validation occurs before freeze: unique immutable IDs, independently
recomputed answers, allowed scorer schema, nonempty prompts, known family/template
labels and no split overlap. Post-freeze exclusions are permitted only for
objectively demonstrated corruption, duplicate scenario allocation, or a wrong
programmatic answer under the already specified semantics, with an exclusion
receipt made without using arm identity or model correctness. Preserve the
original complete results and report sensitivity with all frozen items.

## Primary tests and decision

Every arm receives exactly the same frozen primary examples. Report raw accuracy,
paired wins and losses, two-sided exact McNemar p, paired scenario bootstrap 95%
CI (10,000 replicates, seed 9107), absolute accuracy gain, absolute error
reduction and relative error reduction. Zero baseline errors makes relative
error reduction undefined; it must not be replaced by an invented value.

The confirmatory hierarchy first tests A3 vs A0, then A3 vs AM only if the first
test passes. A recurrence-specific GO requires both comparisons to have positive
95% CI lower bounds and exact McNemar p≤.05. In addition, A3 vs A0 must achieve
at least 2 absolute accuracy points **or** 5% relative error reduction. Secondary
A1/A16 comparisons vs A0 form a separately identified Holm family at .05; they
do not replace a failed A3 primary with a retrospectively chosen primary.

Preservation requires the pooled registered `ordinary_qa_preservation`,
`instruction_evidence` and `retrieval_multihop` families to exclude more than a 2-point loss
(one-sided 95% lower confidence bound greater than −.02), with each family's raw
paired effects disclosed. Require no additional invalid outputs and no safety
stops. The pooled preservation gate may be inconclusive at this sample size.
Quality-per-cost is assessed by accuracy/latency/FLOP frontier; no unregistered
scalar utility or arbitrary latency cutoff is imposed. A cost increase must be
reported even for a positive accuracy outcome.

Report family A0/candidate accuracy, wins/losses, scenario CI where meaningful,
paired canonical-answer NLL delta, output identity and validity. Family and
variant analyses are exploratory; a specialized positive pattern is a hypothesis
for a new separately frozen confirmation, not proof of a router's efficacy.
Planning scenarios in the shared power report govern interpretation: 1,024
examples cannot guarantee adequate power for one- or two-point effects without
knowing paired discordance. Failure to reject does not prove equivalence.

## Immutable execution and failure policy

Use 32 deterministic ID-based shards. Complete matching A0 evidence for a shard
before any candidate arm. Every attempt writes an exclusive directory; never
append to another attempt or retry into failed evidence. Resume by running only
missing arm/shard combinations into fresh directories, then aggregate complete,
nonoverlapping successful shards after all arms finish. A0 and candidates must
match model, source, scorer, dataset, runtime, configuration and registration.
Do not run simultaneous model workers on this constrained CPU host.

Before confirmatory execution, complete all five arms on the frozen 128-item
architecture-development set under this registration. Reject safety failures,
malformed generation or additional invalid outputs. This is a mechanical gate;
poor accuracy alone does not license deleting a control, changing its parameters
or substituting a candidate. A changed mechanism requires a new version and freeze.

Fail closed on nonfinite values, RMS ratios outside [.25,4], generation safety
stops or elapsed deadlines, checkpoint/state mutation, trainable/gradient-bearing
backbone or control parameters, mismatched hashes, changed scorers, missing A0,
cross-track input leakage, evidence overwrite, unregistered arms or settings.
Keep failure receipts and partial outputs; no capability claim from partial runs.
Aggregation always runs the independent raw-evidence audit before issuing a
decision: complete exact role coverage across all five arms, per-case/results
reconciliation, logit artifact hashes, frozen scorer/validity reconstruction,
canonical and control identity, and registered runtime. Linked-diagnostic
auditing additionally requires the complete original-bank evidence and computes
per-variant changes in each arm's paired advantage using scenario resampling.
This audit does not rerun model inference. Every audit writes a new exclusive
receipt, including failed audits; no historical receipt is repaired in place.

No routing, MLA/MHA, MTP, PEFT, backbone training or sealed evaluation is licensed.
If A3 beats A0 but not AM, extra compute remains plausible and recurrence-specific
benefit is unresolved. If neither comparison supports benefit, retain the
negative result. If complete outcomes license future work, the next causal
experiment must receive its own prospective registration and fresh data.

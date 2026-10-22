# IDR-001 protocol — development proposal, not an already frozen native study

## Decision and hypothesis

Keep Rineya's qualified core and adapters unchanged in Stage A. Test whether
balancing helps select useful alternatives for the **same** operation. Enter this
stage only after a development counterfactual matrix shows useful complementarity.
If no eligible expert solves the difficult rows, spend the next budget on competence,
not on a more complex selector. No expert is made eligible by its low usage alone.

The source paper motivates a controller; it does not establish this specialist
routing transfer. A load histogram is secondary evidence, never the primary goal.

## Stages and preconditions

**0 — Custody and diagnostic.** Pin code, base, tokenizer, every adapter, expert
order, coefficient, feature definition, splits, gold interpreter, and compute budget.
Collect all eligible expert outcomes on TRAIN/development data. Count every
counterfactual call, generated token, and elapsed second. Report best-eligible
headroom separately from current accuracy and no-expert-solves cases. Do not use
that oracle as a deployable method. No automatic entry threshold is claimed; a
suggested exploratory screen is at least five percentage points of recoverable
headroom with enough independent parents to quantify uncertainty.

**A — Frozen call routing.** Fit the quality head on TRAIN outcomes only. Calibrate
its predicted probabilities on independent calibration data (the fitted linear
head does not itself guarantee probability calibration). Freeze it before router
selection. Compare none, sign, integral, and ID under uniform and prior targets.
Use identical qualified expert banks, fixed quality predictions, task order,
request budgets, and coefficient values. Whole-call switching must preserve
adapter-off and prevent KV-cache cross-contamination. Evaluate initially with
fixed biases; online adaptation is a separate future study, not enabled here.

**B — Fresh LoRA bank.** After Stage A's entry diagnosis, inject four rank-4 LoRA
experts with Top-2 selection in explicitly approved FP32 dense projections.
Compare none, sign, integral, ID, and auxiliary-loss arms from identical initial
weights and data per seed. No balancing-gradient term is added to the four bias
controller arms. The auxiliary arm uses no bias control. Keep the shared core
frozen. Existing specialist adapters are never overwritten or densely merged.

## Controller choices and ablations

Reference mode uses uniform targets, `bias_limit=null`, `min_expected=0`, and the
paper equation. Guarded mode adds bounded zero-sum projection and count windows
that defer tiny target mass. The bias itself stores integral history, so projected
saturation has no additional hidden accumulator that can wind up indefinitely.

For Stage A, targets are normalized positive priors over the current safe set, or
uniform probabilities over that set. They sum to one per call. Zero-prior experts
are ineligible. On disappearance, there is no new target pressure for an absent
expert; its derivative history resets when a completed update has no target for it.
Deferred windows are accumulated as a whole, not inconsistent expert-wise fragments.

Stage B uses feasible uniform targets `N_valid * K / E`. Padding is excluded. The
MoE implementation does not silently pretend to support arbitrary Top-K nonuniform
marginals. No DDP, asynchronous updates, mixed precision, or gradient-checkpointed
recomputation is supported in the native pilot. For future DDP, globally sum counts
and target mass across microbatches/ranks and update once after a successful step.

Test integral gain and derivative gain independently, including `kd=0`. Allocate
an equal number of development configurations to each method. Test bounded and
unbounded control as separate arms, not interchangeable labels. The built-in
quantile trace is an additional controller reference, not a scored native baseline.

## Challenge banks

Use fresh payloads and parent identities, disjoint across TRAIN, calibration,
development, and confirmation; additionally compare against existing historical
fingerprint registries before native admission. The supplied split checker catches
parent/payload reuse among its inputs, not unseen external histories.

The independent ledger generator provides in-distribution controls, unseen statuses,
large magnitudes, renamed accounts, key-order perturbations, and combined shifts.
Development and confirmation use different held-out status vocabularies. Multiple
variants of one parent are not independent observations. This generator is not a
copy of the RCWM bank or an established benchmark. Pretest its oracle exhaustively
and preregister distributions before model calls.

For BWM, attach the router before a complete learned transition call. Keep proposer
pools separate. Use independent fresh Blocksworld/PlanBench splits and the existing
external plan validator. The provided exact-JSON scorer assesses states only; it
is NOT a plan validator and cannot establish valid-plan rate. Add the benchmark's
existing verifier through the backend/evaluation seam rather than replacing it.

`worlds.self_rollout` chains predictions from previous predicted states and never
teacher-forces the reference trajectory. Desired goals/rewards may be used by a
separate plan evaluator, never to choose a forward expert that predicts a more
favorable future. Test horizons 1, 2, 4, and longer held-out horizons separately.

## Selection, confirmation, and acceptance

Select **one** method/configuration on development data against the strongest
eligible non-ID baseline. Fix either QUALITY or EFFICIENCY before confirmation.
Each confirmation arm gets its own manifest binding specification, complete
controller/quality provenance, data, and native config. The CLI writes a consumed
claim before opening confirmation rows. Interrupted runs remain claimed: document
an amendment and an independently generated bank instead of silently repeating.
Hashes/claims are local process safeguards, not a tamper-proof institutional registry.

Proposed QUALITY criterion: success improvement >= 0.05, paired 95% lower bound > 0,
and every protected family/stratum lower bound >= -0.02. Proposed EFFICIENCY
criterion: actual end-to-end time saving >= 0.15, overall and protected quality
lower bounds >= -0.02. These are proposed thresholds, not predicted gains.

The analysis pairs exact examples and bootstraps parent clusters within families.
Family/stratum protection uses Bonferroni-adjusted intervals. A zero-variance
bootstrap is widened using an unseen-event guard, so ten identical outcomes cannot
spuriously establish a two-point retention margin. Training-seed uncertainty is
not fully estimated by this bootstrap: it conditions on measured seeds. Run at least
three native training seeds, report them separately, and confirm the chosen artifact
with fresh parents. Repeat hardware timing; the current efficiency calculation is
a point estimate, not a confidence interval for latency.

Synthetic and replay evidence cannot become a native qualification. Even a
statistical candidate remains `native_qualified=false` pending custody, parity,
retention, external-verifier, and execution audits. Evidence labels and equal-budget
declarations are not automatically trusted or authenticated by the analysis code.

## Cost and retention accounting

Include quality-feature extraction, quality-head inference, router cost, adapter
switching, cache misses, model calls, tokens, search expansions, and retries.
Amortized model/adapter loading is reported separately. Do not count all-expert
collection as one selected call or treat replay costs as end-to-end measurements.
The quality prediction helper times the head computation only; feature extraction
must be added by a native integration and checked by the timing audit.

Protect each qualified domain and a fresh off-task language suite. For native
promotion add first-token KL, longer generation agreement, exact transition states,
rollout regret, valid-plan rate, and worst-stratum success. These application-specific
retention/verifier measurements are required follow-up work, not implemented by
substituting synthetic scores. Maintain unmerged FP32 requirements for registered
Titan-Sparse adapters and validate the native loader against its reference first.

## Stop conditions

Stop native execution on invalid manifests, dtype/shape mismatch, missing qualified
alternatives, nonfinite loss/gradients, or insufficient free VRAM. Abort rather than
silently reducing precision or altering the checkpoint. Stop prioritizing ID if
integral-only is just as good, actual quality worsens, the oracle shows no routing
opportunity, or apparent improvements disappear under matched budgets. Keep
recurrence/halting and full backbone MoE conversion outside this branch.

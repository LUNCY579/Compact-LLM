# Prospective CPU H3: local fixed-scale sensitivity

This explicit CPU FP32/eager successor preserves the existing H3 question and
seven-prefix population. It is a distinct initially authorized mechanism
experiment, not a bulk restart, runtime-debug retry or controller revision.
The scope finding in `audits/cpu_H3_authority_feasibility_review_001.json`
authorizes preparation, not native dispatch. Old GPU H3, sensitivity, historical
CPU refusal and source/runtime guards remain unchanged.

The question is worth resolving because positive trajectory coherence does not
give signed task progress. Provisional bulk DEV has unit/fixed1 69/192 versus
TAPS/random 67, finite oracle 70 and failed preservation. Only 0.521 percentage
points of finite-grid headroom is available, below the five-point practical margin.
No new controller fit is justified. A native local derivative can test whether
the energy-based scale direction agrees with a real suffix/readout loss direction;
it cannot explain the bulk CUDA fault or establish semantic improvement.

## Fixed operator and derivative

For the input-only prefix x, let s0 be the state after the unchanged ordinary
whole L12-L13 pass. Recompute the same frozen block with unchanged native
conditioning to obtain p2 and the raw proposal d2 = p2 - s0. The externally fixed
controlled state is s(alpha) = s0 + alpha*d2, with the exact p2 value returned
at alpha=1. Unit value must not be reconstructed using cancellation arithmetic.
The ordinary first-pass raw whole-block observation d1 is recorded unchanged.
Raw means before experimental alpha; existing native gates remain inside the
frozen block and are neither removed nor multiplied twice.

Let R denote actual remaining native L14-L27, final normalization, tied output
projection and custom abstention-aware head. With offline target y, define
L(alpha) = CE(R(s(alpha)), y), using only the final current-prefix logit position.
Frozen parameters remain requires_grad=False and grad=None. After controlled
state construction, detach/clone it into the sole requires_grad=True leaf.
autograd.grad(loss, state_leaf) supplies G; no optimizer or parameter backward.
For this fixed schedule, dL/dalpha = dot(G, d2). This is a chain-rule identity,
not a guarantee that coherence or recurrence improves task performance.

The custom UNKNOWN token51378 offset 5.156249985098839 is part of R. A constant
logit offset has zero direct derivative, but changes softmax probabilities and
therefore the CE gradient. Omitting the head changes the measured question.

At alpha=.99, 1 and 1.01, recompute three entire native input-to-suffix rollouts.
Do not reuse a saved controlled state for finite-difference counterfactuals.
Compare adjoint a=dot(G,d2) at unit to c=(L(1.01)-L(.99))/.02. The fixed gate is
abs(a-c) <= .001 + .03*abs(a); do not relax or substitute a different denominator
after observing CPU rounding. Treat abs(a)<=.001 as neutral/indeterminate for
sign analysis, including cases where the mechanical absolute tolerance passes.
Central differences approximate the local derivative and amplify FP32 rounding;
an actual discrepancy is retained as failed qualification, not repaired by tuning.
This local direction does not establish improvement at the much larger TAPS
scale. Its descriptive recommendation remains clip(1+.275*balance,.8,1.2),
with no fitted coefficient or online target use.

## Population and target firewall

Bind all original 16 allowlisted DEV rows before load. The same frozen rule gives
14 input prefixes<=256, two long exclusions, seven eligible inherited canonical
targets and seven short NOT_APPLICABLE target-missing rows. Do not infer a target
from keyword/choice scorer rules or select prefixes by model outcome. Encode only
the first inherited canonical token as offline y. Target IDs never enter input,
proposal, attention/positions, controller history or loss-independent state.
CE receives y after logits have been computed. No full-gold NLL, continuation
teacher forcing, generation, A5 pool access or controller fitting is admitted.

The loss is a first-token surrogate on exposed same-source data with at most five
provisional scenario clusters. It need not reflect complete-answer correctness,
semantic abstention or the six-category bulk bank. Secondary sign/rank observations
remain exploratory; no out-of-family prediction or cross-device gradient pooling.

## CPU contract and qualification

Use a new CPU-native source/entry/supervisor/admission, explicit device=cpu,
FP32 floating parameters/buffers/activations, batch1/nonpad/full prefix<=256,
eager attention and no cache. A state-gradient context is explicit and confined
to this diagnostic wrapper. Do not use a GPU testing flag, disable GPU device
guards, modify the old sensitivity wrapper or silently fall back between devices.
CPU/GPU kernels may differ in FP32 rounding and derivatives; report CPU as a
new host-domain measurement, not exact GPU replication or a speed comparison.

Fresh CPU-native qualification must establish ordinary/noextra identity, unit
legacy equality, .99/1/1.01 value parity against the separately qualified CPU
fixed-scale inference wrapper, actual
suffix/head adjoint/finite-difference agreement, representative and nongold
maximum256 gradient shape, finite logits/loss/proposals/gradients, unchanged state
and assets, cache rejection including object with use_cache=False, masks/positions
and L0-L2 one-shot behavior. Restore exact methods/hooks after normal and forced
exception exit. Verify params remain frozen/gradNone and only state is a leaf.
The expected Python exception does not qualify real poisoned-context cleanup.

Keep unchanged historical CPU floors: parent available RAM>=8.4GiB,
post-import>=7.4GiB, worker RSS<=6GiB and running host reserve>=1GiB. Source-bind
the exact canonical checkpoint/tokenizer/custom head/runtime. A metadata-only
preflight PASS and a fluctuating RAM snapshot qualify neither weights nor tape.
Use sole-worker lock, exclusive output, suspended-before-Job assignment, owned
descendant termination, progress-aware parent deadline and fault-safe receipts.
No other workload/environment/model asset changes.

## Complete count and CPU forecast

Retain the prospective 45 model-forward attempts: 44 expected complete plus one
intentional partial restoration exception. Two precompute cache invocations make
47 API attempts; there are nine state adjoints (two qualification, seven target
measurements). The seven target rows alone require 21 full recomputed forwards
and seven adjoints. Account for every qualification, failed/partial and reused
already-charged baseline call. The loose attempted position bound is 47*256=12032,
not a claim that cache rejection or partial work processes that many positions.
The implemented complete ledger checks 18 value attempts (17 complete/one
partial), 27 complete diagnostic forwards and two cache rejections. Completed
positions are 11*84 + 12*256 + 3*(84+37+30+34+30+27+32) = 4818.

The entire worker has a 1,800-second parent ceiling plus a 1,800-second global
finalization reserve. Historical CPU timings may inform a conservative labelled
estimate; GPU timings/adjoint allowances are not measured CPU evidence. Before
the seven target rows, obtain actual representative and maximum CPU timings and
memory including the gradient tape, imports/load/hash/instrumentation, then admit
the complete remaining fixed plan. A small passing pilot cannot authorize a
truncated matrix. Preserve costs/nulls and stop incomplete if forecast, memory,
identity, finite-difference or other required gates fail; no retry or automatic
device switch is available. Root alone may launch after independent resulting
source/test/admission review and fresh readiness.

# CDE-001 research protocol and native handoff

## Scientific claim ladder

Separate mechanical recurrence, useful recurrence, and efficient deployment.
Carrying a state is not evidence of a quality improvement. A quality improvement
at greater work is not evidence of a better cost frontier. Consistency across
steps is not correctness. An oracle's ability to find a correct candidate is not
the ability of the deployed model/selector to choose it.

The initial apparatus is for same-prefix readouts. It does not model token-level
cache semantics or permit KL comparisons across different generated prefixes.
A native collector must distinguish full-answer evaluation from per-token,
teacher-forced diagnostics. Per-token loss improvements do not establish better
free-running answers. Never attach an oracle answer to a deployment policy.

## A. Fixed-parameter rescue screen — first native priority

Bind one exact recurrent checkpoint (the 344M-class student OR the approximately
596M family member, separately), tokenizer, runtime revision, dtype, decoder
contract, and output budget. Family names or matching tensor shapes are not
sufficient identity. Do not turn ordinary dense Qwen layers into recurrence by
calling them repeatedly without validating the architecture and cache contract.

Start with 128 fresh, parent-disjoint DEV problems for triage, not a significance
claim. Prefer short, exactly checkable arithmetic, logic, constrained planning or
sandboxed code. Freeze maximum generated tokens and prompt formatting. No sealed
Rineya test rows may be used for selecting depths, thresholds or data labels.

Use R in {1,2,4,8}, initially K=1, plus:

- matched nonrecurrent extra-compute baseline;
- same-weight carry versus reset-state control where the host supports it;
- matched nonrecurrent adapter/capacity control when training adds capacity.

Log every depth, including skipped reporting depths: first-correct depth,
final-depth correctness, reversions, answer identity, logit/output discrepancy,
normalized state movement, gate values, forward work, measured FLOPs where
available, coda/readout cost, prefill/decode cost, selection/verifier cost, elapsed
time and peak RAM/VRAM/KV bytes. Missing depth observations cannot establish an
adjacent-depth stopping rule.

A nearly closed gate, changing-but-wrong state or absence of a reproducible
late-rescue population is a reason to stop the tested configuration and diagnose
the host/training. It is not a proof that all recurrence is useless. Do not train
a controller around an ineffective host merely because its outputs look stable.

## B. Depth versus independent width

Cross R and K in {1,2,4,8}. The equal-RK illustration (8,1), (4,2), (2,4), (1,8)
is only a nominal comparison. Encoder/prefill sharing, decoded token counts,
readouts, verifier execution and batching change actual cost. Report separately:

| Endpoint | Required interpretation |
|---|---|
| Mean final-exit accuracy | Raw exit quality averaged over attempts. |
| Selected final-exit accuracy | Quality using a deployable, frozen selector. |
| Oracle best at final depth over width | Diagnostic upper bound for final candidates. |
| Oracle best over depth AND width | Diagnostic upper bound for all observed candidates. |
| Measured end-to-end cost | Native latency/memory/work including selector costs. |

The stdlib apparatus implements first and deterministic-majority selection, not
a learned or executable verifier selector. An identical normalized answer for
the same parent must receive a consistent evaluation verdict. Verification-based
selection belongs in the later verified-resampling integration, not a function
that reads test correctness.

Use nested candidate sets. `nested_seed(base_seed, parent_id, rollout)` excludes
R and K, so widening does not change earlier rollout seeds. Actual host execution
must also preserve prefix RNG streams: equal seeds alone do not prove that
batched stochastic kernels produce nested candidates. Record realized candidate
hashes. State perturbation and text sampling are different methods and must have
separate arms. Never give different noise levels to different depths unnoticed.

`cell_metrics` sums supplied cumulative work; it does not assume that parallel
wall time is the sum, maximum or RK-scaled time. It does not account for shared
prefill automatically. Native collection must explicitly charge shared and
per-candidate components once and measure each requested cell independently.

## C. Training reliable intermediate exits

From one identical qualified initialization and the same trainable parameter
set, compare five arms implemented in `losses.py`:

A: full-depth task CE; B: CE at an explicitly sampled depth; C: mean CE at all
exits; D: full-depth CE plus adjacent-depth distillation; E: full-depth CE plus
maximum-depth-to-all-shallow distillation.

For arm D, with maximum depth M, aligned next-token targets y_t and a response
mask, the proposed LM objective is

    L = CE(p_M, y) + lambda/(M-1) * sum_r KL(stopgrad(p_(r+1)) || p_r).

The implementation applies temperature to both distributions and multiplies KL
by temperature squared. It averages over active response positions and adjacent
pairs, and sums over vocabulary. Prompt/padding positions are excluded. Targets
must already be causally shifted exactly once. All exits must use the SAME
prefix, tokenizer and vocabulary at each supervised position.

Stop-gradient freezes a teacher *within that KL graph*, not across optimizer
steps in a weight-tied network. Retain full-depth task supervision and monitor
full-depth quality. Distillation can converge to identical wrong predictions;
a zero KL is not a success criterion. No teacher confidence or output agreement
may substitute for supervised/verifiable correctness.

The sampled arm takes an explicit sampled_depth to avoid mutating global RNG.
The initial comparison must preregister the depth distribution, lambda grid,
seeds, token budget and training-compute budget. Report data-matched and
compute-matched comparisons separately. A host can truncate execution for arm B;
passing a full-depth tensor to this function does not produce that saving by
itself. Detaching teachers also does not eliminate shared recurrent activations.

Only after individual arms pass should a small factorial compare depth training
with existing Rineya skill updates. Freeze skill-mixture coefficients before
heldout evaluation. Report acquisition AND retention, including general
language, reasoning, existing skill sets, long-context and abstention endpoints.
No additive gain is assumed from the names or orthogonality of interventions.

## D. Calibrated stopping

The prototype policy checks adjacent prediction equality and total-variation
similarity for consecutive pairs. With patience=2 its earliest early exit is
three, not one. This conservative rule is an original apparatus choice, not an
exact FlexLoop controller reproduction. It must be calibrated against a fixed
maximum-depth baseline.

`calibrate` accepts DEV traces only and binds the checkpoint consistency ID,
maximum depth and calibration parent IDs. Replay rejects checkpoint/depth changes.
`replay(..., heldout=True)` requires TEST traces and rejects overlap. Select a
policy under preregistered quality-loss and missed-rescue budgets; the apparatus
defaults to zero allowed DEV degradation and zero missed rescues, falling back
to maximum depth whenever needed. These defaults are not statistically certified
risk bounds. Full-depth training/evaluation remains mandatory.

A missed rescue means the selected exit is wrong and at least one later exit is
correct. The reported rate uses all parents, balanced across their rollouts.
Parent balancing prevents heavily sampled parents from dominating the point
estimate; it does not supply confidence intervals. Before native efficacy claims,
use paired parent-level confidence intervals and a larger untouched confirmation
set, and retain per-family results and repeat-run variance. Do not choose a
policy on those intervals after viewing test outcomes.

`replay` is counterfactual offline analysis. It examines an already collected
full-depth trace and reports supplied work at the selected exit; it cannot
establish actual adaptive latency, memory release or online stopping correctness.
Do not publish replay work savings as wall-clock speedup. A native implementation
must compute only past/current signals online and measure observed stopping,
including the cost of generating readout logits and evaluating the signal.

## E. Self-relabeling — independent data intervention

The implemented `propose` function replays both the previous solution and the
candidate through one supplied verifier callback. It rejects non-TRAIN or
protected parents, invalid new solutions, score ties, unchanged solutions, and
mismatched generation/deployment decoder-contract digests. A valid candidate can
replace an invalid solution; otherwise it must strictly improve one declared
higher-is-better objective. The result contains provenance hashes and verdicts,
not an overwritten dataset. Treat actual file/version storage as a separate
implementation and authorization step.

For binary correctness, first prioritize new solved examples. Do not rank two
correct explanations by a hidden preference for brevity or style. Valid final
answers do not certify intermediate rationales. A verifier must be audited for
exploitability, timeouts, public/private test separation and correctness; code
verification needs an actual sandbox. This module never executes model-written
code and a verifier hash is not proof of a sandbox or soundness.

Compare unchanged-label continuation, ordinary sampling plus verified labels,
and (only if qualified) recurrent search plus verified labels. Record search,
verification and training costs separately. Evaluate the resulting model at a
fixed low inference budget WITHOUT its label-search procedure to distinguish
learned improvement from search-time assistance. Preserve every previous label
version and exclude protected evaluation parents and near-duplicate families.
Parent-ID checks alone cannot detect semantic duplicates; that audit is upstream.

## Native integration checklist and blocking conditions

The next implementer should bind, not bypass, existing qualification machinery:

1. Record checkpoint/tokenizer/runtime hashes and obtain the exact current
   campaign authorization/loader receipt. Preserve native memory guard semantics
   and process-tree accounting. An 8-GB GPU or small parameter count is not an
   automatic fit guarantee; no hardcoded capacity assertion is made here.
2. Implement a native collector using the audited recurrent host. The LoopSpec
   protocol in the parent has `prefill/start/advance/readout/cache_bytes`; those
   names define an interface, not a completed CDE adapter.
3. Prove independent full-prefix versus incremental parity at every exit, plus
   RoPE positions, depth-specific GQA/KV ownership, coda/final-normalization,
   mixed-depth batches, reset/carry controls and stopped-token history semantics.
4. Export same-prefix diagnostic traces separately from full-answer generations.
   Extend CDE's lightweight labels to an actual hashed run manifest and verified
   native receipts. Add heldout parent/near-duplicate checks and measured cost
   receipts. Never mark fixture traces native merely by renaming checkpoint_id.
5. Run the smallest authorized baseline/rescue screen; publish paired results
   and null outcomes before enabling training or a native stopping policy.

Gate summary: G0 source/method contract; G1 exact native host and cache parity;
G2 fixed-depth usefulness versus controls; G3 robust intermediate-exit training;
G4 frozen stopping quality plus actual cost; G5 retention and independent native
confirmation. G0 is limited to abstract-verified, original adaptation in this
commit; G1-G5 are NOT passed. The current native-preflight always blocks.

## Whole-program boundaries

This branch should answer a compute-allocation question, not absorb all Rineya
research. Use the existing sampling, distillation, loader and retention programs
as distinct controls/integration points. Keep architecture efficacy, skill
composition, memory/cache interventions and inference scheduling factorizable.
A later measured success can justify a controlled combination. Until then,
continue reporting failure modes and checkpoint-specific evidence rather than
claiming a universal small-model intelligence multiplier.

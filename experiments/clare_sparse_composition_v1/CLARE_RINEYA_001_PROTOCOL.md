# CLARE-RINEYA-001: Capacity-Aware Sparse Skill Composition

## Status and falsifiable question

OFFLINE IMPLEMENTATION / SYNTHETIC QUALIFICATION. Native qualification BLOCKED.
The mechanism and proposed controls are frozen here; a native efficacy study is
NOT fully registered until the objective, tensor pool, data enrollment, token
budget, runtime and actual artifact identities are bound before execution.

Can learned support allocation outperform equally sparse RANDOM disjoint masks
in retention or held-out composition without starving skill acquisition? Exact
support separation and zero parameter dot products are necessary mechanics, not
success criteria. Nonlinear transformer interactions can remain large or harmful.

Model soup combines already-learned models/deltas. This experiment controls where
the next skill can learn. It retains the additive representation
Delta_s = M_s * D_s and the shared-base lineage rather than introducing task IDs,
MoE routing, separate serving models or inference-time skill selection.

## Parameter space, stages and exact capacity

One immutable base and one explicit, sorted eligible list of contiguous FP32
model parameters are shared by all arms. Every other parameter stays frozen.
No automatic whole-model allocation and no LoRA-factor orthogonality claim.
Select globally across this pool: no enforced layer-specific partitioning.

For n remaining coordinates, allocate
k = min(n, max(1, round_half_up(n * (1-rho)))) when n > 0; k=0 otherwise.
Select exactly k. Learned ties are broken by parameter-name/row-major index.
Capacity exhaustion REFUSES a skill instead of silently reporting a zero update.

Discovery minimizes skill_loss + lambda * L1(change from the incoming state)
on eligible coordinates. Select movement magnitude from available coordinates.
Then restore the EXACT incoming state, discard discovery values and optimizer,
and retrain with ordinary skill loss on the selected mask. Mask gradients before
AdamW.step; project frozen coordinates back after the step and clear corresponding
optimizer moments. Freeze/claim only after a successful complete skill. Exceptions
roll back eligible model state, allocation, Python/Torch RNG and model modes.

The stored canonical increment is FP32(trained - start). Composition is explicit
FP32 accumulation in registered order; tiny subtraction/readdition roundoff is
recorded as `fp32_recomposition_max_abs`. This does not weaken the exact discovery
reset or the frozen-coordinate invariant. There is no norm renormalization.

## Arms and budget fairness

| Arm | Incoming state and final update |
|---|---|
| A dense_sequential | Current shared state; dense retraining; no protection. |
| B independent_add | Each skill starts from the immutable base; dense deltas sum at coefficient 1. |
| C magnitude_sparse | Each skill starts from the immutable base; dense retraining then exact magnitude top-k pruning; sum deltas. |
| D random_disjoint | Current shared state; seeded random unused coordinates; reset and masked retraining. |
| E learned_disjoint | Current shared state; discovery-selected unused coordinates; reset and masked retraining. |

Every arm pays the same configured discovery + retraining step budget on the same
ordered atomic examples. A/B/C/D discard their cost-matched probe; ONLY E uses
it for allocation. This isolates mask information instead of silently giving E
extra optimization compute. Report probe, retraining, token and wall-clock costs
separately. Sparse masking does not make dense autograd or AdamW state sparse.
The actual native token budget must be bound, not inferred from equal step counts.

D/E have identical cumulative quotas. A/B have dense actual masks; their shadow
capacity ledger is ONLY a quota reference, never claimed as sparsity. C uses that
same per-skill quota but its independent supports may overlap. Allocated capacity,
actual mask counts and nonzero deltas are separate fields. Historical `.5/.5`
soup and additive coefficient-1 composition remain distinct operations.

## Small proposed comparison, not a sweep

Primary A/D/E; B/C implemented for follow-up controls, not automatically launched.
Initial rho=.80, order R,U,S,P, seeds 17,29,41. Alternative order P,S,U,R is
configured as a later sensitivity check, not an automatic 24-permutation run.
Supported rho values .80/.90/.95/.975 must not be tuned on held-out answers.
Proposed study configs use 16 discovery + 64 retraining steps per skill, AdamW,
LR=.001, lambda=.0001 and weight decay=0. These are provisional compute controls;
there is no assertion that 64 native steps suffice for acquisition. Failure to
acquire makes the composition comparison inconclusive, not evidence of success.

The separate executed CPU demo uses four regression stand-ins, one seed, 20
coordinates, 4+8 steps and LR=.01. It is NOT this native comparison. Native smoke
uses only 1+1 steps and prompt next-token plumbing, then rolls back. Its objective
is NOT a native skill-training objective and passing it does not authorize a run.

## Data and leakage boundary

Read existing MAXRL-COMP-002 atomic TRAIN manifests only. No regeneration/copies,
bridges, replay, answerability or common warmstart in this first comparison.
R=reference resolution, U=state updates, S=exact subset selection, P=precedence.
Policy sees only system/user messages. A trusted external native loss integration
must keep public verifier specs outside actor inputs, must not import audit
oracles, and must qualify its response/EOS/log-prob masking before use.

Primary unseen signatures supported by inspected corpus contracts: RS, RP, UP,
RUS, RUP, RSP, USP, RUSP. RU/US/SP are separate secondary bridge-signature reports:
not trained in this arm recipe, but not globally absent from corpus TRAIN.
DEV/TEST/CHALLENGE are exposed; they are not sealed. Never select masks, rho,
module pools, coefficients or thresholds using their answers. Exact train/eval
parent disjointness must be rechecked against actual enrollment ledgers on host.

## Four endpoints, plus diagnostics

Acquisition: each R/U/S/P before training and immediately after its own stage.
Retention: all four skills at base and every stage, including source-correct
losses/gains; final-minus-acquisition and forgetting from best observed stage.
Composition: paired accuracy per unseen signature, gains AND damage, parent
counts and uncertainty; no aggregate score hiding signature failures.
General retention: reuse existing guard/retention banks, enroll base predictions
first, retain identical row sets and scorer/tokenizer/decoding metadata.

For every delta pair compute fresh norms, dot/cosine, actual nonzero support
intersection/Jaccard and sign agreement only on overlapping nonzero coordinates.
Zero norms/empty denominators yield null, not fabricated perfect similarity.
Per parameter/layer report eligible, selected, remaining capacity, norm and energy.
Allocated masks are saved explicitly even when selected values train to zero.

At identical inputs/token positions, logit nonadditivity is
I = logits(base+A+B)-logits(base+A)-logits(base+B)+logits(base).
Report raw and vocabulary-centered RMS and raw maximum absolute value. This is
not a correctness, causality or independent-computation metric. Compare matched
base/A/B/A+B outputs with the actual task scorer to identify functional damage.
Optional bounded offline singular-value energy is diagnostic only; no rank
truncation is mixed into training. Gradient-alignment collection is not implemented.

`evaluation.paired` rejects missing/duplicate rows, metadata or identity drift and
nonboolean correctness. Resample whole parent clusters, not sibling rows. Report
per-seed paired effects and variability; descriptive bootstrap intervals are not
automatic selection or multiplicity-adjusted confirmation gates. No early success
rule based on support, cosine, synthetic losses, or one favorable seed.

## Deferred extensions and admission

F shared/private is design-only; `shared_private` adds a shared delta ONCE, not
once per private skill. Before F training, fix shared/private budget accounting,
shared update schedule and retention controls. Gradient-aware coordinate reuse
is a future separate protocol. Neither is silently part of A/D/E.

Stages 3/4 require bound native model/tokenizer/runtime and data, reviewed module
inventory, optimizer/activation memory costs, mask/reset smoke, and a validated
skill objective plus matched evaluation collector. No full-training CLI exists.
A hardware refusal is BLOCKED, not a negative scientific result. See handoff.

Interesting outcomes include E beating D with preserved acquisition/retention,
or exact separation with poor composition, or E approximately equaling D. All
are credible only after native matched, repeated-seed observations. Until then:
MEASURED = CPU mechanics; DERIVED = quota/geometry consequences; HYPOTHESIZED =
native retention/composability improvements.

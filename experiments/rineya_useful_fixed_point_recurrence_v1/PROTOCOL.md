# Research protocol — FP-RINEYA-001

## Decision and scope

Complement existing TAPS, ARMR, compute-depth elasticity (CDE), DEQ/FGD evidence-
function work, LoopSpec and MiMo post-training. Do not replace their findings,
merge sibling experiments silently, or block the primary 0.6B skill-acquisition
program on recurrence research. Preserve this branch even if the first screen is
negative. No frontier-intelligence or published-paper reproduction claim.

The native reference is the audited historical ~0.6B Supa host. The current MiMo
training anchor must be independently bound and qualified for later training.
Do not confuse those checkpoints because both are called Rineya 0.6B.

## Hypotheses

H1: repeated computation can repair answers rather than merely stabilize them.
H2: input carryover/anchor interference is a contributing failure mechanism;
matched orthogonal injection improves on matched additive injection.
H3: useful recurrence can be trained across a *qualified* range of depths without
spending excessive supervision on unhelpful depths.
H4: only after H1-H3, some retained trajectory/cache work can be reduced with a
preregistered quality tolerance and measured total costs.

No hypothesis implies another. Fixed points can be wrong, and a zero update
can be gate suppression. Useful transient answers can precede harmful convergence.

## Equations and implementation distinctions

The interface forms `q = softplus(dt) * (x + U V x)` and
`v = exp(-softplus(dt)*exp(a)) * h`. Its target is either `v+q` or
`v - q*(q.v)/(q.q+eps) + q`, independently per token. Dot products accumulate in
FP32 (FP64 for FP64 inputs). `x` is the original first L12 input, not a later
mutated state. Gradients flow through anchor and carry terms when applicable.

The wrapper uses `c = h+tanh(blend)*(target-h)` and then
`h_next = h + alpha*(F(c)-h)`. The first ordinary block is untouched.
`blend=0` preserves raw recurrence; `alpha=0` preserves the intact output while
still measuring the same extra core-call budget. The first training step usually
updates only blend, followed by Up and then Down. Monitor opening explicitly.

This interpolation and low-rank identity-plus-update anchor adapter are our
adaptations, not the paper's full-matrix injection. Additive and orthogonal arms
must start from identical parameter tensors and have the same parameter count.
Raw recurrence has no added parameters and is a separate capacity baseline.

Diagnostics distinguish `F(c)-h` (raw applied-map proposal) from `F(c)-c`
(operator residual). A small scaled update is not an ungated fixed-point proof.
All runtime records detach and reduce tensors; no retained full hidden trajectory.

## Stage 0: identity and apparatus

Pin checkpoint, custom loader, tokenizer/template, all inherited source files,
exact runtime, source revision, datasets and resource limits. Run the inherited
canonical baseline protocol. Separately qualify the new bridge against unwrapped
execution, zero-start/raw equality, restoration after normal and failing calls,
L0-L2 one-shot, whole-block order, tensor output, masks/positions and no cache.

Initial native scope: one extra L12-L13 block application, at most 16 approved
DEV cases, CPU FP32 and frozen weights. No extrapolation to deeper recurrence.
Fixture pass -> apparatus code reviewed, NOT native gate pass.
Native screen pass -> apparatus qualified on the bound host, NOT efficacy.

## Stage 1: useful-refinement diagnosis

On frozen DEV cases, measure raw/applied update norms, relative change mean/max,
operator residual, injection magnitude, input alignment, suppressed-update and
stationary fractions, state RMS, first-token logit KL, answer corrections/damages,
format/safety failures and complete core-call/token/time costs.

The initial screen executes unwrapped, wrapped one-pass, raw two-pass,
zero-additive, zero-orthogonal and restored arms. Zero-start arms are expected
to be identical to raw; equality cannot establish the utility of either injection.
Do not claim asymptotic convergence from only two applications.

Independent future approved probes may estimate local directional sensitivity
or VJP/JVP behavior; they are NOT implemented or licensed by this release.
Hold input prefix fixed for trajectory/logit comparisons. Free-generation
trajectories with different prefixes cannot be compared as same-input dynamics.

## Stage 2: bounded co-design (reference engine exists; native training disabled)

After explicit host/data/resource qualification, use a small deduplicated
approved TRAIN slice of the 0.6B curriculum. Prefer verifiable SFT initially.
Bind the warmed anchor before deciding whether this historical Supa host transfers.

Compare raw fixed recurrence, trainable additive and trainable orthogonal
interfaces. Use identical frozen core, first pass, tokenization, examples, ordering,
seeds, steps and update scale. Record actual compute, not merely optimizer steps.
An interface may not be inserted into the canonical model's parameter registry.
Only the explicit external interface enters the optimizer. Save adapters separately
into an exclusive run directory; never call save_pretrained on canonical weights.

Keep full backpropagation at this shallow depth. Stage-2 training receipts must
cover token-label masking, scorer parity, gradient checks, base weights/buffers,
rollback, memory and training-data authorization. `fit_interface` supplies a
bounded optimizer engine, NOT those missing native qualifications or a dataset
loader. Its closure must return an approved mean loss and supervised-token count.
Tensor version checks are not a cryptographic memory-state audit.

`admit_splits` validates TRAIN/DEV/RETENTION together for normalized exact-prompt
and cluster overlap. It does not detect semantic near duplicates; perform that
review separately. Never call an optimizer on DEV/RETENTION/CHECK/FINAL/sealed data.
Report seeds separately. Do not tune alpha, steps or thresholds on protected data.

## Stage 3: qualified broader depths and training priors

A support {1,2} has its entire distribution fixed by a fixed expected depth.
Therefore the CPU prior requires at least three support points, but that does NOT
license native R>=3. Native bridge intentionally rejects those requests.

After separate deeper-host qualification, compare fixed target depth, a fixed
sampled-depth distribution and learned distribution with entropy, matched on
actual cumulative compute. Add no-entropy ablation only within approved budget.
Record achieved mean depth, entropy, depth counts, CE and actual core calls.
The current reference uses inverse-perplexity REINFORCE, EMA baseline, mean-depth
penalty and entropy but omits upstream variance rescaling. Its mean is penalized,
not exactly constrained. It is not the published optimizer reproduced numerically.
A training-depth prior is not a question-dependent stopping controller.

Only after separate successes run injection x depth-prior interaction experiments.
Then test TAPS update scaling at fixed operator/budget; never attribute a gain
over alpha=1 to adaptive scheduling without the best fixed-alpha comparator.

## Stage 4: conditional efficiency studies — not implemented natively

Truncated BPTT: first compare to full BPTT and activation checkpointing at the same
depth; preserve live input-anchor gradients in retained steps. Measure gradient
agreement, quality, retained activations, parameters, optimizer state and time.
The reference finite-recurrence function supports full/short windows but does not
prove gradient agreement on native Rineya. Few recurrences offer little saving.

Terminal KV: establish a separate full-cache reference and incremental/chunked/
prefill parity first. ARMR is cache-free, so no duplicated KV exists to remove.
Equal-geometry hypothetical bank counts: full = L+k(R-1), terminal = L. The
paper-like L=4,k=2,R=5 gives 12 vs 4; Rineya L=28,k=2,R=2 gives 30 vs 28 (6.67%
KV-only saving), not 3x total-memory reduction. This release only calculates that
geometry. No cross-layer cache sharing or long-context comprehension claim.

Distilled prefill: need a useful recurrent teacher, extra-student memory accounting,
and quality-vs-latency validation. Endpoint RL: bind rollout/parameter versions,
endpoint residuals, direct gradient paths, and full-replay/VJP comparisons.
Do not start with implicit gradients or assume JVP tooling implements adjoints.

## Outcome and promotion rules

Primary usefulness: paired net correction = (wrong->right - right->wrong)/N,
with separate retention and format failures. Use frozen task-specific binary
success definitions; report graded scores separately. Bootstrap paired clusters,
not seeds or correlated prompts as independent observations. A one-cluster CI is
omitted. DEV intervals are descriptive; no automatic confidence-based promotion.

Stop for identity drift, cache exposure, invalid values, bounds failure, accidental
canonical gradients/mutation, leakage, resource failure, or schedule restrictions.
Return BLOCKED when required evidence/assets are missing, not REJECTED. Preserve
null effects and coherent-but-wrong/zero-gated fixtures. More stationary but no
more useful is not a reasoning improvement and does not authorize efficiency work.

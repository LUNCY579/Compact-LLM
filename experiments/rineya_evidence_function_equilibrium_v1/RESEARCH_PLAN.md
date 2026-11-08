# EFE-001 research programme

Status: **proposed native experiments; CPU reference mechanics only**.
This is a Rineya-specific programme inspired by three papers, not a replication
of their benchmarks or a transfer of their convergence guarantees. Sources and
verification boundaries are in [SOURCES.md](SOURCES.md).

## 1. Evidence and the decision to make

The user-supplied analysis reports INDEP-001 acquisition of 235/384 for one update,
175/384 for a raw two-update sum, 219/384 for a shared scale, and 235/384 for independent
coefficients `(1.00, 0.25)`. It also reports a boundary endpoint of 69/112 -> 85/112,
retention of 39/47, and long-context counts of 159 versus 164/256. These are
**transcript-reported development results, not rerun or independently audited by
EFE-001**. The coefficient-selection bank had already been used. Neither equal
counts nor a nonsignificant difference establish noninferiority.

The reported near-zero weight-delta cosine does not ensure behavioral independence.
The useful next question is whether failure comes from poor coefficients, inadequate
available directions, conflicting objectives, or inadequate computation. Do not
collapse these into an assumed sub-1B architectural ceiling.

The inspected integration branch's `REHEARSAL_COMPOSITION.md` explicitly bars the
Long-Context Demo from training, selection, and confirmation; report-only historical
results must stay report-only. It records different retention tradeoffs across
rehearsal mixtures. EFE-001 does not promote a mixture from that spent instrument.
The supplied analysis additionally flags format/semantic/abstention confounds,
weak acquisition hidden by retention gates, and an indeterminate module-localization
instrument. Preserve those cautions; they are not new EFE measurements.

## 2. Shared protocol: freeze before measuring

Every native session must bind exact source/delta/tokenizer hashes, code commit,
runtime, precision, data content hashes, family/parent identities, seeds, scorer
versions, stopping rules, memory limits, and a new external result directory.
Resolve all `plan.json` prerequisites before execution. The manifest is a queued
plan, **not a completed preregistration**: sample sizes, power, retention margins,
selection budgets, and native resource limits remain unbound.

Partition train, calibration/selection, preservation-calibration, and confirmation
by content and parent family, not just row IDs. Audit against existing quarantine
registries and prior use. Exact hash disjointness does not detect paraphrase leakage.
The CPU guard checks only supplied declarations; a host must recompute hashes and
check provenance. Confirmation outputs must not be exposed to coefficient fitting,
hyperparameter selection, tolerance selection, or stopping-policy optimization.

Primary reports separate per-family acquisition, source-correct losses and gains,
semantic correctness, literal format correctness, and appropriate abstention.
Report both retained/source-correct and all-example denominators. Preserve paired
row-level outcomes and seed identities; do not treat multiple seeds of a prompt
as independent new examples. Predefine paired confidence intervals, multiplicity
handling, a practically meaningful retention margin, and an acquisition floor.
No absence-of-significance shortcut to retention success. A gate cannot pass merely
because the candidate learned less.

Resource reporting includes peak allocated memory, cache tensor bytes, workspace,
prefill and decode times, end-to-end latency, generated-token counts, and failure
rates. Compare matched compute/latency and matched parameters separately rather than
claiming either implies the other. Never silently change precision, truncate inputs,
reduce generation, or drop failed rows to fit a budget.

## 3. EFE-FUNC-001: existing directions before extra capacity

### Hypotheses

F1: frozen independently chosen coefficients can reproduce the development signal
on genuinely fresh admissible instruments. First compare source, each single skill,
raw sum, shared scale, and the frozen `(1.00, 0.25)` recipe. The historical pair is a
candidate to confirm, not an optimum to reuse as fact.

F2: coefficients informed by output effects improve the acquisition/retention
tradeoff over independent grid search and weight/norm-informed allocation at a
matched calibration and selection budget. Independent grid search is a required
strong control; beating only the damaging raw sum is insufficient.

F3: adaptive directions help only when existing directions cannot express the
needed change. Capacity growth is a conditional second experiment, not F2's default.

### Local objective and diagnostics (proposed Rineya design)

For frozen anchor `theta0`, collect directional effects without materializing an
LLM Jacobian: `A_new[:,j] = J_new d_j`, `A_keep[:,j] = J_keep d_j`.
Weight-space orthogonality `d_i^T d_j = 0` does not imply
`(J d_i)^T (J d_j) = d_i^T J^T J d_j = 0`.

Choose an output representation and metric before extracting derivatives. For
example, response-token logits centered across vocabulary avoid treating an
irrelevant shared logit offset as behavioral change. State explicitly whether
coordinates are full logits, fixed log-probability probes, or a sketch; a sketch
can hide missing directions. Equalize family, prompt, and response-token weighting
so long responses or large coordinate blocks do not dominate by accident.

Fit a finite surrogate:

`min_c mean((A_new c + eta g_new)^2)
       + lambda mean((A_keep c)^2) + mu ||c||^2`.

The CPU reference uses augmented least squares, supports rank-deficient bases,
and reports unconstrained span error separately from the residual after retention
and ridge penalties. A zero target produces an undefined relative residual, not a
misleading zero-error capacity certificate. Fitting scalars is optimization/model
selection, even when no checkpoint weights are trained in that step.

The empirical diagnostic `rho = ||(I - Pi_A)g|| / ||g||` is not the adaptive-FGD
paper's certified function-space error bound. Low unconstrained error but worse
penalized fit suggests objective conflict or regularization cost, not necessarily
missing capacity. High reproducible residual motivates an added direction only
after derivative/metric/precision errors have been excluded. Preserving source
outputs indiscriminately can preserve source mistakes; specify which behavior
must be kept and which errors may be repaired.

### Native qualification and falsification

Validate JVPs against central finite differences at multiple small perturbations
on the exact anchor, then compare local predicted output changes with actual
nonlinear checkpoint perturbations. Use fresh calibration probes; account for
all derivative and trial-forward costs. A failed linearization gate blocks fitting
for native interpretation. The CPU fit is unbounded; native use requires a frozen
coefficient box/trust-region policy or a separately qualified constrained solver.
Do not simply clip its output and call that the same optimizer.

Fit on calibration, freeze coefficients and regularizers, export the actual
nonlinear composition, test intended export precision, then confirm. Teacher-forced
loss is diagnostic, not autonomous-generation competence. FP32 subtraction cannot
recover precision already lost in BF16 checkpoint storage. Count repairs and damage
per skill, not just total improvements.

F2 fails at the specified resource budget if it cannot beat the qualified independent
coefficient baseline on the preregistered tradeoff. F3 compares fixed small basis,
fixed larger basis, residual-driven growth, and random growth with matched objectives
and training/resource budgets. A residual drop without native held-out benefit is
not a successful capability result.

## 4. EFE-KV-001: information selection versus protection rules

Random Attention protects prompt information and randomly evicts generated history;
its reported setting is long-reasoning cache eviction [P1]. EFE's indexed reference
is independently written, not a full implementation of its serving system.

Cross three selectors (random, recency, existing content selector) with prompt
protection on/off, plus an uncompressed baseline. Keep generated recent-buffer
protection, eviction schedule, physical budget, precision, and head-sharing layout
identical across selectors. Prompt-unprotected arms are diagnostic ablations only,
not recommendations for shipping. Never tune random seeds on task correctness.

Report three regimes separately: long input/short answer; short input/long generated
reasoning; and once-stated delayed evidence or state. Distinguish a fact in the
protected prompt from one introduced into generated history or a later tool message.
The supplied analysis describes a rare-fact failure in the paper; test such behavior
on Rineya rather than assuming its trace has sufficient redundancy.

With full prompt protection, even deleting every generated KV entry saves at most
`G / (P + G)` token-KV fraction. At `P=4096, G=256`, that is `1/17`, about 5.88%.
This is an analytical upper bound, not a measured memory reduction. Budgets include
prefill, retained generated entries, and the recent buffer. An over-budget protected
set makes an arm ineligible; never silently evict prompt tokens instead.

Qualify original logical positions/RoPE positions after physical compaction, masks,
cache-position counters, shared KV-head versus query-head indexing, batching,
deterministic replay, and no restoration of discarded entries. Repeated sampling
from resident history has a survival/recency bias; it is not uniform reservoir
sampling from the complete original trace. The source model's full-cache-correct
rows define paired damage reporting; also retain all-row outcomes.

The scoring hypothesis fails if the content selector does not improve the specified
quality/cost tradeoff over protected random at matched budgets. Deployment is not
justified when memory savings are negligible, delayed-evidence damage exceeds the
registered margin, or latency/workspace overhead erases the advantage. KV changes
cannot repair forgetting already encoded in model weights.

## 5. EFE-DEQ-001: useful finite refinement before equilibrium training

DEQ models seek `z* = F_phi(z*; h)` and use implicit differentiation [P3]. Constant
memory with respect to effective unrolled depth is not constant total memory or
zero solver cost. Keep this track conditional on an actual useful recurrent mapping.

First measure native quality against refinement depth `1, 2, 4, 8` with matched
training/evaluation contracts. Include source, one-pass sidecar, matched stateless
MLP, state-reset, and state-removal controls. More refinement must demonstrate a
causal task benefit, not merely extra computation or additional parameters.
Predefine maximum refinement budget and report quality and cost at every depth.

Only then test a compact sidecar: `z* = F_phi(z*; h)`; `h' = h + alpha P z*`.
Keep equilibrium dynamics separate from output injection. Zero output gate can
block the task gradient into the sidecar; evaluate a specified warmup, small
nonzero injection, or auxiliary objective. Do not assume identity initialization
is automatically a valid DEQ initialization.

For `F(z)=z+alpha R(z,h)`, `alpha=0` makes every state fixed and `I-J=0` singular.
For damping `F_eta(z)=(1-eta)z+eta F(z)`, the apparent residual is scaled by `eta`.
Therefore the reference solver stops on the **undamped** residual at the returned
state. A numerically converged identity map still fails the implicit-system check.
Small residuals alone do not establish uniqueness, useful reasoning, or a well-
conditioned backward problem.

Qualify implicit gradients against finite differences and adequately converged
explicit references on small cases. Then qualify matrix-free native adjoints,
conditioning, sensitivity to initialization, causal masking/state reset, and solver
failure handling. Jacobian regularization is a relevant stability option [P4],
not an assurance of useful refinement. Never accept unconverged outputs silently;
report failures and a preregistered fallback policy separately.

If quality improves briefly then degrades near equilibrium, finite recurrence may
remain useful but an equilibrium target is not supported for that mapping. DEQ
advancement requires a useful stable state plus a measured quality/memory/latency
tradeoff over strong finite controls. The current CPU solver is forward-only and
the dense adjoint is a toy qualification helper, not a trainable native sidecar.

## 6. Additive and cumulative use without conflating mechanisms

Do not activate all three interventions in the first native comparison. First
confirm composition; separately qualify cache behavior with weights fixed; keep
DEQ conditional. Once individual gates pass, run an allocation x cache factorial
with a fixed checkpoint lineage and full-cache controls. Cache damage must not be
misattributed to weight-space forgetting.

For cumulative skills, preregister at least two admissible skill pairs, both A->B
and B->A orderings, then a later third skill. Freeze earlier decisions before
selecting the next update. Measure retention on every previous skill, broad behavior,
format, abstention, and source-correct examples. Report capability gained per
incremental parameter, optimization, inference-compute, and memory cost.

Adding a refinement sidecar changes the model function and Jacobian; re-extract and
requalify output-direction measurements. Existing coefficients do not remain valid
by assumption. Likewise requalify cache behavior after refinement changes attention.
No claim of lossless, universal, or monotonic skill addition follows from one pair.

Deliverables at each stage: immutable config/data manifest; qualification receipts;
raw paired outputs; metric/scorer versions; acquisition/retention and resource tables;
failures/negative results; and a written promote/hold/reject decision tied to the
frozen criteria. Archive null results rather than redefining a gate after seeing them.

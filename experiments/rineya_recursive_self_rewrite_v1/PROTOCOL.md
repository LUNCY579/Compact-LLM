# RSR-001 experiment protocol

## Hypothesis and causal comparisons

External workflow support can discover useful student executions. Reconstructing
those executions may teach the student to initiate equivalent procedures under H0.
This is a hypothesis, not a native result. Separate discovery gains, distillation,
transfer, retention and cost. A runtime-assisted success is not internalization.

The frozen discovery comparison is H0 resampling versus a portfolio at equal total
attempt allocations and resource caps. Use the same checkpoint, enrolled tasks,
seed schedule, action space and public tools. Report actual resource consumption
and exclusive task coverage, including invalid, truncated, errored and exhausted
attempts. Do not substitute a union of six attempts for a one-attempt baseline.

For training, use a frozen common source pool. Compare untreated base, source-context
SFT, one-pass reconstruction and recursive reconstruction. Treat localized review
as a fifth, separate condition. Keep source task coverage and ancestry identical
where possible; report any source lost to rewrite failure. If coverage differs,
`match_controls` must reject the mechanism-matched claim. A separate full-pipeline
budget comparison can intentionally include filtering and expansion, with that
estimand labeled explicitly.

Match initial checkpoint, actual tokenizer, trainable parameters, seed, supervised
tokens, optimizer updates, learning rate and context. The reference trainer can
match exact supervised-token quotas even when the number of examples differs.
Count forward work and attempted-generation costs separately. More rewrites of one
parent are not more independent task discoveries. Cap retained trajectories per
parent before tokenization; report repeated presentations and row token exposures.

## Data and supervision boundaries

Enroll all known train, calibration, retention and reserved task identities before
collection. Each task requires source revision, immutable parent, role, prompt and
complete source-ancestry declarations. Descendants never cross their source split.
Do not give graders, record stores or expected answers to callbacks. The planner
receives a successful PUBLIC source execution. The critic sees only the public
instruction and proposed procedural runbook, not the source transcript or grader.
All candidate runbooks, revisions, critique outputs and request bindings persist in
private audit logs. Rejected candidates never authorize execution.

Every reconstruction starts an independently created RHD environment. Rerun tools
and retain their actual observations; never carry final artifacts or source hidden
state into the new attempt. Independent seeds are derived without using golds.
Export only complete successful, replay-verified, evidence-supported training
executions. Correct guesses lacking public support do not become SFT. Remove
runbook/controller guidance from reconstructed inputs; loss masking alone is
insufficient. Supervise only the current accepted action and one EOS. Prior accepted
actions and real observations are context. Corrected proposals and reviewer notes
remain private, and correction data is not automatically eligible for DPO/RL.

Native API users must keep task-owning evaluator code out of the worker's process
and restrict file/network access. DTO separation alone is not protection against a
hostile callback. Hashes detect corruption/drift, not a malicious maintainer rewriting
all files and their manifest. Provider accounts and prices are outside this module.

## Staged execution and stop conditions

**E0 — native qualification.** Start with 12 short development tasks. Qualify the
exact local checkpoint loader, RHD envelope, serialized prompt IDs, action parser,
actual tool execution, EOS/masks and resource caps. No training until the student
reliably executes constituent operations. Invalid/loading/OOM failures are not
negative scientific evidence about the RSR mechanism.

**E1 — frozen discovery screen.** Proposed native registration: 256 independent
training parents, two attempts per harness and six H0 controls. This is a proposed
size, not a job launched here. Freeze budgets after E0. A portfolio is useful only
if additional successes justify its actual compute versus resampling. Stop or
simplify tasks if neither strategy creates sufficient evidence-supported sources.

**E2 — reconstruction screen.** At most two candidates, two fresh executions per
accepted candidate, two revisions (three planner rounds total). One-pass has only
one round. Cap exports at four trajectories per parent. Fail closed on schema,
critique-binding, public-evidence or replay failures. Keep all failed attempts in
collection accounting. Self and teacher-assisted lanes have separate manifests.

**E3 — native training mechanism test.** Freeze common source eligibility before
training. Run one host-qualified seed for feasibility, then independent seeds if
resources justify them. Do not add RL, model recurrence, new memory, adapter merging,
or harness evolution to the first causal comparison. The provided trainer performs
a no-replay mechanism screen only; use protected-skill evaluation before retaining
any resulting model. Later independent replay is a separately registered treatment.

**E4 — reviewer-free evaluation.** Run all arms under H0 with the same tools, token
budgets, attempt seeds and stop rules. Report first-attempt success, pass@k,
supported success, tool/format errors, calibrated and retention panels separately,
and repairs versus damage. Pair outcomes by task/attempt/seed; bootstrap by original
parent, not by individual rewritten trajectory. Per-source and per-skill effects
must not be hidden by a positive pooled average.

**E5 — independent confirmation.** Reserve new parents, changed evidence and values,
and withheld skill combinations before adapting the recipe. The current runner
refuses `sealed` tasks. Native owners need a frozen, external confirmation path.
The generated toy catalog does not establish semantic holdout or real-world transfer.

## Decision criteria (proposed, not measured)

A useful initial engineering target is at least +5 percentage points over the
strongest matched trained control on held-out H0 task completion, with a paired
parent interval excluding zero after an adequately powered confirmation run. This
is not a prediction or guarantee. Register a retention noninferiority margin (for
example -2 points) based on application needs BEFORE looking at confirmation data.
Report protected-skill intervals individually; low power is a HOLD, not proof that
nothing was forgotten. Multiple training seeds and an independent panel are needed
before a cumulative-capability claim.

Runtime-only gain means harness value, not weight internalization. Familiar-variant
only gain means narrow transfer. Acquisition with old-skill damage is interference.
Intermediate progress without completion does not satisfy the primary endpoint.
Even a training/evaluation-positive result must pass the existing Rineya deployment,
reload-fidelity, resource and provenance gates before it is retained or shipped.

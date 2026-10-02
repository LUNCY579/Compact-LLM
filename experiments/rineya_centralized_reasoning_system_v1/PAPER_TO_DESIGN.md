# AI-Newton mechanisms translated into falsifiable CRS-001 components

Recorded 2026-10-01. This is an adaptation map, not a reproduction of AI-Newton
physics or evidence that Rineya has discovered concepts. The exact main paper,
four-page supplement and primary-code snapshot are bound in
[SOURCE_AUDIT.md](SOURCE_AUDIT.md) and [audits/source_receipt.json](audits/source_receipt.json).
All primary code locations below use commit
`c143e865be42e067faf64cd8117cbeffd0fccfb6`.

Current implementation boundary: [contracts/CPU_V1.json](contracts/CPU_V1.json)
registers a supplied controlled-grammar reader, transparent supplied controller,
typed primitives and engineering semantics. Native readers, Rineya procedure
discovery, learned scheduling, weight-learning regularities and confirmed
system/model gains need their own receipts. Newly written code is not treated
as qualified until independent frozen-artifact tests pass.

## Separate theory objects

| Object | Operational meaning | Evidence required | Unsupported inference |
| --- | --- | --- | --- |
| Task concept/procedure | Versioned typed input/output or operation graph with applicability, dependencies, provenance, counterexamples and failure behavior | Independent DEV solving checks and later fresh transfer after freezing | A supplied solver is an autonomous discovery; external-library growth is weight learning |
| Learning empirical regularity | Versioned predictor about a declared checkpoint, task family, intervention/dose and acquisition/retention outcome | Registered curves, uncertainty and prospective predictions/interventions | Correlation or a chosen curriculum reveals an internal neural mechanism |
| Internal mechanism hypothesis | Proposed explanation of neural behavior with a distinct intervention and alternative explanations | Separately registered causal controls and replication within scope | Gradient overlap, sparsity or reviewer agreement alone proves mechanism |

These objects can reference the same evidence without sharing status. A good
task procedure may yield eligible training traces; a learning regularity may
recommend testing that procedure's prerequisites. Neither relation silently
upgrades the other's proof.

## 1. Experiment base and explicit theory base

**Paper mechanism:** main pp. 1–2 separates simulator experiments/data generators
from an explicit symbol/concept/law library. Concepts and specific/general laws
retain operational meanings and applicability. Source implementation pointers:
`aiphy/interface.py:191,212`, `aiphy/manager.py:223`, and
`aiphy/general_model.py:15,100`.

**Rineya adaptation:** versioned tasks, parent/split manifests, exact checkpoints,
interventions and immutable receipts form the experiment base. Typed task
procedures and learning regularities form distinct theory collections.

**Implementation contract:** CPU_V1 supplies byte-addressed source, explicit
epistemic states and six primitive operations. Add versioned theory records with
author/proposer/reviser/selector/verifier identities; current code and independent
tests bind a receipt before CPU qualification. A law-library analogue alone does
not establish that model-generated candidates exist.

**Hypothesis H1:** reusable validated task knowledge adds semantic accuracy or
reduces measured cost beyond otherwise identical primitives and state.

**Controls:** B versus C using the same checkpoint, raw source, reader/controller,
verifier, retriever, primitives and ceilings; D diagnostic matched in format/size.

**Limits:** scaffold utility and Rineya discovery are different claims. Freezing
explicit objects improves auditability but does not make their meanings true.

## 2. Typed physical DSL, specialization and generalization

**Paper mechanism:** SM A pp. 1–3 gives an AST-based physical DSL with object
types, evaluation rules, quantified concepts, complete-object sums and
specialization. `src/semantic/generalize.rs:46,91` normalizes object indexing and
checks summation coverage; `src/semantic/specialize.rs:15,71` instantiates objects.

**Rineya adaptation:** restricted typed operation graphs and applicability checks
replace physical equations. Evidence spans, entity types and source states are
explicit. Template/program abstraction generalizes a task procedure; it does not
prove a learned regularity about weights.

**Implementation contract:** validate operation names, types, references, graph
acyclicity, source binding, scope/coverage and total cost before deterministic
execution. Task-supplied rules govern negation, cancellation and ties. No arbitrary
model-generated Python, shell or network request executes.

**Hypothesis H2:** validated operations reduce invalid/unsupported conclusions
under the declared budget while preserving answerable accuracy.

**Controls:** independently solved wrong-type/reference/coverage fixtures, negative
source instructions, explicit contradictions and syntax-versus-semantic scoring;
later compare schema effects separately from task correctness.

**Limits:** formal syntax cannot certify natural-language interpretation. An AST
and the same parser in two components do not establish independent verification.

## 3. Plausible reasoning and counterexample-driven revision

**Paper mechanism:** main p. 3 proposes candidate modifications when an existing
law fails, adding simple terms within a constrained search. Candidate knowledge
precedes verification. Relevant completion interfaces are
`aiphy/general_model.py:340,441,665`.

**Rineya adaptation:** failures on TRAIN/DEV prompt a hypothesis or narrowed
procedure; independent checks and adversarial counterexamples decide revision,
rejection or quarantine. Codex revisions, Rineya inference revisions and optimizer
updates have separate origin/cost records.

**Implementation contract:** preserve immutable candidate versions, failed
counterexamples and applicability changes. Accept only restricted DSL/graph
candidates. Store task-procedure and learning-hypothesis revisions in distinct
ledgers. Initial supplied fixtures test the lifecycle without claiming discovery.

**Hypothesis H3:** counterexample revision improves fresh semantic transfer over
an unrevised eligible candidate at comparable total search resources.

**Controls:** a later registered static-library or resampling/best-of-N control,
same candidate pool/allowed primitives, independent truth checking and all failed
search costs. Begin with H1; do not run every ablation immediately.

**Limits:** self-agreement and external judge confidence are not truth. Test-bank
feedback cannot revise a supposedly frozen procedure or supply training traces.

## 4. Progressive generalization and scope-qualified validation

**Paper mechanism:** main pp. 2–4 distinguishes specific laws from broader laws;
SM A supplies generalized/specialized forms. `general_model.py:823,838` tests a
proposed law across other experiments and records its valid scope.

**Rineya adaptation:** promote a narrow task procedure to broader compositions
only after independent counterexamples and fresh transfer. Learning regularities
progress from observations to prospective prediction and controlled interventions,
with status and assumptions preserved.

**Implementation contract:** freeze selected library/checkpoint/controller/scorer
and expose no cross-instance answer/cache memory. Partition scenario ancestry and
generator templates before rendering. Register which families/compositions are
unfamiliar, and distinguish IID from transfer outcomes.

**Hypothesis H4:** a frozen generalization predicts success on held-out operation
compositions better than a narrow procedure or simple baseline.

**Controls:** new structural parents/compositions, renamed/counterfactual twins
kept within split, shortcut attacks and family-aware uncertainty. Any aggregate
feedback used for later selection consumes the confirmation bank's freshness.

**Limits:** repeatedly rechecking the same simulator family is not CRS fresh
confirmation. Passing finite tests supports recorded scope, not universal truth.

## 5. Simplification and redundant representations

**Paper mechanism:** main p. 3 and SM C p. 4 combine symbolic search with
Rosenfeld–Gröbner differential-algebra simplification. Primary pointers:
`aiphy/general_model.py:1067,1180`, `aiphy/diffalgservice.py:112,132` and
`aiphy/specific_model.py:232`.

**Rineya adaptation:** canonicalize duplicate typed graphs and prefer concise
procedures when tested semantics/applicability remain preserved. Remove redundant
active candidates while retaining their version history and dependencies.

**Implementation contract:** bounded canonicalization is engineering-supplied.
Any learned rewrite needs a separate correctness contract, counterexamples and
fresh transfer. Unknown equivalence, unavailable checker or timeout cannot count
as proven equivalence. Preserve rollback references and invalidation edges.

**Hypothesis H5:** validated simplification reduces library/search/execution cost
without violating a predeclared semantic regression margin.

**Controls:** original versus simplified frozen library on the same tasks;
independent solver, identical scope/ceilings, library-token and cold-start costs.

**Limits:** physical algebraic equivalence does not prove equivalence of language
interpretations or branching procedures. Maple is unnecessary for the CPU slice.
Simplifying working knowledge never authorizes deleting research evidence.

## 6. Recommendation engine and cost-aware exploration

**Paper mechanism:** SM B pp. 3–4 combines discounted rewards/inverse-count
exploration and a learned concept-group weight. `aiphy/ucb.py:22,29` supplies the
experiment value; `aiphy/memory.py:59,175,223,358` manages selection/history.
`recommendation_engine/concept_group.py:5` and its update module expose the small
policy network.

**Rineya adaptation:** choose eligible DEV experiments using predicted information
value, acquisition, retention risk, uncertainty and measured cost. This is a
research scheduler, distinct from the deployed question-answer controller.

**Implementation contract:** start with transparent fixed/randomized scheduling;
log candidate eligibility, selection probability/rationale and all incurred
search/failure/review costs. Store predictions before the next experiment. Add a
learned policy only after sufficient outcome data and a separate comparison.

**Hypothesis H6:** within a declared budget, adaptive scheduling finds a better
acquisition/retention frontier or more informative observations than fixed and
random schedules.

**Controls:** matched checkpoint/search space/envelope, randomized exploration,
independent training draws and fresh evaluation families; separate scheduler
utility from final-recipe efficacy and compare prediction error to simple models.

**Limits:** the inspected repository samples groups differently from one prose
description; CRS does not claim exact reproduction. Policy reward/loss is not
capability gain, calibrated epistemic confidence or an optimal learning pattern.

## 7. Era control under fixed campaign authority

**Paper mechanism:** main p. 3 and SM B begin with tractable trials, filter feasible
experiments, and relax trial time after stagnation. Primary interface:
`aiphy/main.py:217,370,375`.

**Rineya adaptation:** development epochs progress from cheap mechanics to harder
shapes/interventions under predeclared stagnation and admission rules. A larger
proposed trial remains inside owner ceilings and needs shape-specific qualification.

**Implementation contract:** durable DAG/job states, idempotent receipts, one
fenced worker lease, current host/process reconciliation, cumulative rolling
budgets and explicit OUTCOME_UNKNOWN recovery. A stale lease allows reconciliation,
not duplicate launch. Evidence and failure costs survive restart.

**Hypothesis H7:** staged complexity uses the allowed cost more effectively than
a fixed complexity schedule, without degrading retention or completeness.

**Controls:** separately registered fixed-budget schedule at the same total
envelope; charge filtering/qualification/failures; retain unavailable and skipped
conditions. Measure policy utility on DEV before considering confirmation.

**Limits:** exponential time relaxation is a heuristic. The paper's runtime and
host examples neither qualify this laptop nor override two hours/epoch, four
hours/rolling day, or existing numerical/resource gates.

## 8. Learning atlas and trained-model tests are new research

**Paper mechanism boundary:** AI-Newton's reported object is explicit physical
knowledge discovery from simulator data. Its recommendation policy training is
not Rineya weight acquisition, language retention or verified-trace distillation.

**Rineya adaptation:** measure checkpoint-by-skill acquisition/transfer/forgetting
curves and a scoped efficiency frontier. Form empirical predictors of which
prerequisites, update scopes or replay mixtures improve later learning.

**Implementation contract:** separate registered microtraining from procedure
distillation; exact starting/served/exported identities, frozen tensors/update
scope, optimizer/RNG/sampler resume, unique parents/repetition, processed versus
supervised tokens and measured costs. Evaluate both model-only with procedures
disabled and the full system. Freeze noninferiority margins and prospective
checkpoint/skill/stratum analyses.

**Hypothesis H8:** one declared intervention improves acquisition or reduces
forgetting at matched dose within the checkpoint/task/resource scope, and the
atlas predicts a future intervention better than simple baseline-competence or
constant predictors.

**Controls:** ordinary adaptation from the same checkpoint and training draw;
independent draw/order/initialization identities; source/incumbent and previously
acquired skills; fresh transfer, semantic abstention/format separation, exported
serving and delayed retention. Trace distillation additionally needs independently
validated eligible development traces and answer-only supervision control.

**Limits:** this hypothesis is not established by the paper, CPU tests, training
loss, parameter diagnostics or a single selected recipe. Nonsignificance does not
establish retention; unresolved language/long-context coverage stays unmeasured.

## Priority and claim policy

Implement and independently qualify the CPU vertical slice first. Qualify the
exact short native geometry before the 16-case open reader diagnostic. Then
resolve H1 with a controlled open DEV B/C contrast. Use its failure decomposition
to choose the next bounded intervention. A separately qualified learning-atlas
microstudy can proceed before H1 confirmation, with ordinary adaptation labeling.
Fresh confirmation, broader transfer and promotion require frozen instruments,
independent reviews, complete costs and their own evidence.

Keep three scoreboards: model-only, tool-assisted system and trained/distilled
model. Within the system scoreboard distinguish supplied scaffold, Rineya-proposed
discovery and Codex/human-assisted discovery. Clean negative or inconclusive
results narrow a mechanism or budget; preserve the track and its original evidence.

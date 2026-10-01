# CRS-001 pilot proposal: reusable procedures for long-context state tracking

**Draft protocol; not an execution permit or finalized preregistration.** All
native-run and protected-bank gates in [manifest.json](manifest.json) are closed.
Proposed sizes and thresholds below must be reviewed and frozen before opening
new evaluation labels. None of these datasets or results exists in this package.

## Question and falsifiable claims

At fixed model identity and declared resource limits, does reusable, development-
validated procedure memory improve multi-hop state tracking beyond the same
retriever, primitive tools, verifier, and controller without persisted procedures?

A gain from extra calls, a different model, privileged evidence, a hidden solver,
or test feedback does not establish the specific procedure-memory hypothesis.
An observed benefit from a supplied procedure establishes scaffold value, not
autonomous procedure discovery. Track system utility and the mechanism claim
separately.

## Stage 0: implement and qualify CPU mechanics

Future bounded implementation should define evidence and operation schemas,
immutable source references, working-state reset, procedure snapshots, split
isolation, resource accounting, and an independent reference scorer. Test valid
and invalid events, negation, unknown states, cancellation, alias ambiguity,
partial coverage, duplicate records, source updates, malformed operations,
cycles, malicious document instructions, and budget exhaustion.

Verify that evaluation code cannot write procedure memory and that runtime code
cannot access gold answers or oracle state. Test interruption/resume without
mixing instances or erasing failed receipts. These are future CPU controls, not
model-quality evidence. This founding change does not claim they have passed.

## Stage 1: bounded native reader diagnostic

Propose 16 development-only cases with decisive evidence supplied in a short
context. Check extraction of entities, time, modality, negation, confirmation,
and cancellation, along with well-formed operation requests.

The supplied-evidence condition is diagnostic only: it is not a deployable
benchmark result, not an intrinsic capability ceiling, and not a claim that
retrieval is solved. Separate source interpretation from deterministic execution.
If the reader fails, inspect that stage before lengthening the corpus.

A run needs an exact checkpoint/config/tokenizer binding, qualified inference
path, numerical contract, measured host/GPU admission profile, and frozen call,
token, time, and retry limits. No universal safe caps are inferred from an older
experiment. The intended first candidate is the central Titan devfreeze, subject
to qualification. A separately selected Supa run must be labeled as Supa; it
cannot substitute for Titan evidence or silently bypass a blocked Titan gate.

## Stage 2: controlled procedure-memory screen

Proposed scale: 64 development parent problems and 128 fresh evaluation parent
problems, each with paired approximately 4K and 16K source-token versions. Freeze
exact tokenizer-based lengths, rendering, question overhead, and overflow policy
before evaluation. These are source lengths, not promised native model windows.
The active call window stays inside its separately qualified envelope.

Use state tracking with transfers, prerequisites, revocations/cancellations,
aliases, exceptions, and distributed evidence. Include answerable cases plus
separately scored insufficient/contradictory cases. Vary distractor placement and
break lexical shortcuts. Keep all renderings, counterfactuals, and entity-renamed
variants of a parent in the same split. Hold out selected templates or operation
compositions, not merely entity names, when testing procedural transfer.

### Arms

| Arm | Description | Claim it can support |
| --- | --- | --- |
| A | Qualified model-only full input where supported; unsupported lengths explicitly marked unavailable. | Native baseline, never silently truncated full context. |
| B | Same checkpoint with retrieval, working state, primitive operations, verifier, and controller; no persisted task-specific procedure library. | Strong system baseline. |
| C | B plus the frozen development-validated reusable procedure library. | Primary incremental procedure-memory comparison: C versus B. |
| D | B plus irrelevant but format/size-controlled procedure content, used as a diagnostic. | Whether correct procedural information matters, not just extra instructions. |

For a practical system result, compare C to A only on jointly eligible inputs
and disclose all extra resources. Never substitute A for the primary B-versus-C
mechanism control. Supplied-evidence diagnostics remain separately labeled.

Keep base prompts, primitive capabilities, checkpoint, decoding, retriever,
verifier, eligible cases, and resource ceilings identical for B and C apart
from declared procedure access. C may consume fewer calls if its procedures help;
do not pad dummy calls to force equal realized cost. Report accuracy-cost curves
and paired outcomes under common ceilings. Include the library tokens in costs.
B can compose the same primitives during a question; what it lacks is persisted
cross-question procedure knowledge, not basic tool functionality.

Record whether C's procedures were human-authored, model-proposed, revised,
or selected. A separate comparison is required before calling a hand-built
procedure advantage a discovery result. Do not give C gold event records while
requiring B to extract them from natural language.

### Leakage barriers

Procedure proposal, selection, revision, and counterexample search use only
TRAIN/DEV material. Freeze and hash the procedure library, model bindings,
controller, retriever, scorer, prompts, dataset generator, splits, and budgets
before evaluation. Reset instance memory between parent cases. Share no answers,
intermediate conclusions, extraction caches, or evaluator feedback across test
parents. A shared frozen tokenizer or model is not task memory.

A test corpus may be indexed using its permitted source text, but that indexing
is instance-scoped and its complete cost is recorded. Hidden supporting facts
and reference answers are excluded from deployable arms. Keep the scorer out
of the runtime and use an independently checked reference implementation rather
than evaluating a solver against itself. Existing central CHECK/FINAL banks
remain closed; this proposal creates no right to access them.

Online learning across evaluation episodes is a different future experiment;
do not silently enable it here.

## Measurements and decision rules

Primary proposed endpoint: for each answerable parent, average exact-answer
correctness over its two length versions, then compare the paired parent-level
mean for C versus B. Also report 4K and 16K results separately. Use parents as
clusters for confidence intervals; length variants are not independent samples.
Keep counterfactual-pair success as a distinct diagnostic.

Report repairs and damages, answerable accuracy, erroneous abstention,
unsupported-answer rate, correct insufficiency handling, extraction accuracy,
evidence coverage, procedure applicability, and operation validity. A verifier
PASS is not automatically a correct answer. An improved aggregate score cannot
hide damage to answerable tasks or an abstention-only effect.

Record calls, actual input/output tokens for every pass, executor operations,
indexing/scanning time, retries, total wall time, peak process-tree RAM, peak GPU
allocated/reserved memory, external state size, and stop reasons. Count procedure
discovery/validation separately and show cold-start and amortized costs; declare
the amortization denominator. Include all scheduled cases and failures. A
whole-run hardware abort is an incomplete run, not a silently reduced sample.

Proposed go-to-confirmation screen: at least a five-percentage-point C-minus-B
primary gain and a paired parent-cluster 95% confidence interval with lower
bound above zero, with all preregistered correctness, preservation, and resource
guardrails satisfied. This is a proposed screening rule, not a power guarantee
or a final broad-capability claim. Before test access, freeze the interval
method, sampling seed, insufficiency-case mix, preservation panel, acceptable
regression margins, and resource ceilings. A small inconclusive pilot should
trigger a justified independent confirmation design, not test-set tuning.

Negative or inconclusive results remain in the ledger. Interpret no B/C
separation as lack of support for this implementation and budget, not automatic
invalidation of all external memory or all Rineya research. Diagnostic D must
not become a post-hoc replacement primary endpoint.

## Stages 3 and 4: only after the screen

First replicate any gain on independent parent problems and confirm broader
preservation. Then consider public benchmarks: RULER for controlled long-context
operations, BABILong for distributed-fact reasoning, OOLONG for aggregation,
and LongBench v2 for realistic tasks. Primary references are in [SOURCES.md](SOURCES.md).

Pin exact dataset revisions, licenses, prompts, scoring, splits, and eligible
lengths before downloading or evaluating. Label modifications as adaptations;
do not call a custom subset an official leaderboard score. No benchmark or
paper score is a Rineya target forecast.

Only afterward propose distillation into a checkpoint with separate training
permission and the central retention protocol. Evaluate model-only and
restricted-tool settings, new tasks, and unchanged protected behaviors. A
system result alone cannot authorize training or support a native intelligence
claim.

## Pre-execution registration checklist

The future run manifest must bind all identities and artifacts above, completed
CPU controls, exact native qualification scope, independent scorer, proposed
sample/arm order, no-retry or explicit retry policy, total resource caps, and
an owner-approved launch scope. Placeholder budgets, documentation checks, or
inherited central gates are not substitutes. Preserve the founding proposal
and append amendments rather than rewriting its historical status.

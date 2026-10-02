# CRS-001 initial scientific and architecture adversarial review

Review date: 2026-10-01, America/Chicago. Reviewer: delegated scientific-audit
agent `/root/scientific_audit`. Reviewed checkout:
`research/rineya-centralized-reasoning-system-v1` at
`7459b614bcb017ae4f0160b15e0ccd53473bb990`.

Disposition: the founding design supports a scientifically viable **CPU
implementation milestone**, conditional on the contracts below. It is not a
finalized native preregistration, native qualification, efficacy result, or
promotion approval. The findings concern unresolved implementation and
measurement contracts; they are not assertions that an unbuilt system has
already failed. Preserve the founding proposal and append scoped amendments.

## Scope, authority, and reviewer access

Read the complete user request at
`C:/Users/clclc/.codex/attachments/d05aa858-2f19-4093-b86a-c674d09381db/pasted-text-1.txt`
before reviewing. Its local inference and disposable-training authorization is
conditional on scientific and operational prerequisites; its scope does not
open existing central CHECK/FINAL. The historical founding state remains
`DESIGN_RECORDED_EXECUTION_NOT_AUTHORIZED`.

Read root `AGENTS.md`, the charter, all seven founding package records, full
`MODEL_QUALITY_STATUS.md`, central research/evaluation/promotion/resource/train-
scope protocols, Amendments 017–021, next-experiment/README, and retention design
and requirements. Root and central records retain their independent identities.
No ancestor AGENTS file was found at the checked drive/Users/user/BusinessProject
paths. No nested AGENTS file exists in the founding CRS package inventory.

This reviewer shares the coordinator's model lineage, workspace, and host
privileges. The review is a separate reasoning pass over original artifacts,
not an independent ground-truth measurement or an OS-isolated referee. The
coordinator may inspect this report and the same source evidence. No protected
CHECK/FINAL gold, ANY_READ-barred material, or checkpoint tensor was opened. No
model was launched. This reviewer did not audit the full paper or its code;
paper fidelity must rely on the separately pinned source audit, not this report.
Only this audit file is owned or written by this reviewer.

Reviewed core-file SHA256 bindings:

| File | SHA256 |
| --- | --- |
| `AGENTS.md` | `2688c8447722b791d87bcf372d72f487d56b00170b13abd8888267de5c15f4d2` |
| `RINEYA_CENTRALIZED_REASONING_SYSTEM.md` | `614c3b77a73cf1c848d3496c8f9414c1d38f35feb3a3a2f5852de3eb6ed9ffa7` |
| `DESIGN.md` | `3aaf50e6abebd241c8df268ad00fb317ac484532bcc48d8607552eb3c0d3a7f0` |
| `EXPERIMENT_PLAN.md` | `3dc6627ae05c0e41aab0bb00bf9961b5b7a230ddbb06568aefe8aef0d7c6987b` |
| `SOURCES.md` | `7be7f3bb31011b7e6b6ac6a2c4552b10dfa597d986fc331a560d6f8955a4af56` |
| `INTEGRATION_MAP.md` | `fe80d0b925387fc4f9ed68825f9032eb442b97e91ffaed4c55bd98292f5262c2` |
| `MODEL_QUALITY_STATUS.md` | `0c6e03832b2ed882d2dd3aa20c93d5d13e888a78ad4dae87adec562a209be7d9` |

Package filenames in this table resolve within
`experiments/rineya_centralized_reasoning_system_v1/`; repository-root filenames
retain their literal root locations. Hashes identify the reviewed bytes, not
their truth or historical execution. Untracked `audits/` work from other agents
was visible; no reset or unrelated edit was performed.

## Findings and closure criteria

### SCI-01 — B/C parity and supplied-solvers need executable witnesses

**Priority:** blocks a native procedure-memory attribution claim; does not block
CPU infrastructure. **Evidence:** `DESIGN.md:69–89,112–120` defines TRACE_STATE
and distinguishes hand-written solvers; `EXPERIMENT_PLAN.md:69–90` requires
identical primitives, source evidence, and a C-only procedure treatment.

**Impact:** a controller, reader, or compound primitive may silently contain the
entire target algorithm. A scripted CPU C-minus-B result can then reflect
engineering policy rather than checkpoint behavior or Rineya discovery. Giving
C an already parsed event graph while B reads raw text invalidates the contrast.

**Counterexample:** make TRACE_STATE accept a whole source and return its owner;
use it only from C while B must select/join/filter events. C wins although its
library contributes no reusable model decision. Conversely, make both arms call
that solver unconditionally and a null C-minus-B result cannot show that reusable
procedures are generally useless.

**Verification:** record and compare raw-source hashes, checkpoint/settings,
primitive definitions/whitelist, retriever, verifier, controller, common prompt
and budget bindings across B/C. Their sole declared difference is procedure
access. Emit proposer/reviser/selector/verifier identities for every procedure
and reader/controller origin in every receipt. Include an invocation witness
showing B can compose every same primitive within an episode. Mark scripted
readers and complete supplied solvers as infrastructure/scaffold conditions.

### SCI-02 — Gold separation must include generation, verification, and logs

**Priority:** blocks efficacy evidence if breached. **Evidence:**
`DESIGN.md:88–89`; `EXPERIMENT_PLAN.md:29–30,96–107`; user request §5 distinguishes
logical role separation from enforced privilege isolation.

**Impact:** independent scoring fails if deployment receives hidden event graphs,
gold answers, scorer-derived coverage, or intermediate oracle state. A verifier
that shares the same interpretation bug can incorrectly certify a response.

**Counterexample:** runtime and scorer both parse “Morgan did not transfer A to
Riley” using one substring rule. Both move A and CHECK returns PASS; no independent
correctness evidence exists. A “public” source object retaining `gold_owner` or
an answer in filename/log metadata leaks despite a separate scorer function.

**Verification:** construct runtime payloads from an explicit public-field
allowlist. Reference solver uses a separately implemented semantics/algorithm
and never calls runtime TRACE_STATE or extraction. Audit imports and payloads,
including logs/caches/retrieval files; inject forbidden fields and assert
rejection. Use adversarial source fixtures where shared parsing mistakes produce
wrong runtime answers that the scorer detects. Report the actual boundary:
restricted DSL and separate process/interface can constrain a deployed worker,
but same-account Codex peers can access workspace gold. Do not claim OS secrecy
unless separate identities/ACLs or another measured enforcement mechanism exists.

### SCI-03 — Parent separation alone does not establish procedural transfer

**Priority:** blocks fresh-transfer or broad-generalization claims. **Evidence:**
`EXPERIMENT_PLAN.md:54–65`; `MODEL_QUALITY_STATUS.md`, “What exact training corpus”
and QTB-001 partition/audit requirements; central
`evidence/RETENTION_DESIGN.md:17,48–52` documents ancestry and exposure exclusions.

**Impact:** renamed entities/new seeds can hide shared generator templates and
canonical scenario graphs. Exact prompt deduplication misses these shortcuts.
Parent bootstrap handles paired renderings but does not manufacture independent
generator families or unseen compositions.

**Counterexample:** TRAIN “Alex transferred parcel X after authorization” and
CONFIRM “Bea transferred item Q after authorization” have different hashes and
parent IDs but identical event topology/template; a lexical solver transfers
without acquiring the claimed procedure composition.

**Verification:** assign underlying scenario graphs, base generator families,
rendering-template ancestry, paired answerability/counterfactual variants and
composition strata before solving/rendering/model outputs. Preserve all variants
within one split. Freeze specific held-out compositions/families and separate
IID from transfer endpoints. Run normalized/digit-blind/semantic ancestry checks,
question/metadata/position/last-event shortcut baselines and negative controls
on a deliberately leaky generator. State whether intervals are conditional on
the frozen family mix or generalize to independently sampled families. Never
read barred contents to do this; use authorized fingerprint inventories.

### SCI-04 — The 128-parent screen is not a precision guarantee

**Priority:** blocks confirmation readiness until frozen. **Evidence:**
`EXPERIMENT_PLAN.md:116–141` leaves interval method, insufficiency mix, preservation
margins, and other design quantities unset; user request §8 requires prospective
precision/power and sequential/multiplicity rules.

**Impact:** a 5-point screen threshold can be highly variable. Inspecting new
recipes on the same bank consumes its freshness, even if only aggregate results
are exposed. Training-order repetitions do not create new evaluation parents.

**Counterexample:** among 128 binary parents, 25 repairs and 18 damages give
7/128 = 5.47 points. The illustrative independent-parent normal standard error
is approximately 5.10 points, yielding a very wide interval that crosses zero.
This calculation is an illustration, not the registered interval method or a
claim that every 128-parent design lacks power.

**Verification:** freeze parent-level paired endpoint, variants, eligibility,
family weights, interval/resampling method, seed, sample size and precision
design before confirmation. All planned B/C cases remain in the case ledger;
failed/aborted cases cannot disappear or be scored as correct abstention. Treat
a whole-run infrastructure abort as incomplete and report missingness. Maintain
sealed → registered evaluation → consumed transitions; aggregate feedback used
for decisions consumes untouched status. Use fresh confirmation after adaptation.
D and length-specific results remain diagnostic unless prospectively registered
as distinct endpoints with a multiplicity rule.

### SCI-05 — D, abstention, and formatting need separate interpretations

**Priority:** blocks reasoning or preservation claims from aggregate scores.
**Evidence:** `EXPERIMENT_PLAN.md:74,121–124,143–145`; `MODEL_QUALITY_STATUS.md`
documents literal UNKNOWN versus semantic abstention and exposed format effects.

**Impact:** irrelevant procedures can distract D; C>D alone need not identify
procedure value. A library may improve output syntax or decline more questions
while answerable reasoning gets worse. Contradiction and incomplete evidence
need distinct source-level labels and outcome measurements.

**Counterexample:** B answers 60/80 answerable and 0/20 insufficient cases; C
answers 50/80 answerable and abstains correctly on all 20. Aggregate accuracy
rises 60%→70%, while answerable accuracy falls 75%→62.5%.

**Verification:** compare C versus B as primary, with D matched in schema,
serialized size/tokens and access overhead; predeclare D nonapplicability without
introducing false task-specific rules. Report semantic answerable accuracy,
unsupported answers, incorrect abstention, correct insufficiency/contradiction,
strict-format validity, malformed output, termination, evidence coverage, and
extraction separately. Do not infer calibration without an explicit confidence
output and scoring contract. Freeze represented-family regression guardrails.

### SCI-06 — Corpus length, worker context, and native qualification differ

**Priority:** blocks native admission outside an exact qualified envelope.
**Evidence:** `EXPERIMENT_PLAN.md:42–59,71`; central `RESOURCE_POLICY.md:23–35`;
central `README.md`, `NEXT_EXPERIMENT.md`, Amendments 020/021.

**Impact:** historical CE512 mechanics do not qualify 2K/4K reader calls, 4K/16K
attention, sustained training, or a new loader. Titan prefix-versus-bulk likelihood
disagreement remains unresolved, despite implemented backend and short mechanics.
Successful static identity cannot substitute for generation qualification.

**Counterexample:** chunk a 16K corpus into 512-token reads, then label C's score
“16K native attention.” Or load the exact same weights through a new adapter and
reuse the historical loader's zero-adapter parity/resource receipt.

**Verification:** register source-corpus token lengths and actual full model-call
lengths separately, including system prompt, library, working state, output and
overhead. A marks unsupported shapes UNAVAILABLE without truncation. Pin and
qualify exact weight/config/tokenizer/custom code/runtime/dtype/generation path,
reference math, cache/bulk/prefix behavior where used, maximum call shape and
measured process-tree RAM/VRAM. Preserve numerical blockers and tolerances;
resolve them with fresh receipts or register a scoped distinct inference contract
with independent review. A separately qualified Supa subject stays Supa evidence.

### SCI-07 — Learning-atlas gains require matched dose and valid retention

**Priority:** blocks trained-model promotion or learning-efficiency attribution.
**Evidence:** user request §§9–11; `MODEL_QUALITY_STATUS.md`, ordinary baseline
ledger/technical readiness; central `RESEARCH_PROTOCOL.md:69–90`; central
`evidence/RETENTION_DESIGN.md:25–31`.

**Impact:** a winning score with changed checkpoint, data, optimizer, targets,
output format and exposure simultaneously does not identify an intervention.
Nonsignificant regression and tiny source-correct denominators do not prove
noninferiority. More targets or repeats may alter processed compute and
memorization rather than acquisition efficiency.

**Counterexample:** trace supervision improves tool-assisted responses, but all
scores still use the external solver. Claiming internalized weight improvement
is unsupported. Alternatively, train one seed/order and evaluate it repeatedly
on the same skill examples; plotting narrow row×seed intervals overstates evidence.

**Verification:** preregister microtraining separately from validated-procedure
distillation if eligible traces do not yet exist. Pair ordinary control and one
treatment from the same checkpoint and training draw; distinguish data-draw,
initialization, order and evaluation-family seeds. Fix data/exposure/optimizer
choices except the declared intervention; publish any deliberate dose difference.
Measure starting competence, unique parents, repetitions, processed and supervised
tokens, updates, measured compute, failures, and checkpoint curves. Evaluate
trained weights with external procedures disabled plus full-system measurements.
Freeze retention margins, source-correct subset under exact current serving,
skill/stratum panels and uncertainty method before candidates. Include source and
approved incumbent comparators, previous skills and delayed checkpoints. Undefined
or underpowered preservation remains unestablished. Language/long-context gaps
cannot be filled by short deterministic tasks.

### SCI-08 — Learning regularities must predict fresh interventions

**Priority:** blocks supported-mechanism or optimal-pattern claims.
**Evidence:** user request §§10,12; `INTEGRATION_MAP.md` distinguishes research
objects; charter requires preservation even after negative results.

**Impact:** fitting explanations to selected winners yields descriptive
correlations. Shared reviewers' agreement does not establish causality or a
universal optimum; adaptive search costs and failed recipes affect efficiency.

**Counterexample:** rank prerequisite curricula using DEV scores, then call the
winner's same-DEV score a successful prediction. A gradient-overlap correlation
is presented as proof that replay prevents forgetting without a replay control.

**Verification:** ledger each regularity with scope, alternatives, uncertainty,
supporting observations, counterexamples, status and version. Freeze a prediction
before the next run; compare prediction error/calibration with constant and
simple cost/baseline-competence predictors. Test one controlled intervention with
matched dose/randomized order and retain negative outcomes. Bound any frontier
claim to the searched checkpoints/task distribution/recipes/resource envelope.
Separate observation, predictive regularity, replicated intervention effect,
proposed mechanism and supported mechanism within its tested scope.

## Proposed first bounded CPU contract

This contract is a recommendation to the coordinator, not a self-issued grant.

1. Implement one transparent sequential vertical slice under this package:
   immutable UTF-8 byte-addressed source → explicitly labeled fixture/native
   reader → fresh bounded instance state → typed operation graph → deterministic
   verifier → answer/abstention → write-once complete receipt. Source spans bind
   document hash and offset unit. Corrected extractions invalidate dependent
   conclusions. No arbitrary generated code executes.
2. Use newly authored open mechanics worlds covering confirmed/proposed/negated
   transfers, prerequisite failures, aliases, cancellations, equal-time ambiguity,
   explicit contradiction, duplicate evidence, partial coverage and source
   revisions. Rules for cancellation and equal times belong in the supplied
   task contract; domain assumptions cannot be silently invented. Gold/state
   remains scorer-only. Every result labels reader and procedure origins.
3. The scorer independently solves canonical worlds or replays source semantics;
   it shares neither runtime state-transition code nor extraction rules. Test
   successful cases and deliberately incorrect runtime outputs, misleading
   proposals/negation, source-instruction injection, invalid references, typed
   operation errors/cycles, exhausted budgets and missing scorers. Unknown,
   refuted and contradictory states cannot collapse to unsupported answers.
4. Exercise B/C/D interface parity and frozen-library read-only access with
   controlled fixture decisions. Validate reset across parent/arm/length cases,
   public-payload allowlists, deny gold retrieval, stable hashes, and rejected
   evaluation-memory writes. These fixtures qualify plumbing, not Rineya efficacy.
5. Test durable receipts on interrupted writes/corrupt artifacts and preserve
   attempted/returned/failed/pending cases and costs. Freeze the planned case
   ledger before execution and reconcile missing cases on restart. No retry can
   erase the first failure or create another untouched evaluation bank.
6. Independent review receives frozen source/hash bindings, public and scoring
   contracts, original test failures and raw receipts. Close consequential
   findings with tests and re-review the changed contract, within the three-round
   milestone limit. Record real scorer/model privilege boundaries.

Completion permits **CPU_QUALIFIED** for exactly these mechanics. Native efficacy,
procedure discovery, training gain and retention remain UNMEASURED.

## Proposed native and learning-atlas continuation contract

Before any native launch, bind a dated authorization amendment to the user's
actual request; exact asset hashes/path and lineage; clean committed source;
qualified loader/tokenizer/generation path and maximum shapes; current host
admission, lease/fencing and cumulative budgets; independent scorer; public
inputs; complete case plan; and explicit stop/retry rules. A reviewer checks
evidence rather than granting authority. Preserve historical central launch
flags and protected banks.

First run a bounded mechanics qualifier on the exact subject and proposed short
reader envelope, then the separately registered **16 open DEV reader cases**.
Freeze the first proposed geometry at **512 total input tokens and at most 64
generated tokens**, with batch one and sequential calls. The 512 input includes
system/chat-template overhead, source, working state, and procedure content.
Reject overflow; do not truncate decisive evidence. The realized generation
prefix may reach 576 tokens, so qualification must cover that actual maximum,
its cache/position behavior, and matching resource costs. Historical CE512 alone
does not establish this 576-token decoding envelope. Preserve Titan's existing
raw-likelihood limits of 0.03125 maximum error and 0.01 mean error from the failed
capture contract; no after-failure tolerance change is justified by this proposal.
Measure entity/time/modality/negation/confirmation/cancellation extraction and
operation syntax independently; report raw outputs and all failures. This is a
diagnostic, not a power-backed efficacy test or a full retrieval evaluation.
Lengthen only after qualification for the enlarged actual shape. Titan's known
numerical blocker remains closed until fresh evidence resolves the relevant
contract; CPU and separately registered subjects can progress meanwhile.

After qualified extraction and instrument review, freeze an initially small
**open DEV** B/C/D experiment with shared raw inputs, parent/variant allocation,
library and origin identities, common ceilings and arm order. Retain A only at
jointly supported input shapes. Account for library tokens, discovery/validation,
indexing, retries and failures; compare accuracy/cost under common ceilings and
show cold-start and declared-denominator amortized costs. Do not add dummy work.
The founding 64/128 and 4K/16K proposal is a later registered size/shape choice,
not an automatic requirement to launch before fit and precision evidence exists.

Register one independent learning-atlas microstudy when training qualification
passes, even if procedure confirmation is still pending. Start with the frozen
checkpoint, matched ordinary supervised control, and one justified small
treatment; retain acquisition/previous-skill/calibration panels at prospective
checkpoints and record export/served behavior. If data is engineering-supplied,
name it ordinary adaptation or a microstudy. “Validated-procedure distillation”
requires eligible independently verified development traces and a separately
frozen control. Prospective fresh predictions and confirmation remain later gates.

No initial report constitutes independent confirmation, broad preservation,
canonical publication, or release approval. Negative/inconclusive observations
reject or narrow specific hypotheses; preserve CRS-001, its dependencies, open
questions, failures and restart conditions.

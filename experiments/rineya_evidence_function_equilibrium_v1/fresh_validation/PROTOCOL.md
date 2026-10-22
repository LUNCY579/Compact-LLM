# EFE-FUNC-001 first-stage frozen design

September 29, 2026, America/Chicago. **HOLD. No fresh model experiment ran.**
The coefficient candidate remains `(1.00, 0.25)`, selected on spent development data.
The immutable design is [protocol.json](protocol.json). Its completed code/data binding
will be recorded in a create-only external preparation receipt. This is a frozen design
and bank reservation, **not a completed native preregistration**: no fresh payload,
canonical parent/content review, row scorer, or target-host qualification is bound.
Native launch is disabled and cannot be unlocked by editing the JSON.

## Question and contrasts

Test whether adding the frozen second component acquires useful H behavior while
preserving acquired T behavior. Evaluate source, T alone at 1.00, H alone at 1.00,
H alone at 0.25, and the single frozen composition. H at 0.25 is the component-matched
preservation comparator; full H is a prospectively fixed descriptive standalone arm.
This is five serial candidates, one model worker, no composition grid. Raw-sum and
shared-scale controls receive zero calls in this first confirmation budget. The old
shared-scale implementation used per-tensor scales; a global 0.675 would change it.

T comprises ledger replay, prerequisite planning, and register-machine behavior.
H comprises seven boundary categories in the data audit. Fresh tasks must exercise
meaningful correctness and appropriate action, including inapplicable plan steps,
invalid ordering, unsupported answers, and inappropriate abstention on answerable tasks.
Report strata individually; pooled success licenses a pooled claim only.

## Asset and arithmetic binding

[ASSET_AUDIT.md](ASSET_AUDIT.md) records the exact source/tokenizer and endpoint SHA-256s.
The two endpoints are partial absolute FP32 trained parameters. They were reclaimed,
as were their original checkpoints. Saved scripts cannot recreate endpoint bytes;
same-seed training or a published BF16 export is a different object. Existing vault
receipts initially provided historical custody evidence. The subsequent live pinned-
revision check in [vault_availability.json](vault_availability.json) confirms matching
original T/H LFS metadata and access. A T restoration attempt was stopped on extreme
RAM pressure, leaving an incomplete external payload; H was not attempted. No original
endpoint is yet locally hash-verified. The goal bars replacement weights; treating
that as a ban on restoring identical original objects was an overly broad initial
interpretation. Do not substitute weights, and do not retry restoration in this state.

Use the original INDEP builder's FP32 subtraction, FP64 rank-16 matrix SVD, whole
nonmatrix deltas, scaled H plus scaled T update, then source addition and final FP32
rounding. Preserve the original 220 last-20-layer key inventory and 90 untouched source
tensors. Original untouched export storage is BF16; FP32 loading losslessly upcasts it.
Do not silently replace this with HLOC's FP64 subtraction or different addition order.
Loaded-parameter parity and tied weights require native qualification; hashes/norms
alone and new synthetic fixture arithmetic do not establish that qualification.

## Fresh data reservation and exclusion policy

Reserve `EFE-FUNC001-FRESH-CONFIRMATION-RESERVATION-20260929`. No payload exists for
this reservation. Before any candidate generation, bind its JSONL bytes, canonical
parent identities, structural templates/solver graphs, content fingerprints, provenance,
raw oracle data, tokenizer-rendered prompts, all exclusions, and the scorer implementation.
Use saved records first; this preparation did not reopen protected banks.

Exclude all endpoint-training parents, calibration/selection parents, spent RCI-001/002,
historical HSKILL/SELECT views, any protected/reserved H confirmation, and quarantine
registries. The Long Context Demo stays report-only. Exact row/hash disjointness is
necessary but does not detect paraphrase or repeated solver-template leakage. The
saved T selected-parent inventory and H row-order IDs exist; full canonical parent and
semantic-template reconciliation remains unresolved. Endpoint training disjointness
does not establish source-pretraining disjointness.

Qualification and budget-estimation probes are separate development-only parents.
Predeclare all overlength, ambiguous/oracle-invalid, duplicate-parent, or unavailable-
provenance exclusions before outcomes; log reasons and counts. Do not truncate or replace
unfavorable rows. If a confirmation bank influences coefficients, generation, scorers,
margins, or stopping, mark it development and require a new untouched confirmation bank.

Fixed confirmation counts: 20,160 independent parents in each of T, H, answerable, and
broad populations. T has 6,720 parents per domain; H has 2,880 per category. Answerable
and broad strata/provenance must be supplied before registration completes. This is a
sensitivity requirement, not a claim that this many admissible tasks presently exist.
One deterministic generation per parent; multiple seeds/variants do not increase n.

## Scorer and generation contract

T semantic scoring executes an independent oracle against the full final result/state
or valid plan. Numeric substrings and incorrect extracted first answers earn no credit.
H uses frozen row-specific appropriate acts: an irrelevant question or generic refusal
is not a correct clarification. Freeze acceptable acts, dual blinded annotation and
disagreement adjudication before model outcomes. Literal format/sentinel correctness,
appropriate abstention, false abstention, and semantic correctness are separate fields.
Qualification must include intentionally wrong answers, irrelevant questions, malformed
format, repeated/contradictory answers, and unterminated outputs.

Greedy batch one; exact source chat template with thinking disabled; full original KV;
maximum 2,048 prompt tokens plus 128 generated tokens. Bind the complete source EOS
list. No silent precision change, prompt truncation, reduced generation, or omitted row.
Primary success requires semantic correctness and normal termination. Retain token IDs,
raw text, EOS/limit-hit status, each scorer decision, prompt/token counts, timings,
cache bytes, and memory peaks. Worker/resource failure produces an incomplete ledger
and HOLD; it cannot disappear from a denominator.

## Margins, multiplicity and sensitivity

Ten fixed gates share total family alpha 0.05: 0.025 for false PROCEED and 0.025 for
false REJECT, with Bonferroni over ten assertions in each direction. Each acquisition
assertion requires a paired lower confidence bound **above +5 percentage points**.
Each net-preservation assertion requires a lower bound **above -3 points**; corresponding
source/comparator-correct damage upper bounds must be **below 3%**. Repairs cannot offset
unacceptable damage in that conditional endpoint. `primary_gates` in the JSON fixes every
population, comparator, direction and threshold.

For paired gains B and losses C on n parents, the conservative exact lower bound is
`CP_lower(B,n,.00125) - CP_upper(C,n,.00125)`. Marginal binomial coverage plus a union bound
does not assume independence of gains and losses. Conditional damage uses
`CP_upper(C, comparator_correct, .0025)`. Rejection requires the opposite-direction
paired upper bound below the registered acquisition/net threshold, or a conditional
damage lower bound above 3%. Never reduce multiplicity after seeing failures.
Require at least 900 comparator-correct independent parents for each conditional gate;
otherwise HOLD, without outcome-driven supplementation.

[statistics.py](statistics.py) calculates intervals and exact conditional-damage power
using only the standard library. With 900 comparator-correct parents and true damage
1%, up to 13 losses pass the 3% gate, with about 92.7% planning power. Even zero losses
among the historical 47 comparator-correct parents has an adjusted upper bound about
11.97%; it cannot certify 3% preservation. The large fixed n addresses net sensitivity:
at 20,160 maximally discordant pairs with equal totals, the conservative paired lower
bound is about -2.13 points. This net bound alone does not imply preservation: that
extreme example damages every comparator-correct row and fails the conditional gate.
A worst-discordance normal planning calculation gives over
80% power at true zero net change. These are design calculations, not measured endpoint
sensitivity or proof of bank validity. Before native launch, qualify the full scorer,
simulate registered gain/loss alternatives and confirm sensitivity within the fixed
budget. An insensitive or invalid bank holds the experiment.

## Resource envelope and bounded execution

Use the verified compatible `.venv-mq001` runtime; do not replace shared packages.
Self-contained integration is reviewed in [NATIVE_REVIEW.md](NATIVE_REVIEW.md).
The minimum local adapter and its CPU fixtures are engineering preparation, not a
native qualification receipt. No blanket reuse of MQ, CLARE, or historical generation
receipts is permitted.

Check and exclusive-create both `git-common-dir/rineya-central-gpu0.lock` and
`C:/Users/clclc/AppData/Local/Temp/rineya-gpu-0.lock`. Keep both for SVD builds and GPU work;
only release owned tokens. Inspect actual model processes as well. Do not kill unrelated
processes or remove suspected stale locks. After runtime imports, require 4.6 GiB free
host memory and 6,000 MiB free GPU memory; ongoing floors 1.5 GiB host/0.5 GiB GPU.
Keep 4 GiB disk headroom and worker RSS at most 6 GiB. These inherited prerequisites
do not qualify the new 2,176-token maximum envelope.

At most 64 separate qualification calls/8,192 generation tokens. At most 403,200
confirmation calls/51,609,600 generation tokens, five arms, one candidate at a time.
Total native wall budget 30 hours, segments at most 30 minutes, no automatic unchanged
retries. Both time and memory must be enforced by the future supervisor, including
noninterruptible SVD/prefill. The bounded collector/process containment is not implemented
in this preparation, so launch remains disabled. If measured independent qualification
timings cannot support the fixed sensitive design within the total budget, HOLD before
outcomes; do not shrink n or weaken margins. Missing endpoints were established before
inference-cost estimation. Expected inference time/cost remains **unestimated**.

## Decision and deliverables

PROCEED requires complete asset/data/scorer/runtime/max-envelope qualification and all
ten gates on the independently registered bank. HOLD covers missing prerequisites,
incomplete outputs, low correct denominators or inconclusive intervals. REJECT requires
adequate complete confirmation establishing materially unacceptable damage or inadequate
acquisition by the opposite-direction bounds specified above; failing to prove a margin
alone remains HOLD. All prerequisites, complete pairs, fixed n, and denominator floors
must hold before any substantive verdict. No result here promotes a
model or starts JVP fitting, compression, DEQ, CLARE, training, or a third skill.

This session delivers saved-record audits, the frozen design, reproducible CPU-tested
native integration helpers and refusal/resource/statistical code, and create-only
external preparation/test receipts. Native qualification and raw paired outputs are
**NOT_RUN**; the external status receipt records zero rather than manufacturing results.

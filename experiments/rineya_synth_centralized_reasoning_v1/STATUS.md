# Status and next handoff

## 2026-10-04: CPU guards and runtime controls prepared

The [full CPU validation](evidence/inference_path_cpu_validation_20261004.json)
passed all 480 tests without skips, failures or source changes. The independent
review covered prompt-only inference, isolated worker startup, supervisor
ownership and cleanup, and chunk publication/resume cost accounting.

The preserved [task guards](evidence/task_guards_root1_20261004) contain 768 rows
in 192 content components across retrieval, instruction and answerability.
The [long-context guards](evidence/long_context_guards_root1_20261004) contain
2,304 rows in 192 worlds, with complete paired variants at 1,024/2,048/4,096
total-token envelopes. All rows are retained; independent solutions, negative
controls, source exclusions and exact token fitting pass. These are CPU packets,
not native preservation results or enrolled M0-correct cohorts.

The new study backend and chunk runner have CPU random-model/toy controls for
loss, Adam, whole-row dose, RNG, boundary restore and cumulative primitive costs.
Their native sustained fit and fresh-process continuation remain unqualified.
The separate inference contract permits only four deterministic TRAIN prompts
on a fresh unchanged base, repeat/bulk numerical controls and unchanged-base
checks. It retains the Native7 resource and numerical bounds and grants no
DEV collection, scoring or study-training authority.

Next: freeze and independently audit that limited supervised inference scope;
then qualify actual study resume/sustained fit and the separate long-context
profile. Qualified language reference labels and independent reviewer/adjudication
evidence are still missing. Training, protected banks and publication remain
closed. There is no M0/M1/M2 transfer or retention result, and the research
question remains unanswered.

## Latest: short supervised native mechanics passed; comparison pending

Attempt 7 passed the complete supervised short scope in 353.437 seconds, with
normal exit and verified owned-tree/sampler/NVML cleanup. The
[independent audit](evidence/nq001_attempt7_independent_audit_20261003.json)
verified 620 returned API calls, 75 forwards / 16,476 positions, two native and
one reference updates, all 60 CE checks, exact 112 gradient comparisons,
bitwise restoration/continuation and unchanged 310 base tensors. CPU observation
windows stayed below 200 ms (maximum 141 ms); all memory and GPU floors held.
Raw outputs and local private STATE1 are retained in
[attempt 7 evidence](evidence/nq001_attempt7_20261003).

This qualifies `PASS_SHORT_SCOPED_NATIVE_MECHANICS` only. Full E0/E1, sustained
27-update fit, M0/M1/M2 study training, fresh transfer and prior-correct retention
remain unqualified/unmeasured. No protected banks or publication were enabled.

The [DEV statistics policy](DEV_STATISTICS_POLICY.json) and
[23-test CPU receipt](evidence/dev_statistics_cpu_validation_20261003.json)
freeze component-level comparisons, pre-candidate enrollment/cohort hashes,
exact simultaneous retention bounds and separate repairs. Missing qualified
guards or precision remains inconclusive; arithmetic is not grader authority.
The full research goal remains active and unanswered.

## CPU preparation after native qualification

[Compact alias policy 002](EVALUATION_ALIAS_AMENDMENT_002.md) preserves the whole
fresh-family pool and all task conditions while fitting every one of 384 DEV
prompts under the unchanged 480+32 envelope (maximum 452 prompt tokens).
Prompt-only arrays are bound independently of CPU-exposed oracle labels. The
[47-test input/evaluation/schedule receipt](evidence/compact_schedule_cpu_validation_20261003.json)
passes without skips or source changes.

[Study schedule](STUDY_DEV_REGISTRATION_DRAFT.json) freezes 27 full updates,
540 whole sequences per trained arm, identical LR and three nine-update resume
chunks. It preserves 130,562 versus 130,574 non-padding input tokens and the
matched parent vector. Qualification STATE1 is never a study warmstart.
This is a closed registration draft, not training authority. Task-retention
and long-context CPU guard preparation and the actual training/collector
implementation remain ongoing; qualified language reference/review authority
is still missing. No M0/M1/M2 result has been obtained.

## Latest: worker mechanics verified; full supervision still failed

Attempt 6 returned `PASS_SCOPED_NATIVE_MECHANICS`: 40 native training forwards/
backwards, two optimizer updates, 20 independent reference gradients, one
reference update, exact restoration/continuation, and all 310 frozen-base tensor
checks. Its [independent audit](evidence/nq001_attempt6_independent_audit_20261003.json)
verified all 620 API calls, 60 loss checks, 112 gradient comparisons and retained
artifact hashes. The controller aborted on a 219-ms CPU observation window,
above the unchanged 200-ms bound. **Full supervised qualification did not pass.**
All raw measurements, private local STATE1, and source/config snapshots remain in
[attempt 6 evidence](evidence/nq001_attempt6_20261003). No study-training or
transfer/retention result is claimed. The next sampler investigation keeps every
resource, numerical and dose limit fixed and requires CPU controls before launch.

The [direct process/RAM successor](NQ_PROCESS_AMENDMENT_006.md) removes the
measured 25–35-ms RAM-query bottleneck with exact read-only parity controls.
[SYNTH_NQ001_PROC7.json](configs/SYNTH_NQ001_PROC7.json) freezes 25 source and
64 runtime bindings, including System32 kernel32.dll. Its static preflight and
[288 CPU tests](evidence/process7_final_cpu_validation_20261003.json) passed,
without skips or source changes. The next actual scope still needs current
3.3125-GiB RAM admission and full supervised success; no limits were lowered.

The new CPU-only [fresh DEV panel policy](EVALUATION_PANEL_POLICY.json) creates
32 joined components each for R/S/RS and 384 crossed rows. It declares wider,
deeper and UNKNOWN strata plus a held-out ledger rendering. Inventory-completeness
and independent rule-parser fixes now pass negative controls. The
[rebuilt root-bound bundle](evidence/fresh_dev_root2_20261003) and
[CPU receipt](evidence/evaluation_panel_root_control_20261003.json) preserve
all rows and source bindings. Its
measured prompts span 185–572 tokens; 59 rows exceed the current 480-token prompt
allowance plus 32 generation tokens. Preserve every row; native evaluation is
unadmitted until a separate envelope or source amendment is registered. DEV
labels are CPU-exposed, and prior-correct membership remains unenrolled.

## Current native qualification handoff — 2026-10-03

Attempt 5 is now terminal and failed before its first optimizer step: a milestone
producer used an uppercase stage name rejected by the writer. Twenty forward/
loss/backward sequences completed; zero-adapter and capture numeric checks passed.
The supervisor also recorded CPU observation staleness. Preserve the separate
worker and controller errors in [attempt 5 evidence](evidence/nq001_attempt5_20261003).
The [boundary successor](NQ_BOUNDARY_AMENDMENT_005.md) fixes the name contract and
requires actual worker-journal CPU integration plus sampler timing review before
another frozen scope. No scoped native qualification or study pass is recorded.

The [sixth scope](configs/SYNTH_NQ001_BOUND6.json) freezes the corrected producer
and single-query sampler with all earlier dose, resource and numerical limits.
Its [255-test CPU run](evidence/boundary6_final_cpu_validation_20261003.json)
passed without skips, with unchanged before/after source hashes and actual
worker-journal integration. Static preflight passed; native execution is pending.

Earlier attempts 1–4 are terminal and preserved.
Attempt 4 loaded the pinned base and reached one backward call, but the external
GPU observation became stale before any optimizer step. No native qualification
pass or M0/M1/M2 capability result exists. Its raw receipt and source snapshot are
in [evidence/nq001_attempt4_20261003](evidence/nq001_attempt4_20261003).
The [telemetry successor](NQ_TELEMETRY_AMENDMENT_004.md) specifies direct pinned
NVML observations, exact owned-Job worker accounting and durable partial numeric
measurements. It retains every numerical/resource/dose limit and requires passing
CPU/read-only controls and a new committed scope before another attempt.
The [final successor CPU run](evidence/nvml5_final_cpu_validation_20261003.json)
passed all 252 discovered tests without skips, with test IDs and unchanged
before/after source hashes. [SYNTH_NQ001_NVML5.json](configs/SYNTH_NQ001_NVML5.json)
freezes the fifth bounded mechanics scope, including the actual System32 NVML
DLL hash. Its static preflight passed before launch; its native attempt failed
as recorded above.
The research goal remains active; fresh transfer and retention remain unmeasured.

## Later 2026-10-03: CPU_CORPUS_MATERIALIZED_SCHEDULE_MATCHED_NATIVE_UNQUALIFIED

The [matched corpus milestone](CPU_DATA_MILESTONE.md) now has an immutable source,
candidate/student/token pool and paired schedule under
[data/crossed_source2_v1_20261003](data/crossed_source2_v1_20261003), plus a
[separate disk-level audit](evidence/crossed_v2_independent_audit.json).
M1/M2 each have 540 prospective sequences / 27 full updates, with 130,562 versus
130,574 input tokens and 5,614 versus 5,615 supervised tokens. Source parents,
exposures, occurrence order, target format and global cardinality/depth margins
match. Every condition crosses all four renderings; RS remains out of adaptation.
Residual joint/per-update difficulty differences are explicitly reported.
The [final 137-test run](evidence/crossed_corpus_final_tests.json) passed without
skips, including local tokenizer integration and isolated CPU toy helper tests.

V1's exact 736-token residual lower bound violated the 512-token limit. Its
entire pool/code/evidence is preserved in data/crossed_v1_blocked_20261003.
Separately preregistered source-policy 2 fixes the unintended depth change in R
request rebinding without lowering the budget or any checking limit.

The [native proposal](NATIVE_QUALIFICATION_PROPOSAL.json) and CPU precision
helper tests prepare an isolated current-runtime path; they establish no GPU,
pretrained-model, training, transfer or retention result. Next implement and
qualify the bounded owned loader/worker/supervisor, preserving numerical and
resource guards; then freeze fresh baseline guards and atomic/RS evaluation.
The entire M0/M1/M2 research objective remains active and unproved.

## 2026-10-03: CPU_MECHANICS_IMPLEMENTED_NATIVE_UNQUALIFIED

The owner authorized the bounded CPU R/S compiler milestone. It now implements
strict public schemas, two independent solution paths, source-content hashing,
dependency-connected parent splits, surface/condition variants, fail-closed
records and allowlisted transient exports. See [CPU_COMPILER.md](CPU_COMPILER.md)
and [CPU audit receipt](evidence/cpu_audit_2026-10-03_final.json).

All 67 tests pass: 23 retained registration checks, 25 semantic tests including
5,645 independent bit-mask S tasks, and 19 compiler/isolation tests. The default
audit constructs 384 fixtures from 48 parents in 47 components, checks both paths
on every row, verifies 96 answer-changing and 288 answer-preserving witnesses,
and rejects all 1,536 deliberately incorrect candidates. These are CPU mechanics
results, not acquisition, transfer, retention or native qualification evidence.

Transient TRAIN exports have 100 rows per arm with matched source parents,
exposures, skill allocation, final answers and verifier. No training corpus is
saved. Counts/bytes are not measured-token matching. Condition/rendering and
answer-position cues need balanced allocation before native data admission.
CPU eval_fixture labels are exposed and cannot become sealed confirmation.
RS remains excluded from adaptation; held-out RS semantics and label-forgery
rejection are tested. Founding manifest, receipt and other tracks stay intact.

### Owner-delegated pilot choices

The owner subsequently delegated missing decisions. [PILOT_DECISIONS.md](PILOT_DECISIONS.md)
and [PILOT_DECISIONS.json](PILOT_DECISIONS.json) freeze unchanged dense Titan
devfreeze, all-layer rank-8 q/v LoRA (112 FP32 matrices; 1,146,880 trainables),
and 131,072 non-padding tokens per trained arm for the first small DEV dose.
The [exact prospective inventory](pilot_trainable_inventory.json) is retained;
local weight/config/tokenizer hashes were refreshed and match the recorded pins.
No weights were deserialized or model worker started. Hardware inventory
observations do not qualify resource admission or measured training cost.

The research question remains **unanswered**. Compare unchanged M0, verified
surface-varied atomic M1 and condition-diverse atomic M2 on fresh families,
with RS reserved for transfer and disjoint enrolled prior-correct guards.
Familiar-template gains alone cannot establish transfer. Matching must preserve
whole examples, complete parent exposure vectors and optimizer update count.
If the frozen token tolerance cannot be met, record BLOCKED_MATCHING rather
than dropping parents, padding tokens, truncating targets or topping off one arm.

Capture likelihood numerics, current backend math/optimizer/restore, sustained
training fit, balanced tokenized schedules, fresh guards/evaluation, language/
long-context preservation scope, statistics/power, effect/retention margins and
complete run-specific launch registration remain prerequisites. No inference,
training, protected-bank access, weight change, paid call, merge or upload was
performed. Native launch flags remain closed. A null at this small dose cannot
reject condition diversity generally; retain uncertainty and preregister any
successor before fresh evaluation. Preserve CRS-001 and all prior discoveries.

## Founding status and next handoff — 2026-10-02


**State:** `DESIGN_REGISTERED_NATIVE_UNQUALIFIED`.

## Delivered

A dedicated SYNTH-CRS-001 charter, non-discard policy, architecture, controlled
R/S task-diversity proposal, data/leakage contracts, primary source references,
descriptive manifest, and offline registration-validation tests. Work is based
on CRS-001 at `7459b614bcb017ae4f0160b15e0ccd53473bb990`; it does not replace that
track or amend its historical experimental question. See
[evidence/local_validation.json](evidence/local_validation.json) for executed checks.

## Not delivered or established

No task generator, independent task verifier, dataset exporter, online controller,
procedure memory, training driver, or model inference is implemented here. No
corpus is generated or downloaded. No native qualification, training, held-out
measurement, retention/composition effect, cost advantage, or release is claimed.
The manifest checker validates plan invariants only and is not a launch gate.

## Next bounded implementation

Implement deterministic atomic R/S task compilation and independently checked
reference solvers, then lineage-safe variants and export allowlists. Add negative
controls before model work. Keep RS excluded from training. This is the next
engineering milestone, not an automatic job started by this delivery.

Before any native run, identify the actual approved 0.6B checkpoint and tokenizer;
bind hashes, precision, masks, runtime, resource admission, data/guard identities,
statistical thresholds, budgets, and explicit launch approval in a new run
manifest. Reuse qualified existing infrastructure only after checking its exact
scope. Missing information must block a run, not be replaced with guessed defaults.

## Preservation / parking

There is no sunset date. Temporary parking must retain this status, unresolved
questions, exact tip, dependencies, and restart conditions. Rejecting a hypothesis
is allowed; discarding the track or rewriting null results as success is not.
Retirement, archival, replacement, deletion, or obsolescence designation requires
explicit owner approval and the root charter's provenance handoff.

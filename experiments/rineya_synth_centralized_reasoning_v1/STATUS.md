# Status and next handoff

## 2026-10-05 UTC: third M0 batch audited; fresh RAM callback candidate measured

The [third fixed M0 config](configs/SYNTH_SHORT_M0_ATOMIC_BATCH03_A1.json) completed
64 rows in 231.828 seconds. Its
[independent audit](evidence/short_m0_atomic_batch03_attempt1_independent_audit_20261005.json)
passes 22,103 checks: all 64 EOS endings, 514 token prefixes, 516 physical forwards
/ 120,375 positions and 836 returned API calls reconcile. Zero-B parity, all 310
base tensors, resource/freshness limits and owned cleanup pass. The
[591 raw files and 83 source snapshots](evidence/short_m0_atomic_batch03_attempt1_20261005)
are preserved. There are now 192/384 audited atomic/RS DEV outputs, with 127 EOS
endings and 65 generation caps. Three atomic and twelve short task batches remain.
No correctness scoring, enrollment, M1/M2 output or transfer result exists.

After native and audit work became terminal, the
[matched CPU comparison](evidence/worker_ram_probe_cpu_20261005/CPU_ram_callback_comparison_20261005.json)
ran 512 guards per callback under identical current checks. Default guard wall
time was 6.514 seconds, including 5.909 seconds in the RAM callback. The staged
candidate's guard wall time was 0.148 seconds, including 0.00398 seconds in fresh
`native_process.system_memory_status()['available_bytes']` calls. Every guard
and each of its three callbacks returned 512 times without failed/pending work.
Fourteen source and 128 runtime pins match before/after; owned CPU processes are
terminal. Six light controls cover fresh-call counts, API error propagation,
invalid fields/types/ranges and unchanged resource/deadline/heartbeat checks.
[Twenty-nine files](evidence/worker_ram_probe_cpu_20261005/preservation_receipt.json)
retain partial counts before metadata, complete receipts and exact sources.

This is a CPU component improvement in sequential fresh contexts. Volatile RAM
readings are retained without asserting byte equality. Native tensor/body work,
GPU guards, disk-stall prevention, whole-scope fit and continuation remain
unqualified. Original inference helpers remain unchanged. The next training-fit
step is a separately reviewed backend-only RAM-callback amendment, after package
integration and appropriate CPU controls, followed by fresh scoped native A/B
qualification with unchanged model, parameters, dose, optimizer, numerics and
resource/freshness/deadline gates. No study training, unrestricted retry or gate
relaxation is opened by these receipts. Language reference/reviewer authority
and the full three-arm result remain outstanding.

## 2026-10-05 UTC: CPU guard callback costs attributed; helpers unchanged

The [512-call diagnostic](evidence/guard_component_cost_cpu_20261005/CPU_guard_callback_attribution_20261005.json)
retains each original callback and all checks, adding timing wrappers only.
Its loop took 5.661 seconds: available RAM 5.112 seconds, heartbeat reads 0.421
seconds and owned-process RSS 0.108 seconds. Every callback and guard returned
512 times without failed/pending work. Twelve source and 128 runtime pins match
before/after; owned CPU processes are terminal. Six light controls pass, including
metadata-failure preservation and fresh-start runtime-inventory stability.
[Twenty-seven files](evidence/guard_component_cost_cpu_20261005/preservation_receipt.json)
preserve actual counts before optional metadata, complete results and preparatory
source history. This identifies a cost driver in this CPU context; GPU/body work,
historical native causality and full-run fit remain unmeasured.

The [primary-source review](evidence/available_ram_primary_source_review_20261005.json)
binds psutil 7.2.2's page-based available-physical-memory computation and the
existing `native_process.system_memory_status()['available_bytes']` byte field.
Their documented meanings agree, while sequential snapshots are volatile. A
future CPU candidate must retain fresh queries, every existing check/frequency,
numeric floors and failure handling before separate native qualification. No
helper substitution, gate relaxation, performance guarantee or retry is opened.

## 2026-10-05 UTC: second M0 batch audited; CPU guard costs measured separately

The [second fixed config](configs/SYNTH_SHORT_M0_ATOMIC_BATCH02_A1.json) completed
all 64 rows in 458.062 seconds. Its
[independent audit](evidence/short_m0_atomic_batch02_attempt1_independent_audit_20261005.json)
passes 59,507 checks: 1,825 token prefixes, 1,827 physical forwards / 647,404
positions and 2,147 returned API calls reconcile with the fixed packet. Thirty
rows ended at EOS and 34 reached the generation cap. Zero-B parity, all 310 base
tensors, resource/freshness checks and owned cleanup pass. The
[1,902 raw files and 83 source snapshots](evidence/short_m0_atomic_batch02_attempt1_20261005)
are preserved. The [reusable independent auditor](evidence/short_m0_reusable_auditor_cpu_20261005)
retains eight passing light controls and the actual batch01 audit source.

There are now 128 of 384 atomic/RS DEV rows captured and audited, with 63 EOS
endings and 65 generation caps. Four atomic and twelve short task batches remain;
no correctness grades, M0 enrollment, candidate outputs or comparison exist.
Every remaining batch needs its own frozen config, fresh admission and audit.

After all native and audit work stopped, the
[CPU component benchmark](evidence/journal_guard_cost_cpu_20261005/CPU_component_cost_addendum_20261005.json)
replayed the recorded 1,266 primitive calls and 1,120 decoder regions. Each
factor executed 7,722 CPU guard sites derived from the current source; historical
native guard counts were not logged. Journal-only wall time was 1.097 seconds,
CPU-guard-only 85.330 seconds and combined 75.689 seconds. Combined guard time
was 74.409 seconds; durable emits took 1.112 seconds, including the nested stage
evidence emits. These are separate fresh-context observations, not additive
timings or a causal explanation of the earlier native update. The scalar model
has 28 metadata parameters and placeholder tensor/inventory/CE/optimizer bodies.
Native parameter/tensor work and GPU guards remain unmeasured; no full-run
forecast or qualification follows.

The first CPU attempt failed before measurement on a duplicate constructor-owned
receipt write. Its exact source and failure remain preserved. The reviewed
successor consumes the original ownership receipt, passes six light controls and
completed in an owned CPU Job with unchanged limits and verified terminal cleanup.
[All 82 benchmark files](evidence/journal_guard_cost_cpu_20261005/preservation_receipt.json)
and 37 current CPU runtime pins are retained. The next useful timing check is to
separate owned-RSS, available-RAM and heartbeat callback costs under the same
guards. No source integration, resource relaxation, durability reduction or
native retry is opened. Qualified language references/reviewers and sustained
training/continuation still block the full comparison; the question is unanswered.

## 2026-10-05 UTC: first fixed M0 baseline batch completed and audited

The [first atomic DEV batch config](configs/SYNTH_SHORT_M0_ATOMIC_BATCH01_A1.json)
was reviewed at committed source `d8179f54ef651b6f8a7053ab0748d46bd737b508`.
The owned supervisor completed all 64 ordered prompt-only rows in 355.734 seconds
under unchanged resource, freshness and deadline checks. There were 1,679
physical forwards / 570,293 input positions, including two zero-adapter parity
forwards. All 1,999 registered API attempts returned; no backwards, optimizer
steps, checkpoint restores or scores were performed. Thirty-three rows ended
at EOS and 31 reached the 32-token generation cap; these remain distinct.

The [independent terminal audit](evidence/short_m0_atomic_batch01_attempt1_independent_audit_20261005.json)
passes 52,820 checks. All 1,677 token prefixes, final rows, generated-only decode,
finite likelihoods and costs reconcile. Physical zero-B parity and all 310 base
tensors remain unchanged. CPU/GPU freshness, resource floors, the whole-scope
deadline and owned process/sampler/NVML/lease cleanup pass. This qualifies this
one raw collection scope; it establishes no correctness, enrollment, retention,
study-training or transfer result. The
[1,754 raw files, 83 source snapshots and config](evidence/short_m0_atomic_batch01_attempt1_20261005)
are preserved with byte hashes.

Only 64 of the registered 384 atomic/RS DEV rows and none of the 768 short task
guard rows have been captured. The remaining five atomic and twelve task batches
require their own new configs, current admission and independent terminal audits.
Full-context qualification and the language authority gate remain open. The
[language handoff](LANGUAGE_GATE_HANDOFF.md) now correctly distinguishes the
central program's authored/tokenized candidates from its absent qualified
references, reviewers and ratings. Sustained training still lacks successful A/B
continuation evidence. The M0/M1/M2 research question remains unanswered.

## 2026-10-05 UTC: recomputation A aborted its CPU freshness gate

The [amended A scope](configs/SYNTH_BACKENDQ002_RECOMPUTE_A1.json) was contained
and stopped after 193.046 seconds. The
[independent audit](evidence/backendq002_recompute_A_attempt1_independent_audit_20261005.json)
verified 1,745 API attempts / 1,744 returns / one pending outer training forward,
21 CE/backward rows and one complete optimizer update. Recorded decoder work
was 56 no-grad / 590 forward / 588 recompute calls. The
[raw receipts and 24 source snapshots](evidence/backendq002_recompute_A_attempt1_20261005)
are preserved. No terminal worker/region summary, C9, W9/W10 or final base/restore
proof exists. This is not sustained, resume or native-fit qualification.

The terminal CPU sample took 328 ms, including 323.1869 ms in the disk-free
phase, above the unchanged 200-ms freshness bound. The prior start gap was 94 ms;
the exact abort-check time/thread is unlogged. Wall query time includes possible
descheduling, so no filesystem/filter/writer/scheduler cause is established.
Sampled GPU free memory remained above 3,122,126,848 bytes; no direct terminal
Torch peak/free receipt exists. Owned processes, sampler threads, NVML and lease
cleanup are terminal and independently verified.

The [bounded disk investigation](evidence/disk_watchdog_cpu_20261005/preservation_receipt.json)
passed five controls, 96 read-parity brackets and 128 durable writes but did not
reproduce the stall. A verified volume-root probe can preserve the existing
total-volume-free counter; prevention under load remains inconclusive. No
resource/freshness relaxation, stale-value cache, writer durability reduction,
unrelated app termination or unchanged retry is authorized. The study comparison
and qualified language-reference/reviewer evidence remain outstanding.

The [observed timing audit](evidence/recompute_A_observed_timing_audit_20261005.json)
also exposes a separate throughput risk. The first complete update took 59.208
seconds through gradient clearing and 69.994 seconds including its required base
check. At that check, 410.847 seconds remained for nine updates, checkpoint and
witness work, further checks and cleanup. These journal wall intervals include
guards and durable writes; they are neither synchronized GPU timings nor a full
runtime estimate. A disk-only change cannot establish full-scope fit. Measure
matched-count journal/guard overhead on CPU before committing another full A
scope, then obtain the required sustained/checkpoint timing under unchanged gates.

## 2026-10-05 UTC: amended backend and final-adapter helpers verified on CPU

The [recomputation amendment](BACKEND_RECOMPUTATION_AMENDMENT_002.md) is integrated
with explicit decoder-region counts, 82 runtime pins and direct predecessor
region/cost verification. Its [staging evidence](evidence/backend_recompute_scope_cpu_20261005)
and 52 combined root backend controls pass. The
[final-adapter helper](final_adapter_inference.py) requires a complete source-bound
final27/540 three-chunk chain before native restoration, copies only q/v factors,
privately validates archive RNG and preserves caller inference RNG. Its two
CPU controls and [historical/successor evidence](evidence/final_adapter_inference_cpu_20261005)
are retained; native/CUDA restoration remains unqualified.

The complete [571-test CPU run](evidence/recompute_adapter_full_cpu_validation_20261005.json)
passes without skips or source changes. Fresh resource admission and new scoped
native qualification still precede sustained/resume claims. No baseline/study
score or transfer/retention result exists; language-review authority is missing.

## 2026-10-05 UTC: counted full recomputation candidate integrated

The [candidate installer](checkpoint_successor_profile.py) preserves evaluation
mode and parameter objects/storage/dtypes/flags, disables checkpoint early-stop,
uses nonreentrant full decoder recomputation and bypasses it under no-grad.
Seven focused root controls pass, including sticky checkpoint/guard failures,
partial installation, exact attribute restoration and replacement-object
identity rejection. No backend scope or native permission is changed yet.

The [current-source tiny CPU proof](evidence/checkpoint_full_recompute_cpu_20261005/cpu_full_recompute_identity_successor_20261005.json)
passes output/CE numerical equality, all 112 gradient/Adam/RNG comparisons,
independent gradients and in-memory continuation. All 1,680 forward regions and
1,680 recomputations are paired and return normally without failed/pending work;
before/after source hashes match. Historical pre-fix proofs remain preserved.
Saved-storage and tiny-shape limits remain explicit; native VRAM/disk-resume fit
is unqualified. A separate source-bound backend amendment with actual additional
work counts is being prepared under unchanged parameters, dose and gates.

## 2026-10-05 UTC: CPU recomputation mechanism proof passed

The [preserved CPU proof](evidence/checkpoint_memory_cpu_20261005/preservation_receipt.json)
uses a separately declared random tiny 28-layer Qwen, BF16 base/FP32 compute,
evaluation mode, 112 q/v factor tensors and 6,272 tiny trainable parameters.
It retains the native study's chosen 1,146,880 parameters and 131,072-token budget.
Two complete 20-sequence updates match the uncheckpointed path in outputs/CE,
all gradients, clip/Adam state and Python/CPU RNG; an independent 20-row
autograd reference passes and fresh in-memory continuation matches.

First-row saved-for-backward unique storage fell from 286,156 to 12,804 bytes.
This includes saved weight/cast storage and is not peak RAM/VRAM or native-fit
evidence. The original 40-row branch recomputed 1,120 decoder regions; fresh
20-row continuation adds 560. No published weights or CUDA were used. A bounded
successor must account for recomputation explicitly, preserve exact method and
parameter state, pass CPU controls, then obtain separate supervised native and
disk-checkpoint qualification under unchanged gates. No retry is opened here.

## 2026-10-05 UTC: short M0 collector implemented; admission closed

The [short M0 collector CPU receipt](evidence/short_collector_cpu_20261005/cpu_validation.json)
records three reviewed modules and 16 passing root CPU/mock controls. The
complete [555-test CPU run](evidence/short_collector_full_cpu_validation_20261005.json)
also passes without skips or source changes. The
collector binds one of the 18 fixed 64-row batches, preserves token/row partials
and terminal costs, requires unchanged base bytes and physical zero-B parity,
and uses the existing owned-Job/resource/deadline/cleanup algorithm. Decoder
Python/PYD import sources are pinned recursively; unfinished token prefixes
are marked `in_progress`, while EOS and actual 32-token cap semantics remain
distinct. The initial 14-test version and reviewed successor are preserved.

No native collector config, baseline output, score or enrollment exists.
Current VRAM is below the unchanged four-GiB admission floor. Collector
integration grants no training, language/long-context, scoring or retention
qualification. The next qualified run still requires a clean committed source
scope, current host admission and an independent terminal audit.

## 2026-10-05 UTC: backend phase A failed its GPU floor

The first [phase-A scope](configs/SYNTH_BACKENDQ001_A1.json) exited with worker
failure after its first training forward returned. The
[independent audit](evidence/backendq001_A_attempt1_independent_audit_20261005.json)
verified 353 returned API calls and three physical forwards / 585 positions,
including successful zero-adapter parity. Zero losses, backwards, optimizer
steps, C9 checkpoints or W9/W10 witnesses completed. Only three pretraining
base-byte checks completed; there is no final preservation proof or sustained
qualification. [All raw evidence and 22 source snapshots](evidence/backendq001_A_attempt1_20261005)
are retained. Owned processes, samplers, NVML and the lease are terminal.

The direct CUDA diagnostic reported 417,288,192 free bytes, below the unchanged
536,870,912-byte floor. Sampled NVML observations missed that direct/API-time
floor violation; they do not override it. A had more free VRAM at admission
and a lower Torch allocation peak than Native7, while the nonallocator residual
was higher at failure. This does not prove intrinsic profile infeasibility or
identify an app's historical allocation. A separate
[current read-only snapshot](evidence/current_GPU_contention_snapshot_20261005.json)
shows 2,202 MiB free and StarRail among current GPU processes, below the existing
4-GiB native admission requirement. No unrelated app was controlled or stopped.

Native admission and unchanged retry remain closed. Continue CPU collector
integration and a separately evidenced memory/recomputation investigation with
the same parameters, data, optimizer, numerical tolerances and resource floors.
Study training, transfer/retention and the qualified language gate remain open
requirements; the research question is unanswered.

## 2026-10-04: sustained/resume qualification implemented; native fit pending

The reviewed [backend qualification](BACKEND_QUALIFICATION_001.md) is integrated.
Its 45 focused CPU/mock controls pass, including continuous versus fresh-restored
update ten, exact dose/cost checks, predecessor forgery rejection, physical-method
restoration structure, isolated startup, supervisor containment and terminal
cleanup failures. [Staging evidence](evidence/backend_qualification_staging_20261004)
preserves the initial mock-cleanup failure and passing successor.

The first aggregate [539-test root run](evidence/backend_qualification_root_cpu_validation_20261004.json)
found one test-isolation failure: semantic solver modules loaded by other tests
violated a check intended for the prompt adapter's isolated process. The check
now runs the complete fixture under fresh `-I -B` Python; all 13 focused adapter
controls pass. The complete [successor CPU run](evidence/backend_qualification_root_cpu_validation_002_20261004.json)
passes all 539 tests without skips or source changes. No native phase A/B
worker, study checkpoint or acquisition/transfer/retention score exists yet.

## 2026-10-04: short raw-collection CPU work bounds prepared

A [six-test CPU batching plan](evidence/short_collection_plan_cpu_20261004/cpu_validation.json)
retains all 384 atomic/RS DEV and 768 task-guard prompts in 18 fixed batches of
64 rows. Whole-row order and prompt-only arrays are bound before any model
output. Each batch reserves at most 2,050 forwards including two zero-parity
controls; input-position caps sum every possible generated prefix explicitly.
These are prospective work bounds, not measured throughput or native authority.
No baseline response, score or source-correct enrollment was produced. Actual
short DEV collection needs its own committed collector/source/resource scope;
language and long-context qualification remain separate.

## 2026-10-04: long prompt inputs prepared on CPU

The integrated [streaming prompt adapter](long_inference_inputs.py) passed
[13 focused CPU controls](evidence/long_prompt_inputs_root1_cpu_validation_20261004.json)
and a complete isolated-Python run. All 2,304 rows / 5,431,296 prompt positions
are retained at the 992/2,016/4,064 prompt-token envelopes plus 32 generation
tokens. Prompt-only arrays are identical across the original, buffered and
integrated preparations. The integrated run took 15.125 seconds and an OS
lifetime memory peak of 155,480,064 bytes. Source/parser/tokenizer/order/byte
bindings were checked before publication; no oracle sidecar or framework was
used. [Complete attempts, code snapshots and CPU evidence](evidence/long_prompt_inputs_root1_20261004)
are preserved. These inputs grant no native long-context execution, prior-correct
enrollment, language qualification or preservation result.

## 2026-10-04: limited native inference qualification passed

The first four-TRAIN-prompt inference attempt completed in 88.922 seconds with
normal supervised exit and verified owned-tree, sampler and NVML cleanup. The
[independent audit](evidence/infq001_attempt1_independent_audit_20261004.json)
passed 1,270 checks of raw/preserved hashes, source/runtime/model/data bindings,
403 returned API calls, 84 forwards / 28,627 positions, all four exact repeat
and prefix/bulk controls, zero-B physical parity and 310 unchanged base tensors.
Continuous observed CPU freshness stayed below 200 ms (maximum 140 ms); GPU
freshness stayed below two seconds (maximum 516 ms). Resource floors held.
[Raw terminal evidence and source snapshots](evidence/infq001_attempt1_20261004)
and the [frozen scope](configs/SYNTH_INFQ001_TRAIN4.json) are retained.

This is `PASS_SHORT_SCOPED_NATIVE_INFERENCE_MECHANICS` only. No answers were
graded, optimizer steps/backward calls were zero, and full DEV collection,
study training, long-context execution and transfer/retention remain closed
or unqualified. Next work is a separately registered sustained nine-update
backend and fresh-process continuation qualification, plus the long-context
prompt input path. Qualified language reference/review authority and enrolled
guard precision are still requirements. The research question is unanswered.
The [language handoff](LANGUAGE_GATE_HANDOFF.md) records both missing review
authority and the insufficient zero-loss precision of the central 64-world
proposal under SYNTH's existing margin; it grants no qualification or launch.

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

# Prospective native skill acquisition

This package implements the proposed COMP-RS S1 update, data admission,
checkpoint, accounting, and single-arm runner. It does not register a recipe or
authorize training. The existing S0 competence read uses a different, unchanged
checkout.

## Current status

The native S1 recipe is an **unapproved proposal**. Native qualification, measured
operation budgets, completed S0 prerequisites, and separate S1 authorization
are required before a production invocation. Pure fixtures and tiny CPU models
do not supply those requirements. This package currently has no native
qualification entry point.

The production runner rejects an unapproved packet before importing a machine
learning framework. It accepts one named independent arm. Decisions about H1,
later arms, composition, and retention remain governed by the registered study.
A completed donor file is not evidence that its skill was acquired.

## Components

| File | Responsibility |
| --- | --- |
| `data.py` | Verify TRAIN manifests, canonical and policy files, row roles, IDs, prompts, and rendered token census. |
| `plan.py` | Fix exposure order, sample seeds, and separate exposure count C from applied Adam count U. |
| `ce_plan.py` | Bind unapproved 3,000-row CE8 exposure/label plans; permit checkpoint metadata export only for explicit synthetic fixtures. |
| `ce_journal.py` | Persist synthetic CPU CE8 reservations, observed update proofs, C/U and restore fences; refuse native/proposal execution. |
| `ce_controller.py` | Prepare distinct qualification/training requests and shared lifecycle wiring; empty native registries keep execution dormant. |
| `ce_ingestion.py` | Revalidate complete TRAIN/teacher sources, stream tokenizer census and retain at most eight prepared token rows. |
| `ce_qualification.py` | Validate scope-bound qualification metadata and numerical observations; preserve DRAFT_UNQUALIFIED until a real native verifier exists. |
| `protocol.py` | Validate authorization, S0 ancestry, source/runtime bindings, qualification, and measured budgets. |
| `update.py` | Verify sampled-policy identity and raw log probabilities; apply the proposed MaxRL loss; extract a donor relative to the actual warm anchor. |
| `checkpoint.py` | Stream parameters, Adam state, module modes, and process RNG state into two verified rolling checkpoints. |
| `ce_checkpoint.py` | Admit synthetic CPU CE8 storage mechanics with exact row/token/recipe bindings and C=U; deny native scope before IO. |
| `journal.py` | Preserve physical attempts and costs across rollback; bind returned update and restore proofs. |
| `runner.py` | Hold one shared GPU lease, enforce resource budgets, collect 16 trajectories per boundary, and publish a sealed result. |

The proposed atomic unit is 3,000 TRAIN prompts with B=2 and G=8. Uniform groups
contribute zero while the loss denominator remains 16. An entirely uniform
boundary advances C without an Adam step. This is the proposal's behavior, not
a claim about effective learning. MT has two balanced units.

All independent arms start from the same deployed FP32 warm anchor. Extracted
donors use that anchor as their common reference. Sequential arms and alternate
objectives require separate integration and are rejected by production
admission.

## Verification

Run the framework-free suite from the repository root:

```powershell
python -m unittest discover -s experiments/skill_acquisition_native_v1/tests -t . -v
```

Tiny actual CPU tests in `test_update.py` and `test_checkpoint.py` are skipped by
default. Enable them only with resource headroom and no competing native model
worker:

```powershell
$env:COMP_RS_RUN_NATIVE_CPU_TESTS = '1'
python -m unittest experiments.skill_acquisition_native_v1.tests.test_update experiments.skill_acquisition_native_v1.tests.test_checkpoint -v
Remove-Item Env:COMP_RS_RUN_NATIVE_CPU_TESTS
```

These tests verify actual gradients, EOS/length masks, frozen state, donor
extraction, parameters/Adam/RNG resume, and corrupt-checkpoint refusal. Passing
them does not constitute qualification on the registered CUDA model.

## Prospective supervised core

`ce_update.py` supplies the common eight-row teacher-CE boundary for the
unapproved R oracle-supervision alternatives. It uses a labeled-token mean per
row, equal row weights, one-shift target/EOS masks, global norm clipping before
Adam, and frozen-state/U proofs. Reset and failure cleanup remain observable;
an exception after an Adam attempt requires discarding or restoring live state.

The earlier core's five pure and ten actual CPU checks passed in the superseding
`evidence/CE_CORE_CPU_CHECKS_V2_20261001.json` receipt. Review found and fixed a
second-moment sign check that covered only the final chunk. The new multi-chunk
regression proves earlier corruption is rejected before forward, clipping or
Adam work, preserves parameters and optimizer state, and admits the repaired
nonnegative control. The earlier 14-check receipt is retained as historical
evidence. It always refuses native
execution before model access, including forged approval flags. It has no model
loader, experiment driver, evaluation, target policy, retry defaults or grant.
The existing B2 checkpoint/progress adapter must not be presented as a qualified
CE8 adapter. Full integration and native qualification still require the new
operator decision and a separately registered scope.

The current core additionally binds the exact copied prompt/target token IDs and
the static recipe digest in every CE callback. Equal token lengths cannot admit
a substituted teacher, and a different valid optimizer recipe cannot use the
same journal scope. The live boundary/fence fields are excluded from the static
recipe digest. Seven pure and ten actual core controls are included in the
current combined durable-accounting receipt below.

The separate `ce_checkpoint.py` adapter now verifies eight globally unique row
identities per boundary, every completed boundary updated, exact static CE
recipe and prepared-token digest, and CPU parameters/buffers with uninitialized
CUDA. The base storage adapter retains its B2 rules. The combined CE/core/legacy
checkpoint suite passed 43 checks, including 19 actual CPU cases; see the
superseding `evidence/CE8_CHECKPOINT_CPU_CHECKS_V2_20261001.json` receipt.
Review strengthened the fixture's nested restore-progress and ancestor checks;
the original 42-check receipt is preserved.

The new stochastic CE8 integration case uses fresh model, optimizer and adapter
objects in one OS process. Parameters, buffers, Adam moments/steps, Python/NumPy/
CPU RNG and module modes match the uninterrupted second update bit for bit.
The caller installs the returned fence and recomputes actual policy identity;
old CE results and reused restore proofs are rejected by the synthetic fixture.
The corruption control rejects the latest payload without mutation or fallback.
This is mechanics evidence. The in-memory test observer supplies no durable
production CE ledger, fresh-OS-process qualification, native resource census,
experiment driver, acquisition result or training grant.

`ce_plan.py` verified both existing 3,000-row target files and the bound atomic R
manifest into 375 identical eight-row exposure boundaries. Label bindings remain
distinct between canonical and permuted teachers. The independent root export
also pins the completed warm-anchor receipt; recipe, native runtime and full
trainer source remain unresolved. Nine plan controls and three permutation-math
controls passed without framework imports. See
`evidence/CE8_PLAN_PREPARATION_20261001.json` and
`evidence/CE8_PLAN_SOURCE_CHECKS_20261001.json`.

Proposal plans always deny execution and cannot export fixture checkpoint scope.
The preparation references full canonical/policy TRAIN hashes; future native
ingestion must revalidate those files. A target-plan binding and tokenizer census
do not constitute a native recipe, measured resource envelope or learning result.

## Synthetic durable CE accounting

The verified CPU route in `ce_journal.py` binds an explicit
synthetic fixture plan, prepared token/target rows, static recipe, source/runtime
references, anchors and finite declared limits. Proposal/native scope is denied
before path access. Hash-chained, fsynced rows preserve admitted operation,
selected/full-sequence token, hash, IO, metadata and wall reservations across
reopening and rollback. No failed or pending reservation is refunded.

Logical C=U advances only after the complete observed CE result and successful
cleanup. Any incomplete CE attempt requires verified ancestor restoration with
a fresh fence before further CE. Reused or stale restore/update proofs are
rejected. The caller recomputes actual restored policy identity. Unknown
interrupted wall prevents further work and a consistent completion seal.

The combined suite passed **74/74 controls**, including **25 actual CPU cases**,
with no skips/errors, no framework imports during collection and uninitialized
CUDA. Real partial Adam mutation was restored to C0; two-boundary stochastic
resume matched uninterrupted parameters, moments, module modes and Python/NumPy/
CPU RNG bit for bit. Content/recipe/forward/Adam/metadata controls reject work at
their admitted boundaries. See `evidence/CE8_DURABLE_CPU_CHECKS_V2_20261001.json`.

The original 74-case run and its four test failures are retained. Its fixes
changed the purity assertion for mixed test execution and normalized newly
attached dropout modules to eval before the initial snapshot; production source
was unchanged. The earlier 44-case content-binding receipt is also retained.

Profiles are synthetic upper declarations. Recipe hashing precedes the first
kernel callback; constructor validation and bounded replay are setup scope.
A native controller still needs outer admission and measured pricing for that
work, a verified boot identity, actual process/resource guards, full data
revalidation and a separately registered experiment. Reopening uses fresh
objects in one OS process. These controls do not qualify native/full-unit costs,
a fresh-process trainer, acquisition, composition, retention or a training grant.

## Dormant native preparation

The shared controller distinguishes a separately authorized qualification scope
from production training. Qualification has no dependency on an already passed
qualification receipt. Training requires its own scope and qualified evidence.
No native backend or outer authority verifier is registered in this source.
Unapproved requests and invented native factories/permits refuse before work.

The orchestration code admits bootstrap/setup work before its callbacks, binds
the concrete stage/refmap outer adapter, orders source verification, fresh-worker
boot proof, lease, resource guards and watchdog, then checkpoint/update/restore
and publication. Cleanup, watchdog exit and lease release precede the terminal
seal. Failed or suppressed work cannot publish a successful completion.
Injected service fixtures exercise that wiring; they are not process/resource
measurements or an actual native controller implementation.

The new three-module source suite passed **30/30 framework-free tests**, including
cross-module request identity/null qualification-output handling, L2 gradient
criteria, complete summed token census and label binding, source/recipe admission,
failure preservation and lifecycle ordering. See
`evidence/CE8_NATIVE_PREPARATION_SOURCE_CHECKS_20261001.json`.

The actual existing canonical and permuted R teachers were independently
reverified against all source TRAIN/policy data, atomic identities, declared
oracle policy, strict verifier and corpus reward: **6,000 teacher rows checked**.
That run used explicit non-native metadata admission, imported no framework,
loaded no tokenizer/model and read no held-out model outputs. See
`evidence/CE8_REAL_TRAIN_INGESTION_20261001.json`.

Actual maximum prompt, target-plus-EOS and total sequence lengths are selected
independently from rendered row bindings: the prepared ceilings are 1,259, 29
and 1,287 respectively. They need not share a row and must not be combined into
an invented longest example. That source-only verification performed neither
tokenizer rendering nor native qualification.

Subsequent tokenizer-only runs re-rendered all 6,000 targets with the pinned
local tokenizer, without loading a model or initializing CUDA. The actual max
prompt and max total share one row; 89 rows per arm attain max target-plus-EOS.
V2 retains exact prepared-hash input costs: **20,311,962 bytes per arm**.
Both witness receipts are preserved under `evidence/`; they establish row
identities and lengths, and provide no native resource or learning measurement.

## Delayed native implementation

`ce_native_authority.py` and `ce_native_backend.py` now contain source for the
common warm FP32 load, fresh CUDA Adam, CE8 update, native checkpoints, durable
journal linkage, restored-policy fingerprints, donor extraction and failure
cleanup. Native permits bind the recipe, source, runtime, anchors, phase, arm
and concrete observer/resource callbacks. All issuer and dispatch registries
remain empty. Dictionary flags and counterfeit permits refuse before model/IO
work. Source presence does not establish that these operations fit or work on
the GPU.

Native journals require the exact admitted ledger artifact, compact prepared
row bindings and hash-byte costs. Metadata roots must equal the admitted output
directory, so checkpoint metadata and the ledger share the same owned scope.
Restore-required native workers may load the fresh model needed for restoration;
CE updates, checkpoint saves and donor publication still require the installed
restore proof. Prepaid lifetime cleanup can finish after progress or restore
fences change; its original invocation and reservation remain binding.

The controller links the journal before model loading and reads an explicit
checkpoint boundary list. The proposed list is C0, 1, 75, 150, 225, 300, 375.
Proposed budgets and the exact evaluation populations are in
`../skill_acquisition_method_lab_v1/CE8_REGISTRATION_PROPOSAL_20261001.md` and its
JSON. Both operator choices remain unselected and all execution grants false.

The combined regression passed **147/147 checks**, including **33 actual CPU
cases**, with zero skips/errors, unchanged Python source and uninitialized CUDA.
It covers the existing CE8 update/checkpoint/journal paths, legacy B2/donor
behavior and the new refusal/lifecycle rules. The original test receipt and a
classification correction are preserved as
`evidence/CE8_NATIVE_HOOKS_CPU_CHECKS_20261001.json` and
`evidence/CE8_NATIVE_HOOKS_CPU_CLASSIFICATION_20261001.json`: the checker initially
classified seven `NativeCPUTests` as pure tests; execution and pass counts were
unaffected. No optional rerun was needed for that reporting correction.

Pinned raw asset metadata now supplies conservative lower bounds for unavoidable
loader reads and hashes. Zero or undersized profiles fail before asset/model
work. These bounds omit additional work and cannot supply measured upper prices.
When admission expires after prepaid allocation cleanup, held references are
still released and the admission failure remains the primary error.

## Qualification executor and raw readback

`ce_native_qualification.py` contains delayed stage bodies for the common warm
runtime, all 90 maximum-length witnesses, cached versus bulk likelihood,
independent equal-row gradients, declared Adam/clipping batches and saved delta
roundtrip. The independent gradient path disables decoder/head checkpointing;
the candidate uses the registered checkpointed path. Both replay the same RNG
and module modes on the separately declared eight rows. Raw F32 vectors, token
arrays, modes, RNG and operation-order records support later recomputation.
CUDA spans synchronize after the backend's sole runtime configuration.

The controller passes verified worker identity, observer and outer admission to
`begin_qualification` before model loading. Preparation must leave CUDA
uninitialized. Live permit/service objects are excluded from JSON evidence
copies. `PreparedRows.iter_ids` revalidates separately authorized qualification
rows, retains at most eight token rows and preserves training boundary order.

`ce_qualification_artifacts.py` independently recomputes masks, EOS positions,
likelihood errors and streamed F32 gradient L2/coordinate errors. It requires
all 90 witnesses for masks and likelihood, and eight exact rows for gradients.
Its explicit analytical budgets cover raw IO/hash, parsing, arithmetic and
additional hashes. Results establish numeric/artifact consistency and explicitly
leave actual execution verification false.

`ce_qualification_accounting.py` maintains a separate hash-chained qualification
ledger. Explicit qualification Adam batches have their own C/U history. Failed
and pending reservations survive replay; verified CE causal proofs govern
commits. Stage, worker, source, recipe and artifact bindings are retained, with
owned metadata, clipping-order readback and lifetime cleanup accounting.

The combined source suite passed **105/105 controls** without ML framework
imports, models, optimizer execution, skips or errors. Source remained unchanged.
See `evidence/CE8_QUALIFICATION_SOURCE_CHECKS_20261001.json`. These include real
temporary-file readback and durable metadata fixtures; they are not native
numerical or process qualification.

All factory/issuer/adapter/verifier registries remain empty. Native accounting
sealing is explicitly **UNSUPPORTED_UNMEASURED**: durable outer setup admission,
complete constructor/replay/interoperation/terminal wall coverage, platform
directory publication and actual execution verification are still required.
Fresh OS checkpoint workers and resource measurement adapters are unimplemented.
The unapproved registration proposal requires a new source freeze and concrete
qualification accounting/execution artifacts before any native dispatch.

## OS evidence and outer accounting

`ce_bootstrap_accounting.py` retains opaque, source-bound setup prepayments
before dependent constructor work. Duplicate or invalid ticket transitions
refuse before append. The qualification ledger records actual monotonic time
through setup, replay, API barriers, idle intervals and explicit end. Actual
elapsed time and upper reservations are both charged; the physical deadline
remains separate. Interrupted unknown time prevents further work.

`ce_worker_evidence.py` reads actual Windows PID, creation time, boot GUID,
executable image and command line through retained process handles. Live proofs
recheck the handle; termination requires a signaled handle and OS exit code.
The virtual environment launcher and base process image are pinned separately.
Current-process metadata probes explicitly leave native freshness, execution
verification and qualification false.

`ce_resource_evidence.py` samples current-process working set, host RAM, disk
and a pinned NVIDIA query utility. Driver-reserved memory is included in the
six-column GPU record. Barriers use exact CUDA free bytes when available and
a conservative lower bound for rounded CLI values otherwise. Launch disk must
cover both its fixed floor and the declared active reservation. Sampled minima
are labeled as observations at barriers, with no continuous minimum claim.

`ce_execution_verifier.py` derives cost totals, pending and failed reservations,
stage order, wall state and seal consistency from original retained journal
rows. It uses the raw numeric reducers and rejects supplied worker flags.
Results remain **UNQUALIFIED** until actual source-issued worker handles,
bootstrap/clock verification and remaining execution services are verified.

The combined service suite passed **144/144 controls** with unchanged source,
zero skips/errors and no ML framework imports. A real read-only resource probe
and a current-process Windows probe also completed without models. Worker probe
V1/V2 failed on the virtual environment redirector's image/argv assumptions;
their failure records are retained beside the corrected V3 evidence. Resource
probe V2's exact source snapshot is retained as text for provenance.
See `evidence/CE8_OS_SERVICES_SOURCE_CHECKS_20261001.json` and the probe receipts.

All execution/observation/bootstrap verifier registries remain closed. Native
launch/RPC, fresh-process checkpoint comparisons, platform publication
durability and final service/producer integration remain required. The warm
study's stop and pending operator choice still govern any new native treatment.

## Launch, checkpoint and publication adapters

`ce_process_launch.py` implements hidden Windows subprocess launch and bounded
JSON RPC. Launch authority and post-launch observation authority are separate;
both registries remain empty. The fixture route admits only the fixed
`ce_process_fixture` module after explicit resource clearance. Parent identity,
launcher and process image, full argv/environment, source and request hashes,
message sizes, sequence, worker PID and deadlines are bound.

Windows virtual environment launchers may create a separate base-Python child.
The declared redirector topology retains both handles, verifies OS parent and
creation-time lineage, and checks the actual worker's image/argv before RPC.
Terminal evidence uses the same retained handle's PID, creation time, signaled
state and exit code, tied to the image observed while live. Owned cleanup is
prepaid and still releases handles after receipt failures.

`ce_checkpoint_qualification.py` supplies storage for exactly two qualification
CE8 batches and three independent process branches: uninterrupted U2, U1
prefix, and U1 restore followed by U2. Raw parameter/Adam/RNG payload bytes and
canonical module/Python/NumPy metadata are compared. Strict fence ancestry and
bounded manifests are required. Source-registered lease handoff and checkpoint
worker services are still missing; their empty registries refuse before parent
release. The parent reload path verifies its existing runtime and avoids a
second CUDA configuration.

`ce_publication.py` publishes create-only receipts using same-directory Windows
`MoveFileExW(WRITE_THROUGH)`, file fsync and exact readback. Artifact verification
streams large payloads; JSON materialization is bounded. Original replayed costs
and the full publication prepayment are both retained. Receipts remain
`UNQUALIFIED_COMPLETION`, with no execution or learning claim.

The combined source suite passed **175/175 controls**, with unchanged source,
no skips/errors and no ML framework imports. A real hidden JSON-only child
fixture passed handshake, ping, finish, terminal exit0 and cleanup; an actual
Windows small-file publication fixture passed create-only/readback/overwrite
refusal. Two earlier child fixture failures and a permit-order test failure are
retained beside their corrected evidence. See
`evidence/CE8_LAUNCH_CHECKPOINT_SOURCE_CHECKS_20261001.json` and the fixture receipts.

Factory selection for child command construction now uses a JSON module/symbol/
source-hash descriptor. A descriptor cannot register a callable or issue scope.
Actual checkpoint child handlers, GPU lease handoff, final resource/issuer
services and native producer activation remain incomplete. Native models,
discarded qualification updates and training still require the selected
experiment and frozen authorization scope.

The qualification producer remains **DRAFT_UNQUALIFIED**. Its native verifier
registry is empty; flipping an activation flag or providing stage booleans,
unrelated blob hashes or a caller-supplied verifier cannot produce native PASS.
Actual qualification stage execution, execution handles, observation/artifact
linkage, retained-ledger replay, verification pricing and capability activation remain required
after the new experiment decision and registration. Acquisition, composition
and functional retention remain unproven.

## Checkpoint child and controller services

`ce_checkpoint_worker.py` implements the three declared qualification child
branches, exact two-batch accounting, prefix restore, corrupt/stale restore
attempts and full-state readback. A resumed child inherits logical U1 through
an immutable admission bound to the original manifest, raw parent ledger,
source, worker and anchors. It must restore and establish its own fresh fence
before CE continues. Parent costs remain in the parent ledger; failed attempts
remain charged. Cleanup precedes lease return and publication.

`ce_native_services.py` implements stable controller dispatch, separate setup
and native cost ownership, durable bootstrap accounting, resource barriers and
service lifecycle checks. Worker context is finalized before permit issuance.
Preparation leaves CUDA uninitialized for the backend's sole configuration.
Missing source-owned roles refuse before dependent filesystem or model work.

The combined child/service source suite passed **196/196 controls** with exact
source hashes, no skips/errors, no ML framework imports, models or optimizer
steps. See `evidence/CE8_CHILD_SERVICES_SOURCE_CHECKS_20261001.json`. No model or
checkpoint child ran for this source milestone.

The revised unapproved registration budget retains eight restore attempts,
six save allowances and twelve full-state fingerprints. Its analytical totals
fit the proposed checkpoint IO caps; these are reservations, not measured
prices or proof of resource fit. The earlier proposal receipt remains evidence
for its original revision. The new arithmetic receipt is retained in the
method lab's `evidence/CE8_REGISTRATION_CP_COUNT_AMENDMENT_20261001.json`.

Concrete child initialization, process supervision, GPU lease transfer,
publication, issuer and final raw branch reductions still require integration.
Native registries remain empty and the producer remains **DRAFT_UNQUALIFIED**.
The warm stop, pending experiment choice and frozen scope govern native work.
Acquisition, retention and independent composition remain unproved.

## Further review and raw state reduction

Further review separated immutable child request bytes from observed worker
identity and moved branch declarations into the complete derived execution
context. The shared frozen scope is preserved. Worker identity now hashes the
same OS and raw source facts in supervisor and child views; descriptive launcher
provenance remains evidence without changing that identity.

Bootstrap helpers enforce declared and reported work bounds. The stable
dispatcher retains each operation's original cost owner, and failed worker
admission closes its local handle. Child lease contexts cannot suppress a
failed branch. Corruption and stale controls require specific validation
causes and retained operations; infrastructure failures propagate.

`ce_checkpoint_reduction.py` replays separate child ledgers, derives physical
updates separately from inherited logical progress, recomputes returned state
descriptors and compares saved parameter/Adam/RNG bytes. At that source revision,
the complete three-branch envelope-to-ledger-to-endpoint path had not been
exercised end to end. Focused controls did not establish native execution.

`ce_resource_qualification.py` derives resources from retained barrier samples
and selected returned operation counters. Exact CUDA totals govern allocator
fractions; CLI totals remain separate observations. IO observation scope is
retained, including unmeasured external utility OS IO. Reservation ceilings
cannot substitute for actual counters or complete-unit native prices.

The expanded source suite passed **226/226 controls**, with unchanged source,
zero skips/errors and no ML framework imports, models or optimizer steps. See
`evidence/CE8_INTEGRATION_SOURCE_CHECKS_20261001.json`. Native initializer,
process/lease/publication wiring, child
accounting derivation, full pricing and final execution verification remain
required. Registries stay closed, qualification stays **UNQUALIFIED**, and the
broader study remains active with no new native treatment after the warm stop.

## Child initialization, delegation and publication

The complete three-branch raw reduction now has a genuine end-to-end fixture:
independent durable ledgers, inherited U1 plus a returned restore/fresh fence,
four state descriptors and actual small parameter/Adam/RNG files. Rehashed
wrong history, refunded costs, foreign parent and changed endpoint controls
fail. Test coverage is retained as evidence rather than an output authority flag.

`ce_checkpoint_initializer.py` replays original parent header/journal/full
partial snapshot evidence and verifies the original four-file prefix store.
It streams create-only imports and isolated corruption copies, binds actual
returned restores, and finishes the child stage with retained artifacts.
Child qualification and bootstrap directories derive from the common frozen
owner policy; their caps, prices, census and deadlines remain exact.

`ce_checkpoint_lease.py` retains one supervisor lock across serial branches.
Parent GPU use is suspended during delegation. Original current owner markers,
actual retained worker handles and terminal evidence gate transfers; unknown
owners cannot be removed. Source service integration and the lease helper's
complete external IO prices remain unmeasured.

`ce_checkpoint_publication.py` retains saved manifests and control evidence,
then publishes the ended child ledger and branch envelope under independent
outer prepayment after model/lease cleanup. Saved progress is an ancestor of
the installed recovery fence. The exact original accounting owner is pinned.
Publication explicitly leaves OS termination verification to the supervisor.
The child lifecycle now exits delegation while its accounting stage is still
open, then finishes the stage and ends accounting.

The expanded suite passed **263/263 controls** with unchanged source, no skips
or errors, and no ML framework imports, model loads or optimizer execution.
See `evidence/CE8_CHILD_LIFECYCLE_SOURCE_CHECKS_20261002.json`. This is pure
metadata/small-file evidence, not a native CUDA checkpoint qualification.

Native local issuance, process orchestration, full source/worker/bootstrap/clock
verification and complete native unit prices still require integration.
Registry activation and actual discarded native qualification require the
pending selected experiment and frozen authorized scope. Acquisition, functional
retention and independent skill composition remain unproved; the broader goal
is active and incomplete.

## Process and controller integration

`ce_checkpoint_processes.py` supplies parent checkpoint capture, immutable
branch requests, owned launch, exact command dispatch, same-handle terminal
verification, parent restore and a parent-bound raw branch bundle. Growing
parent journals are copied into retained create-only prefixes. Successful
child saves and recovery fences are tied to their original operation rows.
Both child-to-child and parent-to-child saved states must compare bitwise.

`ce_checkpoint_local_issuer.py` implements bounded ready/run/artifact/finish
RPC and paid request/common-scope/delegation derivation. A derived context is
frozen after actual worker/delegation observation and before execution permit
issuance. The native source entry provider is absent, and existing issuer and
verifier registries remain empty. The real native entry path is unexecuted.

Qualification controller services now acquire one lazy supervisor lease before
preparation and bind the exact backend before loading. Supervisor lease and
parent worker refresh costs remain in their original bootstrap owner between
Q stages; child lease costs remain in active child accounting. Reload requires
verified parent use. Imported but uninitialized Torch cannot start CUDA during
cleanup after preparation failure.

The final readback now independently derives Adam/clipping order and resource
observations from original journal prefixes. It replays both retained parent
checkpoint prefixes and all three raw child branches, ties parent/child saved
manifests to publication/recovery, and compares the saved state bytes. Resource
counter scopes remain explicit; complete native prices and OS execution are
separate unverified requirements.

The expanded source suite passed **299/299 controls**, with unchanged source,
zero skips/errors and no framework imports, model loads or optimizer execution.
See `evidence/CE8_PROCESS_INTEGRATION_SOURCE_CHECKS_20261002.json`.

Actual source entry/bootstrap/launch issuers, constructor origin verification,
full OS/source/clock/prepayment verification and measured native qualification
remain required. No native work has run after the warm stop. The pending new
experiment choice and frozen scope govern execution; the broader acquisition,
retention, composition and method-combination objective remains incomplete.

## Selected canonical study and entry verification

`../skill_acquisition_method_lab_v1/evidence/SELECTED_R_ORACLE_SFT_20261002.json`
records an assistant selection of canonical-only R oracle supervision under the
original user's explicit permission to add experiments. It records no human
answer to the optional choice question. The original warm study remains stopped.
This selection grants no qualification, training, evaluation or dispatch scope;
those still require exact registration and verification.

`ce_checkpoint_entry_provider.py` supplies the pre-model source/OS metadata
pipeline. A fixed metadata payment uses the existing durable service bootstrap
before file or process observation, without requiring an early model execution
permit. Native model payments remain after observed delegation/context freeze.
Handle ownership survives post-open accounting failures; redirector and entry
cleanup preserve primary errors. Real prepared-row and retained terminal-prefix
factories, source configuration and actual issuer registrations remain missing.

`ce_bootstrap_verifier.py` independently replays original raw header/journal/
seal, prepayment owners, full failed/pending reservations, transition history,
offsets and recorded monotonic intervals. A historical STARTED artifact ticket
can be linked to its later RETURNED raw transition. This is not body execution
proof. Missing terminal/end coverage and constructor clock gaps stay explicit.

`ce_delta_artifacts.py` streams the exact saved FP32 inventory, checks finite
values, file/data/state digests and norm with bounded memory. Final readback
links runtime loader identities and operation rows, and saved donor reports to
those bytes. Independent full common-anchor/master rebuild and native execution
remain unverified; reported roundtrip booleans cannot certify that missing work.

The V2 combined source suite passed **327/327 controls**, with unchanged source,
zero skips/errors and no framework imports, model loads or optimizer execution.
See `evidence/CE8_ENTRY_BOOTSTRAP_SOURCE_CHECKS_V2_20261002.json`. The earlier
322-check receipt is retained for its earlier exact source revision.

The canonical experiment is selected, while source freeze, registered scopes,
remaining concrete factories and actual native qualification precede dispatch.
Acquisition, retention and independent composition remain unmeasured.

## Production interface

```powershell
python -m experiments.skill_acquisition_native_v1.runner --packet APPROVED_PACKET.json
python -m experiments.skill_acquisition_native_v1.runner --packet APPROVED_PACKET.json --resume --resume-reason 'documented interruption'
```

The approved packet must bind the clean source checkout, complete warm
competence/pilot/enrollment prerequisites, native qualification, numeric cost
limits, measured IO and RAM transients, and an owned output directory. The runner
rechecks host resources after waiting for the lease and before model allocation.
It preserves interrupted attempts and refuses already sealed completion.

`RESULT.json` is complete only with its returned publication operation and the
matching `DONOR_COMPLETE` invocation end in `PHYSICAL_COSTS.jsonl`. Unsealed
results are preserved as partial evidence during a documented resume.

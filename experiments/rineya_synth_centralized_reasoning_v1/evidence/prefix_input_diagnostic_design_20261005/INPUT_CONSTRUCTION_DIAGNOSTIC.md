# One exposed-prefix input-construction diagnostic: source design only

This is a new proposed purpose, `ONE_EXPOSED_PREFIX_CUDA_INT64_INPUT_MASK_ONLY`.
No native worker/contract/supervisor/auditor or launch authority is implemented.
The prefix is exposed failed DEV history, not prompt-only evaluation. It must
never enter M0 population, scoring, training, candidate selection or confirmation.
Success tests only this reporting branch/device health at its actual time. It
does not reproduce prior253 forwards, identify the originating CUDA kernel/TDR
cause, qualify a full M0 batch or establish model readiness.

## Exact input and tensor permission

Root's `ROOT_FAILURE_PREFIX_INPUT_001.json` has file SHA
`195f1b3a658959ccbe517ce5aefaef1d0d2c298db0095ace08f744bd6488feb5`,
body `a22c5b27225fc75ae697a5f06a96da18c1cb995a7bf7c6d03eb05f0cb6f06ec9`
and prefix IDs digest
`e806d781c2c21c15180b6ec1bc714ed5acc7104948c2859ec6b411761a4a19d1`.
It is exactly414 original row20 prompt IDs plus eight already retained generated
IDs, giving one422-token prefix. Retain exact parent config
`b8e2efe1e832ca865bc8a5de8f48b9d601246cd14335a2e6e09ecc4a1362ba00`
and failed row file
`9f4cb0ebca12d4123dd30631ec27314f3344115a68f3e12c17e794016bcea5f1`
joins, without decoding or grading. Every ID is a nonboolean integer0..151935;
attention mask is exactly422 integer ones. No resampling/alternative prefix.

Permission is exactly one CUDA0 `torch.long` IDs tensor of shape(1,422) and one
same-shape/dtype/device `torch.ones_like(ids)` mask. Each has422*8=3376 logical
storage bytes; total6752 logical CUDA tensor bytes. The two existing `tolist`
checks transfer422 integers each (6752 logical value bytes combined). Python
list/allocator/context/driver overhead is measured by unchanged resource guards;
logical bytes do not establish actual RAM/VRAM fit. Zero model loading,
construction, asset reads, forwards, loss, backward, update, decoder or scoring.

## Source-compatible operations and explicit new scope

Reuse the exact existing `inference_qual_worker.make_guard`, physical RAM probe,
owned-Job worker/monitor/lease mechanisms and nonrecursive context Recorder/
GuardTorchProxy/observed_guard where those interfaces fit. D001's contract has
`no_tensor_allocation=true`; it cannot authorize these tensors. Its aggregate
`context_phases`/`lazy_native_api` must not be copied wholesale. Omit model-class
symbol imports, transformer execution and optional Windows PID counters. This is
a separately registered narrower scope, not D001 qualification.

Lazy startup happens only after current owned resource admission. The proposed
success operations are explicitly enumerated in `PROPOSED_DOSE.json`:
initial source/config preflight, ownership verification, disk admission, one
Torch import/version observation, unchanged full post-import growth reserve,
one CUDA context initialization, one direct existing4GiB preload memory check,
exact existing precision/thread configuration and seed101 calls, peak reset,
the twelve tensor/validation operations below, two reference-release operations,
final source/config preflight, four original optional terminal CUDA queries
(is_initialized, max_memory_allocated, max_memory_reserved, mem_get_info), and
three durable worker publications. Existing
configuration/seed behavior is reused, not replaced by new settings. Terminal
query result/error fields are retained in the single final worker_result; no
additional GPU-final file or undeclared publication is implied. No extra current
allocator queries or direct memory queries beyond this explicit list are added.

The input branch preserves the order of `inference_component.py`149–154:

1. `torch.tensor([list(prefix)],dtype=torch.long,device='cuda:0')`, once.
2. `torch.ones_like(ids)`, once.
3. IDs `torch.is_tensor`, shape==(1,422), dtype==torch.long, device=='cuda:0',
   and `ids.tolist()==[prefix]`, each original check exactly once.
4. Mask `torch.is_tensor`, shape==(1,422), dtype==torch.long, device=='cuda:0',
   and `mask.tolist()==[[1]*422]`, each original check exactly once.

Record each actual callable/predicate return/error before moving to the next
existing operation. Preserve the exact source short-circuit order: after an
is_tensor/shape/dtype/device/value predicate fails, do not execute any later
predicate, host transfer or mask check. No failed allocation continues to another
allocation/check. Property summaries use only scalar/tuple/value hashes, never
tensor repr or frame locals.
`tolist` preserves the existing host verification/implicit device interaction.
No added synchronization, `CUDA_LAUNCH_BLOCKING`, empty_cache/device reset,
driver/TDR policy or numerical/resource relaxation. Exact-source-byte CPU
controls must demonstrate the same operator/check order and exceptions.

Eight declared worker guard sites are pre_import (no Torch), after_import,
after_growth, before_context_init, after_context_init, before_input_construction,
after_input_validation and complete. They reuse the original guard implementation
and single existing init/memory-query branching. At most seven initialization
queries and seven corresponding memory queries arise; only pre_import has none.
No per-predicate resource guards are inserted. Additional future guard sites
require a new explicit source/dose budget instead of disappearing from costs.

## Dose, publication and failure requirements

The success path has one IDs allocation, one mask allocation, ten original
validation predicates/host transfers, no model positions and no generated tokens.
It retains original2GiB owned tree RSS cap,1.25GiB available reserve/full import
growth rule,3.3125GiB preworker admission,4GiB preload/admission GPU free,
0.5GiB work GPU free,4GiB disk,600-second whole scope,100ms watchdog,
200ms CPU freshness and all original GPU cadence/freshness/deadline gates.
All startup/configuration/seed/query/predicate/host-transfer/publication/reference
release/exception-frame work must be in the declared operation accounting.

Three worker durable publications are ownership_verified, input_check_summary
(partial failure or complete result, never a generated response) and worker_result.
The phase ledger separately counts create, every marker write/fsync, close and
final summary write. Preserve stdout/stderr and the independent supervisor/
resource/heartbeat/ownership/lease terminal records unchanged. The proposed
35 success operations (31 original design entries plus four terminal queries),
at most one bounded failure-frame capture, eight declared guards and at most14
nested guard queries derive58 maximum phases/116 markers; reserve a hard
128-marker scope budget with existing64KiB/marker and64MiB ledger writer limits.
Worst128*64KiB=8MiB fits that ceiling. Final original primary error plus bounded
exception frames remains a bounded16MiB worker JSON document; no new file or
per-step payload can be silently omitted. This is a source bound, not measured
native fsync speed or runtime fit. Future source/budget reconstruction must check
these exact counts before manifest admission.

On a caught original exception, the accepted bounded helper253caa records only
original Python reporting frames from supplied source/runtime pins. Primary
error type/message stays unchanged. Helper failure adds a separate fixed bounded
diagnostic failure status and cannot erase/replace the primary error. Do not
retain tensors, locals/args/repr, contexts/causes or traceback frame references.
Reference release is counted even on failure, but no CUDA cleanup/reset query is
used to manufacture success. Cleanup/terminal publication failures remain separate.

## Required independent terminal audit before any success claim

The auditor must be a new source-reviewed diagnostic auditor, not the existing
M0 raw or D001 success auditor with a renamed status. Require external config/
profile/input/inventory/auditor/preservation/body pins and actual clean launch
HEAD; complete current executing source/runtime/helper rosters, no count-only
shortcuts or inherited manifest authority. Retain the qualified runtime core and
declare any actually executed added stdlib/helper closure before launch.

Audit all exact original/preserved bytes and source snapshots before/after;
full nested phase order/counts/returns/errors, registered names and scope marker/
IO budgets; actual two tensor constructions and all ten checks with exact422
IDs/mask hashes, shape/dtype/device and6752 logical byte accounting; zero model/
forward/backward/optimizer/decoder/grading work; and every current resource amount,
cadence/freshness/deadline observation. Validate exact owned PID/FILETIME/Job/
reservation/assigned-before-resume/monitor-before-resume identity, sampler/NVML/
Job cleanup, frozen supervisor inventory, and separately anchored current death
observations for every observed owned process birth. Source/helper errors,
unregistered/truncated reporting frames and primary/cleanup/publication errors
must be truthfully retained. A partial/failure is never diagnostic success.

Any positive status must mean only
`ONE_PREFIX_INPUT_MASK_CHECKS_COMPLETE_DEVICE_HEALTH_UNQUALIFIED` for that one
scope. Explicitly keep model_fit, M0_membership, scoring, training, transfer,
retention, confirmation and causal attribution false. Independent source/schema/
CPU review, a new frozen owned manifest and fresh host admission remain required;
no implementation or launch follows this design.

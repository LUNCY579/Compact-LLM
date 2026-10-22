# CPU load source candidate001: OPS review

Date: 2026-10-04. Reviewer `/root/exception_ops`; builder
`/root/desktop_profile_builder`. **SOURCE_APPROVAL_WITHHELD_ROUND1.**
Two material OPS findings below require a separately registered finite source
round2 before any CPU model attempt. Source candidate1 remains frozen; default
three-round limits and all old quarantines remain. This report grants no load,
fit, native qualification or additional probe002 attempt.

This report is finalized only from completed retained inspection/falsifier
outputs before interruption. No unfinished attempt was resumed or new check
performed. All tool executions had completed with exit0; no running/unknown PTY
or exec handle remains from this review. Only this report is written now.
I authored the prior OPS scope advisory, not this source/tests. Shared provenance
is disclosed. There was no actual framework/model import, checkpoint/asset load,
live process/GPU observation, canonical database action or source edit.

## Verified exact subject and retained fixture evidence

FREEZE SHA256:
`386f635dc161cc70d523218d942002461b5cef3c098003884c3f44ef23931514`.
Its complete source/preserved/evidence entries were independently hashed and
matched before the completed falsifier group. Exact candidate pins:

| File | SHA256 |
| --- | --- |
| cpu_native_load.py | `f1a05707751e3e56705b9105357c06f753e94372717dbea086c2cad30e715ec4` |
| tests/test_cpu_native_load.py | `fa37ae75fd563efa12af3eb320173cd51d410482f0c9df11e4fbd9c3d52ff40f` |
| CPU_NATIVE_LOAD_FEASIBILITY_V1.json | `8160ba80aa52a547a96aa605ad31a0a88a3bc605da779de3aef09ca4025c97d8` |
| delegated scope | `e66d181f01a2346617b5433623859e71b518eefe45c896e117d097c2ddeeeb86` |

Retained builder evidence records one no-model fixture invocation,5/5 groups,
2.169s suite/2.562s command. Raw stdlib fixture demonstrates suspended owned Job
flags8704/0x2200,2GiB commit cap, initial RSS2,101,248 bytes, independent deadline
and empty-tree exit. Raw stdoutc71bcd93.../stderra43bf085.../test resultabf0d1cf...
match FREEZE. Those fixtures neither test required Torch import nor the failure
paths below. Original native84531289/contractd79f3b6a, operationsba27ec8a,
probe845d1e66 and reader0299cb37/executionc66a1b64 remain preserved.

## CPU1-OPS-001 — P1: required CUDA definition import is treated as device execution

`forbidden_call` rejects every module beginning torch.cuda for every profiled call,
including name `<module>`. The boundary is armed before import torch. The pinned
torch.__init__ source imports cuda at2255; loading that definition module produces
the prohibited module-body event. Thus the required CPU import path is killed
before any CUDA availability/init/device action or model load. This contradicts
the registered distinction between imported CUDA definitions and initialized
device activity. Source inspection and the no-model decision control establish
the false positive without importing Torch.

Inspected installed Torch main/cuda sources hash51f02753.../674c693e....
`_is_compiled` only tests presence of a compiled symbol. `_lazy_call` queues
callbacks while uninitialized but executes the callback when initialized; it
cannot be broadly permitted as harmless. A future correction must statically
map the exact pinned definition/metadata-only import paths, permit only justified
import-phase behavior/state, and retain fatal availability/init/device/RNG/sync/
tensor-CUDA actions. Unknown callbacks/helpers remain unavailable; do not broadly
allow CUDA calls or spend the one model attempt finding predictable false positives.

## CPU1-OPS-002 — P1: early owned-launcher cleanup and unknown outcome reporting are incomplete

When Popen returns a known suspended launcher but identity/Job setup fails before
`job` is assigned, finalization only waits5s. It has no kill/wait fallback for that
known caller-owned process. It then cancels the deadline; an uncontained suspended
launcher can remain while the ledger correctly holds OUTCOME_UNKNOWN/full120s.
Cleanup must target the exact known Popen process/handle and preserve conservative
unknown accounting whenever complete exit/descendant evidence remains unavailable.
No unrelated process termination or fabricated zero-cost completion is permitted.

The same completed group also demonstrates a related outward-state inconsistency:
after a synthetic completed child, exit-proof failure makes the ledger unknown,
yet `supervise` returns SUCCEEDED because its outward outcome is not updated.
A partial loaded component may remain observed, but overall success cannot
override unknown containment/accounting. Source round2 must report authoritative
unknown status and deny whole-path qualification on that branch.

## Completed one-group falsifier evidence

The group used actual CPU decision/supervisor code with mocked processes, Job,
telemetry and completion, plus real **temporary** OperationsStore fixtures. It
performed no real dispatch, GPU/host query, asset access or canonical DB write.
Its exact returned output was:

```json
{"one_no_model_falsifier_group":[{"case":"definition_module_decision","torch_cuda_module_body_refused":true,"actual_framework_imports":0},{"case":"job_assignment_failure","returned_outcome":"FAILED","scheduler_state":"OUTCOME_UNKNOWN","held_seconds":120.0,"known_launcher_kill_calls":0,"actual_launches":0},{"case":"late_exit_proof_failure","returned_outcome":"SUCCEEDED","scheduler_state":"OUTCOME_UNKNOWN","held_seconds":120.0,"known_launcher_kill_calls":0,"actual_launches":0}],"seconds":0.12056529999972554,"all_physical_freeze_pins_match":true,"actual_processes_GPU_framework_assets_canonical_DB_actions":0,"source_edits":0}
```

No additional falsifier or suite was run. The raw/time above is completed tool
output, not an invented result or real checkpoint observation.

## Disposition

The completed OPS finding set is CPU1-OPS-001/002, including the unknown outward-
status consequence of cleanup failure. Separate SCI completion/dose/late-deadline
findings remain its report's responsibility and must enter Root's complete round2
disposition; no source acceptance follows from fixture5/5 alone.

The new CPU envelope and Job commit/RSS distinction do not lower the old GPU4GiB
floor or relax RAM/numerical/cumulative lease/budget checks. Fit and model bytes
remain unmeasured. Preserve this exact candidate/raw evidence, register the finite
correction within default3, and require fresh frozen source/fixtures/SCI/OPS review
before any120s load. No model attempt, native budget reset, retry or old native/
reader/acquisition reopening is granted.

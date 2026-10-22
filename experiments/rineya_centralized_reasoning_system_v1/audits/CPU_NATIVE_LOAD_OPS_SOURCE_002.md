# CPU load source round2: OPS review

Date: 2026-10-04. Reviewer `/root/exception_ops`; builder
`/root/desktop_profile_builder`. **SOURCE_APPROVAL_WITHHELD_ROUND2.**
Cleanup/accounting corrections close the preceding OPS failure paths at their
reviewed interface, but one required-import bootstrap false positive remains.
Preserve this freeze; no model attempt or further repair inside round2 is approved.
Root may register a finite final round3 under standing authority/default3, with
fresh independent review. Fit and actual Torch import remain unmeasured.

Only this report was written. Review inspected source and retained fixture
artifacts; no source import, tests/falsifier, model/framework/asset load, GPU/
process/database observation or source edit occurred. I authored prior correlated
OPS scope/source reports, not this implementation. Shared provenance is disclosed.

## Exact verified subject

All declared package source/preserved/evidence hashes match physical FREEZE:
`bbf52867244eb6cbcd03b870db7b0501b2793e4e360a7e39e5db0312f36e20e9`.

| Candidate | SHA256 |
| --- | --- |
| cpu_native_load.py | `5c069a26c42bb135ff31dd60dd93a3d8f6aa855db5c5c8c05490d54140532c33` |
| tests/test_cpu_native_load.py | `a2302e5b1e9d7f9f2887d4318a9679f09c8f64b164731563e55a83861b49b8da` |
| CPU_NATIVE_LOAD_FEASIBILITY_V2.json | `019c202afa302a9172a2d025591b9ad19bc508ccae485fbc8290405442287413` |
| finite correction scope | `af6f1d6506899a328845709fac2e9ee42f484effe00cf0c17f9e1d6ec1806229` |

V1/source1/freezes/raw history remain preserved. Original native84531289,
operationsba27ec8a, probe845d1e66, reader0299cb37/executionc66a1b64 and native
contractd79f3b6a remain unchanged. No asset or external runtime-library byte
validation was repeated by this review.

## CPU2-OPS-001 — P1: nested CUDA definition bootstrap still refuses absent initialization metadata

In the inspected pinned torch.cuda.__init__ source674c693e..., lines29–39 import
_device_limits/gds/_utils/graphs/green_contexts/streams **before** the module's
`_initialized=False` assignment at50. The actual profiler therefore sees the main
torch.cuda module present with state=None while a nested definition module starts.

`allowed_cuda_import_call`144–150 permits a nonfunction definition event when
state=False, or when state=None **only** for name `<module>` and module exactly
torch.cuda. A required torch.cuda._device_limits `<module>` event consequently
returns False despite its filename belonging to the pinned definition set.
`forbidden_call` still rejects its torch.cuda prefix. The return guard also rejects
None on every torch.cuda return, so simply allowing that entry would not close
the bootstrap. No device/API action or model construction is needed to encounter
this refusal.

The retained synthetic metadata test uses module torch.cuda for every decision
and False state for class/helper cases. It covers the main None entry but misses
the actual early nested-module/return path. The source-order and predicate proof
is static; no framework import or additional falsifier was performed.

Final-round closure must narrowly map exact pinned bootstrap definition/module/
class events and their None entry/return state before assignment. None must not
permit device functions, initialized state or arbitrary callbacks. Metadata helpers
remain restricted to verified False state/import phase; device availability/init/
device/RNG/sync/CUDA-allocation calls stay fatal. No broad CUDA allowance or model
attempt to discover this already demonstrated false positive is justified.

## Closed operational failure paths and retained limits

The known-launcher helper now obtains PID, creation FILETIME and executable from
the original Popen handle. Pre-Job cleanup requires that exact stable handle
identity, never-resumed state and no descendants, then kill/wait/observed exit.
Unknown or lost handle does not trigger a PID guess and keeps conservative120s
accounting. The retained actual suspended stdlib fixture proves this handle
cleanup with no framework import. No unrelated application control is added.

Outward status now becomes OUTCOME_UNKNOWN when dispatch/exit proof is missing.
Post-cleanup success checks reject absent completion, expiry, resource reason,
nonfinite/late>120s elapsed, and accounting/publication failures cannot return
ordinary SUCCEEDED. Known terminal publication failure is distinguished from
unknown held reservation. Original lease/fence/cumulative budget logic and fixed
single120s/no-retry dose remain; no extra CPU or probe002 attempt is granted.

The source also binds the exact unique ordered310 tensor schedule and strict
integer zero counters/device/dtype before success; independent SCI owns the full
scientific completion assessment. Job0x2200/2GiB committed memory and2GiB tree RSS
remain distinct. Host1.25/pre3.3125/disk4, freshness5s, original four owned exe
identities, suspend/assign-before-resume/GO and old GPU4GiB gate are unchanged.
The selected import policy retains hidden C effects UNMEASURED and has not been
qualified by a live framework import.

Raw fixture output records one5/5-group invocation,0.225s suite/0.563s command;
stdoute4905c59.../stderre8676289.../result50f28a53... match the freeze. Only the known
suspended-launcher cleanup is an actual process fixture; import decisions,
completion/order/dose, deadline/resource and unknown controls are synthetic.
They establish no CPU model fit or current admission.

The complete remaining material OPS finding is CPU2-OPS-001. Hold this candidate
and preserve all outcomes. Source/commit/fixture/live-admission gates and old
quarantines remain closed until a separately frozen accepted resolution exists.

```json
{"review_id":"CRS-CPU-NATIVE-LOAD-OPS-SOURCE-002","reviewer":"/root/exception_ops","builder":"/root/desktop_profile_builder","status":"SOURCE_APPROVAL_WITHHELD_ROUND2","freeze_sha256":"bbf52867244eb6cbcd03b870db7b0501b2793e4e360a7e39e5db0312f36e20e9","material_findings":["CPU2-OPS-001"],"preJob_cleanup_unknown_outward_failure_paths_closed":true,"retained_fixture_groups":5,"reviewer_tests_or_falsifiers":0,"model_framework_asset_GPU_DB_actions":0,"source_edits":0,"actual_CPU_study_attempts":0,"fit_or_launch_approved":false,"default_source_rounds_max":3,"repair_inside_round2":false}
```

# CPU-only runtime artifact batch: OPS review001

2026-10-05. Reviewer `/root/cpu_runtime_control_ops`; coordinator `/root`.
Shared model/workspace and prior OPS scope authorship are disclosed. I did not
author or execute the inspected acquisition/inspection tools. This independent
review reads frozen source/data, hashes local files and inspects ZIP bytes with
PowerShell/.NET; it does not rerun acquisition or dependency resolution, refetch,
extract, import target code, install, create a venv, run tests or observe
processes/GPU/host resources/DB. Only this create-only report is written.

**Verdict: ARTIFACT_BATCH_AND_EXACT_LOCK_ACCEPTED; OFFLINE_SETUP_CONDITIONALLY
ELIGIBLE_AFTER_SEPARATE_SETUP_PATH_TOOL_SCOPE.** The wheel artifacts and finite
stage are accepted at their documented scope. No setup command, Torch bootstrap,
model load, CPU hardware isolation or native/reader qualification is approved by
this review. Missing setup-specific interpreter/installer startup and system
search-path binding is the precise prerequisite to one offline setup, rather
than a reason to refetch or replace the accepted wheels.

## Exact frozen bindings

| Artifact | Physical SHA256 |
| --- | --- |
| FINAL_FREEZE.json | `b8e2c626c1112e18944ede5b066082d11288db496f5c721917700ef6f1322fe1` |
| ARTIFACT_BATCH_AND_HASH_LOCK.json | `f95186e2d56c8039063af8a5f923daad76eefeee2b06515b0af669e30ca7279c` |
| requirements-cpu-lock.txt | `a39fae225772267d5973e88f75e949697d4894969a16080bb101b585411475c7` |
| ACQUISITION_PLAN.json | `d82df9049e0b63c8baa7e708f2b4c2769df5b99fc320f7860f9cc1674336eaef` |
| Preserved preliminary FREEZE.json | `cf3b181d7e3daebf63354844db0d3d7a69d24929bc6d851d96a38db115f3b902` |
| Preserved STATIC_INSPECTION_AND_CLOSURE.json | `3cafa8d37d73557113b8c5935d522197bb666ec0047d7e5c73a1a2158660eb91` |
| acquire_batch.py | `99f30c56ceef4090cd2bc6fd822cd7240c9cc0304210f88e9b00b9ad2495c1c5` |
| inspect_static.py | `e2608322bb848d1a16c6f53aaea848ada7475dc5fb4c02d87cd8caf994d23c9d` |
| Artifact scope003 | `eef6b84f5fd7d69e5d63160508a4aee6ec116dceecd2dc7dfb66ac674a90d498` |
| Metadata selection amendment004 | `7ddb209c5b6bba259ba4507fb20d92410828f4117726fc1f600ad29d05d142be` |
| Official Torch CPU wheel | `17a09465bab2aab8f0f273410297133d8d8fb6dd84dccbd252ca4a4f3a111847` |
| First transport's known HTTP403 receipt | `63577252fe7407296f243d172bee5814f473ec04f7acc12781a2bc92fdFA1cc2` |
| Second transport ATTEMPT.json marker | `d920e8a32efa9aea7d3548545a6c5c41a69680d231753618dd424e695d0d0d03` |
| Second transport WHEEL_INSPECTION.json | `e6172d2cba146a0de39ebbcd483ebea98bc0757a8921adb98b0582447795c756` |
| BASELINE_DISTRIBUTIONS.json | `c3f09cfb8161a835ae0beff6f3a77afcf88670aaa34b40ac4c80429cfef0d7a7` |

Physical hashes and lengths match all30 files listed in FINAL_FREEZE; all25
whole-wheel artifact hashes/sizes, five base interpreter/native files and all15
declared pure-Python packaging inspection-tool source files also match. The
preliminary bindings to primary advisorybac06883 and OPS scopef667f6be match.
Hashes are byte identities, not proof of trust or execution behavior. No package
version string substitutes for the actual artifact pin.

## Dose, transport and preserved preliminary state

The first official download-r2 transfer failed HTTP403 before target open with0
received bytes. Explicit transport002 selected the same filename/hash from the
official download.pytorch.org origin, one<=120s/512MiB attempt; its unknown-until-
verified marker remains preserved. The retained inspection and independently
matching physical113636478-byte wheel reconcile that exact attempt to received
hash-verified bytes. The unknown marker is historical start evidence, not a
current permission for another transfer. No third transport is eligible.

The dependency stage contains exactly24 retained official metadata responses,
831220 response bytes and one24-wheel acquisition batch, each event attempt1,
HASH_VERIFIED, published=physical hash, unchanged official origin. Combined25
wheels total152491224 compressed bytes, with38854746 new dependency bytes.
The final receipt reports295.07117768478383s elapsed from the original stage,
5.186999999999898s batch active time and11.498999999999796s cumulative active
inspection. These measurements are acquisition/inspection costs, not native
model/qualification results; no budget reset follows.

All documented doses remain within scope003 plus the recorded prior amendment:
25<=40 wheels,152491224<=805306368 compressed bytes,24<=80 metadata responses,
831220<=33554432 response bytes,295.071s<=600s,82<=100 selected source/metadata
files,2421905<=8388608 source/metadata bytes,42<=64 selected PE files and
329942576<=1073741824 PE bytes. The amendment explains in advance why75 complete
wheel metadata files plus selected source exceed the original64-file ceiling;
it changes no other limit. It is not an outcome-selected version search.

The preliminary NOT_READY checkpoint deliberately stopped after metadata
closure, before the already registered single artifact batch. Its bytes and
prior costs survive in the final freeze. The copied native-boundary note saying
helper artifacts still need binding is historical: the final batch now binds
them. This review closes that exact artifact-identity gap while retaining the
unresolved runtime path/dispatch/system bindings. Do not rewrite the preliminary
status or infer that its other unresolved entries disappeared.

## ZIP, RECORD, dependency and native payload evidence

Independent ZIP enumeration of all25 artifacts found no unsafe names, absolute/
drive/backslash/traversal ambiguity, case collisions, duplicate archive names,
symlinks, excessive ratios or startup `.pth`, sitecustomize.py/usercustomize.py
members. All RECORD name sets cover all non-directory members, with no duplicate
RECORD paths. The source rejects device names/trailing spaces or dots, encrypted
members and declared byte limits before extraction. No extraction occurred.
Aggregate expanded bytes are570999036, below2GiB; the largest archive has11721
members, below the bound. The helper checks per-wheel expanded caps; the actual
independently summed aggregate establishes the registered total for these bytes.

All75 retained METADATA/WHEEL/RECORD byte hashes independently match the physical
members. All37 wheel PE members, five selected Torch Python sources and two CMake
build-source hashes/sizes match, including their selected RECORD hashes/sizes.
The five additional base Python executable/DLL files produce the final42-file PE
selection. Whole-wheel published hashes bind unselected members; this review does
not claim that every unselected source/native internal behavior was audited.

All24 dependency filenames, origins, published hashes and sizes match the retained
official PyPI JSON and the prewritten acquisition plan. All25 exact lock lines
match received wheel versions/hashes. The source validates cp311 WindowsAMD64,
compatible abi3/pure-Python tags, physical Requires-Python and wheel METADATA
identity against the fixed3.11.15 environment. Its physical Requires-Dist pass
has31 active no-extra requirements, all present in the25-package lock; active
version conflicts, new absent baseline dependencies, extras and direct dependency
URLs are refused. Other versions remain the baseline's exact versions, with
Torch alone changed to2.10.0+cpu. No installer resolver or native import was used.
The hash-bound existing packaging parser was approved inspection tooling under
isolated/no-site Python; the old environment remained read-only.

The selected official artifact supplies stronger evidence than its name alone:
all ten Torch native payloads are AMD64 and bound; version source has CUDA/HIP/
ROCm/XPU metadataNone; Caffe2/Torch generated configuration disables CUDA/HIP/XPU
branches while selecting CPU libraries; selected native import/delay-import
tables bind TorchCPU/OpenMP/core dependencies. Helper native wheels, including
NumPy's bundled BLAS and renamed C++ runtime, are now bound too. This is a useful
CPU-build provenance argument. PE import tables, metadata and CMake alone do not
enumerate computed LoadLibrary/GetProcAddress targets or internal C dispatch.
Runtime `_C._show_config`, actual loaded-module identities and hardware isolation
remain unobserved. Pure CUDA-named Python metadata cannot be treated as proof of
hardware use or as a reason to repair the old quarantined prefix classifier.

## Precise next setup prerequisites and finite resolving scope

**A narrow static setup-path/tool record is necessary before the offline install.**
The five base executable/DLL pins are present, but the actual chosen venv creator/
installer, its startup source or bundled wheels, its immediate native dependency
paths and isolated startup/search policy are not yet bound. In particular system
API-set hosts and the interpreter's System32 imports are unresolved in these
records. An installer that imports already installed target packages, inherits a
global/user site or executes an unexpected startup hook would exceed this scope.

The smallest sufficient next registration is one read-only static invocation
<=60s, no network or target execution, at most64 selected executable/DLL/bundled
tool files/512MiB hashed,32 startup/tool source/data files/8MiB read and64 explicit
import/API-set resolution rows. It must bind the exact installer and creator
(e.g. the selected pinned uv executable, or pinned venv/ensurepip/pip sources and
bundled wheels), base stdlib startup/venv configuration, direct base/installer
DLL imports and their canonical System32/API-set hosts. Declare trusted OS loader
boundary and dynamic-loading limits rather than claim a recursive audit of every
Windows DLL. Use exact named files, not an unbounded directory/process inventory.
Unknown relevant routes or cap exhaustion stop without installation.

That record must also specify canonical absolute `.venv/runtime` destination,
prove it is a distinct new path within this CRS package and not `.venv/artifacts`,
bind artifact/temporary/log roots and refuse existing targets/reparse escapes.
Bind resolved base executable and future launcher identities; disable inherited
PYTHONPATH/PYTHONHOME/user site/global site/current-directory shadowing and
installer configuration/plugins/version checks/network. Use only declared base,
runtime and trusted System32 DLL search roots, recording what Windows resolves
through its loader/API sets. The existing old venv must not enter runtime paths.

**Full target-helper MSVCP140/API-set/system-DLL resolution is a pre-bootstrap
gate, not a necessary prerequisite to copying wheel bytes**, provided this exact
setup never imports those target payloads. Bind base/installer dependencies before
setup; retain the complete target native/search-root and loaded-module proof
requirement before Torch or any helper is actually imported. This distinction
does not assert target binaries safe or permit model/config/tokenizer access.

After that accepted static record and a separately frozen setup scope, one new
venv creation plus one offline install may be eligible, together<=300s and<=2GiB
new environment bytes. Use the accepted25-line hash lock, wheel-only/no-index,
no dependency fallback/resolution, no cache substitution, no builds/hooks/extras,
no bytecode compilation or post-install target imports/tests. Freeze exact tool
commands/environment, scope/attempt identity, write destinations and before/after
artifact/source pins. Use bounded whole-tree supervision/cleanup and record setup
time/bytes/failures; never silently recreate or delete an existing environment.
Failure or unknown completion stops with partial receipts; reconciliation is
required before any further action and does not authorize another install.

One-worker,2GiB tree RSS and separate2GiB Job commit, host reserve1.25GiB,
preworker3.3125GiB, disk4GiB,5s freshness,0.1s watchdog and cumulative native
7200s/epoch14400s/rolling24h with historical47.42100000000573s remain intact.
The setup scope must state its accounting and fresh admission responsibilities;
artifact acceptance is not evidence that host admission currently passes. A
native qualification invocation uses canonical accounting. GPU4GiB floor and
desktop policy, old loader3/3/source quarantine, prior keys, Probe002, canonical
weights and CHECK/FINAL remain unchanged.

After setup, static installed-byte/path verification precedes separately reviewed
source/fixtures/exact-commit/current admission for one<=60s actual no-model
bootstrap. Actual loaded module paths and owned-tree exit require observation.
No model dose, CPU-to-GPU qualification transfer, numerical/reader/learning claim
or full-mission completion follows from artifact acceptance or setup eligibility.

```json
{"review_id":"CRS-CPU-ONLY-RUNTIME-ARTIFACT-OPS-001","reviewer":"/root/cpu_runtime_control_ops","status":"ARTIFACT_BATCH_AND_EXACT_LOCK_ACCEPTED_OFFLINE_SETUP_CONDITIONAL","final_freeze_sha256":"b8e2c626c1112e18944ede5b066082d11288db496f5c721917700ef6f1322fe1","batch_sha256":"f95186e2d56c8039063af8a5f923daad76eefeee2b06515b0af669e30ca7279c","lock_sha256":"a39fae225772267d5973e88f75e949697d4894969a16080bb101b585411475c7","frozen_files_verified":30,"whole_wheels_verified":25,"metadata_member_hashes_verified":75,"native_wheel_members_verified":37,"base_executable_DLL_pins_verified":5,"inspection_packaging_source_pins_verified":15,"archive_startup_hooks":0,"artifact_material_findings":[],"setup_prerequisites":["exact installer/base startup/system path binding","separate bounded setup source/scope/accounting/admission"],"target_runtime_system_dispatch_loaded_module_proof":"DEFERRED_MANDATORY_PRE_BOOTSTRAP","setup_dispatched_or_source_approved":false,"actual_bootstrap_model_GPU_reader_training_qualified":false,"reviewer_network_target_import_install_model_host_GPU_DB_actions":0,"old_caps_keys_quarantines_preserved":true}
```

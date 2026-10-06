# Compiled library observer candidate and startup binding

A bounded native collector candidate now exists. The installed MSVC toolchain
compiled the C/header and a portable mock-notification test executable. The
mock core passed20 cases and123 checks without Windows notification registration.
The candidate DLL is5,632 bytes, SHA256
`8f6ac9f0edcd716b62e06054ec6ca503d961b598177144ffcb1493656e24c679`.

[Independent PE inspection](evidence/native_library_observer_candidate_reviewed_cpu_20261006/review/independent_static/OWNED_OBSERVER_PE_STATIC_REVIEW_1791252310528877500.json)
finds an x64 image with no imports/delay imports or entrypoint and exactly five
expected exports. Callback-reachable calls remain inside the collector. The
[build receipt](evidence/native_library_observer_candidate_reviewed_cpu_20261006/observer_source_work/native_library_observer_build.json)
records exact inputs, compiler/linker bytes, flags and ABI. Its source commit
is the repository ancestor; hashed source inputs were explicitly uncommitted
at build time. No future commit provenance or native permission is fabricated.

The adapter uses counted UTF16 rows and a fixed ABI/prefix, rejects loss,
truncation, invalid names, rewritten prefixes, changed bindings and failed drain.
All selected observer/system binary checks precede the first DLL load in source.
Cleanup attempts the captured transport even when context validation fails;
primary errors survive cleanup errors. Native transport objects remain resident
for process lifetime. A C-owned registry captures the original context, selected
adapter/class/methods and ABI handle. Main binds those exact observer/bootstrap
hooks to its selected loader. Actual source controls use inert transports.

The [failure-only route](evidence/native_library_observer_candidate_reviewed_cpu_20261006/prefix_library_failure_audit.py)
now reads actual worker pre-core failure sidecars through the full preserver,
checks original/copy/source/lease/resource/death facts and replays the external
plan. It returns only INCOMPLETE_FAILURE_PRESERVED, with no certified executing
birth or successful tensor work. Terminal OS/resource facts in its nine CPU
controls are explicit injections. The full success audit's requirements remain.

[Root preserved-copy324-control replay](evidence/native_library_observer_candidate_reviewed_cpu_20261006/ROOT_COMBINED_DLL_CPU_REPLAY_036.json)
passes with unchanged source hashes and no actual forbidden imports. It includes
adapter14, actual module/context/startup/build joins5, registration13 and actual
pre-core failure9 controls. [Acceptance](evidence/native_library_observer_candidate_reviewed_cpu_20261006/ROOT_ACCEPTANCE_038.json)
retains false native qualification. Independent adapter18 covers the earlier
four binding cases; the new fifth compiled-receipt join passed in Root's replay.

## Qualification remains open

The candidate observer DLL has not been loaded or registered. Portable tests,
static call graphs, ABI metadata and source wiring cannot prove actual Windows
notification completeness or post-unregister quiescence. Microsoft's
[callback contract](https://learn.microsoft.com/en-us/windows/win32/devnotes/ldrdllnotification)
constrains calls from the notification; the
[unregister contract](https://learn.microsoft.com/en-us/windows/win32/devnotes/ldrunregisterdllnotification)
is not a measured lifetime/barrier result on this host. Current bootstrap,
loaded-module identities, exact runtime bytes, owned resource fit and terminal
auditor startup/runtime scope still need qualification.

[Observer-only proposal002](evidence/native_library_observer_candidate_reviewed_cpu_20261006/OBSERVER_ONLY_QUALIFICATION_PROPOSAL_002.json)
keeps launch authority false. A complete executable worker/supervisor/audit
registration and explicit new-run permission are required by package AGENTS.md;
the earlier pilot-choice delegation does not authorize this new native scope.
The proposal is not a dispatch manifest. No foreign-cookie or hostile-namespace
requirement is reintroduced.

No model/GPU training, inference, M0 collection or scoring ran. M0 remains256/384
raw unscored rows. Actual matched M1/M2 training, fresh-family and held-out RS
transfer and prior-correct preservation remain unmeasured. All original weights,
resource/numerical floors, protected banks, tracks and negative evidence remain.

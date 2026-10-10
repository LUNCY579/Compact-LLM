# CPU-only offline setup design: OPS review001

2026-10-05. Reviewer `/root/cpu_runtime_control_ops`; coordinator `/root`.
Shared model/workspace and prior artifact/scope reviews are disclosed. This
create-only review reads the frozen prospective proposal and authorization005.
No UV/Python execution, network, target import, installation, test, framework,
model, process/GPU/host-resource/DB action or generic system expansion occurs.
Only this report is written. Static005 is in progress; no completed005 result
was supplied for this initial disposition.

**Verdict: NOT_READY_SETUP_DESIGN_PREREQUISITES; SOURCE_AND_LAUNCH_NOT_APPROVED.**
The finite proposed direction is coherent as a prospective design, but exact UV
provenance/command/query semantics and startup/search boundaries are explicitly
unverified. Those are prerequisites, not details to discover by trying the setup.
An unsupported flag or unsafe query cannot be repaired after the single setup
starts. This is design round1, not an extra old-loader repair or setup attempt.

## Frozen identities and limited scope

Physical PLAN.md SHA256 matches
`57397ff857168d2ae26cd716665f6356e1b3f63a16ae0fff39ff6d130d738d25`;
physical PROPOSAL.json matches
`46192021cf96bad9f3d291e5326a22886d2652b39e10577815169f9c7f86caf3`.
Authorization005 physically matches
`c319e24a1d0066c2735266824990b863a4c08faf4c4c077d55ad649712a0f64e`.
The proposal retains accepted artifact freezeb8e2c626, batchf95186e2 and lock
`a39fae225772267d5973e88f75e949697d4894969a16080bb101b585411475c7`.
Any later registration must also bind current OPS artifact reviewb1005b17 and
current design/static/source dispositions, not only its earlier SCI reference.

The selected local UV executable is reported AMD64/68317696 bytes/SHA256
`5a7ec85884c2ccb1be560cb8fac3eb890df1adf49bfcc070a270ba70401bdd68`.
This review binds the proposal reporting those bytes; it does not independently
establish an official release, publisher hash/signature or UV version/source
correspondence. Embedded flag strings establish neither syntax nor semantics;
missing literal strings establish neither an unsupported flag nor a safe default.

The prospective target `.venv/runtime` is separate from preserved
`.venv/artifacts`; the new cache/work/temp roots are named separately. Initial
absence/ancestor checks are documentary, not live admission. The future
setup_controller.py is explicitly unimplemented. A controller description is
not source acceptance or OS containment evidence.

## OPS-SETUP-D001: exact UV commands and interpreter query remain unbound

The first proposed argv invokes UV venv against the exact base Python with
--no-config/--offline/scoped cache, --no-python-downloads, --no-seed and
--no-project. The second invokes UV pip install with --python BASE and --target
the runtime's site-packages, --no-index/local find-links, --require-hashes,
--no-deps, --only-binary :all:, --no-build, --no-compile-bytecode and copy mode.
Their exact availability, value syntax, interactions and error behavior must be
bound to this selected binary's version before either command is eligible.

The proposed BASE+--target combination plausibly avoids installed runtime hooks,
but that is not yet evidence of actual UV behavior. The proof must establish
which executable UV queries in both operations, exact child arguments, query
source and imports, whether UV queries the generated interpreter at any stage,
and whether installed target directories ever enter its query's sys.path. The
query must be limited to the declared base stdlib path; it cannot import target
payload or execute a generated runtime after package installation. Establish
--no-seed's generated-file behavior and exact --target .data/script/entry-point
scheme. Data placed below an unexpected nested bin directory cannot silently
be accepted as the reviewed ordinary venv scheme.

The no-bytecode claim needs exact query flags and installer semantics. If UV
uses Python -I, Python environment controls such as PYTHONDONTWRITEBYTECODE are
ignored; explicit -B or an independently established equivalent is then needed.
No-user/safe-path controls do not by themselves disable the base interpreter's
system site processing. Bind its .pth/customization/._pth inputs and exact query
imports, including any temporary helper source. Confirm UV does not compile
bytecode, run package entry points or invoke builds despite the proposed flags.
Later absence of .pyc is verification, not permission to test unbound behavior.

The current005 dose permits selected local static evidence only. If it cannot
establish exact-version semantics, freeze NOT_READY. A later separately bounded
official exact-version release/source/CLI binding may resolve these missing
facts under standing authority. It would be new prerequisite evidence before
setup, not a setup retry, installer/version fallback or automatic expansion of005.

## OPS-SETUP-D002: startup, DLL and network boundaries need scoped evidence

Constructing child environment values from an empty mapping, withholding inherited
secrets/config/accelerator/project roots and using an empty controlled cwd is
appropriate. Verify actual startup with the bound UV query and base source,
instead of inferring isolation from variable names. SETUPTOOLS_USE_DISTUTILS=stdlib
may suppress a known distutils .pth branch only after its exact condition/body
is inspected; it does not authorize an unexpected .pth or other customization.
Explicitly retain the base stdlib/global-site distinction and refuse unknown
startup hooks. UV-generated _virtualenv files remain inert installation data
pending later independent startup review; no new runtime invocation follows.

The declared UV directory/base/System32 DLL roots are a policy requirement.
PATH does not prove Windows loader search order, executable-directory shadowing,
KnownDLL/API-set resolution or absence of computed native loads. The selected005
record must bind direct setup-relevant imports and canonical OS/API-set host
disposition, state the trusted Windows loader boundary and note dynamic-loading
limits. A missing relevant application-directory/search-root disposition cannot
be silently replaced by a System32 file with the same name. Do not recursively
audit unrelated system files; unknown relevant routes stop the selected proof.
Full target-helper/Torch system-DLL and actual loaded-module evidence remains
mandatory before bootstrap, rather than being required for inert wheel copying.

The no-network claim similarly requires the bound UV offline/local-only path,
configuration exclusion and no-download semantics. Job membership alone does
not deny sockets or filesystem access; a source policy to reject an attempt
must name an implementable boundary. Describe any remaining enforcement as
cooperative hash-bound behavior, not an OS network or confidentiality sandbox.
Do not invent a firewall, secrecy or exhaustive internal-C guarantee. Native
source/loaded-module validation has specific observable coverage and cannot
prove all hidden Windows behavior.

## OPS-SETUP-D003: controller, resource and verification proof must precede setup

The proposed suspended assignment/kill-on-close/no-breakaway Job, exact owned
PID+creation identities, single300s monotonic deadline including preparation,
verification and empty-tree exit,2GiB RSS and separate2GiB commit,2GiB new runtime
bytes and512MiB cache/temp are suitable finite requirements. They are not yet
implemented or exercised. Freeze source and targeted fixtures that reject
wrong hashes/argv, pre-Job orphan launchers, failed assignments, stale telemetry,
reparse/ancestor drift, disk/byte/deadline breaches, late success, unknown outward
outcomes and incomplete tree exit. Unknown/failure consumes the registered setup
dose, preserves partial runtime/cache and stops; no fallback/retry/delete follows.

Define the thread ceiling precisely in the setup scope. OMP/MKL/OpenBLAS variables
control numerical-library concurrency; they do not establish the Rust UV
installer's thread/concurrency behavior or child-query count. Bind UV's actual
setup concurrency separately. A prospectively proposed UV concurrency setting
or two-CPU affinity policy requires exact semantics and source review; affinity
limits schedulable CPUs, not the number of OS threads. Do not silently rewrite
the existing numerical compute-thread limit or count installer threads as proven
bounded because no framework was imported.

Canonical absence/reparse/root checks must immediately precede writes and remain
valid across the child operations. A Job controls the owned process tree and
resource limits; it does not itself confine child writes to runtime/cache. Bind
the installer write/cache/temp scheme and preserve immutable artifacts/lock/old
environments. Prevent unintended hardlink/reparse/path escape and source drift;
do not claim that post hoc enumeration proves every external write impossible.
Record the enforceable and cooperative limits explicitly.

The complete installed-byte validation must be concrete frozen source before
execution: all25 locked identities, wheel-to-installed path relocation mapping,
RECORD hash/size and membership validation, explicit generated launcher/script/
bookkeeping dispositions, pyvenv.cfg home/include-system-site-packages=false,
startup hook and bytecode refusal,100000-file/byte caps and streaming hashes
inside the remaining300s deadline. A blanket exemption for installer-generated
files would defeat this proof. Installation failure retains partial evidence
without a verification-driven repair. No entry point or runtime Python is run.

Setup scope must bind canonical accounting/attempt/source identities and current
admission obligations while preserving2GiB RSS/commit,1.25GiB host reserve,
3.3125GiB preworker availability,4GiB disk,5s freshness,0.1s watchdog, one native
worker and7200s epoch/14400s rolling24h with historical47.42100000000573s.
Report setup cost separately; do not reset or misclassify native qualification
cost. Old CPU-loader3/3 quarantine, keys, Probe002, GPU policy/floor, weights and
CHECK/FINAL remain unchanged.

## Disposition

Keep the frozen proposal as design round1 and finish only the already registered
static005 proof. D001 remains a concrete setup blocker unless that selected
record supplies exact provenance/command/query evidence. D002/D003 are remaining
source and admission prerequisites, not evidence that the prospective direction
failed an actual setup. Freeze an honest stopping result if gaps persist. No
installer invocation is an acceptable semantics test under this design.

Only after independently accepted prerequisite evidence, concrete controller/
fixture/exact-commit review and a separately recorded one-dose setup scope may
setup become eligible. Actual setup review and new runtime source/loaded-module/
admission gates still precede any no-model bootstrap. Model/reader/numerical/
learning/full-mission claims remain unqualified.

```json
{"review_id":"CRS-CPU-ONLY-RUNTIME-SETUP-DESIGN-OPS-001","reviewer":"/root/cpu_runtime_control_ops","status":"NOT_READY_SETUP_DESIGN_PREREQUISITES","design_round":1,"plan_sha256":"57397ff857168d2ae26cd716665f6356e1b3f63a16ae0fff39ff6d130d738d25","proposal_sha256":"46192021cf96bad9f3d291e5326a22886d2652b39e10577815169f9c7f86caf3","static005_scope_sha256":"c319e24a1d0066c2735266824990b863a4c08faf4c4c077d55ad649712a0f64e","material_prerequisites":["OPS-SETUP-D001 exact UV provenance/argv/base-query/target/no-seed/no-bytecode semantics","OPS-SETUP-D002 scoped startup/DLL/offline boundary evidence","OPS-SETUP-D003 frozen controller/resource/write/deadline/accounting/verification proof"],"static005_completed_evidence_reviewed":false,"source_or_setup_launch_approved":false,"reviewer_UV_Python_network_import_install_test_model_host_GPU_DB_actions":0,"old_caps_keys_quarantines_preserved":true}
```

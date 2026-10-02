# Owned runtime identity diagnostic scientific review 2

Reviewer `/root/runtime_audit`; probe repair builder `/root/scientific_audit`;
original builder `/root`. Recorded 2026-10-02. Disposition:
**REPAIR_REQUIRED: one source-review identity gate remains open. No probe dispatch
or native/reader qualification is approved by this report.**

I independently read the complete frozen probe, its 17 tests, contract and both
first-round reports. I did not implement this probe or its tests. I previously
built the reused native/operations primitives and the separate reader candidate;
that shared engineering involvement, model lineage, workspace privileges and
historical evidence limit independence. This is fresh scrutiny of the new
resolving instrument and its claim scope, not approval of my native or reader
claims. Their whole-path quarantine is unchanged.

## Frozen inputs

| Artifact | SHA256 |
| --- | --- |
| owned_runtime_identity_probe.py | 946b4fbc6cbf013577b1a0e827e902c85a9cdef8382e9a5cb83c687f831d28e9 |
| tests/test_owned_runtime_identity_probe.py | da0c05c7319ec3f90368ed54057791b1a42ef072344f1327f24bed606d3df278 |
| contracts/OWNED_RUNTIME_IDENTITY_DIAGNOSTIC_V1.json | 651cdc696eaa08e1c1b1564542d82238e7e9b57c5373516a8d508742114258f7 |
| first SCI report | ea10b98ec8e4989bf4e33b149bbf2265929dbfad7ccaff30c5773ededc7c4d34 |
| first OPS report | d8a8eab8537cd5acc88edfca4ef8efa2f95e86bbde0cdcc5c4ec7db8271218b0 |
| unchanged native.py | 8453128981280019702dff80ebd5083031f7f2dcd8a7c6c9094587d9a2e377a3 |
| unchanged native contract | d79f3b6a51e939d78a99c5c287ef6f33ef5979cfa50e76ba3d930ed3724f826a |

The three subject hashes were checked before reading and again before writing
this report. The initial source/contract/tests and failed review remain preserved
in `history/owned_runtime_identity_probe_round1/`. The current contract has no
clean committed source identity yet. Source acceptance, if later granted, remains
conditional on both required review roles, exact clean commit/contract bindings,
fresh unchanged admission and one canonical 60-second reservation.

## Executed review controls

The pinned interpreter ran all 17 supplied tests independently: 17 passed, zero
failures/errors/skips, in 3.008 seconds. Six actual suspended, assigned Windows
Job controls executed only stdlib/harness code: five fatal-boundary modes and one
independent deadline while its observer sample blocked. The fatal modes were
main-thread Python, new Python thread, refusal-publication failure, a synthetic
C-call refusal and a Python serialization wrapper. All exited with code 86,
left no first/second/continued forbidden-effect marker, and proved their exact
owned tree empty. They never imported Torch/Transformers/tokenizers/safetensors.
The deadline fixture killed only its owned tree while its sample was blocked.
These are small CPU containment/failure controls, not the actual import probe,
real GPU telemetry or a model result. Mocked resource samples in these fixtures
are explicitly synthetic test inputs.

The other tests exercise individual/distinct hash limits before reads, partial
identity preservation, a fresh psutil object for drift, complete phase and
containment schema, missing/duplicate/reordered/source/boundary corruption,
canonical job/reservation association, cleanup error accumulation, publication
failure and Popen response loss. Temporary CPU SQLite stores retain the full
60-second unknown reservation on lost response and reject another dispatch.
No canonical host database, current host telemetry, GPU API, model asset, model,
tensor or protected bank was accessed in this review. Only the owned report is
changed; review fixtures use their own temporary state and recorded ownership.

## First scientific finding closure

OID-SCI-01 is substantially closed. `FATAL_EXIT` captures the builtin `os._exit`
before third-party imports; both the forbidden-API path and refusal-publication
error path invoke it in `finally`. The callback no longer returns a catchable
exception that disables profiling and permits a second body. The actual contained
fixtures test swallowed `BaseException`, new-thread continuation and failed
publication, not merely one rejected call. Python documents that profile-callback
errors disable the callback; `os._exit` terminates without cleanup/stdio flushing,
so retaining/fsyncing the refusal first and having parent cleanup remain material.
[Python profiling documentation](https://docs.python.org/3.11/library/sys.html#sys.setprofile),
[Python process exit documentation](https://docs.python.org/3.11/library/os.html#os._exit).

OID-SCI-02 is substantially closed. The named Python `torch.serialization`
load/save and underlying `_load`, `_legacy_load`, `_save`, `_legacy_save` wrappers
are now in the preregistered predicate at their Python call identities. Selected
CUDA Python APIs, CUDA-named Torch C calls, named factories, Python nn/Qwen/config/
tokenizer constructors and named safetensors IO remain explicitly selected coverage.
Generic C type construction, calls internal to C extensions and native threads
are not attested observed or zero. Requested-operation zeros are declared source
requests, not measurements of all hidden side effects. No library API is replaced
with a fake successful implementation by the actual probe body.

OID-SCI-03 is substantially closed within the stated bounds. `HashBudget` checks
64 MiB individual and 256 MiB distinct file identities before reading, preserves
size/time/inode consistency and actual cumulative bytes read, and retains known
PID/time/path/membership/RSS when a later hash fails. Repeated reads of the same
identity are possible: the distinct-byte ceiling is not a cumulative-read ceiling.
Observation and guard begin/end/gap fields label 20 ms/100 ms as targets rather
than achieved cadence. A separate timer terminates the exact Job at the absolute
60-second deadline, independent of observer hashing/telemetry; a separate raw RSS
thread does not hash or sample GPU. The real no-framework deadline control and
late-check fixture support this limited enforcement claim. Timer/OS scheduling and
sampling do not establish exhaustive resource or internal-operation coverage.

## Phase, authority and interpretation scope

The child checks the exact fixed KEY RUNNING job, 60-second reservation, owner,
fence, epoch, request hash, recorded launcher and clean source/contract before
framework work; a live shared lease alone rejects. Review/intent hashes, runtime
PID/creation identity, ancestry and the explicit owned query handle are bound.
The temporary query handle closes before imports on success and rejection paths.
These are cooperative protocol bindings over shared privileges, not OS identity
authentication against coherent same-user artifact forgery.

External completion checks the exact 32-event ordered child schedule: clean idle
of at least two seconds; ten indexed source begin/end pairs; two verify_sources
and two runtime metadata observations at indices 0 and 9; Torch import begin,
boundary armed/end; Transformers package begin/end; Qwen3 symbol begin/end;
two-second post-import observation; typed zero requested-operation final marker.
Source/runtime observations bind the frozen GO authority; wrong or incomplete
ordered records, missing containment, partial newline and duplicate JSON keys
reject. The exact Qwen3Config/Qwen3ForCausalLM symbol identities and transitive
module-name manifest are checked. Passive `torch.cuda._initialized` is read from
the existing module dictionary; no experiment-authored CUDA availability/init
query is added.

Raw inventory precedes the unchanged strict guard and links observation IDs,
notification values, timestamps and guard exceptions. Notifications remain
non-guaranteed and a recycled notification PID is not itself an executable
identity. Microsoft explicitly documents these limits.
[Microsoft Job completion-port semantics](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_associate_completion_port).
Child and observer timelines may localize a new refusal interval; coincident
import timing alone cannot establish causal responsibility for the original
attempt004 failure. Only a retained actual identity/path/hash/Job observation can
support a claim about this instrumented run. An unavailable identity, missing
phase or no reproduced refusal leaves the old cause unresolved.

The original registry/native code/math/tolerances are unchanged. No models,
weights, tokenizer/config instances, tensor work, forward/backward, training or
qualification are requested by the specified source body. A completed observation
still has `coverage_complete=false`, no hidden-internal-zero attestation, and all
native/reader/training/old-cause/quarantine-reopening flags false. Its finalization
receipt is distinct from an earlier pending-accounting result. Cleanup exceptions
or uncertain creation/exit retain unknown state and costs. The separate operations
review remains authoritative for complete notification/resource/cleanup safety.

## OID-SCI-04: ordinary aliases permit builder self-approval

**Finding:** `checked_reviews` (lines 275-290) excludes only exact stripped strings
`/root` and `/root/scientific_audit`. It does not normalize slash/backslash,
trailing separators or bare names, nor enforce the contract's explicitly required
science `/root/runtime_audit` and operations `/root/data_audit` identities.

**Independent executed reproducer:** two temporary JSON approvals have the exact
expected binding and distinct role strings. The operations reviewer is
`/root/data_audit`; vary the science reviewer. Production `checked_reviews`
accepts `scientific_audit`, `\root\scientific_audit`, and
`/root/scientific_audit/`. It rejects exact `/root/scientific_audit` and `/root`.
No framework, host-state database or probe is invoked. These are ordinary aliases
of the actual builder, so this violates the builder-cannot-approve provenance gate
and can label one builder as an independent science reviewer.

**Required closure:** apply one explicit normalized identity policy to both roles,
reject builder/original-builder aliases and duplicate normalized identities, and
require the contract's frozen exact role identities (or a separately reviewed
identity amendment). Test bare, canonical, backslash, trailing-separator and case
variants, including using two aliases of one person in different roles. No model,
registry, resource or phase change is needed. Fresh exact-byte targeted source
review is required before the bounded observation can dispatch.

## Disposition

The scientific instrument body and first three scientific controls are now
suitable in the narrow source scope described above, but OID-SCI-04 withholds
launch source approval for these exact bytes. This report approves no execution.
After that consequential identity gate closes and operations accepts its final
bytes, source approval may permit only one bounded 60-second CPU import-observation
attempt with clean source and fresh unchanged host/resource/lease admission.
It never qualifies native containment/math, the reader, training, a discovered
mechanism or a new executable. Native quarantine and the original unresolved
failure persist after every resolver outcome.

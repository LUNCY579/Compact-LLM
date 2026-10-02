# CRS-001 owned runtime identity resolver scientific review, round 1

Recorded 2026-10-02. Reviewer `/root/scientific_audit`; implementation builder
`/root`. Disposition: **REPAIR_REQUIRED; PROBE NOT APPROVED FOR EXECUTION.**
This is the first review of the separate no-model resolving experiment, not a
fourth repair round of the quarantined native candidate.

I authored the prospective design and the earlier narrower no-framework draft,
but did not build this code. This source/test-fixture review therefore offers
fresh implementation scrutiny, with disclosed design involvement; it is not
fully independent experimental design or independent ground truth. Reviewer,
coordinator and other reviewers share model lineage, workspace privileges and
historical evidence. Only source, JSON and small hash receipts were read. The
reviewer did not run the probe, import Torch/Transformers, read weights or labels,
construct models/tensors, or execute native/GPU calls. The builder's reported
seven stdlib tests were inspected, not rerun here.

## Frozen implementation inputs

| Artifact | SHA256 |
| --- | --- |
| owned_runtime_identity_probe.py | ebee87e14e10b602bdd56e9ff3a920a107b361def71baa1c0043f1953c83fb35 |
| tests/test_owned_runtime_identity_probe.py | 2dff8979d6881c0d1829581e6ee2adf09ac140d691fbc757e5fc01d028a481e7 |
| contracts/OWNED_RUNTIME_IDENTITY_DIAGNOSTIC_V1.json | 663ed38a22215808cc14953364bed91fc4e5e6f476e8621a59e117fce8aaa9aa |
| retained scientific design | 74c873a48bcea4cf927ab4501e32475342c8abcb9f398e9d6217f7213f0292f1 |
| unchanged narrower draft | dbe826be7ee8f2e1b54c93d6f8923b7422bf60005aa1d9f5cb96f9b49063b33a |
| preserved native.py | 8453128981280019702dff80ebd5083031f7f2dcd8a7c6c9094587d9a2e377a3 |

The first three hashes were freshly checked. The scientific design is retained
under `contracts/history/OWNED_RUNTIME_IDENTITY_DIAGNOSTIC_V1_74c873a4.json`.
The frozen contract binds the implementation/test/native/operations sources,
selected installed import files, exact registry hash and scope. Source/contract
acceptance would still require a clean committed identity and separate operations
and scientific attestations before the one authorized observation attempt.

## Scientifically appropriate elements

The implementation runs fixed source checks before import, then package Torch,
package Transformers and the original worker's Qwen3Config/Qwen3ForCausalLM
symbol-resolution path. Symbol definitions are distinct from model/config or
tokenizer instances. Transitive safetensors/tokenizers modules are recorded,
without interpreting their presence as weights or tokenizer use. There is no
experiment-authored model construction/forward, dataset access, tensor operation,
CUDA availability/init call, device hiding or precision modification in the
child's prescribed body.

The exact owned Job has notification association before resume and query-only
transfer to a PID/creation-bound runtime. Raw inventory distinguishes unknown
path from wrong hash, including query failures, before invoking unchanged strict
membership/RSS helpers. The original executable registry and native source stay
unchanged. One durable key, lease/reservation and retained outcome proof are
intended to prevent repeats and preserve failed/uncertain cost; operations review
must establish their actual Windows behavior.

Claim flags are appropriately narrow: no native/reader/training qualification,
no reopened quarantine, no original failure cause established, and coverage is
explicitly incomplete. Notification delivery, native threads, C-extension internal
calls and generic C type constructors are not claimed completely observed. A
profiling/sampling/hashing intervention changes timing, so a refusal during this
instrumented run identifies its observed path/phase, not automatically the cause
of attempt004. A clean run cannot establish the absent original executable or
universal CPU-only import behavior.

## OID-SCI-01: the first catchable refusal disables profiling and permits continuation

**Evidence:** `import_boundary` at `owned_runtime_identity_probe.py:60` installs
a Python/C-call profiler, appends a violation and raises NativeBlocked at `:80`.
Checks for retained profiling/violations occur only after an import milestone;
the supervisor detects the refusal file on a later loop iteration.

CPython documents that an error in the profile callback unsets profiling. The
callback exception is a catchable RuntimeError subclass. A library can catch the
first refusal and then execute a second forbidden operation on that thread before
the milestone check or parent observer abort. Raising SystemExit instead also
remains catchable; simply checking profile identity after import is too late.
The initial forbidden call's body may have been blocked while a subsequent body
is unobserved. This contradicts stopping at the first boundary violation.
[Python 3.11 sys.setprofile documentation](https://docs.python.org/3.11/library/sys.html#sys.setprofile).

**Bounded reproducer for repair verification (not run by reviewer):** reuse the
test's synthetic namespace, no framework or GPU. Define fake `torch.cuda`
functions `is_available` and `init`, each appending a distinct marker. Install
the actual profile, call is_available inside `try/except RuntimeError`, then call
init. Inspect markers and sys.getprofile before any reinstatement:

```python
namespace = {'__name__':'torch.cuda', 'effects':[]}
exec("def is_available():\n effects.append('first')\n"
     "def init():\n effects.append('second')", namespace)
profile, violations = probe.import_boundary(refusal_path)
try:
    namespace['is_available']()
except RuntimeError:
    pass
namespace['init']()
print(namespace['effects'], sys.getprofile(), violations)
```

Expected defect on CPython: first marker absent, second present, current profile
None. The existing single-call and new-thread tests stop at the caught first
exception and therefore do not test continuation. Repeat the swallowed-first/
second-side-effect test in a newly created Python thread as well.

**Impact:** the claimed fail-before-body boundary is not maintained after its
first violation. A retained first-refusal log and later FAILED disposition cannot
retroactively prevent an intervening forbidden operation.

**Verification criterion:** durably retain the first refusal, then make child
continuation impossible even if importing code catches BaseException, and retain
parent containment/exit/cost reconciliation. A terminal process abort is one
possible implementation; its durable-publication failure path also needs to stop.
Do not monkeypatch library APIs, suppress the refusal or restore profiling and
continue the import. Parent tests must prove both first and second bodies do not
execute in main and new Python threads, including refusal-publication failure.

## OID-SCI-02: declared checkpoint API coverage does not match the Python predicate

**Evidence:** `forbidden_call` at `:34` denies Torch load/save only for `c_call`.
The installed Torch init reexports `load,save` from `torch.serialization`; those
are Python wrappers. `forbidden_call('torch.serialization','load','call')` and
the analogous save predicate fall through False. The contract's implementation
description says checkpoint load/save APIs are observed and blocked.

**Counterexample/reproducer:** create a fake `torch.serialization` Python load
function whose body appends `read_model_bytes`, install the profile, invoke it.
The current predicate does not refuse its Python body. No actual checkpoint or
Torch import is needed. The same issue applies to the save wrapper. The existing
constructor/factory predicates are also a selected list, not complete tensor,
checkpoint or model-I/O coverage; generic C type constructors remain explicitly
unobserved and must retain that limitation.

**Impact:** describing blocked checkpoint APIs overstates the implemented guard;
an observed import may traverse an uncovered wrapper before any later factory or
CUDA event. This review does not assert that the pinned import path calls load.

**Verification criterion:** include the declared Python checkpoint wrappers and
other expressly claimed named APIs at their actual call-event/module identities,
with no-framework fake-module tests. Bind the exact covered predicate in the
receipt/contract and label uncovered C/type/internal paths as unobserved. Preserve
the zero *experiment-requested* count separately from evidence about actual
transitive operations; no hardcoded zero is a measurement of hidden effects.

## OID-SCI-03: observational hash bounds and achieved cadence are not established

**Evidence:** `inventory` at `:156` invokes `native.sha(executable)` at `:175`
without the contract's 64MiB individual or 256MiB distinct-executable byte check.
It rehashes complete executable files on each sample. The supervisor loop at
`:365` serializes inventory, append/fsync, guard and blocking GPU telemetry before
sleeping .02 seconds; the .1 guard schedule is checked only on subsequent loop
iterations. These sleeps are configured minimum delays, not achieved20ms/100ms
sampling or independent resource watchdogs.

**Counterexample/reproducer:** use a fake observed process/executable with file
size greater than the frozen individual ceiling, and a mocked sha call that
records invocation. The current inventory reaches full hashing without a size
gate. Likewise, inject bounded slow inventory/telemetry callbacks and compare
observation begin/end timestamps; the serial loop's gap exceeds .02/.1 without
being visible as an explicit missed-cadence field. No large file or native process
is needed to test these branches.

**Impact:** an unknown executable's observation can exceed the preregistered I/O
envelope and delay the observer/strict guard. Identity-query failure can also
produce only partial RSS. This is consequential for bounded interpretable
evidence, even though the whole-path candidate remains quarantined and sampled
coverage is already labeled incomplete. Detailed process/resource/monitor safety
remains the independent operations reviewer's responsibility.

**Verification criterion:** enforce individual and distinct executable-byte
bounds before observational hashing, retain explicit oversize/unavailable identity
and refuse without registry growth. Record actual observation/guard begin/end,
timing gaps and any missed cadence; distinguish target intervals from realized
ones. Operations review must verify an independent bounded enforcement path if
an observer/hash/telemetry call blocks, without editing the native helper or
weakening the existing RAM/Job policy. Fake oversize/hash/slow-call tests should
establish refusal or explicit incomplete coverage and retained elapsed costs.

## Claim boundary and next disposition

The proposed no-model import observation is still scientifically useful after
these repairs and operations approval. Its outcomes must remain observed identity
refusal, completed specified phases with sampled/incomplete coverage, or
unresolved/boundary failure. A refusal attributed to a specific pinned import
phase is a new observation under profiling/timing intervention, not proof of
original cause, permission for that executable, or native process containment
qualification. If the resolver does not reproduce the refusal, leave the cause
unresolved and preserve its single-attempt policy.

No model fit/quality, zero hidden CUDA/tensor activity, native numerical result,
efficacy, learning, retention or promotion is established by this source review.
Original native source and .03125/.01 bulk failure remain unchanged; attempt004
47.359-second aggregate failure cost and the three-round quarantine remain.
Root and the operations reviewer were sent this complete first-round finding set
before the next coherent repair freeze. Approval is withheld pending fresh review
of repaired exact bytes and bounded no-framework counterexamples.

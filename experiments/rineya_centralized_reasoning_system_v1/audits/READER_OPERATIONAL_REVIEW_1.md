# CRS reader operational review 1

Recorded 2026-10-02, America/Chicago. Reviewer `/root/data_audit`; not the reader
builder. This is a distinct reader milestone, not another repair round of native
D2. Scope is source and lightweight stdlib CPU review only. No framework,
tokenizer, model, GPU operation, protected bank or unrelated process was used.
Low host RAM and native whole-path/model-dispatch quarantine remain gates.
Shared workspace/model ancestry is disclosed; agreement is not ground truth.

**Disposition: REPAIR_REQUIRED / NOT_ADMITTED.** This report does not approve
a reader qualification launch or serving. A new namespace must not bypass
the quarantined owned-runtime identity/supervision dependency. The distinct
bounded no-model identity diagnostic is a resolving experiment, not permission
for another unchanged model run or whitelist expansion.

## Initial frozen artifacts inspected directly

| Artifact | SHA256 |
| --- | --- |
| `reader.py` | `804110837c260d7dcb67860ca0bdf2a49fd01b1afb7428d2d0e46331335017fd` |
| `reader_execution.py` | `e83984c8ce3d25d44558643b9e3d3440b8706c94f331c6457275aebb18bbd193` |
| `tests/test_reader.py` | `480aa79762bdad1a1a9c936b2c693866be53ee989846bba998a76a71dd9cd85a` |
| `contracts/READER_QUALIFICATION_V1.json` | `f53a116d171730bf5ecc5cb6d23dfd6ec38fa879c8b5b1d01785d2b0124998b6` |
| imported native helper subject | `8453128981280019702dff80ebd5083031f7f2dcd8a7c6c9094587d9a2e377a3` |

Read actual policy/session/worker/validator/supervisor/publication code and
contract before builder interpretation. During review the builder began fixes;
later working hashes differed. This report's initial findings concern the
frozen subjects above. New bytes require a fresh exact binding and targeted
closure; no builder file was edited by this reviewer.

## Executed lightweight evidence

The supplied12 reader tests pass on this reviewer invocation:12/12,0 failures,
0 errors,0.010s. A reviewer-authored temporary-file counterexample under the CRS
results directory built a clearly synthetic, unexecuted hash-valid bundle;
`verify_qualification` accepted it as PASS_EXACT_PREFIX_SERVING_576_SCOPED.
The bundle was never dispatched, saved as native evidence or used to open a
model. Another direct stdlib probe showed an empty reviewer string was accepted.
Temporary files were cleaned; torch/transformers/tokenizers were absent from
sys.modules throughout these probes. Tests demonstrate these narrow mechanics,
not actual model likelihoods, EOS behavior or resource fit.

## Findings

### R-OPS-001 [P1]: hash-valid status-only dependencies can open serving

**Evidence:** Initial `reader.verify_qualification` checks top-level status,
binding and flags, then validates dependency hashes and a few status labels.
It does not replay the complete qualifier result/durable journal, full source/
model/runtime/tokenizer binding, registered blocks, causal controls, or complete
supervisor outcome observation.

**Reproducer executed:** A worker file containing only COMPLETE worker status
and an arbitrary expected binding; supervisor file containing SUCCEEDED,exit0
and owned_tree_exited=true; independent review containing APPROVED status and
that binding; and top-level PASS scope/flags. With actual hashes of these files,
the consumer accepts the bundle, despite zero source qualification/forwards.
Bare caller PASS dictionaries reject, but wrapping claims in hash-valid files
does not turn them into execution evidence.

**Impact:** The serving boundary can consume incomplete or fabricated caller
qualification artifacts. Hash association alone validates bytes, not their
meaning or model identity. Same-user coherent forgery is outside an OS secrecy
claim; ordinary malformed/incomplete bundles still need deterministic rejection.

**Verification:** Require canonical expected bindings derived from current
reviewed contract/source/model/runtime identities, complete immutable worker/
journal/completion/supervisor/observation/tokenizer/runtime/model dependencies,
and re-run their independent validations before admitting serving. Match the
same job/owner/fence/source and complete scheduled controls, reject minimal
bundles, wrong dependency kinds, stale source, missing events and drift. Keep
underlying quarantine and serving registration closed independently of receipts.

### R-OPS-002 [P1]: original qualifier completion is not tied to registered forward geometry

**Evidence:** Initial completion validator counts journal forwards and measured
records and checks per-record prefix self-hashes, mask hash and position IDs.
It does not associate each measured record with its operation call/stage or
the exact scheduled input/retention shape. Saved natural rollouts, parity
blocks, forced prefixes and max-shape causal results are not canonically bound
to their durable measurements. Blocks need only six COMPLETE labels; causal
records need only three equal left/right strings.

**Counterexample from predicates:** A numeric accounting ledger can report the
expected aggregate positions while unrelated measured records carry length1,
six minimal COMPLETE blocks omit actual counts/IDs, and three repeated equal
non-hash causal strings satisfy the initial terminal checks. Aggregate ceilings
alone do not prove the512..576 shape ladder, owned/stock/replay routing or
protected-position tests executed. A reviewer fixture prepared to exercise this
old interface was rejected after concurrent builder changes introduced exact
binding checks; no claim is made that the new unpublished helper passed it.

**Impact:** Corrupt/missing/misassigned qualification evidence can appear
complete without proving the new serving envelope or controls.

**Verification:** Freeze a complete forward plan and bind every attempt/return/
measurement to exact call ID, stage, token prefix/hash, input length, retention
keep/shape, causal/position/hidden witnesses and corresponding saved rollout.
Require exact block identities/counts and canonical journal/result association;
reject missing/duplicated observations, altered prefix/keep, reordered stages,
wrong causal positions, nonhex hashes, bool counts and mismatched scope. Any
new external completion helper must not inherit qualification from D2 labels.

### R-OPS-003 [P2]: empty reviewer identity is accepted

**Evidence/reproducer:** Initial `review_matches` requires a string whose final
slash component is not runtime_audit, but does not require nonempty text.
`reviewer=''` passes with matching APPROVED/binding fields. Similar dependency
review checks are weaker than the normalized native boundary.

**Impact:** An unidentified source can be represented as independent approval.
A builder-written status does not grant authority or ground truth.

**Verification:** Require nonempty normalized identified reviewer, reject short/
canonical/backslash builder identity and missing/blank labels, and bind the
actual independent artifact to exact source/contract/model/runtime. This is
logical provenance validation, not same-user identity authentication.

### R-OPS-004 [P2]: publication prerequisites are checked after model work and job success

**Evidence:** Initial wrapper calls `_supervise_reader_worker` first. That inner
path records SUCCEEDED, reconciles cost and releases the lease before the outer
wrapper verifies that independent review is retained under results and writes
the reusable qualification receipt.

**Impact/counterexample:** An otherwise accepted review path outside results,
or a disk/publication error, can fail final publication only after a model job
has been marked successful. Restart with the same terminal key cannot reproduce
the missing final receipt; treating a new key as a fresh model retry hides the
publication problem. No such model launch was performed in this review.

**Verification:** Validate all path/reference/publication prerequisites before
reservation/dispatch. Separate qualified worker success from complete published
serving qualification, journal durable publication status/idempotency, and
reconcile a lost write from existing verified artifacts without another model
run. Preserve all failure costs, partials and the unqualified outer outcome.

### R-OPS-005 [P1 admission dependency]: reader must inherit quarantine

**Evidence:** The coordinator retained native attempt004 failure from the same
final source at framework import (`unregistered owned runtime executable`),
with current cumulative47.359 native seconds and no active worker. Native
whole-path/model dispatch is now quarantined at its three-round limit. The
reader imports those runtime identity/containment primitives and copies that
supervisor path; an independent new scope does not resolve the underlying fault.

**Impact:** READY reader preflight based only on resources, exact code and its
own review could bypass an unresolved host/runtime identity dependency.

**Verification:** Admission must check the persistent quarantined dependency
and exact disposition/resolving evidence before any reader/model dispatch.
Retain current blocked RAM/GPU identities and don't terminate user apps or
expand registry to fit. A bounded separately reviewed no-model identity probe
may diagnose the dependency; its success cannot itself license reader/model
execution or count as another native repair round.

## Schedule, EOS/reference and serving considerations

The six-block upper schedule arithmetic is correct:278 forwards,146,316 summed
input positions,590 construct/load/copy/forward spans and134 natural owned+
stock emissions. For long native emission countN, the planned forward total
is86+3N; positions are41900+N+3*sum(512+i,i=0..N−1). Forced65 prefixes cover
512..576 and are correctly labeled non-emitting mechanics. A serving cap64
from prompt512 uses prefix575 for its final prediction, not an extra emitted
token at576. Prompt EOS is not an output stop; sampled EOS at cap takes
precedence in the CPU policy tests.

Actual stock generation instrumentation, tokenizer/template controls, raw
argmax tie behavior, history reference likelihood and maximum-shape causal
evidence still need exact native qualification; none is established by fake
CPU callbacks. The old bulk likelihood gate remains failed/closed, and the
new full-prefix deterministic path must not silently qualify bulk SFT/RL.
Natural EOS observation in one trace is not a claim that every possible EOS/
input/status task has been tested. Limits and eligible scopes must remain clear.

ReaderSession uses the same278 qualification-call ceiling for forward/serve.
Serving after a worst-case qualification exhausts it; serving/new-instance
worker budgets, capture costs and immutable model loading need a separate
registered interface rather than an undocumented reset. The CLI currently
denies serve, which correctly leaves that work pending. A caller-provided
ReaderSession binding alone does not attest a different/mutated model object.

The supervisor's copied ordering preserves suspended spawn, exact Job identity,
query-handle closure, fencing, actual process/GPU telemetry, unknown launch
retention, independent completion validation and cumulative budget shape.
Copying reviewed primitives is useful infrastructure; new worker/publication/
scope logic needs its own failure/restart/immutable-receipt tests. The initial12
tests do not exercise those full adapters. No native3 code change or fourth
native retry is approved here.

## Next bounded action

Freeze the reader repair candidate and independent scientific contract, add
lightweight minimal-bundle/missing-journal/source/model/tokenizer/EOS/block/
restart/publication/quarantine negatives, then obtain fresh targeted review.
Resolve the owned-runtime identity issue through its separately registered
no-model diagnostic without granting a model bypass. Keep reader/model dispatch
closed while prerequisite evidence or measured host admission is absent.

Machine-readable initial review equivalent:

```json
{
  "review_id": "CRS-READER-OPS-REVIEW-001",
  "reviewer": "/root/data_audit",
  "builder": "/root/runtime_audit",
  "status": "REPAIR_REQUIRED_NOT_ADMITTED",
  "reader_source_sha256": "804110837c260d7dcb67860ca0bdf2a49fd01b1afb7428d2d0e46331335017fd",
  "execution_source_sha256": "e83984c8ce3d25d44558643b9e3d3440b8706c94f331c6457275aebb18bbd193",
  "contract_sha256": "f53a116d171730bf5ecc5cb6d23dfd6ec38fa879c8b5b1d01785d2b0124998b6",
  "initial_tests_passed": 12,
  "findings": ["R-OPS-001", "R-OPS-002", "R-OPS-003", "R-OPS-004", "R-OPS-005"],
  "quarantine_dependency_resolved": false,
  "native_model_dispatch_approved": false,
  "heavy_imports_by_reviewer": false,
  "model_calls_by_reviewer": 0,
  "gpu_calls_by_reviewer": 0,
  "protected_banks_opened": false,
  "learning_delta_self_approval": false
}
```

# CRS-001 final native operational repair round 3: targeted science review

Recorded 2026-10-02, America/Chicago. Reviewer `/root/scientific_audit`, separate
from builder `/root/runtime_audit`. Initial disposition on the proposed freeze:
**REPAIR REQUIRED; SUPERVISION PROMOTION QUARANTINED** pending the concrete
completion-binding closure below. This is the third/final milestone repair round,
not permission for unlimited additional native retries.

Only this managed worktree's native revision is reviewed. The primary CPU
lifecycle/instrument and previous native evidence/reviews are preserved. No model
was launched/reloaded by this reviewer, no protected gold/new dataset was used,
and no unrelated process was terminated. Reviewers share model lineage and host
privileges; this is a distinct raw-artifact inspection, not an independent
hardware attestation.

## Proposed frozen identities

Paths resolve within the CRS package in
`C:/Users/clclc/.codex/worktrees/crs-native-desktop-v2/ChatBubble-rineya-centralized-reasoning-system-v1`.

| Artifact | Initial round-3 SHA256 |
| --- | --- |
| `native.py` | `c6f6e52792bf86d6a865a1321ac30c3f6b9bfdd11a3a18efb4e4c74119bf5988` |
| `tests/test_native.py` | `6473cf5b830d5cff546a8a816d81a8e91c4a4aebf4dba855c73b1755399f0fb9` |
| `contracts/NATIVE_DIAGNOSTIC_V1.json` | `d18f6429dcffcce23d1b2af0dd3221974398c1bd837c920c5a41c1e7fe4a5714` |
| `NATIVE_OPERATIONAL_AMENDMENT_003.md` | `810ab3350402e272f6ebd4e58a0128b895139fd4cc3794c51c99f4b2bac48b74` |

The original revision-2 contract is preserved under
`contracts/history/NATIVE_DIAGNOSTIC_V1_291b7803.json`. Original 17c01965 worker
observation and FAILED supervisor status/cost are independently reviewed in
[NATIVE_EVIDENCE_REVIEW_1.md](NATIVE_EVIDENCE_REVIEW_1.md). They must not be
rewritten as a complete-path success after a validator repair.

## Targeted design review

The proposed repair replaces `parent.children()` traversal with exact owned Job
membership and explicit process identity observations. A vanished member requires
a fresh Job list showing absence plus original-process exit or PID-reuse evidence.
Live access-denied/unknown telemetry is not treated as successful completion.
After exit zero, the supervisor stops/joins its telemetry thread, retains live
telemetry errors, requires no remaining owned descendant, validates complete
current-source worker/journal evidence and rechecks source/assets before success.
These conditions preserve the intended numerical question and failure accounting.

Read the exact diff against committed 17c01965. Worker code from its committed
source snapshot through all model/Torch/math checks and final result/cost
publication is unchanged. New logic is supervisor sampling/exit reconciliation
and external completion validation. The contract retains all model/config/
tokenizer/runtime, four-forward/524-position, bitwise, baseline, original
0.03125/0.01 failed-capture, mask/position/precision, 600-second and RAM/GPU/disk
limits. Reader576, capture, training and efficacy remain closed. New source/
test/revision/evidence metadata does not transfer old approval to a new commit.

The added validator reconstructs operation counters from durable EVENTS through
the pinned central checker, rejects failed/pending/boolean counters, requires
complete resource-journal/result agreement and six valid raw comparisons plus
four mask/rotary witnesses. Those are meaningful checks, but the missing
measurement-to-trial binding below still allows incomplete or contradictory
raw numerical evidence to pass its promotion boundary.

## NATIVE-SCI-3A — Durable measurement/trial coverage is not enforced

**Priority P1 before diagnostic supervisor acceptance. Evidence:** proposed
`validate_diagnostic_completion` reconstructs operations and joins resources,
but does not require or join the four `kind=measurement` journal payloads to
`result.trials`. Nonbaseline mutation identities/full-output fingerprints are
also not validated by its trial loop.

**Bounded independent reproducer, no model rerun:** load retained actual
`results/d2_17c01965_epoch001_rev2/DIAGNOSTIC_RESULT.json` and `EVENTS.jsonl` into
memory. Preserve the raw files unchanged.

1. Remove all four measurement events from the in-memory journal, retaining
   operation and resource records. Call the validator with original result,
   current cfg and source_commit=17c01965. Observed acceptance still reports
   `durable_journal_verified=true` and `trial_witnesses_verified=4`, with 644
   remaining events.
2. Keep the real full journal, but change nonbaseline trial 1 to mutation_start
   999, protected_positions `[999]` and logits_sha256 `not-a-hash` in a copied
   result. Observed acceptance. The altered saved trial contradicts the real
   measurement payload and registered intervention.

**Impact:** result status and operation totals can coexist with absent raw
measurement records or mismatched trial identity, yet the external boundary
claims complete numerical witness verification. This is a completion instrument
gap, not evidence that the retained physical comparisons were false: the
independent actual-evidence audit joined the original four payloads correctly.

**Verification within this final repair round:** require exactly four durable
measurement payloads equal the four saved trials; reject missing, extra or
altered payloads. Validate registered mutation starts/protected absolute positions,
valid full logits/hidden SHA256s and three finite nonpositive target logprobs.
Retain baseline reference joins and all existing operation/resource/comparison/
mask/position checks. Add missing-measurement and corrupted-trial negative controls
against copied real receipts. This can be a supervisor-only change; the worker's
model/math body and dose need not change or be rerun to test the validator.

If this remains unresolved after the allowed repair round, quarantine the whole
supervision claim and record the resolving bounded experiment; do not label
disappearance or old worker COMPLETE as a newly qualified path.

## Evidence boundaries and retained costs

The old run has valid recorded worker facts: exact Titan load, four returned
131-token forwards/524 positions, 316 closed calls and six protected raw-byte
equalities. Its supervisor remains FAILED for NoSuchProcess on exit despite
exit zero/empty tree. Actual worker/scientific evidence is retained separately
from operational promotion.

34.56199999999808 seconds from that run plus the earlier 1.1090000000003783
seconds remain 35.67099999999846 charged qualification seconds. The proposed
revision neither resets cost nor changes original capture failure into success.
Any future admitted validator test must use the new committed source/contract,
fresh live gates and remaining budgets. No native acquisition, learning,
retention, calibration, GEMM mechanism or procedure-memory effect is inferred.

## Final targeted closure within repair round 3

NATIVE-SCI-3A is closed on the following repaired freeze:

| Artifact | Accepted final SHA256 |
| --- | --- |
| `native.py` | `8453128981280019702dff80ebd5083031f7f2dcd8a7c6c9094587d9a2e377a3` |
| `tests/test_native.py` | `bf0bd76f1d93281dd2a4cbf4581bc30a481e4ae9c4e5d822f3f7e58dbf223067` |
| `contracts/NATIVE_DIAGNOSTIC_V1.json` | `d79f3b6a51e939d78a99c5c287ef6f33ef5979cfa50e76ba3d930ed3724f826a` |

The completion boundary now requires exactly four durable measurement records,
with exact measurement event schema and canonical byte-equivalent payloads to
the saved trial list. Every trial has exact schema, registered stage/mutation
identity, integer protected positions, full valid lowercase SHA256s and three
finite nonpositive original-target logprobs. Canonical comparison distinguishes
integer from float mutation values. Existing resource, operation, mask, position,
baseline and six-comparison checks remain in force.

Independently replayed all original probes against copied retained evidence:
missing all four measurements rejects; corrupted nonbaseline start/positions/
full hash against unchanged real EVENTS rejects; coherently changing both result
and matching measurement payloads still rejects the unregistered identity/hash.
The original unmodified retained worker result/journal validates without a model
reload. Duplicate measurements and other typed/nonfinite corruptions also reject
in the regression suite. No saved failure or worker evidence was modified.

Fresh registered-interpreter test observations: targeted actual-evidence validator
test passed, then **31/31 native CPU controls passed**, reported suite interval
6.383 seconds, with no skips/errors or model execution. These include actual
bounded stdlib Windows probes and a naturally exiting CPU worker whose missing
DIAGNOSTIC_RESULT correctly prevents success despite exit zero/empty tree. The
contract preserves earlier failed CPU repair validations and their causes;
retesting does not erase them or lower thresholds.

Independent AST comparison confirms the complete worker tail beginning its
source snapshot through model/Torch/math/scoring/finalization is unchanged from
committed 17c01965. The protected contract fields are equal to that commit:
assets/runtime/source environment, input/generated IDs, reference endpoints,
causal rule, four-forward/524-position/316-call dose, original failed-capture
tolerances, resource floors/caps, GPU registry and strict owned-Job identity.
This closes external evidence admission; it changes no model or numerical test.

## Final scientific disposition

**SCIENTIFICALLY_APPROVED_FOR_BOUNDED_DIAGNOSTIC**, conditioned on independent
operations acceptance of identical bytes, a clean new commit/exact attestation,
fresh source/asset/runtime/process/GPU admission, one fenced worker and remaining
epoch/rolling budgets. No consequential scientific completion-binding issue
remains for this frozen four-forward diagnostic. This is the targeted closure
of the final existing repair round, not a fourth full review round.

The previous 17c01965 whole path remains FAILED and its actual worker observations
stay separately supported. Its 34.562 seconds plus prior 1.109 seconds remain
35.671 charged qualification seconds. A new source/version may receive one
bounded admitted supervisor validation; it cannot inherit success from the
retained old result or rewrite the failure.

Capture, reader576, training/serving/export/resume, efficacy, learning/retention,
calibration, protected-bank access and mechanism claims remain unqualified.
Only the registered trace-local same-shape protected-row invariance may be
reported after a valid future complete path. Any material operational issue
remaining after this final milestone round must be quarantined with a bounded
resolving experiment instead of automatic repeated model runs.

# CRS native operational review 3: verified exit and complete finalization

Recorded 2026-10-02, America/Chicago. Reviewer `/root/data_audit`, distinct from
builder `/root/runtime_audit`. This is the third and final operational review
round for the milestone, confined to the isolated managed worktree. Shared
workspace/model ancestry is disclosed. No primary source, model asset, protected
bank, original result or user application was changed by this reviewer. No new
model/framework/GPU execution was performed; bounded owned stdlib process
fixtures and read-only retained native evidence were used.

**Final decision: APPROVED_FOR_BOUNDED_DIAGNOSTIC**, conditional on a clean
successor commit, source-bound scientific approval and fresh host/lease/budget
admission. Approval is for operational revision3 of the same four-forward
diagnostic. It is not a complete historical supervisor pass, broad inference,
capture, Reader576, training, model-quality or efficacy qualification.
No material operational finding remains open in the frozen candidate below.
Further unresolved material faults would quarantine this milestone; this report
does not create an additional automatic review/retry round.

## Frozen artifact bindings

| Artifact | SHA256 |
| --- | --- |
| `native.py` | `c6f6e52792bf86d6a865a1321ac30c3f6b9bfdd11a3a18efb4e4c74119bf5988` |
| `tests/test_native.py` | `6473cf5b830d5cff546a8a816d81a8e91c4a4aebf4dba855c73b1755399f0fb9` |
| successor diagnostic contract | `d18f6429dcffcce23d1b2af0dd3221974398c1bd837c920c5a41c1e7fe4a5714` |
| `NATIVE_OPERATIONAL_AMENDMENT_003.md` | `810ab3350402e272f6ebd4e58a0128b895139fd4cc3794c51c99f4b2bac48b74` |
| unchanged operations module | `ba27ec8a31437459b956a5642e6ece4838f0584602db2c93ff7bc61437806e30` |
| preserved revision2 contract history | `291b7803e745ade2ccfcc5a6e9a8eafc31a9c174541ccd5436f730bde57a8707` |

## Raw evidence inspected before repair narrative

Original directory `results/d2_17c01965_epoch001_rev2` was read first: supervisor
result, worker result, containment receipt, stderr, durable event journal and
exit-proof receipt. The original worker source commit is
`17c01965b048a130f82095e2177d626449d0de4a`.

| Immutable observation | Evidence |
| --- | --- |
| Worker COMPLETE_FIXED_SHAPE_DIAGNOSTIC_ONLY | DIAGNOSTIC_RESULT SHA256 `b9a09927bf34e1cd2bcea0dacb9444b5e0508a4477b75ff15ae408469f1c3337` |
| Whole supervisor FAILED, exit0, reason NoSuchProcess(pid626604) | SUPERVISOR_RESULT SHA256 `c2f90ce741a6dbc9f32dc27837a6b3f5242daf7645f9f8b2d2a1b181fff5bffc` |
| Owned Job empty-tree exit, measured34.56199999999808s | Native-outcome receipt SHA256 `e5bb53d7054c70bbbfcfa8fc9df500573f355aa6d16930eb1e39336012fc924f` |

The worker loaded the exact Titan once, returned all four131-token forwards,
accounted316 construct/load/copy/forward calls and524 positions, with no failed
or pending calls, no generation and no optimizer update. Six protected
logit/hidden raw-byte comparisons were identical with numerical max/mean0.
Worker wall time31.718s differs legitimately from full supervisor worker-cost
time34.562s. Peak process-tree RSS was1,462,730,752 bytes. These are actual
retained measurements, not CPU fixtures; numerical truth/scientific scope is
reviewed separately.

The supervisor's NoSuchProcess reason concerns a normal-exit monitoring race;
the original reason record does not retain a full supervisor traceback.
Revision2 source used cached launcher/children traversal. The old WHOLE-PATH
FAILED classification is retained even though useful native component evidence
exists. This review does not rewrite it as success or as a negative model result.
The earlier1.109s plus34.562s preserve **35.671 native qualification seconds**
in cumulative accounting. No revision resets those costs.

## Consequential findings and resolving controls

### NATIVE-O3-001: cached launcher traversal fails during valid terminal exit

**Evidence/impact:** The recorded supervisor reason is NoSuchProcess626604;
worker exit was0 and the complete owned tree was subsequently empty. Root or
child disappearance between process.poll and psutil traversal can invalidate a
monitor sample. Ignoring every NoSuchProcess would also hide a genuinely live
worker with missing telemetry.

**Resolution:** Monitoring now obtains PIDs from the exact owned Job rather than
cached parent.children. A disappeared sampled member requires refreshed exact
Job absence plus OS evidence of exit or a different creation identity after PID
reuse. AccessDenied and other live errors still propagate. After normal exit,
the supervisor stops/joins the GPU collector, retains any live collector error,
requires exit0 and an empty owned Job, then validates complete worker evidence
and current source/assets before success. An empty Job alone is never success.

**Independent tests:** Verified absent/exited PID is accepted; a PID still
associated with the Job or the same live creation identity rejects. The actual
bounded stdlib natural-exit fixture exits0/empty-tree without NoSuchProcess,
but correctly remains FAILED because no real diagnostic result exists. Live
AccessDenied remains an error. These controls do not create synthetic native
success or remove failure costs.

### NATIVE-O3-002: early completion validation accepted malformed evidence

**Reviewer reproducers during this final round:** On an in-memory copy of the
retained real result, boolean model-construct counters were accepted as1,
nonhex `X` repeated64 times was accepted as an equal raw hash, and mask shape
1×1×1×1 was accepted while the validator claimed verified trial witnesses.
The original result file was never changed.

**Impact:** Disappearance recovery could promote a superficially complete but
corrupt artifact. A COMPLETE label or equal hash-shaped strings are insufficient.

**Resolution/final verification:** The final validator rejects those exact
copied-result corruptions, plus changed mask representation, NaN comparison and
stale commit. It enforces strict integer counts, finite numeric zero errors,
valid lowercase SHA256, full expected causal-mask dtype/shape/representation/
actual-match flag, and complete rotary-position dtype/shape/IDs/hash. Six
distinct comparison identities and all four ordered trial witnesses are required.
No scientific tolerance was widened.

### NATIVE-O3-003: final success must bind durable journal and complete resources

**Evidence/impact:** Worker-result counters alone do not prove every attempt,
return or resource-completion record survived publication. The local event
schema uses kind/time; the pinned central checker expects event and its own
optional timing field. Calling it without an explicit adapter is invalid.

**Resolution:** The final validator uses the pinned independent journal checker
with an explicit kind→event adapter and excludes local wall-time fields from
the count-validator input while preserving the raw timestamps. It verifies
monotonic finite recorded times, result/journal counter equality and the complete
resource event vector ending at stage complete. A missing operation return,
trailing/partial journal or missing final resource record rejects success.
The supervisor writes a separate completion-verification receipt with the
durable event SHA256; source/assets and clean-source checks still follow.

**Independent verification:** Reconstructed the original actual journal before
and after repair:648 raw records,316 closed calls,4 returned forwards/524
positions, no pending calls. Final validation of that retained component passes
with the explicitly supplied original source commit. This offline result does
not change its original FAILED supervisor status. Copying counts or dropping a
record cannot produce a qualifying successor result.

## Executed final-round verification

Ran31 frozen native CPU tests using the pinned interpreter, with reviewer-
enforced temporary directories inside the isolated CRS package: **31 passed,
zero failures/errors/skips**,6.056s on this reviewer invocation. The builder's
5.656s is a separate invocation. No model/framework/GPU operations occurred.
Actual owned process fixtures exercise suspended/exact-Job containment, query
handle closure, last-handle tree termination and natural terminal disappearance.
Original lost-Popen/full-reservation, no-dispatch, live resource, source drift,
numerical/reader/training denial controls remain included.

Fresh independent copied-result/exit controls produced:

| Control | Outcome |
| --- | --- |
| Actual retained component+648-event journal | Complete316-call/4-forward/6-comparison/4-trial verification |
| Boolean counters | Rejected as journal/snapshot mismatch |
| Nonhex raw hashes | Rejected |
| Corrupt mask shape or compact representation | Rejected |
| NaN comparison | Rejected |
| Stale source commit | Rejected |
| Refreshed Job absent and OS process absent | Confirmed exit |
| PID still in exact Job | Rejected |
| Same creation identity still live | Rejected |

## Unchanged body, limits and launch boundary

Independent AST comparison against committed17c01965 confirms identical
`worker`, `causal_plan`, `numerical_decision`, `causal_mask_reference_bytes` and
`checked_contract` functions. Direct contract comparison confirms identical
model assets, token trace, four-forward/524-position dose, generation0,
optimizer0, historical tolerances, GPU registry and resource limits.
Source/model/tokenizer/runtime pins, BF16/eager/cache-off math, bitwise causal
criteria, exact owned Job/query closure, fencing and cumulative budgets are
preserved. New helpers validate completion and handle verified exit races;
they do not repair the historical prefix-versus-bulk likelihood discrepancy.

This final review approves one registered bounded validation under the owner's
existing authorization once the unchanged candidate is cleanly committed and
its review/source/scientific/host bindings are refreshed. Missing live telemetry,
OOM/NaN, source/result corruption, failed/pending calls or unknown tree exit
continue to fail closed. Reader576, capture, training/export/resume, procedure
efficacy, acquisition/retention and protected banks remain separate gates.
If a material remaining supervision issue appears, retain raw evidence and
quarantine the claim rather than start an unbounded fourth repair loop.

## Machine-readable review equivalent before commit binding

```json
{
  "review_id": "CRS-NATIVE-OPERATIONAL-REVIEW-003",
  "reviewer": "/root/data_audit",
  "builder": "/root/runtime_audit",
  "client_date": "2026-10-02",
  "status": "APPROVED_FOR_BOUNDED_DIAGNOSTIC",
  "operational_revision": 3,
  "native_source_sha256": "c6f6e52792bf86d6a865a1321ac30c3f6b9bfdd11a3a18efb4e4c74119bf5988",
  "tests_sha256": "6473cf5b830d5cff546a8a816d81a8e91c4a4aebf4dba855c73b1755399f0fb9",
  "contract_sha256": "d18f6429dcffcce23d1b2af0dd3221974398c1bd837c920c5a41c1e7fe4a5714",
  "amendment_sha256": "810ab3350402e272f6ebd4e58a0128b895139fd4cc3794c51c99f4b2bac48b74",
  "source_commit": null,
  "clean_commit_binding_required": true,
  "separate_source_bound_scientific_approval_required": true,
  "tests_passed": 31,
  "tests_failed": 0,
  "tests_errors": 0,
  "tests_skipped": 0,
  "verified_exit_race_control": "PASS",
  "live_missing_telemetry_rejection": "PASS",
  "corrupt_result_and_journal_rejection": "PASS",
  "actual_prior_component_evidence_preserved": true,
  "prior_whole_path_failed_status_preserved": true,
  "prior_native_worker_seconds_preserved": 35.671,
  "native_calls_by_reviewer": 0,
  "gpu_calls_by_reviewer": 0,
  "reader_qualified": false,
  "capture_qualified": false,
  "training_qualified": false,
  "authority_expanded": false,
  "fresh_host_lease_and_budget_admission_required": true,
  "further_material_fault_policy": "retain evidence and quarantine; no automatic fourth repair loop"
}
```

## Final updated closure within the same third round

The c6f6e527/d18f6429 acceptance above is retained as exact-byte review history;
it was **not commit-attested or dispatched**. The independent scientific reviewer
then reproduced a consequential completion-evidence gap: all four measurement
events could be missing, and nonbaseline trial mutation/positions/full hash
could disagree with the durable journal without rejecting completion. Those
findings kept final closure pending; they did not create an automatic fourth
review round or modify the worker's actual numerical evidence.

The final candidate supersedes that pending source binding:

| Final updated artifact | SHA256 |
| --- | --- |
| `native.py` | `8453128981280019702dff80ebd5083031f7f2dcd8a7c6c9094587d9a2e377a3` |
| `tests/test_native.py` | `bf0bd76f1d93281dd2a4cbf4581bc30a481e4ae9c4e5d822f3f7e58dbf223067` |
| diagnostic contract | `d79f3b6a51e939d78a99c5c287ef6f33ef5979cfa50e76ba3d930ed3724f826a` |
| unchanged amendment003 | `810ab3350402e272f6ebd4e58a0128b895139fd4cc3794c51c99f4b2bac48b74` |

**Final updated decision: APPROVED_FOR_BOUNDED_DIAGNOSTIC** for these bytes,
subject to their clean successor commit and matching scientific/live admission
bindings. The earlier pending candidate must not supply that launch binding.

The completion verifier now requires exactly four durable measurement events,
with complete canonical payloads equal to the four saved trials. Independently
of that association it enforces each registered stage, mutation start and exact
protected-position list, two valid full-trial SHA256 strings and three finite
nonpositive model log-probabilities. Missing/duplicate/mismatched measurement
delivery cannot count as complete. Coherently editing both trial and event to
an invalid registered identity also rejects. This is consistency/qualification
within the pinned code contract, not a claim of OS attestation against coherent
same-user forgery of every artifact.

Executed the updated31-test suite using the pinned interpreter and local
worktree temporary directories: **31 passed, zero failures/errors/skips**,
6.596s on this reviewer invocation, zero model/framework/GPU calls. The builder
reports a separate preceding bounded CPU probe timeout/temporary-cleanup failure
retained in its validation history; the repaired registered limits and source
probe were not relaxed to erase that failure.

Independent final controls rejected: all measurement events removed; an
in-order duplicate measurement event; a result/event mutation mismatch; and
coherent invalid full-hash, protected-position, mutation-start or positive
log-probability changes. The actual retained648-event component still validates
with its original source commit, without rewriting the original whole-path
FAILED result. Repeat AST comparison confirms the complete worker and
causal-plan/numerical/mask/limit functions remain identical to17c01965.
Costs35.671s and all predecessor results stay preserved.

No material unresolved finding remains for the updated registered diagnostic
entry/finalization path. No model rerun, reader/training/capture promotion,
protected-bank access or permission expansion occurred in this review.
The final third-round quarantine rule remains applicable to further material
faults after the bounded validation.

Final machine-readable equivalent before successor commit attestation:

```json
{
  "review_id": "CRS-NATIVE-OPERATIONAL-REVIEW-003-FINAL-CLOSURE",
  "reviewer": "/root/data_audit",
  "builder": "/root/runtime_audit",
  "status": "APPROVED_FOR_BOUNDED_DIAGNOSTIC",
  "operational_revision": 3,
  "same_final_review_round": true,
  "native_source_sha256": "8453128981280019702dff80ebd5083031f7f2dcd8a7c6c9094587d9a2e377a3",
  "tests_sha256": "bf0bd76f1d93281dd2a4cbf4581bc30a481e4ae9c4e5d822f3f7e58dbf223067",
  "contract_sha256": "d79f3b6a51e939d78a99c5c287ef6f33ef5979cfa50e76ba3d930ed3724f826a",
  "amendment_sha256": "810ab3350402e272f6ebd4e58a0128b895139fd4cc3794c51c99f4b2bac48b74",
  "source_commit": null,
  "clean_commit_and_scientific_binding_required": true,
  "tests_passed": 31,
  "tests_failed": 0,
  "tests_errors": 0,
  "tests_skipped": 0,
  "measurement_event_and_trial_binding": "PASS",
  "missing_duplicate_corrupt_measurement_rejections": "PASS",
  "unchanged_worker_and_numerical_body": true,
  "prior_native_cost_seconds_preserved": 35.671,
  "prior_whole_path_failed_status_preserved": true,
  "model_calls_by_reviewer": 0,
  "gpu_calls_by_reviewer": 0,
  "reader_qualified": false,
  "capture_qualified": false,
  "training_qualified": false,
  "authority_expanded": false,
  "fresh_host_lease_and_budget_admission_required": true,
  "further_material_fault_policy": "quarantine rather than an automatic fourth repair loop"
}
```

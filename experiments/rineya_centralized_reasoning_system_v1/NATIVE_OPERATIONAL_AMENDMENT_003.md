# Native operational amendment 003: exact Job exit reconciliation

Recorded 2026-10-02, America/Chicago. Builder `/root/runtime_audit`;
base `17c01965b048a130f82095e2177d626449d0de4a`. This is the third and final
operational repair round for this milestone. Material unresolved findings after
targeted review quarantine the supervision claim; there is no automatic retry.
Only the isolated managed worktree's native source/test/contract and this versioned
record change. The primary checkout, weights, operations and earlier evidence
remain untouched. Existing owner authority is unchanged.

Operational revision 2 is preserved create-only at
`contracts/history/NATIVE_DIAGNOSTIC_V1_291b7803.json`, SHA256
`291b7803e745ade2ccfcc5a6e9a8eafc31a9c174541ccd5436f730bde57a8707`.
The original a57 contract history is also retained.

The actual run `results/d2_17c01965_epoch001_rev2` loaded the exact Titan and
completed four fixed 131-token forwards: 316 closed model/load/copy calls,
524 input positions, six protected raw-byte comparisons all identical, and
unchanged-base checks. Its worker status is
`COMPLETE_FIXED_SHAPE_DIAGNOSTIC_ONLY`. Torch peaks were 1.400038 GiB allocated
and 1.4296875 GiB reserved; sampled minimum GPU free memory was 5.404297 GiB.
These are actual native measurements, distinct from CPU fixture results.

The supervisor still failed with `NoSuchProcess(pid=626604)` on a parent-process
enumeration exit race, despite exit code zero and a verified empty owned Job.
The original FAILED supervisor result and complete raw worker evidence remain
unchanged. Its 34.562 worker seconds plus the earlier 1.109 seconds leave
**35.671 native qualification worker seconds charged**. Neither a failure nor
this repair erases cost or turns the old supervisor failure into success.

RAM and GPU monitoring now obtain members from the exact owned Job, avoiding
`parent.children()` after the launcher exits. If a sampled member disappears,
the monitor requires a refreshed exact Job list showing absence and OS evidence
that the recorded process exited or its PID was reused. A live member with
unavailable telemetry remains a failure. Disappearance is never success by
itself. GPU observation errors begun while a worker is live remain failures,
including during shutdown. No OOM, NaN, failed or pending model call is swallowed.

After zero exit, the supervisor stops and joins its telemetry thread, requires no
live-telemetry error and no remaining owned process, then independently validates
the complete worker result. This includes strict integer operation counters,
no pending/failed calls, exact 4-forward/524-position coverage, six valid raw
SHA256 comparisons, and four complete causal-mask/rotary-position witnesses.
The pinned central journal checker independently reconstructs counters from
durable local events through an explicit `kind` to `event` adapter. Raw wall times
remain original evidence. Journal/result mismatches, missing operation returns
and missing final resource-completion records reject promotion.
Only after final source/assets/clean-commit checks may it record success.

CPU controls force a natural subprocess exit between loop entry and sampling;
the supervisor retains exit zero and the empty-tree proof but rejects the missing
diagnostic result. Additional controls reject unproven disappearance, live
access-denied telemetry, pending calls, booleans masquerading as counts, invalid
hashes, NaN and corrupt witness shapes/positions. The real retained native receipt
is read-only evidence for validator tests; no synthetic native success is created.
**31 CPU tests pass**. No model is rerun by the builder.

The worker's model/math body is AST-identical to revision 2 from its committed
source snapshot through finalization. Model/config/tokenizer/runtime identities,
four-forward dose, original failed bulk likelihood limits, precision flags,
bitwise causal criteria, 600 seconds, RAM/GPU/disk floors, 100 ms watchdog,
exact Job containment, fencing and cumulative budgets remain unchanged. Reader576,
capture, training and efficacy remain closed. Fresh targeted operations/science
review, a new clean commit and current admission are required for root's one
bounded validation of this repaired supervisor.

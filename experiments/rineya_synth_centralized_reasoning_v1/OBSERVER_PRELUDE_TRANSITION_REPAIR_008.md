# Observer007 failure and exact owned-path repair

Attempt007 stopped in source_preflight with `exact typed finite initial interpreter
path`. Worker exit2, driver exit1. The other eleven operations were unattempted;
no ctypes bootstrap, observer binding or fixture cycle occurred. Job closure,
sampler/NVML termination, frozen outputs and owned process death are recorded.
[Root review](evidence/observer_startup_attempt007_20261007/ROOT_REVIEWED_FAILURE.json)
and [independent failure audit](evidence/observer_startup_attempt007_20261007/INDEPENDENT_FAILURE_REVIEW.json)
verify ten raw files,27 source snapshots/Git blobs, four leases and43 recorded
dead births. The success auditor correctly refused; its success-qualification
cleanup flag does not contradict the raw completed cleanup evidence.

Source diagnosis: the initial exact four-path profile passed. The prelude then
added its two selected roots. Although the caller projected the six-path expected
state, it reused a helper that still required length4. The resulting failure was
before ctypes and fixture work. Actual six-path state was not persisted; this
ordering diagnosis is source-derived, with no reconstructed runtime claim.

The staged repair changes only the observer worker. Initial four-path validation
remains strict. A separate owned-transition check requires the same original
path container, retained initial profile, exact before+two selected physical roots,
and unchanged flags, prefixes, importer and hook identities. It runs before the
ctypes bootstrap. Foreign path/container/flag/plan/profile changes refuse.

Four authored controls pass2.821 seconds; Root repeats four in2.881. The positive
executes the actual frozen definition prelude, then the second profile check and
owned rollback. This composition was missing from the earlier seven-control
packet; those receipts are retained with their narrower scope. [Independent three repair controls](evidence/tensor_prelude_transition_repair_reviewed_cpu_20261007/independent_review/INDEPENDENT_REVIEW001.json)
pass1.935 seconds. [Worker promotion](evidence/tensor_prelude_transition_repair_reviewed_cpu_20261007/ROOT_PROMOTION_002.json)
verifies the exact one-file change. No native API, ctypes or framework execution
occurs in these controls. A new implementation/registration008 must bind the
changed worker before another native attempt; no unchanged retry is admitted.

The original twelve observer operations, resource limits, ABI, Job/lease/cleanup
and bounded no-model approval remain unchanged. Tensor/model/scoring/training
permission and the actual three-arm transfer/retention result remain absent.

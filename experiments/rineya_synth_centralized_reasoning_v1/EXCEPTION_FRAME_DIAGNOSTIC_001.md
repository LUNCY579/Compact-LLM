# Bounded exception frames; CPU helper accepted

The [preserved helper](evidence/exception_frame_helper_reviewed_cpu_20261005/exception_frames.py)
captures only the original exception's Python traceback filenames, functions and
line numbers through builtin descriptors. It never reads locals, arguments,
exception strings/repr, cause/context chains or tensors. Source/runtime hashes
come from supplied registered maps; the helper does not read or verify files.
Unregistered names are redacted and explicitly labeled.

Bounds are16 frames,512 filename characters,128 function characters and16KiB
serialized ASCII including its checksum. Supplied pin maps have at most4096
entries and4096-character keys. Limits can only tighten. Truncation is explicit,
and hostile attribute/string-subclass hooks are bypassed or rejected.

All12 builder controls and the [independent Root12 run](evidence/exception_frame_helper_reviewed_cpu_20261005/root/ROOT_HELPER_CPU_VALIDATION_001.json)
pass with unchanged source bytes and no skips. The
[focused independent review](evidence/exception_frame_helper_reviewed_cpu_20261005/exception_frames_focused_review_1791222663862818800.json)
also verifies hook avoidance, pin conflicts, fallback bounds, traceback identity,
the exact integration boundary and the diagnostic input. These overlapping
checks establish CPU metadata behavior only.

The [integration proposal](evidence/exception_frame_helper_reviewed_cpu_20261005/PROPOSAL.md)
attaches a separate failure-only diagnostic at the current worker catch site,
retaining the original type/message. Helper failure must retain the primary
exception and a separate bounded diagnostic-failure status. No original worker
or native helper has been changed; no success path invokes frame capture.

The [422-token input proposal](evidence/exception_frame_helper_reviewed_cpu_20261005/root/ROOT_FAILURE_PREFIX_INPUT_001.json)
is the failed row's414 prompt tokens plus eight preserved generated tokens. It
is exposed diagnostic history, not a prompt-only evaluation row; it may not enter
M0, training or grading. A new scope must independently register the smallest
sufficient input/dose and owned admission/audit before any CUDA allocation or
model operation. The older no-model D001 scope forbids tensor allocation and
cannot authorize this successor.

Frames can identify a Python reporting boundary, while an asynchronous CUDA fault
may originate earlier. This helper establishes no cause, cure, native readiness,
model fit, correctness or transfer. CUDA synchronization, driver/TDR settings,
resource floors and numerical checks remain unchanged. The full M0/M1/M2
comparison is still required.

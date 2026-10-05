# Bounded original Python exception frames: evidence-only proposal

The helper is CPU-only and has no file, OS, process, model, tensor, framework,
decoder or GPU APIs. It reads the original exception traceback through the
builtin BaseException descriptor, then only traceback frame/code filename,
function name and line number. It never reads locals, args, exception text/repr,
cause/context chains or tensor contents. It joins exact supplied source/runtime
filename pins without file hashing, path resolution or normalization. Unregistered
filenames/functions are redacted and explicitly marked. Overlapping different
pins reject rather than silently selecting one domain.

Default bounds are16 frames,512 filename chars,128 function chars and16384
serialized ASCII bytes including checksum. Source/runtime pin inputs must be
plain dictionaries, at most4096 combined entries; literal filename keys are
bounded at4096 characters. Limits may only tighten these maxima. Oversized
registered names are redacted; frame-count/serialized-size truncation is explicit
and retains a bounded packet. The packet states supplied pins were not
independently byte-verified and establishes no native qualification or cause.
Diagnostic input validation raises ValueError. Callers must preserve the original
failure if metadata collection fails.

The failed05A2 evidence narrows reporting to the unobserved interval after the
successful component pre-forward guard and before the next counted model API:
input tensor creation, mask creation and exact input/mask host-value checks.
The outer marker alone identifies no individual Python call or originating GPU
kernel. Driver reset event co-occurrence and the later stale-CPU abort do not
establish their causal order or root cause. Traceback frames can identify only
the Python reporting boundary in a future failure; they cannot identify an earlier
asynchronous kernel fault or prove any cure.

## Exact future integration site; not implemented here

The current `short_phase_worker.run` caught-error branch is lines84–85:

```python
except BaseException as caught:
    error={'type':type(caught).__name__,'message':str(caught)}
```

In a **new separately reviewed failure-diagnostic worker/contract profile**, retain
that exact primary error dictionary, then attach a separate bounded
`report['exception_frames']` before the existing worker_failed Journal publication
and terminal preservation. If config/preflight is unavailable, pass empty pin
maps and retain redacted/unregistered status. If helper validation/serialization
fails, attach a fixed bounded `exception_frame_diagnostic_failed` status separately;
do not replace, wrap, swallow or stringify a new error over the primary error.
Use the original caught exception, never `__context__` or `__cause__`.

The successor must reuse the exact phase pipeline and original compute/loader/
decoder/collection helpers. No helper call occurs on success. CPU integration
must prove identical success call/guard/Journal order and returned values, original
error type/message retained when the helper fails, bounded terminal serialization
and explicit stage dispatch refusal. Preserve the four current Phase3 source
files and six independent auditor dependency bytes. The new helper/new worker/
contract/tests and proposal must enter a fresh exact source inventory and reviewed
owned diagnostic manifest; do not adapt the existing native success audit or
reuse the failed attempt as successful evidence.

The original whole64 collector's row order,512/32 generation and2050-forward
contract remain unchanged; helper acceptance does not authorize rerunning that
entire failed batch after recorded driver-reset events. Review the smallest
sufficient diagnostic input/dose before any launch. A separately preregistered
diagnostic-only scope may consider the already observed prospective prefix422
(original414 prompt IDs plus eight retained generated IDs), with exact supplied
input hash and provenance, original compute/guards and separately counted
construction/parity/preservation work. Such a scope contributes no M0/scoring
membership. A standalone prefix does not reproduce or explain prior253-forward
history or driver activity; sufficiency and any broader history must be reviewed
from evidence rather than assumed. Do not implement or launch that scope here.
All numerical and600-second/resource/freshness/ownership floors remain unchanged.
Observer/output-byte costs must be included in the successor's separately
reviewed budget. No new CUDA,
model, synchronization, memory query or OS operation is proposed. Do not enable
CUDA_LAUNCH_BLOCKING, change driver/TDR/settings, lower limits or launch/retry
as a side effect of this helper/proposal. Current native scopes stay unchanged.

This stage is helper metadata and CPU controls only. The actual M0/M1/M2 learning,
transfer and retention comparison remains the goal; reporting-location evidence
is not native readiness, model fit, causal diagnosis or a capability result.

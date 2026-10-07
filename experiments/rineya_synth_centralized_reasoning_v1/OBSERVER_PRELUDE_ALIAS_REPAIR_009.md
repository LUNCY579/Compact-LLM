# Observer008 selection failure and finite typing/cache repair

Attempt008 passed the owned root-transition check and completed the approved
ctypes bootstrap, then withheld observer/fixture work in source_preflight.
Worker exit2 and driver exit1; the other eleven operations were unattempted.
[Root audit](evidence/observer_startup_attempt008_20261007/ROOT_REVIEWED_FAILURE.json)
and [independent failure audit](evidence/observer_startup_attempt008_20261007/INDEPENDENT_FAILURE_REVIEW.json)
verify ten raw files,27 source snapshots/Git blobs, four leases and37 recorded
dead births. Job, sampler, NVML and frozen-output cleanup are recorded complete.

Actual metadata reported three unselected cache paths: argparse, gettext and
struct. Their contents were not read during failed preflight. Only those exact
three candidates are now hashed:138,784 bytes, existing qualified-cache role.
Cache/source equivalence and actual cache execution are not inferred from hashes.

The other two records are CPython3.11 typing.io and typing.re class aliases.
The pinned typing.py source explicitly creates them with _DeprecatedType and
places the class objects in sys.modules. The finite checker requires those exact
two keys, their selected owner source bytes, owner/class/metaclass-code/global
identity and exact export objects. It accesses class metadata without invoking
deprecation/custom hooks. Generic originless/non-module entries remain unknown.
Alias records have no invented source/cache origin or execution-history proof.

Final five authored controls and Root replay pass0.264 seconds. They exercise
real typing aliases, before-introspection refusal, counterfeit identity/source/
export negatives and audit owner rebinding. The first passing variant lacked
safe dispatch ordering and is retained separately. [Independent final four controls](evidence/tensor_prelude_alias_repair_reviewed_cpu_20261007/independent_review/INDEPENDENT_REVIEW002.json)
pass0.251 seconds. [Promotion](evidence/tensor_prelude_alias_repair_reviewed_cpu_20261007/ROOT_PROMOTION_002.json)
verifies the exact three-source delta. No ctypes/native APIs/framework/tensor work occurs in these controls.

Future tensor-loader alias admission remains a separate obligation; this repairs
only the observer measurement. A new implementation and registration009 must bind
the selected worker/prelude/auditor changes and exact three-cache delta before
fresh supervised measurement. Scope, resources, ABI, twelve operations and the
existing no-model owner approval remain unchanged. No tensor/model/scoring/
training/transfer authority follows from either failed008 or these CPU controls.

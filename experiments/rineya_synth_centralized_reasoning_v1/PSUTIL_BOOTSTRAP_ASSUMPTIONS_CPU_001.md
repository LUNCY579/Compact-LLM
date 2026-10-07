# Passive psutil preload assumptions

Original ownership/RSS helpers import psutil before the Torch measurement guard.
Passive source inspection identifies seven Python modules/caches, three selected
extensions and builtin _datetime as a finite candidate preload. All source and
extension descriptors match the selected runtime. Existing cache bytes match
compiled source code graphs and physical filenames; no cache was executed.

One missing ordinary PE dependency is selected at System32/pdh.dll: 360,448
bytes, SHA256 `e19f792a1e033870b263027f181274adff7d7ee1198471336547a77498a6ef4e`.
Its ordinary RPCRT4/ntdll candidates are already selected. Thirty-two API-set/
delay names remain unresolved. This is byte and import metadata, not Windows
loader resolution or observed library effects.

[Independent Root byte/code review](evidence/psutil_bootstrap_assumptions_reviewed_cpu_20261007/ROOT_PASSIVE_PSUTIL_REVIEW001.json)
hashes 1,195,813 bytes, compares all seven cache/source code graphs including
exception/line tables, verifies three extensions and the new pdh bytes, and checks
the exact proposed baseline additions. Original observer009's 63-source/59-cache
maps remain unchanged; these assumptions must be registered separately.

The proposed 39-library baseline is original35 plus three selected extensions
and pdh. It is a falsifiable candidate, not an observation. Own-process preload
must verify actual module/file/spec/cache state and the paired post-provider
baseline. Extra lazy imports, cache drift or additional DLLs must fail and retain
partial evidence. Do not convert this assumption into a permissive library union.

[Preservation](evidence/psutil_bootstrap_assumptions_reviewed_cpu_20261007/ROOT_PRESERVATION_001.json)
binds five files and 48,239 bytes. Generic import qualification and current-main/
preloader integration remain under implementation. No psutil import, DLL, native
API, Torch, CUDA, tensors, model or training ran in this passive work.

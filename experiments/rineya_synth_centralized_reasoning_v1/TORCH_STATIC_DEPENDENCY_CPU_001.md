# Static Torch dependency obligations and selected Windows bytes

Installed Windows Torch source explicitly loads 37 DLLs before importing `_C`.
Passive hashing and PE inspection record 38 selected Torch images, 244 import
edges and 34 unique ordinary candidate identities. Twenty-four candidates are
outside observer009's initial 35-library expectation. Static records retain
117 unresolved/delay edges: 107 API-set entries, eight missing selected names
and two duplicate-basename cases. This is not a dynamic load trace.

Two concrete missing identities are now byte-verified at physical System32 paths:
msvcp140.dll (643,512 bytes) and nvcuda.dll (3,807,352 bytes). The additive tensor
candidate contains 8,716 entries, retaining the prior 8,714 descriptors plus
exactly those two Windows files. It is distinct from the earlier 8,716-entry
observer superset. The 59 observed cache pins remain unchanged. Torch's source
selects its lib-directory asmjit/fbgemm paths; duplicate bin-directory leaves
remain roster entries, not asserted load selections.

[Independent audit](evidence/torch_static_dependency_reviewed_cpu_20261007/independent_review/INDEPENDENT_STATIC_WINDOWS001_SUCCESSOR.json)
recomputes static edge classifications and descriptor joins. It independently
hashes and parses five images totaling 4,480,560 bytes, including both Windows
files, and checks malformed PE refusal. Nine selected pins remain unchanged.
It does not repeat the complete 3.76-GB Torch hash or establish DLL resolution.
The initial review's map-order expectation was corrected as a fixture-only issue;
the static multiset is order-independent.

[Preservation](evidence/torch_static_dependency_reviewed_cpu_20261007/ROOT_PRESERVATION_001.json)
binds fourteen files and 2,342,577 bytes.
[Candidate](evidence/torch_static_dependency_reviewed_cpu_20261007/ADDITIVE_WINDOWS_RUNTIME_CANDIDATE001.json)
retains all native qualification/launch flags false. API-set resolution, delay
loads and exact per-boundary effects remain unresolved. Do not replace those
effects with a broad union allowance or label static imports as observed loads.
No DLL, native API, Torch, CUDA, driver modification, model or training ran.

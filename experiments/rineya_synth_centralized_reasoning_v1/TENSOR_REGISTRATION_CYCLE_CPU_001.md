# Tensor registration commit-cycle candidate: CPU review

The historical native contract requires a config to embed the hash of the commit
that contains it. A separate CPU candidate replaces that construction with an
implementation ancestor in all three packets and an externally selected dispatch
HEAD. Real temporary Git repositories verify the implementation ancestry, clean
dispatch, exact source blobs at implementation and exact config/runtime/registration
blobs at dispatch, plus external file/body anchors and map joins.

Final six authored controls pass on Git helper
`2aa39c31be453c9abea135f0486cb61dcaa71c630a417cfbb02f6315e9ed3b30`.
Root previously replayed six controls on the exact pre-float-fix version, then
checked the final affected overflow/NaN/duplicate parser cases. These are separate
runs. Exact predecessors and failed-review findings remain preserved. [Independent final four controls](evidence/tensor_registration_cycle_reviewed_cpu_20261007/independent_review/INDEPENDENT_REVIEW_004.json)
pass in 2.672 seconds, including real committed packet/map and source-ancestor
mutation refusals.

Reads are bounded to declared size plus one sentinel, parsers reject duplicate
fields and nonfinite numbers, and Git blob size must match before retrieval.
Only exact physical nonexecutable output leaves can be exempted from the clean-tree
check. Dirty/wrong dispatch, nonancestor implementation, packet/source/anchor drift,
map mismatch, forged capability and native entry refuse.

[Preservation](evidence/tensor_registration_cycle_reviewed_cpu_20261007/ROOT_PRESERVATION_001.json)
retains all source/test/history/receipt bytes. The copied contract's amended
metadata adapter is source-inspected but not exercised by these minimal Git
packets. The full amended manifest reader is not wired, native entry unconditionally
refuses, and public contracts remain unchanged. These controls qualify the CPU
Git/byte join, not current runtime/bootstrap/library/model execution.

Full integration must consume the amended metadata, real provider and complete
current runtime/source/cache/library evidence before a concrete native registration
exists. The separate startup loader successor must also cover registered package
and nested submodule search paths; the earlier eight/seven-control integration
exercises a top-level inert source only. No model, CUDA tensor, training, scoring
or transfer result follows from this milestone.

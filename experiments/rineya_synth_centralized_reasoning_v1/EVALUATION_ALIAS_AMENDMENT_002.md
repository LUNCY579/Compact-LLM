# Compact aliases before native DEV outputs — 2026-10-03

The first fresh DEV panel retained all 384 rows but 59 prompts exceeded the
480-token prompt allowance. Its maximum was 572 tokens. The original policy,
both root/staging bundles and all measurements remain preserved; no model was
scored on them and no row was discarded.

[Policy 002](EVALUATION_PANEL_POLICY_002.json) applies one uniform bijection to
reference aliases across both conditions of every R/RS source candidate. Short
lowercase aliases replace longer identifiers. Values, requested items, graph
depth/arity, eligible indices, targets, cardinalities, source order, quotas,
strata, renderings and answer format remain fixed. The training corpus and the
512-token context / 32-token generation bounds are unchanged.

The [complete invariant audit](evidence/fresh_dev_compact002_20261003/alias_amendment_invariance_cpu.json)
checks all 768 candidates / 1,536 cells and 3,072 independent visible comparisons.
Answers, full/projection footprints, original source IDs/hashes, component
membership and selected candidate indices/strata agree with policy 001.
All 384 compact prompts fit: measured lengths are 185–452 tokens. The
[prompt-only token packet audit](evidence/compact_dev_input_token_audit_20261003.json)
binds 116,816 prompt positions, exact chat-template/tokenizer assets, all row IDs,
and masks without reading oracle targets. No padding, packing or truncation is
used. Twenty-eight evaluation tests plus the separate input/schedule controls
pass in the [47-test CPU receipt](evidence/compact_schedule_cpu_validation_20261003.json).

The [immutable compact bundle](evidence/fresh_dev_compact002_20261003) contains
prompt packets, separate CPU-exposed DEV oracles, source snapshots and token
arrays. This is not confirmation data or native evaluation approval. Native
collection, M0 prior-correct enrollment, qualified language/long-context guards,
sustained adaptation, cost ceilings and a complete study launch manifest remain
separate prerequisites. No student DEV output influenced this amendment.

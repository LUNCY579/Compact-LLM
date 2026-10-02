# Learning-data draft scientific finding, round 1

2026-10-02. Reviewer `/root`; builder `/root/data_audit`. Disposition:
**REPAIR_REQUIRED before dataset qualification or model exposure.** This is a
CPU data-validity finding, not a model outcome. Reviewers share model/workspace
context and exposed synthetic development evidence.

The initial draft is preserved before changes in `history/learning_data_draft001/`:
source SHA256 `a85e1f2f9494beb07a2d04ff797a2075ea8066c40d87e9eeb4183d7d62bf1536`,
tests `274a1ac3ff02d51cc226b81697ac57589274bc8ecdc67197549a875cf788fd14`.
Its negative shortcut receipt SHA256 is
`83ae5598f0f184baa251f4922d0461062b45037814f399494ada36466a29da96`.
The contract had not yet been created; no frozen registration/result is replaced.

## LD-SCI-01: queried identifier encodes the target status

`build_train32` assigns `wanted = STATUSES[index % 4]`, while `_world` uses that
same index in every entity/query/alias name. Extracting the trailing integer from
the public queried entity and indexing STATUSES modulo4 predicts32/32 status
labels without interpreting any source event or using the oracle for prediction.
Root independently reproduced the shortcut on all32 parents. The builder's
create-only negative receipt independently confirms it.

The original13 passing tests, including source-runtime/latent-oracle cross-checks,
establish semantic generation consistency only. They cannot establish resistance
to memorization or shortcut learning. Such a bank would confound any later status
acquisition curve with the identifier pattern even with perfectly balanced labels.

Verification criterion: make identifiers and presentation position independent
of target status, while preserving32 unique TRAIN parents and two of every status
per declared family. Measure identifier/position/family shortcuts explicitly.
Preserve independent latent semantics under opaque entity renaming, rendering
order changes and permitted irrelevant facts. Check all known naming aliases,
case/parent metadata exposure, and data provenance; public model prompts must not
include target-related metadata. Shared grammar/template limits remain explicit.

## Additional prequalification control

The draft also varies the number of FACT records by status: one for definite
exclusion, two for supported/unknown and three for contradictory applicability,
with an extra fixed conjunctive fact in one family. Measure this count-only
shortcut and use matched irrelevant facts if needed so source size is not a
target code. Distinguish legitimate causal evidence from irrelevant lexical/count
features; do not erase the measured original shortcut or infer general shortcut
absence from a few probes.

No tokenizer/model/framework/host-DB or protected bank was used. All token/mask
measurements, training/serving qualification and learning gains remain unmeasured.
The corrected generator must undergo fresh frozen-source review within this new
dataset milestone; original V1 TRAIN/DEV data and receipts remain unchanged.

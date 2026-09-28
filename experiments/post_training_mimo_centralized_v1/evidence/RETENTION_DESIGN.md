# Proposed retention design for dense Titan

**Status: PROPOSED_UNQUALIFIED.** Audit date: 2026-09-28. This is a candidate for a future `AMENDMENT_002`, not an accepted amendment, runnable data manifest, native authorization or preservation result. Root decides after reviewing E0. The companion machine-readable proposal is `manifests/RETENTION_REQUIREMENTS.json`.

The exact intended source is `rineya.core.titan-experimental.v1.0.0-devfreeze.28L.supa17-rrp015-control-seed17`, with 596,049,920 serialized parameters and weight SHA256 `537201676814ade103c4efa63f5ebe758ea309971803c4ad633dbca5b6b531b7`. Use `receipts/PRIMARY_STUDENT_RECEIPT.json` for its tokenizer/template binding. Static identity does not qualify the current native runtime. Historical FP32 DenseReference scores cannot supply source-correct membership for a different current runtime.

## Recommended 1,024-row replay candidate

| Component | Replay rows | Parent rule | Qualification still needed |
| --- | ---: | --- | --- |
| Original short retrieval | 256 | One row per fresh semantic world | Independent solver, distractor controls, source-correct enrollment and native token fit |
| Original instruction following | 256 | One row per fresh instruction/world family | Deterministic contracts and negative controls; separate from acquisition parents |
| Original answerability/abstention | 256 | 128 answerable/unanswerable twin parents, two rows each | Missing-evidence and answerable controls; preserve pair integrity and score false abstention separately |
| Existing train-only language preservation | 256 | One row per recovered original parent/family | Provenance, current exclusions, target-quality review, frozen semantic rubric and source-correct enrollment |
| **Total** | **1,024** | **896 proposed parent units, conditional on verified grouping** | **No rows enrolled or accepted by this audit** |

The 768 deterministic rows should come from a new central generator, with its source revision, verifier revision and output hashes frozen before use. It has not been implemented or qualified by this proposal. Fresh identifiers or new random seeds alone do not prove independence; semantic parent/family keys must connect paraphrases, reordered facts and answerability twins. Assign roles before rendering, solving, baseline scoring or selection. Keep all derivatives in the same role and use historical fingerprint inventories without opening protected contents.

The language component should initially consider only the v3 14,000-row preservation slice below. The 256 count is an allocation target, not evidence of enough eligible rows. Freeze a bounded training-side candidate pool, deterministic selection order, scoring contract and cost ceiling before native source-correct screening. Keep answerability twins together; if required eligibility or quotas cannot be met, stop that component and amend the proposal. Do not fill shortages from DEV/CHECK/FINAL or silently replace language with additional deterministic tasks.

All replay must satisfy the registered total-context and target-token bounds under the exact native tokenizer/template; no truncation preserving a target label. The proposed 1,024-token E1 bound is not yet a measured fit statement for these sources.

## Independent fresh preservation evaluation

Propose 256 fresh DEV parents and reserve 512 fresh CHECK parents across retrieval, instruction following, answerability and language: 64/128 parents per domain respectively. Answerability has two rows per parent, so these would be 320 DEV rows and 640 CHECK rows, not 256/512 independent examples multiplied by variants. These counts are proposed operational panels, not a statistical noninferiority certificate. CHECK remains reservation-only until its registered gate; FINAL is unopened and unspecified here.

The language DEV/CHECK parents must be newly authored, independent of both replay and historical selection, with a fixed semantic rubric covering adequate answers, relevance, fluency and role consistency. A single preferred wording is not a uniquely correct label. Exact string matching, short outputs, or valid JSON cannot certify language quality. Proposed language evaluation remains a required **unqualified sentinel**, even if the replay component is prepared.

Long-context preservation is an additional required **unqualified sentinel** before any broad preservation claim. It needs fresh evidence-position worlds balanced across early, middle, late and distributed evidence; a qualified genuinely longer operating length; fixed scoring; and a separate numerical resource budget. Its parent count and lengths remain unset. Context lengths at or below 1,024 tokens can test short retrieval, but cannot establish preservation of the source's historical long-context behavior. The four-domain counts above do not silently include or solve this missing coverage. Before broad promotion, the amendment must reconcile this sentinel with the overall DEV/CHECK counts and preservation endpoints.

For every frozen evaluation bank, first collect all unchanged-source responses under the exact Titan/runtime/decoding contract. Freeze source-correct membership before candidate results. Retain all evaluation parents: report source-correct losses separately from source-incorrect repairs, and report each domain independently. A zero source-correct denominator is undefined. Do not select evaluation parents for easy baseline success or use DEV outcomes to choose replay. Model/template/runtime/scorer identities, generated tokens, EOS/termination and every failed attempt must accompany enrollment.

## Audited source candidates and evidence strength

| Candidate | Revision and identity | Allowed proposed role / limitation |
| --- | --- | --- |
| `Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0/data/sft/rineya_v1_v2_preservation_slice_14000.jsonl` | v3.0; seed `20260818`; 14,000 rows; manifest-recorded 17,341,591 bytes and SHA256 `5ee0c283b290cf89f6fcfbcf52bd2bcd03f6c63bb06793b9fa4b957d91973d50` | Preferred language replay candidate. Existing train-only v1/v2 material, pending review and present-day exclusions. Never a fresh DEV bank. |
| `Rineya_Short_Response_Behavior_Corpus_v1.0/data/sft/rineya_short_response_preservation_slice.jsonl` | v1.0; seed `20260816`; 2,000 rows; checksum-recorded SHA256 `1006c6e1d310b618baaa3e54406369530edca6242da83b5de90b9983e24132a4` | Smaller fallback, not additive to the recommended mix: 550 original core train rows plus 1,450 roleplay train rows. Limited coverage and pending review. |
| MaxRL v2 `replay_train` | Source `847cd27e9427c8c5e3acaf81191b2136a628b130`; archive SHA256 `79b8171e83388ff0bf406328f90b5d74ab4cb6d94f29d0ba66af559cbd9b75da`; manifest SHA256 `86ac8f0d851f2178752357f62249eab5a23fd3de8fa032e7373614891df4cd44` | 1,600 dedicated replay rows in eight deterministic reasoning domains. A 128-per-domain, 1,024-row alternative would be reasoning-only; Titan mastery, broad preservation and current admission are unqualified. |
| `Rineya_LCR_SFT_v1/Rineya_LCR_SFT_v1/data/lcr_sft_v1.jsonl` | LCR SFT v1; checksum-recorded SHA256 `8f0bc0174408eb513851443a6f9c32e2e79a00563fb9d35d659e7d27ab792c9c` | Historical 1,800 TRAIN/200 DEV packs with four related rows each. Ancestor-training/selection exposure requires audit; not a fresh long-context sentinel. |

The v3 manifest was freshly hashed as `077c166e09f64a0001153bf62e0236de26b11202024273b56867d967134bb3bd`; the v1 manifest as `b226cfba067102d1490fd261fd24684ecf434f175dcc19e4b0eef0bb482658e0`. Dataset-file hashes above are recorded metadata claims, not fresh row-level or full-file verification in this audit. MaxRL archive identity was verified in the separate `DATA_PROVENANCE_AUDIT.md`; no archive contents were opened here. Preserve project-internal provenance and unresolved redistribution/review obligations; do not invent license clearance.

V3 is train-only material, assembled from prior v1/v2 training sources. Its historical exact-overlap checks concern an older supplied evaluation snapshot, and its near-duplicate checks were sampled. Randomly splitting this material now does not create fresh independent DEV/CHECK, undo ancestry exposure, or establish semantic family separation. Human review remains recorded as pending. The current source-correct language rubric is also missing.

## Titan lineage and exclusions

- Dense Titan descends from exact dense Supa 1.7, not gated Supa-A1.1. Supa incorporates an LCR Stage-A step-200 adapter trained on Halo and transferred to Nimbus at scale 0.8125. Historical long-context training and selection therefore belong in the ancestry audit.
- Titan RRP-015 used 384 solver-verified one-step Forge RL v2 transition rows in `ledger_replay`, `prerequisite_plan` and `register_machine`, without rehearsal. Its training parents were checked against the historical 384-row transition probe and the 384-row SELECT-VIEWS selection bank. That historical check does not qualify new central splits.
- `experiments/titan_model_soup_mini_v1/EVIDENCE.md` records retention-repository snapshot `90255f33bc78cdfd6872b3ff9cc1982d5b9b5d42` and the D4/SELECT-VIEWS overlap: 275/384 parents, including all 128 prerequisite-plan parents. Descendants cannot inherit evaluation eligibility without an ancestry audit. Use authorized ID/fingerprint records; do not read protected scored contents.
- The 256-row `Rineya_Long_Context_Demo_v1` is SPENT and barred from TRAIN, SELECTION and CONFIRMATION under the historical quarantine. It can supply historical context only for this proposal.
- Residual-v23 direct-reasoning/procedural-RAG DEV was previously selection-exposed. It may support disclosed historical diagnostics, never replay or fresh confirmation. Its `fresh_sealed` populations and Invisible-Limit `test2` are ANY_READ-barred for this work. No protected contents are needed to retain these exclusions.
- Preserve the existing prohibition on `splits_onpanel` and the historical quarantine records. A new role label, new split seed or tokenizer change does not remove exposure.

## Practical next gate

After E0 review, root may freeze `AMENDMENT_002` with exact scopes, parent/row counts, training candidate limits, native baseline cost, language review/scoring, long-context sentinel design and promotion limitations. Then implement/qualify the original deterministic generator, recover language parent provenance, apply authorized fingerprint exclusions, verify selected source bytes and native token lengths, and run bounded baseline enrollment. Freeze the accepted replay/DEV manifests before E1; CHECK remains closed until the registered checkpoint/seed decision. This proposal grants none of those native actions by itself.

An all-original narrow alternative could use 342 retrieval, 342 instruction and 340 answerability replay rows (170 twins), with separately frozen three-domain DEV/CHECK allocation. It requires its own amendment decision and explicitly supports only narrow deterministic preservation. It cannot replace the missing language or long-context sentinels or justify a broad claim.

No dataset rows, CHECK/FINAL contents or native model outputs were read or generated for this document. No dataset was enrolled and no source code was modified.

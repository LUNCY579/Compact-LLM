# Project scripts by phase

| Phase | Folder | Main responsibility | Downstream artifact |
|---|---|---|---|
| 0 | `phase_0_data` | Data contract, validation, leakage control, human audit. | Approved data manifests. |
| 1 | `phase_1_dense_sft` | Dense concise-behavior baseline. | Dense SFT adapter/checkpoint. |
| 2A | `phase_2_layer_selection` | Static selected-layer and module-group tuning study. | Preserved positive or negative training-efficiency result. |
| 2B | `phase_2_layer_selection` | Every-layer deletion, recovery SFT, progressive depth reduction, profile catalog. | Full-quality/balanced/min-memory branches. |
| 2C | `phase_2c_targeted_recovery` | Rank surviving layers and train low-resource selected-layer recovery controls. | Maximum-quality/balanced-training/minimum-training-memory adapter profiles. |
| 3 | `phase_3_preference_alignment` | DPO for concise-correct preference on one selected profile. | Profile-specific aligned adapter. |
| 4 | `phase_4_model_soups` | Compatible delta merging or best-individual decision for that profile. | Accepted Phase 4 dense/full checkpoint. |
| 4B | `phase_4b_structural_reduction` | Post-alignment structural revalidation/refinement and exact checkpoint selection. | `selected_inference_reference.json`. |
| 5 | `phase_5_context_system` | Retrieval, budgets, prefix/KV reuse. | Approved context system. |
| 6 | `phase_6_sparse_simulation` | Activation traces, oracle masks, quality frontier. | Router dataset and frontier. |
| 7 | `phase_7_sparse_runtime` | Strict response control, routers, dense fallback, measured runtime. | Response-control verification, runtime comparison, and optional router checkpoint. |
| 8 | `phase_8_release_candidate` | Strict sealed evaluation, full release audit, and human decision. | Release package with response-control policy. |

## Responsibility boundaries

- Phase 2A asks **where to train** and may remain a negative result.
- Phase 2B asks **which complete layers can be removed and recovered before alignment**.
- Phase 2C asks **which surviving layers should receive the smallest useful recovery adapter**.
- Phase 4B asks **whether the aligned checkpoint should keep, refine, or reject structural reduction**.
- Phase 6 asks **what can be temporarily masked** while preserving dense computation for scientific validation.
- Phase 7 asks **what can be dynamically skipped in the measured runtime**.

A reduced Phase 2B or Phase 2C profile is never silently substituted. Phase 3 resolves exactly one named, human-approved branch, and profile-specific artifacts are written side by side. Phase 5 still requires an approved Phase 4B selected-reference manifest.

- Learned short behavior and hard response control are measured separately: Phase 1 uses `eos_only`, while production-oriented Phases 5–8 default to `one_sentence`.
- The strict boundary is an output/runtime mechanism. It does not authorize whole-layer early exit or reduce checkpoint weight memory.


## Shared evaluation and promotion rule

Every phase uses Quality Evaluator v3 and the same relative-opportunity engine. Each phase declares one primary proportional benefit—training state, loaded weight memory, context/KV tokens, active compute, measured latency/VRAM, or net quality improvement. A technically comparable candidate advances only when that benefit is strictly greater than its combined proportional quality opportunity loss. No phase may add a private absolute score, pass-rate, completion, category, NLL, recall, speedup, or issue-count floor. Informal understandable replies and natural follow-up questions remain valid unless an explicit response contract says otherwise. Human review remains mandatory at stop gates.

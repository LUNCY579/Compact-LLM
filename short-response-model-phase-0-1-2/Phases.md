# Project scripts by phase

| Phase | Folder | Main responsibility | Downstream artifact |
|---|---|---|---|
| 0 | `phase_0_data` | Data contract, validation, leakage control, human audit. | Approved data manifests. |
| 1 | `phase_1_dense_sft` | Dense concise-behavior baseline. | Dense SFT adapter/checkpoint. |
| 2 | `phase_2_layer_selection` | Static selected-layer and module-group study. | Selected training configuration. |
| 3 | `phase_3_preference_alignment` | DPO for concise-correct preference. | Aligned candidate(s). |
| 4 | `phase_4_model_soups` | Compatible delta merging or best-individual decision. | Accepted Phase 4 dense model. |
| 4B | `phase_4b_structural_reduction` | Permanent structural reduction and exact checkpoint selection. | `selected_inference_reference.json`. |
| 5 | `phase_5_context_system` | Retrieval, budgets, prefix/KV reuse. | Approved context system. |
| 6 | `phase_6_sparse_simulation` | Activation traces, oracle masks, quality frontier. | Router dataset and frontier. |
| 7 | `phase_7_sparse_runtime` | Routers, dense fallback, measured runtime. | Runtime comparison and router checkpoint. |
| 8 | `phase_8_release_candidate` | Full release audit and human decision. | Release candidate package. |

## Responsibility boundaries

- Phase 2 asks **where to train**; it does not delete inference layers.
- Phase 4B asks **what can be permanently removed** from the finalized aligned model.
- Phase 6 asks **what can be temporarily masked** while preserving dense computation for scientific validation.
- Phase 7 asks **what can be dynamically skipped in the measured runtime**.

The runner for each phase checks the previous human gate. Phase 5 specifically requires a Phase 4B PASS and a finalized selected-reference manifest.

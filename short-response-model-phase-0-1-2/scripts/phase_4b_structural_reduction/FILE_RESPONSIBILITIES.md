# File responsibilities — Phase 4B structural reduction

The PowerShell runner is the normal entry point. Python executables may be run individually for debugging or resumable work.

| File | Responsibility | Must not be treated as |
|---|---|---|
| `run_phase_4b.ps1` | Resolves the accepted Phase 4 model, creates calibration if absent, builds pruning/depth candidates, evaluates them, ranks them, and writes gate evidence. | Permission to overwrite the selected deployment reference automatically. |
| `build_long_form_calibration.py` | Generates matched approved-short and deliberately-long continuations from non-sealed development prompts. | SFT/DPO training-data generation. The output is calibration-only. |
| `profile_long_form_components.py` | Profiles early-versus-late MLP-channel and attention-head contribution proxies. | Proof that a component has one exclusive semantic function. |
| `build_pruning_plan.py` | Converts profiles into deterministic guided pruning plans and equal-size random controls. | A quality decision. |
| `apply_structured_pruning.py` | Physically slices compatible MLP tensors and optional GQA groups, then smoke-tests and reload-verifies the checkpoint. | A mask-only simulation or a guarantee of speed/quality. |
| `reduce_depth.py` | Physically creates suffix-drop and equal-layer-count uniform depth candidates and verifies reloadability. | Evidence that training-selected layers are the only inference-critical layers. |
| `measure_checkpoint.py` | Measures parameters/checkpoint bytes and records checkpoint metadata. | A latency or VRAM benchmark. |
| `evaluate_candidates.py` | Runs the common behavior evaluator and size measurement for dense and every structural candidate. | The final human choice. |
| `recommend_reference_candidate.py` | Compares structural memory benefit with proportional quality opportunity loss and recommends a Pareto-efficient reduction or dense fallback. | Automatic reference finalization. |
| `create_phase_4b_gate.py` | Creates the recommendation report and pending Phase 4B human stop gate. | Human approval. |
| `finalize_reference_selection.py` | After explicit approval, writes `selected_inference_reference.json` for Phases 5–8. | Candidate evaluation. |
| `__init__.py` | Marks the folder as an importable Python package for tests/helpers. | An executable phase step. |
| `README.md` | Explains Phase 4B scope, candidate types, outputs, and fallback policy. | An execution log. |
| `EXECUTION_MANUAL.md` | Provides normal/resume/finalization commands and review procedure. | Evidence that a candidate passed. |
| `FILE_RESPONSIBILITIES.md` | Documents ownership and boundaries for every file in this folder. | Runtime code. |

Generated reports are evidence, not automatic approval. Keeping the dense reference is a valid Phase 4B outcome.

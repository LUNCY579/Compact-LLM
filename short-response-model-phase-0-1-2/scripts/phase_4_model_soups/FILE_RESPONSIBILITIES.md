# File responsibilities — Phase 4 model soups

The PowerShell runner is the normal entry point. Python executables may be run individually for debugging or resumable work.

| File | Responsibility | Must not be treated as |
|---|---|---|
| `run_phase_4.ps1` | Checks Phase 3 approval, resolves the exact SFT source, trains/evaluates compatible DPO seeds, validates adapters, executes greedy soups, and writes the Phase 4 model manifest. | A human approval mechanism. |
| `build_greedy_soup.py` | Ranks ingredients, materializes and evaluates each greedy trial, applies regression floors, and saves the accepted full checkpoint plus reports. | A theoretical soup planner. It runs real merges/evaluations. |
| `merge_delta_soup.py` | Reconstructs every PEFT delta against the same base, averages deltas in FP32, and saves a standalone Transformers checkpoint. | Quantized-integer averaging or LoRA A/B factor averaging. |
| `validate_ingredients.py` | Checks PEFT configuration and actual saved tensor names/shapes before any merge. | Proof that a soup will improve quality. |
| `resolve_phase_4_model.py` | Reads `phase_4_reference.json` and prints the accepted local full-checkpoint path for Phase 4B. | A gate checker. Phase 4B separately checks human PASS. |
| `greedy_soup_plan.py` | Preserved legacy planning helper for inspecting candidate order. | The production Phase 4 executor. |
| `check_human_gate.py` | Backward-compatible Phase 4 gate wrapper for older commands. | The canonical multi-phase gate checker; active runners use `scripts/common/check_human_gate.py`. |
| `README.md` | Explains Phase 4 scope, inputs, outputs, and practical dense fallback. | An execution log. |
| `EXECUTION_MANUAL.md` | Provides normal/resume commands and human review procedure. | Evidence that a run completed. |
| `FILE_RESPONSIBILITIES.md` | Documents ownership and boundaries for every file in this folder. | Runtime code. |

Generated reports are evidence, not automatic approval. Do not edit a gate to `PASS` until raw outputs and the stated quality/safety floors have been reviewed.

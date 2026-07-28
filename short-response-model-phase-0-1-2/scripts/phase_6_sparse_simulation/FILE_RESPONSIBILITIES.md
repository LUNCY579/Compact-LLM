# File responsibilities — Phase 6 sparse simulation

The PowerShell runner is the normal entry point. Python executables may be run individually for debugging or resumable work.

| File | Responsibility | Must not be treated as |
|---|---|---|
| `run_phase_6.ps1` | Builds non-sealed calibration, profiles dense activations, runs masked generation across ratios, and writes the frontier/gate. | A speed benchmark. Dense computation is intentionally retained. |
| `build_calibration_corpus.py` | Builds a deduplicated, target-stripped, non-sealed calibration corpus with category metadata. | Training data or sealed evaluation. |
| `profile_activations.py` | Saves per-layer router inputs, block scores, oracle masks, token positions, and model provenance. | Evidence that routing is fast. |
| `simulate_sparsity.py` | Measures contribution retention and actual output quality while masking blocks after dense gate/up computation. | A custom sparse kernel. |
| `check_phase_6_gate.py` | Selects a viable active ratio or records the product-safe `DENSE_ONLY` fallback. | Human approval or a wall-clock claim. |
| `check_human_gate.py` | Backward-compatible Phase 6 gate wrapper. | The canonical multi-phase gate checker. |
| `README.md` | Explains calibration, traces, simulation, and dense-only fallback. | An execution log. |
| `EXECUTION_MANUAL.md` | Provides normal/resume commands and raw-generation review steps. | Evidence that sparse runtime will be faster. |
| `FILE_RESPONSIBILITIES.md` | Documents ownership and boundaries for every file in this folder. | Runtime code. |

No speed or memory-reduction claim is valid until Phase 7 measures the actual runtime.

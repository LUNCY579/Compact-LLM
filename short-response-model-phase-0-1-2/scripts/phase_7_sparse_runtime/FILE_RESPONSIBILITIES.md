# File responsibilities — Phase 7 sparse runtime

The PowerShell runner is the normal entry point. Python executables may be run individually for debugging or resumable work.

| File | Responsibility | Must not be treated as |
|---|---|---|
| `run_phase_7.ps1` | Resolves the Phase 6 decision, trains/benchmarks routers when viable, or measures dense-only runtime, then writes deployment evidence. | A requirement to use sparsity. |
| `train_router.py` | Trains one validation-calibrated router per decoder layer and selects confidence thresholds for dense fallback. | A production kernel. |
| `benchmark_runtime.py` | Measures identical dense/routed prompts or dense-only baseline, including latency, VRAM, output quality, active ratios, and fallbacks. | A theoretical FLOP estimator. |
| `resolve_phase_6_ratio.py` | Returns the approved active ratio or `NONE` for dense-only execution. | A gate checker. |
| `check_phase_7_gate.py` | Recommends `DENSE_RUNTIME` or `SPARSE_FEATURE_FLAG` from measured results and writes the pending gate. | Human approval. |
| `KERNEL_CONTRACT.md` | Defines correctness, layout, fallback, telemetry, and benchmarking requirements for a later packed GPU kernel. | An implemented Triton/CUDA kernel. |
| `check_human_gate.py` | Backward-compatible Phase 7 gate wrapper. | The canonical multi-phase gate checker. |
| `README.md` | Explains routed reference behavior, dense fallback, and deployment decisions. | An execution log. |
| `EXECUTION_MANUAL.md` | Provides sparse and dense-only commands plus evidence-review steps. | Evidence that gathered PyTorch will beat dense GEMM. |
| `FILE_RESPONSIBILITIES.md` | Documents ownership and boundaries for every file in this folder. | Runtime code. |

The gathered PyTorch implementation is a correctness/measured reference. Shipping custom sparsity requires measured value on the target hardware.

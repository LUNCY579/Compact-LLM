# Kernel prototype contract

The repository intentionally does not claim that ordinary PyTorch masking is a sparse runtime. A production prototype must use contiguous block layouts, avoid per-token CPU transfers, report router overhead separately, and compare against the exact dense runtime on the same hardware. Start with correctness, then Triton/custom CUDA only when Phase 6 passes.

# Phase 7 Execution Manual

1. Verify Phase 6 human `PASS` and the frozen model hash.
2. Train routers from `router_dataset.pt`.
3. Inspect per-layer validation BCE, weighted recall, accepted fraction, and confidence threshold.
4. Run the dense-versus-routed benchmark with deterministic decoding.
5. Review raw output pairs; runtime improvement cannot compensate for correctness or safety regression.
6. Record active ratio and fallback rate by layer.
7. Test multiple keep ratios and confidence thresholds without tuning on sealed evaluation.
8. Preserve the PyTorch reference result, including negative speedups.
9. Only after quality passes, implement the kernel contract in `KERNEL_CONTRACT.md`.
10. Quantize the deployment checkpoint, retrain/recalibrate routers as needed, and repeat the full benchmark.

Phase 7 passes only with preserved quality and either at least 1.20x measured decode speed or at least 20% lower measured peak VRAM in the exact target runtime.

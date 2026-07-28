# Sparse Kernel Prototype Contract

The PyTorch gather reference is not the production kernel. A production prototype must:

1. keep MLP blocks contiguous and prepacked;
2. avoid per-token CPU/GPU transfers;
3. fuse or overlap router scoring, top-k selection, and block dispatch;
4. use dense prefill unless a separate prefill kernel is measured faster;
5. include router and fallback overhead in wall-clock timing;
6. compare against the exact dense kernel, quantization, batch size, and context length;
7. report output agreement, category quality, active ratio, fallback rate, bytes moved, TTFT, decode time, and peak VRAM/RAM;
8. retain a dense fallback path and fail closed on unsupported shapes;
9. pass correctness against the PyTorch reference before performance tuning;
10. abandon the custom path if measured user-visible value does not justify complexity.

Suggested ladder: PyTorch correctness -> Triton packed-block kernel -> look-ahead routing -> hot/warm/cold placement -> local-runtime integration.

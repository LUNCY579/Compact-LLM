# BAG-001 build verification — 2026-10-04

## Executed locally

- **96 tests passed**; no failures or skipped tests.
- Python **3.13.5**, PyTorch **2.10.0+cpu**; CUDA unavailable.
- `compileall` completed successfully for the experiment subtree.
- The CLI synthetic report and hash-verified capture roundtrip were exercised.
- A tiny attention model exercised all four training arms, deterministic paired
  schedules, selected FP32 parameter updates, fixed evaluation precision and
  frozen-weight preservation. This is fixture evidence, not Rineya training.
- The full-model no-update capture harness was tested on that tiny fixture,
  including both DEV cohorts and a missing requested capture-site failure.

One PyTorch profiler warning concerned event clearing across profiler cycles.
The tests use single-cycle profiles and inspect their returned operator events;
it did not cause a test failure. No remote CI execution is claimed.

## Numerical witness

The Appendix-A-style BF16 **cast emulation** gives an erroneous first dQ coordinate
of `-24.0` at key offset `65536`, while the FP64 reference is `0.0`. GProj's
corrected contraction gives `0.0` in that coordinate. The sweep passes at offsets
`0, 64, 1024, 65536` and widths `2, 64, 128`.

This is not a claim that the CPU math backend produced a -24 error. Native math
and FP32-attention measurements are reported separately. Other gradient
coordinates retain ordinary rounding error; projection does not make the entire
BF16 gradient exact.

## Not executed or established

No exact Rineya checkpoint was loaded, no native Rineya training was performed,
and no acquisition/retention behavioral improvement was measured. Transformers
was not installed in this execution environment. RTX-4070/CUDA behavior, native
loader compatibility, resource fit and fused kernels remain unqualified.

The current correction is a dense research implementation, not an optimized
FA2/FA3 kernel. CPU tests do not establish the paper's training loss or overhead.
Native qualification and a later behavioral study remain necessary.

## Reproduction

Run the scoped pytest and synthetic commands in README.md from the repository
root. Qualification and training output paths are create-only. The source parent
is `03db218ceebc61a2bc04f2482cf7e094045e01c1`; every added file is inside
`experiments/rineya_bf16_attention_gproj_v1/`. Existing files are unchanged.

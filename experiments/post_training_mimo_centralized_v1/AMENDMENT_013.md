# Amendment 013: complete the measured 512-token CE envelope

30 September 2026, America/Chicago. The first streamed unpadded trial from
`a4454e17009c9d3aadd87e2b9b633c2969f5be00` loaded the exact Titan GPU weights,
passed all 32 zero-adapter comparisons, native unpadded causality and all-one-mask
equivalence with zero error. It completed 256/512-token backward and optimizer
fixtures. At 1024, backward returned, but available VRAM was 0.161270 GiB against
the unchanged 0.5 GiB reserve. Preserve the failed result and every raw receipt.
Worker RAM remained within the 1.75 GiB ceiling; supervisor peak was 1.740429 GiB.

The prepared E1 TRAIN manifest has a maximum sequence length of 398 tokens. A
512-token training envelope can retain every registered training row and complete
target without truncation. Prospectively register stage `unpadded_ce512`, scope
`UNPADDED_SINGLE_SEQUENCE_CE512_V1`, and
`configs/E0_LAPTOP_STREAM_UNPADDED_CE512.json`. Use the same exact model, fixed
2 MiB loader, 1.25 GiB host reserve, 1.75 GiB RSS cap, GPU/disk floors, containment,
BF16 base, FP32 rank-8 q/v adapters, optimizer and numerical tolerances.

Use contexts 256 and 512 for the same 32 base and zero-adapter comparisons and
two backward profiles. Retain the 80-token causality and actual all-one-mask
diagnostics, positional EOS labels, independent causal CE, finite confined
gradients, frozen-base verification, adapter save/reload and bitwise resume.
Resume uses the last 512-token fixture. This makes four disposable optimizer
calls and 1792 training-forward positions, with no generation or teacher calls.
The smaller fixture changes the declared RNG/workload trajectory; bitwise resume
is tested within this new stage, not asserted against the failed 1024 stage.

Enforce a hard 512-token input ceiling alongside the existing single-sequence,
all-one-mask, no-cache, no-packing and no-custom-position boundaries. Actual
TRAIN rows still require semantic origin and token/label binding before E1.
The original 1024 registration and failure remain historical evidence; neither
is relabeled. No runtime math, train mode, activation checkpointing, quantization,
CPU model or optimizer offload is introduced by this successor.

Success is `PASS_E0_UNPADDED_CE512_SCOPED`, distinct from full E0 and the earlier
unpadded or resource-only statuses. A separately selected, source-bound evidence
checker must reconcile this exact config, inventories, journal, timings and
containment. It always leaves E1 authorization false. Native capture, preservation,
language grading, long-context runtime and a current E1 plan remain prerequisites.
No scoped qualification demonstrates capability gain or opens CHECK/FINAL.

Activation checkpointing remains a separate prospective option for longer
training contexts if required: local Qwen3 only activates it in train mode, and
non-reentrant checkpointing would need explicit registration, finite-gradient,
RNG/resume and resource measurements. Complete the supported TRAIN envelope first;
retain long-context evaluation as an independent scientific requirement.

# Next experiment

30 September 2026. The user-authorized laptop route now has an actual streamed
GPU measurement. At 5.12 GiB launch headroom, it loaded all 310 tensors exactly,
passed 32 zero-adapter cases, native unpadded causality and all-one-mask equivalence
with zero error. It completed 256/512-token backward and two disposable updates.
At 1024, backward returned but free VRAM fell below the unchanged 0.5 GiB reserve.
Peak process RAM was 1.740 GiB; minimum sampled host headroom was 3.032 GiB.
This is a VRAM limit in the attempted envelope, not a new host-RAM failure.

AMENDMENT_013 registers unpadded_ce512 under the same 1.25 GiB system reserve,
1.75 GiB worker RSS ceiling and fixed 2 MiB loader. All prepared TRAIN sequences
are at most 398 tokens, so 512 can retain the complete dose without truncation.
Complete this scope's four disposable updates, save/reload, bitwise resume and
frozen-base checks under unchanged numerical tolerances. Preserve the failed
1024 trial and its partial counters; its two updates are not acquisition evidence.

Use scripts/run_laptop_adapter_trial.cmd or the regular PowerShell launcher,
which now defaults to unpadded_ce512. It needs a clean committed checkout and
reviewed config/source, waits at most ten minutes for two measurements >=3.0625
GiB, then invokes one contained trial of at most 30 minutes. No automatic retry
or E1 launch. The earlier timed-out controller and latest native controller are
terminal. Actual resources are measured, not inferred from app working sets.

The scoped consistency checker passed 95 stdlib controls; its 512 successor
must bind the distinct config, workload, profile timings, inventories and
containment before accepting a complete scoped native result. It always keeps
E1 authorization false. After scoped qualification, bind current laptop native
capture, queue and E1 successors. Keep one resident GPU worker, streamed
pretokenized microbatch one, GPU adapter/Adam state, adapter-only checkpoints
and sequential evaluation. Activation checkpointing remains a registered-future
option for longer training contexts; it is not introduced into the 512 route.

Language grading, native long-context evaluation and preservation remain separate
prerequisites. E1 seed 101 remains 256 planned steps with 16 acquisition plus
4 replay sequences each; 211/307 are conditional on its DEV screen. Historical
plans require current source/runtime binding. Current best remains unchanged
Titan, acquisition/retention are null and CHECK/FINAL remain closed. The original
verified capability/retention research objective remains incomplete.

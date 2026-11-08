# CDE-001 delivery validation — 2026-09-29

## Executed locally

- Python 3.13.5; PyTorch 2.10.0+cpu; pytest 9.0.2.
- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q experiments/rineya_compute_depth_elasticity_v1/tests`
- **64 passed in 3.56 seconds** on the final added-package source.
- `plan`: exited 0; emitted the 16-cell R/K matrix with explicit proxy-only cost labels.
- `smoke`: exited 0; deterministic scripted DEV/TEST traces, 12 distinct parents per split and 8 rollouts per parent.
- `native-preflight`: exited 2 and emitted BLOCKED; no weights loaded.
- Receipt overwrite refusal, native-block behavior and torch-free CLI import are also tested in subprocesses.

This was an isolated local copy of the added package, not a clone or full
repository regression run. Existing native/LoopSpec suites were not executed.
The scoped GitHub Actions workflow is included for remote CPU reproduction;
its outcome is not asserted by this committed local validation record.

## What the tests establish

They exercise consecutive-depth and same-prefix validation, malformed
probabilities, nested seeds, depth/width selector versus oracle separation,
reversions, late/missed rescues, parent balancing, DEV/TEST separation, frozen
checkpoint/depth policy binding, conservative full-depth fallback, five masked
PyTorch objectives, KL direction/temperature/normalization, detached teacher
gradients, masked-token gradients, and verified relabel proposal guards.

In the intentionally adversarial mixed fixture, zero-tolerance DEV calibration
chooses the full-depth fallback. It does not manufacture a speedup by treating
stable-but-wrong outputs as completed reasoning. A separate all-correct fixture
tests that safe early work reduction can be selected. These are designed method
checks, not empirical Rineya accuracy or risk estimates.

## What was NOT executed or established

No native Rineya loads, forwards, optimizer/training steps, GPU runs, checkpoint
downloads, paid teacher calls, dataset mutation or sealed evaluation occurred.
Loss tests use random tensors and autograd only. No native adapter, recurrent
cache parity, actual latency/memory improvement, learned self-relabel gain,
long-context benefit, or cumulative skill improvement has been demonstrated.

The branch is ready for code review and bounded native integration work, not
model promotion. Follow PROTOCOL.md and preserve the existing campaign gates.

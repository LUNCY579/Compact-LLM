# Validation and execution record

Date: 2026-10-03. Status: **CPU_FIXTURES_PASS_NATIVE_NOT_RUN**.

## Executed

- Scoped pytest suite: **57 passed, 0 failed, 0 skipped**.
- Python compileall: PASS.
- Analytic/synthetic CLI fixture suite: twice; numerical outputs and training
  histories identical after excluding wall-clock fields. Each suite performs
  48 synthetic interface-optimizer steps (96 across two suites). This count
  excludes optimizer calls inside unit tests; those are also synthetic.
- Default native preflight: BLOCKED, exit code 2, as intended.
- Paired-report CLI: covered by tests.

Environment: Python 3.13.5; PyTorch 2.10.0+cpu; pytest 9.0.2;
psutil 7.2.2. No packages were installed to obtain these results.

## Coverage

Tests cover per-token projection, zero anchors, FP16/BF16/FP32/FP64, autograd
gradcheck, NaN/Inf handling, exact zero-start injection, matched parameter counts,
blend opening, live-anchor truncated gradients and full-window gradient parity;
raw-vs-applied suppression, padding masks and logit KL; shape-fixture whole-block
execution, distinct per-layer positions, frozen first-pass/alpha-zero identity,
restoration on success and exceptions, positional cache rejection, overlap and
RMS safety bounds, base gradient exclusion, and no deeper native switch.

Also covered: optimizer/split/resource guards, depth-prior RNG resume and invalid
support, exact-prompt and cluster leakage, binary paired outcomes, no seed mixing,
single-cluster CI omission, immutable runs, digest/path/symlink checks, native
default fail-closed behavior, and no output inside canonical checkpoint trees.

The host-class fixture intentionally has the audited class name/shape so its
wrapper contract can be exercised; it is not real Qwen, Supa or a model checkpoint.
Both additive and orthogonal synthetic regression losses decrease. Which is
lower varies with seed; these constructed losses cannot rank native methods.
The analytic wrong-attractor and identity fixtures are stationary but incorrect,
and the zero-scaled fixture is stationary by suppressing its nonzero update.

## Not executed / remaining qualifications

**Native checkpoint loads: 0. Native optimizer steps: 0. GPU/laptop jobs: 0.**
No downloaded checkpoints, MiMo data ingestion, protected evaluation, changes to
laptop schedules, or game work. No native training, terminal-KV, prefill, endpoint
RL or implicit differentiation experiment. No full inherited test suite, real
custom-loader run, memory feasibility validation, or GitHub Actions green claim.

Native bridge and scorer integration are source-audited and CPU-fixture-tested,
not exercised against the real canonical runtime. Initial native screen is
inference-only and its zero-start arms test identity, not efficacy. Stage-2
optimizer mechanics exist, but native data/gradient/rollback qualification and a
native training entry point remain intentionally deferred.

Machine-readable bounded evidence is in `evidence/cpu_validation.json`.

# Implementation and review boundaries

## Reproduced mechanism versus adaptation

The original implementation independently expresses the paper's bias-corrected
EMA moments and the bounded `1 + rho * (P - gamma*S)/(P + gamma*S + eps)` controller.
No author source is vendored. Defaults are documented against Appendix C.1, not
inferred from an AI summary. This does not reproduce the paper's trained weights,
data, Qwen eta/3 integration, training procedure, kernels, metrics or speedups.

The Rineya changes are explicit: full ARMR residuals; normal whole-block pass as
unit warmup; fresh scheduler on every full-prefix model call; full-prefix statistics
with no native padding; optional masked controller API; per-trajectory seeded
Python RNG for the random control; FP32 moments and scale application; fail-closed
finite checks. For refresh intervals above one, moments observe EVERY update and
only alpha is held. This is a specified Rineya convention, not verified q>1 author
code equivalence. Native default q is one.

## Exactness, state and gating

`apply_update` returns the actual proposal object at alpha one. It does not
reconstruct it through `old + (proposal-old)`, which can introduce cancellation
rounding. Mixed batches also preserve unit rows. Disabled scheduler mode means
unit recurrence, while disabled host integration means no EXTRA block application.
These are intentionally distinct control axes.

The controller rejects changes to sequence identities, shapes, devices or masks
until reset. Its retained state is one mean vector, previous raw update, mask,
scalar energy and alpha per sequence; it does not retain an entire hidden history.
Full native prefixes are batch one; the controller's general batching contract
is tested separately. Per-loop records are scalars and streamed JSONL. Native
controller state never leaks across different token prefixes or independent samples.

The inherited gate is not rescaled twice. For ARMR, delta is the whole operator
proposal minus its input. A future gated adapter should pass its ACTUAL gated
proposal; log both raw and gated norms separately and calibrate afresh. A zero
actual update cannot become useful just by changing alpha. No GRU/gate efficacy
claim follows from the analytic zero-update fixture.

## Split and oracle rules

Calibration validates DEV identity/parent and trace provenance before scores are
used. The immutable plan binds config, DEV records, fixed selections, empirical
alpha pools and disjoint TEST parents. Test evaluation rejects altered configs,
budgets and parents; TEST oracle calls require that frozen DEV binding.

A4-example chooses a constant grid alpha per problem retrospectively. A4-step
exhaustively enumerates all grid schedules only when `5**extra_loops <= 25`.
Every nonconstant schedule recomputes its own live proposals. Constant schedules
reuse already charged fixed-grid results. At four extra loops the step oracle
is NOT_ENUMERATED, not a greedy upper bound. Continuous TAPS may outperform a
finite-grid oracle; capture ratios are therefore not artificially clipped.

Native DEV A4 is per-problem constant only. It is not an exhaustive oracle over
all token-prefix alpha decisions during language generation. Native protected
TEST and extended native horizons are deliberately absent from the initial lane.
The apparatus 'TEST' consists solely of public mathematical fixtures, not the
repository's sealed data. No native dataset is silently generated or changed.

## Timing and resource accounting

Ordinary fixture arms at a fixed budget execute the same proposal/readout count.
The compute ledger also charges all additional oracle search calls. Controller
timing includes moment statistics and scale application; scalar conversion,
readout/proxy work and logging remain separately identifiable or in wall time.
This is a diagnostic-path comparison, not a claim that the ordinary production
baseline pays no controller overhead. Accuracy-target tables are retrospective
budget sweeps, never an online label-aware stopping policy.

Native generation reuses ARMR's exact scorer, token caps, prefix accounting,
first-token logit comparison. The optional inherited full-continuation gold NLL
is deliberately disabled through a fresh function-global dictionary; no inherited
module or generation/scorer bytecode is mutated. Otherwise a sequence-wide alpha
would let future gold tokens affect earlier logits and contaminate the DEV random
pool. Native gold NLL/rank are null in this lane; task scores remain available.
A future NLL collector must replay causal prefixes and keep diagnostic alphas out
of the generation-only random-control pool. The inherited dense-matmul proxy excludes
attention, norms, instrumentation and optional gold-NLL passes; those exclusions
are preserved, not silently described as total FLOPs. New counters include all
model calls and token positions (gold NLL is disabled in this lane). First-pass local core timing
covers the last wrapped layer; extra-pass timing covers the whole recurrent block.
Only end-to-end model/case wall time is an end-to-end latency measurement.

Memory guards are conservative checks, not reservations or hard OS enforcement.
RSS/deadline checks occur between complete forwards; one native forward can cross
a limit before control returns. No CUDA fallback, paid remote job, optimizer,
checkpoint write, model download or automatic dependency installation exists.

## Readiness

The independent CPU fake host tests restoration, positions/masks, normal-pass
history, one-extra-pass recurrence and exact unit behavior. It is NOT a native
Qwen test. A source-bound real model run and complete inherited regression suite
are still required before a native quality claim or production integration.

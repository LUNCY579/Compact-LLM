# Host handoff and native boundary

## Current delivery

The branch supplies a research programme, manifest, standalone NumPy references,
and synthetic tests. It does not bind or load any local Rineya assets. `native`
returns exit code 2 deliberately. Changing `plan.json` cannot unlock missing code.
The remaining work is technical qualification and implementation, not a request
for blanket permission to repeat the user's already requested research work.

## Existing infrastructure: reuse explicitly, do not invent imports

`experiments/titan_tangent_arithmetic_v1/README.md` was successfully inspected on
`research/titan-tangent-task-arithmetic-v1`; its observed blob is recorded in
`SOURCES.md`. The same path was not found at EFE's chosen integration base.
It documents frozen-anchor JVPs, finite-difference diagnostics, data guards and a
native host-qualification protocol. It also distinguishes tangent-trained predictors
from nonlinear merges and teacher-forced scoring from autonomous generation.

A later integration should pin a current source commit, audit its tests/dependencies,
and deliberately port or integrate the needed components. Do not silently import
code from a developer's other worktree or replace its model qualification gates.

The supplied analysis also names
`experiments/rineya_composable_subspace_stack_v1`, but the named historical branch
`research/rineya-composable-subspace-stack-v1` could not be resolved during this
session. Treat that integration as **unverified and optional**, not installed.
Locate an accessible commit before depending on it. No unseen code was copied.

## Stage 0: establish admissibility and resource bounds

Record the anchor/tokenizer/config/delta hashes, stored precision, parameter names,
shapes, tied-weight identity, and checkpoint lineage. Explicitly reject incompatible
anchors, missing keys, nonfinite weights, unsupported quantization, or silent dtype
changes. Save software/device metadata and establish memory headroom without
terminating unrelated jobs. Avoid full Jacobian materialization and unnecessary
simultaneous copies of checkpoints on the local constrained host.

Bind train/calibration/preservation/confirmation roles to raw-content hashes and
parent families. Consult actual quarantine and spent-bank records; the reference
`audit_splits` only checks declarations and cannot prove freshness. Freshness requires
provenance review, including paraphrases and related solver templates. Set selection
budgets, family weights, acquisition floors, noninferiority margins, multiplicity
policy, paired confidence method, and stopping rules before inspecting outcomes.

All native outputs go to a new external session directory. Never write merged
weights over source models or overwrite historical reports. Hash the frozen recipe
and keep per-stage receipts. A failed gate stops that track, not a fallback to a
more favorable precision, dataset, or scoring method.

## Stage 1: functional allocation

Implement a native output-direction exporter with fixed-prefix/teacher-forced
coordinates and explicit output metric. Reuse qualified JVP infrastructure only
after binding it to the exact anchor. Test derivative signs, linearization scale,
shared parameters, dropout/eval mode, and actual nonlinear perturbations. Exclude
confirmation rows from all fits and regularizer searches.

The CPU `fit_coefficients` is an unbounded finite surrogate, with no checkpoint I/O
or trust region. A host must implement the registered bounds/solver, then freeze
coefficients and compare actual nonlinear autonomous generation and intended export
precision. Log the independent-grid control's full cost as well as the functional
method's derivative cost. Do not interpret low surrogate error as native efficacy.

## Stage 2: KV policies

Bind the native cache ABI first. Establish full-cache and no-eviction parity before
physical compaction. Preserve original positional counters and masks. Compare each
KV head's actual resident entries with `select_positions`; query-head multiplicity
must not invent independent stored KV copies. The scorer must operate under the same
protections and resident budget and must not read future tokens or answer labels.

The reference has no key/value tensors, GPU memory accounting, batching semantics,
or kernel implementation. Physical compaction can allocate workspace, so report
peak allocated memory as well as theoretical retained bytes. Keep decode seeds and
selector seeds separate. Use repeated selector seeds fixed before scoring, not
winner-seed selection. Tool-arrival/prefill protection policies must be explicit.

## Stage 3: finite refinement and only then DEQ

Train/evaluate the depth curve against a matched stateless baseline. Audit state
reset and causal state flow. Advance only after native quality gain and a useful
stable attractor are supported under the frozen criteria. Next qualify a small
sidecar's forward solver and matrix-free implicit backward pass. Record every
forward/backward iteration count, residual, condition estimate, solver failure,
output influence, and memory/latency observation.

The CPU dense adjoint must not be expanded into a dense LLM Jacobian. Its finite-
difference test establishes only a small affine example's algebra. Numerical
convergence without task benefit is a negative architecture result, not a pass.

## Results contract

Native reports must identify measurement kind, stage, commit, checkpoint/data/config
hashes, seeds, protocol status, per-row outputs/scorer decisions, and resource usage.
Separate `implementation_qualified`, `native_executed`, `hypothesis_supported`, and
`promotion_eligible`. Include paired gains/losses with denominators and all failed
or excluded arms with reasons. CPU smoke receipts are never native receipts.

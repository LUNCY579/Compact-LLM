# DSD-001 build verification

Implementation session: 2026-10-04; local execution receipts crossed into
2026-10-05 UTC. Parent: `03db218ceebc61a2bc04f2482cf7e094045e01c1`.

## Actually executed

**130 tests passed**, zero failures/skips, one nonfatal PyTorch profiler warning.
Python 3.13.5, PyTorch 2.10.0+cpu, pytest 9.0.2. Scoped compileall also succeeded.
The warning concerns profiler event clearing across cycles; the test uses one
cycle. No GitHub Actions/remote CI execution is claimed.

Tests cover full-vocabulary KL values/gradients/double gradients, teacher detachment,
causal offsets, response means, EMA endpoints/order, independent parameter-bank
copies, exception restoration, no-live-graph swaps, geometry calibration, occupancy,
all axes, zero-context behavior, scorers, data leakage/admission, paired endpoints,
rate cross-fitting, configuration/hash/output guards and CLI commands.

A tiny BF16 attention fixture exercises SFT and both rollout/KL paths, selected
FP32 updates, frozen-state preservation, exact starting conditions, no-update
qualification, end-to-end orchestration with a mocked native loader, failure
receipts and locally saved deltas. A mock is not a native Rineya execution.

An initial zero-context check exposed Adam amplification of roundoff in the naive
reverse-KL autodiff expression. Analytical logit derivatives preserve the same KL
objective and make equal represented distributions stationary. Independent first
and second derivative checks and the zero-context test pass after the correction.

## Controlled execution

**144 runs completed**: two task seeds (16,17), two rollout sources, two KL directions,
18 Appendix-F coupling rates, each for **200 updates** (28,800 total updates).
Reference geometry T4/K8/D64, lambda2.5, kappa.6, feature overlap.5, readout
compatibility0; Adam1e-3, clip10, no weight decay. All 585 nonterminal prefixes use
exact occupancy. No pretrained model, dataset or GPU is involved.

`evidence/controlled_reference.csv` contains every cell's acquisition, retention,
target-path fidelity and initial calibration error. `controlled_summary.json`
records geometry, ranges, source-report hash and limitations. A preceding 40-cell
five-rate diagnostic also ran, but is not pooled into the 144-cell cohort.

One tool wait exceeded its timeout; the process completed and its full 144-cell
JSON was then parsed and checked for unique cells, finite endpoints and 200-update
counts. This is disclosed rather than treating a timeout as a successful exit.

The two-seed slice is **not** the paper's 96-seed cohort, exact RNG replication,
complete regime map or evidence of Rineya benefit. No winner is promoted from it.

## Not executed or established

No exact Rineya weights or native Transformers model were loaded, no CUDA run was
performed, and no native acquisition/retention benefit or 8GB memory fit was tested.
Transformers installation into an isolated test environment failed due unavailable
network/DNS; the existing environment was not upgraded. Native qualification and
explicit operator training remain required. No production or source checkpoint
was modified. All repository changes are additions in the dedicated subtree.

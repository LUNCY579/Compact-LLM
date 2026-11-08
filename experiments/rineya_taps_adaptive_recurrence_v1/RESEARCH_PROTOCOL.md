# TAPS-RINEYA-001: frozen inference-only protocol

Status at registration: no implementation results or native measurements.
Primary question: is recurrent update SCALE a useful control axis beyond fixed
scale, loop count and existing gates? No training, architecture rewrite, model
registry change, protected data access, or promotion by smoke is authorized.

## Hosts and budgets

First native target: historical CPU-FP32/eager Supa-A1.1 ARMR L12 or L12-13,
full-prefix batch one, no padding, no KV, frozen weights and unchanged custom
abstention-aware head. L0-2 are always one-shot. Start with ONE extra application;
zero extra applications is A0. Any larger budget is a separately gated mechanical
screen, not automatically licensed by this harness. Existing ARMR application-3
failure must not be overwritten. Initial normal block evaluation always takes
unit step and seeds history for every instrumented arm. Labels never enter
controller/proposal inputs.

CPU fixture budgets: extra applications {0,1,2,4}; schedule seeds {17,29}.
Fixtures are small known dynamics, NOT reasoning tasks or a Rineya benchmark.
Native DEV: reuse complete existing MERGEDEV-fast and existing mechanical smoke
through ARMR's exact path allowlist/scorer, no favorable subset. No new split
may masquerade as a historically protected split. Preserve family/parent IDs.
Native TEST stays closed in v1; a separate approved frozen policy and dataset
binding are required before a future protected run. Fixture TEST only validates
the apparatus. The generic calibration interface refuses protected parents and
non-DEV rows before any score reads or host execution.

## Registered controls

A0: ordinary native first application, no extra recurrence.
A1: unit recurrence. A2: fixed alpha in {.8,.9,1,1.1,1.2}, applied only to extra
updates. This conservative grid is not asserted numerically qualified natively.
A3: Algorithm-7 TAPS, beta=.8, gamma=1.5, rho=.275, bounds [.8,1.2], warmup=1,
eps=1e-12, q=1, per-sequence element mean. Source: Qwen scheduler settings in
Appendix C.1, adapted to FULL ARMR residual (NOT the paper's eta/3 Euler host).
A4-example: retrospective best constant from A2 for each example.
A4-step: exhaustive finite-grid schedules only when grid_size**extra_loops is
within the registered cap 25. Otherwise explicitly NOT_ENUMERATED, never replaced
by a greedy search labeled upper bound. Search work is recorded separately.
A5: seeded random draws from the DEV A3 empirical alpha pool for that budget.
This matches the DEV marginal distribution in expectation, not sample-by-sample
or TEST trajectories; it does not prove identical conditional distributions.

Best deployable fixed alpha is selected on DEV independently for each registered
budget (score, then smaller alpha for deterministic ties), frozen with config,
DEV-record digest and protected-parent binding. Also report the retrospective
best GLOBAL fixed score on each evaluated split as a NONDEPLOYABLE diagnostic
envelope. Do not confuse that envelope with DEV-selected performance.

Headroom = finite-grid oracle accuracy - retrospective best-global-fixed accuracy.
Capture = (TAPS accuracy - best-global-fixed accuracy)/headroom when headroom
exceeds 1e-12; otherwise null. Keep negative and >1 ratios: TAPS uses continuous
alphas, so a finite-grid oracle is not an upper bound on all real-valued policies.
Compute both constant-selector and enumerated-step versions, with explicit labels.
Constant-selector null does NOT rule out per-step headroom. Binary accuracy may
hide proxy-loss headroom; report both without choosing the favorable endpoint.

## Diagnostics and fairness

For every observed block update log raw delta norm, previous-delta/EMA cosine,
P, S, P/(S+eps), balance, alpha, state norm, actual movement, controller/core
seconds and iteration. Optional terminal readout/logit movement and loss are
kept outside the causal controller. Undefined cosine is null, not invented zero.
Raw proposals are recomputed live for every arm; saved baseline trajectories are
never rescaled as though they were new rollouts. Stream scalar JSONL in exclusive
run directories. Native output fields unavailable at an internal block are null;
final first-token logit/NLL/task metrics remain available through inherited scorer.

Fixed-loop results isolate scale. Curves over registered loop budgets expose
loop-count effects. Accuracy-target loop/time results are retrospective summaries
of complete fixed-budget runs, NOT an online label-aware stopping policy. Count
initial block, extra block calls, block width, diagnostic readouts, all oracle
search calls and controller overhead separately. Native full-prefix token work is
not interchangeable with fixture core calls. Native timing needs paired repeated
runs with rotated arm order, synchronized CUDA if ever qualified, and diagnostics-
on/off timings; instrumented CPU fixture timing is NOT deployable speed evidence.

## Safety and decision gates

No checkpoint download, optimizer, training entrypoint, GPU fallback or author
pickle loading. Native preflight binds local source/checkpoint hashes, inherited
config/data/scorer and exact runtime before import/load; requires >=7.4 GiB free
host RAM (new conservative guard above historical 4.34 GiB peak), bounded context,
per-case deadline, scalar streaming and no unnecessary retained GPU tensors.
Failure is an append-only BLOCKED/FAIL receipt, not a reason to weaken a guard.
The CPU-FP32 lane does not claim to qualify BF16/FP16 or an 8-GiB GPU lane.

Interpretations required: useful scaling axis, adaptive headroom, predictiveness
of dynamics, fixed-loop accuracy, loop savings, actual overhead, cross-task
transfer, interaction with existing gates. Fixture-only delivery => native
recommendation BLOCKED. Native null headroom => reject this selection family,
not recurrence generally. A native follow-up requires paired gains beyond the
DEV-fixed control, independent task/parent replication and acceptable overhead.
Co-design remains NOT_ATTEMPTED unless native adaptive headroom, TAPS benefit or
predictive statistics justify a separately registered training budget.

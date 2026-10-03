# Candidate successor: protect the proposed weight update

This is a mathematical design note, not a trained-model result. Keep CS-PILOT-001 unchanged.

The pilot tests an additive old-task KL objective. A potential next test is a constrained
update with explicit per-family tolerances, using the same source-bound anchor cache.
It should only be tried after inspecting the completed matched pilot.

## Why raw gradient protection is insufficient

Let the new-task gradient be `g_new=(1,1)` and an old-family gradient be `g_old=(1,-0.9)`.
Their dot product is `0.1`. Ordinary descent `d=-g_new` has old-loss directional derivative
`g_old dot d=-0.1`, so a raw-gradient non-conflict check passes.

An adaptive preconditioner `D=diag(0.1,10)` instead proposes `d=-D*g_new=(-0.1,-10)`.
Now `g_old dot d=8.9`: the actual step raises the old loss to first order. Adam moments
can further change this direction. Therefore constraints belong on the proposed **parameter
increment**, including momentum/preconditioning effects, not only on the current gradient.

Even projection onto `g_old dot d <= 0` gives only a local statement. For the smooth old
loss `L(theta)=0.5*||theta+g_old||^2` at `theta=0`, a nonzero tangential step has zero
first-order change but positive finite change `0.5*||d||^2`. No global no-forgetting
guarantee follows from satisfying the linear constraint.

## A testable, bounded procedure

1. Form the new-objective optimizer proposal, retaining the pre-step parameters and optimizer
   state. Use the same scope, rank and training exposure as the matched CE baseline.
2. Compute old-family gradient constraints or a small Fisher approximation on anchor examples.
   Do not combine all families into one average constraint. Project/scale the actual proposed
   increment within the declared trust radius. If the direction is infeasible for useful
   acquisition, record a zero/rejected step instead of silently relaxing retention limits.
3. Evaluate each family's **finite** anchor loss or source-relative KL at the proposed state.
   Accept only within fixed, predeclared tolerances. Otherwise backtrack a bounded number of
   times. Restore optimizer moments consistently on rejected proposals and count all work.
4. Evaluate untouched development examples, including previously source-incorrect examples,
   after the fixed training budget. Compare acquisition and individual losses to CE and KL.
   Anchor acceptance does not prove generalization beyond the anchors.

At exact source initialization, `KL(source || candidate)` and its gradient are zero. Setting
a zero KL tolerance would generally reject every useful change; a finite tolerance is needed.
An additive KL penalty also has no first-update force at the source optimum. That is expected
mathematically, not a caching bug. A trust-region test can detect an excessive proposed first
step, although it costs candidate evaluations.

For LoRA, projection in A/B coordinates and projection in effective `B@A` weight space are
different. Report that choice explicitly. Gauge changes `A→cA, B→B/c` preserve the effective
delta while changing factor norms; this is why `adapter_geometry.py` measures effective
weight-space norms and directions. A projected dense increment may exceed the permitted
rank, so it must not be presented as a free rank-constrained update.

## Predeclared comparisons required

- CE, additive KL, and constrained proposal at matched new-task exposure.
- Cost-aware comparison including extra backward passes and rejected-step work.
- Acquisition-matched and effective-norm-matched CE controls where feasible.
- Per-family old-item losses, source/previous/ever-correct retention, and both task orders.
- A check for plasticity collapse: preserving the base by rejecting all updates is a failure
  to acquire, not success at continual learning.

The anticipated benefit is a more explicit, auditable acquisition/retention tradeoff. It is
not established superiority, and it must not consume confirmation data during development.

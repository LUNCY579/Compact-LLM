# Joint retention constraints for skill composition

Status: prospective mathematics and exact arithmetic examples. No model, bank,
optimizer, execution grant, or change to CE8 qualification is involved.

This extends `retention-projection-v1/RETENTION_PROJECTION.md`. Its empirical
smoothness and gradient-error assumptions remain necessary and unmeasured for
the study models. The result concerns the specified loss groups, not every
behavior, decoding outcome, or meaningful acquisition.

## A joint robust feasible set

At one fixed anchor, define for each retained group

    q_j(d) = ghat_j^T d + sigma_j ||d||_2 + (L_j/2) ||d||_2^2.

Here L_j >= 0 must bound gradient variation on every tested update segment,
and sigma_j >= 0 must bound the Euclidean gradient-estimation error. The
registered allowance is epsilon_j >= 0. Restrict ||d|| <= R to a region where
the bounds hold. The intersection

    C = {d : q_j(d) <= epsilon_j for all j, ||d|| <= R}

is convex, closed, and contains zero. For finite R it is compact. With a fixed
positive definite M, minimizing (d-u)^T M (d-u)/2 over C has a unique solution.
This proves neither feasibility of a desired acquisition margin nor skill gain.

If all allowances are strictly positive and R > 0, zero is a Slater point.
Then the following KKT conditions are necessary and sufficient. Without a
constraint qualification, their sufficiency remains, but finite multipliers
need not characterize every optimum.

    M(d-u) + sum_j lambda_j (ghat_j + L_j d + sigma_j z_j) + nu d = 0
    lambda_j >= 0, nu >= 0
    lambda_j (q_j(d)-epsilon_j) = 0
    nu (||d||^2-R^2) = 0.

For d != 0, z_j = d/||d||. At zero, each z_j can be any vector of norm at most
one. The radius constraint uses (||d||^2-R^2)/2. Thus for nonzero d,

    d = [M + (sum_j lambda_j L_j + nu
              + sum_j lambda_j sigma_j/||d||) I]^-1
        [M u - sum_j lambda_j ghat_j].

This is an implicit characterization, not a closed-form multi-group solver.
Complementarity, the norm, and all group inequalities still need to be solved
and certified together. Averaging group constraints can hide a violated group.

## One projection per group is insufficient

Consider two exact halfspaces in two dimensions:

    C1: x <= 0; C2: y <= x; proposed u = (1,1).

Projecting onto C1 and then C2 gives (1/2,1/2), which violates C1. Reversing
the order gives (0,1), which violates C2. Neither order solves the joint problem.
The joint projection is (0,0): feasible x and y are both nonpositive, so each
squared distance from one is at least one, with equality only at zero.

This is a counterexample to a single sweep of group projections, not a claim
that properly converged alternating projections or Dykstra's method cannot
find an intersection. Parameter-update order also changes optimizer state and
the anchor; these halfspace examples do not model those additional effects.

## Convex mixing has a conditional guarantee

If all donors d_k belong to the SAME C at the SAME anchor, then nonnegative
weights with sum alpha_k <= 1 yield sum alpha_k d_k in C, because the remainder
weight can be assigned to zero. This follows from convexity of every q_j and
the trust ball. It requires common bounds, groups, anchor, and allowances.

An unscaled sum, a negative coefficient, or an extrapolation has no such
guarantee. Even a feasible convex mixture can erase acquisition. A sequential
training permutation cannot inherit this mixture result: its deltas are
computed at different anchors and may use different optimizer states.

## A joint retention requirement can leave no acquisition movement

Let two exact quadratic upper losses have gradients +e1 and -e1, both L=2,
and both allowances zero. Their constraints are

    x + ||d||^2 <= 0; -x + ||d||^2 <= 0.

Adding them gives 2||d||^2 <= 0, hence d=0. A new objective requiring a strictly
positive movement cannot be acquired within these particular constraints.
This example establishes a feasibility limitation, not a universal claim that
retention and acquisition conflict in neural models.

## Evidence and prospective use

`exact_joint_examples.py` checks the projection-order counterexample, the joint
minimum, a robust one-dimensional KKT boundary, convex mixing, extrapolation
failure, and the opposing-quadratic identity using `fractions.Fraction`.
These are exact mathematical examples only. An empirical experiment would
still need separately registered groups, measured or justified bounds, a
numerically certified solver, acquisition and retention endpoints, frozen
source/anchor/runtime identities, and its own phase grant. Existing stop
decisions and the six-update TRAIN-only qualification remain unchanged.

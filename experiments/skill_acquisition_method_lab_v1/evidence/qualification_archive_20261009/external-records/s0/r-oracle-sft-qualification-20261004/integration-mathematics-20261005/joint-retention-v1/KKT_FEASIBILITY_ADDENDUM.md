# Joint-retention KKT feasibility addendum

This corrects the displayed KKT system in `JOINT_RETENTION.md` without changing
its original source or exact-example receipt. The system must include primal
feasibility explicitly:

    q_j(d) <= epsilon_j for every j; ||d|| <= R.

Stationarity, nonnegative multipliers, and complementarity alone are
insufficient: choosing d=u and all multipliers zero satisfies those displayed
relations even when u is infeasible. Interpret the corrected system as primal
feasibility together with every relation shown in the original block.

The domain assumptions are finite-dimensional real vectors, finitely many
retention groups, finite gradients and proposed update, finite nonnegative
L_j, sigma_j and epsilon_j, finite R >= 0, and a symmetric positive definite
matrix M. The compact-set existence and unique-minimum claim then applies.
Slater's condition is available from zero when all epsilon_j > 0 and R > 0.
At R=0 or another degenerate feasible boundary, do not assert necessary finite
KKT multipliers without a constraint qualification. KKT sufficiency still
requires the explicitly included primal feasibility.

All statements remain conditional on the stated valid empirical loss bounds.
No model quantity is measured, no behavioral retention or acquisition result
is claimed, and no registered experiment or CE8 scope is changed.

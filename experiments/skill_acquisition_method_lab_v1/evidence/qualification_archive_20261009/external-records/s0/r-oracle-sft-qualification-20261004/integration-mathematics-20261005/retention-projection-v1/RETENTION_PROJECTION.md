# Prospective retention projection for acquisition and composition

Status: mathematical proposal and exact arithmetic evidence. No execution grant, protocol change, model gain, or retention result follows from this document. It extends the existing eight demonstrations; their counterexamples and previous stop decisions remain valid.

## Conditional thesis

For a registered TRAIN retention group j, let ell_j be its fixed empirical loss at the exact anchor theta, g_j = grad ell_j(theta), and epsilon_j >= 0 its prospectively registered loss allowance. Suppose grad ell_j is L_j-Lipschitz on every point of the segment theta + t delta, 0 <= t <= 1. Then

    ell_j(theta + delta) - ell_j(theta)
      <= q_j(delta) = g_j^T delta + (L_j/2)||delta||_2^2.

Thus q_j(delta) <= epsilon_j for every group is a sufficient empirical group-loss constraint. The proof is the fundamental theorem of calculus: integrate g_j^T delta plus [grad ell_j(theta+t delta)-g_j]^T delta; Cauchy-Schwarz bounds the latter by t L_j ||delta||^2, whose integral is L_j ||delta||^2/2. Convexity of the original loss is unnecessary. L_j must be a valid upper bound, not an observed average Hessian eigenvalue.

If the bound is only local within radius rho, also require ||delta|| <= rho and a neighborhood containing the whole segment. This is a local conditional statement. L_j, gradients, Jacobians, Hessians and numerical errors for the study models are currently unmeasured. TRAIN group loss does not certify every case, decoding, held-out performance, or meaningful skill acquisition.

With estimated gradient ghat_j and a *valid* norm error bound ||g_j-ghat_j|| <= sigma_j, replace q_j by

    ghat_j^T delta + sigma_j ||delta|| + (L_j/2)||delta||^2.

Without justified sigma_j and L_j, this expression is a diagnostic rather than a certificate. A numerical solver also needs an error allowance with the correct direction: its proven constraint upper bound must be <= the registered epsilon_j.

## Closest feasible acquisition update

Given a raw proposed update u and a fixed positive definite metric M, solve

    minimize_delta  (1/2)(delta-u)^T M(delta-u)
    subject to     q_j(delta) <= epsilon_j for each j,
                   ||delta|| <= rho when needed.

For fixed L_j >= 0 this is a convex problem even when the actual neural loss is nonconvex. Zero is feasible for nonnegative allowances. A positive definite M makes the minimizer unique. With strict feasibility, the KKT conditions are necessary and sufficient:

    M(delta-u) + sum_j lambda_j(g_j + L_j delta) + nu delta = 0
    lambda_j >= 0, q_j(delta) <= epsilon_j,
    lambda_j(q_j(delta)-epsilon_j) = 0;
    nu >= 0 and nu(||delta||^2-rho^2) = 0.

Here the radius constraint is written (||delta||^2-rho^2)/2 <= 0. For one constraint and no active radius,

    delta(lambda) = (M + lambda L I)^(-1)(M u - lambda g).

If u is feasible, lambda=0. Otherwise an active root satisfies q(delta(lambda))=epsilon. Where differentiable,

    d q(delta(lambda))/d lambda
      = -(g+L delta)^T(M+lambda L I)^(-1)(g+L delta) <= 0.

This supports a scalar root calculation under regularity; it is not permission to execute a solver. Degenerate singleton feasible sets can fail constraint qualification: g=0, L>0, epsilon=0 requires delta=0, and a finite multiplier may not represent the solution. Do not assert KKT completeness in that case.

### Closed forms for one constraint

For M=I and L>0, completing the square gives a ball with

    c = -g/L; r^2 = ||g||^2/L^2 + 2 epsilon/L.
    delta* = u                          if ||u-c|| <= r,
             c + r (u-c)/||u-c||         otherwise.

For r=0 the answer is c. For L=0 and nonzero g the halfspace solution in metric M is

    delta* = u - max(0, (g^T u-epsilon)/(g^T M^(-1)g)) M^(-1)g.

For L=0, g=0, epsilon>=0 there is no restriction. A halfspace is a finite-step certificate only when the zero-curvature upper bound itself is valid; otherwise it is first-order evidence.

Projection can remove all useful movement or turn a raw new-loss descent direction into ascent. Retention feasibility is not acquisition. For a separate valid new-loss smoothness bound, an acquisition certificate would additionally need

    g_new^T delta + (L_new/2)||delta||^2 <= -gamma, gamma>0.

Its feasibility is not guaranteed, and a loss improvement still does not establish discrete skill gain.

## What the metrics do and do not establish

Euclidean task-vector cosine measures parameter geometry. It need not equal g_j^T delta, and even g_j^T delta=0 permits positive curvature drift. A gradient halfspace corrects that first-order mismatch; the smoothness ball additionally prices a bounded finite-step remainder.

For twice differentiable losses, a justified PSD upper matrix B_j dominating the Hessian along the segment permits the sharper bound g_j^T delta + delta^T B_j delta/2. An anchor Hessian alone is insufficient. An indefinite Hessian is not a convex upper metric, and clipping its negative eigenvalues at the anchor does not prove domination elsewhere.

A PSD functional metric G=J^T W J describes linearized output change when W is PSD. It can be singular and ignores higher-order effects; J, W and the remainder need separate justification. M in the objective controls how expensive changing u is. M, the retention upper matrix B_j, and the functional metric G have distinct roles. A regularized M may be useful computationally, but choosing it does not certify retention. Under theta=A phi, objective and curvature matrices transform by A^T M A and A^T B_j A, while gradients transform by A^T g_j; resetting every transformed matrix to identity changes the intervention.

## Composition and permutation

Independent feasible deltas at one anchor do not imply their sum is feasible. For total Delta=sum_k delta_k, the isotropic bound includes L sum_(a<b) delta_a^T delta_b. Recheck the complete registered sum and its segment, including its floating-point arithmetic, rather than inherit component certificates.

Sequential updates can instead telescope: if at each actual theta_k every group satisfies ell_j(theta_(k+1))-ell_j(theta_k) <= epsilon_(j,k), then final drift <= sum_k epsilon_(j,k). This requires bounds at each changed anchor and a precommitted total allowance; it does not use a stale initial gradient. Different training orders change anchors and optimizer state. Record the optimizer state and order as part of each intervention. No equality of their outcomes follows from projection.

## Exact examples

The companion stdlib Fraction script checks seven newly formulated examples:

1. A smoothness-ball projection reaches its boundary with exact KKT stationarity; a Euclidean-orthogonal task-vector update violates the same loss constraint.
2. Underestimating L changes a claimed zero-drift update into positive drift.
3. An anisotropic objective gives a different, lower-cost halfspace correction than Euclidean projection.
4. Two group constraints retain each group's allowance; averaging them can conceal an individual loss increase.
5. A retention optimum with zero tolerance leaves only delta=0.
6. A retained halfspace projection turns a raw new-loss descent direction into ascent.
7. Two individually feasible finite updates have an infeasible sum because of the cross term.

These are exact mechanical mathematical checks, not neural training, model loading, or evidence that any study model acquires a skill.

## Separate prospective registration needed

Keep current CE8 qualification scope unchanged. A later experiment would need its own exact source, anchor, TRAIN groups, runtime, physical update count, resource/accounting caps and fresh phase grant through the existing owner. No DEV, confirmation evaluation, CE375 modification, promotion or production action is authorized here.

Before estimating any gradients or curvatures, register the empirical loss definition, group memberships, diagnostic sample selection, measurement workload, valid bound assumptions, trust radius, solver metric/tolerance, gradient uncertainty treatment, and failure policy. Register acquisition and per-group retention endpoints independently; preserve existing thresholds and stop decisions. Define whether a comparison studies a projected optimizer update, a post-training weight merge, or sequential training order. Do not mix these as one intervention.

A concrete future comparison is raw u versus Euclidean gradient halfspace versus the smoothness-constrained projection, with common anchor and matched physical update/data budget. Any layer/subspace restriction must be a separately frozen factor, not a post-hoc choice after outcomes. Pilot loss/curvature measurements would remain empirical local diagnostics unless the stated upper bounds are actually justified. Evidence of meaningful acquisition requires the separately registered behavioral endpoints and their allowed evaluation phase.

## Primary paper context

[Adaptive Weight Disentanglement, section 3.2](https://arxiv.org/html/2411.18729v2) obtains its cosine argument after a first-order expansion and substitution of a scaled task vector for the gradient. The projection above is an independent conditional derivation using the actual loss gradient and a remainder upper bound; it does not assert AWD's assumptions hold here.

[Understanding and Enforcing Weight Disentanglement, appendix E.2](https://arxiv.org/html/2604.17078v1) supplies task-feature/NTK context and an approximately whitened covariance assumption in its geometric analysis. No whitening claim is made for this study.

[Task Singular Vectors](https://arxiv.org/abs/2412.00081v3) studies layer task-vector singular representations. A frozen subspace could be a prospective factor, but neither a subspace nor low cosine establishes the constraints above or behavioral success.

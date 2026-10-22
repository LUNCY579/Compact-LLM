# Activation metric geometry for a fixed LoRA layer

Status: conditional mathematics and exact rational toy examples. No models,
scientific rows, activation measurements, optimizer updates, Native dispatch,
evaluation, grants, or changes to qualification eligibility are involved.

## Scope and relation to prior notes

The earlier local LoRA notes derived a Frobenius Gram of reconstructed weight
deltas and showed why factor coordinates depend on the LoRA gauge. The
retention projection notes already distinguished a PSD functional metric from
a loss upper bound. This extension specializes the geometry to a common layer
input distribution and a common output metric, gives the resulting low-rank
trace formula, and treats singular weighted compression explicitly. Whitening
and truncated SVD are established linear algebra; no new algorithm or general
retention theorem is claimed.

All comparisons assume the same anchor, layer, input/output coordinates and
ordinary additive LoRA convention. Let x in R^n have a finite second moment,
and let

    X = E[x x^T] >= 0,
    Q = Q^T >= 0,                 Q in R^(m x m), fixed independently of x,
    D_i = s_i B_i A_i,            B_i in R^(m x r_i), A_i in R^(r_i x n),
    z_i(x) = D_i x.

Scales s_i are finite real numbers and must be included. Different r_i are
allowed. The algebra does not cover arbitrary adapter variants, changed
features/anchors, nonlinear gates, or non-LoRA saved modules.

X is the **uncentered second moment**, not generally the covariance. If
mu=E[x] and Sigma=Cov(x), then X=Sigma+mu mu^T. Replacing X by Sigma changes
the measured quantity by the mean-response term

    (D_i mu)^T Q (D_j mu).

Covariance suffices for this formula when mu=0, or when one explicitly wants
the metric of centered responses instead. A deterministic nonzero input has
zero covariance and can still have a nonzero LoRA response.

## The Gram and its low-rank computation

Define the bilinear functional metric

    <D_i,D_j>_(X,Q) = E[(D_i x)^T Q (D_j x)].

Using E[x^T M x]=tr(M X), valid for every finite matrix M under the stated
moment assumption, and cyclic invariance of trace,

    K_ij = tr(D_i^T Q D_j X)
         = s_i s_j tr(A_i^T B_i^T Q B_j A_j X)
         = s_i s_j tr[(B_i^T Q B_j)(A_j X A_i^T)].

The first factor is r_i by r_j, the second r_j by r_i, so their product is
r_i by r_i. This orientation remains valid for mixed ranks. It avoids
materializing D_i or D_j when the needed factor contractions can be computed;
it does not make estimating or storing X and Q free.

Let Q^(1/2), X^(1/2) be their principal symmetric PSD square roots, and put

    W_i = Q^(1/2) D_i X^(1/2).

Then K_ij=<W_i,W_j>_F. Consequently K is symmetric PSD, and for any finite
real coefficient vector c,

    c^T K c = ||sum_i c_i W_i||_F^2
            = E[||Q^(1/2) (sum_i c_i D_i)x||_2^2] >= 0.

This is a norm of the combined layer response under the chosen metric; it
does not assert additivity of neural losses or task performance. Q=X=I
specializes to the earlier Frobenius Gram. Non-PSD Q or X does not support
the nonnegative squared-response interpretation.

For K_ii,K_jj>0 one can define

    cosine_(X,Q)(i,j) = K_ij / sqrt(K_ii K_jj).

When either diagonal is zero, the cosine is undefined. Adding an arbitrary
epsilon to the denominator changes the diagnostic; it does not identify a
functional direction that the metric cannot observe.

## Gauge invariance and observable equivalence

For invertible R_i in R^(r_i x r_i), the change

    B_i -> B_i R_i,       A_i -> R_i^(-1) A_i

leaves D_i, z_i, W_i and every K_ij unchanged, with s_i fixed. The low-rank
trace expression therefore inherits invariance from the actual layer map,
even for a nonorthogonal R_i. Separate factor cosines are not used.

Because X and Q may be singular, the metric is in general a seminorm:

    ||N||_(X,Q)=0  iff  Q^(1/2) N X^(1/2)=0.

Equivalently, Q^(1/2) N x=0 almost surely: the expectation of its nonnegative
squared norm is zero. In particular,

    N X^(1/2)=0  =>  N x=0 almost surely.

Thus D and D+N can be distinct parameter deltas with exactly the same
activation response on the stipulated input distribution. If Q is also
singular, indistinguishability in this metric can hold while their actual
responses differ in ker(Q). Neither conclusion controls inputs outside the
chosen support. A nonzero parameter delta can be metric-null.

## Euclidean delta cosine can disagree with local functional geometry

The companion script verifies the following exact examples with Q=1:

* D_1=[2,1], D_2=[2,-1] have Frobenius cosine 3/5. For input x=e_2,
  X=diag(0,1), their responses are +1 and -1, so functional cosine is -1.
* [1,1] and [1,-1] are Frobenius orthogonal but have the same functional
  opposition on x=e_2.
* [1,2] and [1,-2] have Frobenius cosine -3/5. On x=e_1 their functional
  cosine is +1.
* D=[0,1] has nonzero Euclidean norm but zero response for X=diag(1,0).
  Its functional cosine with an observed direction is undefined.

These examples concern response alignment, opposition or irrelevance in the
specified local metric. A negative K_ij is not by itself evidence of harmful
task interference; a positive K_ij is not evidence of compatible acquisition.

## When the output metric depends on the input

A downstream Jacobian or Hessian surrogate usually can vary with x. For a
PSD matrix Q(x), the exact definition becomes

    K_ij = E[x^T D_i^T Q(x) D_j x].

Its Gram remains PSD whenever the expectations are finite. The fixed-Q
factorization above generally cannot be recovered by replacing Q(x) with
E[Q(x)]:

    E[Q(x) tensor (x x^T)] need not equal E[Q(x)] tensor E[x x^T].

An exact scalar counterexample has P(x=0)=P(x=2)=1/2, Q(x)=x^2 and D_i=D_j=1.
Then K_ij=E[x^4]=8, whereas E[Q(x)] E[x^2]=2*2=4.

For a finite weighted support with probabilities p_t, use the paired sum

    K_ij = sum_t p_t s_i s_j
           tr[(B_i^T Q(x_t) B_j)(A_j x_t x_t^T A_i^T)].

It preserves the association between x_t and Q(x_t). Column-wise
vectorization gives the general PSD parameter operator
H=E[(x x^T) tensor Q(x)]. Approximating H by a Kronecker product is an
additional assumption/approximation requiring its own error analysis.
Bilateral whitening by one X and one Q solves the separable fixed-Q problem
below; it does not solve arbitrary H-weighted rank approximation.

## Rank-r compression in the separable PSD metric

For a target layer map T, which may be a fixed weighted sum of deltas, consider

    minimize_(rank(D)<=r) J(D),
    J(D) = E[||Q^(1/2)(D-T)x||_2^2]
         = ||Q^(1/2)(D-T)X^(1/2)||_F^2.

Assume finite-dimensional real matrices, fixed PSD X,Q and an integer r>=0.
Let

    C = Q^(1/2) T X^(1/2),
    C_r = a truncated SVD of C retaining at most r singular components,
    D_* = Q^(dagger/2) C_r X^(dagger/2).

Here Q^(dagger/2) is the Moore-Penrose inverse of Q^(1/2), and similarly for
X. This is well-defined even when X or Q is singular.

### Proof, including singular support

For every rank-at-most-r D, N=Q^(1/2) D X^(1/2) also has rank at most r.
Let P project onto range(N). Then

    ||C-N||_F^2 >= ||(I-P)C||_F^2
                  = ||C||_F^2-tr(P C C^T)
                  >= sum_(k>r) sigma_k(C)^2.

To see the last inequality, diagonalize C C^T. In that eigenbasis the diagonal
entries of an orthogonal projector lie in [0,1], sum to rank(P)<=r, and thus
their weighted sum with descending nonnegative eigenvalues is at most the
sum of the top r eigenvalues. This proves the lower bound without assuming
that the input/output metrics are invertible.

C satisfies P_Q C P_X=C, where P_Q and P_X project onto the ranges of Q and
X. Every nonzero left/right singular vector of C lies in the corresponding
range. Therefore C_r also satisfies P_Q C_r P_X=C_r. The pseudoinverse lift
has rank at most r, and

    Q^(1/2) D_* X^(1/2) = P_Q C_r P_X = C_r.

It attains the lower bound, so

    min J(D) = sum_(k>r) sigma_k(C)^2.

If r>=rank(C), the minimum is zero. This need not reconstruct T as a dense
matrix. If either metric is zero, every feasible D has objective zero.
Singular-value ties can make the retained subspace nonunique.

A feasible null addition Z with Q^(1/2) Z X^(1/2)=0 has the same objective,
but D_*+Z is a minimizer of the rank-constrained problem only if
rank(D_*+Z)<=r still holds. Arbitrary null additions can increase rank.
For an SVD C_r=U_r Sigma_r V_r^T, an explicit LoRA representation with scale
one is

    B_*=Q^(dagger/2) U_r Sigma_r^(1/2),
    A_*=Sigma_r^(1/2) V_r^T X^(dagger/2).

Nonzero adapter scale can be absorbed into a factor. Rank-zero and zero
singular components require the corresponding zero representation.

### Exact compression examples

With T=diag(2,1), Q=I and X=diag(1,9), ordinary Euclidean rank-one truncation
diag(2,0) has weighted error 9. Whitening gives C=diag(2,3); keeping its second
component and lifting gives D_*=diag(0,1), with weighted error 4. The
activation-weighted optimum keeps a different direction.

With X=Q=diag(1,0) and T=[[2,7],[11,13]], rank-one D_*=diag(2,0) has zero
metric error although ||T-D_*||_F^2=339. Input-null and output-null entries
are unobserved. Adding diag(0,1) keeps the weighted error zero but raises the
rank to two, so it is infeasible for r=1.

## Interpretation and prospective limits

This metric is exact for the stated linear layer response, distribution and
fixed Q. Across a nonlinear network, inputs and downstream maps can change
when adapters are composed. Layer metrics omit such changes, cross-layer
couplings, higher-order effects and decoding discontinuities unless those
are separately modeled and bounded.

Even the optimal rank-r solution minimizes one weighted mean squared local
error. It can discard a direction needed by a low-weight input group. It does
not guarantee all-skills accuracy, full-model logits, retention, acquisition,
held-out performance, or unchanged decoding. No unmeasured covariance,
Jacobian, curvature, rank or behavioral assumption is asserted for the study.

Any later empirical use needs separate prospective definitions of the common
anchor/layer/scales, admissible input population, X or paired Q(x) estimator,
PSD handling and uncertainty, group weighting, rank and compression policy,
and independent behavioral endpoints. This document supplies no such
execution authority and leaves all existing qualification packets and stop
decisions intact.

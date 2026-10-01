# Mathematical analysis for a bounded R/S acquisition method study

Date: 2026-09-30. Status: planning and analytical derivation. This document does
not grant native training, attest a donor, or report a model-quality result. It
develops the mathematical distinctions needed to compare acquisition,
composition, and retention on one exact Rineya anchor. Equations identified as
original derivations below are calculations for this study; cited experiments
remain evidence about their original models and tasks.

## 1. Current evidence and the common anchor

The completed source competence receipt contains 450 CALIBRATION prompts,
150 each for P, R, and S, with eight completions per prompt. Strict successes
are **0/3,600**: P 0/1,200, R 0/1,200, and S 0/1,200. All 450 strict-reward
groups are all failure; none is mixed or all success. The content diagnostic
finds R 2/1,200 and S 56/1,200, with P 0/1,200. Those diagnostic successes do
not replace the binary reward. These are native inference observations from
the bound source and decoding protocol, not acquisition training. See the
[source competence receipt](C:/Users/clclc/.codex/worktrees/comp-rs-native-q1/ChatBubble-titan-model-soup-mini-v1/experiments/maxrl_comp002_pretraining_v1/evidence/S0_COMPETENCE.json).

Zero observed strict success does not prove that every prompt has population
success probability zero. The prompts are heterogeneous and share parent
worlds, so treating all 3,600 completions as independent observations of one
universal success parameter would overstate the evidence. It does establish
that the recorded groups supply no centered GRPO/MaxRL acquisition update.
Changing the diagnostic parser into a reward after seeing this result would
change the experiment.

The proposed common warmstart is one epoch of supervised training on the
audited 600-row WARMSTART bank, with 75 accumulated updates under the declared
recipe. Its native outcome remains pending in this analysis. Implementation
and synthetic tests are not evidence that it repairs the source's strict
competence. Its eight-prompt envelope check is descriptive and cannot stand
for a complete competence or preservation enrollment. See the
[warmstart implementation](C:/Users/clclc/.codex/worktrees/comp-rs-native-q1/ChatBubble-titan-model-soup-mini-v1/experiments/maxrl_comp002_pretraining_v1/s0_warmstart.py)
and [source snapshot](C:/Users/clclc/.codex/worktrees/comp-rs-native-q1/ChatBubble-titan-model-soup-mini-v1/experiments/maxrl_comp002_pretraining_v1/evidence/WARMSTART_SOURCE_SNAPSHOT_20260930.json).

Let the original source be theta_b. If the common warmstart completes and its
rebuild is qualified, define theta_a = theta_b + delta_warm as the immutable
acquisition anchor. Otherwise theta_a has not been established. Every compared
R and S donor must start from the same theta_a, not separately warmstarted
states. Define delta_R = theta_R - theta_a and delta_S = theta_S - theta_a.
The intended composition is

\[
\theta_{RS}=\theta_a+\alpha_R\delta_R+\alpha_S\delta_S.
\tag{1}
\]

Subtracting theta_b from warmed donors and then adding their deltas to theta_b
would count the warmstart twice at unit coefficients. More generally, it
would apply (alpha_R + alpha_S) delta_warm instead of one delta_warm. The
anchor digest, trainable inventory, construction precision, tied aliases,
and deployed cast therefore belong to the mathematical treatment definition.

Three comparisons have separate denominators: original-source behavior
before/after warmstart, warmed-anchor behavior before/after acquisition, and
newly acquired R/S behavior before/after composition or further acquisition.
Re-enrolling only the warmed model must not erase any earlier loss from an
original-source enrollment when original-source preservation is an endpoint.

## 2. Score identities and assumptions

Fix a prompt x. Let z be a generated trajectory, including its sampled stop
event, and let r(x,z) be the unchanged, theta-independent binary verifier.
Write

\[
p=\Pr_{z\sim\pi_\theta}(r=1),\qquad q=1-p,\qquad
S(z)=\nabla_\theta\log\pi_\theta(z\mid x).
\tag{2}
\]

The following calculations assume fixed support, differentiation under the
expectation, and independent draws from one behavior-policy snapshot. For
0 < p < 1,

\[
E[S]=0,\quad E[rS]=\nabla p,\quad
E[S\mid r=1]=\frac{\nabla p}{p},\quad
E[S\mid r=0]=-\frac{\nabla p}{q}.
\tag{3}
\]

The score is a sum over generated-token log probabilities. If sampling uses
temperature or other warpers, the generating distribution and differentiated
distribution must agree or have a justified correction. Top-k/top-p support
changes require their own treatment; raw-policy scores cannot silently stand
for warped-sampler scores. EOS, padding, truncation, and the single causal
shift must have the exact qualified meaning.

Replacing S by S/length makes E[S/length] generally nonzero, because length
depends on the generated trajectory. Thus per-response mean-token losses
need not have the same expected gradient as the sequence-score identities.
A global fixed scaling is harmless to the direction; response-dependent
scaling is a different estimator. Clipping, multiple updates on one group,
gradient clipping, Adam, and KL terms also require a distinction between the
unclipped population gradient and the actual optimization procedure.

The MaxRL paper derives correctness likelihood and finite-rollout estimators.
Its theorem applies under its score-function assumptions; the native runner
must establish those assumptions rather than inherit them from an algorithm
name. [MaxRL v3, Sections 2–4 and Appendix D](https://arxiv.org/html/2602.02710v3)

## 3. Exact fixed-N MaxRL variants

Take N independent trajectories and K = sum_i r_i. Use S_i for their scores.
Conditional on K = k with 1 <= k <= N, the successful trajectories have mean
score grad p / p. Therefore the success-average estimator

\[
g_{\rm raw}=\mathbf1_{K>0}\frac1K\sum_i r_iS_i
\tag{4}
\]

has expectation

\[
E[g_{\rm raw}]=\Pr(K>0)\frac{\nabla p}{p}
=\frac{1-q^N}{p}\nabla p
=\left(\sum_{j=0}^{N-1}q^j\right)\nabla p.
\tag{5}
\]

This is the gradient of the N-term truncated objective

\[
J_N(p)=-\sum_{t=1}^{N}\frac{q^t}{t}.
\tag{6}
\]

Its derivative is (1-q^N)/p. At fixed p > 0 it tends to 1/p as N increases;
at fixed finite N its low-p limit is N. It does not deliver an infinite
likelihood weight from a group in which no success was sampled.

The unconditional average score V = (1/N) sum_i S_i has E[V] = 0.
Consequently g_raw - cV preserves (5) for a scalar c fixed independently of
the evaluated group. This includes all-failure groups, where g_raw = 0 and
the control-variate term can remain nonzero. Such a term is score noise with
a zero unconditional mean; its realization is not a verifier-discovered
correct direction on an all-failure prompt.

The practical local estimator drops both terms on K = 0:

\[
g_{\rm practical}=\mathbf1_{K>0}
\left(\frac1K\sum_i r_iS_i-\frac1N\sum_iS_i\right).
\tag{7}
\]

Here is a direct derivation of its effective truncation. Independence and
E[(1-r)S] = -grad p give

\[
E[\mathbf1_{K=0}V]
=\frac1N\sum_iE[(1-r_i)S_i]\prod_{j\ne i}E[1-r_j]
=-q^{N-1}\nabla p.
\tag{8}
\]

Since E[V] = 0, E[1_{K>0}V] = q^{N-1} grad p. Subtracting this from (5),

\[
E[g_{\rm practical}]
=\left(\frac{1-q^N}{p}-q^{N-1}\right)\nabla p
=\frac{1-q^{N-1}}{p}\nabla p
=\nabla_\theta J_{N-1}(p).
\tag{9}
\]

This agrees with the paper's v3 practical-estimator theorem: dropping the
all-failure baseline changes order N to order N-1. The paper's asymptotic
GRPO weight is 1/sqrt(pq), whereas a finite group has a different exact weight.
[MaxRL v3, Algorithm 1, Theorem 5 and Appendix E](https://arxiv.org/html/2602.02710v3#A4)

For the more general gated control variate
g_c = 1_{K>0}(successful_mean - cV),

\[
E[g_c]=\left(\frac{1-q^N}{p}-cq^{N-1}\right)\nabla p.
\tag{10}
\]

Changing c is not a pure variance intervention once the baseline is gated.
It also makes all-success groups nonzero unless c = 1. In contrast, with an
ungated V a cross-fitted scalar c can preserve (5). A candidate variance
coefficient is c_star = E[<g_raw-E[g_raw], V>]/E[||V||^2] when the denominator
is positive. Estimating c on the same group can add bias; fitting it on a
separate fixed pilot or cross-fitted batch requires its own budget and scope.
Variance reduction is an empirical claim about the resulting covariance,
not a consequence of centering alone.

## 4. Epsilon, ddof, and finite-group GRPO

Let u = K/N. For any scalar denominator d_K that depends only on K,

\[
g_d=\frac1N\sum_i\frac{r_i-u}{d_K}S_i
\quad\Longrightarrow\quad
E[g_d\mid K=k]=\frac{u(1-u)}{d_k}\frac{\nabla p}{pq}.
\tag{11}
\]

To see this, the conditional score sum separates into k success scores and
N-k failure scores. Their coefficients reduce to
u(1-u)(E[S|success]-E[S|failure]), and the difference equals grad p/(pq).
Uniform groups have zero numerator and are handled separately when d_K = 0.

The local MaxRL denominator is u + epsilon, with zero uniform groups. Its
exact expected weight is the finite binomial sum

\[
w_{M,N,\epsilon}(p)
=\frac1{pq}\sum_{k=1}^{N-1}
{N\choose k}p^kq^{N-k}
\frac{(k/N)(1-k/N)}{k/N+\epsilon}.
\tag{12}
\]

At epsilon = 0 this reduces to (9). For epsilon > 0, each mixed-group
contribution is attenuated by u/(u+epsilon), which is most consequential for
rare-success groups. The local default epsilon = 1e-6 should be recorded;
the exact truncation identity must not be quoted as an exact implementation
identity without accounting for it. Epsilon still leaves uniform groups at
zero. See the [local estimator](C:/Users/clclc/.codex/worktrees/comp-rs-native-q1/ChatBubble-titan-model-soup-mini-v1/experiments/maxrl_verifiable_data_v1/maxrl.py).

For a standard-deviation convention ddof = d in {0,1},

\[
s_{k,d}=\sqrt{\frac{N}{N-d}u(1-u)},
\qquad
w_{G,N,\epsilon,d}(p)
=\frac1{pq}\sum_{k=1}^{N-1}{N\choose k}p^kq^{N-k}
\frac{u(1-u)}{s_{k,d}+\epsilon}.
\tag{13}
\]

The inherited GRPO convention is ddof = 1. At zero epsilon its gradient is
sqrt((N-1)/N) times the ddof = 0 gradient on every mixed group. With a fixed
positive epsilon the ratio is no longer that exact constant. At large N,
(13) approaches 1/sqrt(pq) for zero epsilon. Its small-p finite-N limit is
sqrt(N-1) for ddof = 0 and (N-1)/sqrt(N) for ddof = 1, compared with N-1
for practical MaxRL. These limits follow by retaining the k = 1 term of the
binomial sum, not by taking the population limit first.

On a realized mixed group, zero-epsilon MaxRL/GRPO gradients have ratio

\[
\frac{g_M}{g_{G,d}}
=\frac{s_{k,d}}u
=\sqrt{\frac{N(N-k)}{(N-d)k}}.
\tag{14}
\]

With epsilon in both denominators, replace this by
(s_{k,d}+epsilon)/(u+epsilon). This explicitly identifies the treatment:
relative weight across different success counts, rather than a new
correctness signal.

### The N = 2 degeneracy

When N = 2, the only mixed group has u = 1/2. With ddof = 1,
s = 1/sqrt(2), so zero-epsilon practical MaxRL is sqrt(2) times GRPO.
Both expected gradients point along grad p with constant weights:
w_M = 1 and w_G = 1/sqrt(2). With identical epsilon their ratio is still
constant, (1/sqrt(2)+epsilon)/(1/2+epsilon). Thus N = 2 does not identify
different difficulty reweighting; it primarily exposes optimizer-scale
effects. A native objective study should choose a qualified N >= 3 and
record the convention rather than interpreting N = 2 as a strong objective
comparison. Constant gradient scaling can still change Adam epsilon,
clipping, finite-step dynamics, and stopping, so it does not imply identical
trained checkpoints.

As a separate control, REINFORCE with a leave-one-out reward baseline obeys

\[
r_i-\frac{\sum_{j\ne i}r_j}{N-1}
=\frac{N}{N-1}(r_i-u).
\tag{15}
\]

Its expected gradient is grad p; same-group mean centering without that
factor has expected gradient (N-1)/N grad p. Dividing either by a random
observed success rate is another estimator. RLOO is useful context for
baseline independence, not a silent replacement for the primary GRPO/MaxRL
factor. [RLOO paper](https://arxiv.org/html/2402.14740#S2.SS3),
[DeepSeekMath/GRPO](https://arxiv.org/html/2402.03300)

## 5. Signal probability, rejection, and compute allocation

The probabilities of all failure, mixed outcomes, and all success are

\[
P_0=q^N,\qquad P_{\rm mixed}=1-q^N-p^N,\qquad P_N=p^N.
\tag{16}
\]

Both centered methods are zero on uniform groups. Increasing N changes the
probability of an informative group and, for practical MaxRL, the truncated
objective itself. Under a fixed total rollout allowance C, the number of
distinct groups is approximately C/N; informative groups alone therefore
do not determine efficiency. Their expected count under homogeneous p is
(C/N)(1-q^N-p^N), while group variance, prompt diversity, prefill, variable
response length, and verification costs also change.

DAPO discards uniform groups and refills an effective batch. Its full recipe
also changes clipping, token-loss aggregation, and length reward handling.
Importing the full recipe would change several treatments and the audited
binary reward. A dynamic-sampling-only extension must name precisely which
component it ports. [DAPO, Sections 3.2–3.4](https://arxiv.org/html/2503.14476#S3)

Let a(x) = P_mixed(x), and let g be a fixed-group estimator that is zero on
uniform outcomes. Rejection across x ~ rho produces accepted prompt density
rho_acc(x) proportional to rho(x)a(x). Its within-prompt conditional mean is
E[g|mixed,x] = E[g|x]/a(x). Consequently, under one frozen policy and exact
unbounded rejection,

\[
E_{\rm accepted}[g]
=\frac{E_{x\sim\rho}E[g\mid x]}{E_{x\sim\rho}a(x)}.
\tag{17}
\]

This calculation prevents an overclaim: conditioning does not necessarily
change the expected gradient direction here; it can globally rescale it.
The accepted prompt distribution, gradient noise, attempted costs, effective
step sizes, and number of updates per cost still change. A refill cap,
asynchrony, weighting by variable accepted counts, or updating while refilling
requires a fresh calculation. If only one prompt is repeatedly retried,
conditional normalization is prompt-specific and (17) is not the same
training design.

## 6. Bounded adaptive stopping: cost and estimator bias

The NGU paper continues all-failure prompts probabilistically, retains prior
failures, and uses an asynchronous implementation with age filtering. The
published experiments motivate testing allocation efficiency; they do not
certify effectiveness on short R/S tasks or this hardware. A fixed-snapshot
variant omitting asynchronous buffering must be labeled an adaptation.
[NGU, Section 4 and Appendix E](https://arxiv.org/html/2609.13443#S4)

### Blockwise stopping cost

Consider the following explicitly bounded adaptation. Draw an initial block
of n trajectories from one frozen policy. Stop if the block contains success.
After an all-failure block continue with independent probability gamma;
draw at most L blocks. The probability of reaching block j+1 is
(gamma q^n)^j. Therefore, for total trajectories T,

\[
E[T]
=n\sum_{j=0}^{L-1}(\gamma q^n)^j
=n\frac{1-(\gamma q^n)^L}{1-\gamma q^n},\qquad T\le nL.
\tag{18}
\]

The finite sum defines the case gamma q^n = 1 without division. For gamma < 1,
E[T] <= n/(1-gamma). This is a count bound, not a generated-token or wall-time
bound; each must be separately capped. Treat an initial all-success block as
an uninformative group if using centered advantages. A later success block
is mixed when previous failures are retained, even if its own draws are all
success.

With all previous failures retained, mixed-group acceptance probability is

\[
P_{\rm acc}
=1-q^n-p^n
+(1-q^n)\sum_{j=1}^{L-1}(\gamma q^n)^j.
\tag{19}
\]

For a hardness estimate close to zero, increasing the cap mostly buys more
unsuccessful samples unless success is eventually found. The inherited
practical MaxRL and centered NGU recipes provide zero updates before an
informative group is found. The ungated control-variate variant discussed
above has a different all-failure convention. At the present source
observation, a common qualified
warmstart is a separately registered route to feasibility; additional
sampling is a distinct cost-bearing hypothesis.

### A concrete stopping-bias calculation

For clarity, examine an original scalar stopping scheme with no blocks:
sample until the first success or until M trajectories. Let tau be that
bounded stopping time. On success at t, K = 1. Applying the naive centered
successful-mean estimator gives

\[
h_\tau=\mathbf1_{\rm success}
\left(S_\tau-\frac1\tau\sum_{i=1}^{\tau}S_i\right).
\tag{20}
\]

Conditional on first success at t, the previous t-1 scores have failure mean
-grad p/q. Hence

\[
E[h_\tau\mid \tau=t,\mathrm{success}]
=\frac{t-1}{t}\frac{\nabla p}{pq}.
\tag{21}
\]

The event probability is pq^(t-1), and all failure gives zero. Thus

\[
E[h_\tau]
=\left[\sum_{t=2}^{M}q^{t-2}\frac{t-1}{t}\right]\nabla p.
\tag{22}
\]

This differs from fixed-M practical MaxRL's sum_{j=0}^{M-2} q^j grad p.
For M = 2, (22) is one half grad p while fixed-M practical MaxRL is grad p.
The difference arises from outcome-dependent stopping and its average-score
baseline, even though every trajectory used one frozen policy. Substituting
realized tau into a fixed-N theorem is invalid.

The ungated successful score in this first-success scheme instead has mean
(1-q^M) grad p/p. A bounded-stopping score sum U_tau = sum_{i<=tau} S_i
satisfies

\[
E[U_\tau]=\sum_{i=1}^{M}E[\mathbf1_{\tau\ge i}S_i]=0,
\tag{23}
\]

because reaching draw i depends only on earlier outcomes. A fixed scalar
multiple of U_tau can therefore be an objective-preserving control variate
for that ungated success-score estimator. U_tau/tau generally has nonzero
mean. This is an analytically justified candidate, not a tested native method,
and does not justify stale-policy pooling. Off-policy scores, age cutoffs,
sampling probabilities, and all failed costs need their own receipts.

## 7. Replay signal and local retention penalties

Replay of already mastered prompts often yields all-success groups. Under
the inherited centered objectives these groups have exactly zero advantage,
so more such replay does not directly penalize later degradation. A replay
scheme that preserves them needs an explicit additional signal, for example

\[
L(\theta)=-J_{\rm acquisition}(\theta)
+\lambda E_{x\sim D_{\rm replay}}
D_{\rm KL}(\pi_{\rm ref}(\cdot\mid x)\Vert\pi_\theta(\cdot\mid x)).
\tag{24}
\]

This example uses forward KL; choosing reverse KL, token versus trajectory
KL, fixed prefixes versus sampled prefixes, or supervised replay changes the
recipe. It must be separate from the binary success denominator. Reference
inference, logits, replay tokens, and storage are charged to its budget.
Replay uses declared TRAIN banks, never evaluation guards or held-out RS.

For small delta around a reference model, regularity gives

\[
D_{\rm KL}(\pi_{\rm ref}\Vert\pi_{\rm ref+\delta})
=\tfrac12\delta^TF_{\rm ref}\delta+O(\|\delta\|^3).
\tag{25}
\]

The first derivative vanishes at the reference; its second derivative is the
Fisher under that reference and specified input distribution. A diagonal
approximation yields coordinate-dependent resistance to change. Around a
quadratic acquisition loss with curvature H, a regularized displacement is
(H + lambda F)^(-1) times the acquisition driving vector when the inverse
exists. This local surrogate suggests where acquisition may be attenuated;
it is not a no-forgetting guarantee. [EWC](https://arxiv.org/abs/1612.00796)

The true Fisher averages squared score vectors with labels sampled from the
model. A fixed-label empirical Fisher is a proxy and need not equal the loss
Hessian. Mean squared per-example scores, squared mean gradients, and squared
gradients of a batch loss are different statistics. For independent centered
scores, squaring an unnormalized batch sum can have useful expectation after
scaling, but it is not a finite-sample substitute for per-example Fisher.
The reference, label source, token/sequence reduction, averaging order, and
damping must be declared.

For independent donors, preserving the common anchor uses theta_a as a
reference. In R->S, retaining acquired R uses the actual first-stage R model
as reference and R TRAIN/replay inputs. Original legacy behavior may require
an additional original-source reference. Pooling these references without
weights and denominators changes the hypothesis.

## 8. Sequential permutations and the curvature commutator

An original small-step expansion explains why both orders remain necessary.
Let J_R and J_S be acquisition objectives, g_R and g_S their gradients at
theta_a, and H_R and H_S their Hessians. With one ascent step per skill,

\[
\theta_{R\to S}
=\theta_a+\eta_Rg_R+\eta_Sg_S
+\eta_R\eta_SH_Sg_R+O(\eta_S\eta_R^2),
\tag{26}
\]

\[
\theta_{S\to R}
=\theta_a+\eta_Rg_R+\eta_Sg_S
+\eta_R\eta_SH_Rg_S+O(\eta_R\eta_S^2).
\tag{27}
\]

The common independent sum lacks these cross terms. Their order difference is

\[
\theta_{R\to S}-\theta_{S\to R}
=\eta_R\eta_S(H_Sg_R-H_Rg_S)
+O(\eta_R^2\eta_S+\eta_R\eta_S^2).
\tag{28}
\]

The leading bracket can be nonzero even when g_R and g_S are Euclidean
orthogonal. With multistep Adam or other stateful optimization, replace the
single-step gradient map by the actual update map and its Jacobian; resetting
versus carrying optimizer state is a treatment, not an implementation detail
to choose after observing order effects. Equations (26)–(28) explain possible
order dependence, not its sign or measured size on Rineya.

For two skills the complete permutation panel contains exactly R->S and
S->R. Reuse a first-stage donor only when its anchor, seed, data order, scope,
optimizer state convention, and first-stage budget exactly match the
sequential registration. If the second stage starts from a full donor while
dense composition uses alpha < 1, report both full-donor and operating-dose
controls; their acquired-correct membership is different.

## 9. Composition is a finite difference in function space

Task arithmetic defines updates against one shared initialization. Its
original demonstrations motivate the operator in (1); they do not establish
dependent R/S reasoning or legacy retention on this model.
[Task arithmetic](https://arxiv.org/abs/2212.04089)

The parameter norm identity is

\[
\|a\delta_R+b\delta_S\|^2
=a^2\|\delta_R\|^2+b^2\|\delta_S\|^2
+2ab\langle\delta_R,\delta_S\rangle.
\tag{29}
\]

It describes parameter geometry only. For a predeclared target log-odds
functional f(theta,x) under identical teacher-forced prefixes, define

\[
I(a,b;x)=f(\theta_a+a\delta_R+b\delta_S,x)
-f(\theta_a+a\delta_R,x)-f(\theta_a+b\delta_S,x)+f(\theta_a,x).
\tag{30}
\]

Set h(s,t) = f(theta_a + s delta_R + t delta_S,x). Twice applying the
fundamental theorem of calculus gives the exact identity

\[
I(a,b;x)=\int_0^a\int_0^b
\delta_R^TH_f(\theta_a+s\delta_R+t\delta_S,x)\delta_S\,dt\,ds.
\tag{31}
\]

For small a,b, the leading term is ab delta_R^T H_f(theta_a,x) delta_S.
This derivation requires a smooth fixed-input functional; it does not apply
directly to discontinuous greedy generation or a binary verifier.

A quadratic counterexample makes the geometry boundary concrete. Take
delta_R = e_1, delta_S = e_2 and
H = [[0,1],[1,0]]. Their cosine is zero, but the quadratic function
f(theta) = theta_1 theta_2 gives I = ab. Positive semidefiniteness is not
needed for a logit Hessian; if desired, H = [[1,1/2],[1/2,1]] is positive
definite and also gives nonzero cross interaction. Tiny flattened cosine
can therefore coexist with substantial function interaction.

An exactly linear tangent function has I = 0 in logits. Its thresholded
correctness can still have a nonzero accuracy finite difference. Thus zero
logit interaction neither proves independent correctness nor prevents joint
success. Nonzero I can help or harm, depending on target sign and competing
answers. A normalized kappa must declare its denominator and undefined
near-zero cases; clipping I or the denominator is a robustness analysis,
not the primary estimand.

The tangent-arithmetic study distinguishes training in tangent space from
linearizing already trained nonlinear donors. It finds a single-task
nonlinear advantage and studies weight disentanglement in CLIP models.
Those findings motivate bounded Rineya tests rather than warranting an
assumption that curvature removal improves either acquisition or RS.
[Tangent task arithmetic](https://arxiv.org/html/2305.12827)

## 10. Activation-weighted overlap and effective LoRA geometry

For linear-module deltas D_R,D_S and input second moment Sigma = E[xx^T],

\[
E[\langle D_Rx,D_Sx\rangle]
=\operatorname{tr}(D_R^TD_S\Sigma).
\tag{32}
\]

With an output sensitivity metric G positive semidefinite,

\[
E[(D_Rx)^TG(D_Sx)]
=\operatorname{tr}(G D_S\Sigma D_R^T).
\tag{33}
\]

Sigma is an uncentered second moment unless the input mean is zero; a centered
covariance alone omits the mean contribution. G can be a declared downstream
Fisher/sensitivity approximation, not an unspecified importance score.
Both formulas hold with fixed common inputs. Upstream donor changes alter
those inputs in a nonlinear network, so these are local descriptive
quantities. Freeze capture location, prefix set, dtype, sample count and
memory allowance. Input and output principal-subspace overlap must be
compared within compatible module shapes and weighted by singular energy;
global vector cosine and subspace collision measure different things.

LoRA parameterizes an effective module update as D = cBA, where the scale c
is determined by its declared configuration. For any invertible rank-space
matrix Q,

\[
B'=BQ,\qquad A'=Q^{-1}A,\qquad cB'A'=cBA.
\tag{34}
\]

This gauge freedom makes raw-factor cosine, norms, and coordinate overlap
representation-dependent. Report effective cBA geometry in the actual target
weight convention, and bind rank, scale, transposition, and target modules.
The algebra is an original invariance argument applied to the LoRA
parameterization. [LoRA](https://arxiv.org/abs/2106.09685)

For two donors, the correct effective sum is a c_R B_R A_R + b c_S B_S A_S.
Summing factors and multiplying introduces unwanted cross terms
B_R A_S and B_S A_R. A concatenated representation can realize the effective
sum exactly: concatenate the appropriately scaled B factors horizontally
and stack their A factors vertically. Its rank is at most r_R+r_S. Any later
compression to a smaller rank is a new approximation requiring individual
skill and retention controls. Dense inventory-B training remains the
present candidate scope; LoRA is a separate scope/cost intervention.

## 11. Fisher allocation: preserving a donor and retaining the anchor

Wanda uses magnitude times input-feature norm for within-row ranking.
F-WANDA allocates different row keep counts using a row sensitivity proxy;
a positive rowwise multiplier by itself cannot change within-row rankings.
Its published experiments concern full-weight LLaMA-2 pruning, including an
exact fallback to Wanda under fixed 2:4 density. They do not test this
study's delta pruning, skill composition, or legacy correctness.
[Wanda](https://arxiv.org/html/2306.11695),
[F-WANDA, Section 3](https://arxiv.org/html/2608.00481#S3)

For a delta D and empirical input second moments v_j, a diagonal activation
approximation to the energy lost by dropping entries is

\[
E[\|(D-D_{\rm keep})x\|^2]
\approx\sum_{i,j:(i,j)\ \mathrm{dropped}}D_{ij}^2v_j.
\tag{35}
\]

Ranking |D_ij| sqrt(v_j) therefore preserves this local donor perturbation
under equal row budgets. Adding a row metric omega_i gives the approximate
sum of omega_i D_ij^2 v_j over removed entries. Multiplying each row by
sqrt(omega_i) leaves its order unchanged. A Fisher-inspired allocation must
actually change row counts, for example k_i proportional to sqrt(omega_i),
with exact integer total K, caps, zero-sensitivity handling, and deterministic
ties. The square-root allocation is a specified heuristic; it is not
generally the optimizer of (35) for arbitrary entry distributions.

Preserving a fine-tuned donor's function after removing updates and retaining
the anchor's old function after keeping updates are different objectives.
Use two declared calibration distributions and reference states. Let m_i be
a binary keep mask on delta_i. A diagonal local proxy is

\[
\mathcal R(m)
=\sum_i F_{\rm skill,i}(1-m_i)\delta_i^2
+\lambda\sum_i F_{\rm legacy,i}m_i\delta_i^2,
\qquad\sum_i m_i=K.
\tag{36}
\]

The first term penalizes deviation from the acquired donor when updates are
removed. The second penalizes deviation from the legacy reference when
updates are kept. Using (1-m_i)^2 = 1-m_i and m_i^2 = m_i, its mask-dependent
part is sum_i m_i(lambda F_legacy,i - F_skill,i) delta_i^2. Under this
diagonal, fixed-count approximation, minimize (36) by keeping the K largest

\[
(F_{\rm skill,i}-\lambda F_{\rm legacy,i})\delta_i^2.
\tag{37}
\]

This two-distribution rule is an original proposed acquisition/retention
trade-off, not F-WANDA's published algorithm. The two Fishers may be at
different reference states and need comparable specified reductions and
scales. Off-diagonal curvature, nonzero first derivatives, changes in input
representations, and large deltas limit the approximation. Calibration uses
declared TRAIN/replay inputs, never evaluation guards. Fisher sensitivity
is not direct evidence that a kept entry damages a formerly correct answer.

In a composite with R fixed and S masked, a legacy quadratic approximation
contains (a delta_R + b m*delta_S)^T F_legacy
(a delta_R + b m*delta_S). Even with diagonal F there is an additional
2ab sum_i F_i delta_R,i m_i delta_S,i term. A mask calibrated on S alone
can therefore fail in RS; it must be evaluated in the intended fixed-R
context while retaining its own S-positive control.

At one prospectively chosen keep count, compare dense S, magnitude S,
delta-Wanda S, and Fisher row allocation S, with the same R and coefficients.
Include a direction-preserving attenuation control chosen under a declared
rule, because lower norm and weakened acquisition can reduce observed
damage without improved allocation. Do not silently rescale masked sums.
If a direction or mask changes, reacquire its individual positive-control
evidence before interpreting composition.

Fisher-weighted averaging is a different operator: it takes coordinatewise
precision-weighted means of donor states. Re-expressing those states around
the anchor shows coordinatewise attenuation, not the independent-coefficient
sum in (1). It needs its own dose controls.
[Fisher merging](https://arxiv.org/html/2111.09832)

## 12. Central decision fusion and MiMo teacher fusion

The local Central work supplies an inference composition interface: experts
sharing one tokenizer evaluate the same accepted autoregressive prefix;
each proposes a top-1 token; plurality and a declared tie-break choose the
next token. The mechanics are documented in the
[local mathematical framework](C:/Users/clclc/.codex/worktrees/comp-rs-native-q1/ChatBubble-titan-model-soup-mini-v1/short-response-adapter-consensus-lab/paper/METHODS_AND_MATH.md)
and its [scope statement](C:/Users/clclc/.codex/worktrees/comp-rs-native-q1/ChatBubble-titan-model-soup-mini-v1/short-response-adapter-consensus-lab/documentation/phase2_paper/CENTRAL_CLAIM.md).
Their measured results do not establish efficacy on Rineya R/S.

For expert logits z_j(v), top-1 proposals are invariant to positive
expert-specific scaling and to adding an expert-specific constant across
the vocabulary. Unique plurality winners inherit that invariance. On a
plurality tie between a and b, a summed-log-probability tie-break compares

\[
\sum_j[\log p_j(a)-\log p_j(b)]
=\sum_j[z_j(a)-z_j(b)].
\tag{38}
\]

The log-normalizers cancel. Unequal temperatures can reverse this secondary
ordering; a shared positive temperature preserves it. Arithmetic logit
averaging is also invariant to a per-expert vocabulary-wide additive
constant, since it adds the same constant to every pooled candidate. Its
sensitivity is to relative scaling and more general vocabulary-dependent
shifts, not that additive constant. Probability pooling has different
calibration dependence. These are operator properties, not causal evidence
for an accuracy gain. A K-expert committee must pay measured inference costs
for consultations, caches, and prefix synchronization.

The current official MiMo MOPD card describes domain-specialized RL and SFT
teachers supervising full student rollouts, teacher-prefix continuations,
and demonstration-prefix continuations. It motivates a distinct
teacher-assisted composition route rather than establishing delta-addition
efficacy. [Official MiMo MOPD card](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD#how-mopd2-works)

The following illustrates fusion mathematics, and is not a claim about a
specific MiMo implementation. For fixed common-prefix teacher distributions
q_j and weights w_j summing to one,

\[
\arg\min_p\sum_jw_jD_{\rm KL}(q_j\Vert p)=\sum_jw_jq_j,
\tag{39}
\]

whereas

\[
\arg\min_p\sum_jw_jD_{\rm KL}(p\Vert q_j)
\propto\exp\left(\sum_jw_j\log q_j\right).
\tag{40}
\]

The first follows by differentiating cross entropy with a normalization
multiplier. The second follows by collecting sum_v p(v)log p(v) minus
sum_v p(v) sum_j w_j log q_j(v). Reverse-KL fusion requires appropriate
support; a teacher assigning zero can exclude that token. Domain-conditioned
teachers and student-dependent prefixes are additional choices, so this
fixed-distribution barycenter is not a complete sequence-training algorithm.
Neither distribution fusion is algebraically equal to summing weight deltas.

The [official technical report](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/blob/main/MiMo_V2_6_technical_report.pdf)
was identified, but its PDF body was not retrievable through the research
browser during this analysis. Detailed MixRL equations or efficacy claims
are therefore not attributed to an inspected report. The official card is
the primary source for the mechanics stated here.

## 13. Staged combinations and complete R/S order controls

This matrix is a bounded proposal. Every numerical ceiling, margin, seed
count, coefficient rule, and method trigger must be prospectively frozen
after qualification. No row below authorizes its native execution.

| Stage | Treatment cells | Bound and controls | Falsifiable question |
|---|---|---|---|
| Feasibility | Original source; one common 600-row warmstart; qualified warmed source | One warmstart recipe shared by all later donors; verify rebuilt anchor; remeasure strict competence and source-correct losses | Does the common intervention supply enough mixed strict groups for a bounded objective comparison? |
| Primary acquisition | G-R, G-S, M-R, M-S | Four donors per declared independent seed block; same anchor, inventory, optimizer family, sampler, N, reductions, and tuning allowance | At matched cost ceilings, does M improve held-out atomic acquisition over G by the declared margin? |
| Primary composition | B,R,S,RS within each objective | Seven unique operating models with shared B per seed block; identical solo/RS coefficients; no sum normalization; own-skill gates | Do independently acquired skills improve dependent RS while meeting atomic and legacy loss margins? |
| Sequential order | G: R->S and S->R; M: R->S and S->R | Complete two-order panel; first-stage donors may be reused only under exact bindings; second stages have matched ceilings and optimizer-state convention | Does recency or asymmetric interference explain losses or RS outcomes relative to independent composition? |
| Sampling follow-up | G-fixed, M-fixed, G-adaptive, M-adaptive | At most one bounded adaptive recipe; original fixed baselines retained; attempted cost and policy age recorded | Does allocation improve acquisition per compute, and does that effect differ by objective? |
| Retention follow-up | No replay versus one fixed KL or supervised replay recipe, in both orders | One reference/replay policy and tuning allowance matched across objectives; charge teacher/replay work; enroll first-stage membership | Can old acquired losses decrease while later skill acquisition remains within its declared margin? |
| Composition repair | Dense, magnitude, delta-Wanda, Fisher row allocation on one offender | One fixed keep count and one declared attenuation control; other donor and coefficients fixed; own-skill controls | Does structure improve the acquisition/damage trade-off beyond norm reduction? |
| Conditional combination | Neither intervention, A only, B only, A+B | Only after A and B separately meet development triggers; four-cell small interaction panel, no automatic cross-product of all methods | Does the combined method improve the declared joint endpoints beyond its single-intervention components? |
| Alternative interface | One Central-style committee or one MOPD-inspired distillation route | Separate proposal with its own inference/training costs and data-prefix contracts | Can a different composition operator improve the same endpoints under its declared resource envelope? |

G means the inherited GRPO recipe and M the inherited practical MaxRL recipe,
including the frozen epsilon and reduction. A and B are named, selected
interventions, not arbitrary knobs. The first baseline should remain dense
same-anchor training. Changes in LoRA scope, pruning, projection, tangent
training, architecture, verifier reward, bridge adaptation, and teacher
access are separate treatments. An RS bridge-trained model measures
adaptation to joint tasks, not spontaneous composition from atomic training.

For any metric Y, the four-cell interaction is

\[
\Gamma=Y_{AB}-Y_A-Y_B+Y_0.
\tag{41}
\]

Report Gamma with uncertainty separately for acquisition, RS, retention loss,
and resource costs. An improved combined endpoint is not evidence of synergy
if neither single intervention was measured. A post-hoc choice of A/B or
offending donor based on TEST would turn this into selection on the outcome;
use development triggers and retain selection/tuning costs in the ledger.

## 14. Cost, seeds, memberships, and auditable hypotheses

Independent donors and sequential orders each acquire two skills, but their
resource paths differ. A sequential final state has two stages; dense RS
uses two independently trained donors plus composition/evaluation work.
Equal per-stage limits do not establish equal actual compute. Report a
ledger containing attempted prompt-prefill tokens, generated tokens,
backward/loss tokens, verifier work, updates, reference/teacher work, memory,
wall time, disk use, and tuning costs. Inference committees add their actual
consultation multiplier; pruning/dense materialization alone does not prove
runtime speed or memory savings.

Run-level replication comes from independent training seed blocks. A seed
block pairs methods by declared initialization/data-order conditions but
each diverged learner samples its own current-policy trajectories. Do not
reuse a competing policy's response bank. Parent worlds are the evaluation
resampling unit; child views, repeated decodes, sparse variants, and repeated
reads are not independent training replications. A single seed supports a
conditional description, not a population claim about training variability.

The governing COMP-RS amendment already fixes the grid
{0.125, 0.25, 0.375, 0.50, 0.75, 1.00}: alpha_star is the smallest coefficient
at the acquisition maximum, but the deployed coefficient is the **median**
on the nine-row plateau (lower median for an even count). Preserve that rule
for the registered study. A new method panel must freeze its coefficient rule
and equal tuning allowance before reading its dose response. In equation
(1), its solo R and S controls use precisely the coefficients deployed in
RS. Keep raw-donor scores available because dose selection is part of the
treatment policy. Composition acquired-R membership is enrolled from
theta_a + alpha_R delta_R; sequential full-donor acquired-R membership is
enrolled from theta_R. A membership can be empty, in which case its retention
ratio is undefined rather than perfect.

For a reference-correct set E_ref and candidate correctness C,

\[
\mathrm{loss}_{\rm ref}(C)
=\frac{\sum_{x\in E_{\rm ref}}\mathbf1[C(x)=0]}{|E_{\rm ref}|}.
\tag{42}
\]

State the actual lost counts, denominators, and families. New rescues on
previously wrong examples do not cancel these losses. On each dependent RS
parent, retain the correctness vector (B,R,S,RS). Composition-only repair is
B=R=S=0, RS=1; joint damage is max(R,S)=1, RS=0. Strict outcomes remain primary;
bounded content diagnoses wrapper failures separately. Pass@k is any-success
coverage under sampling, not selected-answer accuracy unless that selection
procedure actually runs.

Predeclare the following hypotheses and their material margins rather than
substituting nonsignificance for equivalence:

1. Acquisition: each donor exceeds its own immutable-anchor atomic outcome
   by a meaningful margin under the declared cost ceiling.
2. Composition: RS improves dependent RS relative to each same-dose solo,
   while atomic and reference-correct losses meet their separate bounds.
3. Order: R->S and S->R differ in losses of their enrolled first skills or
   dependent RS beyond the order margin; both are required to interpret it.
4. Allocation: one adaptive method improves acquisition per actual compute
   and its hard/easy trade-off relative to fixed groups, with an objective
   interaction estimate.
5. Retention: one explicit replay/penalty reduces acquired losses while later
   acquisition meets its bound; uniform-success replay alone is not assumed
to supply protection.
6. Mask allocation: the selected mask improves damage at a declared skill
   acquisition level beyond equal-count magnitude/Wanda and attenuation
   controls. Lower norm, Fisher proxy, or cosine alone cannot pass it.
7. Combination: the four-cell interaction in (41) is favorable by its
   registered margin for the stated endpoint, with the costs included.

Multiplicity, parent-cluster uncertainty, run variation, final source-correct
membership, and independent evaluation must be fixed before final held-out
reads. The existing exposed banks constrain interpretation; planning and
analytical audits do not make them sealed or certify real-world retention.

## 15. Completed analytical audit and its boundary

The companion audit reports EXACT_TOY_IDENTITIES_PASS for 572 finite-group
cases and 72 blockwise stopping cases. Zero-epsilon MaxRL enumeration agrees
with its closed-form Bernoulli-logit expectation to a maximum absolute error
of 8.92e-16. The associated mathematical suite reported 46 passing tests.
The create-only [audit receipt](evidence/exact-20260930/RECEIPT.json) binds the
code and output hashes; the [complete result](evidence/exact-20260930/EXACT_AUDIT.json)
contains the source evidence references, control cases and full tables.
The [enumerator](exact_math.py), [audit entrypoint](run_audit.py), and
[mathematical tests](tests/test_exact_math.py) specify the actual toy estimators.

For the scalar Bernoulli-logit toy, S_i = r_i-p and grad p = pq. At N = 8,
p = 0.05 and epsilon = 0, the results are:

| Estimator | Expected logit gradient | Variance |
|---|---:|---:|
| REINFORCE | 0.04750000 | 0.00535859 |
| Same-group centered | 0.04156250 | 0.00374387 |
| RLOO | 0.04750000 | 0.00488996 |
| GRPO, ddof = 1 | 0.10988357 | 0.02429279 |
| Practical MaxRL, dropped all-failure baseline | 0.28657957 | 0.16288922 |
| Success-only MaxRL | 0.31975059 | 0.20152262 |
| MaxRL with unconditional score control variate | 0.31975059 | 0.14443518 |

These numbers verify expected scaling and, in this toy, variance reduction
at an unchanged success-only objective. Comparing raw variances across
different expected-gradient scales is not a direct ranking of training
quality; objective weight, normalized signal/noise, optimizer behavior,
prompt mixture, and model geometry differ. The
[finite-group table](evidence/exact-20260930/fixed_group_expectations.csv)
includes both epsilon settings and both standard-deviation conventions.

For same-snapshot failure continuation with n = 4, L = 4, gamma = 1 and
p = 0.05, exact enumeration gives expected draws 12.07314708 with maximum
16. The expected pooled practical logit gradient is 0.47169917, whereas
fixed-N = 16 practical MaxRL gives 0.50987333. Thus the random-total pool
does not inherit the fixed-maximum-group objective. This comparison verifies
an estimator distinction; it does not establish superior improvement per
compute. See the
[stopping table](evidence/exact-20260930/stopped_sampling_cost.csv).
The scalar first-success calculation (20)–(23) is an additional analytical
derivation; the 72 receipt cases enumerate blockwise continuation rather
than claiming coverage of every scalar stopping variant.

The rejection control uses two pass rates (0.02, 0.5), equal original prompt
weights, and N = 8. Mixed-group acceptance is 0.57071224; accepted prompt
weights become (0.13074626, 0.86925374). The fixed-proposal expected gradient
is 0.31266536. Dividing by acceptance gives global-rejection expectation
0.54785116, while a fixed accepted quota per prompt gives 0.68299248.
This verifies the normalization distinction in (17), not an acquisition
claim about an adaptive prompt curriculum.

The quadratic controls show:

- Orthogonal initial gradients give an order difference (0.007, -0.002),
  equal to the Hessian commutator prediction. The audit uses gradient descent
  on quadratic losses; the second-order cross-term sign is also positive
  because the two negative first-order steps multiply.
- Orthogonal deltas give functional interaction 0.0875 at a = 0.5,
  b = 0.25 and cross-Hessian entry 0.7, matching (31).
- Two update rows with Frobenius dot zero have activation-weighted cosine
  0.98019802 under second moment diag(100,1). Flattened orthogonality does
  not remove activation-weighted overlap.
- The diagonal two-distribution mask keeps zero-based coordinates [1,2],
  attaining objective 11, equal to exhaustive enumeration over fixed-count
  masks. Rowwise positive score multiplication leaves row rankings unchanged.
- A LoRA gauge change leaves a donor's effective update unchanged while
  changing the result of averaging raw factors. Effective-update averaging
  remains a separate operator.
- Uniform-success RL replay gives zero gradient, while a declared reference
  KL has a nonzero derivative once the current Bernoulli model drifts from
  that reference. The KL derivative agrees with a finite-difference check.

These checks establish algebra for defined scalar/quadratic examples and
executable estimator conventions. The receipt records native_models_run = 0.
It cannot validate the native loader, on-policy collector, warmstart efficacy,
learned donor competence, real-model curvature, allocation causality, or any
retention margin. The native source remains 0/3,600 strict successes and the
common warmstart outcome remains pending until separate native evidence
changes that status.

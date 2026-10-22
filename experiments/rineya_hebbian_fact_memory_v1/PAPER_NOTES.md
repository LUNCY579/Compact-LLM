# Source and implementation notes

Primary source: Garcia et al., *MLPs are Hebbians: Constructing Efficient
Fact-Storing MLPs for Transformers*, arXiv:2607.10034v1 (2026-07-10).
https://arxiv.org/abs/2607.10034
https://arxiv.org/html/2607.10034v1

Official implementation inspected, not vendored:
https://github.com/HazyResearch/hebbian-mlps/tree/41443978d2261f30a9749fe7a286c95be08b331e
See `SOURCES.json` for exact upstream paths and blob hashes. This package is an
independent equation-based implementation, not a copy of the upstream package.
Upstream's license is Apache-2.0; no upstream datasets/weights are redistributed.

## Mathematical contract

Use row-major keys K and targets Y. With width M,
`Phi = (K A.T) * (K G.T)` and prediction `Phi B.T`.
Independent Gaussian A/G divided by `M**0.25` implement Appendix B.2's
`(a.x)*(g.x)/sqrt(M)` features, whose expected kernel is `(x.z)**2`.
The gate is the identity: this is **not Qwen3's SiLU-gated MLP**.

Raw readout is `B = Y.T Phi / N`; diagonal whitening divides each column
by `mean(Phi**2)+ridge`. Full whitening and the two-factor data-dependent
construction correspond to Algorithm 1 and Appendix B.2.5.

Two deliberate, test-covered conventions avoid conflating different equations:

* `mean_loss` (default): minimize `||Phi W-Y||^2/N + lambda*||W||^2`.
  Both primal and dual Gram systems use **N*lambda**.
* `paper_branch`: reproduce the released branch convention: primal uses
  **N*lambda**, dual uses **lambda**. They differ at the same nonzero lambda.
  This is a compatibility setting, not a mathematically equivalent route switch.

At lambda=0 the implementation uses a minimum-norm SVD solution. Recovering any
original B from its outputs requires full column rank. With deficient features,
only the fitted outputs on their span are guaranteed; the original B is not
identified. Unit tests demonstrate both cases. No inverse is silently fabricated.

For data-dependent features, the initial raw B0 remains fixed through two
matrix-free linear factor fits (G then A); the final readout is reconstructed.
The default `vector` objective follows the paper's vector-target formulation.
`upstream_score` instead fits candidate scores against the identity matrix,
matching the inspected released implementation's coupled value-Gram objective.
Projection ridge is an explicit stabilizer; positive ridge is not an exact
unregularized reproduction. Data-dependent initialization uses unnormalized
Gaussian factors, separately from normalized random-feature arms. True CG
residuals must qualify; reaching the iteration limit is a failure, not a fit.

## What each experiment means

Strict recall requires a positive true-versus-best-wrong margin. An argmax tie
is not a successful association. Synthetic labels may be many-to-one; competitor
IDs refer to distinct candidate vectors. Kernel-based signal/crosstalk accounting
uses the actual readout normalization, not an arbitrarily scaled feature Gram.
The reported Cauchy bound is a diagnostic, not a finite-sample capacity theorem.

An unavoidable boundary is `Phi(x)=Phi(-x)` for this bias-free quadratic map.
It cannot distinguish arbitrary antipodal keys. Synthetic random embeddings
and native contextual hidden states do not automatically share favorable geometry.

Our SSFR-style reader uses explicit key/query tag channels, identity V/O,
no residual bypass, fresh evaluation contexts, and a mapping-A to mapping-B swap
with the reader frozen. It is a simplified mechanism test, not an exact paper
Transformer reproduction. Repeated contexts of one fact are clustered, not new
independent facts. The stale A memory is a required negative control for B.

The paper's Qwen3-0.6B-Base study uses captured layer-14 activations. Its
specialized fact-editing model is not evidence that one may safely replace an
arbitrary pretrained assistant's MLP. Our native bridge therefore fits existing
MLP outputs and measures drift first. Supplying new factual target vectors,
proving their semantic correctness, and evaluating language edits are subsequent
research, **not functionality that this reconstruction screen claims to solve**.

## Rineya-specific hypotheses, not established conclusions

The useful connection to composable skills is functional interference: how much
a newly stored association changes old outputs when activations overlap. This
may complement existing parameter-subspace experiments; it does not equate
orthogonal parameter deltas with orthogonal keys or zero forgetting. First
qualify activation geometry and nonzero interventions on the exact 0.6B checkpoint.
Keep recurrence, harness, and generic skill training as separate interventions
until fact-memory effects can be isolated. Replacing every MLP is out of scope.

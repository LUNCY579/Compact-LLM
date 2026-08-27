# PCSM theory and engineering invariants

## 1. Problem statement

Let a native transformer parameter be

\[
W_t \in \mathbb{R}^{d_{out}\times d_{in}}.
\]

PCSM generates, once per frozen plan,

\[
M_t \in \{0,1\}^{d_{out}\times d_{in}},\qquad
A_t \in [a_{min},a_{max}]^{d_{out}\times d_{in}}.
\]

`M` answers **where may the target learn?** and `A` answers **how strongly may
the selected coordinate learn?** Neither object provides an arbitrary signed
parameter delta. The task gradient supplies the optimization direction. Peer
information affects plasticity only.

The idealized SGD expression is

\[
W_t^{(n+1)} = W_t^{(n)} - \eta M_t \odot A_t \odot
\nabla_{W_t}\mathcal{L}.
\]

For AdamW, applying `A` to the raw gradient is insufficient because a positive
constant can cancel between first and second moments. PCSM therefore applies
`A` to the realized, normalized update; Section 9 gives the exact rule.

## 2. Architecture-derived peer eligibility

Every rank-two native parameter is indexed by its module-tree site, repeated
block index, module type, and semantic role. For target `t`,

\[
\mathcal{P}_t = \{j \ne t : role(j)=role(t)\}.
\]

Examples include `mlp.down_proj` with `mlp.down_proj` and `q_proj` with
`q_proj`. PCSM does not silently widen the peer set to unrelated roles.

The target is removed before attention is calculated, so self advice is absent
by construction and \(\alpha_{tt}=0\) exactly. If no compatible peer remains,
the production path fails closed. An empty-peer mode exists only as an
explicit target-only control.

The real Phase-C implementation further requires all same-role matrices to have
equal shape. The generic property-map library supports deterministic projection
for unequal shapes, but Phase C v1 does not use that escape hatch.

## 3. Compact FP32 descriptors

The descriptor function summarizes a tensor rather than embedding its complete
coordinate array. All reduction and scoring arithmetic is FP32, even when the
native model is BF16.

The accepted descriptor width is 16-32; the qualified default is 24. Let
\(rms(X)=\sqrt{mean(X^2)}\). The default feature order is:

| Index | Descriptor value |
|---:|---|
| 1 | signed weight mean / weight RMS |
| 2 | weight standard deviation / RMS |
| 3 | mean absolute weight / RMS |
| 4 | maximum absolute weight / RMS |
| 5-6 | mean and standard deviation of row RMS / matrix RMS |
| 7-8 | mean and standard deviation of column RMS / matrix RMS |
| 9 | positive fraction |
| 10 | fraction with `abs(W) >= RMS(W)` |
| 11 | fraction with `abs(W) >= 2 RMS(W)` |
| 12 | finite-value fraction |
| 13-14 | `log1p(rows)`, `log1p(columns)` |
| 15 | normalized repeated-layer depth in `[0,1]` |
| 16 | stable SHA-256-derived role code in `[-1,1]` |
| 17-19 | activation signed mean, std, and mean absolute value, each / activation RMS |
| 20-22 | calibration-gradient signed mean, std, and mean absolute value, each / gradient RMS |
| 23-24 | maximum row RMS and maximum column RMS / matrix RMS |

Missing optional activations or gradients contribute zeros; non-finite inputs
are sanitized. Positive rescaling leaves the normalized geometry features
unchanged. Descriptor widths above 24 add negative/zero fractions, minimum
row/column RMS, centered RMS, log aspect ratio, and normalized third/fourth
moments.

In the current real Phase C, the target descriptor includes its calibration
gradient; peer activation and gradient descriptors are not collected, so their
optional slots are zero. This is a documented scope limitation, not silently
imputed evidence.

## 4. Transparent peer attention

For peer `j`, the implemented attention logit is

\[
\ell_{jt} =
\frac{cos(z_t,z_j)}{\tau}
+ r_\theta([z_t;z_j;z_t-z_j])
- \lambda_d |d_t-d_j|,
\]

with

\[
r_\theta(x)=s_r\tanh(MLP_\theta(x)),\qquad
\alpha_{jt}=softmax_{j\in\mathcal{P}_t}(\ell_{jt}).
\]

The final residual-network layer is initialized to all zeros. Consequently the
bootstrap router is exactly descriptor cosine similarity (plus any explicitly
configured depth penalty), not a random routing policy. The default real plan
uses `tau=1`, residual bound `s_r=0.25`, and depth penalty `lambda_d=0`.

The residual may be calibrated jointly with the generator. Every plan records
peer names, similarities, residuals, depth distances, logits, alphas, and an
explicit `self_alpha: 0.0`. These values make routing inspectable; they are not
causal explanations without the pre-registered ablations.

The descriptor peer context is

\[
c_t=\sum_j\alpha_{jt}z_j.
\]

## 5. Structural property maps

Raw cross-layer coordinate identity is not assumed. For each peer matrix, PCSM
constructs nonnegative structural properties.

Centered element magnitude:

\[
P_j^{elem}[a,b]=
\left|\frac{W_j[a,b]-mean(W_j)}
{rms(W_j-mean(W_j))}\right|.
\]

Mean-normalized row and column energy:

\[
r_j[a]=\frac{\sqrt{mean_b(W_j[a,b]^2)+\epsilon}}
{mean_a\sqrt{mean_b(W_j[a,b]^2)+\epsilon}},
\]

\[
c_j[b]=\frac{\sqrt{mean_a(W_j[a,b]^2)+\epsilon}}
{mean_b\sqrt{mean_a(W_j[a,b]^2)+\epsilon}}.
\]

Joint leverage:

\[
P_j^{joint}[a,b]=r_j[a]c_j[b].
\]

The row and column vectors are broadcast to matrix shape when used as coordinate
features. If a generic caller requests a different target shape, the library
uses deterministic nearest-coordinate projection along each dimension. That is
a coarse structural projection only; it is never described as neuron identity.

Projected peer maps are attention-averaged:

\[
\bar P_t=\sum_{j\in\mathcal{P}_t}\alpha_{jt}\Pi_{j\rightarrow t}(P_j).
\]

Static plan generation detaches attention before caching this aggregate. Joint
attention calibration explicitly opts into differentiable aggregation and
recomputes it on each calibration step.

## 6. Eight coordinate features and score head

For each target coordinate, the generator materializes eight explicit maps:

| Symbol | Implementation name | Definition |
|---|---|---|
| \(x_1\) | `task_gradient_magnitude` | RMS-normalized `abs(task gradient)` |
| \(x_2\) | `peer_element_magnitude` | aggregated peer element magnitude |
| \(x_3\) | `peer_row_energy` | aggregated peer row energy, broadcast |
| \(x_4\) | `peer_column_energy` | aggregated peer column energy, broadcast |
| \(x_5\) | `peer_joint_leverage` | aggregated peer joint leverage |
| \(x_6\) | `target_element_magnitude` | target centered element magnitude |
| \(x_7\) | `peer_target_structural_difference` | absolute difference between mean peer and target structural templates |
| \(x_8\) | `gradient_peer_interaction` | normalized gradient magnitude times peer structural template |

Each map is standardized independently:

\[
\hat x_f=\frac{x_f-mean(x_f)}{max(std(x_f),10^{-6})}.
\]

The global context concatenates the 24-value target descriptor, a 24-value task
gradient descriptor, and the 24-value peer summary. A small network
`Linear(72,64) -> GELU -> Linear(64,11)` produces eight feature logits and three
scale controls. The feature weights are

\[
\beta_t=softmax(h_\beta([z_t;z_g;c_t])),
\qquad
S_t[a,b]=\sum_{f=1}^{8}\beta_{t,f}\hat x_f[a,b].
\]

The feature names and weights are persisted per target.

## 7. Exact stable TopK

For a requested fraction `p` over `N` coordinates, production plan generation
resolves K once using round-half-up behavior:

\[
K=max(1,min(N,\lfloor Np+0.5\rfloor)).
\]

The stable selector sorts scores descending and resolves equal scores by the
lower flat row-major index. It returns exactly K binary values:

\[
M=TopKMask(S,K),\quad M\in\{0,1\}^N,\quad ||M||_0=K.
\]

The selected mask is frozen after calibration. In the current static protocol,
support-union K equals declared K; no later training step regenerates support.

## 8. Exact-forward smooth straight-through estimator

Generator meta-calibration needs a gradient through support selection. With
detached K-th score threshold \(q_K\), the smooth surrogate is

\[
M_{soft}=\sigma((S-stopgrad(q_K))/T).
\]

A custom autograd operation is used:

- `forward`: return the hard mask bytes produced by stable TopK;
- `backward`: multiply incoming gradient by
  \(M_{soft}(1-M_{soft})/T\).

The forward path never uses an algebraic hard-plus-soft cancellation, so it is
bitwise identical to the exact binary hard mask. K=0 and K=N have explicit
threshold handling. The current surrogate does not perform an additional soft
mass normalization.

## 9. Bounded post-Adam scale

Let the peer template be the mean of the four peer property channels. Three
context-head outputs `(b,s,p)` produce

\[
A_t[a,b]=a_{min}+(a_{max}-a_{min})\sigma\left(
b+s\widehat{S_t}[a,b]+p\widehat{P_t}[a,b]
\right).
\]

The qualified default bounds are `[0.25,1.0]`. Plans reject non-finite,
non-positive, or out-of-bound values. Because `A` is strictly positive, peers
cannot flip the task-gradient contribution. The optional AdamW decay term can,
as in ordinary AdamW, contribute its own signed shrinkage direction; it is
reported separately from the peer semantics claim.

## 10. Exact-support AdamW

For selected flattened coordinate `i`, with gradient `g_i`,

\[
m_i^{(n)}=\beta_1m_i^{(n-1)}+(1-\beta_1)g_i,
\]

\[
v_i^{(n)}=\beta_2v_i^{(n-1)}+(1-\beta_2)g_i^2,
\]

\[
d_i=\frac{\hat m_i}{\sqrt{\hat v_i}+\epsilon}+\lambda W_i,
\qquad
W_i\leftarrow W_i-\eta A_i d_i.
\]

All three update mechanisms are support-restricted:

1. only selected gradient elements are gathered;
2. first and second moments exist only as K-length FP32 arrays;
3. normalized update and decoupled decay are evaluated only for selected
   values;
4. `A` multiplies the complete selected AdamW direction after normalization;
5. a single `index_copy_` writes only selected native coordinates.

The module also disables every unplanned parameter with `requires_grad=False`
and installs a coordinate hook that zeros unselected gradients on a planned
tensor. The optimizer's indexed gather/write remains the decisive protection
against stale momentum and weight decay.

Optimizer checkpoints contain plan hash, support hash, exact int64 indices,
K-length FP32 moments, step, stabilizer, and optimizer hyperparameters. Resume
fails if any plan identity, shape, support, or stabilizer differs. No dense stale
moment state is accepted.

The central invariant is bitwise:

\[
M_i=0 \implies bits(W_i^{(n)})=bits(W_i^{(0)})
\]

for all optimizer steps absent external mutation, including BF16/FP16 native
weights, momentum, nonzero weight decay, and resumed state.

## 11. Immutable plan and byte-identical reload

A rich plan records at least:

- base checkpoint SHA-256 fingerprint;
- parameter name, role, module type, layer depth, and shape;
- requested fraction, exact K, and full binary support;
- full bounded scale matrix;
- peer names and attention vector;
- feature names and weights;
- generator architecture/configuration;
- per-mask SHA-256, support hash, plan hash, and safetensors file hash.

Plans are published as `plan.json` plus `tensors.safetensors` through a staging
directory and atomic rename. Existing destinations are refused. Loading rejects
missing or extra files, wrong base identity, malformed fields, tensor hash
mismatch, mask hash mismatch, non-binary masks, or invalid scales.

The workflow is:

\[
load\ base \rightarrow calibrate\ without\ step \rightarrow save\ plan
\rightarrow discard \rightarrow reload\ identical\ base \rightarrow apply.
\]

For real Phase C, each seed/budget/arm cell loads a fresh model. The manifest
also binds the model config, tokenizer, frozen sample content and token lengths,
gradient caches, and exact plan cell before evaluation.

## 12. Post-update native delta audit

The auditor compares all before/after native parameters without dtype
conversion. A coordinate is “changed” if any raw byte differs, so `+0.0` to
`-0.0` and changed NaN payloads are detectable.

For every tensor and in aggregate it records:

- planned K and support-union size/fraction;
- total and changed native coordinates;
- selected changed count and outside-support changed count;
- L2 and max-absolute deltas inside and outside support;
- plan/support hashes and example leakage coordinates.

Any changed bit outside the declared support makes the run invalid. The real
runner separately fingerprints every unplanned parameter before and after the
step; that hash must also remain identical.

## 13. Optional structural regularizer

The core method uses no structural regularizer (`coefficient=0`). The independent
ablation, if run, compares normalized row and column energy profiles:

\[
R_{struct}=||r(W'_t)-\bar r_{peer}||_2^2+
||c(W'_t)-\bar c_{peer}||_2^2.
\]

It does not minimize raw `||W_t-W_j||`, which could erase specialization. Any
regularizer-on result must be labeled separately; it cannot be bundled into the
primary PCSM claim.

## 14. Complexity and memory

Let `P` be eligible peer count, `D` descriptor width, `F=8` coordinate features,
and `N=d_out*d_in`.

| Operation | Current asymptotic cost | Notes |
|---|---:|---|
| Descriptor scan | `Theta(PN)` | 24 FP32 outputs per matrix |
| Peer attention | `Theta(PD + PDH)` | Small residual MLP; negligible versus maps |
| Property-map construction | `Theta(PN)` | Current real builder caches all peer maps |
| Coordinate feature/scoring | `Theta(FN)` | Dense FP32 planning tensors |
| Stable TopK | `O(N log N)` | Current implementation uses stable full `argsort` |
| AdamW dynamic moments | `Theta(K)` | Two K-length FP32 arrays |
| Native optimizer write | `Theta(K)` | Indexed gather/update/write |
| Model forward/backward | Dense model cost | No sparse kernel is used |

For the Phase-C target `1024 x 3072`, `N=3,145,728`:

| Buffer | Bytes | MiB |
|---|---:|---:|
| BF16 target weight | 6,291,456 | 6 |
| FP32 score matrix | 12,582,912 | 12 |
| FP32 stabilizer matrix | 12,582,912 | 12 |
| Boolean/uint8 mask | 3,145,728 | 3 |
| Eight-channel FP32 local feature tensor | 100,663,296 | 96 |
| One four-channel FP32 structural tensor | 50,331,648 | 48 |

These are individual buffers, not a claim about exact process peak. The current
real builder also caches property maps for 27 same-role peers, and PyTorch
temporaries coexist during generation. Resource preflight records GPU and
system memory before real execution; target plans are processed sequentially.

The run artifact reports sparse optimizer identity/state as approximately
`16K` bytes (int64 support plus two FP32 moments). Selected BF16 delta plus
indices is approximately `10K` bytes; selected stabilizers add `4K` bytes when
materialized. The persisted rich plan intentionally stores a dense uint8 mask
and dense FP32 `A` for auditability.

Sparse optimizer state and delta storage do **not** imply faster training. Dense
forward/backward matrix multiplications still dominate, and stable full sort
adds planning cost. Wall-clock, tokens/second, peak allocated/reserved VRAM,
gradient storage, optimizer bytes, and delta bytes must be measured rather than
inferred.

## 15. Limitations and interpretation

- A structural coordinate map is not a neuron-alignment theorem.
- Learned attention is associational until causal ablations support more.
- Positive synthetic evidence only establishes that the implementation can
  exploit peer-dependent structure when the synthetic optimum is constructed
  from that structure.
- Real peers approximately equal to shuffled peers would reject the
  coworker-coordinate hypothesis even if both beat simple baselines.
- Real peers above shuffled but equal to target-only may reflect generator
  capacity rather than peer information.
- A one-step advantage may not survive multi-step SFT dynamics.
- Improvement with retention collapse is not a successful adaptation result.
- If only `A` helps, the evidence supports optimizer conditioning, not support
  selection.
- The first real test covers one model, one semantic role, and one target
  matrix. Generalization requires replication.
- No inference-time, biological, universal cooperation, novelty, or speed claim
  follows from the design.


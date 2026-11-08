# TGSA architecture specification

## 1. Research object

Trajectory-Guided Sequential Attention replaces the independent parallel-head ensemble inside one attention block with an ordered sequence of head computations. The head index is treated as a discrete computational-depth coordinate. A shared carrier tracks the evolving representation, while finite differences of that carrier estimate the direction and stability of computation.

This document defines the Phase-0 semantics that all optimized or Qwen-compatible implementations must preserve.

## 2. Notation and tensor contract

Let:

- `B` be batch size;
- `T` be sequence length;
- `D` be model width;
- `H` be the number of sequential heads;
- `d = D/H` be head width;
- `X ∈ R^(B×T×D)` be the attention-block input.

Split the features into ordered groups:

```math
X=[x_1\|x_2\|\cdots\|x_H],\qquad x_j\in\mathbb{R}^{B\times T\times d}.
```

TGSA also maintains a carrier:

```math
c_j\in\mathbb{R}^{B\times T\times d}.
```

All carrier states share one coordinate system. This is essential: subtracting arbitrary raw head outputs is poorly identified when different heads occupy unrelated learned subspaces.

Initialization:

```math
c_0=P_0X,\qquad c_{-1}=c_0.
```

The implementation realizes the equivalent first-step condition by setting the previous carrier equal to `c[0]`, so the first velocity is exactly zero.

## 3. Finite-difference trajectory

The first-order trajectory signal is:

```math
v_{j-1}=c_{j-1}-c_{j-2}.
```

Because head depth is discrete with unit spacing, this is a finite-difference approximation to change along computational depth:

```math
\frac{dc}{dj}\bigg|_{j-1}\approx c_{j-1}-c_{j-2}.
```

It is fully differentiable with respect to every upstream parameter that produced the two carrier states.

The optional second-order quantity is reserved for a later ablation:

```math
a_{j-1}=v_{j-1}-v_{j-2}
       =c_{j-1}-2c_{j-2}+c_{j-3}.
```

Acceleration is intentionally excluded from the primary Phase-1 operator to keep causal attribution clean.

## 4. Linear-collapse lemma

A naïve formula such as

```math
z_j=A c_{j-1}+B c_{j-2}+C(c_{j-1}-c_{j-2})
```

reduces to

```math
z_j=(A+C)c_{j-1}+(B-C)c_{j-2}.
```

Therefore, simply appending a linearly projected difference does not by itself establish a distinct mechanism. It may only reparameterize a two-state linear recurrence.

TGSA crosses this identifiability boundary by using the trajectory through nonlinear and multiplicative controls:

- learned nonlinear encoding;
- state-, velocity-, predictor-, and correction gates;
- Q/K/V conditioning;
- a token-dependent attention prior;
- predictor–corrector dynamics;
- later, differentiable compression and stopping.

The ablation ladder includes a raw-delta arm precisely to measure whether the stronger control formulation adds value beyond a reparameterizable residual.

## 5. Trajectory encoder

For head `j`, let `s[j-1]` be the selected control signal. The primary TGSA operator uses `s[j-1]=v[j-1]`; the history control defined below uses a matched non-directional signal. Then:

```math
m_j=\Phi_j(\operatorname{RMSNorm}(s_{j-1})).
```

The learned reference encoder is low-rank:

```math
\Phi_j(v)=U_j\operatorname{SiLU}(V_jv),
```

where `V[j] ∈ R^(r×d)`, `U[j] ∈ R^(d×r)`, and `r << d` when scaled. The final projection is initialized with a small nonzero variance. This keeps the initial trajectory perturbation controlled while permitting immediate gradient flow through both factors.

Ablation modes:

- `none`: `m[j]=0`;
- `raw`: the selected control signal is passed directly;
- `learned`: the selected control signal uses the low-rank nonlinear encoder.

### Parameter-identical history control

The primary TGSA arms select the finite-difference velocity as their control signal. A deliberately strong no-delta control instead selects

```math
\mu_{j-1}=\tfrac{1}{2}(c_{j-1}+c_{j-2}).
```

Both `history_control` and `learned_delta` then use the same encoder and the same downstream pathways. They have exactly the same parameter count. Given the current carrier, both also expose equivalent information about the preceding state:

```math
c_{j-2}=c_{j-1}-v_{j-1}
```

and

```math
c_{j-2}=2\mu_{j-1}-c_{j-1}.
```

Thus their comparison targets the directional finite-difference inductive bias, not extra history or capacity. The first head receives a zero control signal in both arms because no inter-head transition exists yet.

## 6. Bounded control gates

The reference kernel computes feature-wise gates from the current feature group, carrier, and encoded control signal:

```math
[g_j^s,g_j^v,g_j^p]
=\sigma\left(W_j^g[x_j;c_{j-1};m_j]+b_j^g\right).
```

Their meanings are:

- `g^s`: amount of predicted carrier injected into the head input;
- `g^v`: amount of transformed control signal injected (velocity in TGSA arms);
- `g^p`: predictor step size.

The predictor gate is bounded by a configured maximum:

```math
\alpha_j=\alpha_{\max}g_j^p.
```

This prevents an unconstrained trajectory extrapolation from destabilizing the carrier.

## 7. Predictor

The pre-attention predicted state is:

```math
\hat c_j=c_{j-1}+\alpha_j\odot P_jm_j.
```

This step asks, “Given the current state and its recent direction, where should computation move before performing another retrieval?”

It is disabled in cascade, raw-delta, learned-delta, and QKV-only controls as specified in `TGSAConfig.for_ablation`.

## 8. Sequential head input

The feature group is reconstructed with state and trajectory controls:

```math
z_j=x_j
+g_j^s\odot S_j\hat c_j
+\lambda_v g_j^v\odot R_jm_j.
```

The final term is absent when trajectory injection is disabled. Unlike the weak linear formulation, `m[j]` affects the input through a nonlinear encoder and an input-dependent multiplicative gate.

## 9. Trajectory-conditioned Q/K/V

The strongest reference operator allows distinct trajectory transformations for the retrieval request, memory index, and retrieved content:

```math
q_j=W_j^Q(z_j+I_j^Qm_j),
```

```math
k_j=W_j^K(z_j+I_j^Km_j),
```

```math
u_j=W_j^V(z_j+I_j^Vm_j).
```

These paths have different interpretations:

- Q conditioning changes what the current step seeks;
- K conditioning changes how the current step indexes available information;
- V conditioning changes what information is delivered after selection.

They must be ablated separately in later mechanistic work even though Phase 1 first evaluates them jointly.

## 10. Soft trajectory-aware token significance

For each token:

```math
r_{j,t}=\sigma\left(w_j^r[\hat c_{j,t};m_{j,t}]+b_j^r\right).
```

The mechanism does **not** assume that larger velocity is always more important. A learned function determines whether a changing, stable, or contradictory representation should receive more attention.

The score enters attention as a differentiable key prior:

```math
\bar r_{j,t}=r_{\min}+(1-r_{\min})r_{j,t},
```

```math
L_j=\frac{q_jk_j^\top}{\sqrt d}+M_{causal}+B_{external}
+\lambda_j\log(\bar r_j).
```

`r_min > 0` prevents `log(0)`. The Phase-0 kernel never performs hard token deletion. Hard or top-k compression belongs to a later phase only after the soft mechanism has been isolated.

## 11. Attention correction

```math
A_j=\operatorname{softmax}(L_j),
```

```math
h_j=O_j(A_ju_j).
```

The correction gate observes the predicted state, attention result, and trajectory encoding:

```math
\beta_j=\beta_{\max}\sigma\left(W_j^c[\hat c_j;h_j;m_j]+b_j^c\right).
```

The next carrier is:

```math
c_j=\operatorname{RMSNorm}
\left(\hat c_j+\beta_j\odot C_jh_j\right).
```

This is a learned predictor–corrector step:

```text
state → trajectory prediction → attention retrieval → state correction
```

rather than a one-shot parallel ensemble.

## 12. Output

The per-head attention outputs are concatenated and projected:

```math
y=W^O[h_1\|h_2\|\cdots\|h_H].
```

The reference module returns `y` without the enclosing Transformer residual. A model integration must retain the host block’s original pre/post-normalization and residual convention.

## 13. Causal semantics

Each head uses the same upper-triangular causal constraint. For token `t`, the initial carrier is token-local, and every head may attend only to positions `≤t`. By induction over head index:

1. `c[0,t]` depends only on `X[t]`;
2. if `c[j-1,t]` and `c[j-2,t]` contain no future information, then `v[j-1,t]`, `m[j,t]`, gates, and the predictor contain no future information;
3. causal attention at head `j` retrieves only positions `≤t`;
4. therefore `c[j,t]` remains future-independent.

The test suite perturbs an entire suffix and requires all prefix outputs to remain invariant within numerical tolerance.

## 14. Differentiability

Every Phase-0 operation is differentiable almost everywhere:

- subtraction;
- RMS normalization;
- low-rank matrix products;
- SiLU and sigmoid gates;
- attention softmax;
- logarithm on a strictly positive bounded score;
- residual and correction updates.

The test suite verifies that gradients reach the learned trajectory encoder, the input, and all ablation arms without NaN/Inf.

## 15. Telemetry contract

Summary telemetry has shape `[B,H]` and records:

- velocity RMS;
- acceleration RMS;
- cosine between consecutive velocities;
- carrier-update RMS;
- state, control-signal, predictor, and correction gate means;
- token-importance mean/min/max;
- attention entropy.

Full telemetry optionally records carrier states, velocity tensors, token scores, and attention maps. Full attention maps are forbidden in long-context production runs unless explicitly required because storage is `O(BHT²)`.

Interpretation is preregistered cautiously:

- declining velocity may indicate convergence, but could also indicate collapse;
- negative consecutive velocity cosine may indicate oscillation, but not necessarily error;
- sparse importance may indicate useful selection, but not necessarily faithful explanation.

Correlations become evidence only after held-out validation.

## 16. Complexity

For the clear independent-KV reference implementation:

- attention arithmetic remains approximately `O(BHT²d)=O(BT²D)`;
- head computations are serialized, reducing parallel throughput;
- control projections add approximately `O(BTDd)` with coefficients depending on enabled paths;
- summary telemetry is `O(BH)` storage;
- full attention telemetry is `O(BHT²)` storage.

Thus the theoretical attention FLOP order is not multiplied by `H`, but realized latency can rise sharply because work that was parallel is now sequential and current kernels are unfused. Quality must therefore be evaluated on a Pareto surface against prefill latency, decode latency, memory, and realized FLOPs.

## 17. Autoregressive KV-cache semantics

At decode time, token `t` traverses heads sequentially. Head-specific K/V for token `t` are produced and appended after the corresponding head computes its trajectory-conditioned state. Previous tokens’ head-specific K/V are already cached.

Three cache variants must be kept distinct:

1. **independent sequential KV** — one K/V stream per head; most expressive;
2. **grouped sequential KV** — query heads share K/V within native GQA groups;
3. **native anchored KV** — retain pretrained K/V and condition mainly Q/carrier paths; cheapest retrofit.

The reference kernel implements the semantic equivalent of variant 1 for full-sequence training. `QWEN_INTEGRATION_PLAN.md` specifies the production transition.

## 18. Stability invariants

Every implementation must preserve:

- a shared carrier coordinate system;
- finite `importance_floor` before logarithm;
- bounded predictor and correction step sizes;
- no all-future attention row under causal masking;
- finite softmax accumulation (float32 in the reference path);
- explicit telemetry opt-out for latency measurement;
- no hidden hard-pruning in soft-significance arms;
- exact mechanism flags for registered ablations.

A faster kernel that violates these invariants is a different experiment.

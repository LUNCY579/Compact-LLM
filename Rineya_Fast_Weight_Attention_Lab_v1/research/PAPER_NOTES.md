# Paper notes — Fast Weight Attention / FALCON (arXiv:2608.27763)

Source: Zhang et al., "Fast Weight Attention for Continual Learning", arXiv:2608.27763
(also titled FALCON on the project page). Notes below are taken from the paper,
not from social-media summaries.

This file has three labeled parts. A result in this lab that depends on part C
must **not** be described as a FALCON result.

## A. Mechanisms reproduced from the paper

### A.1 Temporal alignment and read-after-write

The paper fixes **read-after-write (RAW)** autoregressive semantics: at token
index `t` the model reads from the **updated** state `(S_t, …)` and uses that
representation to predict token `t+1`.

For the prefix-prediction / next-latent objective, the local fast-memory
example revealed at step `t` is the **prefix-aligned** pair

```
(x_t, y_t) = (φ(k_{t-1}), v_t)
```

not the common same-step pair `(φ(k_t), v_t)`. Same-step association remains
causal; it optimizes a **different** internal objective. This lab uses the
prefix-aligned pair.

At the boundary `t = 1` the paper uses the sentinel `x_1 = 0` ("no causal pair
yet"): no write, read from `S_0` (zeros here).

### A.2 Read

After the write,

```
m_t = S_t^T φ(q_t)
```

(the paper writes `o_t = S_t^T q_t` with `q_t := φ(q_t)`).

### A.3 Falcon-1 / Falcon-2 / Falcon-3 (regression family)

All three use the same causal pair. Residual:

```
r_t = v_t − S_{t−1}^T x_t
```

**Falcon-1** (scalar NLMS):

```
S_t = (1 − η_t λ_t) S_{t−1} + η_t x_t r_t^T
η_t = β_t / (‖x_t‖² + λ_t + ε)
```

**Falcon-2** (per-column extension; **this lab**):

```
S_t = S_{t−1} (I − λ_t diag(η_t)) + x_t (η_t ⊙ r_t)^T
η_{j,t} = β_{j,t} / (‖x_t‖² + λ_t + ε)
```

`η_t ∈ R^{d_v}` is one rate per value channel. Descent requires `0 < β_{j,t} < 2`.

**Falcon-3**: sliding-window mini-batch regression over the last `B` causal
pairs. **Not implemented** in v1 (correctness-first recurrent form only).

### A.4 Inner-product family (Falcon-1A / 2A / 3A)

These write the target directly (`y_t` / `v_t`) instead of the residual `r_t`.
The denominator is write-magnitude normalization rather than curvature
normalization. **Not the default** in this lab.

### A.5 Plasticity and forgetting

- Plasticity: dimensionless gain `β` (scalar or per-column).
- Forgetting: ridge shrinkage `λ_t ≥ 0`. When `λ_t > 0` the paper uses
  **positive-decay renormalization** for chunk-parallel kernels. The sequential
  recurrence above already includes `(I − λ diag(η))`.

### A.6 Recurrent state shape

Per head, `S ∈ R^{d_x × d_v}`. In the paper's multi-head factorization each
head has its own `S`. This lab uses **KV-head** factorization (8 heads) with
`d_x = d_v = rank` (default 16), a bounded low-rank state.

### A.7 Numerical stabilization

- `ε > 0` in the NLMS denominator (paper: any `ε > 0` only decreases `η` and
  preserves descent). Lab default `ε = 1e-6`.
- `β` is parameterized so the realized value stays in `(0, 2)` via a scaled
  sigmoid.
- Recurrence accumulates in float32 even when the backbone is float16.
- Implementations the paper also discusses (masked-parallel, chunk-parallel WY /
  Gram forms) are **out of v1 scope**.

### A.8 What "continual learning" means here

The paper treats the **state transition as an online learning rule** on a
bounded recurrent matrix. That is **inference-time / sequence-time fast-weight
state**, not a permanent change to the slow weights. This lab uses:

- fast-weight state
- recurrent memory
- inference-time state update
- online local update

It does **not** claim lifelong learning or that model parameters are
fine-tuned during inference.

## B. Engineering adaptations for Supa-A1.1

These are **not** FALCON paper claims.

1. **Auxiliary pathway.** Standard Transformer attention is left intact. FWA
   runs beside it and merges with a learned gate `α` initialized at **0**:

   ```
   h'_t = h_attn,t + α_t · P(m_t)
   ```

   When `α = 0` the merge is a hard skip, so unmodified Supa must be recovered.

2. **Feature maps `φ`.** Learned linear maps from the post-layernorm hidden
   state (the attention module's input) into per-KV-head rank-`R` features,
   plus RMSNorm. The paper allows a general `φ`; we do not use the paper's
   kernel feature map.

3. **Low rank.** `R = 16` rather than `head_dim = 128`, to stay inside an
   8 GB-class envelope.

4. **Placement.** 1–3 decoder layers only. P1 uses layer 14 (0-based middle of
   28). Attention is **not** replaced (Arm E deferred).

5. **Frozen backbone.** Only FWA tensors train. `use_cache = False` during
   training and recurrent eval (KV cache and this recurrence are different
   memories; mixing them is refused, not silently degraded).

6. **Arm B.** Same tensors and one-step write/read algebra, but **no state
   carry** (`S` is reset every token; same-step `(k_t, v_t)`). This is the
   parameter-matched static control, not a FALCON variant.

## C. Novel Rineya-specific hypothesis (Arm D only)

**Not part of FALCON.**

Let `s_t = σ(w^T h_t + b) ∈ (0, 1)` be a token write-significance.

```
β_t = β_min + s_t (β_max − β_min)
λ_t = λ_max − s_t (λ_max − λ_min)
```

Interpretation (hypothesis, not a result): high significance → stronger write,
slower forget; low significance → weak write, faster forget.

Arm C and Arm D differ by this gate only. Do not assume it helps.

---

## Addendum 1 — retention horizon of the Falcon-2 recurrence (measured)

Appended after the P2 post-mortem. Part A analysis (a property of the paper's
algebra), plus one correction to part B.

### A.9 The `β = 1` deadbeat point, and what sets the memory horizon

Two *different* forgetting scales act in A.3, and only one is `λ`.

**Ridge decay** applies to every stored direction:

```
decay = 1 − λ η,    η = β / (‖x‖² + λ + ε)
```

**Replacement along the written direction.** Projecting the Falcon-2 update onto
the current key `x`:

```
S_t^T x = (1 − λη) S_{t−1}^T x + η‖x‖² (v − S_{t−1}^T x)
        = (decay − η‖x‖²) S_{t−1}^T x + η‖x‖² v
```

so retention along `x` is `decay − η‖x‖²`. Since `η‖x‖² = β‖x‖²/(‖x‖²+λ+ε)`,
this is `≈ 1 − β` for small `λ`. At **`β = 1` the term vanishes**: the write
*completely replaces* whatever was stored along that direction. This is the
NLMS/RLS deadbeat (exact-projection) point. `β ∈ (0,2)` is the descent interval
(A.3), but `β = 1` is its most destructive interior point for retention.

With the P1/P2 initialisation (`β_raw = 0 → β = 1.0`, `λ_raw = −2 → λ = 0.1269`)
and an RMSNorm'd rank-16 key (`‖x‖² = 16`):

- `η = 0.0622`
- ridge decay `= 0.99211` per token → **half-life 87.5 tokens**
- retention along the written direction `= 0.992107 − 0.992129 = −2.2e−5 ≈ 0`

Measured on the trained P2 adapters, not derived from the prompt. The state norm
saturates (`‖S_t‖_F` peaks at t≈390 on a 990-token item, then declines), which
is the signature of a bounded state at its steady state rather than one
accumulating a long prefix.

**Consequence for experiment design.** The horizon is set by `(β, λ, ‖x‖²)`, and
`‖x‖²` is pinned to `rank` by the RMSNorm on the key features. So the horizon is
effectively an initialisation choice. Testing long-context retention with a
horizon shorter than the context tests nothing. P2b matches
`β_raw = −1.1` (`β = 0.499`, ~50 % replacement) and `λ_raw = −5.9`
(`λ = 2.74e−3`, half-life ~8100 tokens) across arms B/C/D.

This is an observation about the paper's recurrence, not a modification of it.

### Correction to B.1

B.1 states the merge gate is "initialized at **0**". That is what both YAML
configs declare, but `train.py` hardcoded `init_alpha = 0.05` and never read the
config key, so **every P1/P2 run started at α = 0.05.**

The `α = 0` claim in B.1 is also load-bearing in a way worth recording: `P` is
zero-initialised, so at exactly `α = 0` both `dL/dα` and `dL/dP` are identically
zero and the whole block is a permanent no-op. `α = 0` is therefore correct as
the *equivalence* check (the zero-gate test in ZERO_TRAINING_SANITY.md) but
cannot be a *training* initialisation. From P2b the value is passed by the
caller and recorded in the training manifest.

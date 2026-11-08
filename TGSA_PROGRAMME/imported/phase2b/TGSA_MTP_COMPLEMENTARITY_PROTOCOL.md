# TGSA + Multi-Token Prediction Complementarity Protocol

**Status:** implementation-complete protocol; synthetic smoke validated; real
Rineya evidence not yet run on this branch.

## 1. Research question

Trajectory-Guided Sequential Attention (TGSA) deliberately introduces serial
head-level refinement. Multi-token prediction (MTP) may recover part of the
resulting inference latency by proposing several future tokens from one causal
state. The combined hypothesis has two logically separate parts:

1. **quality axis:** TGSA improves or preserves model quality relative to a
   parallel-attention control; and
2. **serial-decoding axis:** MTP reduces the number of serial target-model
   forwards required to produce the same greedy token sequence.

A wall-clock speedup is not assumed. Sequential TGSA heads increase work per
forward, while MTP reduces the number of forwards. Both quantities must be
measured on the target hardware before any end-to-end speed claim.

## 2. Mechanisms

### 2.1 TGSA

The existing reference implementation keeps a shared carrier across sequential
heads, forms finite differences between carrier states, and optionally uses the
encoded trajectory for state injection, Q/K/V conditioning, prediction,
correction, and token-importance biasing. Phase 2 does not alter those semantics.
The default combined candidate is the existing `full` arm.

### 2.2 MTP

The ordinary LM head remains the sole predictor for offset 1. For every causal
hidden state `h_t`, auxiliary residual heads predict offsets `2..K`:

```math
z_t^{(k)} = h_t + \sigma(g_k)\,U_k\,\mathrm{SiLU}(D_k\,\mathrm{RMSNorm}(h_t)),
```

```math
p(x_{t+k}\mid x_{\le t}) = \mathrm{softmax}(E z_t^{(k)}), \qquad k=2,\ldots,K.
```

`U_k` is zero-initialized, so every head begins as the identity in the parent
vocabulary geometry when `E` is tied to the model output embedding. Auxiliary
heads never feed the main LM logits.

The joint objective is:

```math
\mathcal L = \mathcal L_{AR} + \lambda
\frac{\sum_{k=2}^{K} w_k \mathcal L_k}
     {\sum_{k=2}^{K} w_k},
\qquad w_k=\rho^{k-2}.
```

Normalizing the horizon weights prevents the auxiliary loss magnitude from
increasing merely because a longer horizon is selected.

## 3. Leakage and masking invariants

- Future tokens are labels only; they are never inputs to an auxiliary head.
- Padding must be valid at both the source and target positions.
- In SFT, `target_mask` is true only for final-assistant tokens. Prompt tokens
  may provide context but never contribute target loss.
- The offset-1 and offset-`k` objectives share one explicit mask convention.
- A chat-template prompt must be a token-prefix of the complete row; otherwise
  the real-model trainer fails closed rather than guessing the boundary.

## 4. Primary 2 x 2 design

Every seed contains a complete paired block:

| Cell | Attention | Objective | Decoder |
|---|---|---|---|
| `parallel_ar` | parallel MHA | next-token only | greedy / verified reference |
| `parallel_mtp` | parallel MHA | AR + future offsets | verified MTP |
| `tgsa_ar` | TGSA `full` | next-token only | greedy / verified reference |
| `tgsa_mtp` | TGSA `full` | AR + future offsets | verified MTP |

All cells contain the same MTP parameter structure. AR cells keep those tensors
dormant and frozen. Within each attention family, AR and MTP cells are exact
deep copies at initialization. Across families, every shape-compatible tensor is
copied from one canonical source; structurally different attention tensors are
recorded explicitly.

### Secondary capacity controls

`parallel_aux_next` and `tgsa_aux_next` use the same auxiliary heads and number
of supervised positions, but every auxiliary head predicts offset 1. These
non-gating controls distinguish a future-horizon inductive bias from merely
adding trainable auxiliary capacity or extra gradient signal.

A shuffled-batch future target is implemented as a negative causal control. It
requires a fixed-point-free donor map and is never a primary arm.

## 5. Exact verified decoding

MTP proposals are not trusted directly.

1. Offset 1 comes from the main LM head.
2. Offsets 2..K come from auxiliary heads at the same causal state.
3. The target model evaluates the proposed block once.
4. Proposals are accepted in order while they equal the main LM argmax.
5. At the first mismatch, the main LM token is emitted and later proposals are
   discarded.
6. A fully accepted verification result is reused as the next proposal state.

This produces exactly the ordinary greedy sequence. Exact equality is a hard
gate. The decoder reports both the speedup factor and the fractional round reduction:

```math
\text{serial speedup}=\frac{\text{generated tokens}}{\text{target-model forward calls}},
\qquad
\text{round reduction}=1-\frac{\text{target-model forward calls}}{\text{generated tokens}}.
```

The factorial gate uses the fractional reduction because a threshold such as `0.10` has the direct meaning “10% fewer serial rounds.” Both are algorithmic proxies, not wall-clock or FLOP claims.

## 6. Estimands

For every metric, effects are computed inside seed before aggregation:

- `mtp_on_parallel = parallel_mtp - parallel_ar`
- `mtp_on_tgsa = tgsa_mtp - tgsa_ar`
- `tgsa_without_mtp = tgsa_ar - parallel_ar`
- `tgsa_with_mtp = tgsa_mtp - parallel_mtp`
- `interaction = (tgsa_mtp - tgsa_ar) - (parallel_mtp - parallel_ar)`
- `combined_vs_parallel_ar = tgsa_mtp - parallel_ar`

Signs are oriented so positive means improvement. Missing cells or duplicate
seed/cell records fail closed. A positive interaction is reported separately
from practical complementarity: MTP can compensate for latency additively
without a statistically super-additive interaction.

## 7. Gates

A combined `GO` requires all of the following in a confirmatory configuration:

1. every verified decode is exactly greedy-equivalent;
2. attaching dormant MTP heads and the zero-start retrofit preserve the main
   path at initialization within the registered tolerance;
3. the closed evidence verifier passes;
4. the arm-blind competence report is `GO` before confirmatory dispatch;
5. `tgsa_mtp` is non-inferior to `parallel_ar` on the registered quality metric;
6. MTP improves TGSA's serial-round metric by the registered minimum.

A positive efficiency interaction can be made mandatory in a later
preregistration, but is not silently required in the initial complementarity
study.

## 8. Real Rineya retrofit path

The combined branch selectively reuses the validated pretrained-retrofit
safeguards:

- Qwen3/Rineya geometry, GQA, RoPE, Q/K normalization, and cache semantics;
- zero-start TGSA conditioning;
- deterministic layer conversion and tensor-reuse manifests;
- main-logit drift measured in the actual dtype and attention implementation;
- adapter-only checkpointing; and
- progressive warm start: `adapters -> attention -> converted_layers -> all`.

MTP wraps final hidden states and the existing output embedding. It does not
replace `lm_head`, modify cache objects, or change ordinary logits. The first
real run should convert a small distributed layer set, validate drift and exact
decoding, train only TGSA/MTP adapters, then expand placement only after quality
preservation is demonstrated.

## 9. Claim ceiling

The synthetic recurrence study can validate implementation, causal masking,
paired initialization, exact decoding, and statistical plumbing. It cannot
establish broad Rineya quality, long-context improvement, or wall-clock speedup.
Those require a real pretrained checkpoint, held-out Rineya evaluation, and
hardware measurements under the deployment kernel.

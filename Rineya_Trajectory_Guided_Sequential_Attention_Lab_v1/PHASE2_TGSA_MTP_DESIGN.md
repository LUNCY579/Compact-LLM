# TGSA Phase 2: competence-gated multi-token prediction complementarity

**Implementation status:** reference implementation and non-evidentiary rehearsal are available.

**Parent baseline:** `research/tgsa-phase1-v4-competence-first` at
`8b5737e260b5478427b8bb033cffa394a5f3fc5d`.

**Current claim ceiling:** this phase can validate mechanics, gradients, paired
initialization, future-token learning, development-set contrasts, and the shape
of a future speculative decoder. It does **not** yet establish TGSA efficacy,
Rineya quality gains, or tokens-per-second gains. Phase 1 v4 produced a frozen
regime-family **rejection** (`V4_REGIME_FAMILY_REJECTED`): both competence
measurements returned `NO_GO`, the pointer families remained inert, and
`no_primary_family_saturated` closed the ladder. The reference TinyLM has no
production KV-cache verifier. Old-band Phase 2A runs remain development-only.

---

## 1. Research hypothesis

TGSA and multi-token prediction (MTP) address different bottlenecks.

* TGSA makes attention heads sequential and exposes a differentiable state
  trajectory. Its intended benefit is better iterative refinement, at the cost
  of less head-level parallelism.
* MTP asks one current representation to predict several future token horizons.
  It can provide denser training supervision. If future proposals are accepted
  by a target verifier, it can also amortize one expensive target pass over
  several generated tokens.

The complementarity hypothesis is therefore two-part:

1. **quality:** TGSA's state trajectory may make farther-future predictions more
   coherent, while MTP may force TGSA's final representation to preserve a
   longer planning horizon;
2. **runtime:** a cache-aware MTP verifier may recover part of TGSA's sequential
   inference cost by accepting multiple proposed tokens per target pass.

The second statement is a runtime hypothesis, not a consequence of MTP training
accuracy. It receives a separate later benchmark gate.

---

## 2. Existing TGSA semantics remain authoritative

For layer `l`, token position `t`, and sequential head `j`, the reference kernel
maintains carrier states

\[
  c_{l,t}^{(0)}, c_{l,t}^{(1)}, \ldots, c_{l,t}^{(H)}.
\]

The registered finite-difference coordinate is

\[
  v_{l,t}^{(j)} = c_{l,t}^{(j)} - c_{l,t}^{(j-1)}.
\]

The information-matched history coordinate is

\[
  m_{l,t}^{(j)} = \frac{c_{l,t}^{(j)} + c_{l,t}^{(j-1)}}{2}.
\]

The existing TGSA operator already emits the tensor supplied to its trajectory
control path as full telemetry with shape

\[
  S_l \in \mathbb{R}^{B \times H \times T \times d_h}.
\]

Phase 2 transposes and flattens this exact tensor to

\[
  \tau_l \in \mathbb{R}^{B \times T \times D},
\]

without inventing a second trajectory definition. Consequently:

* `learned_delta + trajectory MTP` receives finite differences;
* `history_control + trajectory MTP` receives matched two-state history means;
* `raw_delta`, `qkv_conditioned`, `predictor_corrector`, and `full` retain their
  registered TGSA semantics;
* head 0 remains the registered zero-control head;
* ordinary TGSA calls remain unchanged and may continue to request no telemetry
  or detached telemetry.

This is the load-bearing identifiability choice. A separately designed MTP
trajectory encoder would confound “MTP helps” with “an unmatched side channel
was added.”

---

## 3. Sequential MTP reference module

Let `h_t` be the final normalized decoder state and let `E` and `W_LM` be the
backbone's shared token embedding and LM head. The ordinary next-token
prediction is

\[
  p_t^{(1)} = \operatorname{softmax}(W_{LM}h_t).
\]

For horizon `k = 2, ..., K`, the implemented reference module uses

\[
\begin{aligned}
  e_t^{(k-1)} &= E(x_{t+k-1}), \\
  q_t &= P_\tau(\operatorname{RMSNorm}(\tau_t)), \\
  r_t^{(k)} &= [z_t^{(k-1)}; e_t^{(k-1)}; q_t], \\
  z_t^{(k)} &= z_t^{(k-1)} +
    \sigma(G_kr_t^{(k)}) \odot
    U_k\operatorname{SiLU}(D_k\operatorname{RMSNorm}(r_t^{(k)})), \\
  p_t^{(k)} &= \operatorname{softmax}
    (W_{LM}\operatorname{RMSNorm}(z_t^{(k)})).
\end{aligned}
\]

During training, `x[t+k-1]` is teacher-forced. During recursive drafting, it is
the preceding MTP argmax. Token embedding and LM output weights are shared by
reference rather than copied, so no second vocabulary matrix is introduced.

The transition output projections use the same tiny nonzero initialization in
every experimental cell. This makes the cold transition near-identity while
keeping the trajectory-control route differentiable on the first backward pass.

---

## 4. Parameter-matched MTP conditions

Every condition instantiates the identical MTP module with byte-identical
initial weights.

| condition | auxiliary loss | MTP control input | purpose |
|---|---:|---|---|
| `dormant` | excluded | zeros | parameter-structure and training-budget control |
| `backbone` | active | zeros | isolates future-token supervision |
| `trajectory` | active | registered TGSA control tensor | tests trajectory content beyond MTP alone |

`dormant` still computes MTP logits for diagnostics, but the auxiliary branch is
not attached to the training objective. It is intentionally not represented as
`0 × loss`, because zero-valued gradients could still permit decoupled weight
decay to alter dormant parameters.

`parallel_mha` has no trajectory coordinate. `cascade + trajectory` is excluded
from the core evidentiary matrix because exposing a diagnostic carrier
difference that the cascade itself does not consume would create an unmatched
side channel. Cascade remains the sequential-depth control through its
`dormant` and `backbone` cells.

---

## 5. Training objective

The existing answer-only autoregressive objective is preserved on the registered
answer mask:

\[
  \mathcal{L}_{AR}=\operatorname{CE}(p_t^{(1)},x_{t+1}).
\]

For each horizon `k >= 2`, MTP uses all valid, non-padded causal positions:

\[
  \mathcal{L}_{MTP}^{(k)}=
  \operatorname{CE}(p_t^{(k)},x_{t+k}).
\]

The normalized auxiliary objective is

\[
  \mathcal{L}_{MTP}=
  \frac{\sum_{k=2}^{K}\alpha_k\mathcal{L}_{MTP}^{(k)}}
       {\sum_{k=2}^{K}\alpha_k},
\]

and active MTP cells optimize

\[
  \mathcal{L}=\mathcal{L}_{AR}+\lambda_{MTP}\mathcal{L}_{MTP}.
\]

Normalization prevents a change in horizon count or weighting from silently
changing the total auxiliary-loss scale.

Because dense future supervision includes prompt syntax as well as task answers,
the report separately measures **answer-horizon accuracy**: whether horizon `k`
can predict the task answer from `k-1` positions earlier. This prevents apparent
success based only on predictable formatting tokens.

---

## 6. Staged experimental matrix

### Stage 0 — mechanical validation

The unit suite executes all registered TGSA arms:

* `cascade`
* `history_control`
* `raw_delta`
* `learned_delta`
* `qkv_conditioned`
* `predictor_corrector`
* `full`

It checks future-target alignment, byte-matched initialization, unchanged
ordinary logits at initialization, differentiability of the existing control
coordinate, dormant-branch isolation, evidence schemas, and confirmatory
fail-closed behavior.

### Stage 1 — eight-cell core rehearsal

The supplied CPU smoke configuration uses:

| TGSA arm | dormant | backbone MTP | trajectory MTP |
|---|---:|---:|---:|
| `cascade` | yes | yes | no |
| `history_control` | yes | yes | yes |
| `learned_delta` | yes | yes | yes |

This is sufficient to validate the core MTP interaction and
trajectory-specificity estimands without multiplying hypotheses before
competence and MTP depth are calibrated.

### Stage 2 — competence-certified calibration

Phase 1 v4 will not return `GO` on this band. Calibrate a small frozen
development grid only after a **new** competence preregistration produces
`GO` on a learnable, non-saturated primary-family regime. Vary horizon,
auxiliary-loss weight, transition rank, and training budget. Selection must
be arm-blind: use the weakest required cell's absolute competence, not a
favorable learned-delta contrast.

### Stage 3 — full TGSA ladder

After a core setting is frozen, screen `raw_delta`, `qkv_conditioned`,
`predictor_corrector`, and `full` with the same MTP state, initialization,
examples, order, optimizer, and update budget. These are secondary mechanism
analyses unless explicitly powered and preregistered.

### Stage 4 — family-model retrofit

Port only a selected mechanism to an actual Supa/Rineya checkpoint. Preserve the
checkpoint's tokenizer, RoPE, GQA, tied embeddings, cache semantics, and ordinary
next-token logits at installation.

### Stage 5 — speculative runtime evidence

Implement recursive drafting plus a cache-aware exact target verifier. Benchmark
accepted tokens per verifier call, draft cost, target cost, rollback correctness,
TTFT, TPOT, decode tokens/s, VRAM, GPU utilization, and output parity. Training
horizon accuracy alone is never labeled a speed result.

---

## 7. Registered contrasts

For held-out answer accuracy `A`, the implementation emits paired per-seed
estimands.

**MTP effect within arm `a`:**

\[
  \Delta_{MTP}(a)=A(a,backbone)-A(a,dormant).
\]

**TGSA-mechanism × MTP interaction:**

\[
\begin{aligned}
  I_{MTP}={}&[A(learned\_delta,backbone)-A(learned\_delta,dormant)]\\
             &-[A(history\_control,backbone)-A(history\_control,dormant)].
\end{aligned}
\]

**Trajectory-readout specificity:**

\[
\begin{aligned}
  I_{\tau}={}&[A(learned\_delta,trajectory)-A(learned\_delta,backbone)]\\
              &-[A(history\_control,trajectory)-A(history\_control,backbone)].
\end{aligned}
\]

A trajectory readout gain shared equally by finite differences and matched
history means supports a generic recurrent side channel, not specifically finite
differences. The current rehearsal analyzer reports these estimands
descriptively and refuses to create post-hoc success thresholds.

---

## 8. Evidence package

Every run preserves:

* exact config and dataset-manifest hashes;
* Git head, expected ancestor, branch, and dirty-state snapshot;
* byte hashes for the initial backbone and MTP head;
* proof that MTP heads are identical across cells and backbones are identical
  within TGSA arm across MTP modes;
* an auxiliary-only gradient audit, including direct gradient on the existing
  TGSA control tensor;
* fixed-step training log and final checkpoint hash;
* per-example development predictions;
* answer accuracy/NLL and per-family accuracy;
* token-horizon and answer-horizon accuracy/NLL;
* a teacher-forced consecutive-future-token quality surrogate explicitly marked
  “not throughput”;
* non-gating timing and parameter counts;
* machine-readable run and study reports validated against JSON schemas.

---

## 9. Competence and leakage gates

Phase 1 v4 records `V4_REGIME_FAMILY_REJECTED`. Therefore:

1. a non-confirmatory rehearsal may run on development data to validate code and
   estimate training behavior;
2. the test split is materialized and hash-sealed but never read by this runner;
3. `confirmatory=true` requires `registration.status="frozen"`;
4. it also requires a current Phase-1 competence report accepted by `require_go`
   with verdict `GO`;
5. that check occurs before either output or dataset creation;
6. MTP depth, loss weight, optimization budget, cells, estimators, thresholds,
   and test-bank hash must then be frozen before test evaluation.

A missing, malformed, obsolete-schema, or `NO_GO` report fails closed. The v4
pilot mean cannot be substituted for a competence verdict.

---

## 10. Decision boundaries

The current pipeline may establish that the combined workflow is executable and
that its differentiable routes, controls, artifacts, and estimands behave as
specified. It may not yet establish:

* that TGSA improves capability;
* that MTP improves TGSA or Rineya quality;
* that the interaction is positive;
* that future-token accuracy translates into accepted speculative tokens;
* that TGSA's speed cost is recovered;
* that the TinyLM result generalizes to a Rineya-family checkpoint.

Those claims require, in order, competence, powered paired evidence, a real
checkpoint retrofit, and an end-to-end speculative decoding benchmark.

# DELTA-SUBSPACE-GEOMETRY-001 — near-orthogonal deltas share a subspace

Date: 2026-09-22. Read-only analysis. **No model was loaded, no row was scored, no quality,
retention or release claim is made.** Artifacts: `SUBSPACE.json`, `CONTROLS.json`.

## Question

`SOUP-MINI-001` measured cos(dSparse, dOpallo) = 0.0025 and the peer experiment
`TITAN-OPALLO-SKILL-VECTOR-001` reproduced it. The natural reading is "these two skills use
different parts of the network, so composing them should be nearly free." This asks whether
the elementwise cosine actually licenses that reading.

It does not, and the reason is measurable.

## Why elementwise cosine is the wrong statistic

For a linear layer `W = W0 + a·dS + b·dO` the perturbations add in the forward pass, and the
expected overlap of the two contributions is

```
E[⟨dS·x, dO·x⟩] = tr(dSᵀ dO Σ),      Σ = E[x xᵀ]
```

Under **isotropic** inputs (Σ = I) this collapses to `⟨dS, dO⟩_F` — the elementwise inner
product, which is ~0 here. But transformer activations are strongly anisotropic. `tr(dSᵀ dO Σ)`
can be large while `tr(dSᵀ dO)` ≈ 0, if the two deltas align inside the high-variance input
directions and merely cancel in sign when averaged over all coordinates.

So the elementwise cosine answers the isotropic question. The real one is whether the deltas
compete for the *dominant* directions.

## Measurement

For each of the 140 trained 2-D weight matrices (blocks 8–27), using the **unrounded FP32**
deltas, I measured how much of one delta's energy lies inside the other's top-32 singular
subspace, normalized by the chance value `k/dim`:

```
collision_in = ‖dS · V_dO‖²_F / ‖dS‖²_F  ÷  (k / n_in)
```

`1.0` = isotropic relative to the other delta (elementwise cosine tells the whole story).
`≫1` = the deltas compete for the same subspace.

Both deltas turn out to be effectively **rank ≈ 18** out of 1024–3072 dimensions.

## Result, with controls

| | median collision_in |
|---|---|
| **Same-block dSparse in dOpallo's subspace** | **3.15** |
| Cross-block (same sublayer kind, different depth) | 1.38 |
| dSparse in W0's own top-32 subspace | 1.35 |
| Haar-random subspace (null) | **0.99** |

The Haar null lands at 0.99, so the estimator is calibrated and chance really is 1.0. The two
rival explanations both fail:

- **"Both deltas just align with the base model."** Refuted — alignment with W0 is 1.35, barely
  above chance.
- **"Any two deltas of this shape collide this much."** Refuted — the cross-block control pairs
  real deltas of identical shape, rank, scale and training recipe, differing *only* in depth,
  and gives 1.38.

The collision is **specific to the same layer**, and it is structured by sublayer:

| sublayer | same-block | cross-block | ratio |
|---|---|---|---|
| `self_attn.o_proj` | **7.38** | 1.28 | **5.75×** |
| `mlp.down_proj` | **5.93** | 1.00 | **5.94×** |
| `self_attn.v_proj` | 4.32 | 1.85 | 2.33× |
| `mlp.gate_proj` | 3.15 | 1.41 | 2.23× |
| `self_attn.k_proj` | 3.11 | 1.56 | 1.99× |
| `mlp.up_proj` | 2.94 | 1.38 | 2.14× |
| `self_attn.q_proj` | 2.82 | 1.37 | 2.07× |

`o_proj` and `down_proj` are exactly the two matrices that **write back into the residual
stream**. The matrices that *read* from it (q, k, up, gate) collide at roughly half that rate.

> The two skills read somewhat different features but route them through **the same write-out
> directions at the same depth**. The machinery is not disjoint.

Meanwhile `collision_out` (median 1.45) is far below `collision_in` (3.55), so the overlap is
concentrated on the input side of those write matrices.

## The follow-up test — and it refuted my own hypothesis

Sharing a subspace is not the same as contending inside it. If the near-zero cosine were sign
cancellation that anisotropy would undo, then restricting the inner product to the dominant
directions should make the alignment jump. I measured
`cos_eff(V) = ⟨dS·V, dO·V⟩ / (‖dS·V‖‖dO·V‖)` for four candidate subspaces (`EFFECTIVE_COSINE.json`):

| subspace V (top-32 right singular) | median \|cos\| | mean \|cos\| | max \|cos\| | frac > 0.05 |
|---|---|---|---|---|
| global (no restriction) | 0.0031 | 0.0043 | 0.0172 | 0.00 |
| Haar random | 0.0051 | 0.0065 | 0.0340 | 0.00 |
| W0's own top-32 | 0.0063 | 0.0090 | 0.0444 | 0.00 |
| dSparse's top-32 | 0.0098 | 0.0133 | 0.0614 | 0.01 |
| **dOpallo's top-32** | **0.0124** | 0.0156 | 0.0628 | 0.02 |

Restricting to the most contended directions multiplies the alignment by about **4×** — and
leaves it at **0.012**. The worst single matrix reaches 0.063; 98% stay under 0.05.

**So the sign-cancellation hypothesis is refuted.** The deltas share a subspace, but *within*
that subspace they remain very nearly orthogonal. The direction of inference that breaks is the
alarming one: I argued the near-zero global cosine might be an artifact that realistic
anisotropy would expose, and it is not. A 4× amplification off a 0.003 base is still negligible.

What survives from the collision finding is narrower but still real: **the two deltas are not
disjoint machinery** — they occupy overlapping rank-~18 subspaces at the same depth, concentrated
in the residual-stream write matrices. What does *not* survive is any claim that this implies
meaningful functional interference.

**Consequence for mechanism design:** this kills the obvious intervention. Projecting dOpallo off
dSparse's top-32 subspace in `o_proj`/`down_proj` would cost roughly 12% of dOpallo's energy
there to remove a cross term already at |cos| ≈ 0.012. That is a bad trade, and it should not be
built. The analysis is worth more for the arm it prevents than for one it suggests.

## What this does and does not establish

**Does:** two things, one positive and one negative.
1. The premise "near-zero cosine ⇒ disjoint machinery" is **false** for this pair, with three
   controls excluding the obvious rivals. The deltas overlap in the same low-rank subspaces at
   the same depth, most in the residual-stream write matrices.
2. Nevertheless the alignment *inside* those shared subspaces stays at ~0.012, so the geometric
   interference term is small **and robust to a 4× anisotropy stress test**. On the interference
   axis specifically, the near-orthogonality holds up better than I initially argued.

**Does not:** establish that composing these donors is cheap. Interference is only one of three
cost terms, and the measured evidence says it is the smallest one — see the attenuation figures
below. This is also structural only: no activations, no true Σ, no forward pass. The four
candidate subspaces are proxies, and a real Σ could still weight directions none of them cover.

## Why "orthogonal ⇒ nearly free composition" still does not follow

Three cost terms, only one of which orthogonality touches:

| term | measured size | does orthogonality help? |
|---|---|---|
| **Attenuation** — each donor at half strength in a 0.5/0.5 soup | 235 → 223 = **12 rows** | **No.** Property of the coefficient, not the angle. |
| **Cross-skill interference** | 223 → 219 = **4 rows**, paired p = .45 | Yes — and this analysis says it is genuinely small. |
| **Donor's own retention damage** | Opallo long-context strict 161 → 135/256 | **No.** Imported at any coefficient. |

The strength-matched decomposition is from `ATTENUATION_CORRECTION_20260918`. The term
orthogonality governs is the one that was already nonsignificant; the dominant term is
attenuation, which a 0.5/0.5 average pays by construction. That is the concrete case for
independent full-strength coefficients over an average — not a new mechanism, but the one the
registered soup gives away.

## Supporting measurements made along the way

Both source reconstructions were independently verified before use:

- `int001-20260917/trained_fp32.safetensors` is **absolute unrounded FP32 Titan weights** on the
  220 trained tensors: `bf16(X)` reproduces published Titan on 220/220 tensors, while the
  "it is a delta" hypothesis fails 220/220 — a discriminating control.
- The peer's Opallo reconstruction is **sound**: `bf16(trainable[mapped])` reproduces published
  Opallo on 220/220 tensors and 0 elements differ; all 90 untrained tensors equal Supa exactly.
  The `encoder.*` → `model.layers.*` remap has two shape-degenerate pairs
  (`input_norm`↔`post_norm`, `q_norm`↔`k_norm`) where a swap would be silent; both swaps were
  tested and each produces 40/220 mismatches, so the check genuinely discriminates.

Unrounded vs stored norms, and the resulting quantization noise:

| | unrounded FP32 | published BF16 | quantization noise ‖q‖ |
|---|---|---|---|
| Titan delta | 0.829831 | 0.803470 | 0.542661 (**65.4%** of the delta) |
| Opallo delta | 1.356960 | 1.403826 | 0.688213 (**50.7%** of the delta) |

cos(q_Titan, q_Opallo) = 0.0013, i.e. the two storage noises are independent. **Near-orthogonality
is not a quantization artifact**: the clean cos(dTitan, dOpallo) = 0.003044 versus 0.002486 for
the BF16 pair, and cos(dSparse, dOpallo_unrounded) = 0.002547.

Note the tension worth keeping: BF16 storage injects noise at 50–65% of the learned update's
norm, yet recorded acquisition moved only 235 → 234. Attenuating a delta to half strength — a
*smaller* norm change, but a structured one — cost 12 rows. **Frobenius norm is a poor predictor
of functional effect; direction is doing the work.** Do not read the 65% as a defect in the
published artifacts.

Titan-Sparse is confirmed to be an exact mask of the unrounded Titan delta: 99.87% of its
non-zeros are bitwise-equal to `dTitan_unrounded`, and cos(dSparse, dTitan) = 0.836382 equals
‖dSparse‖/‖dTitan‖ exactly — the algebraic signature of a projection. The top-25% of elements
carries 83.6% of the energy.

## Reproduction

```bash
python subspace.py --k 32 --output SUBSPACE.json
python controls.py
```

# ROTATION-001 — does the residual scale with the rotation angle?

**Registered before the load-bearing quantity is measured.** Read the disclosure in §2 first:
part of this is a LOOK and is labelled as one; the registered prediction is about a cosine that
has **not** been computed.

---

## 1. The question

DIRCTL-001 asks whether the direction residual is real. **This asks whether it is orderly.**

If "direction matters at fixed magnitude" is a real property rather than a quirk of one
intervention, then **how far** an update is rotated away from the collinear ray should predict
**how much** unexercised skill it spares. Three off-ray arms are available across two lanes:

| lane | off-ray arm | produced by | cos to its lane's ray | residual |
|---|---|---|---|---|
| LWF | λ = 0.25 | anchor loss during training | 0.5243 | +5.5 pp |
| LWF | λ = 0.5 | anchor loss during training | 0.3486 | +8.9 pp |
| **RRP-015** | **k = 0.25** | **magnitude masking, post-hoc** | **NOT MEASURED** | ~+2.7 pp |

**Two different mechanisms produce the rotation** — a training-time loss term and a post-hoc
per-tensor mask. If both land on one rotation-residual relationship, the effect belongs to the
geometry and not to either mechanism.

## 2. Disclosure — what is a LOOK here and what is not

**LOOK.** The k = 0.25 residual is already derivable from published tables and I derived it
before writing this. Its alias score is **0.650** (`gen-D`, 234/360) and its norm fraction is
**0.836** of the full delta (`delta-compression-is-magnitude` receipt). Interpolating the
RRP-015 α ladder (α = 0.75 → 0.633, α = 1.00 → 0.603) at 0.836 gives ≈ 0.623, hence ≈ **+2.7 pp**.
**That number is not evidence produced by this experiment and is not treated as such.**

**NOT measured, and this is the registered quantity: `cos(δ_full, δ_k025)`.** The programme has
measured this delta's norms under every parameterisation and has never once measured the *angle*
between a compressed arm and its parent. Nothing in any receipt records it.

## 3. Registered prediction

> **k = 0.25's cosine to the full delta will be HIGH — above 0.90 — consistent with its small
> residual, and the three off-ray arms will order the same way on rotation as they do on
> residual.**

Stated as an ordering, since three points across two lanes cannot support a fitted slope:

    rotation (1 - cos):   k0.25  <  lambda0.25  <  lambda0.5
    residual (pp):        +2.7   <  +5.5        <  +8.9

## 4. What would disconfirm

- **k = 0.25's cosine comes back LOW (< 0.7).** Then a heavily rotated arm has a small residual,
  the ordering breaks, and rotation does **not** predict sparing. The direction effect would
  then be a property of *how* the rotation was produced — anchoring specifically — and the
  geometric reading is wrong. **This is the outcome I consider most informative and I am not
  expecting it, which is exactly why it is written down.**
- **Cosine high but the LWF ordering inverts under DIRCTL-001's measured residuals.** Then the
  ordering claim fails on its own lane and nothing cross-lane should be attempted.

## 5. Caveats binding on any reading

1. **Cross-lane comparison is not licensed for levels.** RRP-015 and LWF are different training
   runs reaching different acquisition (235 vs 198). Each lane's residual is computed against
   **its own** collinear ray, so the residuals are internally valid; the *ordering* across lanes
   is the only cross-lane claim made, and no slope is fitted.
2. **n = 3.** No p-value is computed for the ordering and none should be. This is a shape check
   that either holds or does not.
3. **The k = 0.25 residual rests on an interpolation** of the α ladder between 0.75 and 1.00 —
   the same weakness DIRCTL-001 exists to remove in the LWF lane, and it is **not** removed here.
   If the ordering holds, the honest next step is a norm-matched collinear control at 0.836 in
   the RRP-015 lane, which is one arm and no training.
4. Rotation and magnitude are **not** independent across these arms; each off-ray arm also has a
   different norm. The residual already conditions on norm by construction, so this is handled —
   but it is why the collinear calibration has to be per-lane and cannot be shared.
5. MECHANISM SCOUT. No seal, no read budget. `quality_improvement_established` false,
   `release_permitted` false.

## 6. Cost

One cosine pass over two models on CPU (`analysis/delta_norms.py`, streamed, ~2.2 GiB peak).
**Zero model calls, zero training, no new arms.** Runs alongside DIRCTL-001's generation.

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

---

# ADDENDUM 1 — ROTATION-002, registered before the arm is built

**Written after ROTATION-001's cosine was read and before ROTATION-002's arm exists.** The
prereg's own §5 already committed to this follow-up conditionally; this fixes its predictions.

ROTATION-001 established that `k = 0.25` is a **projection** — `cos ≡ norm ratio` exactly — so it
sits at cosine **0.836382** from the collinear ray at its own norm. Its +2.7 pp residual is
still measured against an **interpolation** of the α ladder between α = 0.75 (0.633) and
α = 1.00 (0.603). One arm removes that.

    ROTATION-002:  theta_supa + 0.836382 * (theta_trained - theta_supa)

Built by INT-001's own hash-bound `build_arm.py`, scored on the identical 360 rendered ALIAS-001
prompts with the validated harness, then deleted. The scalar `0.836382` is
`||d_k025|| / ||d_full||` recomputed from `ROTATION_NORMS.json`, and the achieved norm is
verified against `||d_k025|| = 0.694055` before scoring.

## What this is really testing

Not rotation. **Whether `delta-compression-is-magnitude` holds on a preservation endpoint with
adequate power.** That conclusion — mask, rank and scalar are one lever acting through magnitude
— was established on acquisition plus preservation endpoints of n = 47, where a 2.7 pp effect is
one row. This bank is n = 360, where it is about ten.

| account | `k=0.25` vs the α = 0.836382 scalar |
|---|---|
| **ONE LEVER HOLDS on preservation** | the two tie — difference inside ±2.5 pp, McNemar not significant |
| **ONE LEVER FAILS on preservation** | the **mask wins** by ~2.7 pp, significant; compression guidance needs qualifying |
| **REVERSED** | the scalar wins; no account predicts it, reported as unexplained |

**Primary: exact two-sided McNemar, mask vs scalar, on 360 paired rows. One comparison, so no
multiplicity correction** — Holm over a family of one is the identity.

**Registered direction: mask > scalar.** I am registering the direction I already expect from
the interpolation, which is weaker than a blind prediction, and I am saying so rather than
presenting it as foresight.

**Secondary, free:** this measures the RRP-015 α ladder at 0.836, inside the 0.75→1.00 gap, so
it also says how accurate that interpolation was. Note this gap is **0.25 wide** where the LWF
interpolation DIRCTL-001 is correcting spanned **0.416 → 0.832**, so the error here should be
smaller and a near-zero correction would be unsurprising rather than reassuring.

## What would disconfirm the reading

- **Tie.** Then one-lever extends to a well-powered preservation endpoint and is *strengthened*,
  and ROTATION-001's §4 flag is withdrawn in the record.
- **The interpolation turns out badly wrong** (scalar far from 0.623). Then the +2.7 pp figure
  in ROTATION-001 §1 was itself unreliable and every residual in this programme computed against
  an α-ladder interpolation needs re-deriving — which would be the most consequential outcome of
  the three and is why the interpolation error is reported whatever happens.

One delta, one bank, one draw. MECHANISM SCOUT; nothing here promotes anything.

---

# OUTCOME — recorded here; full write-up in `probe/ROTATION001_RESULT.md`

**Registered prediction: cos(δ_full, δ_k025) > 0.90. MISSED — measured 0.836382.** The
registered disconfirmer (< 0.7) did not fire either, so neither branch resolved cleanly.

**What it returned instead:** `cos` equals the norm ratio **to six decimals**, because
`<δ_full, δ_k025> = ‖δ_k025‖²` with a difference of exactly **0.000e+00**. A coordinate mask is
a **projection**, so for any mask `cos ≡ norm ratio` and **rotation and dose are the same
parameter** — which is why the three-arm ordering the prereg asked for is **not reported**.

**ROTATION-002 (Addendum 1): ONE_LEVER_FAILS_ON_PRESERVATION.** Mask 0.6500 against a
norm-matched scalar's 0.6222 — **+2.8 pp, 11 gains / 1 loss, p = 0.0063.** The α-ladder
interpolation it relied on was accurate to **0.1 pp**, so the most consequential registered
disconfirmer did not fire. Downstream: `ALIAS_RESOLUTION_INVESTIGATION.md` Amendment 6.

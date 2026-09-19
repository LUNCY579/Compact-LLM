# ROTATION-001 — the registered threshold missed, and the reason is worth more than the prediction

**Prereg `probe/ROTATION001_PREREG.md`, committed before the measurement.** Receipt
`rineya-dataset-retention-records/ROTATION_NORMS.json`. Zero model calls, zero training, no new
arms — one CPU cosine pass. MECHANISM SCOUT.

---

## 1. The registered prediction, and it is wrong

> **Registered: k = 0.25's cosine to the full delta will be HIGH — above 0.90.**

**Measured: 0.836382.** The threshold missed. It is not below the registered disconfirmer of
0.7 either, so neither branch fired cleanly, and the honest statement is that **my point
prediction was wrong and the experiment did not return the outcome I wrote down.**

The ordering *did* hold:

| arm | produced by | cos to ray | **rotation (1 − cos)** | residual |
|---|---|---:|---:|---:|
| k = 0.25 | magnitude mask | 0.8364 | **0.164** | +2.7 pp |
| λ = 0.25 | anchor loss | 0.5243 | **0.476** | +5.5 pp |
| λ = 0.5 | anchor loss | 0.3486 | **0.651** | +8.9 pp |

But **the ordering should not be banked**, for a structural reason the measurement itself
revealed, and that reason is the actual result here.

## 2. What the measurement found: k = 0.25 is a PROJECTION, not a rotation

`cos(δ_full, δ_k025) = 0.836382` and `‖δ_k025‖ / ‖δ_full‖ = 0.836382` — **identical to six
decimals.** That is an identity, not a coincidence:

    <d_full, d_k025> = 0.481712461
    ||d_k025||^2     = 0.481712461
    difference       = 0.000e+00   (exactly zero)

So `<d_full - d_k025, d_k025> = 0`: **the masked-away part is exactly orthogonal to the retained
part.** Which is what a coordinate mask must do — `δ_k025` is `δ_full` restricted to the
top-magnitude coordinates and `δ_full − δ_k025` lives on the complementary coordinates, so the
two are coordinate-disjoint and therefore orthogonal.

**Consequence, and it is general: for ANY coordinate mask, `cos ≡ norm ratio` identically.**

## 3. Why that breaks the three-point ordering

**Rotation and dose are the same parameter for a mask, and independent parameters for the
anchor.**

| arm | cos to ray | norm ratio | independent? |
|---|---:|---:|:--:|
| k = 0.25 | 0.836382 | **0.836382** | **no — identical** |
| λ = 0.25 | 0.5243 | 0.8924 | yes |
| λ = 0.5 | 0.3486 | 0.8171 | yes |

A mask cannot trace out the rotation axis, because you cannot move its angle without moving its
norm by exactly the same factor. So the k = 0.25 point is **not a third observation of the same
kind** as the two anchored points, and a monotone ordering across all three describes two
different geometries sharing an axis label.

**The k = 0.25 point remains individually valid.** Its residual is computed against a *scaled*
arm at the same norm, so it is still a genuine same-norm / different-direction comparison. What
is not valid is treating the three as one trend. **I am not reporting a rotation–residual
relationship.** The prereg asked for an ordering; the ordering exists and is uninterpretable,
which is a different and more useful answer than either registered branch.

## 4. The finding this hands to the programme, which is larger than what it was asked

`delta-compression-is-magnitude` concluded that mask, rank truncation and scalar scaling are
**one lever acting through magnitude**, on the grounds that at matched norm they are
behaviourally indistinguishable.

This measurement shows those arms are **not** directionally identical: at norm ratio 0.836 the
mask sits at cosine 0.836 from the scalar — a 33° separation — and the alias bank puts it
**+2.7 pp above** what the collinear ray predicts at that norm.

**That does not overturn the earlier conclusion, and it flags exactly where it might not hold.**
The endpoints that established "one lever" were acquisition (where the arms did match) and
preservation endpoints with far less power than this one — `references_retained` is n = 47 where
the alias bank is n = 360. A ~2.7 pp effect on 47 rows is one row and invisible; on 360 rows it
is about ten and measurable.

**So the honest statement is narrower than the memory currently records: mask/rank/scalar are
one lever *for acquisition*, and whether that holds *for preservation* has never been tested at
adequate power.**

## 5. The follow-up, already registered in the prereg and now clearly required

> "If the ordering holds, the honest next step is a norm-matched collinear control at 0.836 in
> the RRP-015 lane, which is one arm and no training."

The +2.7 pp still rests on an **interpolation** of the α ladder between 0.75 (0.633) and 1.00
(0.603) — precisely the weakness DIRCTL-001 exists to remove in the LWF lane and which is **not**
removed here. The fix is identical and costs one arm:

    ROTATION-002:  theta_supa + 0.836382 * (theta_trained - theta_supa)

scored on the same 360 prompts, giving a **direct paired McNemar between the mask and a scalar
at identical norm**. If the mask wins there, "one lever" is false on preservation and the
programme's compression guidance needs qualifying. If it ties, the earlier conclusion extends to
a well-powered preservation endpoint and is strengthened.

## 6. Scope

One delta, one bank, one draw. The cosine identity in §2 is exact arithmetic and holds for any
coordinate mask; everything else here is a single measurement.
`quality_improvement_established` false, `release_permitted` false.

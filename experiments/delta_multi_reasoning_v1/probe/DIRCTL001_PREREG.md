# DIRCTL-001 — turning the direction result into a direct measurement

**Written and committed BEFORE either arm is built.** No arm exists, no prediction file exists.

Amendment 4 of `ALIAS_RESOLUTION_INVESTIGATION.md` reported that the anchored arms preserve alias
binding **beyond what their magnitude explains** — +5.5 pp at λ = 0.25 and +8.9 pp at λ = 0.5 —
and called it the first result in this programme that moves the frontier by choosing **direction**
rather than **dose**.

**That claim currently rests on an interpolation, and this experiment exists to remove it.**

---

## The weakness in my own result

The residual is measured against a **piecewise-linear interpolation** of a three-point ray:

| calibration point | ‖δ‖ | alias |
|---|---:|---:|
| Supa | 0.000000 | 0.750 |
| own-α0.5 | 0.416144 | 0.658 |
| λ = 0 | 0.832289 | 0.553 |

The anchored arms sit at **0.742705** and **0.680032** — both inside the wide gap between 0.4161
and 0.8323, where the curve is assumed straight and has never been measured. **If the true
magnitude curve is convex there, the residual is overstated and the headline is wrong.**

`probe/LWF_ALIAS_PREREG.md` caveat 1 already said a two-point interpolation "cannot detect
curvature, so a residual near zero is weaker evidence than a residual clearly away from zero."
The residual came back clearly away from zero, which makes the curvature question load-bearing
rather than academic.

## The design — same direction, identical norm, no curve fitting

Build two scalar arms on **λ = 0's own direction**, each scaled to land **exactly** on an
anchored arm's norm:

    theta(a) = theta_supa + a * (theta_lambda0 - theta_supa)

| control | matches | target ‖δ‖ | scalar `a` |
|---|---|---:|---:|
| `sc-m025` | λ = 0.25 | 0.742705 | **0.892364** |
| `sc-m05` | λ = 0.5 | 0.680032 | **0.817062** |

`a` is exact because the ray is linear in `a` by construction: ‖δ(a)‖ = a·‖δ(λ=0)‖. The
constants above are recomputed from `DELTA_NORMS.json` in the driver rather than typed, and the
build **verifies** the achieved norm before scoring; a mismatch beyond 1e-6 aborts.

Built with the peer's own `lwf001-20260918/scale_own_delta.py`, the same script that produced
`ctrl-own-alpha0.5`, so the construction is identical to the existing calibration point.

Scored on the **identical 360 rendered ALIAS-001 prompts** with `probe/generate_alias.py`, the
harness validated by reproducing the frozen source's existing 64-row chunk byte-for-byte, 64/64.

**The comparison becomes a direct paired McNemar between two real arms at the same norm on the
same direction. No interpolation appears anywhere in it.**

## Registered predictions

| account | `sc-m025` vs λ=0.25 | `sc-m05` vs λ=0.5 |
|---|---|---|
| **DIRECTION IS REAL** (Amendment 4 as written) | scalar **below** anchored by ~5.5 pp, significant | scalar **below** by ~8.9 pp, significant |
| **INTERPOLATION ARTIFACT** | scalar within ±2.5 pp of anchored | scalar within ±2.5 pp of anchored |

**Primary: exact two-sided McNemar on each pair, Holm-corrected over the family of two.**
Registered direction: anchored > scalar. Two comparisons, so Holm is not the identity here and
is applied rather than gestured at.

**Point predictions, stated so they can miss:** `sc-m025` ≈ 0.576 and `sc-m05` ≈ 0.591, the
interpolated values. I expect these to be wrong in level — the α-ladder taught me that absolute
levels move for mundane reasons — and the verdict does not rest on them. It rests on the paired
contrast, which is what `probe/LWF_ALIAS_PREREG.md` should have registered the first time.

### Secondary 1 — curvature, a free read

The two new arms give a **five-point collinear ladder** in the LWF lane at ‖δ‖ =
0.000 / 0.416 / 0.680 / 0.743 / 0.832. That is the first direct measurement of the magnitude
curve's shape in this lane, and it puts an error bar on any residual computed here in future.
Reported as convex / linear / concave against the chord, with no verdict attached.

### Secondary 2 — a replication of ALIAS-002 on a different ray

ALIAS-002 found a perfect monotone dose response (ρ = −1.000, exact permutation p = 0.0167) on
the **RRP-015** ray. Those same five collinear points on the **LWF** ray test whether that
replicates on an independently trained delta. **Registered: negative and monotone.** At n = 5
the exact permutation floor is 0.0167 again.

## What would disconfirm, and what I do about it

- **Scalars land on the anchored arms** → the residual was an interpolation artifact.
  **Amendment 4's headline is retracted**, in the record, in those words.
- **Scalars land *below* the interpolation** → the curve is convex, the residual was
  *understated*, and the direction effect is larger than reported.
- **Non-monotone alias across the five collinear points** → the dose response does not replicate
  on this ray, which would be a bigger problem than anything about direction and would be
  reported as such rather than absorbed.

## Caveats stated before the numbers

1. **The scalar arms are post-hoc edits of a trained delta; the anchored arms were trained under
   a different objective.** They therefore differ in more than direction. I do not think this
   breaks the comparison — the question is only whether a same-direction update at the same
   distance damages alias as much, and provenance is not in that question — but it is the seam,
   it is named here, and the peer session has been asked to attack it before the GPU is spent.
2. **This does not resolve the confound registered in `LWF_ALIAS_PREREG.md` Addendum 2.** Even if
   the residual survives exactly, it still cannot separate *the anchor protects* from *that
   direction damages less*. It establishes only that the effect is not an artifact of
   interpolation. The direction control that would separate them needs a training run.
3. **Absolute rates in the `high` salience condition remain unreadable** by construction; paired
   contrasts are readable in every condition.
4. One training draw, one seed, one bank. MECHANISM SCOUT — no seal, no read budget.
   `quality_improvement_established` false, `release_permitted` false.

## Resource discipline

Two arms at ~2.27 GB on 40 GB free. Both are **deleted after scoring**, as the α-ladder arms
were; the predictions and the build receipts are the evidence and they are kept.

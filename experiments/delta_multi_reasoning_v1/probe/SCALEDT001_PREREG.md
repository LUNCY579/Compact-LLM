# SCALED-T-001 — is H's skill free at matched magnitude?

> **STATUS: REGISTERED, NOT BUILT.** No arm exists and nothing has been scored as of this line.
> Registered before the build because the arm's whole purpose is to adjudicate a number
> (TH16s = 219) that is already known, and a control designed after its target is not a control.

**Class: MECHANISM SCOUT.** Zero training. Pure weight arithmetic over deltas reconstructed from
files that survived reclamation. Promotes nothing.

## The question

`TH16s` scores **219** against `T16`'s **235**. I nearly attributed that −16 to skill
interference. It is not obviously that, because the rescale is applied to the **sum**:

    TH16s = s_t * ( P16(delta_T,t) + P16(delta_H,t) )
    s_t   = ||P16(delta_T,t)|| / ||P16(delta_T,t) + P16(delta_H,t)||      per 2D tensor
    s_t over 220 tensors: min 0.5175, median 0.6750, max 0.7496, 0 fallbacks

**So T's own component inside TH16s sits at `s_t` of its truncated strength. TH16s is not "full
T plus attenuated H" — it is 0.675 × (T + H).** Comparing it to T16 at full strength compares
two different T magnitudes and charges the difference to H.

An interpolation against INT-001's α ladder predicts T alone at that attenuation would score
**~221.9** against TH16s's observed **219** — a residual of −2.9, inside the 7-row floor. That
is suggestive and it is **an interpolation, not a measurement**. The ray band across the full
`s_t` range is **195–226**, which is wide enough that 219 sitting mid-band settles nothing.

> **This experiment replaces that interpolation with a built arm and a PAIRED test.**

## Construction

    SCALED_T = theta_supa + s_t * P16(delta_T,t)          per 2D tensor, FP32

Identical to TH16s in every respect **except that δ_H's directions are absent**. Same `s_t`, same
truncation rank, same source, same per-tensor treatment, same FP32 path, 1D norm deltas handled
as COMPOSE-001 handled them.

**`s_t` is READ from `mag001-20260918/arm-TH16s/MATERIALIZATION.json` → `per_tensor_scales`, not
recomputed.** Recomputation is how `arm-rebuild-s10` diverged from `arm-s10` at the ninth
decimal; reading the recorded vector removes that source entirely and guarantees the two arms
share the scale exactly.

**Ingredients, all verified present:**

| file | MB | role |
|---|---:|---|
| `Rineya-Supa-1.7.0-devfreeze/model.safetensors` | 1137 | source θ |
| `int001-20260917/trained_fp32.safetensors` | 1201 | δ_T = trained − source |
| `compose001-20260918/trained_H_fp32.safetensors` | 1201 | δ_H (not needed here; listed for the sibling control) |

## BUILD GATE — the canary, and it must pass before any arm is written

`per_tensor_scales` records **`norm_T` per tensor = ‖P₁₆(δ_T,t)‖** as TH16s computed it. My
recomputed truncation must reproduce all **220** of those values.

> **Refuse to write the arm unless every tensor's recomputed ‖P₁₆(δ_T,t)‖ matches the recorded
> `norm_T` to relative 1e-6.** If it does not, my SVD truncation is not theirs, the control is
> not matched, and the comparison is void — report the divergence and stop.

This is the canary rule applied at build time: **compute a quantity whose answer is already
recorded, in the same pass.** Without it a silently different truncation would produce a
plausible number and no way to notice.

## Primary endpoint — a PAIRED contrast, not a ray

**`SCALED_T` against `TH16s`**, acquisition on `select-views-001`
(`views_sha256 7dd53f2a…`), exact McNemar on the per-view `canonical` flags.

Both arms are **deterministic transforms of the same trained deltas**, so this is a
**WITHIN-draw** contrast, valid at its own discordance. The ~20-row cross-draw floor does **not**
apply and must not be quoted against it.

## Registered predictions

| # | prediction | what it would mean |
|---|---|---|
| **P1** | `SCALED_T − TH16s` net **≤ 7** in absolute value, McNemar **p > 0.05** | **H's skill is FREE at matched magnitude.** The whole 60-row composition cost is magnitude; composition buys a second skill for the shrinkage you would pay anyway |
| **P2** | `SCALED_T − TH16s` net **≥ +10** at **p ≤ 0.05** | **The attenuation account is WRONG.** H's presence costs acquisition at matched magnitude and the −16 is real interference |
| **P3** *(secondary)* | `SCALED_T` lands in **212–226** | reproduces the interpolated ray's central estimate; a miss does **not** falsify P1/P2, it falsifies the ray |

**Between +8 and +9, or a net ≥ 10 at p > 0.05, is INDETERMINATE** and reports as *"not powered
to distinguish"*. Registered now so it cannot be resolved by preference later.

## What would make this uninterpretable, declared in advance

- **Build-gate failure** (canary mismatch) → void, report the divergence, do not score.
- **A discordance too small to adjudicate.** If `SCALED_T` and `TH16s` differ on fewer than 6
  views, no net can reach p ≤ 0.05 and the result reports as **UNCERTIFIABLE**, not as P1. This
  is the `references_retained` trap and it is registered here before it can be discovered.
- Scoring on any bank other than `7dd53f2a…`, or under any digest other than `a329eb7bb06f`.

## Scope

Acquisition only. It says nothing about long context, abstention, retention, or the boundary
instrument — and notably **nothing about whether H's skill survives**, which `J_decision` on
`HSKILL-VIEWS-002` already answers for TH16s and which this arm does not carry at all.

`process-transition-probe-001` is not used. **REPORT-ONLY; promotes nothing.**

## Cost and the current blocker

Build: elementwise FP32, streamed tensor-by-tensor, ~2.27 GiB of new disk against **20.56 GiB
free**. No GPU, no training, no lease.

**Scoring is BLOCKED on host RAM**: a probe needs ~5.2 GiB free and the host has **2.30 GiB**.
The guard is not weakened. The arm can be built and left ready.

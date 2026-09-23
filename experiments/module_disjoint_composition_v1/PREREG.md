# MODULE-DISJOINT-001 — preregistration

Written 2026-09-23, **before any arm in this experiment was built or scored.**

## The idea, and why it is not already closed

Composition has three cost terms, measured:

| term | size | governed by |
|---|---|---|
| **attenuation** | 12 rows | the coefficient |
| interference | 4 rows, p=.45 | the angle between deltas |
| donor's own damage | Opallo LC 161→135/256 | neither |

The dominant term is attenuation, and a 0.5/0.5 soup pays it by construction. Independent
coefficients (`ADD-100`) avoid it by putting both donors at 1.0 — but then both write into the
same coordinates, and `DELTA-SUBSPACE-GEOMETRY-001` showed they occupy overlapping rank-~18
subspaces at the same depth (collision 3.15× chance vs 1.38 cross-block).

**Module-disjoint composition takes a third route: give each donor its own coordinates.**

```
W = Supa + dSparse|ATTN + dOpallo|MLP
```

`ATTN` = `q/k/v/o_proj` + `input_layernorm` + `post_attention_layernorm` + `q/k_norm`;
`MLP` = `gate/up/down_proj`. The two donors then share **no parameter at all**. Both sit at
coefficient 1.0, nothing rescales the sum, and the attenuation term is not paid — each donor is
at full strength *on the coordinates it keeps*.

Measured split over the 220 trained tensors (blocks 8–27):

| class | params | share of ‖dSparse‖² | share of ‖dOpallo‖² |
|---|---|---|---|
| ATTN | 125,875,200 (40.0%) | 40.2% | 39.2% |
| MLP | 188,743,680 (60.0%) | 59.8% | 60.8% |

‖dSparse\|ATTN‖ = 0.440062 (of 0.694055). ‖dOpallo\|MLP‖ = 1.057716 (of 1.356960).

Energy share tracks parameter share to within 1pp for both donors, so this partition is
**not** selected to favour either one.

### Prior art, checked

- `MODCOEF-001` (2026-09-23) closed per-module coefficients **for the certification goal** on
  **one** delta: "Per-module coefficients do not buy certification", and found `A_mlp_zero` at
  γ=0 "remains the best… most acquisition per unit norm". That is a within-donor compression
  result. It does **not** address two donors on disjoint module sets, and its own framing calls
  the practical reading "compression, not certification".
- `DPLOC-001` (2026-09-23) measured attention at **2.00×** random function density for
  acquisition and MLP at **0.66×** (p = 5.4e-07 / 3.6e-08 on matched-norm controls). Its
  retention limb returned `UNINFORMATIVE_BY_DISCORDANCE` — a power failure, not a null.
- Neither ran a cross-donor module-disjoint arm. This is new.

## Arms

Built from the **unrounded FP32** deltas (both reconstructions independently verified 220/220).

| arm | construction |
|---|---|
| `SPARSE_ATTN` | Supa + dSparse\|ATTN — the precondition arm |
| `MODULE_DISJOINT` | Supa + dSparse\|ATTN + dOpallo\|MLP — the mechanism |

Existing arms already scored on this bank serve as comparators: `BASE` 0/192, `SPARSE` 102/192,
plus `ADD-100` and `AVG` from the run currently in flight.

## Registered predictions, before building

Scored on `FRESH-DUAL-SKILL-001` `T_BANK_V2` (sha256 `fc094136…`), transition exact match, n=192.
`SPARSE` = 102/192 is the reference.

**P1 — does attention carry the transition skill disproportionately?**
`SPARSE_ATTN` keeps 40.2% of dSparse's squared norm.

- *norm-proportional account*: ≈ 41/192
- *DPLOC 2.00× attention-density account*: ≈ 82/192 (2 × 40%)
- **Registered**: `SPARSE_ATTN` ≥ **62/192** (i.e. > 60% of SPARSE) confirms disproportionate
  attention carriage. Below 50 falsifies the transfer of DPLOC's density result to this bank and
  **kills the mechanism outright**, because the precondition fails.

**P2 — does disjointness actually buy non-interference?**

- **Registered**: |`MODULE_DISJOINT` − `SPARSE_ATTN`| ≤ **6 rows**, and McNemar p > 0.05.
  Adding Opallo on strictly disjoint coordinates should not move the transition score.
- If `MODULE_DISJOINT` falls materially below `SPARSE_ATTN`, then interference is **not**
  coordinate-mediated — it is routed through the residual stream, which is a result in itself
  and contradicts the disjoint-coordinates premise.

**P3 — the comparison that matters.**

- **Registered**: `MODULE_DISJOINT` > `ADD-100` on transition would mean disjoint allocation
  beats same-coordinate full-strength composition. `MODULE_DISJOINT` ≈ `ADD-100` means
  coordinate sharing was never the problem, and the extra machinery buys nothing.

## Limits, stated now

- **Half the claim is untestable here.** This bank measures transition only. Whether
  `dOpallo|MLP` retains Opallo's arithmetic is the other half of the mechanism and needs the
  arithmetic lane, which does not exist (see `../fresh_dual_skill_instrument_v1/ARITHMETIC_LANE.md`).
  **No joint-composition or "acquires a skill for free" claim can be made from this experiment.**
  If P1 and P2 pass, the honest statement is "the transition half is preserved"; the arithmetic
  half remains unmeasured.
- n=192, exact match. Only large effects resolve.
- Attempt 1. If an arm fails to build or the gate on the bank is later withdrawn, this is
  reported as-is; the bank is fixed and will not be regenerated for this experiment.
- No authorization is granted or implied. SOUP-MINI-001's `protocol.json` is untouched. GPU work
  takes the shared lease.

---

# Amendment 1 — a missing control, added BEFORE any arm in this experiment was scored

No arm of this experiment has been run through the model. `SPARSE_ATTN` and `MODULE_DISJOINT`
are built and bitwise-verified but unscored, and this amendment is written before that.

## The confound

`SPARSE_ATTN` has ‖Δ‖ = 0.4401 against full `SPARSE` at 0.6941 — a ratio of **0.634**. The
recorded dose result is that acquisition benefit **saturates between roughly 0.25 and 0.675 of
trained strength** while cost keeps accruing. 0.634 sits inside that plateau.

So if `SPARSE_ATTN` scores near `SPARSE`, P1 as registered cannot distinguish:

- **module account** — attention carries the transition skill disproportionately (DPLOC 2.00×), or
- **dose account** — *any* 63.4%-norm version of dSparse scores near 102, and the module
  restriction is irrelevant.

The registered P1 thresholds (≥62 confirms, <50 kills) were derived against a
*norm-proportional* baseline of ~41. That baseline is wrong: it assumes score scales with norm,
which the dose plateau says it does not. **P1 as written would have credited the module account
for an effect the dose account predicts equally well.** This is the candidate-set problem — a
discriminating test against the wrong rival.

## The control

`SCALAR_MATCHED` = Supa + α·dSparse over **all 220 tensors**, with α chosen so the total norm
equals `SPARSE_ATTN`'s exactly:

```
α = ‖dSparse|ATTN‖ / ‖dSparse‖ = 0.4400615543769171 / 0.6940550846962167 = 0.634044…
```

Same magnitude, same donor, same bank, differing **only** in whether the delta is restricted to
attention coordinates or spread across all modules at reduced amplitude.

## P1, restated

The comparison is no longer against a norm-proportional guess but against a built arm:

- **P1′ (module account)**: `SPARSE_ATTN` − `SCALAR_MATCHED` ≥ **+10 rows** with McNemar
  p < 0.05. Attention coordinates beat matched-magnitude spread.
- **P1′ null (dose account)**: |`SPARSE_ATTN` − `SCALAR_MATCHED`| ≤ 6 rows. The module
  restriction buys nothing beyond its magnitude, and the mechanism is a dose effect wearing a
  module label.
- **P1′ reversed**: `SCALAR_MATCHED` − `SPARSE_ATTN` ≥ +10 rows. Spreading beats concentrating
  at matched norm, which would contradict DPLOC's density result on this bank.

`SCALAR_MATCHED` also anchors the plateau claim itself: if it scores near `SPARSE`'s 102 despite
carrying 63.4% of the norm, that independently reproduces the dose-plateau result on a fresh
bank, which is worth having regardless of how P1′ lands.

P2 and P3 are unchanged. The Amendment-1 stopping rule of the instrument experiment does not
apply here; this is a control being added before measurement, not a bank being regenerated
after it.

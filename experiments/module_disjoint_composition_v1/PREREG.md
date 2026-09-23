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

---

# Amendment 2 — a better question, registered before any arm here is scored

Still true that no arm of this experiment has been run through a model. `SPARSE_ATTN`,
`MODULE_DISJOINT` and `SCALAR_MATCHED` are built and bitwise-verified, unscored.

## What changed upstream

`FRESH-DUAL-SKILL-001` returned. Two things reshape this experiment:

1. **The attenuation motivation is weaker than when I registered P1–P3.** Independent full
   coefficients already cost only **+5 rows (p = 0.33, ns)** of the donated skill, against
   **+33 (p = 3.6e-08)** for the 0.5/0.5 average. `ADD-100` already solves attenuation. So
   "module-disjoint avoids the attenuation tax" is no longer a gap worth much — `ADD-100` has
   no meaningful tax to avoid.
2. **A new defect appeared that module-disjointness plausibly *does* address.** The composed
   model imports Opallo's **trailing-period answer convention**: 178/192 rows at β=1.0, 63 at
   β=0.75, 0 at β≤0.5. Titan's contract is strict, so a strict scorer reads this as a 96-row
   collapse. This is a real regression in contract compliance even though it is not capability
   loss.

## The question this experiment can now answer, that nothing else can

**Where does the output convention live — attention or MLP?**

`MODULE_DISJOINT` takes Opallo's contribution **only from MLP** and Titan-Sparse's **only from
attention**. So it is a clean ablation of the convention's locus:

- If the trailing-period convention is carried by **Opallo's MLP** delta → `MODULE_DISJOINT`
  emits trailing periods.
- If it is carried by **Opallo's attention** delta → `MODULE_DISJOINT` emits none, because
  Opallo's attention is excluded entirely.

This is not answerable from any arm scored so far: every existing arm takes Opallo whole.

## Registered predictions, before scoring

Primary endpoint is **trailing-period emission count out of 192**, alongside strict and
period-tolerant scores. `ADD-O75` (63 periods) and `ADD-100` (178) are the comparators;
`SPARSE` (0) and `AVG` (0) are the clean references.

**P4 — locus of the convention.**
- **MLP-carried**: `MODULE_DISJOINT` emits **≥ 40** trailing periods.
- **Attention-carried**: `MODULE_DISJOINT` emits **≤ 10**.
- Between 10 and 40 is **distributed**, and the arm cannot localise it.

**P5 — built-in attribution control.** `SPARSE_ATTN` contains **no Opallo at all** and must emit
**0** trailing periods. If it emits any, the convention is not coming from Opallo and P4 is
uninterpretable. This control can fail and is the reason to trust P4 if it does not.

**P1′ (unchanged from Amendment 1)** — `SPARSE_ATTN` vs `SCALAR_MATCHED` at matched norm
(0.4401, matched to 1.5e-08) decides module-account vs dose-account for the transition skill:
+10 rows with p < 0.05 confirms the module account, |diff| ≤ 6 says the module restriction is a
dose effect wearing a module label.

**P2/P3 restated against period-tolerant scores**, since strict scores on any arm containing
Opallo now measure convention transfer rather than skill:
- P2: |`MODULE_DISJOINT` − `SPARSE_ATTN`| ≤ 6 rows, period-tolerant.
- P3: `MODULE_DISJOINT` vs `ADD-100` (97 period-tolerant) decides whether coordinate disjointness
  buys anything over same-coordinate full-strength composition.

## Stated plainly

P1–P3 were registered against a motivation that has since weakened. I am not deleting them —
they are answered as written — but the honest headline of this experiment is now **P4**, and P4
was formulated after seeing the upstream result. It is registered before this experiment's
measurement, not before all measurement, and that distinction belongs in any write-up.

---

# Amendment 3 — P5 is an absence check, and its positive control already exists

Written after `SPARSE_ATTN` scored but **before** `SCALAR_MATCHED` and `MODULE_DISJOINT` were
read, so P4 and P1′ are still unread at the time of writing.

A peer session (DOSELORA-001) pointed out a real weakness in P5 as registered. P5 says
`SPARSE_ATTN` contains no Opallo and must therefore emit **0** trailing periods. That is an
**absence check**, and an absence check returns the passing answer when the detector is broken —
a zero is equally consistent with "no convention transfer" and "my period counter does not work".

The fix costs nothing, because the positive control is already measured on the *same detector*:

| arm | trailing periods | role |
|---|---|---|
| `ADD-100` | **178/192** | **positive control — the detector fires** |
| `ADD-O75` | 63/192 | detector is graded, not binary |
| `SPARSE`, `AVG` | 0/192 | detector does not fire without a full-strength donor |
| `SPARSE_ATTN` | *(to be read)* | the absence check |

So P5's zero is interpretable only *because* the same counter, on the same bank, with the same
scoring path, returned 178 on ADD-100 and 63 on ADD-O75. Stated as a rule:

> **An absence check is interpretable only when the same detector has been shown to fire on
> achievable data.** Here that evidence was already in hand and merely needed naming.

This does not change P5's threshold or any prediction. It records why the zero, if it comes,
should be believed — and that if `ADD-100` had *not* shown 178, P5 would have been uninterpretable
regardless of what `SPARSE_ATTN` returned.

## One thing already visible and worth flagging before the rest is read

`SPARSE_ATTN` scored 68/192 with `copied_input_verbatim` = **22**, against `SPARSE`'s 4. Restricting
the delta to attention coordinates roughly quintuples verbatim input-copying. That is a distinct
failure mode from the punctuation convention and was not predicted by anything registered here.
Whether it is a property of the module restriction or of the reduced magnitude is exactly what
`SCALAR_MATCHED` (same norm, spread across all modules) will separate.

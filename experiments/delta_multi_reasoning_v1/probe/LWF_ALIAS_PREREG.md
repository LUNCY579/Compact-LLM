# LWF-ALIAS — does the anchor protect an unexercised skill, or is it just a norm knob?

**Written and committed BEFORE any prediction file is read.** The four generation jobs were
launched first and are running as this is written; that is deliberate and is the same order used
for ALIAS-002. Generation is answer-neutral. Scoring is not, and no scoring has happened.

This is a **joint experiment with the peer session** and neither side can run it alone: the three
anchored arms are theirs (`lwf001-20260918/`, same lane, seed 17, 28 update groups byte-identical
to `control-PLAN.json`, same teacher cache, `chunk_tokens` 64, betas 0.9/0.95), and the only
instrument that measures role→person binding is ours. **Gate v2 has no binding endpoint.**

**MECHANISM SCOUT.** No seal, no read budget, openly re-runnable. Nothing here licenses
promotion; `quality_improvement_established` false, `release_permitted` false.

---

## Why this is worth running even though LWF-001 is falsified

The peer session has already falsified LWF-001 on its own terms, and reported it plainly:

| arm | acquisition | retained /47 |
|---|---:|---:|
| λ = 0 | 198 | 36 |
| λ = 0.25 | 197 | 36 |
| λ = 0.5 | **125** | 39 |
| **own α = 0.5** (halve λ=0's finished delta) | **208** | **40** |

A post-hoc scalar beats the anchored arm by **+83 acquisition and +1 retained**, and beats the
unanchored parent on both axes. So the anchor is not a dose knob in disguise — it is a *worse*
dose knob than multiplication.

**That makes this test more interesting, not less.** If the anchor nonetheless protects alias
binding, it is a **selective** effect surviving an otherwise-dominated intervention. If it does
not, an intervention that loses on every axis has no remaining defence.

## The standing prediction this tests

Registered by a previous session before any of this was measured:

> **The anchor helps most on skills absent from the training mixture.**

`alias_resolution` is the measured instance. RRP-015's 384-row training file contains **zero role
phrases and zero person names**, verified independently on both sides, and the regression is
replicated at **1 repair / 54 damages across three banks, p = 3.1e-15**.

## The confound this design exists to remove

ALIAS-002 established a monotone dose response — alias accuracy falls with ‖δ‖, ρ = −1.000 in
every salience condition — and `analysis/coupling_ladder.py` (a LOOK) found the damage tracks
**magnitude** rather than marginal acquisition: cost per acquisition row spans **160×** across
the α ladder, and the top rung buys **+1 row of 384** while costing **20.8% of the total alias
damage** at McNemar p = 0.0034.

**An anchor reduces ‖δ‖ as a side effect.** So "the anchored arm loses less alias" is exactly
what a pure magnitude account predicts, with no protection of anything. Any test that does not
control the norm cannot distinguish the two, and would re-derive the programme's own denominator
mistake in a new coordinate.

## The design

**Score all four arms on the identical 360 rendered ALIAS-001 prompts**, byte-identical prompts
file, same chunked harness (`probe/generate_alias.py`), same interpreter. That harness was
reconstructed for this run and **validated by reproducing the frozen source's existing 64-row
chunk byte-for-byte, 64/64**, so conditions are matched by demonstration rather than by claim.

**The within-lane norm calibration is the control that makes this readable.** `own-α0.5` is
λ=0's own delta multiplied by 0.5 — *same direction, half the norm*. Together with λ=0 it gives a
two-point magnitude curve **calibrated inside the LWF lane itself**, so the anchored arms can be
compared against their own lane rather than against RRP-015's direction.

    residual(arm) = alias_observed(arm) − alias_predicted_from_norm(arm)

where the prediction interpolates the within-lane {λ=0, own-α0.5} pair on ‖δ‖.

### Registered predictions

| account | residual for λ = 0.25 and λ = 0.5 | Spearman(λ, alias) |
|---|---|---|
| **ANCHOR PROTECTS** (the standing prediction) | **positive and increasing in λ** | positive |
| **MAGNITUDE ONLY** | **≈ 0, inside noise** | positive but fully explained by ‖δ‖ |
| **NO EFFECT** — the teacher cache never saw a role binding, so it cannot protect one | ≈ 0 | ≈ 0 |

**Primary: the residual.** A raw Spearman over three λ points cannot separate the first two
accounts and is registered as reported-only, not verdict-bearing. At n = 3 its floor p is 0.333
in any case, which is not a test.

**Secondary, and directly actionable:** `own-α0.5` vs `λ=0.5`, paired McNemar on 360 rows. The
peer has already shown own-α wins acquisition 208 vs 125 and retention 40 vs 39. If it also wins
or ties on alias, **the anchor is dominated on all three axes** and closes. If the anchor wins on
alias alone, that is the selective effect and it is worth a registered follow-up.

**Also reported:** paired McNemar λ=0 vs each arm; per-salience-condition splits; damage and
repair never netted; every subgroup rate scored over the rows the *baseline* got right.

## What would disconfirm

- **Residuals ≈ 0 with λ=0.5's alias clearly above λ=0's** — the anchor moved alias only by
  shrinking the update, i.e. it is a norm knob and the standing prediction is wrong as stated.
- **Residuals negative** — the anchor *costs* alias binding beyond its norm, which no account
  predicts and which would be reported as unexplained rather than absorbed.
- **`own-α0.5` ≥ `λ=0.5` on alias** — an otherwise-dominated intervention loses its last axis.
- **All four arms flat** — the anchor neither helps nor hurts here, and the standing prediction
  is simply not instantiated by this skill.

## The caveats, stated before the numbers

1. **Different lane, different direction.** The LWF arms are a different training run from
   RRP-015, so their δ points in a different direction. A cross-lane comparison to the α ladder
   would confound direction with magnitude. **That is exactly why the calibration is within-lane**
   — but a two-point interpolation is a straight line through two points and cannot detect
   curvature, so a residual near zero is weaker evidence than a residual clearly away from zero.
2. **Absolute rates in the `high` salience condition are not readable** — the answer is the most
   frequent person there by construction and `most_frequent_candidate_in_context` scores 1.000.
   Paired contrasts remain readable in every condition, because a deterministic answer-agnostic
   rule emits the same string for both arms and cannot create a discordant pair.
3. **One training draw per arm, one seed.** Three λ points is not a dose curve.
4. **A mechanism scout is not a confirmation.** Whatever this returns, it cannot promote an
   intervention, and the confirmation instruments remain spent.

---

# ADDENDUM 1 — this also tests the coupling LOOK, and λ = 0.25 is the discriminating cell

**Added before any prediction file was read; the four generations were still running.** It
registers a reading of the *already-registered* residual, and adds no new endpoint.

`analysis/coupling_ladder.py` found, as a LOOK, that damage tracks **magnitude** rather than
marginal acquisition — cost per acquisition row spans 160× across the α ladder. That LOOK needs
a TEST on independent arms. These arms are it, for free, because the lane varies acquisition at
roughly fixed recipe:

| arm | acquisition /384 | anchored |
|---|---:|:--:|
| λ = 0 | 198 | no |
| λ = 0.25 | **197** | yes |
| λ = 0.5 | **125** | yes |

**λ = 0.25 acquires essentially exactly what λ = 0 acquires (197 vs 198) while carrying an
active anchor. λ = 0.5 acquires 73 rows fewer.** That dissociation is what makes the three
accounts separable, and it is worth writing down that **ANCHOR PROTECTS and STRONG CAPACITY make
the same prediction at λ = 0.5** — both expect a positive residual there, for different reasons.
Reporting a positive λ=0.5 residual as support for whichever account is preferred would be
exactly the post-hoc move this programme keeps having to undo.

| account | residual at λ = 0.25 | residual at λ = 0.5 |
|---|---|---|
| **ANCHOR PROTECTS** — the anchor preserves unexercised skills | **positive** | positive |
| **STRONG CAPACITY** — damage is the price of acquisition | **≈ 0** (it acquired the same, so it spent the same) | positive (it acquired 73 rows fewer) |
| **MAGNITUDE ONLY** — damage is the price of moving the weights | ≈ 0 | **≈ 0** |

**So λ = 0.25 is the cell that separates the anchor from capacity, and λ = 0.5 is the cell that
separates capacity from magnitude.** Neither alone decides anything.

The existing `verdict_rule` in `probe/score_lwf_alias.py` is unchanged and still governs the
recorded verdict; this addendum only fixes in advance which account a given residual pattern
supports, so that it cannot be chosen afterwards.

**A power caveat stated now.** These are single arms at n = 360 with a ±2.5 pp registered band,
so a residual near zero is "not detectably different from the magnitude prediction" and is not
positive evidence of exact zero. With three λ points and one seed, nothing here can settle
capacity; it can only fail to support it, or support it.

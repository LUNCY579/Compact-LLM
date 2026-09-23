# MODULE-DISJOINT-001 — result

Date: 2026-09-23. Bank `T_BANK_V2.json` sha256 `fc094136…`, 192 rows, under the shared GPU lease.
All predictions registered in `PREREG.md` + Amendments 1–3 before the corresponding arms were read.

**No quality, retention or release claim.** One skill, one instrument, n = 192.

## Scores

| arm | construction | ‖Δ‖ | strict | period-tolerant | trailing periods | copied input |
|---|---|---|---|---|---|---|
| `SPARSE` | full dSparse | 0.6941 | 102 | 102 | 0 | 4 |
| `SCALAR_MATCHED` | 0.634·dSparse, all modules | 0.4401 | **88** | 88 | 0 | 12 |
| `MODULE_DISJOINT` | dSparse\|ATTN + dOpallo\|MLP | — | **73** | 73 | **0** | 17 |
| `SPARSE_ATTN` | dSparse\|ATTN | 0.4401 | **68** | 68 | **0** | 22 |
| `ADD-100` | dSparse + dOpallo | 1.524 | 6 | **97** | **178** | 1 |
| `ADD-O75` | dSparse + 0.75·dOpallo | 1.232 | 63 | 98 | 63 | 4 |
| `AVG` | 0.5/0.5 | 0.762 | 69 | 69 | 0 | 15 |

## P4 — the output convention is carried by ATTENTION, not MLP

`MODULE_DISJOINT` takes Opallo's contribution **only from MLP**, at coefficient 1.0 — that is
‖dOpallo\|MLP‖ = 1.058, i.e. 78% of the full Opallo delta's norm and 60.8% of its energy.

**It emits 0 trailing periods out of 192.**

Registered thresholds were ≥40 for MLP-carried, ≤10 for attention-carried. The result is 0.

| | periods | role |
|---|---|---|
| `ADD-100` (Opallo whole, β=1.0) | **178** | positive control — the detector fires |
| `ADD-O75` (Opallo whole, β=0.75) | 63 | detector is graded, not binary |
| `MODULE_DISJOINT` (Opallo MLP only, β=1.0) | **0** | the test |
| `SPARSE_ATTN` (no Opallo at all) | **0** | P5 absence control |

Both controls hold. The detector fires at 178 on the same bank, same scoring path, and returns 0
when Opallo's attention block is withheld while 78% of its norm is still present.

> **The trailing-period answer convention lives in Opallo's attention delta.** Sixty percent of
> the donor's energy can be composed at full strength without importing its output format.

This is a mechanistic localisation of a *surface* property, and it is actionable: an arm that
takes Opallo whole but attenuates only its attention block should keep most of the donor while
dropping the convention. Not built here.

## P1′ — REVERSED. At matched norm, spreading beats concentrating.

| | period-tolerant | | |
|---|---|---|---|
| `SPARSE_ATTN` → `SCALAR_MATCHED` | 68 → **88** | net **+20** (lost 2, gained 22) | **p = 3.6e-05** |

Registered outcomes were: module account ≥ +10 for `SPARSE_ATTN`; dose account \|diff\| ≤ 6;
**reversed** ≥ +10 for `SCALAR_MATCHED`, "which would contradict DPLOC's density result on this
bank." The reversal fired, significantly.

**This does not say `DPLOC-001` is wrong.** It measured attention at 2.00× function density with
matched-norm random controls at p < 1e-6 on its own bank, for acquisition. What fails is the
*generalisation* "attention is function-dense". Same base, same donor family, different skill and
contract, opposite ordering — so **module function-density is skill-specific.**

### The +20 is not cleanly a density statement

`copied_input_verbatim` rises monotonically with attention-restriction: `SPARSE` 4 →
`SCALAR_MATCHED` 12 → `MODULE_DISJOINT` 17 → `SPARSE_ATTN` 22. Attention-only roughly
quintuples verbatim echoing relative to the full delta.

Two readings with different consequences, and this experiment does not separate them:

- *"attention delivers less function per unit norm"* → allocate elsewhere.
- *"attention-only is a broken construction that degrades into copying"* → the comparison is
  against a damaged arm, not a fair module contrast.

The copying column belongs next to the +20, not below it. (Point due to DOSELORA-001.)

## P2 — CONFIRMED. Disjoint coordinates really do avoid interference.

| | | | |
|---|---|---|---|
| `SPARSE_ATTN` → `MODULE_DISJOINT` | 68 → 73 | net +5 (lost 6, gained 11) | p = 0.33, ns |

Adding Opallo at **full strength** on strictly disjoint coordinates moves the transition score by
5 rows, not significant, and in the *positive* direction. The mechanism does what it was designed
to do.

## P3 — but it is dominated, so it should not be used.

| | | | |
|---|---|---|---|
| `ADD-100` → `MODULE_DISJOINT` | 97 → 73 | net **−24** (lost 32, gained 8) | p = 1.8e-04 |

Same-coordinate full-strength composition beats coordinate-disjoint composition by 24 rows,
period-tolerant. Disjointness buys clean formatting and costs 24 rows of skill.

**The honest recommendation is `ADD-100` plus a one-character post-process**, not
`MODULE_DISJOINT` — unless the output convention cannot be repaired downstream, in which case
`MODULE_DISJOINT` trades 24 rows for a clean contract.

## Synthesis

The mechanism works and is still the wrong choice. P2 confirms disjoint coordinates eliminate
interference; P4 shows they also eliminate the convention transfer. But P1′ shows the precondition
is bad — attention is a weak carrier of *this* skill — so the arm starts from 68 instead of 102
and never recovers. **A mechanism can be sound and dominated at the same time**, and only the
matched-norm control distinguished that from "the mechanism failed".

> **SUPERSEDED BY P6 — read the P6 section below.** "Attention is a weak carrier of this skill"
> is the wrong reading. `SPARSE_MLP` carries 50% more energy than `SPARSE_ATTN` and scores one
> row *lower* (67 vs 68), and it loses to *its* matched-norm scalar by 27 rows (p = 1.1e-07)
> against attention's 20 (p = 3.6e-05). Both classes behave identically. The operative variable
> is **restriction**, not attention. The conclusion that the arm is dominated is unchanged; the
> reason given here for it is not.

Had `SCALAR_MATCHED` not been built, `SPARSE_ATTN` at 68 against a norm-proportional expectation
of ~41 would have read as confirmation of the module account. Amendment 1 added that control
after noticing the norm ratio sat inside the dose plateau; it changed the conclusion.

## Incidental, unregistered

`SCALAR_MATCHED` keeps 88/102 = **86.3% of the skill at 63.4% of the norm** — sublinear and
saturating, the same *shape* as the recorded dose curve, on a bank sharing no rows with anything
that curve was fitted on. Not a replication: no pre-declared threshold, and the curvature
differs. At the same norm fraction the peer's dense limb sits at 98.6% and its LoRA limb at
89.9% against this arm's 86.3% — and **the peer's two limbs differ from each other more than its
LoRA limb differs from this one**, so curvature is parameterization-dependent rather than a
constant of the base. Report as "same shape, different curvature". (Comparison supplied by
DOSELORA-001.)

## What this does not establish

- Transition only. Whether `dOpallo|MLP` carries Opallo's *arithmetic* is unmeasured — the
  arithmetic lane does not exist. **No joint-composition claim.**
- P4 localises the convention to attention **for this donor pair**. Whether output conventions
  are attention-carried in general is one observation, not a law.
- P1′ and P3 are period-tolerant comparisons; the registered primary remains strict exact match.

---

# P6 — the other half, and it weakens my own P4 generalisation

Registered in Amendment 4 with thresholds before building. Built and scored inside a single
lease covering construction as well as inference.

| arm | ‖Δ‖ | energy | period-tolerant | contract-obedient | copied input |
|---|---|---|---|---|---|
| `BASE` | 0 | 0% | 0 | **56/192 (29%)** | 2 |
| `SPARSE_ATTN` | 0.4401 | 40.2% | 68 | 191/192 | 21 |
| `SCALAR_MATCHED` | 0.4401 | — | **88** | 191/192 | 12 |
| `SPARSE_MLP` | 0.5367 | 59.8% | 67 | **192/192 (100%)** | 21 |
| `SCALAR_MATCHED_MLP` | 0.5367 | — | **94** | 192/192 | 7 |
| `MODULE_DISJOINT` | — | — | 73 | 191/192 | 15 |
| `SPARSE` | 0.6941 | 100% | 102 | 192/192 | 4 |

## P6 answer: output SHAPE is not module-specific. P4 does not generalise.

`SPARSE_MLP` withholds attention entirely and scores **192/192 contract-obedience**. Registered
thresholds were ≤70% for attention-carried, ≥95% for not-module-specific. It came in at 100%.

**So "attention carries the contract" is false as stated, in my own lane.** I registered that
this outcome would weaken my P4 generalisation and it does.

What survives is narrower and more interesting, because "contract" turns out to be two things
that behave **oppositely under the same manipulation**:

| contract property | carrier | evidence |
|---|---|---|
| **output shape** (well-formed JSON object / bare integer) | **neither — module-insensitive** | either class at 40–60% energy installs it at ~100%; base is 29% |
| **punctuation convention** (trailing period) | **attention** | MLP-only donor at 60.8% energy emits **0/192** periods vs 178 for the whole donor |

P4 stands as measured; only the generalisation from it dies. A donor's *surface convention* is
attention-carried; a donor's *output shape compliance* is carried by any module class and is
cheap. Both would have been called "the contract".

## The skill is DISTRIBUTED — both module restrictions lose to matched-norm scalars

| contrast (period-tolerant) | | net | p |
|---|---|---|---|
| `SPARSE_ATTN` → `SCALAR_MATCHED` | 68 → 88 | **+20** | 3.6e-05 |
| `SPARSE_MLP` → `SCALAR_MATCHED_MLP` | 67 → 94 | **+27** | 1.1e-07 |

Both module-restricted arms lose heavily to a plain scalar at their *own* norm, in the same
direction, both significant. This is the secondary outcome registered in Amendment 4:

> **The transition skill is distributed. Module identity buys nothing in either direction, and
> restriction itself costs far more than the norm it removes.**

`SPARSE_MLP` carries **50% more energy** than `SPARSE_ATTN` (0.5367 vs 0.4401) and scores one
row lower (67 vs 68). Restriction to either class collapses to the same place regardless of how
much norm is kept.

## The P1′ confound is retired

P1′ could not separate *"attention delivers less function per unit norm"* from *"attention-only
is a broken construction that degrades into copying"*. The MLP arm settles it:

```
copied input:  SPARSE 4  |  SCALAR_MATCHED_MLP 7  |  SCALAR_MATCHED 12  |  SPARSE_MLP 21  |  SPARSE_ATTN 21
```

Both restricted arms sit at 21; both scalars sit well below. **Restriction per se drives the
model toward echoing its input — not attention-restriction specifically.** So the +20 on P1′ was
never an attention-vs-MLP density statement; it was restriction-vs-spreading, and P1′'s
apparent support for "attention is a poor carrier" was the wrong reading of the right number.

Corrected joint statement: **module function-density, as measured by matched-norm ablation on
this skill, is zero in both directions.** DPLOC-001's 2.00× attention advantage does not
transfer here, and neither does its mirror image.

## Dose, incidentally, again

| norm fraction | period-tolerant | % of `SPARSE` |
|---|---|---|
| 0.634 | 88 | 86.3% |
| 0.773 | 94 | 92.2% |
| 1.000 | 102 | 100% |

Sublinear and saturating on a third point. Still unregistered, still an observation.

## Standing limits

- Transition only; the arithmetic lane does not exist, so no joint-composition claim.
- P4's localisation is one donor pair and one convention. P6 shows the *category* it belongs to
  is narrower than "the contract".
- All contrasts period-tolerant; the registered primary remains strict exact match.

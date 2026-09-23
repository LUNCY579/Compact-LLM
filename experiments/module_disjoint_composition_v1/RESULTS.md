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

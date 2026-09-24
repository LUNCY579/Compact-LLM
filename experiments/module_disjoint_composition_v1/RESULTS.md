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

> **WITHDRAWN — Erratum 1 (2026-09-24).** "Different skill and contract" is false. DPLOC-001
> scored acquisition with `process_probe.py --instrument select-views-001` (DPLOC001_PREREG.md
> §4; run_dploc.py:65): the same three strict-JSON transition families as T_BANK_V2. The skill was
> held **constant**. What differs is the **control**: DPLOC used a *random per-element mask*
> removing the same squared norm; this experiment used a *uniform scalar*. See Erratum 1 below.

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
  *(Corrected 2026-09-24: Opallo trained on zero arithmetic-class rows — see
  `../fresh_dual_skill_instrument_v1/RESULTS.md` Erratum 1 — so there is no Opallo arithmetic skill
  for `dOpallo|MLP` to carry. The accurate limit is that this bank measures only whether Titan's
  transition skill survives; any transfer from Opallo must be read on an endpoint where Opallo has a
  specific, measurable effect.)*
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

> **WITHDRAWN — Erratum 1 (2026-09-24).** This experiment never measured module function-density
> in DPLOC's sense, so it cannot say density is zero, nor that DPLOC's result "does not transfer".
> What it measured is restriction versus *uniform shrinkage* at matched norm. See Erratum 1.

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


---

# Erratum 1 (2026-09-24) — the DPLOC comparison changed the wrong variable

Found by adversarial review; the reviewer's verifier died on a session limit, so the key fact was
re-derived by hand from `dploc001-20260923/DPLOC001_PREREG.md` §3–§4 and `run_dploc.py:65`.

## What was claimed

That P1′ (attention-only loses to a matched-norm scalar by +20) and DPLOC-001 (attention carries
acquisition at 2.00× random function density) disagree because they measure **different skills**,
so "module function-density is skill-specific."

## What is true

**Same skill.** DPLOC's acquisition endpoint is `select-views-001`: 384 rows, 128 each of
`ledger_replay`, `prerequisite_plan`, `register_machine`, strict JSON, 0/384 references ending in
`.`. T_BANK_V2 is the same three families under the same contract. The variable I blamed was the
one held fixed.

**Different control — and that is decisive.** The two designs match norm against different
references:

| | removes | compares module-restriction against |
|---|---|---|
| DPLOC-001 | the module's delta | a **random per-element mask** removing the same squared norm |
| this experiment | the other module's delta | a **uniform scalar** α applied to every element |

These answer different questions. Given the recorded dose plateau — benefit saturates well below
trained strength — a uniform scalar at 63–77% of norm stays near full function (here 88 and 94 of
102). **Any structured removal is expected to lose to uniform shrinkage** on a saturating curve,
whichever module it keeps. So P1′ and P6 are predicted by the plateau alone and carry no
information about *which* module carries function. DPLOC's random-mask control is the one that
isolates location.

**The two results are not in conflict.** "Attention beats random-element removal at matched norm"
(DPLOC) and "attention-only loses to uniform shrinkage at matched norm" (here) can both be true.
Other differences also remain uncontrolled — delta (DPLOC's ablation target vs dSparse), decode
budget (128 vs 48), bank (select-views-001 vs fresh T_BANK_V2) — and I had collapsed a
four-variable gap into one variable, choosing the wrong one.

## What survives, and what does not

**Survives (measurements unchanged; content-clean rows in parentheses):**
- `SPARSE_ATTN` 68 (62) vs `SCALAR_MATCHED` 88 (81) at ‖Δ‖ 0.4401
- `SPARSE_MLP` 67 (61) vs `SCALAR_MATCHED_MLP` 94 (85) at ‖Δ‖ 0.5367
- `SPARSE_MLP` ≈ `SPARSE_ATTN` despite 50% more energy
- restricted arms echo the input at 21–22 against 7–12 for scalars

**Practical reading that does follow:** *if a delta's norm must be reduced, shrink it uniformly
rather than ablating a module class* — on this skill and delta, uniform shrinkage keeps 19–24 more
rows at matched norm.

**Withdrawn:** "module function-density is skill-specific"; "density is zero in both directions";
"DPLOC's 2.00× does not transfer here"; and every downstream use of them, including the
allocation-invariance conjunction "the optimal allocation differs by skill" in the memory file.

**The test that would actually speak to location** is DPLOC's design on a fresh bank: a random
per-element mask matched to the squared norm `SPARSE_ATTN` removes, scored against `SPARSE_ATTN`.
If attention-restriction beats it, DPLOC's location effect replicates on an independent
instrument. Not run here.

Direction: an over-claimed mechanism is withdrawn; no measured number is retracted.

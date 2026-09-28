# FRESH-DUAL-SKILL-001 — result

Date: 2026-09-23. Bank `T_BANK_V2.json` sha256 `fc094136…`, 192 rows, exact-match, greedy, FP32,
collected under the shared GPU lease. Instrument admitted as **attempt 2 of 2** (see `PREREG.md`).

**No quality, retention or release claim. One skill, one instrument, n=192.**

## Headline

> **CONFIRMATION RESULT (2026-09-24, CONFIRM-001, fresh content-disjoint bank, 864 rows, read
> once; `../composition_confirmation_v1/RESULTS.md`).** Attenuation replicates and is confirmed:
> AVG loses 116/447 (26%), and ADD-100 beats AVG by +82 (C1, C2 pass). **The claim below that
> interference is "not distinguishable from zero" does NOT replicate:** under the registered
> period-tolerant rule, ADD-100 loses 56 rows and gains 22 (net −34, p = 1.5e-4). That is 3.8–11.4%
> of SPARSE's skill, and the registered ≤ 10% margin (C3) failed. Every lost row is a genuine content
> error: none is formatting residue.

> **CORRECTED 2026-09-24 (Erratum 3).** The paragraph that stood here led with a post-hoc result.
> Read in registration order:

**On the registered primary endpoint (strict exact match), the registered prediction FAILED by
its own pre-stated falsifier.** PREREG.md registered "attenuation cost > interference cost" and
wrote: *"If instead ADD-100 collapses while AVG holds, the near-orthogonality reading is wrong."*
ADD-100 fell to 6/192 while AVG held at 69; interference 96 rows (p = 7.0e-26) against
attenuation 33 (p = 3.6e-08).

**A post-hoc decomposition of the errors then shows what the strict endpoint was measuring.**
88 of the 99 rows ADD-100 loses relative to SPARSE are the correct answer plus one trailing
period. Under a period-tolerant rule — chosen after seeing the generations — interference is
5 rows (p = 0.33, ns) against attenuation 33, and on the 181 content-clean rows 4 (p = 0.45)
against 29 (p = 4.2e-07).

So: the registered test failed; an unregistered analysis explains the failure as a format
transfer rather than a capability loss. The tolerant result is a well-supported *explanation*,
not a *confirmation* — confirming it needs a fresh unexposed bank with the tolerant rule
registered in advance.

## The scores, and the artifact that nearly buried them

| arm | α (Sparse) | β (Opallo) | **strict** | **period-tolerant** |
|---|---|---|---|---|
| BASE (Supa) | 0 | 0 | 0 | 0 |
| **SPARSE** | 1.0 | 0 | **102** | **102** |
| AVG | 0.5 | 0.5 | 69 | 69 |
| ADD-75 | 0.75 | 0.75 | 59 | 94 |
| ADD-O75 | 1.0 | 0.75 | 63 | 98 |
| ADD-S75 | 0.75 | 1.0 | 4 | 89 |
| **ADD-100** | 1.0 | 1.0 | **6** | **97** |
| CONFLICT-LOCAL | 1.0 | 1.0* | 6 | 97 |
| OPALLO | 0 | 1.0 | 1 | 8 |

Read strictly, ADD-100 collapses from 102 to 6 and the obvious conclusion is that full-strength
composition annihilates the skill. That conclusion is **wrong**.

**88 of the 99 rows ADD-100 loses relative to SPARSE are the correct answer with a single trailing period.** *(Corrected in Erratum 3: this line originally read "91 of ADD-100's 96 lost rows", which divided ADD-100's total gold+"." count (91) by the NET loss (96).)*

```
gold {"A":27,"B":22,"C":-7,"D":33}     gen {"A":27,"B":22,"C":-7,"D":33}.
gold 67                                 gen 67.
```

Verified exhaustively: every recovered row equals `gold + "."` exactly — not whitespace, not
multiple periods, nothing else. SPARSE and AVG produce **zero** trailing periods; ADD-100 and
ADD-S75 produce **178/192**.

### Why: the donor transfers its punctuation convention

Opallo's own training answers all end in a period — `64.`, `40.`, `$4.`, `10:25 AM.` — and the
lab's own arithmetic scorer `architecture_benchmark_001/retention.py` contains
`_strip_one_period`: *"Remove at most one closing sentence period."* Period-tolerance is the
**registered convention of Opallo's domain**. Titan's transition contract is strict.

So the composed model inherits Opallo's surface convention and is then judged by Titan's scorer.
The β-ladder in trailing periods (0 → 0 → 63 → 178) is a **format-convention transfer**, not a
capability loss. It appears at β ≥ 0.75 and is total at β = 1.0.

This is the programme's "read the text, not just the structure" rule biting hard: a one-character
convention read as a 96-row capability collapse.

## The registered decomposition

| | strict | period-tolerant |
|---|---|---|
| **attenuation** = SPARSE − AVG | +33, lost 36 / gained 3, p = 3.6e-08 | **+33, p = 3.6e-08** |
| **interference** = SPARSE − ADD-100 | +96, p = 7.0e-26 | **+5, lost 11 / gained 6, p = 0.33** |

Under the rule that does not penalise the imported convention, **attenuation is 6.6× interference
and interference is not distinguishable from zero.**

This **replicates** `ATTENUATION_CORRECTION_20260918` (235 → 223 → 219: attenuation 12 rows,
second task 4 more, p = .45) on a fresh, oracle-verified, uncontaminated bank at a different n
and with different donors. Two independent instruments, same decomposition.

## What this means for "can we add a skill with barely any loss?"

On the **loss** side, for the skill already present: **yes, if you use independent coefficients.**
5 rows out of 102, not significant. Near-orthogonality does buy cheap composition on the
interference axis — that part of the geometric story holds.

Three qualifications, all load-bearing:

1. **Do not average.** The 0.5/0.5 soup costs 33 rows at p = 3.6e-08. The registered
   `SOUP-MINI-001` arm set is built entirely from 0.5 coefficients and therefore pays the
   dominant cost in every arm.
2. **The convention transfer is a real regression**, even though it is not capability loss. Any
   downstream consumer doing exact match breaks completely. It is trivially repairable by
   stripping one period, but it must be *known about* — and nothing in weight-space geometry
   predicted it.
3. **The acquisition side is unmeasured.** This bank shows Titan's skill survives. It does not
   show Opallo's skill arrived. Until the arithmetic lane exists
   (`ARITHMETIC_LANE.md`), "acquires a new skill" is not tested — only "does not lose the old one".

## Controls and negative results

- `BASE` 0/192 and `OPALLO` 1/192 (8 period-tolerant): the bank is specific to Titan's skill, and
  Opallo contributes essentially nothing to it. So the surviving 97 in ADD-100 is Titan's skill,
  not a shared one.
- `CONFLICT-LOCAL` is byte-identical in outcome to ADD-100 (6 / 97, same shape counts). It
  attenuates two tiny norm tensors and changes nothing — confirming the audit panel's finding
  that it is a no-op arm, now on a second instrument.
- Family structure of the strict collapse: `prerequisite_plan` 37 → 2 at β = 0.75 while
  `register_machine` is untouched (33 → 34). Period-tolerant, that asymmetry is much reduced —
  it was largely the JSON-integer families being easiest to break with punctuation.

## Provenance

- Oracle validated 384/384 against the recorded bank before any row was generated.
- Payloads disjoint from 6,610 historical payloads across every records-tree jsonl.
- Template self-test reproduces 384 recorded prompts byte-exactly, with a negative control.
- Decode budget verified non-censoring (`DECODE_BUDGET_QC.md`): longest gold 35 chars, longest
  correct generation 34–35, donor never reaches the 48-token cap.
- Collected under the shared GPU lease after an earlier un-leased pass was discarded.

## Registered-primary statement

The registered primary endpoint is strict exact match, and under it ADD-100 scores 6/192. That
number stands and is not retroactively changed. The period-tolerant column is a **secondary,
post-hoc decomposition of the errors**, reported because it identifies *what* the strict endpoint
is measuring. Both are true; only together are they honest.


---

# Erratum 1 (2026-09-23) — the period convention was attributed to the wrong evidence

Line 46 above says *"Opallo's own training answers all end in a period — `64.`, `40.`, `$4.`,
`10:25 AM.`"*. **Those four strings are not Opallo's training answers.** They are gold answers of
the 150-row arithmetic panel (`architecture_benchmark_001/data/monitor_probe.jsonl`, sha
`5f6df1c1…`; they occur there 2, 1, 3 and 3 times). Opallo never trained on that panel or on its
classes.

What Opallo actually trained on, re-derived from the raw files:

- Checkpoint `915af77f` (the `OPALLO` arm's source) consumed **41 rows**
  (`train_row_ids_consumed` in `evidence/retention-design-v1/ARM_DOSE_B28last20_lr1e-05_b5200_seed17_wsd0.2-0.2_b20.95.json`)
  from `outputs/splits_large/train.jsonl` (sha `cd57e650…`), spanning **24 conversational reply
  classes** (`natural_follow_up` 5, `counting_sets` 2, `schedule_ordering` 2, …).
- That 10,000-row pool contains **zero** rows of `percentage_reasoning`, `ratio_proportion` or
  `time_duration`. Opallo's 73/150 (BF16) / 74/150 (FP32) on the panel is **transfer**, not
  in-domain skill.
- Final character of the 41 training answers: **`.` 34, `?` 6, `!` 1.** Median length 5 words,
  maximum 11. Four are bare numerics followed by a period: `52.`, `22.`, `14.`, `1.25 liters.`

**Which direction of the inference this breaks.** The *evidence* was wrong; the *mechanism* is not
weakened. "The composed model inherits a terse, period-terminated answer habit from the Opallo
delta" is supported more directly by Opallo's real training rows (83% end in `.`, including bare
numbers written `52.`) than by the panel golds that were quoted. The attention localisation in
`MODULE-DISJOINT-001` is unaffected.

**What is weakened:** the sentence *"Period-tolerance is the registered convention of Opallo's
domain."* `_strip_one_period` is a property of the arithmetic **panel's scorer**, not of any domain
Opallo was trained in. The two-scorer hazard in `CROSS_LANE_HAZARD_scorer_pair.md` stands as a
statement about the **instruments**. It should not be read as "each donor matches its own scorer".

No number in this file changes.

---

# Erratum 2 (2026-09-24) — T_BANK_V2 is ID-disjoint, not content-disjoint

The Provenance section says payloads are *"disjoint from 6,610 historical payloads across every
records-tree jsonl."* **That is true only at the level of the full payload string.** The exclusion
keyed on the entire normalized payload, including fields the answer does not depend on (`tick` in
ledger/register rows; `cost` and `name` in prerequisite rows). A generated row that matches a
historical row on every answer-determining field and differs only in `cost` passed as "fresh".

Re-keyed on operative fields only (`CONTENT_CONTAMINATED_ROWS.json`):

| bank | full-payload collisions | operative-content collisions | same (before-state, answer) |
|---|---|---|---|
| T_BANK (v1, rejected) | 0 | 0 | 5 |
| **T_BANK_V2** | **0** | **4** | **11** |

All in `prerequisite_plan`. **The v2 repair caused it.** Resampling from the recorded input
marginals (single-bit `add`, 0–1-bit `delete`, 1–5-bit states) shrank that family's state space
enough that generated rows landed on recorded ones. Matching the distribution bought difficulty
parity at the price of literal overlap — a trade-off neither PREREG amendment anticipated.

**The contamination is real in effect, not only in label.** On the 11 dirty rows SPARSE scores
9/11 (82%) against 93/181 (51%) on clean rows, while BASE scores 0/11 — so the rows are not
intrinsically easy. That is a memorization signature.

## Every conclusion survives on the 181 content-clean rows

| contrast (period-tolerant) | all 192 | clean 181 |
|---|---|---|
| attenuation SPARSE → AVG | −33, p = 3.6e-08 | **−29, p = 4.2e-07** |
| interference SPARSE → ADD-100 | −5, p = 0.33 | **−4, p = 0.45**, ns |
| P1′ SPARSE_ATTN → SCALAR_MATCHED | +20, p = 3.6e-05 | **+19, p = 6.6e-05** |
| P6 SPARSE_MLP → SCALAR_MATCHED_MLP | +27, p = 1.1e-07 | **+24, p = 8.0e-07** |
| P2 SPARSE_ATTN → MODULE_DISJOINT | +5, p = 0.33 | **+6, p = 0.21**, ns |
| P3 ADD-100 → MODULE_DISJOINT | −24, p = 1.8e-04 | **−21, p = 7.5e-04** |

Gates on the clean subset, thresholds scaled to n = 181: G1 base 0 ≤ 36.2, G2 donor 93 ≥ 72.4,
G3 separation 93 ≥ 54.3. All pass.

Every registered verdict keeps its direction and its significance class; magnitudes shrink by
10–15% because trained arms lose the rows they had memorized, and every arm containing dSparse
loses them roughly equally. **Absolute levels** on this bank were inflated by ~5% for trained arms;
**within-bank contrasts** were not materially affected.

## What changes

- The bank must be described as **content-overlapping on 11/192 rows (5.7%)**, not "disjoint".
- It is **exposed**: nine plus five arms have been scored on it, and it informed Amendments 1–4 of
  MODULE-DISJOINT-001. It is a development bank, not a confirmation instrument, independent of the
  overlap. (`titan_opallo_composition_audit_v2/COEFFICIENT_PLAN.json` classifies it as not reusable
  for confirmation; this erratum supplies the specific mechanism.)
- **Rule for the next generator:** key exclusion on answer-determining fields, and check the
  (input-state, answer) pair as well — see memory `id-disjoint-is-not-content-disjoint`. And when a
  repair narrows a sampling distribution, re-run the collision check: v1 had zero operative overlaps
  and v2 had four.


---

# Erratum 3 (2026-09-24) — headline order, the 91/96 arithmetic, and residual Opallo-skill wording

Found by an adversarial review (reviewer returned; its verifier died on a session limit, so each
item below was re-derived by hand before being written).

1. **Headline order.** The original headline presented the period-tolerant decomposition as the
   result and left the registered primary to a closing section. It now leads with the registered
   outcome — which failed by its own falsifier — and labels the tolerant analysis as the post-hoc
   explanation it is. The numbers were always correct; the *order* implied a confirmation.
2. **"91 of 96" divided a total by a net.** ADD-100 emits exactly `gold+"."` on 91 rows overall.
   SPARSE→ADD-100 strict is lost 99 / gained 3 / net −96. Of the 99 lost rows, **88** are
   `gold+"."`. Corrected in place above. Same conclusion, right arithmetic.
3. **Residual wording that treats arithmetic as Opallo's skill.** After Erratum 1 established that
   Opallo trained on zero arithmetic-class rows, "acquisition side is unmeasured … does not show
   Opallo's skill arrived" (qualification 3 above) presupposes a donated arithmetic skill that
   never existed. The accurate statement: this bank measures only whether Titan's transition skill
   survives composition. Whether composition *transfers* anything from Opallo depends on an
   endpoint where Opallo has a measurable, specific effect — see `O_SKILL_DEFINABILITY.md`
   consequence 4 and FA-001's O-side, not an arithmetic bank.

Direction of these corrections: all three narrow over-claims. None retracts a measured number.

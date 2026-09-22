# COMPOSE4B-003 — complete the 3x2 factorial, and run the control that WITHDREW a 596M finding

> **STATUS: REGISTERED, NOT BUILT.** No arm exists, nothing is rendered, nothing is scored as of
> this line. Records directory `compose4b003-20260921` claimed **`mkdir` create-only** before
> this file was written. Committed with the arm directories absent.

**Class: CONTROL + FACTORIAL COMPLETION. Zero training.** Two new arms built by tensor
arithmetic over adapters already verified. Promotes nothing, releases nothing.
`quality_claim: false`, `release_permitted: false`.

---

## 1. Why this experiment exists — it is a control I owe, not a new idea

COMPOSE4B-002 published two things. One of them is safe and one of them is **confounded in a way
the 596M lane has already caught once, in a document I had not read to the end.**

`COMPOSITION_IS_A_REPAIR_20260920.md` **Amendment 2** withdraws its own Finding 2:

> *"The repair is attenuation, not composition."* SCALED-H-001 built `theta_supa + s_t·P16(delta_H)`
> — delta_H at the composite's attenuation with **no delta_T at all** — and recovered **18 of 19**
> answerable rows on its own (p 7.6e-06). `SCALED_H` also holds **J_decision 0.7589, the highest
> of any arm on record, with no delta_T in it.**

Checking my own two claims against that control:

| COMPOSE4B-002 claim | confounded? |
|---|---|
| **P1 `th_s - scaled_t` sensitivity +19** (nd 31, p 8.8e-4) | **No.** Both arms hold delta_T at the identical per-module `s_t`; only delta_H's presence differs. Survives. |
| **§5 descriptive `th_s - delta_t` sensitivity +26** (nd 38, p 2e-5) | **Yes — exactly as Finding 2 was.** delta_T is attenuated *and* delta_H is added in one step. This is the comparison a reader makes against the shipped single-skill arm, and **I published it with no `scaled_h` arm in existence on this base.** |

**The 4B lane has never run delta_H at attenuation without delta_T, and has never run delta_T at
full strength with delta_H present.** Those are two empty cells of one factorial, and both are
adapter arithmetic. This experiment fills them.

## 2. The design — a 3x2 factorial, of which four cells are already scored

Coefficients are **per 2D tensor**. `s_t` denotes the per-module vector in
`compose4b001-20260921/PER_TENSOR_SCALES.json`; its median is 0.5869, min 0.4587, max 0.7176.

| | **delta_H = 0** | **delta_H at s_t** | **delta_H at 1.0** |
|---|---|---|---|
| **delta_T = 0** | `base` (scored) | **`scaled_h` — NEW** | `h_only` (scored) |
| **delta_T at s_t** | `scaled_t` (scored) | `th_s` (scored) | — |
| **delta_T at 1.0** | `delta_t` (scored) | **`indep` — NEW** | — |

Two new arms close the factorial. The four scored arms are **re-used exactly as scored** — no
re-generation, no re-scoring, identical 224 row ids — so this experiment's GPU cost is two arms.

    scaled_h = s_t * delta_H                     (the SCALED-H-001 control, on this base)
    indep    = 1.0 * delta_T + s_t * delta_H     (independent coefficients; the INDEP-001 cell)

**Why delta_H sits at `s_t` in `indep` and not at 0.25.** Holding delta_H at the identical
per-module `s_t` across every composed cell makes **delta_T's coefficient the only thing that
varies**, which is the separability question. 0.25 is a 596M-calibrated operating point and
importing it here would vary two things at once. `s_t` (median 0.5869) also sits inside the band
H-ALPHA-001 measured as **boundary-saturated, alpha 0.20–0.675** — the only endpoint for which
magnitude invariance was actually measured, and the reason a same-magnitude anchor is admissible
on boundary and **not** on long context, where INDEP-002 found real damage between 0.25 and 0.50.

## 3. Construction — extending the existing builder, not writing a third one

BESTARM-001's rule, adopted verbatim: **"a third implementation is a third chance to disagree."**
Every construction bug this record has suffered came from re-implementing arithmetic that already
existed. So both arms are built by `build_compose_003.py`, which **imports `build_compose.py`'s
own `modules()` and `eff()`** and reuses its concatenation pattern.

For a LoRA module, effective `dW = (alpha/r)·B@A`. Both arms are **exact**, not approximations:

    scaled_h : A' = A_H,              B' = s·B_H          r'=r,  alpha'=alpha
               (alpha/r)·(s·B_H)@A_H == s·delta_H

    indep    : A' = [A_T ; A_H],      B' = [B_T | s·B_H]  r'=2r, alpha'=2·alpha
               (2a/2r)·B'@A' == (a/r)(B_T A_T) + s·(a/r)(B_H A_H) == delta_T + s·delta_H

Note the difference from `th_s`, which multiplies **the whole** of `B'` by `s`. Here only the
**H half** is scaled. That one-column-block distinction *is* the experiment.

**`s_t` is READ BACK, never recomputed.** `PER_TENSOR_SCALES.json` sha256
`ebabde31a6d8af6b3d52ae133bc0f9cc653de7afd9a2af66489ee6c2e8288d12`, 252 modules, verified at
build time. SCALED-T-001 adopted read-back after a recomputed scale diverged at the ninth
decimal; a rebuild that silently drifted would make `th_s` and `indep` incomparable.

## 4. Build gates — each one is shown able to FAIL and to PASS before it is trusted

| # | gate | fails when | passes when |
|---|---|---|---|
| 1 | `PER_TENSOR_SCALES.json` sha256 == `ebabde31…`, 252 modules | the scales file was rebuilt or drifted | the recorded file is byte-identical |
| 2 | delta_T and delta_H `adapter_config` agree on `r`, `lora_alpha`, `lora_dropout`, `bias`, `peft_type`, target modules | the two adapters were trained under different parameterizations | they match, as `build_compose.py` already verified |
| 3 | adapter key sets identical | a key silently dropped a term | equal sets |
| 4 | `rel_err(reconstructed, target) <= TOL` per module, both arms | the concatenation or the scale placement is wrong | error at float32 noise |
| 5 | **magnitude identity**: `norm(scaled_h) == s·norm(delta_H)` and `norm(indep - delta_T) == s·norm(delta_H)`, per module | `s` applied to the wrong block, or to both blocks | holds to `TOL` |
| 6 | **POSITIVE CONTROL** — a deliberate mutation (scale the T half instead of the H half) must make gate 4 or 5 FAIL | the gate is vacuous and would pass anything | the mutation is detected and reported |

**`TOL = 2e-6`.** This is **SCALED-T-001's own amended numerical tolerance, adopted unchanged,
not invented here.** float32 adapter storage has eps 1.19e-7, so the 1e-9 that COMPOSE4B-001
first tried is unachievable *in principle* — it refused at 6.1e-8, which was the guard being too
strict, not the arithmetic being wrong. `build_compose.py`'s observed worst error was **6.13e-8,
about 33x inside this tolerance.** A tolerance that cannot be met is not a gate.

Gate 6 exists because an absence check and a passing check return the same answer. Gates 4 and 5
are run **against a deliberately wrong arm first**, and the failure recorded, before any real arm
is written.

## 5. Endpoints — the sealed rule, imported and never reimplemented

Instrument **HSKILL-VIEWS-002**, `HSKILL_VIEWS_V2.jsonl` sha256 `ba1f2982fd5d08aa…`,
rule `boundary_rule.py` sha256 `416cfb6aec8aa282…`, 224 rows = 112 boundary + 112
`arithmetic_short_answer`, `view_role: ADMITTED_FOR_SELECTION`.
**`HSKILL-CONFIRM-002` is not opened and must not be.**

- **sensitivity** = abstains on the 112 boundary rows
- **specificity_decision** = attempts on the 112 answerable rows
- **specificity_competence** = attempts **and** is correct on the 112 answerable rows
- **J_decision** = sensitivity + specificity_decision − 1

## 6. Multiplicity, fixed before the data

The instrument has now carried 16 arms. **That is not the multiplicity that matters.**
Multiplicity is about how many hypotheses the data is allowed to choose between, not how many
times the rows have been read; `ADMITTED_FOR_SELECTION` permits the reuse, and selection does not
need family-wise control the way confirmation does.

**What does need control is this experiment's own registered family.** Fixed now:

> **Holm–Bonferroni across the three registered predictions P1, P2, P3, m = 3, fixed regardless
> of outcome.** Each contrast reports `b`, `c`, `nd`, exact two-sided p, `min_attainable_p`,
> MDE at the observed discordance, and its ceiling. **No instrument-level claim is made.**

**An unsatisfiability check on the Holm procedure itself, done in advance.** Holm's most
stringent step is `0.05/3 = 0.01667`. The minimum attainable exact p is 0.0312 at nd = 6 and
0.0156 at nd = 7. **Therefore a contrast with nd <= 6 cannot clear Holm's first step no matter
how the pairs split.** Such a contrast is reported `UNINFORMATIVE_BY_DISCORDANCE` with its nd and
`min_attainable_p` printed; it is **not** counted as evidence of absence, and m stays 3.

MDE at n = 112, exact paired McNemar, two-sided 0.05, computed before any arm exists:

| nd | 4 | 6 | 8 | 10 | 14 | 20 | 24 | 30 | 36 | 42 | 50 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MDE | **n/a** | 6 | 8 | 8 | 10 | 10 | 12 | 12 | 14 | 14 | 16 | 18 |

## 7. Registered predictions

Ceilings are stated **now**, from the already-scored comparator arms, not discovered afterwards.
A contrast fires iff `net >= MDE(nd)` **and** its Holm-adjusted p <= 0.05. If
`ceiling < MDE(nd)` it is reported `UNINFORMATIVE_BY_CEILING`.

### P1 — PRIMARY. `indep - delta_t` on SENSITIVITY. Does the gate arrive at FULL delta_T?

Every composed arm scored on this base so far holds delta_T **attenuated**. P1 asks the question
INDEP-001 answered on the 596M — *does a second skill arrive at its own coefficient without
paying for it out of the first* — in the one cell that has never been run here.

- 596M analogue, INDEP-001: boundary **69 → 85 = +16, p = 0.0015**, acquisition unchanged.
- `delta_t` sensitivity is **67/112**, so the **ceiling is 45** — stated in advance, ample.
- **Registered band: net +8 … min(30, 45). Point prediction +16.**

### P2 — `indep - delta_t` on SPECIFICITY_COMPETENCE. Does the gate cost answering?

This is *"is composition free"* asked where it has never been asked on this base: with delta_T at
full strength rather than at 0.5869.

- `delta_t` competence is **57/112** — gain ceiling **55**, loss ceiling **57**.
- **Registered as a POWERED NULL OR NOTHING.** No tolerance margin is registered: at n = 112 even
  a perfect tie bounds the loss-side excess only at ~7.9 rows (Clopper–Pearson), so a tighter
  margin would be unsatisfiable by construction. A null here is reported **with its equivalence
  bound**, never as "no effect".

### P3 — CONTROL. `th_s - scaled_h` on SPECIFICITY_COMPETENCE. Does delta_T contribute anything?

**This is the control Amendment 2 is made of, and it is the one that can retract my own result.**
`th_s` holds delta_T at `s_t` and delta_H at `s_t`; `scaled_h` holds delta_H at the identical
`s_t` with no delta_T. The difference is delta_T's presence, with attenuation matched.

- `th_s` competence is **66/112**; `h_only` (delta_H at full strength) is **54/112**; `base` is
  **3/112** — and the base's 3 is a *container* failure, so any adapter fixes the format and
  competence differences between adapters are about arithmetic.
- **Registered band: net >= +MDE(nd) in favour of `th_s`.** I predict delta_T contributes real
  arithmetic competence.
- On the 596M this contrast was **unresolvable** — nd 1 and nd 4, `min_attainable_p` 1.0 and
  0.12. If nd is that small here, P3 reports `UNINFORMATIVE_BY_DISCORDANCE` and **licenses
  nothing in either direction.**

### DESCRIPTIVE — no prediction, not in the Holm family

- `indep - th_s` on both endpoints — the delta_T-magnitude contrast at delta_H held fixed.
  Completes the factorial and is the cleanest available read on whether attenuating delta_T costs
  competence.
- `th_s - scaled_h` on SENSITIVITY — the direct 4B analogue of SCALED-H-001's boundary contrast.
- `J_decision` for all seven arms; per-family sensitivity across the seven boundary classes.

## 8. What each outcome licenses — including the one that retracts me

| outcome | reading |
|---|---|
| **P1 fires** | the gate skill arrives at its **own** coefficient with delta_T at full strength. INDEP-001 replicates cross-base, and the composite is not buying the gate with delta_T's magnitude. |
| **P1 does not fire** | delta_H's boundary contribution **requires** delta_T to be attenuated — i.e. the 4B's composed gain is a magnitude effect on delta_T, not an allocation result. This does **not** touch COMPOSE4B-002 P1, which matched attenuation. |
| **P2 null with a reported bound** | composition is free *on competence, at full delta_T*, to within the stated equivalence bound. Never "no cost". |
| **P3 fires** | delta_T contributes arithmetic the attenuated gate cannot supply. `th_s` is a genuine composite and **RESULT_002 §5 stands.** |
| **P3 DOES NOT fire** | **the competence half of RESULT_002 §5 is confounded with attenuation exactly as Finding 2 was, and I will amend RESULT_002 to say so.** Registered here, before the data, so that the amendment is an outcome and not a defeat. |
| **P3 `UNINFORMATIVE_BY_DISCORDANCE`** | the instrument cannot separate them at this n. Reported as such; **§5 is qualified as UNTESTED, not as vindicated.** A check that could not have failed is not a check. |

## 9. What this does NOT license, whatever happens

- **Not a release, not a promotion.** One instrument, one base, one draw.
- **Not "composition is free" in general.** One delta_T/delta_H pair, one coefficient vector.
- **Not a damage result on the endpoints INDEP-001 named.** `references_retained` /47 and the
  150-row arithmetic panel remain **structurally unrunnable** on a 4B adapter (both require Supa
  sidecars and a Supa-hashed materialization receipt); long context remains SPENT/REPORT-ONLY.
- **Not a claim about delta_H at its own 0.25 operating point**, which is not built here.
- **No unregistered direction is ever promoted.** COMPOSE4B-002's P2 came in at +15 in a
  direction I had not registered and was reported as nothing. The same rule applies here.

## 10. Envelope

2 new arms x 224 rows x 48 new tokens, greedy, `enable_thinking=False`. ~4 min/arm ≈ **8 min
GPU**, one lease. Four existing arms are re-used as scored and cost nothing. Adapter arithmetic is
CPU and runs while the lease is held by another lane. Disk: two LoRA adapters plus text output,
well under 500 MB. **14 GiB free-disk floor and the host-RAM gate enforced unchanged; neither is
lowered.** Zero training, zero optimizer steps, one battery, no retry registered.

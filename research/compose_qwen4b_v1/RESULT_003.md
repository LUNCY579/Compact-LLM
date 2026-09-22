# COMPOSE4B-003 — the control that could have retracted me FIRED, and a 596M withdrawal only HALF-replicates

**2026-09-22 · 2 new arms × 224 rows · zero training · registered in `PREREG_003.md`, committed
with `scored/` verifiably empty and the records directory containing that file alone.**

`quality_claim` false. `release_permitted` false. **No arm promoted.**

---

## 1. The canary, before any 003 number was read

`analyze_003.py` was committed with `scored/` empty. Its statistics are **imported** from
`analyze.py`; only the row extraction is new, because HSKILL's schema differs from the
acquisition schema. That is exactly where this programme keeps getting hurt — four wrong
extractors in one day, none caught by reading code — so the script **refuses to read a 003
number** unless it first reproduces COMPOSE4B-002's three published contrasts:

| check | published | recomputed |
|---|---|---|
| `th_s − scaled_t` sensitivity | +19, nd 31, p 0.00088 | **+19, nd 31, p 0.000878** |
| `th_s − scaled_t` competence | +15, nd 35, p 0.01667 | **+15, nd 35, p 0.016674** |
| `h_only − base` sensitivity | +2, nd 48, p 0.885 | **+2, nd 48, p 0.885433** |

**PASS.** Validated by collision with known numbers, not by reading.

## 2. The completed 3×2 factorial

| arm | δ_T | δ_H | sensitivity /112 | decision /112 | competence /112 | J_decision |
|---|---|---|---:|---:|---:|---:|
| `base` | 0 | 0 | 77 | 111 | 3 | +0.6786 |
| `delta_t` | 1.0 | 0 | 67 | 112 | 57 | +0.5982 |
| `scaled_t` | s_t | 0 | 74 | 112 | 51 | +0.6607 |
| `h_only` | 0 | 1.0 | 79 | **89** | 54 | +0.5000 |
| **`scaled_h`** ← new | 0 | s_t | 91 | **102** | 53 | +0.7232 |
| `th_s` | s_t | s_t | 93 | 112 | 66 | +0.8304 |
| **`indep`** ← new | 1.0 | s_t | **95** | **112** | 64 | **+0.8482** |

Four arms are re-used exactly as COMPOSE4B-002 scored them — same driver, same rule, same 224 row
ids, and the render gate confirmed `prompts_sha256 25d3b50f…` byte-identical, so every contrast
below is genuinely paired.

## 3. Registered outcomes — Holm–Bonferroni, m = 3, fixed before the data

| | contrast | b/c | nd | net | exact p | Holm adj | MDE | ceiling | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| **P1** | `indep − delta_t` sensitivity | 32/4 | 36 | **+28** | 1.94e-06 | 5.82e-06 | 14 | 45 | **FIRES** |
| **P3** | `th_s − scaled_h` competence | 19/6 | 25 | **+13** | 0.01463 | 0.02927 | 11 | 59 | **FIRES** |
| P2 | `indep − delta_t` competence | 13/6 | 19 | +7 | 0.1671 | 0.1671 | 11 | 55 | **null, as registered** |

### P1 fires, and larger than its anchor

Adding δ_H at its own coefficient, with **δ_T at full strength**, buys **+28 boundary rows**
(32 gains against 4 losses). Registered band was **+8 … min(30, 45)** with a point prediction of
+16 from the 596M's INDEP-001. Observed **+28 — inside the band, and 12 rows above the anchor.**

Worth stating plainly: `delta_t` abstains on **67/112, which is 10 rows BELOW the untouched base
(77).** The task delta *costs* abstention. `indep` reaches **95** — so δ_H does not merely repair
δ_T's damage, it clears the base by 18.

### P3 fires — so the clause that would have retracted me is not triggered

`th_s` and `scaled_h` hold δ_H at the identical per-module `s_t`; the only difference is δ_T's
presence. **δ_T contributes +13 competence rows (19 gains / 6 losses, p = 0.0146).**

PREREG_003 §8 registered: *"if P3 does not fire, the competence half of RESULT_002 §5 is
confounded with attenuation exactly as Finding 2 was, and I will amend RESULT_002 to say so."*
**P3 fired. RESULT_002 §5 stands**, and it stands because a control that could have killed it was
run, not because it was never tested.

### P2 is a null, and it is reported with its bound

+7 in the *favourable* direction, nd 19, p = 0.167 — below the MDE of 11. Registered as a
**powered null or nothing**, so it is the null. With its equivalence bound:

> **At 95 % confidence, adding δ_H at its own coefficient costs at most ~1.1 competence rows**
> (Clopper–Pearson upper bound 0.530 on 6 losses of 19 discordant pairs).

That is a far tighter statement than COMPOSE4B-002 could make, where even a perfect tie bounded
the loss only at ~7.9 rows. **`indep − delta_t` on the decision endpoint is a perfect tie: 0
gains, 0 losses, nd 0 — both arms attempt all 112 answerable rows.** The gate costs nothing at
all on whether the model answers.

## 4. THE MAIN FINDING — a 596M withdrawal replicates on one endpoint and fails on the other

`COMPOSITION_IS_A_REPAIR` **Amendment 2** withdrew its own Finding 2 with SCALED-H-001:
*"the repair is attenuation, not composition"* — δ_H attenuated, with **no δ_T**, recovered 18 of
19 answerable rows by itself, and `SCALED_H` held the highest `J_decision` of any arm on record.

Running that same control on the 4B for the first time:

| | 596M (SCALED-H-001) | **4B (here)** | |
|---|---|---|---|
| `th_s − scaled_h` **boundary** | nd 4, min p 0.12, unresolvable | **+2, nd 4, min p 0.125, unresolvable** | **replicates exactly** |
| attenuation recovers answering | **18 of 19** | **13 of 22** (`scaled_h − h_only` decision **+13, p 0.0072**) | replicates in direction |
| δ_T's further contribution to answering | none measurable | **`scaled_h` stalls at 102/112; δ_T takes it to 112/112** | **does NOT replicate** |
| δ_T's contribution to competence | not resolvable | **+13, p 0.0146 (P3)** | **does NOT replicate** |
| best `J_decision` | `SCALED_H`, with no δ_T in it | `indep` **0.8482** > `th_s` 0.8304 > `scaled_h` **0.7232** | **does NOT replicate** |

> **δ_T's contribution is ENDPOINT-SPECIFIC: nothing measurable on the gate, significant on
> answering.** The boundary half of Amendment 2 replicates to the discordance count. The
> answering half does not survive the change of base.

What survives cross-base is the *shape* of Finding 3 — **the asymmetry is an endpoint property,
not a skill property** — and what does not is Finding 2's stronger reading that δ_T adds nothing
at all. **A withdrawal can itself be base-specific.**

## 5. The over-dose account, converted from interpretation to measurement

`scaled_h − base` on sensitivity is **+14 (21 gains / 7 losses, nd 28, p = 0.0125).**

COMPOSE4B-002's registered control P3 was `h_only − base` on sensitivity with a band of ≥ +20,
and it came in at **+2, p 0.885 — a registered miss.** The only difference between that arm and
this one is **dose**: α = 1.0 versus the per-module `s_t`.

**I am not using this to rescue the failed registered control.** That clause stands as written,
exactly as it did in RESULT_002 — *"I do not get to swap a passing measure in for a failed
registered one."* What it does is convert RESULT_002 §4's *interpretation* ("δ_H is over-dosed,
not weak") into a **direct measurement on the registered control's own contrast form**: the same
delta, the same comparator, the same endpoint, a different dose, +2 → +14.

This is [[delta-magnitude-saturates-early]] on a third instrument, and the answering side is the
sharper version: `h_only` attempts **89/112**, `scaled_h` **102/112**, `th_s`/`indep` **112/112**.

## 6. What must NOT be read out of this

- **`indep` being the best arm on record here is a DESCRIPTIVE RANKING, not a significant
  difference.** `indep − th_s` is **nd 2** on sensitivity (min attainable p 0.5) and **nd 8,
  net −2, p 0.727** on competence. The two composites are nearly identical; at nd 2 the
  instrument cannot certify any difference. **Nothing is promoted.**
- **Not a release.** One instrument, one base, one draw, one δ_T/δ_H pair, one coefficient vector.
- **Not a damage result on the endpoints INDEP-001 named.** `references_retained` /47 and the
  150-row arithmetic panel remain structurally unrunnable against a 4B adapter; long context
  remains SPENT / REPORT-ONLY.
- **P2's +7 is not a gain.** It is a null in the favourable direction, reported with its bound.
- **D4's +14 does not make COMPOSE4B-002's P3 pass.** It failed as written and stays failed.
- **Not a claim about δ_H at 0.25**, which was not built.

## 7. Provenance

Instrument `HSKILL_VIEWS_V2.jsonl` sha256 `ba1f2982…`, rule `boundary_rule.py` sha256
`416cfb6a…` — **imported, never reimplemented**. `HSKILL-CONFIRM-002` untouched.
Render gate 224/224 tokenizer agreement, mutation control fires, `prompts_sha256 25d3b50f…`
byte-identical to COMPOSE4B-002's. `enable_thinking=False`, `max_new_tokens=48`, greedy.

Arms built by `build_compose_003.py`, which **imports** `build_compose.py`'s `modules()` and
`eff()`. `s_t` **read back** from `PER_TENSOR_SCALES.json` (sha256 `ebabde31…`, 252 modules),
never recomputed. **Build gate 6 was a positive control and ran BEFORE any correct arm was
written**: two deliberate mutations — `s` applied to the T half instead of the H half, and `s`
omitted from `scaled_h` — were detected at **4.87e-01** and **3.95e-01** against TOL 2e-6; the
real arms then landed at **3.41e-08**, ~59× inside. The tolerance is SCALED-T-001's own amended
value, adopted unchanged.

Statistics: exact paired McNemar, MDE at observed discordance, Clopper–Pearson equivalence
bounds, Holm–Bonferroni at m = 3 fixed regardless of outcome. The registered unsatisfiability
guard (nd ≤ 6 cannot clear Holm's strictest step of 0.0167) did not bind — the smallest
registered nd was 19.

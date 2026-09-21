# COMPOSE4B-001 — is a second skill still free at matched magnitude when the base is not capacity-bound?

> **STATUS: REGISTERED, NOT BUILT. No arm exists and nothing has been scored as of this line.**
> Registered before any training because the result it adjudicates (SCALED-T-001's P1, a
> **powered null**) is already known, and a replication designed after seeing its target is not
> a replication.

**Class: MECHANISM REPLICATION ON A DIFFERENT BASE.** One training run; everything else is
weight arithmetic. Promotes nothing, releases nothing, and permits no quality claim.

---

## 1. The question, and why the existing answer may not generalise

SCALED-T-001 established, on the 596M Supa base:

> `SCALED_T` 223 vs `TH16s` 219 — net **+4**, nd 16, exact McNemar **p = 0.4545**, against a
> **10-row MDE**. A powered null. **H's skill is free at matched T-magnitude**; the whole
> 60-row composition cost is magnitude, not interference.

That is the single most useful positive mechanism this programme has produced, and it has a
confound that the 596M base cannot rule out.

**Every 596M arm is dominated by Qwen3-4B + LoRA on both axes** (acquisition 270 vs 232, long
context 180 vs 164, literal UNKNOWN 9 vs 12 — QWENFT001). `T16` scores 235/384 on a base whose
best-ever arm is 235/384. **The 596M is at or near its own ceiling**, and a skill added to a
saturated base may be "free" precisely *because* nothing further fits.

On Qwen3-4B, `T` scores 270/384 — there is real headroom. So:

> **If composition is free only because the 596M is saturated, a 4B base should show MORE
> interference, not less. If it is a property of low-rank delta composition, the null should
> replicate with headroom to spare.**

This is a directional, falsifiable discriminator between two readings of a result that is
currently being used as a general claim.

## 2. Construction — mirrors SCALED-T-001 exactly, in LoRA space

Let `δ_T` and `δ_H` be the **effective weight deltas** of each adapter, per target module:

    δ = (alpha / r) * B @ A          materialised in FP32, per 2D tensor

    TH_s     = s_t * ( δ_T + δ_H )
    SCALED_T = s_t * δ_T
    s_t      = ||δ_T|| / ||δ_T + δ_H||        Frobenius, PER 2D TENSOR

`SCALED_T` is identical to `TH_s` in every respect **except that δ_H's directions are absent** —
same `s_t`, same base, same per-tensor treatment, same FP32 path. That is what isolates *skill
interference* from *magnitude attenuation*.

**`s_t` is computed ONCE when `TH_s` is built, written to `TH_s`'s materialisation receipt, and
then READ by the `SCALED_T` build — never recomputed.** SCALED-T-001 adopted this after
`arm-rebuild-s10` diverged from `arm-s10` at the ninth decimal under recomputation. Reading the
recorded vector guarantees the two arms share the scale exactly.

### Arms

| arm | construction | status |
|---|---|---|
| `T` | `δ_T` at full strength | **EXISTS** — QWENFT001 `run/adapter`, scored 270/384 |
| `H` | new LoRA on H's rows | **requires the one training run** |
| `TH_s` | `s_t·(δ_T + δ_H)` | weight arithmetic |
| `SCALED_T` | `s_t·δ_T` | weight arithmetic — **the control** |

## 3. Data — transfers exactly, verified by hash

`δ_H` trains on `compose001-20260918/H/H-prepared/microfit-train.jsonl`
(experiment `RRP-009-PROCESS-MIX`): **648 rows, 54,217 processed tokens, 6,274 supervised
tokens**, precomputed `ids` and `labels`.

Those rows transfer to Qwen3-4B **without retokenization** because the Supa source tokenizer and
Qwen3-4B's are byte-identical on all four files — verified here, not assumed:

| file | sha256 | |
|---|---|---|
| `tokenizer.json` | `aeb13307a71a…` | IDENTICAL |
| `tokenizer_config.json` | `d5d09f07b48c…` | IDENTICAL |
| `vocab.json` | `ca10d7e9fb3e…` | IDENTICAL |
| `merges.txt` | `8831e4f1a044…` | IDENTICAL |

This is the same equality QWENFT001 relied on for `δ_T`, re-verified for `δ_H`'s file set.

## 4. Frozen training configuration for δ_H

Identical to QWENFT001's δ_T run in every knob that is not the data:

- Qwen3-4B, **4-bit NF4** base (BF16 is 7.49 GiB against ~7.4 GiB free GPU; 4-bit is forced,
  and QWENFT001's separately-measured 4-bit zero-shot control remains the valid comparator)
- LoRA **r = 16**, same target modules (attention + MLP projections), same alpha, dropout 0
- **peak LR 1e-4**, same schedule, TF32 **off**, `float32_matmul_precision("highest")`
- one exposure, same row order as the prepared file, create-only output
- **no learning-rate search, no second training arm, no retry**

Any deviation forced by the harness is recorded as an amendment **before** the run, not after.

## 4a. AMENDMENT 1 — what δ_H actually teaches, and three declared divergences

Registered **before any arm exists**. Facts traced to `compose001-20260918/H/H-PLAN.json` and
`H-PREPARATION.json` and verified here, not taken on report.

**δ_H is an ABSTENTION / BOUNDARY skill, not a second generator.** Family composition:
clarification 124, safety_boundary 118, clarification_request 93, insufficient_information 85,
boundary_setting 82, uncertainty 82, knowledge_boundary 64. It teaches *when to decline, when to
ask for clarification, and when the context does not determine an answer.*

**This sharpens the hypothesis rather than merely describing it.** This programme has already
found that *magnitude is an acquisition-only variable, so **gates compose and generators do
not***. δ_H is a **gate** and δ_T is a **generator**, so SCALED-T-001's null is the *predicted*
case, not a surprise. COMPOSE4B-001 therefore tests something specific: **does the
gate-composes-freely result survive when the base has headroom**, or was it the 596M's ceiling
doing the work? P2 firing would damage the general claim considerably more than it would damage
this pairing.

It also means **`H` alone should score LOW on `select-views-001`** — that bank measures T's
transition contract, which H was never trained to do. A low `H` score is expected and is *not*
evidence of a failed training run. The registered 0–200 band stands, and a score near the
bottom of it is unremarkable.

### Declared divergences from δ_H's original 596M run

| knob | δ_H original (596M) | here (4B) | why |
|---|---|---|---|
| peak LR | **1e-05** | **1e-04** | original was FP32 full-tensor MQ; this is LoRA r=16, which needs the higher rate. **Matches δ_T's 4B regime**, which is the axis that must be held constant for the composition contrast to mean anything |
| parameterization | FP32 tail blocks | LoRA r=16 | tail-20 of Qwen3-4B is 2.02B params / 30.08 GiB in FP32 against 15.22 GiB host RAM — arithmetically impossible, as QWENFT001 recorded |
| base precision | FP32 | 4-bit NF4 | BF16 is 7.49 GiB against ~7.5 GiB free GPU |

**Matching δ_T's regime is the deliberate choice.** The contrast is `TH_s` vs `SCALED_T`, both
built from the same two deltas; if δ_H were trained under a different optimizer regime than δ_T,
any interference measured could be charged to the regime instead of to the composition.

**Dose is matched on TOKENS, not rows.** δ_H is 648 rows / 54,217 processed tokens against δ_T's
384 rows / 54,172 — `transition_token_budget_matched: 54172` in the preparation receipt. The
648-vs-384 asymmetry is the *matching axis*, not a defect. H's rows are short
(`max_rendered_tokens: 101`).

**Naming trap, recorded so nobody greps past it:** δ_H's row ids carry a `rehearsal:` prefix.
These are **not** replay of earlier training — they are a distinct corpus for a distinct skill.
The collision matters because this programme measured rehearsal *harmful*, and the prefix will
mislead anyone matching on it.

### Disjointness must be checked with a positive control

δ_H's `selection_sha256` is `7dd53f2a751b54c3…`, **byte-identical to
`select-views-001/SELECT_VIEWS.jsonl`** — this experiment's acquisition bank. That is benign:
`selection_sha256` records the pool a draw was *audited against*, not what was trained on, and
the dmrf session verified 384 views vs 384 train rows with 0 overlap on id, stem, parent_id and
source_id.

**But their first version of that check was VACUOUS and nearly shipped.** View ids carry an
`RRP-select:` prefix and train ids do not, so the two sides *could not have matched* even if
every row were identical — and it returned 0, which is also what a passing check returns.

> **Registered requirement: before any contrast is reported, the train/eval disjointness check
> must (a) assert both sides share an id namespace, and (b) pass a self-intersection positive
> control proving a match was detectable. A 0 is informative only after a match was possible.**

## 4b. AMENDMENT 2 — registered while the battery is running and NO score exists

Both items below are registered **before any arm has been scored** — `scored/` is empty and the
384-row battery is mid-flight. That timing is the whole point of item 2.

### (i) The build-gate tolerance change, registered rather than merely recorded

The first build gate used `TOL = 1e-9` and **refused** at a worst relative error of 6.131e-08.
It was then changed to `2e-06` and passed with ~33× margin.

**Why this is not widening-after-failure.** The general test: *widening is illegitimate when the
new tolerance is chosen to admit the observed number; legitimate when the old tolerance was
wrong for reasons visible **without** the observation.*

- `1e-9` is **below float32 machine epsilon (1.19e-7)**. The adapters are stored in float32, so
  a 1e-9 agreement between a float64 reconstruction and float32-stored weights is unachievable
  **in principle**, whatever the arithmetic does. That was true before the gate ever ran.
- `2e-06` is not a number invented to fit: it is **SCALED-T-001's own amended numerical
  tolerance** for this exact class of check — the parent construction's calibrated value.
- The 33× margin is therefore a *consequence*, not a target.

A provenance string inside `PER_TENSOR_SCALES.json` is a **record**, not a **registration**, and
the two cases above are indistinguishable from outside once a result exists. Hence this
amendment. No arm had been written when the gate refused — the gate precedes the write loop — so
the retry was clean and nothing was overwritten.

### (ii) PRE-REGISTERED INTERPRETATION CONSTRAINT: the 4B is the HARSHER test

Registered now because it forecloses a reading that would otherwise be available *after* the
number lands, and which I would be tempted by.

| | 596M (SCALED-T-001) | 4B (here) |
|---|---:|---:|
| `s_t` min | 0.5175 | **0.4587** |
| `s_t` median | **0.6750** | **0.5869** |
| `s_t` max | 0.7496 | 0.7176 |
| tensors / modules | 220 | 252 |
| fallbacks | 0 | 0 |

**δ_H is relatively larger against δ_T on the 4B, so T is attenuated *harder* inside `TH_s`
here — to 0.587 of its strength rather than 0.675.**

> **Therefore: if P1 holds and the null replicates, it may NOT be explained away as "the 596M
> null was just mild attenuation." The 4B attenuates T more, not less. A replication here is a
> STRONGER result than the original, not a weaker one.**
>
> **And symmetrically: if P2 fires and interference appears, it may NOT be attributed solely to
> headroom.** Harsher attenuation is an alternative sufficient cause, and this design cannot
> separate the two. A P2 result therefore licenses *"interference appears under harsher
> attenuation on a base with headroom"* and **not** *"headroom causes interference."*
> Separating them needs a third arm at the 596M's `s_t`, which is not built and is not
> registered here.

This constraint binds both directions, which is what makes it a constraint rather than a hedge.

### (iii) Norm-check scope, noted

The build gate verifies the **sum** is represented exactly and checks every magnitude with a
plain dense Frobenius norm (`matrix_norm(..., ord="fro")` over the materialised `B@A` product),
**not** a top-16 singular-value sum. That matters because the concatenated `TH_s` adapter is
rank 32, so any rank-16 norm shortcut would be wrong for it. No such shortcut is used.

## 4c. AMENDMENT 3 — H is PRESENT AND FUNCTIONAL at s_t, not scaled into irrelevance

Registered before any score exists. Forecloses the two cheapest dismissals of either outcome.

H-ALPHA-001 (`halpha001-20260921/HALPHA001_RESULT.md`, 4 arms, 896 rows, zero training)
measured δ_H's own magnitude curve on the 596M, on the HSKILL-VIEWS-002 abstention instrument:

| α | sensitivity /112 | specificity /112 | `J_decision` |
|---:|---:|---:|---:|
| 0.00 *(frozen source)* | 71 | 111 | 0.6250 |
| **0.25** | 86 | **112** | **0.7679** ← H's operating point |
| 0.50 | 85 | 112 | 0.7589 |
| 0.675 | 86 | 111 | 0.7589 |
| 0.85 | 85 | 104 | 0.6875 |
| 1.00 *(as trained)* | 88 | **93** | 0.6161 |

**My `s_t` median is 0.5869 — inside the flat, benign band (0.25–0.675)**, above H's optimum
and well below where specificity collapses (0.85 → 104, 1.00 → 93). Paired damage from 0.25 to
0.675 is undetectable (0 gains / 1 loss, p = 1.0).

> **Therefore, if P1 holds (the null replicates), it may NOT be dismissed as "s_t shrank H into
> irrelevance."** At 0.587 δ_H is at essentially full effectiveness on its own skill — J 0.7589
> at 0.675 against 0.7679 at its 0.25 optimum. H is present and functional in `TH_s`, not
> annihilated. A null therefore means *a functional second skill costs nothing*, which is the
> claim of interest.
>
> **And if P2 fires, "H is over-strength and simply damaging" is a WEAKER explanation than it
> looks**, because on the 596M analogue 0.587 sits in the benign band — H only becomes
> destructive above 0.85.

**Both readings carry the same caveat, stated now rather than after the number:** that curve is
596M, and its instrument is *abstention* (HSKILL-VIEWS-002), not *acquisition*
(`select-views-001`). It governs how strong H is at **H's own skill**; it does **not** govern how
much H's directions perturb **T's** acquisition, and no 4B equivalent of it exists. Transferring
it is an assumption, and it is labelled as one here rather than relied on silently.

With Amendment 2(ii), three alternative causes are now explicitly on the record for a P2 result —
headroom, harsher attenuation (`s_t` 0.587 vs 0.675), and H's absolute strength — and **this
design separates none of them.**

## 4d. AMENDMENT 4 — the allocation critique, and a correction to my own Amendment 3

Registered with `scored/` empty. Absorbs INDEP-001 and two peer audits.

### (i) Correction: which band 0.587 is in depends on the criterion

Amendment 3 called `s_t` = 0.5869 "inside the flat, benign band (0.25–0.675)." That is true
under one criterion and false under another, and saying only the favourable one was
over-claiming. H-ALPHA-001 Amendment 1's finer grid gives **two** bands:

| criterion | band | is 0.587 in it? |
|---|---|---|
| statistically indistinguishable from the max | α ∈ [0.10, 0.675] | **yes** |
| perfect specificity 112/112 | α ∈ [0.05, 0.50] | **NO — above it** |

Specificity is 112 at α ≤ 0.50 and 111 at 0.675, so **0.587 sits just past the point where
δ_H's specificity stops being perfect.** Amendment 3's conclusion survives — H is *functional*,
not scaled into irrelevance — but "benign" was the wrong word and is withdrawn.

### (ii) INDEP-001: the matched-SUM construction has a known allocation inefficiency

INDEP-001 measured `‖P₁₆(δ_T)‖ = 0.674` and `‖P₁₆(δ_H)‖ = 0.710` on the 596M — **δ_H is the
larger delta** — and argues the composition penalty is an *allocation error*, not interference:
a single scale on the **sum** forces δ_T down to pay for δ_H at several times the strength δ_H
needs.

That critique applies to **this** construction, which is the matched-SUM one. Quantified here:

| | value |
|---|---:|
| `s_t` (median) | 0.5869 |
| δ_T pays | **41.3 %** of its trained strength |
| δ_H sits at | 0.5869 of its trained strength |
| δ_H's operating point (H-ALPHA argmax) | ~0.25 |
| δ_H over-strength factor | **2.35×** |

> **This does NOT invalidate the contrast.** `TH_s` and `SCALED_T` both hold δ_T at exactly
> 0.5869; the only difference between them is whether δ_H's directions are present. The
> contrast therefore still isolates *H's presence at matched T-magnitude*, which is what it was
> registered to measure.
>
> **What it changes is scope.** COMPOSE4B-001 tests the **matched-SUM** construction, which
> INDEP-001 gives independent reason to think is a suboptimal allocation. So:
> - **a P1 null becomes a STRONGER claim** — H's presence costs nothing *even at 2.35× its own
>   operating point*;
> - **a P2 result is confounded with over-allocation** and may not be read as "composition
>   interferes." Independent per-skill coefficients (INDEP-001's arm) are the design that
>   separates them, and that arm is **not built here**.

This is the fourth unseparated cause, alongside the three in Amendment 2(ii). The design
separates none of them, and the writeup will say so rather than selecting one.

### (iii) Two audits passed, recorded because their failure modes are live in this programme

**Verdict-table exhaustiveness.** A peer found their own guard table had a hole that their only
real datapoint fell into — a *net* threshold that would have stamped a well-powered null
(9/11, nd 20, p 0.8238) as a failure. Mine was enumerated over **every** (b, c) with b + c ≤ 60:
branch counts P1 648 / P2 611 / P3 611 / UNINFORMATIVE 21, **zero holes**, and the defensive
`else` is provably unreachable. The verdict keys on the **ordered pair** through `p` and `nd`,
never on `net` alone — `net` is used only for direction after significance is established.

**Power is now reported directly, not only as an MDE.** Added
`min_attainable_p_at_observed_nd` = the p of an all-one-way split at the observed discordance.
If it exceeds α the contrast **cannot** reach significance however the pairs fall, and that is
now a first-class `UNINFORMATIVE` verdict rather than a null. Significance is unattainable
below **nd = 6** (nd 5 → min p 0.0625).

**Template-disjointness**, on a peer's warning that row- and parent-disjointness can both hold
while template-disjointness fails. Digit-blinded stems (every digit run → `#`, so digit
resamples of one template collide): 166 distinct view stems, 647 distinct H stems,
**overlap 0**, positive control detects an injected stem. Families are disjoint by construction
— views are `ledger_replay / prerequisite_plan / register_machine`, H is
`rehearsal_{safety_boundary, clarification, …}`.

## 4e. AMENDMENT 5 — P1 delivers a POWERED NULL, not an equivalence certification

Registered with `scored/` empty. This withdraws a second phrase of mine.

Amendment 3 and the P1 branch both said a null would mean *"a functional second skill **costs
nothing**."* **That is an equivalence claim, and this design cannot support it.** Withdrawn.

**Detection and equivalence are different questions with different sample-size requirements on
the very same rows.** `min_attainable_p` answers *could a difference have been detected*.
It says nothing about *can the difference be bounded below a margin*. Computed exactly for this
bank — one-sided 95% Clopper–Pearson upper bound on the loss-side excess `(c − b)`, in rows of
384:

| `nd` | perfect tie (b = c) | one row better than tie |
|---:|---:|---:|
| 10 | 5.55 | 3.93 |
| 16 | 7.08 | 5.33 |
| 20 | **7.92** | 6.12 |
| 30 | 9.67 | 7.80 |
| 51 | 11.54 | 9.61 |
| 57 | 12.22 | 10.29 |

> **Even a PERFECT TIE cannot certify equivalence tighter than about 8–12 rows at any plausible
> discordance.** Registering a tolerance margin below that would be registering an unsatisfiable
> gate — a verdict that no achievable data can deliver.

**So no tolerance/non-inferiority limb is registered here, and none will be added later.** P1
delivers a **powered null**: *no cost detected, and a cost of ≥ MDE rows would have been
detected.* The analysis now emits `equivalence_upper_bound_rows` alongside it so a reader sees
what the data actually bound, which at this n is loose.

**Cross-checked against an independent implementation.** The dmrf session computed the same
quantity for their n = 112 instrument; at nd = 20 we agree to the hundredth on all three cases
they published — perfect tie **7.92**, one worse **9.65**, one better **6.12**. Two
implementations, three full values, no disagreement.

This is the same defect class as an unsatisfiable build gate: a criterion written before anyone
checked whether achievable data could ever meet it. The difference is that this one was caught
*before* the data rather than after, which is the only time the fix is free.

## 5. Registered predictions

Primary contrast is **`TH_s` vs `SCALED_T`** on acquisition, `select-views-001`, paired exact
McNemar. Both arms are transforms of the **same two deltas**, so this is a **within-draw**
comparison and the cross-training `|net| ≥ 20` floor does **not** apply — that floor governs
comparisons across independent training draws.

| # | registered | reading if it fires |
|---|---|---|
| **P1** | `\|net\| ≤ MDE` and **p > 0.05** | **REPLICATES.** A second skill is free at matched magnitude on a base with headroom, so freeness is a property of low-rank composition, not of saturation |
| **P2** | `TH_s − SCALED_T` net **≤ −MDE** at **p ≤ 0.05** | **FAILS TO REPLICATE, informatively.** Interference is real once the base is not saturated, and SCALED-T-001's null was a ceiling artifact. This would materially weaken the composition finding |
| **P3** | net **≥ +MDE** at p ≤ 0.05 | H's presence *helps* at matched magnitude — unregistered direction, reported, claimed as nothing |

**MDE is computed from the OBSERVED discordance `nd` and reported alongside every contrast**, as
SCALED-T-001 did (nd 16 → MDE 10). It is **not** assumed from the 596M run: 4B discordance is a
different quantity and pretending otherwise would import a floor across a change of structure —
a failure already on this programme's record. **A null is reported as a null only if it is
powered at the observed nd; an underpowered null is reported as UNINFORMATIVE.**

Bands, registered before any arm exists:

| arm | acquisition /384 band |
|---|---|
| `H` alone | 0–200 (H is not the transition contract; this is descriptive, not a target) |
| `TH_s` | 200–280 |
| `SCALED_T` | 200–280 |

Preservation endpoints — long context /256 and literal `UNKNOWN` /64 — are **report-only** on
every arm, are **not** used to select anything, and carry no prediction.

## 6. Scoring

`references_retained` is **not** an endpoint of this experiment. If any future reader scores one
of these arms on it, **GATEFIX-001's three-valued rule governs** (CERTIFIED_PASS /
UNCERTIFIABLE / CERTIFIED_FAIL by exact 95% Clopper–Pearson position), and per SCOREFIX-001 that
rule is **currently reachable from no live scoring path**. Wiring it is a **build gate for this
experiment's scoring path**, not a retrofit — no arm here may be scored on that endpoint by a
module that applies the superseded two-valued rule.

## 7. Resource envelope and what is NOT yet granted

- **One** supervised training launch, **zero** optimizer retries
- Three arm materialisations (`H` merge-equivalent, `TH_s`, `SCALED_T`), FP32 CPU arithmetic
- Probes: 4 arms × acquisition, plus report-only preservation legs
- Adapters are ~100 MB, not 2.2 GB merges — this experiment's disk cost is **modest by design**;
  no full-precision base export is produced

**BLOCKED AT REGISTRATION TIME — recorded so the block is prospective, not an excuse later:**

| gate | required | measured 2026-09-21T01:42Z |
|---|---|---|
| free disk | ≥ 14.000 GiB floor | **11.847 GiB — BELOW FLOOR by 2.153** |
| host RAM | (4B lane gate TBD) | 4.828 GiB |
| GPU lease | free | free |
| GPU memory | ~7.4 GiB for 4-bit | 7.539 GiB free |

**No training may start while the shared floor is breached.** Three other roots grew 5.8 GiB in
the preceding four hours (`binding-integration001` 2.418, `halpha001` 1.712, `scaledh001` 1.711),
so the breach is live work by other sessions, not residue, and is **not mine to reclaim
unilaterally.**

**Authorization:** the operator's explicit instruction *"run the composition experiment on the 4B
base."* A numeric grant artifact (`GRANT.json`) must be written before the training launch; this
registration does not itself authorize spending.

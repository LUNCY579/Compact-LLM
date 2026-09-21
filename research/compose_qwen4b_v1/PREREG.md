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

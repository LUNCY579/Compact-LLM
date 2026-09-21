# COMPOSE4B-001 — P1 HELD. A second skill is free at matched magnitude on a base with headroom.

> **CONFIRMED 2026-09-21.** The integrity check named in §5 has run and **passed**: 48 TH_s rows
> regenerated on a quiet card are **byte-identical** to the stored run, 0 divergences, with a
> positive control confirming a mutation would have been detected. Contention moved timing, not
> outputs. `DETERMINISM_CHECK.json`.

**2026-09-21 · 3 arms × 384 rows · one training run · registered in `PREREG.md` with seven
amendments, every one committed with `scored/` verifiably empty.**

`quality_claim` false. `release_permitted` false. **No arm promoted.**

## 1. The result

| arm | construction | acquisition /384 | registered band |
|---|---|---:|---|
| `TH_s` | `s_t·(δ_T + δ_H)` | **262** | 200–280 ✓ |
| `SCALED_T` | `s_t·δ_T` | **254** | 200–280 ✓ |
| `H_only` | `δ_H` alone | **2** | 0–200 ✓ |

**Primary contrast, `TH_s − SCALED_T`, exact paired McNemar on `select-views-001`
(`views_sha256 7dd53f2a…`, endpoint `canonical`, within-draw):**

| gains | losses | net | nd | exact p | MDE at nd | min attainable p |
|---:|---:|---:|---:|---:|---:|---:|
| 14 | 6 | **+8** | 20 | **0.1153** | **10** | **1.9e-06** |

**Verdict: P1 — a null at the realized discordance.** `|net| = 8 ≤ MDE = 10`, and significance
*was* attainable at this discordance (min p 1.9e-06), so this is a null the instrument could
have broken and did not. It is **not** an equivalence certification.

> ### CORRECTION 1 — the power sentence was wrong and is withdrawn
>
> An earlier version read *"a cost of ≥ MDE rows would have been detected."* **That is false as
> a statement about the design's power**, and it is withdrawn from this document, from
> `ANALYSIS.json` and from `analyze.py`.
>
> MDE is computed **at the realized nd**, and nd = 20 is the single luckiest value available:
>
> | nd | 20 | 22 | 24 | 30 | 34 | 40 | 50 | 60 | 80 | 100 |
> |---|---|---|---|---|---|---|---|---|---|---|
> | MDE | **10** | 12 | 12 | 12 | 14 | 14 | 16 | 18 | 20 | 22 |
>
> A net of 10 reaches p ≤ 0.05 at nd = 20 (p = 0.0414) and at **no other nd in that range**. A
> genuine 10-row cost would generically arrive with *more* discordance than 20 — 10 systematic
> flips plus symmetric noise — and at nd > 20 it would **not** have been detected.
>
> **The only supported sentence is conditional:** *had these 20 discordant pairs split 15/5 or
> worse against TH_s, that split would have been significant (p = 0.0414).* That is a statement
> about this realization, not about the experiment's power against a 10-row effect.

## 2. What makes this stronger than the 596M result, not weaker

INDEP-001 showed on the 596M that a second skill costs zero acquisition **when the allocation is
fixed** — independent per-skill coefficients, +3.4 % magnitude, 235 = T16's 235.

**COMPOSE4B-001 finds no cost without fixing the allocation**, and against four measured
handicaps registered *before* the data:

| handicap | 596M | here |
|---|---:|---:|
| δ_T attenuation inside the composite | 0.675 | **0.5869** (δ_T pays 41.3 %) |
| δ_H vs its own operating point (~0.25) | ~2.7× | **2.35×** |
| `cosine(δ_T, δ_H)` | 0.0027 | **+0.0923** (~34× less orthogonal) |
| `‖δ_H‖ / ‖δ_T‖` | 1.053 | **1.3163** |

**The gate skill's directions cost nothing even when the composite is built the wrong way.**

> ### CORRECTION 2 — "δ_H cannot do the transition contract at all" is REFUTED by its own file
>
> Withdrawn. `H_only` scores **2/384 canonical but 76/384 contract_tolerant**, with
> `format_only_gain = 74` and `parsed = 356/384`. Of its 384 predictions, **214 are single-key
> dicts** (`{"after": {...}}`) and **74 of those unwrap to the EXACT reference**. The scorer
> applies the frozen unwrap set only in the tolerant branch, never in canonical.
>
> Against the 4-bit zero-shot base on the same bank and ids, `H_only` **wins**:
> 76 vs 54, gains 41 / losses 19, **net +22, nd 60, p = 0.0062**.
>
> **That is a CONTAINER failure, not a capability failure.** δ_H does acquire transition
> capability over the base; it just emits it in the wrong wrapper. Calling 2/384 "the
> load-bearing control" read a formatting artifact as an absence of skill.

**The mechanism conclusion survives, on different evidence.** Of the **14 gains** that make up
the +8, `H_only` solves **0 at canonical and 0 at contract_tolerant**. So the +8 is still not H
contributing task skill on those rows — it is H's directions failing to displace T's, which is
what near-orthogonality predicts. The claim is unchanged; the evidence for it is the
gain-row overlap, not H's aggregate score.

## 3. What the data bound, and what they do not

`equivalence_upper_bound_rows = 0.31` — one-sided 95 % CP upper bound on the loss-side excess.
Verified by two independent implementations (direct binomial inversion and incomplete-beta),
agreeing to 5 decimal places, with the peer-confirmed perfect-tie sanity value 7.92 reproduced.

**That 0.31 is tight only because the observed direction favours `TH_s`.** A perfect tie at
nd = 20 bounds at **7.92** rows; the reversed split (6 gains / 14 losses) bounds at **14.42**.
**No tolerance margin was registered** (Amendment 5 established that none below ~8 rows is
satisfiable at this n), so this bound is reported **descriptively** and is not a pre-registered
equivalence verdict.

> **CORRECTION 3.** An earlier version said the reversed split "would have been ~9.7 rows."
> That number is reproducible from nothing — the tie is 7.92184 and the reversal is 14.41785.
> Withdrawn and replaced with both computed values.

## 3a. The aggregate null is a CANCELLATION, not a flat result

Per-family, exploratory (family contrasts were **not** registered and cannot move the verdict):

| family | TH_s | SCALED_T | gains | losses | net | nd | p | MDE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `prerequisite_plan` | 88 | 79 | 10 | 1 | **+9** | 11 | **0.0117** | 9 |
| `ledger_replay` | 87 | 89 | 2 | 4 | −2 | 6 | 0.6875 | 6 |
| `register_machine` | 87 | 86 | 2 | 1 | +1 | 3 | 1.0 | — |

**The entire +8 is one family, and there it is significant** — p = 0.0117 survives Holm over
three tests (0.05/3 = 0.0167). The registered aggregate is P1 while the only family with real
discordance points the *other* registered direction (P3, "H's presence helps"). This is
exploratory and changes no verdict, but reporting the aggregate alone would hide that the null
is an average over one significantly improved family and one mildly degraded one.

## 3b. The comparison a reader will actually make is UNINFORMATIVE

`TH_s` 262 vs **unattenuated δ_T** 270 (same bank, same ids): gains 16, losses 24, **net −8,
nd 40, p = 0.268, MDE 14**. By `analyze.py`'s own rule `|net| 8 < MDE 14` with p > α lands in
the branch labelled **UNINFORMATIVE** — it cannot be called a null.

**So the composite is not shown to match the shipped single-skill arm.** What is shown is that
*given* T has been attenuated to `s_t`, adding H costs nothing detectable. The attenuation
itself is a separate and larger question this experiment does not settle.

## 4. What this does NOT license

Registered before the data and binding now:

- **Not "composition is free" in general.** One pairing, one base, one draw, one bank.
- **Not a statement about damage.** Only acquisition was measured. Retention, the arithmetic
  panel, long context and literal `UNKNOWN` are **NOT MEASURED** for any arm here — the same gap
  INDEP-001 identified across every prior composition arm.
- **Not an equivalence claim.** §3.
- **Not a resolution of the five confounds.** Headroom, attenuation, H's absolute strength,
  allocation and orthogonality all differ between the two bases and **this design separates
  none of them.** A P2 would have been confounded; a P1 is *robust to* all five, which is why
  the null is the informative outcome here and the alternative would not have been.

## 5. OUTSTANDING: a determinism check on `TH_s`

`TH_s` was generated under GPU contention from another lane. Wall clock attributes it exactly:

| arm | wall | s/row | excess vs clean |
|---|---:|---:|---:|
| `TH_s` | 2412.9 s | 6.28 | **+1764 s (~29 min)** |
| `SCALED_T` | 649.3 s | **1.69** | 0 — clean |
| `H_only` | 821.6 s | 2.14 | +173 s |

**The contended arm is the one that scored higher**, so the risk is not neutral: if contention
perturbed generation, it perturbed the arm carrying the +8. Greedy decoding with TF32 off is
deterministic and this programme has measured probe determinism at exactly 0/256 — but that is
an argument, not a measurement on these bytes.

**A spot-check regenerating a `TH_s` subset and comparing against the stored rows is queued for
the next free GPU window. If it diverges, this contrast is BLOCKED, not caveated.**

(An earlier note of mine said the check was queued on `SCALED_T`. That was wrong — `SCALED_T` is
the clean arm. Corrected here.)

## 6. Provenance

One training run: δ_H, LoRA r=16, peak 1e-4, 28 groups, 648 rows, 54,217 processed / 6,274
supervised tokens, loss 6.854 → 1.584. δ_T reused from QWENFT001 (`run/adapter`, unmodified).
Build gate PASS: worst relative error 6.131e-08 against the 2e-06 tolerance, `‖TH_s‖ = ‖δ_T‖` to
5.756e-08, 252 modules, 0 fallbacks. `s_t` computed once and read thereafter, never recomputed.
Train/eval disjointness established on **content** with a positive control after id-namespace
matching was shown vacuous: 0 exact overlap, 0 of 648 H prompts sharing a 20-gram with any view,
0 digit-blinded template collisions.
